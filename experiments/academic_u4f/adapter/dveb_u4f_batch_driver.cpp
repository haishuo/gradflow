// U4-F batched adapter around the immutable DVEB scalar scheduling ABI v1.
//
// This file owns only row-wise padding, allocation, transfer, timing, and
// result export. It deliberately contains no WENO mathematics.
#include "weno5_schedule_abi_v1.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr long kGhost = 3;
constexpr int kAbi = DVEB_SCALAR_SCHEDULE_ABI_V1;

struct Arguments {
    long size = 0;
    long batch = 0;
    std::string backend;
    std::string mode;
    std::string input;
    std::string output;
    int warmups = 5;
    int samples = 20;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr, "u4f-dveb-batch: %s\n", message);
    std::exit(2);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "u4f-dveb-batch: %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(1);
    }
}

Arguments parse(int argc, char** argv) {
    Arguments result;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) usage("every option requires a value");
        const std::string option = argv[i++];
        const std::string value = argv[i];
        if (option == "--size") result.size = std::strtol(value.c_str(), nullptr, 10);
        else if (option == "--batch") result.batch = std::strtol(value.c_str(), nullptr, 10);
        else if (option == "--backend") result.backend = value;
        else if (option == "--mode") result.mode = value;
        else if (option == "--input") result.input = value;
        else if (option == "--output") result.output = value;
        else if (option == "--warmups") result.warmups = std::atoi(value.c_str());
        else if (option == "--samples") result.samples = std::atoi(value.c_str());
        else usage("unknown option");
    }
    if (result.size <= 0 || result.batch <= 0 || result.input.empty()) {
        usage("size, batch, and input are required");
    }
    if (result.backend != "cpu" && result.backend != "cuda") usage("invalid backend");
    if (result.mode != "qualify" && result.mode != "resident") usage("invalid mode");
    if (result.warmups < 0 || result.samples <= 0) usage("invalid sample counts");
    return result;
}

void read_and_pad(const Arguments& args, std::vector<double>& field) {
    const long stride = args.size + 2 * kGhost;
    std::FILE* stream = std::fopen(args.input.c_str(), "rb");
    if (!stream) usage("unable to open input");
    for (long b = 0; b < args.batch; ++b) {
        double* row = field.data() + b * stride;
        if (std::fread(row + kGhost, sizeof(double), args.size, stream) !=
            static_cast<size_t>(args.size)) {
            std::fclose(stream);
            usage("incomplete input");
        }
        for (long j = 0; j < kGhost; ++j) {
            row[j] = row[kGhost + args.size - kGhost + j];
            row[kGhost + args.size + j] = row[kGhost + j];
        }
    }
    double extra = 0.0;
    if (std::fread(&extra, sizeof(double), 1, stream) != 0) {
        std::fclose(stream);
        usage("input contains extra values");
    }
    std::fclose(stream);
}

void write_interior(const Arguments& args, const std::vector<double>& field) {
    if (args.output.empty()) return;
    const long stride = args.size + 2 * kGhost;
    std::FILE* stream = std::fopen(args.output.c_str(), "wb");
    if (!stream) usage("unable to open output");
    for (long b = 0; b < args.batch; ++b) {
        const double* row = field.data() + b * stride + kGhost;
        if (std::fwrite(row, sizeof(double), args.size, stream) !=
            static_cast<size_t>(args.size)) {
            std::fclose(stream);
            usage("incomplete output write");
        }
    }
    std::fclose(stream);
}

void abi_check(dveb_scalar_status_v1 status, const char* operation,
               const char* error) {
    if (status != DVEB_SCALAR_OK_V1) {
        std::fprintf(stderr, "u4f-dveb-batch: %s failed (%d): %s\n", operation,
                     static_cast<int>(status), error);
        std::exit(1);
    }
}

dveb_scalar_result_v1 empty_result() {
    dveb_scalar_result_v1 result{};
    result.struct_size = sizeof(result);
    result.abi_version = kAbi;
    return result;
}

void print_policy(const char* kind, const char* target,
                  const dveb_scalar_result_v1& result) {
    std::printf(
        "U4F_%s target=%s cpu_loop=%d cuda_block=%d reuse=%d launches=%d "
        "scratch_bytes=%llu elements=%llu synchronized=%d\n",
        kind, target, result.selected_cpu_loop, result.selected_cuda_block,
        result.selected_reuse, result.numerical_launches,
        static_cast<unsigned long long>(result.scratch_bytes),
        static_cast<unsigned long long>(result.elements_written),
        result.synchronized);
}

void report(const Arguments& args, const std::vector<double>& output) {
    const long stride = args.size + 2 * kGhost;
    bool finite = true;
    double checksum = 0.0;
    double maximum = 0.0;
    for (long b = 0; b < args.batch; ++b) {
        const double* row = output.data() + b * stride + kGhost;
        for (long i = 0; i < args.size; ++i) {
            finite = finite && std::isfinite(row[i]);
            checksum += row[i];
            maximum = std::fmax(maximum, std::fabs(row[i]));
        }
    }
    std::printf(
        "U4F_RESULT finite=%d checksum=%.17g maximum=%.17g batch=%ld size=%ld\n",
        finite ? 1 : 0, checksum, maximum, args.batch, args.size);
    if (!finite) std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    const Arguments args = parse(argc, argv);
    if (dveb_scalar_schedule_abi_version() != kAbi) usage("ABI version mismatch");

    const long stride = args.size + 2 * kGhost;
    const size_t padded_elements = static_cast<size_t>(args.batch) * stride;
    const size_t padded_bytes = padded_elements * sizeof(double);
    std::vector<double> input(padded_elements);
    std::vector<double> output(padded_elements);
    read_and_pad(args, input);

    const bool use_cuda = args.backend == "cuda";
    dveb_scalar_create_v1 create{};
    create.struct_size = sizeof(create);
    create.abi_version = kAbi;
    create.target_mask = use_cuda ? DVEB_SCALAR_TARGET_CUDA_V1
                                  : DVEB_SCALAR_TARGET_CPU_V1;
    create.n = args.size;
    create.nb = args.batch;
    create.ghost = kGhost;
    create.cpu_threads = 1;
    dveb_scalar_schedule_context_v1* context = nullptr;
    char error[512]{};
    abi_check(dveb_scalar_schedule_create_v1(&create, &context, error, sizeof(error)),
              "create", error);

    double* device_input = nullptr;
    double* device_output = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t end_event = nullptr;
    if (use_cuda) {
        cuda_check(cudaMalloc(&device_input, padded_bytes), "cudaMalloc(input)");
        cuda_check(cudaMalloc(&device_output, padded_bytes), "cudaMalloc(output)");
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
        cuda_check(cudaEventCreate(&start_event), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&end_event), "cudaEventCreate(end)");
        cuda_check(cudaMemcpyAsync(device_input, input.data(), padded_bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "initial H2D");
        cuda_check(cudaStreamSynchronize(stream), "initial synchronize");
    }

    dveb_scalar_run_v1 run{};
    run.struct_size = sizeof(run);
    run.abi_version = kAbi;
    run.cpu_loop = DVEB_SCALAR_CPU_AUTO_V1;
    run.cuda_block = DVEB_SCALAR_CUDA_AUTO_V1;
    run.reuse = DVEB_SCALAR_REUSE_AUTO_V1;
    run.dx = 1.0 / static_cast<double>(args.size);
    run.em = 1.0;
    run.input = use_cuda ? static_cast<const void*>(device_input)
                         : static_cast<const void*>(input.data());
    run.output = use_cuda ? static_cast<void*>(device_output)
                          : static_cast<void*>(output.data());
    run.padded_elements = padded_elements;
    run.stream = use_cuda ? static_cast<void*>(stream) : nullptr;

    dveb_scalar_result_v1 query = empty_result();
    abi_check(dveb_scalar_schedule_query_v1(
                  context,
                  use_cuda ? DVEB_SCALAR_TARGET_CUDA_V1 : DVEB_SCALAR_TARGET_CPU_V1,
                  &run, &query, error, sizeof(error)),
              "query", error);
    print_policy("QUERY", args.backend.c_str(), query);

    const bool one_call = args.mode == "qualify";
    const int total = one_call ? 1 : args.warmups + args.samples;
    dveb_scalar_result_v1 last = empty_result();
    if (!use_cuda) {
        for (int repetition = 0; repetition < total; ++repetition) {
            const auto started = std::chrono::steady_clock::now();
            last = empty_result();
            abi_check(dveb_scalar_schedule_run_cpu_padded_v1(
                          context, &run, &last, error, sizeof(error)),
                      "CPU run", error);
            const double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            if (!one_call && repetition >= args.warmups) {
                std::printf("U4F_SAMPLE %.17g\n", milliseconds);
            }
        }
    } else {
        for (int repetition = 0; repetition < total; ++repetition) {
            cuda_check(cudaEventRecord(start_event, stream), "event record start");
            last = empty_result();
            abi_check(dveb_scalar_schedule_run_cuda_padded_v1(
                          context, &run, &last, error, sizeof(error)),
                      "CUDA run", error);
            cuda_check(cudaEventRecord(end_event, stream), "event record end");
            cuda_check(cudaEventSynchronize(end_event), "event synchronize");
            float milliseconds = 0.0f;
            cuda_check(cudaEventElapsedTime(&milliseconds, start_event, end_event),
                       "event elapsed");
            if (!one_call && repetition >= args.warmups) {
                std::printf("U4F_SAMPLE %.17g\n", static_cast<double>(milliseconds));
            }
        }
        cuda_check(cudaMemcpyAsync(output.data(), device_output, padded_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "final D2H");
        cuda_check(cudaStreamSynchronize(stream), "final synchronize");
    }

    print_policy("RUN", args.backend.c_str(), last);
    write_interior(args, output);
    report(args, output);

    if (use_cuda) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
        cudaStreamDestroy(stream);
        cudaFree(device_input);
        cudaFree(device_output);
    }
    abi_check(dveb_scalar_schedule_destroy_v1(context, error, sizeof(error)),
              "destroy", error);
    return 0;
}
