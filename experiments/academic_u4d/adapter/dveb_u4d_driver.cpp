// U4-D research adapter around unmodified DVEB-generated WENO-5 launchers.
#include "weno5_gen.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct Arguments {
    long size = 0;
    std::string backend;
    std::string mode;
    std::string input;
    std::string output;
    int warmups = 5;
    int samples = 20;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr, "u4d-dveb: %s\n", message);
    std::exit(2);
}

Arguments parse(int argc, char** argv) {
    Arguments result;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) usage("every option requires a value");
        const std::string option = argv[i++];
        const std::string value = argv[i];
        if (option == "--size") result.size = std::strtol(value.c_str(), nullptr, 10);
        else if (option == "--backend") result.backend = value;
        else if (option == "--mode") result.mode = value;
        else if (option == "--input") result.input = value;
        else if (option == "--output") result.output = value;
        else if (option == "--warmups") result.warmups = std::atoi(value.c_str());
        else if (option == "--samples") result.samples = std::atoi(value.c_str());
        else usage("unknown option");
    }
    if (result.size <= 0 || result.input.empty()) usage("size and input are required");
    if (result.backend != "cpu" && result.backend != "cuda") usage("invalid backend");
    if (result.mode != "qualify" && result.mode != "resident" &&
        result.mode != "transfer" && result.mode != "launch") usage("invalid mode");
    if (result.mode == "transfer" && result.backend != "cuda") {
        usage("transfer mode is CUDA-only");
    }
    if (result.warmups < 0 || result.samples <= 0) usage("invalid sample counts");
    return result;
}

void read_state(const Arguments& args, const dveb::FieldView& field) {
    std::FILE* stream = std::fopen(args.input.c_str(), "rb");
    if (stream == nullptr) usage("unable to open input");
    for (long i = 0; i < args.size; ++i) {
        double value = 0.0;
        if (std::fread(&value, sizeof(double), 1, stream) != 1) {
            std::fclose(stream);
            usage("incomplete input");
        }
        dveb::at(field, 0, i) = value;
    }
    double extra = 0.0;
    if (std::fread(&extra, sizeof(double), 1, stream) != 0) {
        std::fclose(stream);
        usage("input contains extra values");
    }
    std::fclose(stream);
}

void write_rhs(const Arguments& args, const dveb::FieldView& field) {
    if (args.output.empty()) return;
    std::FILE* stream = std::fopen(args.output.c_str(), "wb");
    if (stream == nullptr) usage("unable to open output");
    for (long i = 0; i < args.size; ++i) {
        const double value = dveb::at_ro(field, 0, i);
        if (std::fwrite(&value, sizeof(double), 1, stream) != 1) {
            std::fclose(stream);
            usage("incomplete output write");
        }
    }
    std::fclose(stream);
}

void report(const dveb::FieldView& field) {
    double checksum = 0.0;
    double maximum = 0.0;
    bool finite = true;
    for (long i = 0; i < field.n; ++i) {
        const double value = dveb::at_ro(field, 0, i);
        checksum += value;
        maximum = std::fmax(maximum, std::fabs(value));
        finite = finite && std::isfinite(value);
    }
    std::printf(
        "U4D_RESULT finite=%d checksum=%.17g maximum=%.17g\n",
        finite ? 1 : 0, checksum, maximum);
}

}  // namespace

int main(int argc, char** argv) {
    const Arguments args = parse(argc, argv);
    constexpr long kGhost = 3;
    const double dx = 1.0 / static_cast<double>(args.size);
    dveb::HostField input_owner(args.size, 1, kGhost);
    dveb::HostField output_owner(args.size, 1, kGhost);
    const dveb::FieldView input = input_owner.view();
    const dveb::FieldView output = output_owner.view();
    read_state(args, input);

    const bool one_call = args.mode == "qualify" || args.mode == "launch";
    const int total = one_call ? 1 : args.warmups + args.samples;
    if (args.backend == "cpu") {
        for (int repetition = 0; repetition < total; ++repetition) {
            dveb::exchange_ghosts_host(input);
            const auto started = std::chrono::steady_clock::now();
            k_weno5_rhs_cpu(input, output, dx, std::fabs(A));
            const double milliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
            if (!one_call && repetition >= args.warmups) {
                std::printf("U4D_SAMPLE %.17g\n", milliseconds);
            }
        }
    } else {
        dveb::verify_targets(12, 0);
        dveb::CudaField device_input_owner(args.size, 1, kGhost);
        dveb::CudaField device_output_owner(args.size, 1, kGhost);
        const dveb::FieldView device_input = device_input_owner.view();
        const dveb::FieldView device_output = device_output_owner.view();
        if (args.mode != "transfer") dveb::upload(input, device_input);
        cudaEvent_t start_event = nullptr;
        cudaEvent_t end_event = nullptr;
        if (args.mode == "resident") {
            DVEB_CUDA_CHECK(cudaEventCreate(&start_event));
            DVEB_CUDA_CHECK(cudaEventCreate(&end_event));
        }
        for (int repetition = 0; repetition < total; ++repetition) {
            std::chrono::steady_clock::time_point started;
            if (args.mode == "transfer") {
                started = std::chrono::steady_clock::now();
                dveb::upload(input, device_input);
            }
            dveb::exchange_ghosts_cuda(device_input);
            if (args.mode == "resident") {
                DVEB_CUDA_CHECK(cudaEventRecord(start_event));
            }
            k_weno5_rhs_cuda(device_input, device_output, dx, std::fabs(A));
            double milliseconds = 0.0;
            if (args.mode == "resident") {
                DVEB_CUDA_CHECK(cudaEventRecord(end_event));
                DVEB_CUDA_CHECK(cudaEventSynchronize(end_event));
                float elapsed = 0.0f;
                DVEB_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, end_event));
                milliseconds = static_cast<double>(elapsed);
            } else if (args.mode == "transfer") {
                dveb::download(device_output, output);
                milliseconds = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
            } else {
                dveb::sync_cuda();
            }
            if (!one_call && repetition >= args.warmups) {
                std::printf("U4D_SAMPLE %.17g\n", milliseconds);
            }
        }
        if (start_event != nullptr) DVEB_CUDA_CHECK(cudaEventDestroy(start_event));
        if (end_event != nullptr) DVEB_CUDA_CHECK(cudaEventDestroy(end_event));
        if (args.mode != "transfer") dveb::download(device_output, output);
    }

    write_rhs(args, output);
    report(output);
    return 0;
}
