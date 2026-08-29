#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error_ = (call);                                           \
    if (error_ != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string(#call) + ": " +                      \
                               cudaGetErrorString(error_));                      \
    }                                                                           \
  } while (0)

namespace {

constexpr int kComponents = 5;
constexpr int kAxes = 3;
constexpr float kGamma = 1.4f;
constexpr float kGammaMinusOne = 0.4f;
constexpr float kEpsilon = 1.0e-6f;
constexpr float kCfl = 0.1f;

struct Options {
  int size = 128;
  int steps = 1;
  int warmups = 5;
  int repetitions = 30;
  std::string output_initial;
  std::string output_state;
};

struct DeviceState {
  const float* data;
  int n;
  std::int64_t cells;
};

__host__ __device__ __forceinline__ int wrap(int value, int n) {
  if (value < 0) value += n;
  if (value >= n) value -= n;
  return value;
}

__host__ __device__ __forceinline__ std::int64_t index3(
    int x, int y, int z, int n) {
  return (static_cast<std::int64_t>(z) * n + y) * n + x;
}

__device__ __forceinline__ std::int64_t offset_index(
    int x, int y, int z, int axis, int offset, int n) {
  if (axis == 0) x = wrap(x + offset, n);
  if (axis == 1) y = wrap(y + offset, n);
  if (axis == 2) z = wrap(z + offset, n);
  return index3(x, y, z, n);
}

__device__ __forceinline__ void load_state(
    const DeviceState& state, std::int64_t cell, float q[kComponents]) {
#pragma unroll
  for (int component = 0; component < kComponents; ++component) {
    q[component] = state.data[static_cast<std::int64_t>(component) *
                                  state.cells +
                              cell];
  }
}

__device__ __forceinline__ float primitive_flux(
    const float q[kComponents], int axis, float flux[kComponents]) {
  const float reciprocal_density = 1.0f / q[0];
  const float u = q[1] * reciprocal_density;
  const float v = q[2] * reciprocal_density;
  const float w = q[3] * reciprocal_density;
  const float pressure = kGammaMinusOne *
      (q[4] - 0.5f * q[0] * (u * u + v * v + w * w));
  const float sound = sqrtf(kGamma * pressure * reciprocal_density);
  const float velocities[3] = {u, v, w};
  const float normal_velocity = velocities[axis];

  flux[0] = q[axis + 1];
  flux[1] = q[1] * normal_velocity;
  flux[2] = q[2] * normal_velocity;
  flux[3] = q[3] * normal_velocity;
  flux[axis + 1] += pressure;
  flux[4] = normal_velocity * (q[4] + pressure);
  return fabsf(normal_velocity) + sound;
}

__device__ __forceinline__ void js_weights(
    float a, float b, float c, float d, float e, float weights[3]) {
  const float d20 = a - 2.0f * b + c;
  const float d21 = b - 2.0f * c + d;
  const float d22 = c - 2.0f * d + e;
  const float d10 = a - 4.0f * b + 3.0f * c;
  const float d11 = b - d;
  const float d12 = 3.0f * c - 4.0f * d + e;
  const float beta0 = 1.0833333333333333f * d20 * d20 + 0.25f * d10 * d10;
  const float beta1 = 1.0833333333333333f * d21 * d21 + 0.25f * d11 * d11;
  const float beta2 = 1.0833333333333333f * d22 * d22 + 0.25f * d12 * d12;
  const float e0 = kEpsilon + beta0;
  const float e1 = kEpsilon + beta1;
  const float e2 = kEpsilon + beta2;
  const float alpha0 = 0.1f / (e0 * e0);
  const float alpha1 = 0.6f / (e1 * e1);
  const float alpha2 = 0.3f / (e2 * e2);
  const float reciprocal = 1.0f / (alpha0 + alpha1 + alpha2);
  weights[0] = alpha0 * reciprocal;
  weights[1] = alpha1 * reciprocal;
  weights[2] = alpha2 * reciprocal;
}

__device__ __forceinline__ float reconstruct_left(
    const float values[5], const float weights[3]) {
  const float p0 = (2.0f * values[0] - 7.0f * values[1] +
                    11.0f * values[2]) /
                   6.0f;
  const float p1 = (-values[1] + 5.0f * values[2] +
                    2.0f * values[3]) /
                   6.0f;
  const float p2 = (2.0f * values[2] + 5.0f * values[3] - values[4]) /
                   6.0f;
  return weights[0] * p0 + weights[1] * p1 + weights[2] * p2;
}

__device__ __forceinline__ void u0_face_flux(
    const DeviceState& state, int x, int y, int z, int axis,
    float output[kComponents]) {
  float q[6][kComponents];
  float flux[6][kComponents];
  float density[6];
  float alpha = 0.0f;
#pragma unroll
  for (int sample = 0; sample < 6; ++sample) {
    const std::int64_t cell = offset_index(x, y, z, axis, sample - 2, state.n);
    load_state(state, cell, q[sample]);
    density[sample] = q[sample][0];
    alpha = fmaxf(alpha, primitive_flux(q[sample], axis, flux[sample]));
  }

  float positive_weights[3];
  float negative_weights[3];
  js_weights(density[0], density[1], density[2], density[3], density[4],
             positive_weights);
  js_weights(density[5], density[4], density[3], density[2], density[1],
             negative_weights);

#pragma unroll
  for (int component = 0; component < kComponents; ++component) {
    float positive[5];
    float negative[5];
#pragma unroll
    for (int sample = 0; sample < 5; ++sample) {
      positive[sample] =
          0.5f * (flux[sample][component] + alpha * q[sample][component]);
      const int reversed = 5 - sample;
      negative[sample] =
          0.5f * (flux[reversed][component] - alpha * q[reversed][component]);
    }
    output[component] = reconstruct_left(positive, positive_weights) +
                        reconstruct_left(negative, negative_weights);
  }
}

__device__ __forceinline__ void atomic_max_positive(float* address, float value) {
  atomicMax(reinterpret_cast<unsigned int*>(address), __float_as_uint(value));
}

__global__ void cfl_kernel(DeviceState state, float inverse_spacing,
                           float* maximum) {
  float local = 0.0f;
  for (std::int64_t cell =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < state.cells;
       cell += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    float q[kComponents];
    load_state(state, cell, q);
    const float reciprocal_density = 1.0f / q[0];
    const float u = q[1] * reciprocal_density;
    const float v = q[2] * reciprocal_density;
    const float w = q[3] * reciprocal_density;
    const float pressure = kGammaMinusOne *
        (q[4] - 0.5f * q[0] * (u * u + v * v + w * w));
    const float sound = sqrtf(kGamma * pressure * reciprocal_density);
    local = fmaxf(local, (fabsf(u) + fabsf(v) + fabsf(w) + 3.0f * sound) *
                             inverse_spacing);
  }

  extern __shared__ float shared[];
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomic_max_positive(maximum, shared[0]);
}

__global__ void finish_cfl_kernel(float* value) { value[0] = kCfl / value[0]; }

__global__ void face_kernel(DeviceState state, float* faces) {
  for (std::int64_t cell =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < state.cells;
       cell += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int x = static_cast<int>(cell % state.n);
    const int y = static_cast<int>((cell / state.n) % state.n);
    const int z = static_cast<int>(cell /
                                   (static_cast<std::int64_t>(state.n) * state.n));
#pragma unroll
    for (int axis = 0; axis < kAxes; ++axis) {
      float numerical[kComponents];
      u0_face_flux(state, x, y, z, axis, numerical);
#pragma unroll
      for (int component = 0; component < kComponents; ++component) {
        faces[(static_cast<std::int64_t>(axis) * kComponents + component) *
                  state.cells +
              cell] = numerical[component];
      }
    }
  }
}

__global__ void update_kernel(DeviceState state, const float* faces,
                              const float* dt, float inverse_spacing,
                              float* output) {
  for (std::int64_t cell =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < state.cells;
       cell += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int x = static_cast<int>(cell % state.n);
    const int y = static_cast<int>((cell / state.n) % state.n);
    const int z = static_cast<int>(cell /
                                   (static_cast<std::int64_t>(state.n) * state.n));
    const std::int64_t left_cells[3] = {
        index3(wrap(x - 1, state.n), y, z, state.n),
        index3(x, wrap(y - 1, state.n), z, state.n),
        index3(x, y, wrap(z - 1, state.n), state.n)};
#pragma unroll
    for (int component = 0; component < kComponents; ++component) {
      float divergence = 0.0f;
#pragma unroll
      for (int axis = 0; axis < kAxes; ++axis) {
        const std::int64_t base =
            (static_cast<std::int64_t>(axis) * kComponents + component) *
            state.cells;
        divergence += faces[base + cell] - faces[base + left_cells[axis]];
      }
      output[static_cast<std::int64_t>(component) * state.cells + cell] =
          state.data[static_cast<std::int64_t>(component) * state.cells + cell] -
          dt[0] * inverse_spacing * divergence;
    }
  }
}

std::vector<float> initialize_vortex(int n) {
  const std::int64_t cells = static_cast<std::int64_t>(n) * n * n;
  std::vector<float> state(static_cast<std::size_t>(kComponents * cells));
  const double pi = std::acos(-1.0);
  const double coefficient = 5.0 / (2.0 * pi * std::exp(-0.5));
  for (int z = 0; z < n; ++z) {
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        const double px = 10.0 * x / n;
        const double py = 10.0 * y / n;
        const double radius_squared =
            (px - 5.0) * (px - 5.0) + (py - 5.0) * (py - 5.0);
        const double exponential = std::exp(-0.5 * radius_squared);
        const double u = -coefficient * exponential * (py - 5.0);
        const double v = coefficient * exponential * (px - 5.0);
        const double temperature =
            1.0 - 0.5 * coefficient * coefficient * exponential * exponential *
                      ((kGamma - 1.0) / kGamma);
        const double pressure =
            std::pow(temperature, kGamma / (kGamma - 1.0));
        const double density = pressure / temperature;
        const double energy =
            pressure / (kGamma - 1.0) + 0.5 * density * (u * u + v * v);
        const std::int64_t cell = index3(x, y, z, n);
        state[cell] = static_cast<float>(density);
        state[cells + cell] = static_cast<float>(density * u);
        state[2 * cells + cell] = static_cast<float>(density * v);
        state[3 * cells + cell] = 0.0f;
        state[4 * cells + cell] = static_cast<float>(energy);
      }
    }
  }
  return state;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto require_value = [&]() -> const char* {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + argument);
      return argv[++i];
    };
    if (argument == "--size") options.size = std::stoi(require_value());
    else if (argument == "--steps") options.steps = std::stoi(require_value());
    else if (argument == "--warmups") options.warmups = std::stoi(require_value());
    else if (argument == "--repetitions")
      options.repetitions = std::stoi(require_value());
    else if (argument == "--output-initial")
      options.output_initial = require_value();
    else if (argument == "--output-state")
      options.output_state = require_value();
    else throw std::runtime_error("unknown argument: " + argument);
  }
  if (options.size < 8 || options.steps < 1 || options.warmups < 0 ||
      options.repetitions < 1) {
    throw std::runtime_error("invalid size, steps, warmups, or repetitions");
  }
  return options;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

void write_state(const std::string& path, const std::vector<float>& state) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot open output state: " + path);
  output.write(reinterpret_cast<const char*>(state.data()),
               static_cast<std::streamsize>(state.size() * sizeof(float)));
  if (!output) throw std::runtime_error("failed to write output state: " + path);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const std::int64_t cells =
        static_cast<std::int64_t>(options.size) * options.size * options.size;
    const std::size_t state_elements =
        static_cast<std::size_t>(kComponents * cells);
    const std::size_t state_bytes = state_elements * sizeof(float);
    const std::size_t face_bytes =
        static_cast<std::size_t>(kAxes) * state_bytes;
    std::vector<float> initial = initialize_vortex(options.size);
    std::vector<float> result(state_elements);
    if (!options.output_initial.empty()) {
      write_state(options.output_initial, initial);
    }

    float *d_initial = nullptr, *d_q = nullptr, *d_next = nullptr;
    float *d_faces = nullptr, *d_dt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_initial, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_q, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_next, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_faces, face_bytes));
    CUDA_CHECK(cudaMalloc(&d_dt, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_initial, initial.data(), state_bytes,
                          cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    int blocks = static_cast<int>((cells + threads - 1) / threads);
    blocks = std::min(blocks, 65535);
    const float inverse_spacing = static_cast<float>(options.size) / 10.0f;

    auto launch = [&](float*& q, float*& next) {
      for (int step = 0; step < options.steps; ++step) {
        CUDA_CHECK(cudaMemsetAsync(d_dt, 0, sizeof(float)));
        cfl_kernel<<<blocks, threads, threads * sizeof(float)>>>(
            DeviceState{q, options.size, cells}, inverse_spacing, d_dt);
        finish_cfl_kernel<<<1, 1>>>(d_dt);
        face_kernel<<<blocks, threads>>>(
            DeviceState{q, options.size, cells}, d_faces);
        update_kernel<<<blocks, threads>>>(
            DeviceState{q, options.size, cells}, d_faces, d_dt,
            inverse_spacing, next);
        std::swap(q, next);
      }
    };

    for (int warmup = 0; warmup < options.warmups; ++warmup) {
      CUDA_CHECK(cudaMemcpy(d_q, d_initial, state_bytes, cudaMemcpyDeviceToDevice));
      float* q = d_q;
      float* next = d_next;
      launch(q, next);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t started{}, stopped{};
    CUDA_CHECK(cudaEventCreate(&started));
    CUDA_CHECK(cudaEventCreate(&stopped));
    std::vector<double> observations;
    observations.reserve(options.repetitions);
    float* final_q = d_q;
    for (int repetition = 0; repetition < options.repetitions; ++repetition) {
      CUDA_CHECK(cudaMemcpy(d_q, d_initial, state_bytes, cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaDeviceSynchronize());
      float* q = d_q;
      float* next = d_next;
      CUDA_CHECK(cudaEventRecord(started));
      launch(q, next);
      CUDA_CHECK(cudaEventRecord(stopped));
      CUDA_CHECK(cudaEventSynchronize(stopped));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, started, stopped));
      observations.push_back(static_cast<double>(elapsed_ms));
      final_q = q;
    }

    CUDA_CHECK(cudaMemcpy(result.data(), final_q, state_bytes,
                          cudaMemcpyDeviceToHost));
    float final_dt = 0.0f;
    CUDA_CHECK(cudaMemcpy(&final_dt, d_dt, sizeof(float), cudaMemcpyDeviceToHost));

    bool finite = true;
    double checksum = 0.0;
    double maximum_absolute = 0.0;
    for (float value : result) {
      finite = finite && std::isfinite(value);
      checksum += static_cast<double>(value);
      maximum_absolute =
          std::max(maximum_absolute, std::fabs(static_cast<double>(value)));
    }
    if (!options.output_state.empty()) write_state(options.output_state, result);

    const double median_ms = median(observations);
    const double cells_per_second =
        static_cast<double>(cells) * options.steps / (median_ms * 1.0e-3);
    const double faces_per_second =
        static_cast<double>(kAxes) * cells * options.steps /
        (median_ms * 1.0e-3);

    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"contract\": \"u0_unique_f32_component_shared_density_local_lf_forward_euler_v1\",\n";
    std::cout << "  \"gpu\": \"" << properties.name << "\",\n";
    std::cout << "  \"compute_capability\": \"" << properties.major << "."
              << properties.minor << "\",\n";
    std::cout << "  \"size\": " << options.size << ",\n";
    std::cout << "  \"cells\": " << cells << ",\n";
    std::cout << "  \"steps\": " << options.steps << ",\n";
    std::cout << "  \"warmups\": " << options.warmups << ",\n";
    std::cout << "  \"repetitions\": " << options.repetitions << ",\n";
    std::cout << "  \"median_device_ms\": " << median_ms << ",\n";
    std::cout << "  \"cells_per_second\": " << cells_per_second << ",\n";
    std::cout << "  \"directional_faces_per_second\": " << faces_per_second
              << ",\n";
    std::cout << "  \"final_dt\": " << final_dt << ",\n";
    std::cout << "  \"finite\": " << (finite ? "true" : "false") << ",\n";
    std::cout << "  \"checksum_float64\": " << checksum << ",\n";
    std::cout << "  \"maximum_absolute\": " << maximum_absolute << ",\n";
    std::cout << "  \"peak_allocated_bytes\": "
              << (3 * state_bytes + face_bytes + sizeof(float)) << ",\n";
    std::cout << "  \"samples_device_ms\": [";
    for (std::size_t i = 0; i < observations.size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << observations[i];
    }
    std::cout << "]\n";
    std::cout << "}\n";

    CUDA_CHECK(cudaEventDestroy(started));
    CUDA_CHECK(cudaEventDestroy(stopped));
    CUDA_CHECK(cudaFree(d_initial));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_faces));
    CUDA_CHECK(cudaFree(d_dt));
    return finite ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
