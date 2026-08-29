#include "runner.h"
#include "shu_math.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t error_ = (call); \
  if (error_ != cudaSuccess) throw std::runtime_error(std::string(#call) + ": " \
      + cudaGetErrorString(error_)); \
} while (0)

namespace {

using shu3d::StateView;
using shu3d::Vec5;

__device__ void atomic_max_positive(float* address, float value) {
  atomicMax(reinterpret_cast<unsigned int*>(address), __float_as_uint(value));
}

__global__ void timestep_kernel(StateView state, float inverse_spacing,
                                float* maximum) {
  const int n = state.intervals;
  const std::int64_t total = static_cast<std::int64_t>(n) * n * n;
  for (std::int64_t t = blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int x = static_cast<int>(t % n) + 1;
    const int y = static_cast<int>((t / n) % n) + 1;
    const int z = static_cast<int>(t / (static_cast<std::int64_t>(n) * n)) + 1;
    const Vec5 q = shu3d::load_global(state, x, y, z);
    const float inverse_density = 1.0f / q.v[0];
    const float u = q.v[1] * inverse_density;
    const float v = q.v[2] * inverse_density;
    const float w = q.v[3] * inverse_density;
    const float pressure = shu3d::gamma_minus_one * (q.v[4]
        - 0.5f * q.v[0] * (u * u + v * v + w * w));
    const float sound = sqrtf(shu3d::gamma_value * pressure * inverse_density);
    atomic_max_positive(maximum, (fabsf(u) + fabsf(v) + fabsf(w)
        + 3.0f * sound) * inverse_spacing);
  }
}

__global__ void finish_timestep_kernel(float* value) { value[0] = 0.1f / value[0]; }

__global__ void alpha_kernel(StateView state, int axis, float* alpha) {
  const int side = state.side;
  const std::int64_t lines = static_cast<std::int64_t>(side) * side;
  for (std::int64_t line = blockIdx.x; line < lines; line += gridDim.x) {
    float local[3] = {1.0e-15f, 1.0e-15f, 1.0e-15f};
    int x = 1, y = 1, z = 1;
    if (axis == 0) { y = static_cast<int>(line % side); z = static_cast<int>(line / side); }
    if (axis == 1) { x = static_cast<int>(line % side); z = static_cast<int>(line / side); }
    if (axis == 2) { x = static_cast<int>(line % side); y = static_cast<int>(line / side); }
    for (int coordinate = threadIdx.x; coordinate < side; coordinate += blockDim.x) {
      if (axis == 0) x = coordinate;
      if (axis == 1) y = coordinate;
      if (axis == 2) z = coordinate;
      const Vec5 q = shu3d::load_oriented(state, axis, x, y, z, 0);
      Vec5 flux{};
      float velocity, sound, enthalpy, tangent[2];
      shu3d::primitive_and_flux(q, flux, velocity, sound, enthalpy, tangent);
      local[0] = fmaxf(local[0], fabsf(velocity - sound));
      local[1] = fmaxf(local[1], fabsf(velocity));
      local[2] = fmaxf(local[2], fabsf(velocity + sound));
    }
    extern __shared__ float shared[];
    float* s0 = shared;
    float* s1 = shared + blockDim.x;
    float* s2 = shared + 2 * blockDim.x;
    s0[threadIdx.x] = local[0]; s1[threadIdx.x] = local[1]; s2[threadIdx.x] = local[2];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
        s0[threadIdx.x] = fmaxf(s0[threadIdx.x], s0[threadIdx.x + stride]);
        s1[threadIdx.x] = fmaxf(s1[threadIdx.x], s1[threadIdx.x + stride]);
        s2[threadIdx.x] = fmaxf(s2[threadIdx.x], s2[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      alpha[line * 3] = shu3d::lf_enlargement * s0[0];
      alpha[line * 3 + 1] = shu3d::lf_enlargement * s1[0];
      alpha[line * 3 + 2] = shu3d::lf_enlargement * s2[0];
    }
  }
}

__global__ void rhs_kernel(StateView state, float* result, const float* alpha_x,
                           const float* alpha_y, const float* alpha_z,
                           float inverse_spacing) {
  const int side = state.side;
  const std::int64_t total = state.cells;
  for (std::int64_t t = blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int x = static_cast<int>(t % side);
    const int y = static_cast<int>((t / side) % side);
    const int z = static_cast<int>(t / (static_cast<std::int64_t>(side) * side));
    const std::int64_t lines[3] = {
      static_cast<std::int64_t>(z) * side + y,
      static_cast<std::int64_t>(z) * side + x,
      static_cast<std::int64_t>(y) * side + x
    };
    const float* alphas[3] = {alpha_x, alpha_y, alpha_z};
    float total_rhs[5] = {};
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      float a[3];
#pragma unroll
      for (int k = 0; k < 3; ++k) a[k] = alphas[axis][lines[axis] * 3 + k];
      const Vec5 derivative = shu3d::directional_derivative(
          state, axis, x, y, z, inverse_spacing, a);
      shu3d::add_oriented(total_rhs, derivative, axis);
    }
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      result[static_cast<std::int64_t>(component) * state.cells + t] = total_rhs[component];
    }
  }
}

__global__ void update_kernel(StateView base, StateView stage, StateView deriv,
                              float* output, const float* dt, int which) {
  const std::int64_t total = base.cells;
  const int side = base.side;
  for (std::int64_t t = blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int x = static_cast<int>(t % side);
    const int y = static_cast<int>((t / side) % side);
    const int z = static_cast<int>(t / (static_cast<std::int64_t>(side) * side));
    const Vec5 q0 = shu3d::load_global(base, x, y, z);
    const Vec5 qs = shu3d::load_global(stage, x, y, z);
    const Vec5 r = shu3d::load_global(deriv, x, y, z);
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value;
      if (which == 1) value = q0.v[component] + dt[0] * r.v[component];
      else if (which == 2) value = 0.75f * q0.v[component]
          + 0.25f * (qs.v[component] + dt[0] * r.v[component]);
      else value = (q0.v[component]
          + 2.0f * (qs.v[component] + dt[0] * r.v[component])) / 3.0f;
      output[static_cast<std::int64_t>(component) * total + t] = value;
    }
  }
}

void alphas_and_rhs(StateView q, float* result, float* alpha[3],
                    float inverse_spacing, int blocks) {
  const int alpha_threads = 256;
  const std::size_t shared = 3 * alpha_threads * sizeof(float);
  const int lines = q.side * q.side;
  for (int axis = 0; axis < 3; ++axis) {
    alpha_kernel<<<lines, alpha_threads, shared>>>(q, axis, alpha[axis]);
  }
  rhs_kernel<<<blocks, 128>>>(q, result, alpha[0], alpha[1], alpha[2], inverse_spacing);
}

}  // namespace

RunResult run_cuda(const std::vector<float>& initial, int intervals, int steps) {
  const int side = intervals + 1;
  const std::int64_t cells = static_cast<std::int64_t>(side) * side * side;
  const std::size_t state_bytes = initial.size() * sizeof(float);
  const std::size_t alpha_bytes = static_cast<std::size_t>(side) * side * 3 * sizeof(float);
  float *q, *q1, *q2, *r, *dt, *alpha[3];
  CUDA_CHECK(cudaMalloc(&q, state_bytes)); CUDA_CHECK(cudaMalloc(&q1, state_bytes));
  CUDA_CHECK(cudaMalloc(&q2, state_bytes)); CUDA_CHECK(cudaMalloc(&r, state_bytes));
  CUDA_CHECK(cudaMalloc(&dt, sizeof(float)));
  for (int axis = 0; axis < 3; ++axis) CUDA_CHECK(cudaMalloc(&alpha[axis], alpha_bytes));
  CUDA_CHECK(cudaMemcpy(q, initial.data(), state_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  const int blocks = static_cast<int>((cells + 127) / 128);
  const int capped_blocks = blocks > 65535 ? 65535 : blocks;
  const float inverse_spacing = static_cast<float>(intervals) / 10.0f;
  cudaEvent_t started{}, stopped{};
  CUDA_CHECK(cudaEventCreate(&started));
  CUDA_CHECK(cudaEventCreate(&stopped));
  CUDA_CHECK(cudaEventRecord(started));
  for (int step = 0; step < steps; ++step) {
    CUDA_CHECK(cudaMemsetAsync(dt, 0, sizeof(float)));
    const std::int64_t interior = static_cast<std::int64_t>(intervals) * intervals * intervals;
    const int cfl_blocks = static_cast<int>((interior + 255) / 256);
    timestep_kernel<<<cfl_blocks > 65535 ? 65535 : cfl_blocks, 256>>>(
        StateView{q, intervals, side, cells}, inverse_spacing, dt);
    finish_timestep_kernel<<<1, 1>>>(dt);
    alphas_and_rhs(StateView{q, intervals, side, cells}, r, alpha,
                   inverse_spacing, capped_blocks);
    update_kernel<<<capped_blocks, 128>>>(
        StateView{q, intervals, side, cells}, StateView{q, intervals, side, cells},
        StateView{r, intervals, side, cells}, q1, dt, 1);
    alphas_and_rhs(StateView{q1, intervals, side, cells}, r, alpha,
                   inverse_spacing, capped_blocks);
    update_kernel<<<capped_blocks, 128>>>(
        StateView{q, intervals, side, cells}, StateView{q1, intervals, side, cells},
        StateView{r, intervals, side, cells}, q2, dt, 2);
    alphas_and_rhs(StateView{q2, intervals, side, cells}, r, alpha,
                   inverse_spacing, capped_blocks);
    update_kernel<<<capped_blocks, 128>>>(
        StateView{q, intervals, side, cells}, StateView{q2, intervals, side, cells},
        StateView{r, intervals, side, cells}, q1, dt, 3);
    float* old = q; q = q1; q1 = old;
  }
  CUDA_CHECK(cudaEventRecord(stopped));
  CUDA_CHECK(cudaEventSynchronize(stopped));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, started, stopped));

  RunResult result;
  result.state.resize(initial.size());
  CUDA_CHECK(cudaMemcpy(result.state.data(), q, state_bytes, cudaMemcpyDeviceToHost));
  result.execution_seconds = static_cast<double>(elapsed_ms) * 1.0e-3;
  result.peak_bytes = state_bytes * 4 + alpha_bytes * 3 + sizeof(float);
  cudaFree(q); cudaFree(q1); cudaFree(q2); cudaFree(r); cudaFree(dt);
  for (int axis = 0; axis < 3; ++axis) cudaFree(alpha[axis]);
  CUDA_CHECK(cudaEventDestroy(started));
  CUDA_CHECK(cudaEventDestroy(stopped));
  return result;
}
