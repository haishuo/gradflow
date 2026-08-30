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

#ifndef GRADFLOW_RECOVERY_LEVEL
#define GRADFLOW_RECOVERY_LEVEL 0
#endif

#ifndef GRADFLOW_FACE_THREADS
#define GRADFLOW_FACE_THREADS 256
#endif

#ifndef GRADFLOW_G6_REGISTER_LIMIT
#define GRADFLOW_G6_REGISTER_LIMIT 0
#endif

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

const char* contract_name() {
#ifdef GRADFLOW_G6_CONTRACT
  return GRADFLOW_G6_CONTRACT;
#endif
  if (GRADFLOW_RECOVERY_LEVEL == 1) {
    return "r1_unique_strict_f32_component_shared_density_local_lf_forward_euler_v1";
  }
  if (GRADFLOW_RECOVERY_LEVEL == 2) {
    return "r2_unique_strict_f32_component_separate_weights_local_lf_forward_euler_v1";
  }
  if (GRADFLOW_RECOVERY_LEVEL == 3) {
    return "r3_unique_strict_f32_roe_characteristic_local_lf_forward_euler_v1";
  }
  if (GRADFLOW_RECOVERY_LEVEL == 4) {
    return "r4_unique_strict_f32_roe_characteristic_line_lf_forward_euler_v1";
  }
  if (GRADFLOW_RECOVERY_LEVEL == 5) {
    return "r5_unique_strict_f32_roe_characteristic_line_lf_ssprk3_face_once_v1";
  }
  if (GRADFLOW_RECOVERY_LEVEL == 6) {
    return "r6_unique_strict_f32_shu_characteristic_line_lf_ssprk3_face_once_v1";
  }
  if (GRADFLOW_RECOVERY_LEVEL == 7) {
    return "r6q_arbitrary_state_rhs_unique_strict_f32_shu_face_once_v1";
  }
  if (GRADFLOW_RECOVERY_LEVEL == 8) {
    return "p1_shared_pencil_unique_strict_f32_shu_fused_update_v1";
  }
  return "u0_unique_f32_component_shared_density_local_lf_forward_euler_v1";
}

struct Options {
  int size = 128;
  int steps = 1;
  int warmups = 5;
  int repetitions = 30;
  std::string input_state;
  std::string mode = "step";
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

__device__ __forceinline__ void orient_state(
    const float global[kComponents], int axis, float oriented[kComponents]) {
  oriented[0] = global[0];
  oriented[4] = global[4];
  if (axis == 0) {
    oriented[1] = global[1];
    oriented[2] = global[2];
    oriented[3] = global[3];
  } else if (axis == 1) {
    oriented[1] = global[2];
    oriented[2] = global[1];
    oriented[3] = global[3];
  } else {
    oriented[1] = global[3];
    oriented[2] = global[1];
    oriented[3] = global[2];
  }
}

__device__ __forceinline__ void unorient_flux(
    const float oriented[kComponents], int axis, float global[kComponents]) {
  global[0] = oriented[0];
  global[4] = oriented[4];
  if (axis == 0) {
    global[1] = oriented[1];
    global[2] = oriented[2];
    global[3] = oriented[3];
  } else if (axis == 1) {
    global[1] = oriented[2];
    global[2] = oriented[1];
    global[3] = oriented[3];
  } else {
    global[1] = oriented[2];
    global[2] = oriented[3];
    global[3] = oriented[1];
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

__device__ __forceinline__ float shu_nonlinear_correction(const float h[4]) {
  const float t1 = h[0] - h[1];
  const float t2 = h[1] - h[2];
  const float t3 = h[2] - h[3];
  const float a = h[0] - 3.0f * h[1];
  const float b = h[1] + h[2];
  const float c = 3.0f * h[2] - h[3];
  const float indicator1 = 13.0f * t1 * t1 + 3.0f * a * a;
  const float indicator2 = 13.0f * t2 * t2 + 3.0f * b * b;
  const float indicator3 = 13.0f * t3 * t3 + 3.0f * c * c;
  const float d1 = (kEpsilon + indicator1) * (kEpsilon + indicator1);
  const float d2 = (kEpsilon + indicator2) * (kEpsilon + indicator2);
  const float d3 = (kEpsilon + indicator3) * (kEpsilon + indicator3);
  float weight1 = d2 * d3;
  const float weight2 = 6.0f * d1 * d3;
  float weight3 = 3.0f * d1 * d2;
  const float reciprocal = 1.0f / (weight1 + weight2 + weight3);
  weight1 *= reciprocal;
  weight3 *= reciprocal;
  return (weight1 * (t2 - t1) +
          (0.5f * weight3 - 0.25f) * (t3 - t2)) /
         3.0f;
}

__device__ __forceinline__ void roe_matrices(
    const float left_state[kComponents], const float right_state[kComponents],
    float left[5][5], float right[5][5]) {
  const float inverse_left = 1.0f / left_state[0];
  const float inverse_right = 1.0f / right_state[0];
  const float ul = left_state[1] * inverse_left;
  const float vl = left_state[2] * inverse_left;
  const float wl = left_state[3] * inverse_left;
  const float ur = right_state[1] * inverse_right;
  const float vr = right_state[2] * inverse_right;
  const float wr = right_state[3] * inverse_right;
  const float pl = kGammaMinusOne *
      (left_state[4] -
       0.5f * left_state[0] * (ul * ul + vl * vl + wl * wl));
  const float pr = kGammaMinusOne *
      (right_state[4] -
       0.5f * right_state[0] * (ur * ur + vr * vr + wr * wr));
  const float hl = (pl + left_state[4]) * inverse_left;
  const float hr = (pr + right_state[4]) * inverse_right;
  const float root_left = sqrtf(left_state[0]);
  const float root_right = sqrtf(right_state[0]);
  const float fraction = root_left / (root_left + root_right);
  const float complement = 1.0f - fraction;
  const float u = fraction * ul + complement * ur;
  const float v = fraction * vl + complement * vr;
  const float w = fraction * wl + complement * wr;
  const float h = fraction * hl + complement * hr;
  const float kinetic = 0.5f * (u * u + v * v + w * w);
  const float sound = sqrtf(kGammaMinusOne * (h - kinetic));

  right[0][0] = 1.0f; right[1][0] = u - sound; right[2][0] = v;
  right[3][0] = w; right[4][0] = h - u * sound;
  right[0][1] = 0.0f; right[1][1] = 0.0f; right[2][1] = 1.0f;
  right[3][1] = 0.0f; right[4][1] = v;
  right[0][2] = 0.0f; right[1][2] = 0.0f; right[2][2] = 0.0f;
  right[3][2] = 1.0f; right[4][2] = w;
  right[0][3] = 1.0f; right[1][3] = u; right[2][3] = v;
  right[3][3] = w; right[4][3] = kinetic;
  right[0][4] = 1.0f; right[1][4] = u + sound; right[2][4] = v;
  right[3][4] = w; right[4][4] = h + u * sound;

  const float reciprocal_sound = 1.0f / sound;
  const float b1 = kGammaMinusOne * reciprocal_sound * reciprocal_sound;
  const float b2 = kinetic * b1;
  left[0][0] = 0.5f * (b2 + u * reciprocal_sound);
  left[0][1] = -0.5f * (b1 * u + reciprocal_sound);
  left[0][2] = -0.5f * b1 * v; left[0][3] = -0.5f * b1 * w;
  left[0][4] = 0.5f * b1;
  left[1][0] = -v; left[1][1] = 0.0f; left[1][2] = 1.0f;
  left[1][3] = 0.0f; left[1][4] = 0.0f;
  left[2][0] = -w; left[2][1] = 0.0f; left[2][2] = 0.0f;
  left[2][3] = 1.0f; left[2][4] = 0.0f;
  left[3][0] = 1.0f - b2; left[3][1] = b1 * u;
  left[3][2] = b1 * v; left[3][3] = b1 * w; left[3][4] = -b1;
  left[4][0] = 0.5f * (b2 - u * reciprocal_sound);
  left[4][1] = -0.5f * (b1 * u - reciprocal_sound);
  left[4][2] = -0.5f * b1 * v; left[4][3] = -0.5f * b1 * w;
  left[4][4] = 0.5f * b1;
}

__device__ __forceinline__ void u0_face_flux(
    const DeviceState& state, int x, int y, int z, int axis,
    const float* line_alpha, float output[kComponents]) {
  float q[6][kComponents];
  float flux[6][kComponents];
  float density[6];
  float alpha = 0.0f;
#pragma unroll
  for (int sample = 0; sample < 6; ++sample) {
    const std::int64_t cell = offset_index(x, y, z, axis, sample - 2, state.n);
    float global[kComponents];
    load_state(state, cell, global);
#if GRADFLOW_RECOVERY_LEVEL >= 3
    orient_state(global, axis, q[sample]);
#else
#pragma unroll
    for (int component = 0; component < kComponents; ++component) {
      q[sample][component] = global[component];
    }
#endif
    density[sample] = q[sample][0];
#if GRADFLOW_RECOVERY_LEVEL >= 3
    alpha = fmaxf(alpha, primitive_flux(q[sample], 0, flux[sample]));
#else
    alpha = fmaxf(alpha, primitive_flux(q[sample], axis, flux[sample]));
#endif
  }

#if GRADFLOW_RECOVERY_LEVEL >= 3
  float left[5][5];
  float right[5][5];
  roe_matrices(q[2], q[3], left, right);
  float characteristic[5];
#pragma unroll
  for (int family = 0; family < 5; ++family) {
    const float split_alpha = line_alpha == nullptr
        ? alpha
        : (family == 0 ? line_alpha[0]
                       : (family == 4 ? line_alpha[2] : line_alpha[1]));
#if GRADFLOW_RECOVERY_LEVEL >= 6
    float positive[4];
    float negative[4];
#pragma unroll
    for (int candidate = 0; candidate < 4; ++candidate) {
      const int negative_candidate = 4 - candidate;
      float projected_positive = 0.0f;
      float projected_negative = 0.0f;
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        const float positive_flux_difference =
            flux[candidate + 1][component] - flux[candidate][component];
        const float positive_state_difference =
            q[candidate + 1][component] - q[candidate][component];
        const float negative_flux_difference =
            flux[negative_candidate + 1][component] -
            flux[negative_candidate][component];
        const float negative_state_difference =
            q[negative_candidate + 1][component] -
            q[negative_candidate][component];
        projected_positive += left[family][component] *
            (0.5f * (positive_flux_difference +
                     split_alpha * positive_state_difference));
        projected_negative += left[family][component] *
            (0.5f * (negative_flux_difference +
                     split_alpha * negative_state_difference) -
             negative_flux_difference);
      }
      positive[candidate] = projected_positive;
      negative[candidate] = projected_negative;
    }
    characteristic[family] = shu_nonlinear_correction(positive) +
                             shu_nonlinear_correction(negative);
#else
    float positive[5];
    float negative[5];
#pragma unroll
    for (int sample = 0; sample < 5; ++sample) {
      float projected_positive = 0.0f;
      float projected_negative = 0.0f;
      const int reversed = 5 - sample;
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        projected_positive += left[family][component] *
            (0.5f * (flux[sample][component] +
                     split_alpha * q[sample][component]));
        projected_negative += left[family][component] *
            (0.5f * (flux[reversed][component] -
                     split_alpha * q[reversed][component]));
      }
      positive[sample] = projected_positive;
      negative[sample] = projected_negative;
    }
    float positive_weights[3];
    float negative_weights[3];
    js_weights(positive[0], positive[1], positive[2], positive[3], positive[4],
               positive_weights);
    js_weights(negative[0], negative[1], negative[2], negative[3], negative[4],
               negative_weights);
    characteristic[family] = reconstruct_left(positive, positive_weights) +
                             reconstruct_left(negative, negative_weights);
#endif
  }
  float oriented_output[5];
#pragma unroll
  for (int component = 0; component < 5; ++component) {
    float value = 0.0f;
#pragma unroll
    for (int family = 0; family < 5; ++family) {
      value += right[component][family] * characteristic[family];
    }
#if GRADFLOW_RECOVERY_LEVEL >= 6
    value += (-flux[1][component] +
              7.0f * (flux[2][component] + flux[3][component]) -
              flux[4][component]) /
             12.0f;
#endif
    oriented_output[component] = value;
  }
  unorient_flux(oriented_output, axis, output);
#else

#if GRADFLOW_RECOVERY_LEVEL < 2
  float positive_weights[3];
  float negative_weights[3];
  js_weights(density[0], density[1], density[2], density[3], density[4],
             positive_weights);
  js_weights(density[5], density[4], density[3], density[2], density[1],
             negative_weights);
#endif

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
#if GRADFLOW_RECOVERY_LEVEL >= 2
    float positive_weights[3];
    float negative_weights[3];
    js_weights(positive[0], positive[1], positive[2], positive[3], positive[4],
               positive_weights);
    js_weights(negative[0], negative[1], negative[2], negative[3], negative[4],
               negative_weights);
#endif
    output[component] = reconstruct_left(positive, positive_weights) +
                        reconstruct_left(negative, negative_weights);
  }
#endif
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

__global__ void alpha_kernel(DeviceState state, int axis, float* alphas) {
  const std::int64_t lines = static_cast<std::int64_t>(state.n) * state.n;
  for (std::int64_t line = blockIdx.x; line < lines; line += gridDim.x) {
    float local_minus = 1.0e-15f;
    float local_center = 1.0e-15f;
    float local_plus = 1.0e-15f;
    for (int coordinate = threadIdx.x; coordinate < state.n;
         coordinate += blockDim.x) {
      int x = 0;
      int y = 0;
      int z = 0;
      if (axis == 0) {
        x = coordinate;
        y = static_cast<int>(line % state.n);
        z = static_cast<int>(line / state.n);
      } else if (axis == 1) {
        x = static_cast<int>(line % state.n);
        y = coordinate;
        z = static_cast<int>(line / state.n);
      } else {
        x = static_cast<int>(line % state.n);
        y = static_cast<int>(line / state.n);
        z = coordinate;
      }
      float global[kComponents];
      float oriented[kComponents];
      load_state(state, index3(x, y, z, state.n), global);
      orient_state(global, axis, oriented);
      const float inverse_density = 1.0f / oriented[0];
      const float velocity = oriented[1] * inverse_density;
      const float tangent0 = oriented[2] * inverse_density;
      const float tangent1 = oriented[3] * inverse_density;
      const float pressure = kGammaMinusOne *
          (oriented[4] - 0.5f * oriented[0] *
              (velocity * velocity + tangent0 * tangent0 +
               tangent1 * tangent1));
      const float sound = sqrtf(kGamma * pressure * inverse_density);
      local_minus = fmaxf(local_minus, fabsf(velocity - sound));
      local_center = fmaxf(local_center, fabsf(velocity));
      local_plus = fmaxf(local_plus, fabsf(velocity + sound));
    }

    extern __shared__ float shared[];
    float* minus = shared;
    float* center = shared + blockDim.x;
    float* plus = shared + 2 * blockDim.x;
    minus[threadIdx.x] = local_minus;
    center[threadIdx.x] = local_center;
    plus[threadIdx.x] = local_plus;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
        minus[threadIdx.x] =
            fmaxf(minus[threadIdx.x], minus[threadIdx.x + stride]);
        center[threadIdx.x] =
            fmaxf(center[threadIdx.x], center[threadIdx.x + stride]);
        plus[threadIdx.x] =
            fmaxf(plus[threadIdx.x], plus[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      alphas[line * 3] = 1.1f * minus[0];
      alphas[line * 3 + 1] = 1.1f * center[0];
      alphas[line * 3 + 2] = 1.1f * plus[0];
    }
  }
}

__global__ void face_kernel(DeviceState state, const float* alphas,
                            float* faces) {
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
      const float* line_alpha = nullptr;
#if GRADFLOW_RECOVERY_LEVEL >= 4
      const std::int64_t line = axis == 0
          ? static_cast<std::int64_t>(z) * state.n + y
          : (axis == 1 ? static_cast<std::int64_t>(z) * state.n + x
                       : static_cast<std::int64_t>(y) * state.n + x);
      line_alpha = alphas +
          (static_cast<std::int64_t>(axis) * state.n * state.n + line) * 3;
#endif
      u0_face_flux(state, x, y, z, axis, line_alpha, numerical);
#pragma unroll
      for (int component = 0; component < kComponents; ++component) {
        faces[(static_cast<std::int64_t>(axis) * kComponents + component) *
                  state.cells +
              cell] = numerical[component];
      }
    }
  }
}

__global__ void pencil_kernel(DeviceState base_state, DeviceState state,
                              const float* alphas, const float* dt,
                              float inverse_spacing, float* partial,
                              float* output, int axis, int stage) {
  const std::int64_t line = blockIdx.x;
  const int coordinate = threadIdx.x;
  const bool active = coordinate < state.n;
  int x = 0;
  int y = 0;
  int z = 0;
  if (axis == 0) {
    x = coordinate;
    y = static_cast<int>(line % state.n);
    z = static_cast<int>(line / state.n);
  } else if (axis == 1) {
    x = static_cast<int>(line % state.n);
    y = coordinate;
    z = static_cast<int>(line / state.n);
  } else {
    x = static_cast<int>(line % state.n);
    y = static_cast<int>(line / state.n);
    z = coordinate;
  }

  extern __shared__ float shared_faces[];
  if (active) {
    float numerical[kComponents];
    const float* line_alpha = alphas + line * 3;
    u0_face_flux(state, x, y, z, axis, line_alpha, numerical);
#pragma unroll
    for (int component = 0; component < kComponents; ++component) {
      shared_faces[component * state.n + coordinate] = numerical[component];
    }
  }
  __syncthreads();

  if (!active) return;
  const int left = wrap(coordinate - 1, state.n);
  const std::int64_t cell = index3(x, y, z, state.n);
#pragma unroll
  for (int component = 0; component < kComponents; ++component) {
    const float divergence =
        shared_faces[component * state.n + coordinate] -
        shared_faces[component * state.n + left];
    const std::int64_t element =
        static_cast<std::int64_t>(component) * state.cells + cell;
    if (axis == 0) {
      partial[element] = divergence;
    } else if (axis == 1) {
      partial[element] += divergence;
    } else {
      const float total_divergence = partial[element] + divergence;
      if (stage == 0) {
        output[element] = -inverse_spacing * total_divergence;
      } else {
        const float euler_value =
            state.data[element] -
            dt[0] * inverse_spacing * total_divergence;
        if (stage == 1) {
          output[element] = euler_value;
        } else if (stage == 2) {
          output[element] = 0.75f * base_state.data[element] +
                            0.25f * euler_value;
        } else {
          output[element] =
              (base_state.data[element] + 2.0f * euler_value) / 3.0f;
        }
      }
    }
  }
}

__global__ void update_kernel(DeviceState base_state, DeviceState state,
                              const float* faces,
                              const float* dt, float inverse_spacing,
                              float* output, int stage) {
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
      const std::int64_t element =
          static_cast<std::int64_t>(component) * state.cells + cell;
      const float euler_value =
          state.data[element] - dt[0] * inverse_spacing * divergence;
      if (stage == 0) {
        output[element] = -inverse_spacing * divergence;
      } else if (stage == 1) {
        output[element] = euler_value;
      } else if (stage == 2) {
        output[element] = 0.75f * base_state.data[element] +
                          0.25f * euler_value;
      } else {
        output[element] =
            (base_state.data[element] + 2.0f * euler_value) / 3.0f;
      }
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
    else if (argument == "--input-state")
      options.input_state = require_value();
    else if (argument == "--mode")
      options.mode = require_value();
    else if (argument == "--output-initial")
      options.output_initial = require_value();
    else if (argument == "--output-state")
      options.output_state = require_value();
    else throw std::runtime_error("unknown argument: " + argument);
  }
  if (options.size < 6 || options.steps < 1 || options.warmups < 0 ||
      options.repetitions < 1) {
    throw std::runtime_error("invalid size, steps, warmups, or repetitions");
  }
  if (options.mode != "step" && options.mode != "rhs") {
    throw std::runtime_error("mode must be step or rhs");
  }
  if (options.mode == "rhs" && options.steps != 1) {
    throw std::runtime_error("rhs mode requires exactly one step");
  }
#if GRADFLOW_RECOVERY_LEVEL >= 8
  if (options.size > 256) {
    throw std::runtime_error("shared-pencil contract requires size <= 256");
  }
#endif
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

std::vector<float> read_state(const std::string& path, std::size_t elements) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open input state: " + path);
  const std::streamsize expected =
      static_cast<std::streamsize>(elements * sizeof(float));
  if (input.tellg() != expected) {
    throw std::runtime_error("input state has unexpected byte length");
  }
  input.seekg(0);
  std::vector<float> state(elements);
  input.read(reinterpret_cast<char*>(state.data()), expected);
  if (!input) throw std::runtime_error("failed to read input state: " + path);
  return state;
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
    std::vector<float> initial = options.input_state.empty()
        ? initialize_vortex(options.size)
        : read_state(options.input_state, state_elements);
    std::vector<float> result(state_elements);
    if (!options.output_initial.empty()) {
      write_state(options.output_initial, initial);
    }

    float *d_initial = nullptr, *d_q = nullptr, *d_next = nullptr;
    float *d_stage1 = nullptr, *d_stage2 = nullptr;
    float *d_faces = nullptr, *d_dt = nullptr, *d_alphas = nullptr;
    CUDA_CHECK(cudaMalloc(&d_initial, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_q, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_next, state_bytes));
#if GRADFLOW_RECOVERY_LEVEL >= 5
    CUDA_CHECK(cudaMalloc(&d_stage1, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_stage2, state_bytes));
#endif
#if GRADFLOW_RECOVERY_LEVEL < 8
    CUDA_CHECK(cudaMalloc(&d_faces, face_bytes));
#endif
    CUDA_CHECK(cudaMalloc(&d_dt, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dt, 0, sizeof(float)));
#if GRADFLOW_RECOVERY_LEVEL >= 4
    const std::size_t alpha_bytes =
        static_cast<std::size_t>(kAxes) * 3 * options.size * options.size *
        sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_alphas, alpha_bytes));
#else
    const std::size_t alpha_bytes = 0;
#endif
    CUDA_CHECK(cudaMemcpy(d_initial, initial.data(), state_bytes,
                          cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    int blocks = static_cast<int>((cells + threads - 1) / threads);
    blocks = std::min(blocks, 65535);
    constexpr int face_threads = GRADFLOW_FACE_THREADS;
    int face_blocks = static_cast<int>((cells + face_threads - 1) / face_threads);
    face_blocks = std::min(face_blocks, 65535);
#ifdef GRADFLOW_G6_CONTRACT
    cudaFuncAttributes face_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&face_attributes, face_kernel));
    int face_active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &face_active_blocks_per_sm, face_kernel, face_threads, 0));
    const double face_theoretical_occupancy_percent =
        100.0 * face_active_blocks_per_sm * face_threads /
        properties.maxThreadsPerMultiProcessor;
#endif
    const float inverse_spacing = static_cast<float>(options.size) / 10.0f;

    auto launch = [&](float*& q, float*& next) {
#if GRADFLOW_RECOVERY_LEVEL >= 8
      auto pencil_stage = [&](float* base_state, float* stage_state,
                              float* scratch, float* output, int stage) {
        constexpr int alpha_threads = 256;
        const int line_blocks = options.size * options.size;
        for (int axis = 0; axis < kAxes; ++axis) {
          float* axis_alphas =
              d_alphas + static_cast<std::int64_t>(axis) * 3 * options.size *
                  options.size;
          alpha_kernel<<<line_blocks, alpha_threads,
                         3 * alpha_threads * sizeof(float)>>>(
              DeviceState{stage_state, options.size, cells}, axis,
              axis_alphas);
          pencil_kernel<<<line_blocks, threads,
                          kComponents * options.size * sizeof(float)>>>(
              DeviceState{base_state, options.size, cells},
              DeviceState{stage_state, options.size, cells}, axis_alphas,
              d_dt, inverse_spacing, scratch, output, axis, stage);
        }
      };
      if (options.mode == "rhs") {
        pencil_stage(q, q, next, d_stage1, 0);
        q = d_stage1;
        return;
      }
      for (int step = 0; step < options.steps; ++step) {
        CUDA_CHECK(cudaMemsetAsync(d_dt, 0, sizeof(float)));
        cfl_kernel<<<blocks, threads, threads * sizeof(float)>>>(
            DeviceState{q, options.size, cells}, inverse_spacing, d_dt);
        finish_cfl_kernel<<<1, 1>>>(d_dt);
        pencil_stage(q, q, next, d_stage1, 1);
        pencil_stage(q, d_stage1, next, d_stage2, 2);
        pencil_stage(q, d_stage2, d_stage1, next, 3);
        std::swap(q, next);
      }
#else
      auto spatial = [&](float* stage_state) {
#if GRADFLOW_RECOVERY_LEVEL >= 4
        constexpr int alpha_threads = 256;
        const int alpha_blocks = options.size * options.size;
        for (int axis = 0; axis < kAxes; ++axis) {
          alpha_kernel<<<alpha_blocks, alpha_threads,
                         3 * alpha_threads * sizeof(float)>>>(
              DeviceState{stage_state, options.size, cells}, axis,
              d_alphas + static_cast<std::int64_t>(axis) * 3 * options.size *
                  options.size);
        }
#endif
        face_kernel<<<face_blocks, face_threads>>>(
            DeviceState{stage_state, options.size, cells}, d_alphas, d_faces);
      };
      if (options.mode == "rhs") {
        spatial(q);
        update_kernel<<<blocks, threads>>>(
            DeviceState{q, options.size, cells},
            DeviceState{q, options.size, cells}, d_faces, d_dt,
            inverse_spacing, next, 0);
        std::swap(q, next);
        return;
      }
      for (int step = 0; step < options.steps; ++step) {
        CUDA_CHECK(cudaMemsetAsync(d_dt, 0, sizeof(float)));
        cfl_kernel<<<blocks, threads, threads * sizeof(float)>>>(
            DeviceState{q, options.size, cells}, inverse_spacing, d_dt);
        finish_cfl_kernel<<<1, 1>>>(d_dt);
#if GRADFLOW_RECOVERY_LEVEL >= 5
        spatial(q);
        update_kernel<<<blocks, threads>>>(
            DeviceState{q, options.size, cells},
            DeviceState{q, options.size, cells}, d_faces, d_dt,
            inverse_spacing, d_stage1, 1);
        spatial(d_stage1);
        update_kernel<<<blocks, threads>>>(
            DeviceState{q, options.size, cells},
            DeviceState{d_stage1, options.size, cells}, d_faces, d_dt,
            inverse_spacing, d_stage2, 2);
        spatial(d_stage2);
        update_kernel<<<blocks, threads>>>(
            DeviceState{q, options.size, cells},
            DeviceState{d_stage2, options.size, cells}, d_faces, d_dt,
            inverse_spacing, next, 3);
#else
        spatial(q);
        update_kernel<<<blocks, threads>>>(
            DeviceState{q, options.size, cells},
            DeviceState{q, options.size, cells}, d_faces, d_dt,
            inverse_spacing, next, 1);
#endif
        std::swap(q, next);
      }
#endif
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
    std::cout << "  \"contract\": \"" << contract_name() << "\",\n";
    std::cout << "  \"mode\": \"" << options.mode << "\",\n";
    std::cout << "  \"gpu\": \"" << properties.name << "\",\n";
    std::cout << "  \"compute_capability\": \"" << properties.major << "."
              << properties.minor << "\",\n";
    std::cout << "  \"size\": " << options.size << ",\n";
    std::cout << "  \"cells\": " << cells << ",\n";
    std::cout << "  \"steps\": " << options.steps << ",\n";
    std::cout << "  \"warmups\": " << options.warmups << ",\n";
    std::cout << "  \"repetitions\": " << options.repetitions << ",\n";
#ifdef GRADFLOW_G6_CONTRACT
    std::cout << "  \"face_threads\": " << face_threads << ",\n";
    std::cout << "  \"declared_register_limit\": "
              << GRADFLOW_G6_REGISTER_LIMIT << ",\n";
    std::cout << "  \"compiled_face_registers_per_thread\": "
              << face_attributes.numRegs << ",\n";
    std::cout << "  \"compiled_face_local_bytes_per_thread\": "
              << face_attributes.localSizeBytes << ",\n";
    std::cout << "  \"compiled_face_static_shared_bytes\": "
              << face_attributes.sharedSizeBytes << ",\n";
    std::cout << "  \"face_active_blocks_per_sm\": "
              << face_active_blocks_per_sm << ",\n";
    std::cout << "  \"face_theoretical_occupancy_percent\": "
              << face_theoretical_occupancy_percent << ",\n";
#endif
    std::cout << "  \"median_device_ms\": " << median_ms << ",\n";
    std::cout << "  \"cells_per_second\": " << cells_per_second << ",\n";
    std::cout << "  \"directional_faces_per_second\": " << faces_per_second
              << ",\n";
    std::cout << "  \"final_dt\": " << final_dt << ",\n";
    std::cout << "  \"finite\": " << (finite ? "true" : "false") << ",\n";
    std::cout << "  \"checksum_float64\": " << checksum << ",\n";
    std::cout << "  \"maximum_absolute\": " << maximum_absolute << ",\n";
    std::cout << "  \"peak_allocated_bytes\": "
              << ((GRADFLOW_RECOVERY_LEVEL >= 8
                       ? 5 * state_bytes
                       : (GRADFLOW_RECOVERY_LEVEL >= 5 ? 5 : 3) * state_bytes +
                             face_bytes) +
                  sizeof(float) + alpha_bytes)
              << ",\n";
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
    if (d_stage1 != nullptr) CUDA_CHECK(cudaFree(d_stage1));
    if (d_stage2 != nullptr) CUDA_CHECK(cudaFree(d_stage2));
    if (d_faces != nullptr) CUDA_CHECK(cudaFree(d_faces));
    CUDA_CHECK(cudaFree(d_dt));
    if (d_alphas != nullptr) CUDA_CHECK(cudaFree(d_alphas));
    return finite ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
