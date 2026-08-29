#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define SHU_HD __host__ __device__ __forceinline__
#else
#define SHU_HD inline
#endif

namespace shu3d {

constexpr int equations = 5;
constexpr float gamma_value = 1.4f;
constexpr float gamma_minus_one = 0.4f;
constexpr float weno_epsilon = 1.0e-6f;
constexpr float lf_enlargement = 1.1f;

struct Vec5 {
  float v[equations];
};

struct StateView {
  const float* data;
  int intervals;
  int side;
  std::int64_t cells;
};

SHU_HD int periodic_coordinate(int coordinate, int intervals) {
  while (coordinate < 0) coordinate += intervals;
  while (coordinate > intervals) coordinate -= intervals;
  return coordinate == 0 ? intervals : coordinate;
}

SHU_HD std::int64_t cell_index(const StateView& q, int x, int y, int z) {
  x = periodic_coordinate(x, q.intervals);
  y = periodic_coordinate(y, q.intervals);
  z = periodic_coordinate(z, q.intervals);
  return (static_cast<std::int64_t>(z) * q.side + y) * q.side + x;
}

SHU_HD Vec5 load_global(const StateView& q, int x, int y, int z) {
  const std::int64_t cell = cell_index(q, x, y, z);
  Vec5 result{};
#pragma unroll
  for (int component = 0; component < equations; ++component) {
    result.v[component] = q.data[static_cast<std::int64_t>(component) * q.cells + cell];
  }
  return result;
}

SHU_HD Vec5 orient(const Vec5& global, int axis) {
  Vec5 result{};
  result.v[0] = global.v[0];
  result.v[4] = global.v[4];
  if (axis == 0) {
    result.v[1] = global.v[1];
    result.v[2] = global.v[2];
    result.v[3] = global.v[3];
  } else if (axis == 1) {
    result.v[1] = global.v[2];
    result.v[2] = global.v[1];
    result.v[3] = global.v[3];
  } else {
    result.v[1] = global.v[3];
    result.v[2] = global.v[1];
    result.v[3] = global.v[2];
  }
  return result;
}

SHU_HD Vec5 load_oriented(const StateView& q, int axis, int x, int y, int z,
                          int offset) {
  if (axis == 0) x += offset;
  if (axis == 1) y += offset;
  if (axis == 2) z += offset;
  return orient(load_global(q, x, y, z), axis);
}

SHU_HD void primitive_and_flux(const Vec5& q, Vec5& flux, float& velocity,
                               float& sound, float& enthalpy,
                               float tangential[2]) {
  const float inverse_density = 1.0f / q.v[0];
  velocity = q.v[1] * inverse_density;
  tangential[0] = q.v[2] * inverse_density;
  tangential[1] = q.v[3] * inverse_density;
  const float velocity_squared = velocity * velocity
      + tangential[0] * tangential[0] + tangential[1] * tangential[1];
  const float pressure = gamma_minus_one
      * (q.v[4] - 0.5f * q.v[0] * velocity_squared);
  sound = sqrtf(gamma_value * pressure * inverse_density);
  enthalpy = (pressure + q.v[4]) * inverse_density;
  flux.v[0] = q.v[1];
  flux.v[1] = q.v[1] * velocity + pressure;
  flux.v[2] = q.v[2] * velocity;
  flux.v[3] = q.v[3] * velocity;
  flux.v[4] = velocity * (pressure + q.v[4]);
}

SHU_HD float nonlinear_correction(const float h[4]) {
  const float t1 = h[0] - h[1];
  const float t2 = h[1] - h[2];
  const float t3 = h[2] - h[3];
  const float a = h[0] - 3.0f * h[1];
  const float b = h[1] + h[2];
  const float c = 3.0f * h[2] - h[3];
  const float indicator1 = 13.0f * t1 * t1 + 3.0f * a * a;
  const float indicator2 = 13.0f * t2 * t2 + 3.0f * b * b;
  const float indicator3 = 13.0f * t3 * t3 + 3.0f * c * c;
  const float d1 = (weno_epsilon + indicator1) * (weno_epsilon + indicator1);
  const float d2 = (weno_epsilon + indicator2) * (weno_epsilon + indicator2);
  const float d3 = (weno_epsilon + indicator3) * (weno_epsilon + indicator3);
  float weight1 = d2 * d3;
  const float weight2 = 6.0f * d1 * d3;
  float weight3 = 3.0f * d1 * d2;
  const float reciprocal_sum = 1.0f / (weight1 + weight2 + weight3);
  weight1 *= reciprocal_sum;
  weight3 *= reciprocal_sum;
  return (weight1 * (t2 - t1)
      + (0.5f * weight3 - 0.25f) * (t3 - t2)) / 3.0f;
}

SHU_HD void roe_matrices(const Vec5& left_state, const Vec5& right_state,
                         float left[5][5], float right[5][5]) {
  Vec5 unused_flux{};
  float ul, cl, hl, utl[2];
  float ur, cr, hr, utr[2];
  primitive_and_flux(left_state, unused_flux, ul, cl, hl, utl);
  primitive_and_flux(right_state, unused_flux, ur, cr, hr, utr);
  const float root_left = sqrtf(left_state.v[0]);
  const float root_right = sqrtf(right_state.v[0]);
  const float fraction = root_left / (root_left + root_right);
  const float one_minus = 1.0f - fraction;
  const float u = fraction * ul + one_minus * ur;
  const float v = fraction * utl[0] + one_minus * utr[0];
  const float w = fraction * utl[1] + one_minus * utr[1];
  const float h = fraction * hl + one_minus * hr;
  const float q = 0.5f * (u * u + v * v + w * w);
  const float c = sqrtf(gamma_minus_one * (h - q));

  right[0][0] = 1.0f; right[1][0] = u - c; right[2][0] = v;
  right[3][0] = w; right[4][0] = h - u * c;
  right[0][1] = 0.0f; right[1][1] = 0.0f; right[2][1] = 1.0f;
  right[3][1] = 0.0f; right[4][1] = v;
  right[0][2] = 0.0f; right[1][2] = 0.0f; right[2][2] = 0.0f;
  right[3][2] = 1.0f; right[4][2] = w;
  right[0][3] = 1.0f; right[1][3] = u; right[2][3] = v;
  right[3][3] = w; right[4][3] = q;
  right[0][4] = 1.0f; right[1][4] = u + c; right[2][4] = v;
  right[3][4] = w; right[4][4] = h + u * c;

  const float reciprocal_sound = 1.0f / c;
  const float b1 = gamma_minus_one * reciprocal_sound * reciprocal_sound;
  const float b2 = q * b1;
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

SHU_HD Vec5 numerical_flux(const StateView& state, int axis, int x, int y,
                           int z, int face, const float alpha3[3]) {
  Vec5 q[6];
  Vec5 flux[6];
  float velocity[6], sound[6], enthalpy[6], tangent[6][2];
#pragma unroll
  for (int sample = 0; sample < 6; ++sample) {
    int offset = face - 2 + sample;
    int center = axis == 0 ? x : (axis == 1 ? y : z);
    q[sample] = load_oriented(state, axis, x, y, z, offset - center);
    primitive_and_flux(q[sample], flux[sample], velocity[sample], sound[sample],
                       enthalpy[sample], tangent[sample]);
  }

  float left[5][5], right[5][5];
  roe_matrices(q[2], q[3], left, right);
  float characteristic[5];
#pragma unroll
  for (int family = 0; family < 5; ++family) {
    const float alpha = family == 0 ? alpha3[0]
        : (family == 4 ? alpha3[2] : alpha3[1]);
    float positive[4], negative[4];
#pragma unroll
    for (int candidate = 0; candidate < 4; ++candidate) {
      const int pd = candidate;
      const int nd = 4 - candidate;
      float projected_positive = 0.0f;
      float projected_negative = 0.0f;
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        const float pdf = flux[pd + 1].v[component] - flux[pd].v[component];
        const float pdu = q[pd + 1].v[component] - q[pd].v[component];
        const float split_positive = 0.5f * (pdf + alpha * pdu);
        const float ndf = flux[nd + 1].v[component] - flux[nd].v[component];
        const float ndu = q[nd + 1].v[component] - q[nd].v[component];
        const float split_negative = 0.5f * (ndf + alpha * ndu) - ndf;
        projected_positive += left[family][component] * split_positive;
        projected_negative += left[family][component] * split_negative;
      }
      positive[candidate] = projected_positive;
      negative[candidate] = projected_negative;
    }
    characteristic[family] = nonlinear_correction(positive)
        + nonlinear_correction(negative);
  }

  Vec5 result{};
#pragma unroll
  for (int component = 0; component < 5; ++component) {
    float nonlinear = 0.0f;
#pragma unroll
    for (int family = 0; family < 5; ++family) {
      nonlinear += right[component][family] * characteristic[family];
    }
    const float central = (-flux[1].v[component]
        + 7.0f * (flux[2].v[component] + flux[3].v[component])
        - flux[4].v[component]) / 12.0f;
    result.v[component] = nonlinear + central;
  }
  return result;
}

SHU_HD Vec5 directional_derivative(const StateView& state, int axis, int x,
                                   int y, int z, float inverse_spacing,
                                   const float alpha3[3]) {
  const int coordinate = axis == 0 ? x : (axis == 1 ? y : z);
  const Vec5 left = numerical_flux(state, axis, x, y, z, coordinate - 1, alpha3);
  const Vec5 right = numerical_flux(state, axis, x, y, z, coordinate, alpha3);
  Vec5 result{};
#pragma unroll
  for (int component = 0; component < 5; ++component) {
    result.v[component] = (left.v[component] - right.v[component])
        * inverse_spacing;
  }
  return result;
}

SHU_HD void add_oriented(float global[5], const Vec5& oriented, int axis) {
  global[0] += oriented.v[0];
  global[4] += oriented.v[4];
  if (axis == 0) {
    global[1] += oriented.v[1]; global[2] += oriented.v[2];
    global[3] += oriented.v[3];
  } else if (axis == 1) {
    global[2] += oriented.v[1]; global[1] += oriented.v[2];
    global[3] += oriented.v[3];
  } else {
    global[3] += oriented.v[1]; global[1] += oriented.v[2];
    global[2] += oriented.v[3];
  }
}

SHU_HD void primitive_speeds(const Vec5& global, float speeds[3]) {
  const float inverse_density = 1.0f / global.v[0];
  const float u = global.v[1] * inverse_density;
  const float v = global.v[2] * inverse_density;
  const float w = global.v[3] * inverse_density;
  const float pressure = gamma_minus_one * (global.v[4]
      - 0.5f * global.v[0] * (u * u + v * v + w * w));
  const float sound = sqrtf(gamma_value * pressure * inverse_density);
  speeds[0] = u;
  speeds[1] = v;
  speeds[2] = w;
  // Callers combine the relevant velocity with sound.
  speeds[0] = u;
  speeds[1] = v;
  speeds[2] = w;
}

}  // namespace shu3d

