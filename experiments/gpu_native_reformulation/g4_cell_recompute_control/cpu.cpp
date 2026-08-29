#include "runner.h"
#include "shu_math.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using shu3d::StateView;
using shu3d::Vec5;

std::int64_t raw_cell(int side, int x, int y, int z) {
  return (static_cast<std::int64_t>(z) * side + y) * side + x;
}

void compute_alphas(const std::vector<float>& state, int intervals,
                    std::vector<float> alpha[3]) {
  const int side = intervals + 1;
  const std::int64_t cells = static_cast<std::int64_t>(side) * side * side;
  const StateView view{state.data(), intervals, side, cells};
  const std::int64_t lines = static_cast<std::int64_t>(side) * side;
  for (int axis = 0; axis < 3; ++axis) alpha[axis].assign(lines * 3, 1.0e-15f);

  for (int axis = 0; axis < 3; ++axis) {
#pragma omp parallel for schedule(static)
    for (std::int64_t line = 0; line < lines; ++line) {
      float maximum[3] = {1.0e-15f, 1.0e-15f, 1.0e-15f};
      int x = 1, y = 1, z = 1;
      if (axis == 0) { y = static_cast<int>(line % side); z = static_cast<int>(line / side); }
      if (axis == 1) { x = static_cast<int>(line % side); z = static_cast<int>(line / side); }
      if (axis == 2) { x = static_cast<int>(line % side); y = static_cast<int>(line / side); }
      for (int coordinate = 0; coordinate <= intervals; ++coordinate) {
        if (axis == 0) x = coordinate;
        if (axis == 1) y = coordinate;
        if (axis == 2) z = coordinate;
        const Vec5 q = shu3d::load_oriented(view, axis, x, y, z, 0);
        Vec5 flux{};
        float velocity, sound, enthalpy, tangent[2];
        shu3d::primitive_and_flux(q, flux, velocity, sound, enthalpy, tangent);
        maximum[0] = std::max(maximum[0], std::fabs(velocity - sound));
        maximum[1] = std::max(maximum[1], std::fabs(velocity));
        maximum[2] = std::max(maximum[2], std::fabs(velocity + sound));
      }
      for (int family = 0; family < 3; ++family) {
        alpha[axis][line * 3 + family] = shu3d::lf_enlargement * maximum[family];
      }
    }
  }
}

float timestep(const std::vector<float>& state, int intervals) {
  const int side = intervals + 1;
  const std::int64_t cells = static_cast<std::int64_t>(side) * side * side;
  const StateView view{state.data(), intervals, side, cells};
  const float inverse_spacing = static_cast<float>(intervals) / 10.0f;
  float maximum = 0.0f;
#pragma omp parallel for collapse(3) reduction(max:maximum) schedule(static)
  for (int z = 1; z <= intervals; ++z) {
    for (int y = 1; y <= intervals; ++y) {
      for (int x = 1; x <= intervals; ++x) {
        const Vec5 q = shu3d::load_global(view, x, y, z);
        const float inverse_density = 1.0f / q.v[0];
        const float u = q.v[1] * inverse_density;
        const float v = q.v[2] * inverse_density;
        const float w = q.v[3] * inverse_density;
        const float pressure = shu3d::gamma_minus_one * (q.v[4]
            - 0.5f * q.v[0] * (u * u + v * v + w * w));
        const float sound = std::sqrt(shu3d::gamma_value * pressure * inverse_density);
        const float local = (std::fabs(u) + std::fabs(v) + std::fabs(w)
            + 3.0f * sound) * inverse_spacing;
        maximum = std::max(maximum, local);
      }
    }
  }
  return 0.1f / maximum;
}

void rhs(const std::vector<float>& state, std::vector<float>& result,
         int intervals, const std::vector<float> alpha[3]) {
  const int side = intervals + 1;
  const std::int64_t cells = static_cast<std::int64_t>(side) * side * side;
  const std::int64_t lines = static_cast<std::int64_t>(side) * side;
  const StateView view{state.data(), intervals, side, cells};
  const float inverse_spacing = static_cast<float>(intervals) / 10.0f;
#pragma omp parallel for collapse(3) schedule(static)
  for (int z = 0; z <= intervals; ++z) {
    for (int y = 0; y <= intervals; ++y) {
      for (int x = 0; x <= intervals; ++x) {
        float total[5] = {};
        const std::int64_t line_x = static_cast<std::int64_t>(z) * side + y;
        const std::int64_t line_y = static_cast<std::int64_t>(z) * side + x;
        const std::int64_t line_z = static_cast<std::int64_t>(y) * side + x;
        const std::int64_t line_ids[3] = {line_x, line_y, line_z};
        for (int axis = 0; axis < 3; ++axis) {
          float a[3];
          for (int k = 0; k < 3; ++k) a[k] = alpha[axis][line_ids[axis] * 3 + k];
          const Vec5 derivative = shu3d::directional_derivative(
              view, axis, x, y, z, inverse_spacing, a);
          shu3d::add_oriented(total, derivative, axis);
        }
        const std::int64_t cell = raw_cell(side, x, y, z);
        for (int component = 0; component < 5; ++component) {
          result[static_cast<std::int64_t>(component) * cells + cell] = total[component];
        }
      }
    }
  }
}

void update(const std::vector<float>& base, const std::vector<float>& stage,
            const std::vector<float>& deriv, std::vector<float>& output,
            int intervals, float dt, int which) {
  const int side = intervals + 1;
  const std::int64_t cells = static_cast<std::int64_t>(side) * side * side;
  const StateView base_view{base.data(), intervals, side, cells};
  const StateView stage_view{stage.data(), intervals, side, cells};
  const StateView deriv_view{deriv.data(), intervals, side, cells};
#pragma omp parallel for collapse(3) schedule(static)
  for (int z = 0; z <= intervals; ++z) {
    for (int y = 0; y <= intervals; ++y) {
      for (int x = 0; x <= intervals; ++x) {
        const std::int64_t cell = raw_cell(side, x, y, z);
        const Vec5 q0 = shu3d::load_global(base_view, x, y, z);
        const Vec5 qs = shu3d::load_global(stage_view, x, y, z);
        const Vec5 r = shu3d::load_global(deriv_view, x, y, z);
        for (int component = 0; component < 5; ++component) {
          float value;
          if (which == 1) value = q0.v[component] + dt * r.v[component];
          else if (which == 2) value = 0.75f * q0.v[component]
              + 0.25f * (qs.v[component] + dt * r.v[component]);
          else value = (q0.v[component]
              + 2.0f * (qs.v[component] + dt * r.v[component])) / 3.0f;
          output[static_cast<std::int64_t>(component) * cells + cell] = value;
        }
      }
    }
  }
}

}  // namespace

RunResult run_cpu(const std::vector<float>& initial, int intervals, int steps) {
  const std::int64_t entries = static_cast<std::int64_t>(initial.size());
  std::vector<float> q = initial, q1(entries), q2(entries), r(entries);
  std::vector<float> alpha[3];
  const auto started = std::chrono::steady_clock::now();
  for (int step = 0; step < steps; ++step) {
    const float dt = timestep(q, intervals);
    compute_alphas(q, intervals, alpha); rhs(q, r, intervals, alpha);
    update(q, q, r, q1, intervals, dt, 1);
    compute_alphas(q1, intervals, alpha); rhs(q1, r, intervals, alpha);
    update(q, q1, r, q2, intervals, dt, 2);
    compute_alphas(q2, intervals, alpha); rhs(q2, r, intervals, alpha);
    update(q, q2, r, q1, intervals, dt, 3);
    q.swap(q1);
  }
  const auto stopped = std::chrono::steady_clock::now();
  RunResult result;
  result.state = std::move(q);
  result.execution_seconds = std::chrono::duration<double>(stopped - started).count();
  result.peak_bytes = static_cast<std::uint64_t>(entries) * sizeof(float) * 4;
  return result;
}

