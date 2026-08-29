#include "runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> initialize(int intervals) {
  const int side = intervals + 1;
  const std::int64_t cells = static_cast<std::int64_t>(side) * side * side;
  std::vector<float> state(static_cast<std::size_t>(cells) * 5);
  const float spacing = 10.0f / static_cast<float>(intervals);
  const float pi = 4.0f * std::atan(1.0f);
  const float coefficient = 5.0f / (2.0f * pi * std::exp(-0.5f));
  for (int z = 0; z < side; ++z) {
    for (int y = 0; y < side; ++y) {
      const float py = y * spacing;
      for (int x = 0; x < side; ++x) {
        const float px = x * spacing;
        const float radius_squared = (px - 5.0f) * (px - 5.0f)
            + (py - 5.0f) * (py - 5.0f);
        const float exponential = std::exp(-0.5f * radius_squared);
        const float vx = -coefficient * exponential * (py - 5.0f);
        const float vy = coefficient * exponential * (px - 5.0f);
        const float temperature = 1.0f - 0.5f * coefficient * coefficient
            * exponential * exponential * 0.4f / 1.4f;
        const float pressure = std::pow(temperature, 1.4f / 0.4f);
        const float density = pressure / temperature;
        const float values[5] = {density, density * vx, density * vy, 0.0f,
            pressure / 0.4f + 0.5f * density * (vx * vx + vy * vy)};
        const std::int64_t cell = (static_cast<std::int64_t>(z) * side + y) * side + x;
        for (int component = 0; component < 5; ++component) {
          state[static_cast<std::int64_t>(component) * cells + cell] = values[component];
        }
      }
    }
  }
  return state;
}

std::vector<float> read_unique_and_duplicate(
    const std::string& path, int intervals) {
  const std::int64_t unique_cells =
      static_cast<std::int64_t>(intervals) * intervals * intervals;
  const std::size_t unique_elements =
      static_cast<std::size_t>(5 * unique_cells);
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) throw std::runtime_error("could not open input state");
  std::vector<float> unique(unique_elements);
  const std::size_t count =
      std::fread(unique.data(), sizeof(float), unique_elements, file);
  const int trailing = std::fgetc(file);
  std::fclose(file);
  if (count != unique_elements || trailing != EOF) {
    throw std::runtime_error("input state has unexpected byte length");
  }

  const int side = intervals + 1;
  const std::int64_t duplicated_cells =
      static_cast<std::int64_t>(side) * side * side;
  std::vector<float> duplicated(
      static_cast<std::size_t>(5 * duplicated_cells));
  for (int component = 0; component < 5; ++component) {
    for (int z = 0; z < side; ++z) {
      for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
          const std::int64_t source =
              (static_cast<std::int64_t>(z % intervals) * intervals
               + y % intervals) * intervals + x % intervals;
          const std::int64_t target =
              (static_cast<std::int64_t>(z) * side + y) * side + x;
          duplicated[static_cast<std::int64_t>(component) * duplicated_cells
                     + target] =
              unique[static_cast<std::int64_t>(component) * unique_cells
                     + source];
        }
      }
    }
  }
  return duplicated;
}

}  // namespace

int main(int argc, char** argv) {
  std::string target;
  int size = 0, steps = 1;
  std::string input_state, output;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--target") == 0 && i + 1 < argc) target = argv[++i];
    else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--input-state") == 0 && i + 1 < argc) input_state = argv[++i];
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output = argv[++i];
    else { std::fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]); return 2; }
  }
  if ((target != "cpu" && target != "cuda") || size < 4 || steps < 1) {
    std::fprintf(stderr, "usage: g4_cell_recompute --target {cpu|cuda} --size N --steps S [--input-state unique.f32] [--output file]\n");
    return 2;
  }
  try {
    const auto process_started = std::chrono::steady_clock::now();
    std::vector<float> initial = input_state.empty()
        ? initialize(size)
        : read_unique_and_duplicate(input_state, size);
    RunResult result = target == "cpu" ? run_cpu(initial, size, steps)
                                        : run_cuda(initial, size, steps);
    double checksum = 0.0;
    bool finite = true;
    for (float value : result.state) { checksum += value; finite = finite && std::isfinite(value); }
    if (!output.empty()) {
      std::FILE* file = std::fopen(output.c_str(), "wb");
      if (!file) throw std::runtime_error("could not open output file");
      std::fwrite(result.state.data(), sizeof(float), result.state.size(), file);
      std::fclose(file);
    }
    const double process_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - process_started).count();
    std::printf("{\"contract\":\"g4_cell_recompute_interface_cuda_event_v1\","
                "\"storage\":\"duplicated_periodic\","
                "\"lane\":\"cell-recompute-%s\",\"size\":%d,\"steps\":%d,"
                "\"execution_seconds\":%.9f,\"process_seconds_after_main\":%.9f,"
                "\"peak_bytes\":%llu,\"checksum\":%.17g,\"finite\":%s}\n",
                target.c_str(), size, steps, result.execution_seconds, process_seconds,
                static_cast<unsigned long long>(result.peak_bytes), checksum,
                finite ? "true" : "false");
    return finite ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "shu ceiling failed: %s\n", error.what());
    return 2;
  }
}
