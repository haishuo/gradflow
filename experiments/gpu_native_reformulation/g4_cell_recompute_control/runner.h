#pragma once

#include <cstdint>
#include <vector>

struct RunResult {
  std::vector<float> state;
  double execution_seconds = 0.0;
  std::uint64_t peak_bytes = 0;
};

RunResult run_cpu(const std::vector<float>& initial, int intervals, int steps);
RunResult run_cuda(const std::vector<float>& initial, int intervals, int steps);

