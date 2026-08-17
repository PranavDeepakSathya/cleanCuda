#pragma once

#include "tensor.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace cleancuda::lab {

struct CheckOptions {
  double atol = 1e-5;
  double rtol = 1e-5;
  bool equal_nan = false;
  bool print = true;
  std::string name = "output";
};

struct CheckResult {
  bool passed = false;
  std::size_t mismatches = 0;
  std::size_t worst_index = 0;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  double actual_at_worst = 0.0;
  double expected_at_worst = 0.0;

  explicit operator bool() const noexcept { return passed; }
};

template <typename Actual, typename Expected>
CheckResult check_close(const Tensor<Actual>& actual, const Tensor<Expected>& expected, CheckOptions options = {}) {
  if (actual.shape() != expected.shape()) {
    throw std::invalid_argument("check_close shape mismatch: " + format_shape(actual.shape()) + " vs " +
                                format_shape(expected.shape()));
  }

  CheckResult result;
  const Actual* actual_values = actual.host_data();
  const Expected* expected_values = expected.host_data();
  for (std::size_t i = 0; i < actual.numel(); ++i) {
    const double observed = static_cast<double>(actual_values[i]);
    const double wanted = static_cast<double>(expected_values[i]);
    const bool both_nan = std::isnan(observed) && std::isnan(wanted);
    const double absolute = std::abs(observed - wanted);
    const double relative = absolute / std::max(std::abs(wanted), std::numeric_limits<double>::min());
    const bool close = observed == wanted || (both_nan && options.equal_nan) ||
                       absolute <= options.atol + options.rtol * std::abs(wanted);
    if (!close) ++result.mismatches;
    if (absolute >= result.max_absolute_error) {
      result.max_absolute_error = absolute;
      result.max_relative_error = relative;
      result.worst_index = i;
      result.actual_at_worst = observed;
      result.expected_at_worst = wanted;
    }
  }
  result.passed = result.mismatches == 0;

  if (options.print) {
    std::cout << '[' << (result.passed ? "PASS" : "FAIL") << "] " << options.name
              << "  shape=" << format_shape(actual.shape()) << "  mismatches=" << result.mismatches << '/'
              << actual.numel() << "  max_abs=" << result.max_absolute_error
              << "  max_rel=" << result.max_relative_error;
    if (!result.passed) {
      std::cout << "  worst[" << result.worst_index << "]=" << result.actual_at_worst
                << " expected=" << result.expected_at_worst;
    }
    std::cout << '\n';
  }
  return result;
}

template <typename Function>
float benchmark_ms(Function&& function, int warmup = 10, int iterations = 100) {
  for (int i = 0; i < warmup; ++i) function();
  CCL_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start;
  cudaEvent_t stop;
  CCL_CUDA_CHECK(cudaEventCreate(&start));
  CCL_CUDA_CHECK(cudaEventCreate(&stop));
  CCL_CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) function();
  CCL_CUDA_CHECK(cudaEventRecord(stop));
  CCL_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0.0f;
  CCL_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  CCL_CUDA_CHECK(cudaEventDestroy(start));
  CCL_CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed / static_cast<float>(iterations);
}

inline void check_last_kernel(const char* context = "kernel launch") {
  cuda_check(cudaGetLastError(), context, __FILE__, __LINE__);
}

}  // namespace cleancuda::lab
