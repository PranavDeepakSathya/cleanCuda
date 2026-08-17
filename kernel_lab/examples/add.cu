#include <cleancuda/lab/all.cuh>

#include <iostream>

using cleancuda::lab::CheckOptions;
using cleancuda::lab::Tensor;

__global__ void add_kernel(const float* left, const float* right, float* output, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = left[index] + right[index];
}

int main(int argc, char** argv) {
  const std::int64_t count = argc > 1 ? std::stoll(argv[1]) : 1 << 20;
  auto left = Tensor<float>::randn({count}, 0.0, 1.0, 1);
  auto right = Tensor<float>::randn({count}, 0.0, 1.0, 2);
  auto output = Tensor<float>::empty({count});

  const int threads = 256;
  const int blocks = static_cast<int>((output.numel() + threads - 1) / threads);
  add_kernel<<<blocks, threads>>>(left.device_data(), right.device_data(), output.mutable_device_data(), output.numel());
  cleancuda::lab::check_last_kernel("add_kernel");

  const auto expected = cleancuda::lab::ops::add(left, right);
  const auto result = cleancuda::lab::check_close(
      output, expected, CheckOptions{.atol = 1e-6, .rtol = 1e-6, .name = "add"});

  const float milliseconds = cleancuda::lab::benchmark_ms([&] {
    add_kernel<<<blocks, threads>>>(left.device_data(), right.device_data(), output.mutable_device_data(), output.numel());
  });
  std::cout << "time   : " << milliseconds << " ms\n";
  return result ? 0 : 1;
}
