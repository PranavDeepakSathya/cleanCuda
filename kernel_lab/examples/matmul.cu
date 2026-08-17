#include <cleancuda/lab/all.cuh>

#include <iostream>

using cleancuda::lab::CheckOptions;
using cleancuda::lab::Tensor;

__global__ void matmul_kernel(const float* left, const float* right, float* output, int m, int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || column >= n) return;

  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    accumulator += left[row * k + inner] * right[inner * n + column];
  }
  output[row * n + column] = accumulator;
}

int main(int argc, char** argv) {
  const int size = argc > 1 ? std::stoi(argv[1]) : 256;
  auto left = Tensor<float>::randn({size, size}, 0.0, 0.2, 1);
  auto right = Tensor<float>::randn({size, size}, 0.0, 0.2, 2);
  auto output = Tensor<float>::empty({size, size});

  const dim3 threads(16, 16);
  const dim3 blocks((size + threads.x - 1) / threads.x, (size + threads.y - 1) / threads.y);
  matmul_kernel<<<blocks, threads>>>(
      left.device_data(), right.device_data(), output.mutable_device_data(), size, size, size);
  cleancuda::lab::check_last_kernel("matmul_kernel");

  const auto expected = cleancuda::lab::ops::matmul(left, right);
  const auto result = cleancuda::lab::check_close(
      output, expected, CheckOptions{.atol = 1e-3, .rtol = 1e-3, .name = "matmul"});

  const float milliseconds = cleancuda::lab::benchmark_ms([&] {
    matmul_kernel<<<blocks, threads>>>(
        left.device_data(), right.device_data(), output.mutable_device_data(), size, size, size);
  });
  const double tflops = 2.0 * size * size * size / (milliseconds * 1e-3) / 1e12;
  std::cout << "time   : " << milliseconds << " ms\n"
            << "speed  : " << tflops << " TFLOP/s\n";
  return result ? 0 : 1;
}
