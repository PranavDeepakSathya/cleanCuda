#pragma once

#include "tensor.cuh"

#include <functional>
#include <stdexcept>

namespace cleancuda::lab::ops {

template <typename T, typename Function>
Tensor<T> unary(const Tensor<T>& input, Function function) {
  Tensor<T> output(input.shape());
  const T* source = input.host_data();
  T* destination = output.mutable_host_data();
  for (std::size_t i = 0; i < input.numel(); ++i) destination[i] = static_cast<T>(function(source[i]));
  return output;
}

template <typename T, typename Function>
Tensor<T> binary(const Tensor<T>& left, const Tensor<T>& right, Function function) {
  if (left.shape() != right.shape()) throw std::invalid_argument("binary op requires equal shapes");
  Tensor<T> output(left.shape());
  const T* lhs = left.host_data();
  const T* rhs = right.host_data();
  T* destination = output.mutable_host_data();
  for (std::size_t i = 0; i < left.numel(); ++i) destination[i] = static_cast<T>(function(lhs[i], rhs[i]));
  return output;
}

template <typename T>
Tensor<T> add(const Tensor<T>& left, const Tensor<T>& right) {
  return binary(left, right, std::plus<T>{});
}

template <typename T>
Tensor<T> sub(const Tensor<T>& left, const Tensor<T>& right) {
  return binary(left, right, std::minus<T>{});
}

template <typename T>
Tensor<T> mul(const Tensor<T>& left, const Tensor<T>& right) {
  return binary(left, right, std::multiplies<T>{});
}

template <typename T>
Tensor<T> relu(const Tensor<T>& input) {
  return unary(input, [](T value) { return std::max(value, T{}); });
}

template <typename T>
Tensor<T> exp(const Tensor<T>& input) {
  return unary(input, [](T value) { return std::exp(value); });
}

template <typename T>
Tensor<T> matmul(const Tensor<T>& left, const Tensor<T>& right) {
  if (left.ndim() != 2 || right.ndim() != 2) throw std::invalid_argument("matmul expects two 2D tensors");
  const std::int64_t m = left.size(0);
  const std::int64_t k = left.size(1);
  const std::int64_t n = right.size(1);
  if (right.size(0) != k) throw std::invalid_argument("matmul inner dimensions do not match");

  Tensor<T> output = Tensor<T>::zeros({m, n});
  const T* a = left.host_data();
  const T* b = right.host_data();
  T* c = output.mutable_host_data();
  for (std::int64_t row = 0; row < m; ++row) {
    for (std::int64_t column = 0; column < n; ++column) {
      T accumulator{};
      for (std::int64_t inner = 0; inner < k; ++inner) {
        accumulator += a[row * k + inner] * b[inner * n + column];
      }
      c[row * n + column] = accumulator;
    }
  }
  return output;
}

template <typename T>
T sum(const Tensor<T>& input) {
  return std::accumulate(input.host_data(), input.host_data() + input.numel(), T{});
}

}  // namespace cleancuda::lab::ops
