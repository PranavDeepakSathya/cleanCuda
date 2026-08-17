#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <numeric>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cleancuda::lab {

inline void cuda_check(cudaError_t error, const char* expression, const char* file, int line) {
  if (error == cudaSuccess) return;
  std::ostringstream message;
  message << "CUDA error: " << cudaGetErrorString(error) << "\n  call: " << expression
          << "\n  at: " << file << ':' << line;
  throw std::runtime_error(message.str());
}

#define CCL_CUDA_CHECK(call) ::cleancuda::lab::cuda_check((call), #call, __FILE__, __LINE__)

using Shape = std::vector<std::int64_t>;

inline std::size_t shape_numel(const Shape& shape) {
  if (shape.empty()) return 1;
  std::size_t result = 1;
  for (std::int64_t extent : shape) {
    if (extent < 0) throw std::invalid_argument("tensor extents must be non-negative");
    const auto value = static_cast<std::size_t>(extent);
    if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error("tensor shape is too large");
    }
    result *= value;
  }
  return result;
}

inline std::string format_shape(const Shape& shape) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) out << ", ";
    out << shape[i];
  }
  out << ']';
  return out.str();
}

template <typename T>
class Tensor {
 public:
  using value_type = T;

  explicit Tensor(Shape shape)
      : shape_(std::move(shape)), host_(shape_numel(shape_)) {}

  Tensor(std::initializer_list<std::int64_t> shape) : Tensor(Shape(shape)) {}

  ~Tensor() { release_device(); }

  Tensor(const Tensor& other) : shape_(other.shape_), host_(other.numel()) {
    std::copy_n(other.host_data(), other.numel(), host_.data());
  }

  Tensor& operator=(const Tensor& other) {
    if (this == &other) return *this;
    release_device();
    shape_ = other.shape_;
    host_.resize(other.numel());
    std::copy_n(other.host_data(), other.numel(), host_.data());
    state_ = State::host;
    return *this;
  }

  Tensor(Tensor&& other) noexcept
      : shape_(std::move(other.shape_)), host_(std::move(other.host_)), device_(other.device_), state_(other.state_) {
    other.device_ = nullptr;
    other.state_ = State::host;
  }

  Tensor& operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    release_device();
    shape_ = std::move(other.shape_);
    host_ = std::move(other.host_);
    device_ = other.device_;
    state_ = other.state_;
    other.device_ = nullptr;
    other.state_ = State::host;
    return *this;
  }

  static Tensor empty(Shape shape) { return Tensor(std::move(shape)); }

  static Tensor zeros(Shape shape) {
    Tensor result(std::move(shape));
    std::fill(result.host_.begin(), result.host_.end(), T{});
    return result;
  }

  static Tensor ones(Shape shape) { return full(std::move(shape), T{1}); }

  static Tensor full(Shape shape, T value) {
    Tensor result(std::move(shape));
    std::fill(result.host_.begin(), result.host_.end(), value);
    return result;
  }

  static Tensor arange(std::int64_t count, T start = T{}, T step = T{1}) {
    Tensor result({count});
    for (std::size_t i = 0; i < result.numel(); ++i) {
      result.host_[i] = static_cast<T>(start + static_cast<T>(i) * step);
    }
    return result;
  }

  static Tensor rand(Shape shape, T low = T{}, T high = T{1}, std::uint64_t seed = 0) {
    static_assert(std::is_arithmetic_v<T>, "rand currently supports arithmetic tensor types");
    Tensor result(std::move(shape));
    std::mt19937_64 generator(seed);
    if constexpr (std::is_integral_v<T>) {
      std::uniform_int_distribution<T> distribution(low, high);
      for (T& value : result.host_) value = distribution(generator);
    } else {
      std::uniform_real_distribution<double> distribution(static_cast<double>(low), static_cast<double>(high));
      for (T& value : result.host_) value = static_cast<T>(distribution(generator));
    }
    return result;
  }

  static Tensor randn(Shape shape, double mean = 0.0, double stddev = 1.0, std::uint64_t seed = 0) {
    static_assert(std::is_floating_point_v<T>, "randn requires a floating-point tensor type");
    Tensor result(std::move(shape));
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> distribution(mean, stddev);
    for (T& value : result.host_) value = static_cast<T>(distribution(generator));
    return result;
  }

  const Shape& shape() const noexcept { return shape_; }
  std::int64_t size(std::size_t dimension) const { return shape_.at(dimension); }
  std::size_t ndim() const noexcept { return shape_.size(); }
  std::size_t numel() const noexcept { return host_.size(); }
  std::size_t bytes() const noexcept { return numel() * sizeof(T); }
  bool has_device_storage() const noexcept { return device_ != nullptr; }

  Tensor& reshape(Shape shape) {
    if (shape_numel(shape) != numel()) throw std::invalid_argument("reshape changes tensor element count");
    shape_ = std::move(shape);
    return *this;
  }

  const T* host_data() const {
    sync_host();
    return host_.data();
  }

  T* mutable_host_data() {
    sync_host();
    state_ = State::host;
    return host_.data();
  }

  const T* device_data() const {
    sync_device();
    return device_;
  }

  T* mutable_device_data() {
    sync_device();
    state_ = State::device;
    return device_;
  }

  Tensor& to_device() {
    sync_device();
    return *this;
  }

  Tensor& to_host() {
    sync_host();
    return *this;
  }

  void mark_device_modified() {
    if (!device_) throw std::logic_error("tensor has no device storage");
    state_ = State::device;
  }

  T& operator[](std::size_t index) { return mutable_host_data()[index]; }
  const T& operator[](std::size_t index) const { return host_data()[index]; }

  T& at(std::initializer_list<std::int64_t> indices) { return mutable_host_data()[offset(indices)]; }
  const T& at(std::initializer_list<std::int64_t> indices) const { return host_data()[offset(indices)]; }

  std::string summary(std::size_t limit = 8) const {
    std::ostringstream out;
    out << "Tensor(shape=" << format_shape(shape_) << ", values=[";
    const T* values = host_data();
    const std::size_t shown = std::min(limit, numel());
    for (std::size_t i = 0; i < shown; ++i) {
      if (i) out << ", ";
      out << values[i];
    }
    if (shown < numel()) out << ", ...";
    out << "])";
    return out.str();
  }

 private:
  enum class State { host, device, synced };

  void allocate_device() const {
    if (device_ || bytes() == 0) return;
    CCL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_), bytes()));
  }

  void sync_device() const {
    allocate_device();
    if (bytes() != 0 && state_ == State::host) {
      CCL_CUDA_CHECK(cudaMemcpy(device_, host_.data(), bytes(), cudaMemcpyHostToDevice));
      state_ = State::synced;
    }
  }

  void sync_host() const {
    if (bytes() != 0 && state_ == State::device) {
      CCL_CUDA_CHECK(cudaMemcpy(host_.data(), device_, bytes(), cudaMemcpyDeviceToHost));
      state_ = State::synced;
    }
  }

  void release_device() noexcept {
    if (device_) cudaFree(device_);
    device_ = nullptr;
  }

  std::size_t offset(std::initializer_list<std::int64_t> indices) const {
    if (indices.size() != shape_.size()) throw std::out_of_range("incorrect number of tensor indices");
    std::size_t result = 0;
    std::size_t stride = numel();
    std::size_t dimension = 0;
    for (std::int64_t index : indices) {
      const auto extent = static_cast<std::size_t>(shape_[dimension]);
      if (index < 0 || static_cast<std::size_t>(index) >= extent) throw std::out_of_range("tensor index out of range");
      stride = extent == 0 ? 0 : stride / extent;
      result += static_cast<std::size_t>(index) * stride;
      ++dimension;
    }
    return result;
  }

  Shape shape_;
  mutable std::vector<T> host_;
  mutable T* device_ = nullptr;
  mutable State state_ = State::host;
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& tensor) {
  return out << tensor.summary();
}

}  // namespace cleancuda::lab
