# cleanCuda kernel lab

A small, standalone loop for writing a CUDA kernel, checking it against a C++ reference, and timing it without building a Torch extension.

## Run an experiment

```bash
./kernel_lab/run kernel_lab/examples/add.cu --gpu 0 --arch sm_120
./kernel_lab/run kernel_lab/examples/matmul.cu --gpu 1 --arch sm_120 -- 512
```

The first positional argument is any standalone `.cu` file. Arguments after `--` are passed to its executable. The runner:

1. compiles with `nvcc`;
2. includes `kernel_lab/include`, the cleanCuda root, and the source's directory;
3. caches the executable by source contents and compiler options;
4. selects the process GPU through `CUDA_VISIBLE_DEVICES`;
5. runs the resulting binary.

Use `--arch auto` when `nvidia-smi` is available, or give `90`, `sm_90`, `120`, and so on explicitly. See every switch with:

```bash
./kernel_lab/run --help
```

Useful options include `-I path`, `-DNAME=value`, `--nvcc-flag=...`, `--debug`, `--force`, `--compile-only`, `--dry-run`, and `--verbose`.

## Run on Modal

Install and authenticate Modal once:

```bash
pip install modal
modal setup
```

Then select a remote GPU and let the remote machine detect its architecture:

```bash
./kernel_lab/modal --source kernel_lab/examples/add.cu --gpu L40S
./kernel_lab/modal --source kernel_lab/examples/matmul.cu --gpu RTX-PRO-6000 --arch sm_120 --args "512"
./kernel_lab/modal --source my_kernel.cu --gpu H100 --arch sm_90 \
  --define "TILE_M=128 TILE_N=128" --nvcc-flags "--use_fast_math"
```

The Modal runner uploads the selected `.cu` file and the `.cu`, `.cuh`, `.h`, and `.hpp` files beside it, mounts cleanCuda for shared headers, compiles in an NVIDIA CUDA 12.8 development image, runs on the requested GPU, and returns the native program output. Compiled binaries are cached in a Modal Volume by source and compiler flags.

Modal GPU names compatible with the default CUDA 12.8 image include `T4`, `L4`, `A10`, `L40S`, `A100`, `A100-40GB`, `A100-80GB`, `RTX-PRO-6000`, `H100`, `H200`, and `B200`. B300 requires a CUDA 13.1+ image and is intentionally not included in this runner's default image yet.

## Write a kernel

```cpp
#include <cleancuda/lab/all.cuh>

using cleancuda::lab::CheckOptions;
using cleancuda::lab::Tensor;

__global__ void square(const float* input, float* output, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = input[i] * input[i];
}

int main() {
  auto input = Tensor<float>::randn({1 << 20}, 0.0, 1.0, 7);
  auto output = Tensor<float>::empty(input.shape());

  const int threads = 256;
  const int blocks = (input.numel() + threads - 1) / threads;
  square<<<blocks, threads>>>(input.device_data(), output.mutable_device_data(), input.numel());
  cleancuda::lab::check_last_kernel("square");

  const auto expected = cleancuda::lab::ops::unary(input, [](float x) { return x * x; });
  const auto result = cleancuda::lab::check_close(
      output, expected, CheckOptions{.atol = 1e-6, .rtol = 1e-6, .name = "square"});
  return result ? 0 : 1;
}
```

## Tensor API

`Tensor<T>` owns row-major host storage and lazily allocated CUDA storage. Transfers happen only when the opposite side is requested.

Factories:

```cpp
Tensor<float>::empty({m, n});
Tensor<float>::zeros({m, n});
Tensor<float>::ones({m, n});
Tensor<float>::full({m, n}, 3.0f);
Tensor<float>::arange(1024);
Tensor<float>::rand({m, n}, -1.0f, 1.0f, seed);
Tensor<float>::randn({m, n}, mean, stddev, seed);
```

Access and movement:

```cpp
t.shape();
t.size(0);
t.ndim();
t.numel();
t.bytes();
t.reshape({m, n});
t.host_data();
t.mutable_host_data();
t.device_data();
t.mutable_device_data();
t.to_host();
t.to_device();
t.at({row, column});
std::cout << t << '\n';
```

Calling `mutable_device_data()` marks the GPU copy as newest, so a later reference check automatically copies the result back. If a kernel writes through a pointer retained from an earlier call, call `tensor.mark_device_modified()` after the launch.

## Reference operations and checking

Built-in CPU references are `ops::add`, `sub`, `mul`, `relu`, `exp`, `matmul`, and `sum`. New elementwise operations do not need library changes:

```cpp
auto expected = cleancuda::lab::ops::unary(x, [](float value) {
  return custom_reference(value);
});

auto expected = cleancuda::lab::ops::binary(a, b, [](float left, float right) {
  return custom_reference(left, right);
});
```

`check_close` checks `abs(actual - expected) <= atol + rtol * abs(expected)` and prints the shape, mismatch count, maximum absolute and relative errors, and the worst element. `benchmark_ms` uses CUDA events and returns the average kernel time.

## Layout

```text
kernel_lab/
├── include/cleancuda/lab/  # header-only tensor, references, checks
├── examples/               # complete experiments
├── run                     # short launcher entry point
├── run.py                  # local compiler/GPU/architecture driver
├── modal                   # short Modal entry point
└── modal_run.py            # selectable remote Modal GPU runner
```
