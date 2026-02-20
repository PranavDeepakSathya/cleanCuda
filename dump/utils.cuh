#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <type_traits> // for if constexpr
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cuda/ptx>
#include <numeric>
#include <iomanip>
#include <cudaTypedefs.h>
#include <cuda/barrier>

// 1. The Essential Macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__device__ inline
int elect_sync() {
  int pred = 0;
  asm volatile(
    "{\n"
    ".reg .pred P;\n"
    "elect.sync _|P, %1;\n"
    "@P mov.s32 %0, 1;\n"
    "}"
    : "+r"(pred) : "r"(0xFFFF'FFFF)
  );
  return pred;
}

template <typename T>
__device__ inline
T warp_uniform(T x) {
  return __shfl_sync(0xFFFF'FFFF, x, 0);
}

