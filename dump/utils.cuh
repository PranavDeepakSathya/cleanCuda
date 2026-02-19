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