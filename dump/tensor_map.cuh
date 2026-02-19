#pragma once
#include "utils.cuh"

class TmaLoader {
public:
  static PFN_cuTensorMapEncodeTiled_v12000 get_encoder() {
    static PFN_cuTensorMapEncodeTiled_v12000 func_ptr = load_symbol();
    return func_ptr;
  }

private:
  static PFN_cuTensorMapEncodeTiled_v12000 load_symbol() {
    cudaDriverEntryPointQueryResult driver_status;
    void* ptr = nullptr;

    cudaError_t err = cudaGetDriverEntryPointByVersion(
      "cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &driver_status
    );

    if (err != cudaSuccess || driver_status != cudaDriverEntryPointSuccess) {
      fprintf(stderr, "[TMA Error] Failed to load cuTensorMapEncodeTiled. Requires CUDA Driver 12+.\n");
      exit(EXIT_FAILURE);
    }

    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }
};

template <typename T>
class TmaDescriptor {
public:
  static CUtensorMap create_2d_row_major(
    T* global_address,
    std::pair<uint64_t, uint64_t> logical_shape,
    std::pair<uint32_t, uint32_t> box_shape,
    CUtensorMapSwizzle swizzle_mode = CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion l2_promo = CU_TENSOR_MAP_L2_PROMOTION_NONE
  ) {
    auto [rows, cols] = logical_shape;
    auto [box_r, box_c] = box_shape;

    uint64_t g_dims[2] = {cols, rows};
    uint64_t g_strides[1] = {cols * sizeof(T)};
    uint32_t b_dims[2] = {box_c, box_r};
    uint32_t e_strides[2] = {1, 1};

    return create_raw(
      global_address,
      2,
      g_dims,
      g_strides,
      b_dims,
      e_strides,
      swizzle_mode,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      l2_promo
    );
  }

  static CUtensorMap create_2d_col_major(
    T* global_address,
    std::pair<uint64_t, uint64_t> logical_shape,
    std::pair<uint32_t, uint32_t> box_shape,
    CUtensorMapSwizzle swizzle_mode = CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion l2_promo = CU_TENSOR_MAP_L2_PROMOTION_NONE
  ) {
    auto [rows, cols] = logical_shape;
    auto [box_r, box_c] = box_shape;

    uint64_t g_dims[2] = {rows, cols};
    uint64_t g_strides[1] = {rows * sizeof(T)};
    uint32_t b_dims[2] = {box_r, box_c};
    uint32_t e_strides[2] = {1, 1};

    return create_raw(
      global_address,
      2,
      g_dims,
      g_strides,
      b_dims,
      e_strides,
      swizzle_mode,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      l2_promo
    );
  }

  static CUtensorMap create_raw(
    T* global_address,
    uint32_t rank,
    uint64_t* global_dims,
    uint64_t* global_strides,
    uint32_t* box_dims,
    uint32_t* element_strides,
    CUtensorMapSwizzle swizzle,
    CUtensorMapInterleave interleave,
    CUtensorMapL2promotion l2_promo
  ) {
    CUtensorMap tma_map{};
    auto encoder = TmaLoader::get_encoder();

    CUresult res = encoder(
      &tma_map,
      get_data_type(),
      rank,
      global_address,
      global_dims,
      global_strides,
      box_dims,
      element_strides,
      interleave,
      swizzle,
      l2_promo,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
      const char* err_str;
      cuGetErrorName(res, &err_str);
      fprintf(stderr, "[TMA Build Error] %s\n", err_str);
      exit(1);
    }

    return tma_map;
  }

private:
  static CUtensorMapDataType get_data_type() {
    if constexpr (std::is_same_v<T, float>) return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    if constexpr (std::is_same_v<T, __half>) return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    if constexpr (std::is_same_v<T, nv_bfloat16>) return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    if constexpr (std::is_same_v<T, int32_t>) return CU_TENSOR_MAP_DATA_TYPE_INT32;
    if constexpr (std::is_same_v<T, uint8_t>) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }
};