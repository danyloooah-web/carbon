
NAME = "v4_swizzle_128x4.cuh"
SOURCE = r"""#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <cstdint>

namespace sglang::bundle_v9 {

inline constexpr uint32_t kSwizzleThreads = 256;

template <bool kUsePDL>
__global__ void __launch_bounds__(kSwizzleThreads) swizzle_col_to_128x4_kernel(
    const int32_t* __restrict__ src,
    int4* __restrict__ dst_vec,
    uint32_t rows,
    uint32_t cols,
    int64_t col_stride,
    uint32_t total_quads) {
  using namespace device;
  PDLWaitPrimary<kUsePDL>();
  const uint32_t q = blockIdx.x * kSwizzleThreads + threadIdx.x;
  if (q < total_quads) {
    const uint32_t lane = q & 31u;
    const uint32_t tc = q >> 5;
    const uint32_t col = tc % cols;
    const uint32_t tile = tc / cols;
    const uint32_t r0 = tile * 128u + lane;
    const int32_t* col_ptr = src + static_cast<int64_t>(col) * col_stride;
    const int32_t w0 = (r0 + 0u < rows) ? col_ptr[r0 + 0u] : 0;
    const int32_t w1 = (r0 + 32u < rows) ? col_ptr[r0 + 32u] : 0;
    const int32_t w2 = (r0 + 64u < rows) ? col_ptr[r0 + 64u] : 0;
    const int32_t w3 = (r0 + 96u < rows) ? col_ptr[r0 + 96u] : 0;
    dst_vec[q] = make_int4(w0, w1, w2, w3);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct SwizzleColTo128x4 {
  static void run(
      const tvm::ffi::TensorView src,
      const tvm::ffi::TensorView dst_u8,
      const int64_t rows,
      const int64_t cols,
      const int64_t col_stride) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    auto R = SymbolicSize{"rows"};
    auto C = SymbolicSize{"cols"};
    auto S = SymbolicSize{"col_stride"};
    auto B = SymbolicSize{"out_bytes"};
    TensorMatcher({R, C}).with_strides({1, S}).with_dtype<int32_t>().with_device(dev).verify(src);
    TensorMatcher({B}).with_strides({1}).with_dtype<uint8_t>().with_device(dev).verify(dst_u8);
    const uint32_t r = static_cast<uint32_t>(rows);
    const uint32_t c = static_cast<uint32_t>(cols);
    const uint32_t tiles = (r + 127u) / 128u;
    const uint32_t quads = tiles * c * 32u;
    RuntimeCheck(R.unwrap() == rows && C.unwrap() == cols && S.unwrap() == col_stride, "swizzle: shape mismatch");
    RuntimeCheck(B.unwrap() == static_cast<int64_t>(quads) * 16, "swizzle: dst size mismatch");
    RuntimeCheck(reinterpret_cast<uintptr_t>(dst_u8.data_ptr()) % 16u == 0, "swizzle: dst 16B alignment");
    if (quads == 0u) return;
    const uint32_t blocks = (quads + kSwizzleThreads - 1u) / kSwizzleThreads;
    LaunchKernel(blocks, kSwizzleThreads, dev.unwrap()).enable_pdl(kUsePDL)(
        swizzle_col_to_128x4_kernel<kUsePDL>,
        static_cast<const int32_t*>(src.data_ptr()),
        static_cast<int4*>(dst_u8.data_ptr()),
        r, c, col_stride, quads);
  }
};

}
"""
