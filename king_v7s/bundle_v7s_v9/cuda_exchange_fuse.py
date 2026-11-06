
NAME = "exchange_fuse.cuh"
TAG = "6ea910cac1010f2e"
WRAPPERS = (('q8_gather_pack', 'V4::q8_gather_pack'), ('q8_place', 'V4::q8_place'))

SOURCE = r"""#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace v4 {

__device__ __forceinline__ uint32_t cvt_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

__device__ __forceinline__ void ue8m0_scale(float amax, uint32_t& sf, float& mult) {
  const float normalized = __fmul_rn(amax, __uint_as_float(0x3B124925u));
  const int bits = __float_as_int(normalized);
  const int exponent = (bits >> 23) & 255;
  const int mantissa = bits & 0x7FFFFF;
  const bool bump = (mantissa != 0) && !((exponent == 0) && (mantissa <= 0x400000));
  const int e = min(exponent + (bump ? 1 : 0), 254);
  const int s = (normalized <= 0.0f) ? 0 : e;
  sf = static_cast<uint32_t>(s);
  mult = (s == 0) ? 0.0f : __int_as_float((254 - s) << 23);
}

constexpr int kSrcGroups = 48;
constexpr int kWidth = kSrcGroups * 32;
constexpr int kPacked = kWidth + kSrcGroups;
constexpr int kWorld = 4;
constexpr int kGroups = kWorld * kSrcGroups;

template <int kRows>
__global__ __launch_bounds__(kSrcGroups * kRows) void q8_pack_kernel(const uint4* __restrict__ values,
                                                                     uint8_t* __restrict__ packed, int rows) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRows + static_cast<int>(threadIdx.x) / kSrcGroups;
  const int g = static_cast<int>(threadIdx.x) % kSrcGroups;
  const bool live = row < rows;
  uint4 v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = make_uint4(0u, 0u, 0u, 0u);
  if (live) {
    const uint4* src = values + static_cast<int64_t>(row) * (kWidth / 8) + g * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __ldg(src + j);
  }
  float f[32];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t w[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[8 * j + 2 * i] = bf16_lo(w[i]);
      f[8 * j + 2 * i + 1] = bf16_hi(w[i]);
    }
  }
  float amax = fabsf(f[0]);
#pragma unroll
  for (int i = 1; i < 32; ++i) amax = fmaxf(amax, fabsf(f[i]));
  uint32_t sf;
  float mult;
  ue8m0_scale(amax, sf, mult);
  uint32_t qw[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float t[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) t[k] = fminf(__fmul_rn(__fadd_rn(f[4 * i + k], 0.0f), mult), 448.0f);
    qw[i] = cvt_e4m3x2(t[0], t[1]) | (cvt_e4m3x2(t[2], t[3]) << 16);
  }
  uint32_t word = sf;
  word |= __shfl_down_sync(0xffffffffu, word, 1) << 8;
  word |= __shfl_down_sync(0xffffffffu, word, 2) << 16;
  if (!live) return;
  uint8_t* dst = packed + static_cast<int64_t>(row) * kPacked;
  reinterpret_cast<uint4*>(dst + g * 32)[0] = make_uint4(qw[0], qw[1], qw[2], qw[3]);
  reinterpret_cast<uint4*>(dst + g * 32)[1] = make_uint4(qw[4], qw[5], qw[6], qw[7]);
  if ((lane & 3) == 0) reinterpret_cast<uint32_t*>(dst + kWidth)[g >> 2] = word;
}

__global__ __launch_bounds__(kGroups) void q8_place_kernel(const uint8_t* __restrict__ received, uint4* __restrict__ q,
                                                           uint32_t* __restrict__ s, int mine, int n_local) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x;
  const int gg = threadIdx.x;
  const int src = gg / kSrcGroups;
  const int g = gg - src * kSrcGroups;
  uint4 c0 = make_uint4(0u, 0u, 0u, 0u), c1 = c0;
  uint32_t sf = 0;
  if (row < mine) {
    const uint8_t* p = received + (static_cast<int64_t>(src) * mine + row) * kPacked;
    c0 = __ldg(reinterpret_cast<const uint4*>(p + g * 32));
    c1 = __ldg(reinterpret_cast<const uint4*>(p + g * 32) + 1);
    sf = __ldg(p + kWidth + g);
  }
  if (row < n_local) {
    uint4* dst = q + (static_cast<int64_t>(row) * kGroups + gg) * 2;
    dst[0] = c0;
    dst[1] = c1;
  }
  uint32_t word = sf;
  word |= __shfl_down_sync(0xffffffffu, word, 1) << 8;
  word |= __shfl_down_sync(0xffffffffu, word, 2) << 16;
  if ((lane & 3) == 0) {
    s[static_cast<int64_t>(row >> 7) * (kGroups * 32) + (gg >> 2) * 128 + (row & 31) * 4 + ((row >> 5) & 3)] = word;
  }
}

__device__ __forceinline__ float e4m3_to_f32(uint32_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
  return __half2float(__half(h));
}

template <int kRows>
__global__ __launch_bounds__(kSrcGroups * kRows) void q8_gather_pack_kernel(
    const uint8_t* __restrict__ weight, const uint8_t* __restrict__ scale, const int64_t* __restrict__ ids,
    uint8_t* __restrict__ packed, int rows, int64_t row_lo, int64_t row_hi) {
  constexpr int kDim = 256, kBlk = 32, kCols = kWidth / kDim;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRows + static_cast<int>(threadIdx.x) / kSrcGroups;
  const int g = static_cast<int>(threadIdx.x) % kSrcGroups;
  const bool live = row < rows;
  const int j = g / (kDim / 32), k = g % (kDim / 32);
  float f[32];
  if (live) {
    const int64_t idx = __ldg(ids + static_cast<int64_t>(row) * kCols + j);
    const bool owned = idx >= row_lo && idx < row_hi;
    const int64_t local = owned ? idx - row_lo : 0;
    uint4 w0 = make_uint4(0u, 0u, 0u, 0u), w1 = w0;
    uint32_t e = 0;
    if (owned) {
      const uint4* src = reinterpret_cast<const uint4*>(weight + local * kDim + k * 32);
      w0 = __ldg(src);
      w1 = __ldg(src + 1);
      e = __ldg(scale + local * (kDim / kBlk) + k);
    }

    const float sc = (e == 0) ? __uint_as_float(0x00400000u) : __int_as_float(static_cast<int>(e << 23));
    const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const float prod = __fmul_rn(e4m3_to_f32((ww[i >> 2] >> (8 * (i & 3))) & 0xFFu), sc);

      f[i] = owned ? __bfloat162float(__float2bfloat16_rn(prod)) : 0.0f;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 32; ++i) f[i] = 0.0f;
  }
  float amax = fabsf(f[0]);
#pragma unroll
  for (int i = 1; i < 32; ++i) amax = fmaxf(amax, fabsf(f[i]));
  uint32_t sf;
  float mult;
  ue8m0_scale(amax, sf, mult);
  uint32_t qw[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float t[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) t[kk] = fminf(__fmul_rn(__fadd_rn(f[4 * i + kk], 0.0f), mult), 448.0f);
    qw[i] = cvt_e4m3x2(t[0], t[1]) | (cvt_e4m3x2(t[2], t[3]) << 16);
  }
  uint32_t word = sf;
  word |= __shfl_down_sync(0xffffffffu, word, 1) << 8;
  word |= __shfl_down_sync(0xffffffffu, word, 2) << 16;
  if (!live) return;
  uint8_t* dst = packed + static_cast<int64_t>(row) * kPacked;
  reinterpret_cast<uint4*>(dst + g * 32)[0] = make_uint4(qw[0], qw[1], qw[2], qw[3]);
  reinterpret_cast<uint4*>(dst + g * 32)[1] = make_uint4(qw[4], qw[5], qw[6], qw[7]);
  if ((lane & 3) == 0) reinterpret_cast<uint32_t*>(dst + kWidth)[g >> 2] = word;
}

}

struct V4 {

  static void q8_pack(tvm::ffi::TensorView values, tvm::ffi::TensorView packed) {
    using namespace sglang::host;
    auto R = SymbolicSize{"rows"};
    auto dev = SymbolicDevice{};
    TensorMatcher({R, v4::kWidth}).with_strides({v4::kWidth, 1}).with_device<kDLCUDA>(dev)
        .with_dtype<sglang::bf16_t>().verify(values);
    TensorMatcher({R, v4::kPacked}).with_strides({v4::kPacked, 1}).with_device<kDLCUDA>(dev)
        .with_dtype<uint8_t>().verify(packed);
    RuntimeCheck(reinterpret_cast<uintptr_t>(values.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 == 0, "v4 EQ8: misaligned buffers");
    const int64_t rows = R.unwrap();
    if (rows == 0) return;
    RuntimeCheck(rows < (int64_t(1) << 31) / v4::kPacked, "v4 EQ8: too many rows");
    constexpr int kRows = 4;
    LaunchKernel(dim3(static_cast<unsigned>((rows + kRows - 1) / kRows)), v4::kSrcGroups * kRows, dev.unwrap())(
        v4::q8_pack_kernel<kRows>, static_cast<const uint4*>(values.data_ptr()),
        static_cast<uint8_t*>(packed.data_ptr()), static_cast<int>(rows));
  }

  static void q8_gather_pack(tvm::ffi::TensorView weight, tvm::ffi::TensorView scale, tvm::ffi::TensorView ids,
                             tvm::ffi::TensorView packed, int64_t row_lo, int64_t row_hi) {
    using namespace sglang::host;
    auto SR = SymbolicSize{"shard_rows"};
    auto R = SymbolicSize{"rows"};
    auto dev = SymbolicDevice{};
    TensorMatcher({SR, 256}).with_strides({256, 1}).with_device<kDLCUDA>(dev).verify(weight);
    TensorMatcher({SR, 8}).with_strides({8, 1}).with_device<kDLCUDA>(dev).with_dtype<uint8_t>().verify(scale);
    TensorMatcher({R, 6}).with_strides({6, 1}).with_device<kDLCUDA>(dev).with_dtype<int64_t>().verify(ids);
    TensorMatcher({R, v4::kPacked}).with_strides({v4::kPacked, 1}).with_device<kDLCUDA>(dev)
        .with_dtype<uint8_t>().verify(packed);
    RuntimeCheck(weight.dtype().bits == 8, "v4 EQ8G: the table must be 1-byte e4m3");
    RuntimeCheck(row_hi - row_lo == SR.unwrap(), "v4 EQ8G: row_hi - row_lo must be the shard's rows");
    RuntimeCheck(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 == 0, "v4 EQ8G: misaligned buffers");
    const int64_t rows = R.unwrap();
    if (rows == 0) return;
    RuntimeCheck(rows < (int64_t(1) << 31) / v4::kPacked, "v4 EQ8G: too many rows");
    constexpr int kRows = 4;
    LaunchKernel(dim3(static_cast<unsigned>((rows + kRows - 1) / kRows)), v4::kSrcGroups * kRows, dev.unwrap())(
        v4::q8_gather_pack_kernel<kRows>, static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(scale.data_ptr()), static_cast<const int64_t*>(ids.data_ptr()),
        static_cast<uint8_t*>(packed.data_ptr()), static_cast<int>(rows), row_lo, row_hi);
  }

  static void q8_place(tvm::ffi::TensorView received, tvm::ffi::TensorView q, tvm::ffi::TensorView s) {
    using namespace sglang::host;
    auto WM = SymbolicSize{"world_rows"};
    auto NL = SymbolicSize{"n_local"};
    auto dev = SymbolicDevice{};
    TensorMatcher({WM, v4::kPacked}).with_strides({v4::kPacked, 1}).with_device<kDLCUDA>(dev)
        .with_dtype<uint8_t>().verify(received);
    TensorMatcher({NL, v4::kGroups * 32}).with_strides({v4::kGroups * 32, 1}).with_device<kDLCUDA>(dev).verify(q);
    RuntimeCheck(q.dtype().bits == 8, "v4 EQ8: q must be a 1-byte dtype");
    RuntimeCheck(WM.unwrap() % v4::kWorld == 0, "v4 EQ8: received rows must be 4 * mine");
    const int64_t mine = WM.unwrap() / v4::kWorld, n_local = NL.unwrap();
    RuntimeCheck(n_local >= mine, "v4 EQ8: n_local < mine");
    const int64_t n_pad = (n_local + 127) / 128 * 128;
    TensorMatcher({n_pad * v4::kGroups}).with_device<kDLCUDA>(dev).with_dtype<uint8_t>().verify(s);
    RuntimeCheck(reinterpret_cast<uintptr_t>(received.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(s.data_ptr()) % 4 == 0,
                 "v4 EQ8: misaligned buffers");
    if (n_pad == 0) return;
    RuntimeCheck(n_pad * v4::kGroups < (int64_t(1) << 31), "v4 EQ8: too many rows");
    LaunchKernel(dim3(static_cast<unsigned>(n_pad)), v4::kGroups, dev.unwrap())(
        v4::q8_place_kernel, static_cast<const uint8_t*>(received.data_ptr()), static_cast<uint4*>(q.data_ptr()),
        static_cast<uint32_t*>(s.data_ptr()), static_cast<int>(mine), static_cast<int>(n_local));
  }
};
"""
