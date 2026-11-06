
NAME = "prefill_fuse.cuh"
TAG = "f4dd688ccb81c0d2"
WRAPPERS = (('qnorm', 'V4::qnorm'), ('wob_scale', 'V4::wob_scale'), ('engram_place', 'V4::engram_place'), ('engram_quant', 'V4::engram_quant'), ('dequant_v41', 'V4::dequant_v41'), ('swa_ids', 'V4::swa_ids'), ('index_q', 'V4Idx::index_q'))

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

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float rsqrt_approx_ftz(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

__device__ __forceinline__ uint32_t cvt_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ uint32_t cvt_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

constexpr int kQK = 1280;
constexpr int kQChunks = kQK / 256;
constexpr int kQGroups = kQK / 32;
constexpr int kQKTiles = kQGroups / 4;
constexpr int kQThreads = 256;

__device__ __forceinline__ float thread_sumsq(const float (&v)[8]) {
  float s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = __fmul_rn(v[i], v[i]);
  const float a0 = __fadd_rn(s[0], s[2]), a1 = __fadd_rn(s[1], s[3]);
  const float b0 = __fadd_rn(s[4], s[6]), b1 = __fadd_rn(s[5], s[7]);
  return __fadd_rn(__fadd_rn(a0, b0), __fadd_rn(a1, b1));
}

__device__ __forceinline__ float bfly(float t, int off) { return __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, off)); }

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

template <int kRows, int kMinBlocks>
__global__ __launch_bounds__(kQThreads, kMinBlocks) void qnorm_mxfp8_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t sx, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y,
    uint8_t* __restrict__ q, uint32_t* __restrict__ s, int m, int m_pad, float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * (kQThreads / 32) + (threadIdx.x >> 5);
  const int row0 = warp * kRows;
  if (row0 >= m_pad) return;
  uint4 xv[kRows][kQChunks];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const int row = row0 + r;
    if (row < m) {
      const __nv_bfloat16* src = x + static_cast<int64_t>(row) * sx + 8 * lane;
#pragma unroll
      for (int c = 0; c < kQChunks; ++c) xv[r][c] = __ldg(reinterpret_cast<const uint4*>(src + 256 * c));
    }
  }
  uint4 wv[kQChunks];
#pragma unroll
  for (int c = 0; c < kQChunks; ++c) wv[c] = __ldg(reinterpret_cast<const uint4*>(w + 256 * c + 8 * lane));
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const int row = row0 + r;
    if (row >= m_pad) break;
    const int64_t tile_words = static_cast<int64_t>(row >> 7) * (kQKTiles * 128);
    const int slot = (row & 31) * 4 + ((row >> 5) & 3);
    if (row >= m) {
      if (lane < kQKTiles) s[tile_words + lane * 128 + slot] = 0u;
      continue;
    }
    float part[kQChunks];
#pragma unroll
    for (int c = 0; c < kQChunks; ++c) {
      const uint32_t xx[4] = {xv[r][c].x, xv[r][c].y, xv[r][c].z, xv[r][c].w};
      float v[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[2 * i] = bf16_lo(xx[i]);
        v[2 * i + 1] = bf16_hi(xx[i]);
      }
      float t = thread_sumsq(v);
      t = bfly(t, 16);
      t = bfly(t, 8);
      t = bfly(t, 4);
      t = bfly(t, 2);
      t = bfly(t, 1);
      part[c] = t;
    }

    const float zero = 0.0f;
    const float a0 = __fadd_rn(part[0], part[4]), a1 = __fadd_rn(part[1], zero);
    const float a2 = __fadd_rn(part[2], zero), a3 = __fadd_rn(part[3], zero);
    const float total = __fadd_rn(__fadd_rn(a0, a2), __fadd_rn(a1, a3));
    const float inv = rsqrt_approx_ftz(__fadd_rn(div_full(total, static_cast<float>(kQK)), eps));
    const int64_t base = static_cast<int64_t>(row) * kQK + 8 * lane;
#pragma unroll
    for (int c = 0; c < kQChunks; ++c) {
      const uint32_t xx[4] = {xv[r][c].x, xv[r][c].y, xv[r][c].z, xv[r][c].w};
      const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w};
      uint32_t yw[4];
      float yf[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float lo = __fmul_rn(__fmul_rn(inv, bf16_lo(xx[i])), bf16_lo(ww[i]));
        const float hi = __fmul_rn(__fmul_rn(inv, bf16_hi(xx[i])), bf16_hi(ww[i]));
        yw[i] = cvt_bf16x2(lo, hi);
        yf[2 * i] = bf16_lo(yw[i]);
        yf[2 * i + 1] = bf16_hi(yw[i]);
      }
      *reinterpret_cast<uint4*>(y + base + 256 * c) = make_uint4(yw[0], yw[1], yw[2], yw[3]);
      float amax = fabsf(yf[0]);
#pragma unroll
      for (int i = 1; i < 8; ++i) amax = fmaxf(amax, fabsf(yf[i]));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
      uint32_t sf;
      float mult;
      ue8m0_scale(amax, sf, mult);
      uint32_t qw[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float t[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) t[j] = fminf(fmaxf(__fmul_rn(yf[4 * i + j], mult), -448.0f), 448.0f);
        qw[i] = cvt_e4m3x2(t[0], t[1]) | (cvt_e4m3x2(t[2], t[3]) << 16);
      }
      *reinterpret_cast<uint2*>(q + base + 256 * c) = make_uint2(qw[0], qw[1]);

      const int half = lane & 16;
      const uint32_t b0 = __shfl_sync(0xffffffffu, sf, half + 0);
      const uint32_t b1 = __shfl_sync(0xffffffffu, sf, half + 4);
      const uint32_t b2 = __shfl_sync(0xffffffffu, sf, half + 8);
      const uint32_t b3 = __shfl_sync(0xffffffffu, sf, half + 12);
      if ((lane & 15) == 0) {
        const int kt = 2 * c + (half >> 4);
        s[tile_words + kt * 128 + slot] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
      }
    }
  }
}

__global__ __launch_bounds__(256) void wob_scale_kernel(
    const int32_t* __restrict__ in, int64_t s0, int64_t s1, int rows, int kw_count, int64_t units, int4* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  const int64_t warps = static_cast<int64_t>(gridDim.x) * 8;
  for (int64_t u = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5); u < units; u += warps) {
    const int64_t tile = u / kw_count;
    const int64_t kw = u - tile * kw_count;
    int32_t v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int64_t mrow = tile * 128 + j * 32 + lane;
      v[j] = mrow < rows ? __ldg(in + mrow * s0 + kw * s1) : 0;
    }
    out[u * 32 + lane] = make_int4(v[0], v[1], v[2], v[3]);
  }
}

__device__ __forceinline__ uint32_t plus_zero(uint32_t p) {
  return cvt_bf16x2(__fadd_rn(bf16_lo(p), 0.0f), __fadd_rn(bf16_hi(p), 0.0f));
}

template <int kWorld, int kChunks, int kUnroll>
__global__ __launch_bounds__(256) void engram_place_kernel(const uint4* __restrict__ recv, uint4* __restrict__ out, int mine) {
  const int total = mine * kWorld * kChunks;
  const int stride = gridDim.x * blockDim.x;
  for (int i0 = blockIdx.x * blockDim.x * kUnroll + threadIdx.x; i0 < total; i0 += stride * kUnroll) {
    uint4 v[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      const int i = i0 + k * blockDim.x;
      if (i < total) {
        const int ms = i / kChunks;
        const int c = i - ms * kChunks;
        const int mrow = ms / kWorld;
        const int src_s = ms - mrow * kWorld;
        v[k] = __ldg(recv + (static_cast<int64_t>(src_s) * mine + mrow) * kChunks + c);
      }
    }
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      const int i = i0 + k * blockDim.x;
      if (i < total) {
        uint4 o;
        o.x = plus_zero(v[k].x);
        o.y = plus_zero(v[k].y);
        o.z = plus_zero(v[k].z);
        o.w = plus_zero(v[k].w);
        out[i] = o;
      }
    }
  }
}

__global__ __launch_bounds__(192) void engram_quant_kernel(
    const uint4* __restrict__ recv, uint4* __restrict__ q, uint32_t* __restrict__ s, int mine, int n_local) {
  constexpr int kSrcGroups = 48, kGroups = 192;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x;
  const int g = threadIdx.x;
  const int src = g / kSrcGroups;
  const int gs = g - src * kSrcGroups;
  uint4 v[4];
  if (row < mine) {
    const uint4* src_p = recv + (static_cast<int64_t>(src) * mine + row) * (kSrcGroups * 4) + gs * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __ldg(src_p + j);
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = make_uint4(0u, 0u, 0u, 0u);
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
  if (row < n_local) {
    uint32_t qw[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float t[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) t[k] = fminf(__fmul_rn(__fadd_rn(f[4 * i + k], 0.0f), mult), 448.0f);
      qw[i] = cvt_e4m3x2(t[0], t[1]) | (cvt_e4m3x2(t[2], t[3]) << 16);
    }
    uint4* dst = q + (static_cast<int64_t>(row) * kGroups + g) * 2;
    dst[0] = make_uint4(qw[0], qw[1], qw[2], qw[3]);
    dst[1] = make_uint4(qw[4], qw[5], qw[6], qw[7]);
  }
  uint32_t word = sf;
  word |= __shfl_down_sync(0xffffffffu, word, 1) << 8;
  word |= __shfl_down_sync(0xffffffffu, word, 2) << 16;
  if ((lane & 3) == 0) {
    s[static_cast<int64_t>(row >> 7) * (kGroups * 32) + (g >> 2) * 128 + (row & 31) * 4 + ((row >> 5) & 3)] = word;
  }
}

__device__ __forceinline__ float ue8m0_to_fp32(uint32_t b) {
  return b == 0u ? __uint_as_float(0x00400000u) : (b == 255u ? __uint_as_float(0x7FC00000u) : __uint_as_float(b << 23));
}

__device__ __forceinline__ float2 e4m3x2_to_f32x2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  return __half22float2(*reinterpret_cast<__half2*>(&h));
}

template <typename IdT>
__global__ __launch_bounds__(256) void dequant_v41_fp8_kernel(
    const uint8_t* __restrict__ cache, const IdT* __restrict__ ids, __nv_bfloat16* __restrict__ out, int64_t out_stride,
    int64_t n, int64_t bytes_per_page, int64_t page_size, int64_t data_stride, int64_t scale_base, int64_t scale_stride) {
  const int64_t t = (static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x) >> 4;
  const int hl = threadIdx.x & 15;
  if (t >= n) return;
  const int64_t loc = static_cast<int64_t>(ids[t]);
  const int64_t page = loc / page_size, in_page = loc - page * page_size;
  const uint8_t* base = cache + page * bytes_per_page;
  const uint4* data = reinterpret_cast<const uint4*>(base + in_page * data_stride + 32 * hl);
  const uint4 d0 = __ldg(data), rehome = __ldg(data + 1);
  const float scale = ue8m0_to_fp32(__ldg(base + scale_base + in_page * scale_stride + hl));
  const uint32_t dw[8] = {d0.x, d0.y, d0.z, d0.w, rehome.x, rehome.y, rehome.z, rehome.w};
  uint32_t o[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 lo = e4m3x2_to_f32x2(static_cast<uint16_t>(dw[i] & 0xFFFFu));
    const float2 hi = e4m3x2_to_f32x2(static_cast<uint16_t>(dw[i] >> 16));
    o[2 * i] = cvt_bf16x2(__fmul_rn(lo.x, scale), __fmul_rn(lo.y, scale));
    o[2 * i + 1] = cvt_bf16x2(__fmul_rn(hi.x, scale), __fmul_rn(hi.y, scale));
  }
  uint4* dst = reinterpret_cast<uint4*>(out + t * out_stride + 32 * hl);
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
}

__global__ __launch_bounds__(256) void swa_ids_kernel(
    int32_t* __restrict__ out, const int32_t* __restrict__ first_pos, const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ req_pool, const int32_t* __restrict__ req_to_token, int64_t rtt_stride,
    const int64_t* __restrict__ full_to_swa, int num_reqs, int64_t total) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
  if (t >= total) return;
  int lo = 0, hi = num_reqs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (static_cast<int64_t>(__ldg(offsets + mid)) <= t) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int64_t start = __ldg(offsets + lo);
  if (t < start || t >= static_cast<int64_t>(__ldg(offsets + lo + 1))) return;
  const int64_t pos = static_cast<int64_t>(__ldg(first_pos + lo)) + (t - start);
  const int64_t full =
      static_cast<int64_t>(__ldg(req_to_token + static_cast<int64_t>(__ldg(req_pool + lo)) * rtt_stride + pos));
  out[t] = static_cast<int32_t>(__ldg(full_to_swa + full));
}

}

struct V4 {

  static void qnorm(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView y, tvm::ffi::TensorView q,
                    tvm::ffi::TensorView s, double eps) {
    using namespace sglang::host;
    auto M = SymbolicSize{"rows"};
    auto dev = SymbolicDevice{};
    TensorMatcher({M, v4::kQK}).with_strides({-1, 1}).with_device<kDLCUDA>(dev).with_dtype<sglang::bf16_t>().verify(x);
    TensorMatcher({v4::kQK}).with_device<kDLCUDA>(dev).with_dtype<sglang::bf16_t>().verify(w);
    TensorMatcher({M, v4::kQK}).with_strides({v4::kQK, 1}).with_device<kDLCUDA>(dev).with_dtype<sglang::bf16_t>().verify(y);
    TensorMatcher({M, v4::kQK}).with_strides({v4::kQK, 1}).with_device<kDLCUDA>(dev).verify(q);
    const int64_t m = M.unwrap();
    const int64_t m_pad = (m + 127) / 128 * 128;
    TensorMatcher({v4::kQGroups * m_pad}).with_device<kDLCUDA>(dev).with_dtype<uint8_t>().verify(s);
    RuntimeCheck(x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                 "v4 QNORM: x rows must be 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(s.data_ptr()) % 16 == 0,
                 "v4 QNORM: buffers must be 16-byte aligned");
    RuntimeCheck(m < (int64_t(1) << 30), "v4 QNORM: too many rows");
    if (m == 0) return;
    const auto device = dev.unwrap();
    auto launch = [&](auto kernel, int rows) {
      const int64_t warps = (m_pad + rows - 1) / rows;
      const unsigned blocks = static_cast<unsigned>((warps + 7) / 8);
      LaunchKernel(dim3(blocks), v4::kQThreads, device)(
          kernel, static_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int64_t>(x.stride(0)),
          static_cast<const __nv_bfloat16*>(w.data_ptr()), static_cast<__nv_bfloat16*>(y.data_ptr()),
          static_cast<uint8_t*>(q.data_ptr()), static_cast<uint32_t*>(s.data_ptr()), static_cast<int>(m),
          static_cast<int>(m_pad), static_cast<float>(eps));
    };
    launch(v4::qnorm_mxfp8_kernel<2, 2>, 2);
  }

  static void wob_scale(tvm::ffi::TensorView sf, tvm::ffi::TensorView out, int64_t blocks) {
    using namespace sglang::host;
    auto R = SymbolicSize{"rows"};
    auto KW = SymbolicSize{"words"};
    auto dev = SymbolicDevice{};
    TensorMatcher({R, KW}).with_strides({-1, -1}).with_device<kDLCUDA>(dev).with_dtype<int32_t>().verify(sf);
    const int64_t rows = R.unwrap(), kw = KW.unwrap();
    const int64_t tiles = (rows + 127) / 128;
    TensorMatcher({tiles * 128 * kw * 4}).with_device<kDLCUDA>(dev).with_dtype<uint8_t>().verify(out);
    RuntimeCheck(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "v4 WOBSF: out must be 16-byte aligned");
    RuntimeCheck(rows < (int64_t(1) << 30) && kw < (int64_t(1) << 20), "v4 WOBSF: shape too large");
    if (rows == 0 || kw == 0) return;
    const int64_t units = tiles * kw;
    const int64_t need = (units + 7) / 8;
    const unsigned grid = static_cast<unsigned>(blocks > 0 ? (need < blocks ? need : blocks) : need);
    LaunchKernel(dim3(grid), 256, dev.unwrap())(
        v4::wob_scale_kernel, static_cast<const int32_t*>(sf.data_ptr()), static_cast<int64_t>(sf.stride(0)),
        static_cast<int64_t>(sf.stride(1)), static_cast<int>(rows), static_cast<int>(kw), units,
        static_cast<int4*>(out.data_ptr()));
  }

  static void engram_place(tvm::ffi::TensorView recv, tvm::ffi::TensorView out, int64_t world, int64_t blocks) {
    using namespace sglang::host;
    auto WM = SymbolicSize{"world_rows"};
    auto WD = SymbolicSize{"width"};
    auto OR = SymbolicSize{"out_rows"};
    auto OW = SymbolicSize{"out_width"};
    auto dev = SymbolicDevice{};
    TensorMatcher({WM, WD}).with_strides({-1, 1}).with_device<kDLCUDA>(dev).with_dtype<sglang::bf16_t>().verify(recv);
    TensorMatcher({OR, OW}).with_strides({-1, 1}).with_device<kDLCUDA>(dev).with_dtype<sglang::bf16_t>().verify(out);
    const int64_t width = WD.unwrap();
    RuntimeCheck(world > 0 && WM.unwrap() % world == 0, "v4 XPLACE: recv rows must be world * mine");
    const int64_t mine = WM.unwrap() / world;
    RuntimeCheck(OW.unwrap() == world * width && OR.unwrap() >= mine, "v4 XPLACE: out must be [>= mine, world * width]");
    RuntimeCheck(recv.stride(0) == width && out.stride(0) == world * width, "v4 XPLACE: contiguous rows only");
    RuntimeCheck(width % 8 == 0 && reinterpret_cast<uintptr_t>(recv.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "v4 XPLACE: 16-byte vectors only");
    if (mine == 0 || width == 0) return;
    RuntimeCheck(world == 4 && width == 1536, "v4 XPLACE serves the 1017 Engram exchange (world 4, width 1536)");
    RuntimeCheck(mine * world * (width / 8) < (int64_t(1) << 31), "v4 XPLACE: too many rows");
    constexpr int kUnroll = 4;
    const int64_t total = mine * 4 * 192;
    const int64_t need = (total + 256 * kUnroll - 1) / (256 * kUnroll);
    const unsigned grid = static_cast<unsigned>(blocks > 0 ? (need < blocks ? need : blocks) : need);
    LaunchKernel(dim3(grid), 256, dev.unwrap())(
        v4::engram_place_kernel<4, 192, kUnroll>, static_cast<const uint4*>(recv.data_ptr()),
        static_cast<uint4*>(out.data_ptr()), static_cast<int>(mine));
  }

  static void engram_quant(tvm::ffi::TensorView recv, tvm::ffi::TensorView q, tvm::ffi::TensorView s, int64_t world) {
    using namespace sglang::host;
    auto WM = SymbolicSize{"world_rows"};
    auto WD = SymbolicSize{"width"};
    auto NL = SymbolicSize{"n_local"};
    auto OW = SymbolicSize{"out_width"};
    auto dev = SymbolicDevice{};
    TensorMatcher({WM, WD}).with_strides({-1, 1}).with_device<kDLCUDA>(dev).with_dtype<sglang::bf16_t>().verify(recv);
    TensorMatcher({NL, OW}).with_strides({-1, 1}).with_device<kDLCUDA>(dev).verify(q);
    const int64_t width = WD.unwrap(), n_local = NL.unwrap();
    RuntimeCheck(world > 0 && WM.unwrap() % world == 0, "v4 XQUANT: recv rows must be world * mine");
    const int64_t mine = WM.unwrap() / world;
    RuntimeCheck(OW.unwrap() == world * width && n_local >= mine && q.stride(0) == world * width &&
                 recv.stride(0) == width, "v4 XQUANT: q must be [n_local >= mine, world * width] contiguous");
    RuntimeCheck(q.dtype().bits == 8, "v4 XQUANT: q must be a 1-byte dtype");
    RuntimeCheck(width % 32 == 0, "v4 XQUANT: width must be whole 32-element groups");
    const int64_t groups = world * width / 32;
    RuntimeCheck(groups % 4 == 0, "v4 XQUANT: the group count must be a multiple of 4");
    const int64_t n_pad = (n_local + 127) / 128 * 128;
    TensorMatcher({n_pad * groups}).with_device<kDLCUDA>(dev).with_dtype<uint8_t>().verify(s);
    RuntimeCheck(reinterpret_cast<uintptr_t>(recv.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(s.data_ptr()) % 4 == 0, "v4 XQUANT: misaligned buffers");
    if (n_pad == 0) return;
    RuntimeCheck(world == 4 && width == 1536, "v4 XQUANT serves the 1017 Engram exchange (world 4, width 1536)");
    RuntimeCheck(n_pad * 768 < (int64_t(1) << 31), "v4 XQUANT: too many rows");
    LaunchKernel(dim3(static_cast<unsigned>(n_pad)), 192, dev.unwrap())(
        v4::engram_quant_kernel, static_cast<const uint4*>(recv.data_ptr()), static_cast<uint4*>(q.data_ptr()),
        static_cast<uint32_t*>(s.data_ptr()), static_cast<int>(mine), static_cast<int>(n_local));
  }

  static void dequant_v41(tvm::ffi::TensorView cache, tvm::ffi::TensorView ids, tvm::ffi::TensorView out, int64_t page_size,
                          int64_t data_stride, int64_t scale_base, int64_t scale_stride) {
    using namespace sglang::host;
    auto N = SymbolicSize{"tokens"};
    auto dev = SymbolicDevice{};
    auto id_dtype = SymbolicDType{};
    TensorMatcher({-1, -1}).with_device<kDLCUDA>(dev).verify(cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_dtype).with_device<kDLCUDA>(dev).verify(ids);
    TensorMatcher({N, 1, 512}).with_strides({-1, -1, 1}).with_dtype<sglang::bf16_t>().with_device<kDLCUDA>(dev).verify(out);
    RuntimeCheck(cache.dtype().bits == 8, "v4 DEQ: the cache must be a 1-byte view");
    const int64_t n = N.unwrap();
    const int64_t bytes_per_page = cache.size(1);
    RuntimeCheck(bytes_per_page % 16 == 0 && data_stride % 16 == 0 && reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0 &&
                 out.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                 "v4 DEQ: 16-byte rows only");
    if (n == 0) return;
    const unsigned grid = static_cast<unsigned>((n + 15) / 16);
    auto launch = [&](auto kernel, auto* ids_ptr) {
      LaunchKernel(dim3(grid), 256, dev.unwrap())(
          kernel, static_cast<const uint8_t*>(cache.data_ptr()), ids_ptr, static_cast<__nv_bfloat16*>(out.data_ptr()),
          static_cast<int64_t>(out.stride(0)), n, bytes_per_page, page_size, data_stride, scale_base, scale_stride);
    };
    if (id_dtype.is_type<int32_t>()) {
      launch(v4::dequant_v41_fp8_kernel<int32_t>, static_cast<const int32_t*>(ids.data_ptr()));
    } else {
      launch(v4::dequant_v41_fp8_kernel<int64_t>, static_cast<const int64_t*>(ids.data_ptr()));
    }
  }

  static void swa_ids(tvm::ffi::TensorView out, tvm::ffi::TensorView first_pos, tvm::ffi::TensorView offsets,
                      tvm::ffi::TensorView req_pool, tvm::ffi::TensorView req_to_token, tvm::ffi::TensorView full_to_swa) {
    using namespace sglang::host;
    auto T = SymbolicSize{"total"};
    auto R = SymbolicSize{"reqs"};
    auto R1 = SymbolicSize{"reqs_plus_one"};
    auto dev = SymbolicDevice{};
    TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(out);
    TensorMatcher({R}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(first_pos);
    TensorMatcher({R1}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(offsets);
    TensorMatcher({R}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(req_pool);
    TensorMatcher({-1, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(req_to_token);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(dev).verify(full_to_swa);
    RuntimeCheck(R1.unwrap() == R.unwrap() + 1, "v4 SWAIDS: offsets must have reqs + 1 entries");
    const int64_t total = T.unwrap();
    if (total == 0 || R.unwrap() == 0) return;
    LaunchKernel(dim3(static_cast<unsigned>((total + 255) / 256)), 256, dev.unwrap())(
        v4::swa_ids_kernel, static_cast<int32_t*>(out.data_ptr()), static_cast<const int32_t*>(first_pos.data_ptr()),
        static_cast<const int32_t*>(offsets.data_ptr()), static_cast<const int32_t*>(req_pool.data_ptr()),
        static_cast<const int32_t*>(req_to_token.data_ptr()), static_cast<int64_t>(req_to_token.stride(0)),
        static_cast<const int64_t*>(full_to_swa.data_ptr()), static_cast<int>(R.unwrap()), total);
  }
};

#include <sgl_kernel/deepseek_v4/fp4_utils.cuh>

namespace v4 {

struct IdxqParams {
  const uint4* __restrict__ input;
  const float* __restrict__ freqs;
  const void* __restrict__ positions;
  uint32_t* __restrict__ payload;
  int32_t* __restrict__ scale;
  const uint16_t* __restrict__ head_weights;
  float* __restrict__ weights_out;
  float weight_scale;
  uint32_t num_rows;
  uint32_t heads;
};

__device__ __forceinline__ uint32_t index_pack_exponent(float amax) {
  const auto exponent = sglang::deepseek_v4::fp8::cast_to_ue8m0(fmaxf(amax / 6.0f, 1.0e-4f));
  return static_cast<uint32_t>(min(max(exponent, 1), 254));
}

__device__ __forceinline__ uint32_t clear_negative_zero(uint32_t packed) {
  const auto any_magnitude = (packed | (packed >> 1) | (packed >> 2)) & 0x11111111u;
  return packed & ((any_magnitude << 3) | 0x77777777u);
}

__device__ __forceinline__ float block_max4(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

template <int kWarpHeads, typename PosT>
__global__ __launch_bounds__(256, 4) void index_q_tok_kernel(const IdxqParams p, uint32_t num_tokens) {
  namespace fp4 = sglang::deepseek_v4::fp4;
  constexpr int kHeads = 32;
  constexpr int kWarpsPerToken = kHeads / kWarpHeads;
  constexpr int kIters = kWarpHeads / 2;
  const int lane = threadIdx.x & 31;
  const int hl = lane & 15;
  const int sub = lane >> 4;
  const uint32_t gw = blockIdx.x * 8 + (threadIdx.x >> 5);
  const uint32_t token = gw / kWarpsPerToken;
  const int head0 = static_cast<int>(gw % kWarpsPerToken) * kWarpHeads;
  if (token >= num_tokens) return;
  float fr[8];
  if (hl >= 8) {
    const int64_t pos = static_cast<int64_t>(static_cast<const PosT*>(p.positions)[token]);
    const float4* f = reinterpret_cast<const float4*>(p.freqs + pos * 64 + 8 * (hl - 8));
    const float4 f0 = __ldg(f), f1 = __ldg(f + 1);
    fr[0] = f0.x; fr[1] = f0.y; fr[2] = f0.z; fr[3] = f0.w; fr[4] = f1.x; fr[5] = f1.y; fr[6] = f1.z; fr[7] = f1.w;
  }
  const int64_t row_base = static_cast<int64_t>(token) * kHeads + head0;
  uint4 xin[kIters];
#pragma unroll
  for (int it = 0; it < kIters; ++it) xin[it] = __ldg(p.input + (row_base + 2 * it + sub) * 16 + hl);
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int64_t row = row_base + 2 * it + sub;
    const uint32_t xx[4] = {xin[it].x, xin[it].y, xin[it].z, xin[it].w};
    float d[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      d[2 * i] = bf16_lo(xx[i]);
      d[2 * i + 1] = bf16_hi(xx[i]);
    }
    if (hl >= 8) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float tx = d[2 * i], ty = d[2 * i + 1], fx = fr[2 * i], fy = fr[2 * i + 1];
        const float yy = __fmul_rn(ty, fy);
        const float yx = __fmul_rn(ty, fx);
        const uint32_t b = cvt_bf16x2(__fmaf_rn(tx, fx, -yy), __fmaf_rn(tx, fy, yx));
        d[2 * i] = bf16_lo(b);
        d[2 * i + 1] = bf16_hi(b);
      }
    }
    float amax = fabsf(d[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) amax = fmaxf(amax, fabsf(d[i]));
    const auto sc = fp4::block_scale(block_max4(amax));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const auto qv = fp4::fake_quant_x2(float2{d[2 * i], d[2 * i + 1]}, sc.x, sc.y);
      d[2 * i] = qv.x;
      d[2 * i + 1] = qv.y;
    }
    amax = fabsf(d[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) amax = fmaxf(amax, fabsf(d[i]));
    const uint32_t exponent = index_pack_exponent(block_max4(amax));
    const float inv = sglang::deepseek_v4::fp8::inv_scale_ue8m0(static_cast<int32_t>(exponent));
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const auto code = __nv_cvt_float2_to_fp4x2(float2{__fmul_rn(d[2 * i], inv), __fmul_rn(d[2 * i + 1], inv)},
                                                 __NV_E2M1, cudaRoundNearest);
      word |= static_cast<uint32_t>(code) << (8 * i);
    }
    p.payload[row * 16 + hl] = clear_negative_zero(word);
    const int half = lane & 16;
    const uint32_t e0 = __shfl_sync(0xffffffffu, exponent, half + 0);
    const uint32_t e1 = __shfl_sync(0xffffffffu, exponent, half + 4);
    const uint32_t e2 = __shfl_sync(0xffffffffu, exponent, half + 8);
    const uint32_t e3 = __shfl_sync(0xffffffffu, exponent, half + 12);
    if (hl == 0) {
      p.scale[row] = static_cast<int32_t>(e0 | (e1 << 8) | (e2 << 16) | (e3 << 24));
      const float wv = __fmul_rn(__uint_as_float(static_cast<uint32_t>(__ldg(p.head_weights + row)) << 16), p.weight_scale);
      p.weights_out[row] = bf16_lo(cvt_bf16x2(wv, 0.0f));
    }
  }
}

}

struct V4Idx {

  static void index_q(tvm::ffi::TensorView input, tvm::ffi::TensorView freqs_cis, tvm::ffi::TensorView positions,
                      tvm::ffi::TensorView payload, tvm::ffi::TensorView scale, tvm::ffi::TensorView head_weights,
                      tvm::ffi::TensorView weights_out, double weight_scale) {
    using namespace sglang;
    using namespace sglang::host;
    auto N = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"heads"};
    auto R = SymbolicSize{"num_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({N, H, 128}).with_dtype<bf16_t>().with_device(device_).verify(input);
    TensorMatcher({-1, 64}).with_dtype<fp32_t>().with_device(device_).verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device_).verify(positions);
    TensorMatcher({R, 64}).with_dtype<int8_t>().with_device(device_).verify(payload);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(scale);
    TensorMatcher({N, H}).with_dtype<bf16_t>().with_device(device_).verify(head_weights);
    TensorMatcher({N, H}).with_dtype<fp32_t>().with_device(device_).verify(weights_out);
    RuntimeCheck(R.unwrap() == N.unwrap() * H.unwrap(), "v4 IDXQ: payload rows must be tokens x heads");
    RuntimeCheck(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(freqs_cis.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(payload.data_ptr()) % 4 == 0, "v4 IDXQ: misaligned buffers");
    RuntimeCheck(R.unwrap() < (int64_t(1) << 31), "v4 IDXQ: too many rows");
    const auto num_rows = static_cast<uint32_t>(R.unwrap());
    if (num_rows == 0) return;
    const auto params = v4::IdxqParams{
        .input = static_cast<const uint4*>(input.data_ptr()),
        .freqs = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .payload = static_cast<uint32_t*>(payload.data_ptr()),
        .scale = static_cast<int32_t*>(scale.data_ptr()),
        .head_weights = static_cast<const uint16_t*>(head_weights.data_ptr()),
        .weights_out = static_cast<float*>(weights_out.data_ptr()),
        .weight_scale = static_cast<float>(weight_scale),
        .num_rows = num_rows,
        .heads = static_cast<uint32_t>(H.unwrap()),
    };
    RuntimeCheck(H.unwrap() == 32, "v4 IDXQ serves 32 heads");
    const auto tokens = static_cast<uint32_t>(N.unwrap());
    const unsigned blocks = static_cast<unsigned>((static_cast<uint64_t>(tokens) * 2 + 7) / 8);
    if (pos_dtype.is_type<int32_t>()) {
      LaunchKernel(blocks, 256, device_.unwrap())(v4::index_q_tok_kernel<16, int32_t>, params, tokens);
    } else {
      LaunchKernel(blocks, 256, device_.unwrap())(v4::index_q_tok_kernel<16, int64_t>, params, tokens);
    }
  }
};
"""
