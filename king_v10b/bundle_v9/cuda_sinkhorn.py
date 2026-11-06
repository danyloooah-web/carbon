
NAME = "hc_sinkhorn_reduce.cuh"

SOURCE = r"""#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace v4_hcsk {

constexpr int kThreads = 128;
constexpr int kMix = 24;
constexpr int kVec = kMix / 4;
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float ex2_approx(float a) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float sqrt_approx_ftz(float a) {
  float r;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float add_f(float a, float b) {
  float r;
  asm("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float sub_f(float a, float b) {
  float r;
  asm("sub.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float mul_f(float a, float b) {
  float r;
  asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float max_f(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float bfly(float v, int mask) { return __shfl_xor_sync(kFull, v, mask, 32); }

template <int kRows, int kSlices, bool kResidual>
__global__ void __launch_bounds__(kThreads) reduce_sinkhorn_kernel(
    const float* __restrict__ part_mix, const float* __restrict__ residual, const float* __restrict__ part_sq,
    const float* __restrict__ scale, const float* __restrict__ base, float* __restrict__ pre,
    float* __restrict__ post, float* __restrict__ comb, int m, float inv_k, float rms_eps, int slices, int iters,
    float eps) {
  constexpr int kLoads = (kRows * kSlices * kVec + kThreads - 1) / kThreads;
  constexpr int kSqLoads = (kRows * kSlices + kThreads - 1) / kThreads;
  __shared__ float4 s_mix[kRows][kSlices * kVec];
  __shared__ float s_sq[kRows][kSlices];
  __shared__ float s_sum[kRows][kMix + 1];
  const int row0 = blockIdx.x * kRows;
  const int rows = min(kRows, m - row0);
  const int t = threadIdx.x;
  const int n4 = slices * kVec;
  const int total = rows * n4;
  float4 v[kLoads];
  float4 w[kResidual ? kLoads : 1];
#pragma unroll
  for (int j = 0; j < kLoads; ++j) {
    const int i = t + j * kThreads;
    if (i < total) {
      const int rr = kRows == 1 ? 0 : i / n4;
      const int rest = i - rr * n4;
      const int s = rest / kVec;
      const int q = rest - s * kVec;
      const size_t at = (size_t(s) * m + row0 + rr) * kMix;
      v[j] = __ldcg(reinterpret_cast<const float4*>(part_mix + at) + q);
      if constexpr (kResidual) w[j] = __ldcg(reinterpret_cast<const float4*>(residual + at) + q);
    }
  }
  float sq[kSqLoads];
#pragma unroll
  for (int j = 0; j < kSqLoads; ++j) {
    const int i = t + j * kThreads;
    if (i < rows * slices) {
      const int rr = kRows == 1 ? 0 : i / slices;
      sq[j] = __ldcg(part_sq + size_t(i - rr * slices) * m + row0 + rr);
    }
  }
#pragma unroll
  for (int j = 0; j < kLoads; ++j) {
    const int i = t + j * kThreads;
    if (i < total) {
      const int rr = kRows == 1 ? 0 : i / n4;
      float4 x = v[j];
      if constexpr (kResidual) {
        x.x = add_f(x.x, w[j].x);
        x.y = add_f(x.y, w[j].y);
        x.z = add_f(x.z, w[j].z);
        x.w = add_f(x.w, w[j].w);
      }
      s_mix[rr][i - rr * n4] = x;
    }
  }
#pragma unroll
  for (int j = 0; j < kSqLoads; ++j) {
    const int i = t + j * kThreads;
    if (i < rows * slices) {
      const int rr = kRows == 1 ? 0 : i / slices;
      s_sq[rr][i - rr * slices] = sq[j];
    }
  }
  __syncthreads();
  if (t < rows * (kMix + 1)) {
    const int rr = kRows == 1 ? 0 : t / (kMix + 1);
    const int c = t - rr * (kMix + 1);
    float acc = 0.f;
    if (c < kMix) {
      const float* col = reinterpret_cast<const float*>(s_mix[rr]) + c;
#pragma unroll 8
      for (int s = 0; s < slices; ++s) acc = add_f(acc, col[s * kMix]);
    } else {
#pragma unroll 8
      for (int s = 0; s < slices; ++s) acc = add_f(acc, s_sq[rr][s]);
    }
    s_sum[rr][c] = acc;
  }
  __syncthreads();
  const int wr = t >> 5;
  if (wr >= rows) return;
  const int row = row0 + wr;
  const float* sum_r = s_sum[wr];
  const int lane = t & 31;
  const int j = lane & 3;
  const int r = (lane >> 2) & 3;
  const float log2e = __int_as_float(0x3FB8AA3B);
  const float neg_log2e = __int_as_float(0xBFB8AA3B);
  const float rsq = div_full(1.0f, sqrt_approx_ftz(fma_rn(inv_k, sum_r[kMix], rms_eps)));
  const float s0 = scale[0], s1 = scale[1], s2 = scale[2];

  float z = fma_rn(mul_f(sum_r[j], rsq), s0, base[j]);
  float sig = div_full(1.0f, add_f(ex2_approx(mul_f(z, neg_log2e)), 1.0f));
  const float pre_v = add_f(sig, eps);
  z = fma_rn(mul_f(sum_r[4 + j], rsq), s1, base[4 + j]);
  sig = div_full(1.0f, add_f(ex2_approx(mul_f(z, neg_log2e)), 1.0f));
  const float post_v = add_f(sig, sig);
  if (lane < 4) {
    pre[row * 4 + j] = pre_v;
    post[row * 4 + j] = post_v;
  }

  float c = fma_rn(mul_f(sum_r[8 + r * 4 + j], rsq), s2, base[8 + r * 4 + j]);
  float mx = max_f(c, bfly(c, 2));
  mx = max_f(mx, bfly(mx, 1));
  const float e = ex2_approx(mul_f(sub_f(c, mx), log2e));
  float sum = add_f(e, bfly(e, 2));
  sum = add_f(sum, bfly(sum, 1));
  c = add_f(div_full(e, sum), eps);
  sum = add_f(c, bfly(c, 8));
  sum = add_f(sum, bfly(sum, 4));
  c = div_full(c, add_f(sum, eps));
#pragma unroll 1
  for (int it = 1; it < iters; ++it) {
    sum = add_f(c, bfly(c, 2));
    sum = add_f(sum, bfly(sum, 1));
    c = div_full(c, add_f(sum, eps));
    sum = add_f(c, bfly(c, 8));
    sum = add_f(sum, bfly(sum, 4));
    c = div_full(c, add_f(sum, eps));
  }
  if (lane < 16) comb[row * 16 + r * 4 + j] = c;
}

}

template <int kSlices, bool kResidual>
struct HcskReduceSinkhorn {
  static void
  run(tvm::ffi::TensorView part_mix,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView part_sq,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView base,
      tvm::ffi::TensorView pre,
      tvm::ffi::TensorView post,
      tvm::ffi::TensorView comb,
      float inv_k,
      float rms_eps,
      int64_t iters,
      float eps) {
    using namespace sglang::host;
    using v4_hcsk::kMix;
    auto slices = SymbolicSize{"slices"};
    auto m = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({slices, m, kMix}).with_dtype<float>().with_device(dev).ensure_alignment(16).verify(part_mix);
    if constexpr (kResidual) {
      TensorMatcher({slices, m, kMix}).with_dtype<float>().with_device(dev).ensure_alignment(16).verify(residual);
    }
    TensorMatcher({slices, m}).with_dtype<float>().with_device(dev).verify(part_sq);
    TensorMatcher({3}).with_dtype<float>().with_device(dev).verify(scale);
    TensorMatcher({kMix}).with_dtype<float>().with_device(dev).verify(base);
    TensorMatcher({m, 4}).with_dtype<float>().with_device(dev).verify(pre).verify(post);
    TensorMatcher({m, 4, 4}).with_dtype<float>().with_device(dev).verify(comb);
    const int64_t rows = m.unwrap();
    const int64_t num_slices = slices.unwrap();
    RuntimeCheck(num_slices >= 1 && num_slices <= kSlices, "hcsk: slices must be 1..", kSlices, ", got ", num_slices);
    RuntimeCheck(rows <= (int64_t{1} << 20), "hcsk: rows must be <= 2^20, got ", rows);
    RuntimeCheck(iters >= 1, "hcsk: sinkhorn iters must be >= 1");
    if (rows == 0) return;
    const float* pm = static_cast<const float*>(part_mix.data_ptr());
    const float* rs = kResidual ? static_cast<const float*>(residual.data_ptr()) : pm;
    LaunchKernel(dim3(static_cast<unsigned>(rows)), v4_hcsk::kThreads, dev.unwrap())
        .launch(
            v4_hcsk::reduce_sinkhorn_kernel<1, kSlices, kResidual>,
            pm,
            rs,
            static_cast<const float*>(part_sq.data_ptr()),
            static_cast<const float*>(scale.data_ptr()),
            static_cast<const float*>(base.data_ptr()),
            static_cast<float*>(pre.data_ptr()),
            static_cast<float*>(post.data_ptr()),
            static_cast<float*>(comb.data_ptr()),
            static_cast<int>(rows),
            inv_k,
            rms_eps,
            static_cast<int>(num_slices),
            static_cast<int>(iters),
            eps);
  }
};
"""
