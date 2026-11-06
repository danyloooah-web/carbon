"""king-v13s TAIL-NORM device text (tail_norm.py): the decode step's final seam post, mhc_combine and RMSNorm in one
launch, one 320-thread CTA a row.

- R' (the final mHC residual, returned as pre_hc_head): king-v6 SEAM-POST's arithmetic, line for line
  (hoist_pdl.SEAM_SOURCE): fma(post[i], x, comb[i] * r0), then fma(comb[4j + i], r_j, acc) in j order, RNE bf16; that
  is stock's _mhc_post_split_h_kernel (enable_fp_fusion=False, explicit tl.fma).
- h = mhc_combine(R', pre): Triton's _mhc_combine_kernel, acc = 0, then acc += pre[k] * float(R'[k]) for k = 0..3,
  RNE bf16 store. `comb_form` selects how that loop compiled: 0 = fused fma from +0 (Triton's default fp fusion),
  1 = unfused multiply then add from +0, 2 = first product unfused, then fma. The prepare-time proof picks the form
  that matches the stock kernel bit for bit.
- y = RMSNorm(h) (norm != 0): flashinfer 0.7.2's CuTe-DSL RMSNormKernel at H = 5120 (cluster_n 1, 128 threads, 2 rows a
  CTA, 64 threads a row, vec 8, 10 vector blocks): thread t of a row holds columns 512 v1 + 8 t + v0 (v0 0..7, v1 0..9),
  its partial is the ordered reduction of x^2 in that order from +0 (`red`: 0 = mul.rn then add.rn, 1 = fma), a
  butterfly over lanes 1, 2, 4, 8, 16, then the two warp sums of the row added once; mean = sum / 5120 (`div`: 0 =
  div.rn, 1 = div.full, 2 = div.approx, 3 = times the rounded reciprocal), rstd = rsqrt.approx(mean + eps),
  y = RNE bf16((h * rstd) * (w + 0)). Here
  warps 0-1 replay exactly those 64 threads' partials from the row's h in shared memory; every thread then normalizes
  its own columns. The proof picks (red, div) or leaves the norm to flashinfer.
Every arithmetic step uses an explicit-rounding intrinsic or PTX instruction, so no compiler flag can contract or
reorder it. `griddepcontrol.launch_dependents` is issued after the CTA's last global store (fixer, fatal review F2: the
first build issued it right after the loads, so a PDL successor reading R', h or y before its own griddepcontrol.wait
could have seen partial rows; king_v10b's R' and h came behind a non-PDL Triton combine). That gives up nothing
measured: in the first build's GPU trace the next kernel in the graph (the LM head's cuBLAS GEMM) starts ~3 us after
this kernel ends, i.e. it is not launched early.
"""

NAME = "dec_tail_norm.cuh"
ENTRY = "tail_norm::TailNorm<320>::run"
SOURCE = r"""#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {
namespace tail_norm {

constexpr int kHidden = 5120;
constexpr int kNormThreadsPerRow = 64;  // flashinfer RMSNormKernel, H 5120
constexpr int kNormVecBlocks = 10;

__device__ __forceinline__ float combine4(const float (&pre)[4], const float (&u)[4], int form) {
  float acc = 0.0f;
  if (form == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = __fmaf_rn(pre[i], u[i], acc);
  } else if (form == 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = __fadd_rn(acc, __fmul_rn(pre[i], u[i]));
  } else {
    acc = __fmul_rn(pre[0], u[0]);
#pragma unroll
    for (int i = 1; i < 4; ++i) acc = __fmaf_rn(pre[i], u[i], acc);
  }
  return acc;
}

__device__ __forceinline__ float div_by_hidden(float sum, int form) {
  float out;
  if (form == 0) {
    out = __fdiv_rn(sum, static_cast<float>(kHidden));
  } else if (form == 1) {
    asm("div.full.f32 %0, %1, %2;" : "=f"(out) : "f"(sum), "f"(static_cast<float>(kHidden)));
  } else if (form == 2) {
    asm("div.approx.f32 %0, %1, %2;" : "=f"(out) : "f"(sum), "f"(static_cast<float>(kHidden)));
  } else {
    out = __fmul_rn(sum, __frcp_rn(static_cast<float>(kHidden)));
  }
  return out;
}

__device__ __forceinline__ float rsqrt_approx(float v) {
  float out;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(out) : "f"(v));
  return out;
}

template <int Threads>
__global__ __launch_bounds__(Threads) void tail_norm_kernel(const bf16_t* x_in, const bf16_t* r_in,
                                                            const float* post_in, const float* comb_in,
                                                            const float* pre_in, const bf16_t* w_in, float eps,
                                                            int comb_form, int red, int div, int norm,
                                                            bf16_t* r_out, bf16_t* h_out, bf16_t* y_out) {
  using namespace device;
  using V = AlignedVector<bf16x2_t, 4>;
  constexpr int kVectors = kHidden / 8;
  constexpr int kVPT = kVectors / Threads;
  static_assert(kVPT * Threads == kVectors && Threads % 32 == 0 && Threads >= kNormThreadsPerRow);
  static_assert(kNormThreadsPerRow * 8 * kNormVecBlocks == kHidden);
  __shared__ __align__(16) bf16_t s_h[kHidden];
  __shared__ float s_part[2];

  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;

  float comb[16], post[4], pre[4];
#pragma unroll
  for (int i = 0; i < 16; ++i) comb[i] = comb_in[row * 16 + i];
#pragma unroll
  for (int i = 0; i < 4; ++i) post[i] = post_in[row * 4 + i];
#pragma unroll
  for (int i = 0; i < 4; ++i) pre[i] = pre_in[row * 4 + i];

  V xh[kVPT], rh[kVPT][4];
#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    xh[j].load(x_in + row * kHidden, tid + j * Threads);
#pragma unroll
    for (int r = 0; r < 4; ++r) rh[j][r].load(r_in + row * 4 * kHidden + r * kHidden, tid + j * Threads);
  }
  V hv[kVPT];
#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    const int vid = tid + j * Threads;
    V x = xh[j], residual[4], upd[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) residual[r] = rh[j][r];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const auto xx = cast<fp32x2_t>(x[k]);
        const auto r0 = cast<fp32x2_t>(residual[0][k]);
        const auto r1 = cast<fp32x2_t>(residual[1][k]);
        const auto r2 = cast<fp32x2_t>(residual[2][k]);
        const auto r3 = cast<fp32x2_t>(residual[3][k]);
        float a = __fmaf_rn(post[i], xx.x, __fmul_rn(comb[i], r0.x));
        float b = __fmaf_rn(post[i], xx.y, __fmul_rn(comb[i], r0.y));
        a = __fmaf_rn(comb[4 + i], r1.x, a);
        b = __fmaf_rn(comb[4 + i], r1.y, b);
        a = __fmaf_rn(comb[8 + i], r2.x, a);
        b = __fmaf_rn(comb[8 + i], r2.y, b);
        a = __fmaf_rn(comb[12 + i], r3.x, a);
        b = __fmaf_rn(comb[12 + i], r3.y, b);
        upd[i][k] = cast<bf16x2_t>(fp32x2_t{a, b});
      }
      upd[i].store(r_out + row * 4 * kHidden + i * kHidden, vid);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float ua[4], ub[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const auto f = cast<fp32x2_t>(upd[i][k]);
        ua[i] = f.x;
        ub[i] = f.y;
      }
      hv[j][k] = cast<bf16x2_t>(fp32x2_t{combine4(pre, ua, comb_form), combine4(pre, ub, comb_form)});
    }
    hv[j].store(h_out + row * kHidden, vid);
    if (norm) hv[j].store(s_h, vid);
  }
  if (!norm) {
    // Tier B: R' and h are this CTA's last global stores; only now may a PDL successor launch.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    return;
  }

  __syncthreads();
  if (tid < kNormThreadsPerRow) {
    float local = 0.0f;
#pragma unroll
    for (int v1 = 0; v1 < kNormVecBlocks; ++v1) {
      V c;
      c.load(s_h, v1 * kNormThreadsPerRow + tid);  // columns 512 v1 + 8 tid .. + 7
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const auto f = cast<fp32x2_t>(c[k]);
        if (red == 0) {
          local = __fadd_rn(local, __fmul_rn(f.x, f.x));
          local = __fadd_rn(local, __fmul_rn(f.y, f.y));
        } else {
          local = __fmaf_rn(f.x, f.x, local);
          local = __fmaf_rn(f.y, f.y, local);
        }
      }
    }
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) local = __fadd_rn(local, __shfl_xor_sync(0xffffffffu, local, offset));
    if ((tid & 31) == 0) s_part[tid >> 5] = local;
  }
  __syncthreads();
  const float sum = __fadd_rn(s_part[0], s_part[1]);
  const float rstd = rsqrt_approx(__fadd_rn(div_by_hidden(sum, div), eps));
#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    const int vid = tid + j * Threads;
    V wv, yv;
    wv.load(w_in, vid);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const auto hf = cast<fp32x2_t>(hv[j][k]);
      const auto wf = cast<fp32x2_t>(wv[k]);
      const float ya = __fmul_rn(__fmul_rn(hf.x, rstd), __fadd_rn(wf.x, 0.0f));
      const float yb = __fmul_rn(__fmul_rn(hf.y, rstd), __fadd_rn(wf.y, 0.0f));
      yv[k] = cast<bf16x2_t>(fp32x2_t{ya, yb});
    }
    yv.store(y_out + row * kHidden, vid);
  }
  // Tier A: R', h and y are stored; only now may a PDL successor launch.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

template <int Threads>
struct TailNorm {
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView residual, tvm::ffi::TensorView post,
                  tvm::ffi::TensorView comb, tvm::ffi::TensorView pre, tvm::ffi::TensorView weight,
                  tvm::ffi::TensorView r_out, tvm::ffi::TensorView h_out, tvm::ffi::TensorView y_out, float eps,
                  int64_t comb_form, int64_t red, int64_t div, int64_t norm) {
    using namespace host;
    auto m = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, kHidden}).with_dtype<bf16_t>().with_device(dev).verify(x).verify(h_out).verify(y_out);
    TensorMatcher({m, 4, kHidden}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(r_out);
    TensorMatcher({m, 4}).with_dtype<float>().with_device(dev).verify(post).verify(pre);
    TensorMatcher({m, 4, 4}).with_dtype<float>().with_device(dev).verify(comb);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    for (const auto* ptr : {x.data_ptr(), residual.data_ptr(), r_out.data_ptr(), h_out.data_ptr(), y_out.data_ptr(),
                            weight.data_ptr()}) {
      CHECK_HOST(reinterpret_cast<uintptr_t>(ptr) % 16 == 0) << "BF16 pointers must be 16-byte aligned";
    }
    CHECK_HOST(0 <= comb_form && comb_form <= 2 && 0 <= red && red <= 1 && 0 <= div && div <= 3 &&
               0 <= norm && norm <= 1) << "tail_norm: variant out of range";
    const int64_t rows = m.unwrap();
    if (rows == 0) return;
    LaunchKernel(dim3(rows), Threads, dev.unwrap())
        .launch(tail_norm_kernel<Threads>, static_cast<const bf16_t*>(x.data_ptr()),
                static_cast<const bf16_t*>(residual.data_ptr()), static_cast<const float*>(post.data_ptr()),
                static_cast<const float*>(comb.data_ptr()), static_cast<const float*>(pre.data_ptr()),
                static_cast<const bf16_t*>(weight.data_ptr()), eps, static_cast<int>(comb_form),
                static_cast<int>(red), static_cast<int>(div), static_cast<int>(norm),
                static_cast<bf16_t*>(r_out.data_ptr()), static_cast<bf16_t*>(h_out.data_ptr()),
                static_cast<bf16_t*>(y_out.data_ptr()));
  }
};

}  // namespace tail_norm
}  // namespace sglang
"""
