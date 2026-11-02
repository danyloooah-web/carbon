// EXM (dspd/exm.py; dspd-w33e, SHIPPED OFF until the GPU checks G-EXM-0..6 of struct/w33_exm/LANE_EXM.md): the exact
// split mHC boundary at the decode / target-verify boundaries. Shared constants and the six FMA-contraction pins.
//
// Provenance: every kernel of EXM is written fresh in this lane from DeepGEMM 0.2.3.post2's mega_mhc semantics as
// derived in struct/w33_exm/EXM_SPEC.md (sections 5-6: impls/ and epilogue/sm100_mega_mhc.cuh). DeepGEMM's
// own headers are INCLUDED, never copied, from the installed deep_gemm package's include directory (cute, cutlass,
// deep_gemm), for the helpers EXM must share with stock bit for bit: math::warp_reduce_sum / warp_reduce_max,
// math::fast_rcp, math::get_packed_bf16_amax, math::get_ue8m0_sf_exp / get_ue8m0_sf_inv,
// math::scale_bf16x2_into_fp8x4, ptx::accumulate_square, the cute UMMA / TMEM atoms and mma::sm100::make_umma_desc.
// The split-critical-kernel + side-stream-stats idea is the rivals' route R1 / R6 read as TEXT in our own summary
// (struct/decdev_1031/R_code.md section 1); no rival file was opened in this lane and no rival code is used.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/ptx/utils.cuh>

// FMA contraction pins (EXM_SPEC 4.4; FIX of struct/w33_exm/REVIEW_exm_exact.md F1 / F5): stock writes six plain-C
// `a * b + c` sites; whether each compiles to ONE FFMA (1) or to FMUL then FADD (0) is decided by DeepJIT's nvcc, not
// by the source. EXM never relies on its own compiler (every operation is an _rn intrinsic) and follows these pins.
// Defaults = what stock's SASS shows for all four arena instances <5120, {40, 27}, 148, shifted, bf16, {fp8, -}, 0>
// built with DeepJIT's flags by the local nvcc 13.0.88 (struct/w33_exm/review_cc/stock*.sass, [M] INDICATIVE):
//   K1 FFMA, K2 FMUL then FFMA, K3 FFMA, K4 FMUL then FADD (the multiply sits inside the lane < 2 branch,
//   epilogue:323-332, so nothing fuses it), K5 / K6 FMUL; SHFL; FADD (unfused).
// G-EXM-0 (struct/w33_exm/gpu/g_exm_bits.py sass) re-decides every pin on the box from the real cached cubins; the
// install self-test's K4-band rows and the Sinkhorn of every row then check the pins against the live stock kernel.
#ifndef EXM_K1
#define EXM_K1 1  // epilogue:43   hc_norm_sqr_sum * (1 / 20480) + hc_norm_eps
#endif
#ifndef EXM_K2
#define EXM_K2 1  // epilogue:53   hc_output * hc_rms_scale * scales[k] + bases[n]  (FMUL, then this pin)
#endif
#ifndef EXM_K3
#define EXM_K3 1  // epilogue:69   value * fast_rcp(row_sum) + sinkhorn_eps
#endif
#ifndef EXM_K4
#define EXM_K4 0  // epilogue:332  x1_sqr_sum * kInvHidden + eps
#endif
#ifndef EXM_K5
#define EXM_K5 0  // epilogue:81, math.cuh:334   row sum of value = v * rcp: (value + shfl_xor(value, 2)), then xor 1
#endif
#ifndef EXM_K6
#define EXM_K6 0  // epilogue:73,83  column sum of value = v * rcp: value + shfl_xor(value, 4), then xor 24
#endif

namespace sglang {
namespace exm {

inline constexpr uint32_t kHidden = 5120;     // hidden_size
inline constexpr uint32_t kRoutes = 4;        // hc_mult
inline constexpr uint32_t kOutputs = 24;      // kRoutes * (kRoutes + 2): pre | post | comb
inline constexpr uint32_t kBlockM = 64;       // stock's m-block (rows per UMMA tile)
inline constexpr uint32_t kBlockK = 64;       // hidden per k-block
inline constexpr uint32_t kKBlocks = kHidden / kBlockK;  // 80
inline constexpr uint32_t kMaxSplits = 40;    // EXM serves S in {40, 27} (148 SMs, <= 320 rows)
inline constexpr uint32_t kSFWords = kHidden / 128;      // 40 UE8M0 words per row
inline constexpr uint32_t kGemmPartial = kBlockM * kOutputs;  // floats per task (stock's Workspace)
// the two compile-time constants stock folds (EXM_SPEC 4.4): 1/5120 = 0x394ccccd, 1/20480 = 0x384ccccd
inline constexpr float kInvHidden = 1.0f / static_cast<float>(kHidden);
inline constexpr float kInvHcHidden = 1.0f / static_cast<float>(kRoutes * kHidden);

// split geometry of stock's kernel (impls:69-76): split s owns k-blocks [first, first + count)
SGL_DEVICE uint32_t split_first_kb(const uint32_t s, const uint32_t num_splits) {
  return s * (kKBlocks / num_splits) + min(s, kKBlocks % num_splits);
}
SGL_DEVICE uint32_t split_num_kb(const uint32_t s, const uint32_t num_splits) {
  return kKBlocks / num_splits + (s < kKBlocks % num_splits ? 1u : 0u);
}

SGL_DEVICE float k1_rms_arg(const float a, const float b, const float c) {
  if constexpr (EXM_K1) return __fmaf_rn(a, b, c);
  else return __fadd_rn(__fmul_rn(a, b), c);
}
SGL_DEVICE float k2_affine(const float out, const float rr, const float scale, const float base) {
  const float prod = __fmul_rn(out, rr);  // the left multiply of `out * rr * scale` is never fused
  if constexpr (EXM_K2) return __fmaf_rn(prod, scale, base);
  else return __fadd_rn(__fmul_rn(prod, scale), base);
}
SGL_DEVICE float k3_row_norm(const float v, const float rcp, const float eps) {
  if constexpr (EXM_K3) return __fmaf_rn(v, rcp, eps);
  else return __fadd_rn(__fmul_rn(v, rcp), eps);
}
SGL_DEVICE float k4_rms_arg(const float a, const float b, const float c) {
  if constexpr (EXM_K4) return __fmaf_rn(a, b, c);
  else return __fadd_rn(__fmul_rn(a, b), c);
}
// The Sinkhorn sums whose first addend is a product v = a * b (epilogue:81-83). Stock's warp_reduce_sum<4> is
// value = value + shfl_xor(value, 2), then + shfl_xor(., 1) (math.cuh:307,334-335); its get_col_sum is value +
// shfl_xor(value, 4), then + shfl_xor(., 24). A fused toolchain keeps v for the shuffle and folds the own product into
// the first add: fma(a, b, shfl(v)). `v` must be __fmul_rn(a, b) (rounded, never contracted).
SGL_DEVICE float k5_row_sum(const float a, const float b, const float v) {
  const float o = __shfl_xor_sync(0xffffffffu, v, 2);
  float s;
  if constexpr (EXM_K5) s = __fmaf_rn(a, b, o);
  else s = __fadd_rn(v, o);
  return __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
}
SGL_DEVICE float k6_col_sum(const float a, const float b, const float v) {
  const float o = __shfl_xor_sync(0xffffffffu, v, 4);
  float s;
  if constexpr (EXM_K6) s = __fmaf_rn(a, b, o);
  else s = __fadd_rn(v, o);
  return __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 24));
}

SGL_DEVICE nv_bfloat162 bf2_of(const uint32_t word) {
  return *reinterpret_cast<const nv_bfloat162*>(&word);
}
SGL_DEVICE uint32_t word_of(const nv_bfloat162 v) {
  return *reinterpret_cast<const uint32_t*>(&v);
}

SGL_DEVICE void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

}  // namespace exm
}  // namespace sglang
