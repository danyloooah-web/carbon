NAME = "moe/moe_front.cuh"
SOURCE = r"""
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>
#include <sgl_kernel/mbarrier.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::dec_front {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::pack_fp8;

inline constexpr int kRouterWarps = 4;
inline constexpr int kQuantWarps = 4;
inline constexpr int kRouterThreads = kRouterWarps * 32;
inline constexpr int kThreads = (kRouterWarps + kQuantWarps) * 32;
inline constexpr int kPadGroupsPerThread = 2;
inline constexpr uint32_t kAll = 0xffffffffu;
inline constexpr uint32_t kNoPick = 0xffffffffu;

struct Params {
  const float* logits;
  const float* bias;
  const float* bias_alt;
  const void* input_ids;
  const void* num_live;
  const bf16_t* x;
  fp8_e4m3_t* buf_x;
  int32_t* buf_x_sf;
  int64_t* buf_topk_idx;
  float* buf_topk_weights;
  int32_t* shared_x_sf;
  int64_t logits_stride;
  int64_t ids_stride;
  int64_t alt_token;
  int64_t x_sf_stride;
  int64_t shared_col_stride;
  uint32_t shared_block_m;
  uint32_t rows;
  uint32_t pad_group_begin;
  uint32_t pad_group_end;
  bool ids64;
  bool live64;
  float scale;
  float renorm_eps;
};

__device__ __forceinline__ float stock_sqrtsoftplus(const float x) {
  float r;
  asm("{\n"
      ".reg .f32 t, j, jj, y, e2, s, e, u, m, v, f, rr, kf, p, q, res, alt;\n"
      ".reg .b32 jb, ub, kb, eb, mb, vb;\n"
      ".reg .pred big, ovf, zero, gt20;\n"
      "fma.rn.ftz.f32 t, %1, 0f3BBB989D, 0f3F000000;\n"
      "cvt.ftz.sat.f32.f32 t, t;\n"
      "fma.rm.ftz.f32 j, t, 0f437C0000, 0f4B400001;\n"
      "add.rn.f32 jj, j, 0fCB40007F;\n"
      "neg.f32 jj, jj;\n"
      "fma.rn.ftz.f32 y, %1, 0f3FB8AA3B, jj;\n"
      "fma.rn.ftz.f32 y, %1, 0f32A57060, y;\n"
      "ex2.approx.ftz.f32 e2, y;\n"
      "mov.b32 jb, j;\n"
      "shl.b32 jb, jb, 23;\n"
      "mov.b32 s, jb;\n"
      "mul.rn.f32 e, e2, s;\n"
      "add.rz.ftz.f32 u, e, 0f3F800000;\n"
      "mov.b32 ub, u;\n"
      "add.s32 kb, ub, -1061158912;\n"
      "and.b32 kb, kb, -8388608;\n"
      "mov.b32 eb, e;\n"
      "sub.s32 mb, eb, kb;\n"
      "mov.b32 m, mb;\n"
      "sub.s32 vb, 1082130432, kb;\n"
      "mov.b32 v, vb;\n"
      "fma.rn.ftz.f32 f, 0f3E800000, v, 0fBF800000;\n"
      "add.rn.f32 rr, f, m;\n"
      "cvt.rn.f32.s32 kf, kb;\n"
      "mul.rn.f32 kf, kf, 0f34000000;\n"
      "fma.rn.ftz.f32 p, 0fBD39BF78, rr, 0f3DD80012;\n"
      "fma.rn.ftz.f32 p, p, rr, 0fBE0778E0;\n"
      "fma.rn.ftz.f32 p, p, rr, 0f3E146475;\n"
      "fma.rn.ftz.f32 p, p, rr, 0fBE2A68DD;\n"
      "fma.rn.ftz.f32 p, p, rr, 0f3E4CAF9E;\n"
      "fma.rn.ftz.f32 p, p, rr, 0fBE800042;\n"
      "fma.rn.ftz.f32 p, p, rr, 0f3EAAAAE6;\n"
      "fma.rn.ftz.f32 p, p, rr, 0fBF000000;\n"
      "mul.rn.f32 q, rr, p;\n"
      "fma.rn.ftz.f32 q, q, rr, rr;\n"
      "fma.rn.ftz.f32 res, kf, 0f3F317218, q;\n"
      "setp.ge.u32 big, eb, 2139095040;\n"
      "setp.ge.s32 ovf, eb, -1082130431;\n"
      "and.pred ovf, ovf, big;\n"
      "fma.rn.ftz.f32 alt, e, 0f7F800000, 0f7F800000;\n"
      "selp.f32 res, alt, res, ovf;\n"
      "setp.eq.f32 zero, e, 0f00000000;\n"
      "and.pred zero, zero, big;\n"
      "selp.f32 res, 0f80000000, res, zero;\n"
      "setp.gt.f32 gt20, %1, 0f41A00000;\n"
      "selp.f32 res, %1, res, gt20;\n"
      "sqrt.approx.ftz.f32 %0, res;\n"
      "}"
      : "=f"(r)
      : "f"(x));
  return r;
}

__device__ __forceinline__ float add_rn(const float a, const float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float mul_rn(const float a, const float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float div_full(const float a, const float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ uint32_t rank_key(const float f) {
  uint32_t u = __float_as_uint(f);
  if (u == 0x80000000u) u = 0u;
  return u ^ ((u >> 31) != 0u ? 0xffffffffu : 0x80000000u);
}

template <int kExperts, int kTopK, int kHidden, bool kRenorm, bool kScale, bool kSharedSF>
__global__ void __launch_bounds__(kThreads) moe_front_kernel(const Params __grid_constant__ p) {
  using namespace device;
  static_assert(kExperts % kRouterThreads == 0, "each router lane scores a whole number of experts");
  static_assert(kTopK > 4 && kTopK <= 8, "the routed sum follows Triton's 8-slot tree");
  static_assert(kRouterWarps * kTopK <= 32, "one merge warp holds every candidate");
  static_assert(kHidden % (kQuantWarps * 8 * 32) == 0, "whole quant passes of 32 groups");
  constexpr int kPerThread = kExperts / kRouterThreads;
  constexpr int kCands = kRouterWarps * kTopK;
  constexpr int kGroups = kHidden / 32;
  constexpr int kPasses = kGroups / (kQuantWarps * 8);

  __shared__ float s_act[kExperts];
  __shared__ uint2 s_cand[kCands];
  __shared__ float s_pick_act[8];
  __shared__ int32_t s_pick_id[8];

  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;

  device::PDLWaitPrimary<true>();

  if (blockIdx.x >= p.rows) {

    const uint32_t first_slot = p.rows * kTopK;
    const uint32_t end_slot = p.pad_group_end * 4u;
    const uint32_t base =
        p.pad_group_begin + ((blockIdx.x - p.rows) * kThreads + threadIdx.x) * kPadGroupsPerThread;
#pragma unroll
    for (int j = 0; j < kPadGroupsPerThread; ++j) {
      const uint32_t g = base + j;
      if (g >= p.pad_group_end) break;
      const uint32_t s = g * 4u;
      if (s >= first_slot) {
        *reinterpret_cast<float4*>(p.buf_topk_weights + s) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *reinterpret_cast<longlong2*>(p.buf_topk_idx + s) = make_longlong2(-1, -1);
        *reinterpret_cast<longlong2*>(p.buf_topk_idx + s + 2) = make_longlong2(-1, -1);
      } else {
#pragma unroll
        for (uint32_t k = 0; k < 4u; ++k) {
          if (s + k >= first_slot && s + k < end_slot) {
            p.buf_topk_weights[s + k] = 0.0f;
            p.buf_topk_idx[s + k] = -1;
          }
        }
      }
    }
    device::PDLTriggerSecondary<true>();
    return;
  }

  const uint32_t row = blockIdx.x;
  if (warp >= static_cast<uint32_t>(kRouterWarps)) {

    using InputVec = AlignedVector<bf16x2_t, 4>;
    using OutputVec = AlignedVector<fp8x2_e4m3_t, 4>;
    const uint32_t q = warp - kRouterWarps;
    const auto token_in = p.x + static_cast<uint64_t>(row) * kHidden;
    const auto token_out = p.buf_x + static_cast<uint64_t>(row) * kHidden;
    InputVec in[kPasses];
#pragma unroll
    for (int i = 0; i < kPasses; ++i) in[i].load(token_in, (i * kQuantWarps * 8 + q * 8) * 4 + lane);
    uint32_t sf_word = 0;
#pragma unroll
    for (int i = 0; i < kPasses; ++i) {
      const uint32_t chunk = (i * kQuantWarps * 8 + q * 8) * 4 + lane;
      float local_max = 0.0f;
      float vals[8];
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k) {
        const auto [v0, v1] = cast<fp32x2_t>(in[i][k]);
        vals[2 * k + 0] = v0;
        vals[2 * k + 1] = v1;
        local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
      }
      local_max = warp::reduce_max<4>(local_max);
      const float absmax = fmaxf(local_max, 1e-10f);
      const float raw_scale = absmax / math::FP8_E4M3_MAX;
      const uint32_t ue8m0_exp = cast_to_ue8m0(raw_scale);
      const float inv_scale = __uint_as_float((127u + 127u - ue8m0_exp) << 23);
      OutputVec out;
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k) out[k] = pack_fp8(vals[2 * k + 0] * inv_scale, vals[2 * k + 1] * inv_scale);
      out.store(token_out, chunk);

      const uint32_t sf = ue8m0_exp & 0xffu;
      const uint32_t b = lane & 16u;
      const uint32_t word = __shfl_sync(kAll, sf, b) | (__shfl_sync(kAll, sf, b + 4) << 8) |
                            (__shfl_sync(kAll, sf, b + 8) << 16) | (__shfl_sync(kAll, sf, b + 12) << 24);
      if (lane == static_cast<uint32_t>(i) || lane == static_cast<uint32_t>(16 + i)) sf_word = word;
    }

    if (lane % 16u < static_cast<uint32_t>(kPasses)) {
      const uint32_t i = lane % 16u;
      const uint32_t col = (i * kQuantWarps * 8 + q * 8) / 4 + lane / 16u;
      p.buf_x_sf[static_cast<int64_t>(row) * p.x_sf_stride + col] = static_cast<int32_t>(sf_word);
      if constexpr (kSharedSF) {
        const uint32_t m = row % p.shared_block_m;
        const uint32_t shared_row = row / p.shared_block_m * ((p.shared_block_m + 127u) / 128u * 128u) +
                                    m / 128u * 128u + m % 32u * 4u + m % 128u / 32u;
        p.shared_x_sf[static_cast<int64_t>(shared_row) + static_cast<int64_t>(col) * p.shared_col_stride] =
            static_cast<int32_t>(sf_word);
      }
    }
    device::PDLTriggerSecondary<true>();
    return;
  }

  int64_t live_rows = p.rows;
  if (p.num_live != nullptr) {
    const int64_t n = p.live64 ? *static_cast<const int64_t*>(p.num_live)
                               : static_cast<int64_t>(*static_cast<const int32_t*>(p.num_live));
    live_rows = n < live_rows ? n : live_rows;
  }
  const bool live = static_cast<int64_t>(row) < live_rows;

  if (live) {
    const float* lrow = p.logits + static_cast<int64_t>(row) * p.logits_stride;
    const float* bias = p.bias;
    if (p.input_ids != nullptr) {
      const int64_t at = static_cast<int64_t>(row) * p.ids_stride;
      const int64_t token = p.ids64 ? static_cast<const int64_t*>(p.input_ids)[at]
                                    : static_cast<int64_t>(static_cast<const int32_t*>(p.input_ids)[at]);
      if (token == p.alt_token) bias = p.bias_alt;
    }
    uint32_t key[kPerThread];
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int e = static_cast<int>(threadIdx.x) + i * kRouterThreads;
      const float act = stock_sqrtsoftplus(lrow[e]);
      s_act[e] = act;
      float biased = add_rn(act, bias[e]);
      biased = biased == biased ? biased : __int_as_float(0x7f800000);
      key[i] = rank_key(biased);
    }

#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      uint32_t offer_key = 0;
      int offer = 0;
#pragma unroll
      for (int i = 0; i < kPerThread; ++i) {
        if (key[i] > offer_key) {
          offer_key = key[i];
          offer = i;
        }
      }
      const uint32_t offer_id = threadIdx.x + static_cast<uint32_t>(offer) * kRouterThreads;
      const uint32_t best = __reduce_max_sync(kAll, offer_key);
      const uint32_t win = __reduce_min_sync(kAll, offer_key == best ? offer_id : kNoPick);
      if (offer_id == win) {
#pragma unroll
        for (int i = 0; i < kPerThread; ++i) key[i] = offer == i ? 0u : key[i];
      }
      if (lane == static_cast<uint32_t>(k)) s_cand[warp * kTopK + k] = make_uint2(best, win);
    }
  }
  asm volatile("bar.sync 1, %0;" ::"n"(kRouterThreads) : "memory");

  if (warp != 0) {
    device::PDLTriggerSecondary<true>();
    return;
  }
  int32_t pick = -1;
  float w = 0.0f;
  if (live) {

    const uint2 mine = lane < static_cast<uint32_t>(kCands) ? s_cand[lane] : make_uint2(0u, kNoPick);
    uint32_t rank = 0;
#pragma unroll
    for (int j = 0; j < kCands; ++j) {
      const uint2 other = s_cand[j];
      rank += (other.x > mine.x || (other.x == mine.x && other.y < mine.y)) ? 1u : 0u;
    }
    if (lane < static_cast<uint32_t>(kCands) && rank < static_cast<uint32_t>(kTopK)) {
      s_pick_id[rank] = static_cast<int32_t>(mine.y);
      s_pick_act[rank] = s_act[mine.y];
    }
    __syncwarp();

    float slot[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) slot[k] = k < kTopK ? s_pick_act[k] : 0.0f;
    const float even = add_rn(add_rn(slot[0], slot[2]), add_rn(slot[1], slot[3]));
    const float odd = add_rn(add_rn(slot[4], slot[6]), add_rn(slot[5], slot[7]));
    const float sum = add_rn(even, odd);
    if (lane < static_cast<uint32_t>(kTopK)) {
      pick = s_pick_id[lane];
      w = s_pick_act[lane];
    }
    if constexpr (kRenorm) {
      const float norm = p.renorm_eps > 0.0f ? add_rn(sum, p.renorm_eps) : (sum > 0.0f ? sum : 1.0f);
      w = div_full(w, norm);
    }
  }
  if constexpr (kScale) w = mul_rn(w, p.scale);
  if (lane < static_cast<uint32_t>(kTopK)) {
    const int64_t at = static_cast<int64_t>(row) * kTopK + lane;
    p.buf_topk_idx[at] = static_cast<int64_t>(pick);
    p.buf_topk_weights[at] = w;
  }
  device::PDLTriggerSecondary<true>();
}

template <int kExperts, int kTopK, int kHidden, bool kRenorm, bool kScale, bool kSharedSF>
struct MegaFront {
  static void run(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::Optional<tvm::ffi::TensorView> bias_alt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> input_ids,
      const int64_t alt_token,
      const tvm::ffi::Optional<tvm::ffi::TensorView> num_live,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView buf_x,
      const tvm::ffi::TensorView buf_x_sf,
      const tvm::ffi::TensorView buf_topk_idx,
      const tvm::ffi::TensorView buf_topk_weights,
      const tvm::ffi::Optional<tvm::ffi::TensorView> shared_x_sf,
      const int64_t shared_block_m,
      const double scale,
      const double renorm_eps,
      const bool use_pdl) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto P = SymbolicSize{"padded_max"};
    auto S = SymbolicSize{"logits_stride"};
    auto T = SymbolicSize{"input_ids_stride"};
    auto X = SymbolicSize{"x_sf_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, kExperts}).with_strides({S, 1}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({kExperts}).with_strides({1}).with_dtype<float>().with_device(device_).verify(bias);
    TensorMatcher({R, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(x);
    TensorMatcher({P, kHidden}).with_dtype<int8_t, fp8_e4m3_t>().with_device(device_).verify(buf_x);
    TensorMatcher({P, kHidden / 128}).with_strides({X, 1}).with_dtype<int32_t>().with_device(device_).verify(buf_x_sf);
    TensorMatcher({P, kTopK}).with_dtype<int64_t>().with_device(device_).verify(buf_topk_idx);
    TensorMatcher({P, kTopK}).with_dtype<float>().with_device(device_).verify(buf_topk_weights);
    const auto rows = R.unwrap();
    const auto padded = P.unwrap();
    RuntimeCheck(rows > 0 && rows <= padded, "moe_front: 0 < rows <= padded_max");
    RuntimeCheck(padded * kTopK % 4 == 0 && padded * kTopK <= 0x7fffffff, "moe_front: P * top_k slots");
    RuntimeCheck(reinterpret_cast<uintptr_t>(buf_topk_idx.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(buf_topk_weights.data_ptr()) % 16 == 0,
                 "moe_front: top-k buffers are 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(buf_x.data_ptr()) % 16 == 0,
                 "moe_front: x and buf.x are 16-byte aligned");
    RuntimeCheck(X.unwrap() >= kHidden / 128 && reinterpret_cast<uintptr_t>(buf_x_sf.data_ptr()) % 4 == 0,
                 "moe_front: buf.x_sf rows");
    RuntimeCheck(shared_x_sf.has_value() == kSharedSF, "moe_front: shared_x_sf must be given iff the module writes it");

    auto params = Params{
        .logits = static_cast<const float*>(logits.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .bias_alt = nullptr,
        .input_ids = nullptr,
        .num_live = nullptr,
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .buf_x = static_cast<fp8_e4m3_t*>(buf_x.data_ptr()),
        .buf_x_sf = static_cast<int32_t*>(buf_x_sf.data_ptr()),
        .buf_topk_idx = static_cast<int64_t*>(buf_topk_idx.data_ptr()),
        .buf_topk_weights = static_cast<float*>(buf_topk_weights.data_ptr()),
        .shared_x_sf = nullptr,
        .logits_stride = S.unwrap(),
        .ids_stride = 0,
        .alt_token = alt_token,
        .x_sf_stride = X.unwrap(),
        .shared_col_stride = 0,
        .shared_block_m = 1,
        .rows = static_cast<uint32_t>(rows),
        .pad_group_begin = static_cast<uint32_t>(rows * kTopK / 4),
        .pad_group_end = static_cast<uint32_t>(padded * kTopK / 4),
        .ids64 = false,
        .live64 = false,
        .scale = static_cast<float>(scale),
        .renorm_eps = static_cast<float>(renorm_eps),
    };
    if constexpr (kSharedSF) {
      auto SR = SymbolicSize{"shared_sf_rows"};
      auto SC = SymbolicSize{"shared_sf_col_stride"};
      const auto sf = shared_x_sf.value();
      TensorMatcher({SR, kHidden / 128}).with_strides({1, SC}).with_dtype<int32_t>().with_device(device_).verify(sf);
      RuntimeCheck(shared_block_m > 0, "moe_front: shared_block_m must be positive");
      const auto need = (rows + shared_block_m - 1) / shared_block_m * ((shared_block_m + 127) / 128 * 128);
      RuntimeCheck(SR.unwrap() >= need, "moe_front: shared scale buffer is too small for shared_block_m");
      RuntimeCheck(SC.unwrap() >= SR.unwrap(), "moe_front: shared scale columns must not overlap");
      params.shared_x_sf = static_cast<int32_t*>(sf.data_ptr());
      params.shared_col_stride = SC.unwrap();
      params.shared_block_m = static_cast<uint32_t>(shared_block_m);
    }
    RuntimeCheck(!input_ids.has_value() || bias_alt.has_value(), "moe_front: input_ids needs bias_alt");
    if (input_ids.has_value()) {
      TensorMatcher({kExperts}).with_strides({1}).with_dtype<float>().with_device(device_).verify(bias_alt.value());
      TensorMatcher({R}).with_strides({T}).with_dtype<int32_t, int64_t>().with_device(device_).verify(input_ids.value());
      params.bias_alt = static_cast<const float*>(bias_alt.value().data_ptr());
      params.input_ids = input_ids.value().data_ptr();
      params.ids_stride = T.unwrap();
      params.ids64 = input_ids.value().dtype().bits == 64;
    }
    if (num_live.has_value()) {
      const auto& live = num_live.value();
      const auto dtype = live.dtype();
      RuntimeCheck(dtype.code == kDLInt && (dtype.bits == 32 || dtype.bits == 64),
                   "moe_front: num_token_non_padded is an int32 or int64 scalar");
      params.num_live = live.data_ptr();
      params.live64 = dtype.bits == 64;
    }
    const uint32_t pad_groups = params.pad_group_end - params.pad_group_begin;
    const uint32_t per_cta = kThreads * kPadGroupsPerThread;
    const uint32_t pad_ctas = (pad_groups + per_cta - 1) / per_cta;
    LaunchKernel(static_cast<uint32_t>(rows) + pad_ctas, kThreads, device_.unwrap())
        .enable_pdl(use_pdl)(moe_front_kernel<kExperts, kTopK, kHidden, kRenorm, kScale, kSharedSF>, params);
  }
};

inline constexpr int kV2Warps = 16;
inline constexpr int kV2Stages = 1;
inline constexpr uint32_t kV2PrefixMask = 0xfffffe00u;

template <int kExperts, int kHidden, int kWarps, int kStages>
struct V2Smem {
  static constexpr uint32_t kXBytes = kHidden * 2;
  static constexpr uint32_t kLBytes = kExperts * 4;
  static constexpr uint32_t kGroups = kHidden / 32;
  static constexpr size_t kStageX = 0;
  static constexpr size_t kStageL = kStageX + size_t(kWarps) * kStages * kXBytes;
  static constexpr size_t kAct = kStageL + size_t(kWarps) * kStages * kLBytes;
  static constexpr size_t kBias = kAct + size_t(kWarps) * kLBytes;
  static constexpr size_t kSf = kBias + 2 * size_t(kLBytes);
  static constexpr size_t kBars = (kSf + size_t(kWarps) * kGroups + 7) / 8 * 8;
  static constexpr size_t kBytes = kBars + size_t(kWarps) * kStages * 2 * sizeof(uint64_t);
};

__device__ __forceinline__ void v2_bulk_load(void* dst, const void* src, const uint32_t bytes, uint64_t* bar) {
  device::ptx::mbar_arrive_expect_tx(bar, bytes);
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   device::ptx::to_shared(dst)),
               "l"(src), "r"(bytes), "r"(device::ptx::to_shared(bar))
               : "memory");
}

__device__ __forceinline__ int64_t v2_token(const Params& p, const uint32_t row) {
  const int64_t at = static_cast<int64_t>(row) * p.ids_stride;
  return p.ids64 ? static_cast<const int64_t*>(p.input_ids)[at]
                 : static_cast<int64_t>(static_cast<const int32_t*>(p.input_ids)[at]);
}

template <int kExperts, int kTopK, int kHidden, bool kRenorm, bool kScale, bool kSharedSF, int kWarps, int kStages>
__global__ void __launch_bounds__(kWarps * 32, 1) moe_front_v2_kernel(const Params __grid_constant__ p) {
  using namespace device;
  using L = V2Smem<kExperts, kHidden, kWarps, kStages>;
  static_assert(kExperts == 12 * 32, "12 experts a lane, read as three float4");
  static_assert(kExperts <= 512, "an expert id fits the 9-bit low field of the selection word");
  static_assert(kTopK == 6, "7 selection rounds: the top-6 and the boundary check");
  static_assert(kHidden % 512 == 0, "whole quant passes of 16 groups");
  constexpr int kPerLane = kExperts / 32;
  constexpr int kGroups = kHidden / 32;
  constexpr int kPasses = kGroups / 16;

  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  uint8_t* const wx = smem + L::kStageX + size_t(warp) * kStages * L::kXBytes;
  float* const wl = reinterpret_cast<float*>(smem + L::kStageL + size_t(warp) * kStages * L::kLBytes);
  float* const wact = reinterpret_cast<float*>(smem + L::kAct + size_t(warp) * L::kLBytes);
  float* const s_bias = reinterpret_cast<float*>(smem + L::kBias);
  uint8_t* const wsf = smem + L::kSf + size_t(warp) * kGroups;
  uint64_t* const wbar = reinterpret_cast<uint64_t*>(smem + L::kBars) + size_t(warp) * kStages * 2;

  device::PDLWaitPrimary<true>();
  for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(kExperts); i += blockDim.x) {
    s_bias[i] = p.bias[i];
    s_bias[kExperts + i] = p.bias_alt != nullptr ? p.bias_alt[i] : 0.0f;
  }
  if (threadIdx.x == 0) {
    uint64_t* const bars = reinterpret_cast<uint64_t*>(smem + L::kBars);
    for (int i = 0; i < kWarps * kStages * 2; ++i) ptx::mbar_init(bars + i, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const uint32_t rows = p.rows;
  const uint32_t gw = blockIdx.x * kWarps + warp;
  const uint32_t tw = gridDim.x * kWarps;
  int64_t live_rows = rows;
  if (p.num_live != nullptr) {
    const int64_t n = p.live64 ? *static_cast<const int64_t*>(p.num_live)
                               : static_cast<int64_t>(*static_cast<const int32_t*>(p.num_live));
    live_rows = n < live_rows ? n : live_rows;
  }

  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      const uint32_t row = gw + s * tw;
      if (row < rows) {
        v2_bulk_load(wx + s * L::kXBytes, p.x + static_cast<uint64_t>(row) * kHidden, L::kXBytes, wbar + 2 * s);
        v2_bulk_load(wl + s * kExperts, p.logits + static_cast<int64_t>(row) * p.logits_stride, L::kLBytes,
                     wbar + 2 * s + 1);
      }
    }
  }
  int64_t tok[kStages];
#pragma unroll
  for (int s = 0; s < kStages; ++s) {
    const uint32_t row = gw + s * tw;
    tok[s] = (p.input_ids != nullptr && row < rows) ? v2_token(p, row) : 0;
  }

  {
    const uint32_t first_slot = rows * kTopK;
    const uint32_t end_slot = p.pad_group_end * 4u;
    for (uint32_t g = p.pad_group_begin + gw * 32u + lane; g < p.pad_group_end; g += tw * 32u) {
      const uint32_t s = g * 4u;
      if (s >= first_slot) {
        *reinterpret_cast<float4*>(p.buf_topk_weights + s) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *reinterpret_cast<longlong2*>(p.buf_topk_idx + s) = make_longlong2(-1, -1);
        *reinterpret_cast<longlong2*>(p.buf_topk_idx + s + 2) = make_longlong2(-1, -1);
      } else {
#pragma unroll
        for (uint32_t k = 0; k < 4u; ++k) {
          if (s + k >= first_slot && s + k < end_slot) {
            p.buf_topk_weights[s + k] = 0.0f;
            p.buf_topk_idx[s + k] = -1;
          }
        }
      }
    }
  }

  uint32_t low[kPerLane];
#pragma unroll
  for (int n = 0; n < kPerLane; ++n) low[n] = 511u - (128u * (n / 4) + 4u * lane + (n % 4));
  const uint32_t rot = (lane >> 2) & 1u;

  uint32_t i = 0;
#pragma unroll 1
  for (uint32_t row = gw; row < rows; row += tw, ++i) {
    const uint32_t s = i % kStages;
    const uint32_t parity = (i / kStages) & 1u;
    const uint32_t next = row + kStages * tw;

    ptx::mbar_wait_parity(wbar + 2 * s, parity);
    {
      const uint8_t* const xs = wx + s * L::kXBytes;
      fp8_e4m3_t* const token_out = p.buf_x + static_cast<uint64_t>(row) * kHidden;
#pragma unroll 2
      for (int ps = 0; ps < kPasses; ++ps) {
        const uint8_t* const src = xs + ps * 1024 + lane * 32;
        const uint4 in0 = *reinterpret_cast<const uint4*>(src + 16 * rot);
        const uint4 in1 = *reinterpret_cast<const uint4*>(src + 16 * (rot ^ 1u));
        const uint32_t w[8] = {in0.x, in0.y, in0.z, in0.w, in1.x, in1.y, in1.z, in1.w};
        float local_max = 0.0f;
        fp32x2_t vals[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {

          const float v0 = __uint_as_float(w[k] << 16);
          const float v1 = __uint_as_float(w[k] & 0xffff0000u);
          vals[k] = make_float2(v0, v1);
          local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
        }
        local_max = warp::reduce_max<2>(local_max);
        const float absmax = fmaxf(local_max, 1e-10f);
        const float raw_scale = absmax / math::FP8_E4M3_MAX;
        const uint32_t ue8m0_exp = cast_to_ue8m0(raw_scale);
        const float inv_scale = __uint_as_float((127u + 127u - ue8m0_exp) << 23);
        const fp32x2_t inv2 = make_float2(inv_scale, inv_scale);
        AlignedVector<fp8x2_e4m3_t, 4> o[2];
#pragma unroll
        for (int k = 0; k < 8; k += 2) {

          const fp32x2_t p0 = __fmul2_rn(vals[k], inv2);
          const fp32x2_t p1 = __fmul2_rn(vals[k + 1], inv2);
          o[k / 4][k % 4] = pack_fp8(p0.x, p0.y);
          o[k / 4][k % 4 + 1] = pack_fp8(p1.x, p1.y);
        }

        const uint2 a = *reinterpret_cast<const uint2*>(o[0].data());
        const uint2 b = *reinterpret_cast<const uint2*>(o[1].data());
        const uint4 out = rot ? make_uint4(b.x, b.y, a.x, a.y) : make_uint4(a.x, a.y, b.x, b.y);
        *reinterpret_cast<uint4*>(token_out + ps * 512 + lane * 16) = out;
        wsf[ps * 16 + (lane >> 1)] = static_cast<uint8_t>(ue8m0_exp);
      }
    }
    __syncwarp();
    if (lane == 0 && next < rows) {
      v2_bulk_load(wx + s * L::kXBytes, p.x + static_cast<uint64_t>(next) * kHidden, L::kXBytes, wbar + 2 * s);
    }
    {

      const int32_t* const words = reinterpret_cast<const int32_t*>(wsf);
      uint32_t shared_row = 0;
      if constexpr (kSharedSF) {
        const uint32_t m = row % p.shared_block_m;
        shared_row = row / p.shared_block_m * ((p.shared_block_m + 127u) / 128u * 128u) + m / 128u * 128u +
                     m % 32u * 4u + m % 128u / 32u;
      }
#pragma unroll
      for (uint32_t col = lane; col < static_cast<uint32_t>(kGroups / 4); col += 32u) {
        const int32_t word = words[col];
        p.buf_x_sf[static_cast<int64_t>(row) * p.x_sf_stride + col] = word;
        if constexpr (kSharedSF) {
          p.shared_x_sf[static_cast<int64_t>(shared_row) + static_cast<int64_t>(col) * p.shared_col_stride] = word;
        }
      }
    }

    ptx::mbar_wait_parity(wbar + 2 * s + 1, parity);
    const bool live = static_cast<int64_t>(row) < live_rows;
    int32_t pick = -1;
    float w = 0.0f;
    if (live) {
      const float* const lrow = wl + s * kExperts;
      const float* const bias = s_bias + ((p.input_ids != nullptr && tok[0] == p.alt_token) ? kExperts : 0);
      uint32_t key[kPerLane];
      uint32_t word[kPerLane];
#pragma unroll
      for (int q = 0; q < kPerLane / 4; ++q) {
        const float4 l4 = *reinterpret_cast<const float4*>(lrow + 128 * q + 4 * lane);
        const float4 b4 = *reinterpret_cast<const float4*>(bias + 128 * q + 4 * lane);
        const float lv[4] = {l4.x, l4.y, l4.z, l4.w};
        const float bv[4] = {b4.x, b4.y, b4.z, b4.w};
        float av[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int n = 4 * q + j;
          const float act = stock_sqrtsoftplus(lv[j]);
          av[j] = act;
          float biased = add_rn(act, bv[j]);
          biased = biased == biased ? biased : __int_as_float(0x7f800000);
          key[n] = rank_key(biased);
          word[n] = (key[n] & kV2PrefixMask) | low[n];
        }
        *reinterpret_cast<float4*>(wact + 128 * q + 4 * lane) = make_float4(av[0], av[1], av[2], av[3]);
      }

      uint32_t c1 = word[0], c2 = 0u, c3 = 0u;
#pragma unroll
      for (int n = 1; n < kPerLane; ++n) {
        const uint32_t t1 = umin(c1, word[n]);
        c1 = umax(c1, word[n]);
        const uint32_t t2 = umin(c2, t1);
        c2 = umax(c2, t1);
        c3 = umax(c3, t2);
      }
      uint32_t prev = 0u, mine = 0u;
      bool exact = false;
#pragma unroll
      for (int r = 0; r <= kTopK; ++r) {
        const uint32_t best = __reduce_max_sync(kAll, c1);
        if (r > 0) exact = exact || ((prev ^ best) & kV2PrefixMask) == 0u;
        prev = best;
        if (lane == static_cast<uint32_t>(r)) mine = best;
        if (c1 == best) {
          c1 = c2;
          c2 = c3;
          c3 = 0u;
        }
        if (r == kTopK - 1) exact = exact || __any_sync(kAll, c1 == 0u);
      }
      int32_t mine_id = static_cast<int32_t>(511u - (mine & 511u));
      if (exact) {

        uint32_t taken = 0u;
#pragma unroll 1
        for (uint32_t r = 0; r < static_cast<uint32_t>(kTopK); ++r) {
          uint32_t bk = 0u, bn = 0u;
#pragma unroll
          for (int n = 0; n < kPerLane; ++n) {
            const bool better = ((taken >> n) & 1u) == 0u && key[n] > bk;
            bk = better ? key[n] : bk;
            bn = better ? static_cast<uint32_t>(n) : bn;
          }
          const uint32_t bid = 128u * (bn / 4u) + 4u * lane + bn % 4u;
          const uint32_t best = __reduce_max_sync(kAll, bk);
          const uint32_t win = __reduce_min_sync(kAll, bk == best ? bid : kNoPick);
          if (bid == win) taken |= 1u << bn;
          if (lane == r) mine_id = static_cast<int32_t>(win);
        }
      }
      __syncwarp();
      const float a = lane < static_cast<uint32_t>(kTopK) ? wact[mine_id] : 0.0f;

      float slot[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) slot[k] = k < kTopK ? __shfl_sync(kAll, a, k) : 0.0f;
      const float even = add_rn(add_rn(slot[0], slot[2]), add_rn(slot[1], slot[3]));
      const float odd = add_rn(add_rn(slot[4], slot[6]), add_rn(slot[5], slot[7]));
      const float sum = add_rn(even, odd);
      if (lane < static_cast<uint32_t>(kTopK)) {
        pick = mine_id;
        w = a;
      }
      if constexpr (kRenorm) {
        const float norm = p.renorm_eps > 0.0f ? add_rn(sum, p.renorm_eps) : (sum > 0.0f ? sum : 1.0f);
        w = div_full(w, norm);
      }
    }
    if constexpr (kScale) w = mul_rn(w, p.scale);
    if (lane < static_cast<uint32_t>(kTopK)) {
      const int64_t at = static_cast<int64_t>(row) * kTopK + lane;
      p.buf_topk_idx[at] = static_cast<int64_t>(pick);
      p.buf_topk_weights[at] = w;
    }
    __syncwarp();
    if (lane == 0 && next < rows) {
      v2_bulk_load(wl + s * kExperts, p.logits + static_cast<int64_t>(next) * p.logits_stride, L::kLBytes,
                   wbar + 2 * s + 1);
    }
#pragma unroll
    for (int k = 0; k + 1 < kStages; ++k) tok[k] = tok[k + 1];
    tok[kStages - 1] = (p.input_ids != nullptr && next < rows) ? v2_token(p, next) : 0;
  }
}

template <int kExperts, int kTopK, int kHidden, bool kRenorm, bool kScale, bool kSharedSF, int kWarps = kV2Warps,
          int kStages = kV2Stages>
struct MegaFrontV2 {
  static constexpr auto kernel =
      moe_front_v2_kernel<kExperts, kTopK, kHidden, kRenorm, kScale, kSharedSF, kWarps, kStages>;
  static constexpr size_t kSmemBytes = V2Smem<kExperts, kHidden, kWarps, kStages>::kBytes;

  static void run(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::Optional<tvm::ffi::TensorView> bias_alt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> input_ids,
      const int64_t alt_token,
      const tvm::ffi::Optional<tvm::ffi::TensorView> num_live,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView buf_x,
      const tvm::ffi::TensorView buf_x_sf,
      const tvm::ffi::TensorView buf_topk_idx,
      const tvm::ffi::TensorView buf_topk_weights,
      const tvm::ffi::Optional<tvm::ffi::TensorView> shared_x_sf,
      const int64_t shared_block_m,
      const double scale,
      const double renorm_eps,
      const bool use_pdl) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto P = SymbolicSize{"padded_max"};
    auto T = SymbolicSize{"input_ids_stride"};
    auto X = SymbolicSize{"x_sf_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({R, kExperts}).with_strides({kExperts, 1}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({kExperts}).with_strides({1}).with_dtype<float>().with_device(device_).verify(bias);
    TensorMatcher({R, kHidden}).with_strides({kHidden, 1}).with_dtype<bf16_t>().with_device(device_).verify(x);
    TensorMatcher({P, kHidden}).with_dtype<int8_t, fp8_e4m3_t>().with_device(device_).verify(buf_x);
    TensorMatcher({P, kHidden / 128}).with_strides({X, 1}).with_dtype<int32_t>().with_device(device_).verify(buf_x_sf);
    TensorMatcher({P, kTopK}).with_dtype<int64_t>().with_device(device_).verify(buf_topk_idx);
    TensorMatcher({P, kTopK}).with_dtype<float>().with_device(device_).verify(buf_topk_weights);
    const auto rows = R.unwrap();
    const auto padded = P.unwrap();
    RuntimeCheck(rows > 0 && rows <= padded, "moe_front_v2: 0 < rows <= padded_max");
    RuntimeCheck(padded * kTopK % 4 == 0 && padded * kTopK <= 0x7fffffff, "moe_front_v2: P * top_k slots");
    RuntimeCheck(reinterpret_cast<uintptr_t>(buf_topk_idx.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(buf_topk_weights.data_ptr()) % 16 == 0,
                 "moe_front_v2: top-k buffers are 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(buf_x.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
                 "moe_front_v2: x, buf.x and the logits are 16-byte aligned");
    RuntimeCheck(X.unwrap() >= kHidden / 128 && reinterpret_cast<uintptr_t>(buf_x_sf.data_ptr()) % 4 == 0,
                 "moe_front_v2: buf.x_sf rows");
    RuntimeCheck(shared_x_sf.has_value() == kSharedSF,
                 "moe_front_v2: shared_x_sf must be given iff the module writes it");

    auto params = Params{
        .logits = static_cast<const float*>(logits.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .bias_alt = nullptr,
        .input_ids = nullptr,
        .num_live = nullptr,
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .buf_x = static_cast<fp8_e4m3_t*>(buf_x.data_ptr()),
        .buf_x_sf = static_cast<int32_t*>(buf_x_sf.data_ptr()),
        .buf_topk_idx = static_cast<int64_t*>(buf_topk_idx.data_ptr()),
        .buf_topk_weights = static_cast<float*>(buf_topk_weights.data_ptr()),
        .shared_x_sf = nullptr,
        .logits_stride = kExperts,
        .ids_stride = 0,
        .alt_token = alt_token,
        .x_sf_stride = X.unwrap(),
        .shared_col_stride = 0,
        .shared_block_m = 1,
        .rows = static_cast<uint32_t>(rows),
        .pad_group_begin = static_cast<uint32_t>(rows * kTopK / 4),
        .pad_group_end = static_cast<uint32_t>(padded * kTopK / 4),
        .ids64 = false,
        .live64 = false,
        .scale = static_cast<float>(scale),
        .renorm_eps = static_cast<float>(renorm_eps),
    };
    if constexpr (kSharedSF) {
      auto SR = SymbolicSize{"shared_sf_rows"};
      auto SC = SymbolicSize{"shared_sf_col_stride"};
      const auto sf = shared_x_sf.value();
      TensorMatcher({SR, kHidden / 128}).with_strides({1, SC}).with_dtype<int32_t>().with_device(device_).verify(sf);
      RuntimeCheck(shared_block_m > 0, "moe_front_v2: shared_block_m must be positive");
      const auto need = (rows + shared_block_m - 1) / shared_block_m * ((shared_block_m + 127) / 128 * 128);
      RuntimeCheck(SR.unwrap() >= need, "moe_front_v2: shared scale buffer is too small for shared_block_m");
      RuntimeCheck(SC.unwrap() >= SR.unwrap(), "moe_front_v2: shared scale columns must not overlap");
      params.shared_x_sf = static_cast<int32_t*>(sf.data_ptr());
      params.shared_col_stride = SC.unwrap();
      params.shared_block_m = static_cast<uint32_t>(shared_block_m);
    }
    RuntimeCheck(!input_ids.has_value() || bias_alt.has_value(), "moe_front_v2: input_ids needs bias_alt");
    if (input_ids.has_value()) {
      TensorMatcher({kExperts}).with_strides({1}).with_dtype<float>().with_device(device_).verify(bias_alt.value());
      TensorMatcher({R}).with_strides({T}).with_dtype<int32_t, int64_t>().with_device(device_).verify(input_ids.value());
      params.bias_alt = static_cast<const float*>(bias_alt.value().data_ptr());
      params.input_ids = input_ids.value().data_ptr();
      params.ids_stride = T.unwrap();
      params.ids64 = input_ids.value().dtype().bits == 64;
    }
    if (num_live.has_value()) {
      const auto& live = num_live.value();
      const auto dtype = live.dtype();
      RuntimeCheck(dtype.code == kDLInt && (dtype.bits == 32 || dtype.bits == 64),
                   "moe_front_v2: num_token_non_padded is an int32 or int64 scalar");
      params.num_live = live.data_ptr();
      params.live64 = dtype.bits == 64;
    }
    const DLDevice dev = device_.unwrap();
    int sms = 0;
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev.device_id));

    static bool smem_set[64] = {};
    RuntimeCheck(dev.device_id >= 0 && dev.device_id < 64, "moe_front_v2: device id");
    if (!smem_set[dev.device_id]) {
      int smem_optin = 0;
      RuntimeDeviceCheck(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev.device_id));
      RuntimeCheck(static_cast<size_t>(smem_optin) >= kSmemBytes, "moe_front_v2: too little shared memory a block");
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      smem_set[dev.device_id] = true;
    }
    const int64_t ctas_for_rows = (rows + kWarps - 1) / kWarps;
    const uint32_t ctas = static_cast<uint32_t>(ctas_for_rows < sms ? ctas_for_rows : sms);
    LaunchKernel(ctas, kWarps * 32, dev, kSmemBytes).enable_pdl(use_pdl)(kernel, params);
  }
};

}
"""
