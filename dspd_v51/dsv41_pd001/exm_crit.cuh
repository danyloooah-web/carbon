// EXM critical kernel (dspd/exm.py; dspd-w33e, SHIPPED OFF; NEVER run on a GPU yet): the main-stream half of one
// shifted mHC boundary, row-parallel, no grid barrier, no tensor core. One CTA of 320 threads per row t, any number of
// rows, PDL-launchable (griddepcontrol.wait before the first global access; dependents triggered at exit).
//
// It writes exactly what stock's deep_gemm.mega_mhc writes for row t apart from the next layer's stats:
//   new_residual[t, o, :]  (EXM_SPEC 5.1)    y_bf16[t, :] = the normalized row (5.5; stock's x1 staging is overwritten)
//   y_fp8[t, :] + the 40 UE8M0 words of row t (5.6; attention form; the MoE-input form only under kSBM > 0, unexported)
// Arithmetic, op for op, as derived in struct/w33_exm/EXM_SPEC.md (written fresh here, DeepGEMM helpers included):
//   phase 1, one thread per 8 hidden (any map: elementwise): pv_o = bf16(fma_i(res_i, comb[i][o], x * post_o)) with
//     __fmul2_rn / __ffma2_rn in stock's order (impls:484-500), x1 = bf16(fma_o(pv_o, pre_o, +0))
//     (impls:479,502-506,520);
//     x1 of the row stays in shared memory (10 KB)
//   phase 2, ONE THREAD PER STOCK CHAIN: thread tid = 8 s + 4 p + c runs split s, post-worker parity p, column pair c:
//     fma.rn.f32.bf16 over k-blocks KB(s) ascending, then atoms u = p, p+2, p+4, p+6, on hidden pair kb*32 + 4u + c
//     (impls:458-461,517-523; ptx::accumulate_square), then .x + .y, math::warp_reduce_sum<4> (xor 2, xor 1: lanes
//     4p..4p+3 are aligned), then (0 + W0) + W1 (impls:367-375); thread 0 sums the S split values 0..S-1 from +0
//     (epilogue:323-330), rms = rsqrtf(K4) * scale (epilogue:331-332)
//   phase 3, stock's norm lane map (epilogue:338-384): warp w handles 256-element packs w, w + 10; lane l owns 8
//     elements; normalize_bf16_pair; for FP8: get_packed_bf16_amax, warp_reduce_max<4> on the packed pair (second
//     half +0 as stock's absent token), get_ue8m0_sf_exp, scale_bf16x2_into_fp8x4, the two __byte_perm packs, lane % 16
//     == 0 stores word 2 * pack + lane / 16 at [t * stride_token + word * stride_word]; SF padding never written
#pragma once

#include "exm_common.cuh"

namespace sglang {
namespace exm {

inline constexpr uint32_t kCritThreads = 320;  // >= 8 * kMaxSplits chain threads; 10 warps for the 20 norm packs

struct CritParams {
  const bf16_t* x;             // [T, 5120]
  const bf16_t* residual;      // [T, 4, 5120]
  const float* pre;            // [T, 4]   shifted_prev_mix
  const float* post;           // [T, 4]   post_mix
  const float* comb;           // [T, 4, 4] comb_res_mix[t][i][o]
  const bf16_t* weight;        // [5120]   the next norm's weight
  bf16_t* new_residual;        // [T, 4, 5120]
  bf16_t* y_bf16;              // [T, 5120]
  fp8_e4m3_t* y_fp8;           // [T, 5120] (FP8 forms)
  int32_t* y_sf;               // primary SF words: [t * sf_stride_token + w * sf_stride_word]
  int32_t* y_shared_sf;        // MoE form only: [srow(t) + w * shared_stride_word]
  int64_t sf_stride_token;
  int64_t sf_stride_word;
  int64_t shared_stride_word;
  uint32_t num_splits;         // S of the call (40 or 27)
  float eps;                   // rmsnorm_eps
  float scale;                 // rmsnorm_scale (1.0 from the engine)
};

// stock's shared-SF row (epilogue:116-122), MoE-input form only
template <uint32_t kSBM>
SGL_DEVICE uint32_t shared_sf_row(const uint32_t t) {
  const uint32_t i = t % kSBM;
  return t / kSBM * ((kSBM + 127u) & ~127u) + (i & ~127u) + (i & 31u) * 4u + ((i >> 5) & 3u);
}

template <bool kUsePDL, bool kFP8, uint32_t kSBM>
__global__ void __launch_bounds__(kCritThreads, 1) exm_crit_kernel(const __grid_constant__ CritParams p) {
  __shared__ __align__(16) uint32_t s_x1[kHidden / 2];  // the row's x1 as bf16 pairs
  __shared__ float s_split[kMaxSplits];
  __shared__ float s_rms;
  const uint32_t t = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5;
  if constexpr (kUsePDL) pdl_wait();

  // ---- phase 1: new residual and x1 (elementwise)
  float pre[kRoutes], post[kRoutes], comb[kRoutes * kRoutes];
  {
    const float4 a = reinterpret_cast<const float4*>(p.pre)[t];
    const float4 b = reinterpret_cast<const float4*>(p.post)[t];
    pre[0] = a.x, pre[1] = a.y, pre[2] = a.z, pre[3] = a.w;
    post[0] = b.x, post[1] = b.y, post[2] = b.z, post[3] = b.w;
#pragma unroll
    for (uint32_t i = 0; i < kRoutes; ++i) {
      const float4 c = reinterpret_cast<const float4*>(p.comb)[t * kRoutes + i];  // comb[t][i][0..3]
      comb[4 * i + 0] = c.x, comb[4 * i + 1] = c.y, comb[4 * i + 2] = c.z, comb[4 * i + 3] = c.w;
    }
  }
  const uint64_t row_x = static_cast<uint64_t>(t) * kHidden;
  const uint64_t row_r = static_cast<uint64_t>(t) * kRoutes * kHidden;
  for (uint32_t q = tid; q < kHidden / 8; q += kCritThreads) {
    const uint4 xv = *reinterpret_cast<const uint4*>(p.x + row_x + q * 8);
    uint4 rv[kRoutes];
#pragma unroll
    for (uint32_t i = 0; i < kRoutes; ++i)
      rv[i] = *reinterpret_cast<const uint4*>(p.residual + row_r + i * kHidden + q * 8);
    uint32_t nv[kRoutes][4];
    uint32_t x1v[4];
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const float2 xf = __bfloat1622float2(bf2_of((&xv.x)[j]));
      float2 x1 = {0.0f, 0.0f};  // stock's `float2 x1[2] = {}`: the chain starts at +0 (Z-0)
#pragma unroll
      for (uint32_t o = 0; o < kRoutes; ++o) {
        float2 v = __fmul2_rn(xf, {post[o], post[o]});
#pragma unroll
        for (uint32_t i = 0; i < kRoutes; ++i)
          v = __ffma2_rn(__bfloat1622float2(bf2_of((&rv[i].x)[j])), {comb[4 * i + o], comb[4 * i + o]}, v);
        const nv_bfloat162 vb = __float22bfloat162_rn(v);  // the shared rounding boundary (impls:496-498)
        nv[o][j] = word_of(vb);
        x1 = __ffma2_rn(__bfloat1622float2(vb), {pre[o], pre[o]}, x1);
      }
      x1v[j] = word_of(__float22bfloat162_rn(x1));
    }
#pragma unroll
    for (uint32_t o = 0; o < kRoutes; ++o)
      *reinterpret_cast<uint4*>(p.new_residual + row_r + o * kHidden + q * 8) =
          make_uint4(nv[o][0], nv[o][1], nv[o][2], nv[o][3]);
    *reinterpret_cast<uint4*>(&s_x1[q * 4]) = make_uint4(x1v[0], x1v[1], x1v[2], x1v[3]);
  }
  __syncthreads();

  // ---- phase 2: stock's x1 square chains, one thread per (split, parity, column pair)
  const uint32_t num_splits = p.num_splits;
  {
    const uint32_t s = tid >> 3, par = (tid >> 2) & 1u, c = tid & 3u;
    float2 q = {0.0f, 0.0f};
    if (s < num_splits) {
      const uint32_t kb0 = split_first_kb(s, num_splits), nkb = split_num_kb(s, num_splits);
      for (uint32_t kb = kb0; kb < kb0 + nkb; ++kb) {
#pragma unroll
        for (uint32_t u = par; u < 8; u += 2)
          deep_gemm::ptx::accumulate_square(q, bf2_of(s_x1[kb * 32 + u * 4 + c]));
      }
    }
    // every lane of every warp takes part in the shuffles (lanes past the last split carry +0 and are dropped)
    const float w = deep_gemm::math::warp_reduce_sum<4>(__fadd_rn(q.x, q.y));
    const float w1 = __shfl_xor_sync(0xffffffffu, w, 4);
    if ((tid & 7u) == 0 && s < num_splits) s_split[s] = __fadd_rn(__fadd_rn(0.0f, w), w1);
  }
  __syncthreads();
  if (tid == 0) {
    float tot = 0.0f;
    for (uint32_t s = 0; s < num_splits; ++s) tot = __fadd_rn(tot, s_split[s]);
    s_rms = __fmul_rn(rsqrtf(k4_rms_arg(tot, kInvHidden, p.eps)), p.scale);
  }
  __syncthreads();
  const float rms = s_rms;

  // ---- phase 3: normalized row (+ FP8 pair) in stock's norm lane map
  for (uint32_t pack = warp; pack < kHidden / 256; pack += kCritThreads / 32) {
    const uint32_t h = pack * 256 + lane * 8;
    const uint4 xv = *reinterpret_cast<const uint4*>(&s_x1[h / 2]);
    const uint4 wv = *reinterpret_cast<const uint4*>(p.weight + h);
    nv_bfloat162 y[4];
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {  // normalize_bf16_pair (epilogue:112-114): x1 * rms first, then * weight
      const float2 v = __fmul2_rn(__bfloat1622float2(bf2_of((&xv.x)[j])), {rms, rms});
      y[j] = __float22bfloat162_rn(__fmul2_rn(v, __bfloat1622float2(bf2_of((&wv.x)[j]))));
    }
    *reinterpret_cast<uint4*>(p.y_bf16 + row_x + h) = make_uint4(word_of(y[0]), word_of(y[1]), word_of(y[2]),
                                                                 word_of(y[3]));
    if constexpr (kFP8) {
      const nv_bfloat162 amax = deep_gemm::math::warp_reduce_max<4>(
          nv_bfloat162(deep_gemm::math::get_packed_bf16_amax(y), __ushort_as_bfloat16(0)));
      const uint32_t e = deep_gemm::math::get_ue8m0_sf_exp(amax.x);
      const nv_bfloat162 inv = __bfloat162bfloat162(deep_gemm::math::get_ue8m0_sf_inv<nv_bfloat16>(e));
      const uint2 f = {deep_gemm::math::scale_bf16x2_into_fp8x4(y[0], y[1], inv, inv),
                       deep_gemm::math::scale_bf16x2_into_fp8x4(y[2], y[3], inv, inv)};
      *reinterpret_cast<uint2*>(p.y_fp8 + row_x + h) = f;
      const uint32_t paired = __byte_perm(e, __shfl_xor_sync(0xffffffffu, e, 4), 0x1140);
      const uint32_t word = __byte_perm(paired, __shfl_xor_sync(0xffffffffu, paired, 8), 0x5410);
      if ((lane & 15u) == 0) {
        const uint32_t w = pack * 2 + lane / 16;
        p.y_sf[static_cast<int64_t>(t) * p.sf_stride_token + static_cast<int64_t>(w) * p.sf_stride_word] =
            static_cast<int32_t>(word);
        if constexpr (kSBM > 0)
          p.y_shared_sf[static_cast<int64_t>(w) * p.shared_stride_word + shared_sf_row<kSBM>(t)] =
              static_cast<int32_t>(word);
      }
    }
  }
}

// Host side. A form: run(x, residual, pre, post, comb, weight, new_residual, y_bf16, y_fp8, y_sf, S, eps, scale,
// stream) with y_sf the engine's [T, 40] column-major view (strides (1, align4(T))); F form: run_f without the pair.
// `stream` is the raw cudaStream_t of torch's current stream, passed explicitly (exm.py), never resolved here.
template <bool kUsePDL>
struct ExmCrit {
  static void check_common(const tvm::ffi::TensorView x, const tvm::ffi::TensorView residual,
                           const tvm::ffi::TensorView pre, const tvm::ffi::TensorView post,
                           const tvm::ffi::TensorView comb, const tvm::ffi::TensorView weight,
                           const tvm::ffi::TensorView new_residual, const tvm::ffi::TensorView y_bf16,
                           host::SymbolicSize& M, host::SymbolicDevice& device) {
    using namespace host;
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, kRoutes, kHidden}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({M, kRoutes}).with_dtype<float>().with_device(device).verify(pre);
    TensorMatcher({M, kRoutes}).with_dtype<float>().with_device(device).verify(post);
    TensorMatcher({M, kRoutes, kRoutes}).with_dtype<float>().with_device(device).verify(comb);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, kRoutes, kHidden}).with_dtype<bf16_t>().with_device(device).verify(new_residual);
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device).verify(y_bf16);
    for (const auto& v : {x, residual, pre, post, comb, weight, new_residual, y_bf16})
      RuntimeCheck(reinterpret_cast<uintptr_t>(v.data_ptr()) % 16u == 0, "EXM: 16-byte alignment");
  }

  static CritParams params(const tvm::ffi::TensorView x, const tvm::ffi::TensorView residual,
                           const tvm::ffi::TensorView pre, const tvm::ffi::TensorView post,
                           const tvm::ffi::TensorView comb, const tvm::ffi::TensorView weight,
                           const tvm::ffi::TensorView new_residual, const tvm::ffi::TensorView y_bf16,
                           const int64_t num_splits, const double eps, const double scale) {
    host::RuntimeCheck(num_splits >= 1 && num_splits <= static_cast<int64_t>(kMaxSplits), "EXM: split count");
    return CritParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .residual = static_cast<const bf16_t*>(residual.data_ptr()),
        .pre = static_cast<const float*>(pre.data_ptr()),
        .post = static_cast<const float*>(post.data_ptr()),
        .comb = static_cast<const float*>(comb.data_ptr()),
        .weight = static_cast<const bf16_t*>(weight.data_ptr()),
        .new_residual = static_cast<bf16_t*>(new_residual.data_ptr()),
        .y_bf16 = static_cast<bf16_t*>(y_bf16.data_ptr()),
        .y_fp8 = nullptr,
        .y_sf = nullptr,
        .y_shared_sf = nullptr,
        .sf_stride_token = 0,
        .sf_stride_word = 0,
        .shared_stride_word = 0,
        .num_splits = static_cast<uint32_t>(num_splits),
        .eps = static_cast<float>(eps),      // C++ float as stock's host takes it (binary32 RN of the Python float)
        .scale = static_cast<float>(scale),
    };
  }

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView residual, const tvm::ffi::TensorView pre,
                  const tvm::ffi::TensorView post, const tvm::ffi::TensorView comb, const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView new_residual, const tvm::ffi::TensorView y_bf16,
                  const tvm::ffi::TensorView y_fp8, const tvm::ffi::TensorView y_sf, const int64_t num_splits,
                  const double eps, const double scale, const int64_t stream) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto W = SymbolicSize{"sf_word_stride"};
    device.set_options<kDLCUDA>();
    check_common(x, residual, pre, post, comb, weight, new_residual, y_bf16, M, device);
    TensorMatcher({M, kHidden}).with_dtype<int8_t, fp8_e4m3_t>().with_device(device).verify(y_fp8);
    TensorMatcher({M, kSFWords}).with_strides({1, W}).with_dtype<int32_t>().with_device(device).verify(y_sf);
    const auto rows = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(rows > 0 && W.unwrap() >= static_cast<int64_t>(rows), "EXM: SF word stride below the row count");
    RuntimeCheck(reinterpret_cast<uintptr_t>(y_fp8.data_ptr()) % 16u == 0, "EXM: y_fp8 alignment");
    auto prm = params(x, residual, pre, post, comb, weight, new_residual, y_bf16, num_splits, eps, scale);
    prm.y_fp8 = static_cast<fp8_e4m3_t*>(y_fp8.data_ptr());
    prm.y_sf = static_cast<int32_t*>(y_sf.data_ptr());
    prm.sf_stride_token = 1;
    prm.sf_stride_word = W.unwrap();
    LaunchKernel(rows, kCritThreads, reinterpret_cast<cudaStream_t>(stream))
        .enable_pdl(kUsePDL)(exm_crit_kernel<kUsePDL, true, 0>, prm);
  }

  static void run_f(const tvm::ffi::TensorView x, const tvm::ffi::TensorView residual,
                    const tvm::ffi::TensorView pre, const tvm::ffi::TensorView post, const tvm::ffi::TensorView comb,
                    const tvm::ffi::TensorView weight, const tvm::ffi::TensorView new_residual,
                    const tvm::ffi::TensorView y_bf16, const int64_t num_splits, const double eps,
                    const double scale, const int64_t stream) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    device.set_options<kDLCUDA>();
    check_common(x, residual, pre, post, comb, weight, new_residual, y_bf16, M, device);
    const auto rows = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(rows > 0, "EXM: empty call");
    const auto prm = params(x, residual, pre, post, comb, weight, new_residual, y_bf16, num_splits, eps, scale);
    LaunchKernel(rows, kCritThreads, reinterpret_cast<cudaStream_t>(stream))
        .enable_pdl(kUsePDL)(exm_crit_kernel<kUsePDL, false, 0>, prm);
  }
};

}  // namespace exm
}  // namespace sglang
