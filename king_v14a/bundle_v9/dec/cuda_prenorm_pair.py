NAME = "mhc/dec_prenorm_pair.cuh"

SOURCE = r"""#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

#include <cstdint>

namespace dec_prenorm_pair {

using namespace deep_gemm;

constexpr uint32_t kK = 20480, kN = 24, kSplits = 16, BM = 64, BN = 32, BK = 64;
constexpr uint32_t kKBPerSplit = kK / BK / kSplits;

template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
__device__ __forceinline__ uint32_t swz(const uint32_t& offset, const uint32_t& lane_idx) {
  const auto bank_group_idx = offset + lane_idx * (kSwizzleMode / kSwizzleBase);
  constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
  constexpr bool kHasShortcut = (kSwizzleMode / kSwizzleBase) == kNumBankGroups;
  auto row = kHasShortcut ? (offset / kNumBankGroups + lane_idx) : (bank_group_idx / kNumBankGroups);
  auto col = kHasShortcut ? (offset) : (bank_group_idx % kNumBankGroups);
  col ^= row % (kSwizzleMode / kSwizzleBase);
  return row * 128 + col * kSwizzleBase;
}

template <uint32_t kNumRB>
struct Plan {
  static constexpr uint32_t A_RB = BM * BK * 2;
  static constexpr uint32_t B = BN * BK * 4;
  static constexpr uint32_t STAGE = kNumRB * A_RB + 2 * B;
  static constexpr uint32_t FIT = 200u * 1024u / STAGE;
  static constexpr uint32_t STAGES = FIT < 2 ? 2 : (FIT > 8 ? 8 : FIT);
  static constexpr uint32_t CAST = (8 - kNumRB) > 6 ? 6 : (8 - kNumRB);
  static constexpr uint32_t BAR = STAGES * STAGE;
  static constexpr uint32_t BYTES = BAR + (2 * STAGES + 2 * CAST + 1) * 8 + 16;
  static constexpr uint32_t TMEM_D = BK * CAST;
  static constexpr uint32_t TMEM_COLS = 512;
  static_assert(TMEM_D + kNumRB * 2 * BN <= TMEM_COLS);
};

template <uint32_t kNumRB>
__global__ void __launch_bounds__(256, 1)
prenorm_pair_kernel(const uint32_t shape_m, const __grid_constant__ cute::TmaDescriptor tm_a,
          const __grid_constant__ cute::TmaDescriptor tm_hi, const __grid_constant__ cute::TmaDescriptor tm_lo,
          float* __restrict__ mix_hi, float* __restrict__ mix_lo, float* __restrict__ sqr_sum) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
  using P = Plan<kNumRB>;
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr uint32_t kStages = P::STAGES;
  constexpr auto kMajor = cute::UMMA::Major::K;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto lane_idx = ptx::get_lane_idx();
  extern __shared__ __align__(1024) uint8_t smem[];

  auto sa = [&](const uint32_t st, const uint32_t rb) {
    return reinterpret_cast<nv_bfloat16*>(smem + st * P::STAGE + rb * P::A_RB);
  };
  auto sb = [&](const uint32_t st, const uint32_t h) {
    return reinterpret_cast<float*>(smem + st * P::STAGE + kNumRB * P::A_RB + h * P::B);
  };
  auto bars = reinterpret_cast<Barrier*>(smem + P::BAR);
  Barrier* full = bars;
  Barrier* empty = bars + kStages;
  constexpr uint32_t kCast = P::CAST;
  Barrier* full_cast = bars + 2 * kStages;
  Barrier* empty_cast = bars + 2 * kStages + kCast;
  Barrier* tmem_full = bars + 2 * kStages + 2 * kCast;
  auto tmem_ptr = reinterpret_cast<uint32_t*>(bars + 2 * kStages + 2 * kCast + 1);

  if (warp_idx == 0 and cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(&tm_a);
    cute::prefetch_tma_descriptor(&tm_hi);
    cute::prefetch_tma_descriptor(&tm_lo);
  }
  if (warp_idx == 1 and cute::elect_one_sync()) {
    #pragma unroll
    for (uint32_t i = 0; i < kStages; ++i) {
      full[i].init(1);
      empty[i].init(1);
    }
    #pragma unroll
    for (uint32_t i = 0; i < kCast; ++i) {
      full_cast[i].init(128);
      empty_cast[i].init(1);
    }
    tmem_full->init(1);
    cutlass::arch::fence_barrier_init();
  } else if (warp_idx == 2) {
    cute::TMEM::Allocator1Sm().allocate(P::TMEM_COLS, tmem_ptr);
  }
  __syncthreads();

  const uint32_t g = blockIdx.x, G = gridDim.x;
  const uint32_t nsplit = (kSplits - g + G - 1) / G;
  const uint32_t T = nsplit * kKBPerSplit;

  if (warp_idx == 0) {
    if (cute::elect_one_sync()) {
      for (uint32_t t = 0; t < T; ++t) {
        const uint32_t st = t % kStages;
        empty[st].wait(((t / kStages) & 1) ^ 1);
        const uint32_t s = g + (t / kKBPerSplit) * G;
        const uint32_t k_idx = (s * kKBPerSplit + t % kKBPerSplit) * BK;
        #pragma unroll
        for (uint32_t rb = 0; rb < kNumRB; ++rb)
          tma::copy<BK, BM, 128>(&tm_a, &full[st], sa(st, rb), k_idx, rb * BM);
        tma::copy<BK, BN, 128>(&tm_hi, &full[st], sb(st, 0), k_idx, 0);
        tma::copy<BK, BN, 128>(&tm_lo, &full[st], sb(st, 1), k_idx, 0);
        full[st].arrive_and_expect_tx(P::STAGE);
      }
    }
  } else if (warp_idx == 1) {
    constexpr uint32_t UMMA_K = 32 / sizeof(float);
    constexpr uint32_t SBK = 128 / sizeof(float);
    using umma_t = cute::SM100_MMA_TF32_TS<cutlass::tfloat32_t, cutlass::tfloat32_t, float, BM, BN, kMajor, kMajor>;
    auto instr_desc = cute::UMMA::make_instr_desc<cutlass::tfloat32_t, cutlass::tfloat32_t, float, BM, BN, kMajor, kMajor>();
    const auto& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
    for (uint32_t t = 0; t < T; ++t) {
      const uint32_t st = t % kStages, kb = t % kKBPerSplit;
      #pragma unroll
      for (uint32_t rb = 0; rb < kNumRB; ++rb) {
        const uint32_t u = t * kNumRB + rb, cs = u % kCast;
        full_cast[cs].wait((u / kCast) & 1);
        ptx::tcgen05_after_thread_sync();
        const uint32_t d = P::TMEM_D + rb * 2 * BN;
        #pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
          auto b_desc = mma::sm100::make_umma_desc<kMajor, BN, SBK, 128>(sb(st, h), 0, 0);
          const uint32_t base_lo = b_desc.lo;
          #pragma unroll
          for (uint32_t k = 0; k < BK / UMMA_K; ++k) {
            const uint32_t atom_idx = (k * UMMA_K) / SBK;
            const uint32_t in_atom_idx = (k * UMMA_K) % SBK;
            const uint32_t offset = atom_idx * BN * SBK;
            b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajor, BN, 128, float>(base_lo, offset, in_atom_idx);
            umma_t::fma(BK * cs + k * UMMA_K, b_desc, d + h * BN, kb > 0 or k > 0, runtime_instr_desc);
          }
        }
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&empty_cast[cs]));
      }
      cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&empty[st]));
      if (kb == kKBPerSplit - 1) cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full));
    }
  } else if (warp_idx >= 4) {
    const uint32_t sw = warp_idx - 4;
    constexpr uint32_t kNumLoads = BK / 8;
    float2 sum[kNumRB][2];
    #pragma unroll
    for (uint32_t rb = 0; rb < kNumRB; ++rb) sum[rb][0] = sum[rb][1] = float2{0, 0};
    uint32_t split_iter = 0;
    for (uint32_t t = 0; t < T; ++t) {
      const uint32_t st = t % kStages, kb = t % kKBPerSplit;
      full[st].wait((t / kStages) & 1);
      #pragma unroll
      for (uint32_t rb = 0; rb < kNumRB; ++rb) {
        const uint32_t u = t * kNumRB + rb, cs = u % kCast;
        const auto base = reinterpret_cast<uint8_t*>(sa(st, rb)) + sw * (BM / 4) * 128;
        uint32_t v[2][kNumLoads];
        #pragma unroll
        for (uint32_t i = 0; i < kNumLoads; i += 2) {
          auto p = base + swz<128>(i + lane_idx / 16, lane_idx % 16);
          ptx::SM90_U32x4_LDSM_N::copy(v[0][i + 0], v[1][i + 0], v[0][i + 1], v[1][i + 1], p);
        }
        empty_cast[cs].wait(((u / kCast) & 1) ^ 1);
        float2 f[2][kNumLoads];
        const auto& up = reinterpret_cast<uint32_t*>(&f[0]);
        const auto& dn = reinterpret_cast<uint32_t*>(&f[1]);
        #pragma unroll
        for (uint32_t i = 0; i < kNumLoads; ++i) {
          #pragma unroll
          for (uint32_t w = 0; w < 2; ++w) {
            f[w][i] = __bfloat1622float2(*reinterpret_cast<nv_bfloat162*>(&v[w][i]));
            sum[rb][w] = __ffma2_rn(f[w][i], f[w][i], sum[rb][w]);
          }
          const auto idx_0 = i * 2, idx_1 = i * 2 + 1;
          cute::SM100_TMEM_STORE_16dp256b1x::copy(up[idx_0], up[idx_1], dn[idx_0], dn[idx_1], cs * BK + i * 8);
        }
        cutlass::arch::fence_view_async_tmem_store();
        cutlass::arch::fence_view_async_shared();
        ptx::tcgen05_before_thread_sync();
        full_cast[cs].arrive();
      }
      if (kb == kKBPerSplit - 1) {
        const uint32_t s = g + split_iter * G;
        tmem_full->wait(split_iter & 1);
        ptx::tcgen05_after_thread_sync();
        #pragma unroll
        for (uint32_t rb = 0; rb < kNumRB; ++rb) {
          const uint32_t m = rb * BM + sw * (BM / 4) + lane_idx;
          #pragma unroll
          for (uint32_t h = 0; h < 2; ++h) {
            float* dst = (h ? mix_lo : mix_hi) + (static_cast<size_t>(s) * shape_m + m) * kN;
            #pragma unroll
            for (uint32_t i = 0; i < kN / 4; ++i) {
              uint32_t r[4];
              cute::SM100_TMEM_LOAD_32dp32b4x::copy(P::TMEM_D + rb * 2 * BN + h * BN + i * 4, r[0], r[1], r[2], r[3]);
              cutlass::arch::fence_view_async_tmem_load();
              if (lane_idx < 16 and m < shape_m)
                *reinterpret_cast<uint4*>(dst + i * 4) = make_uint4(r[0], r[1], r[2], r[3]);
            }
          }
          #pragma unroll
          for (uint32_t w = 0; w < 2; ++w) {
            const auto reduced = math::warp_reduce_sum<4>(sum[rb][w].x + sum[rb][w].y);
            const uint32_t mq = rb * BM + sw * (BM / 4) + lane_idx / 4 + w * 8;
            if (lane_idx % 4 == 0 and mq < shape_m) sqr_sum[s * shape_m + mq] = reduced;
            sum[rb][w] = float2{0, 0};
          }
        }
        ptx::tcgen05_before_thread_sync();
        ++split_iter;
      }
    }
  }
  __syncthreads();
  if (warp_idx == 2) cute::TMEM::Allocator1Sm().free(0, P::TMEM_COLS);
#endif
}

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

inline EncodeFn encoder() {
  static EncodeFn fn = [] {
    void* f = nullptr;
    cudaDriverEntryPointQueryResult q;
    const cudaError_t err =
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q);
    sglang::host::RuntimeCheck(err == cudaSuccess && q == cudaDriverEntryPointSuccess && f != nullptr,
                               "pnorm: cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<EncodeFn>(f);
  }();
  return fn;
}

inline CUtensorMap encode(const void* p, CUtensorMapDataType type, uint32_t elem, uint64_t rows, uint32_t box_inner,
                          uint32_t box_rows) {
  CUtensorMap tm;
  cuuint64_t dims[2] = {kK, rows};
  cuuint64_t strides[1] = {cuuint64_t(kK) * elem};
  cuuint32_t box[2] = {box_inner, box_rows};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = encoder()(&tm, type, 2, const_cast<void*>(p), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "pnorm: tensor map encode failed ", int(r));
  return tm;
}

template <uint32_t kNumRB>
void launch(const void* x, const void* hi, const void* lo, float* mh, float* ml, float* sq, uint32_t m, uint32_t ctas,
            DLDevice dev) {
  using P = Plan<kNumRB>;
  auto k = prenorm_pair_kernel<kNumRB>;
  static const bool attr = [&] {
    sglang::host::RuntimeCheck(
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(P::BYTES)) == cudaSuccess,
        "pnorm: shared memory attribute");
    return true;
  }();
  (void)attr;
  const CUtensorMap ta = encode(x, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, m, BK, BM);
  const CUtensorMap th = encode(hi, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 4, kN, 128 / 4, BN);
  const CUtensorMap tl = encode(lo, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 4, kN, 128 / 4, BN);
  sglang::host::LaunchKernel(dim3(ctas), dim3(256), dev, P::BYTES)(k, m, ta, th, tl, mh, ml, sq);
}

struct PrenormPair {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView hi, const tvm::ffi::TensorView lo,
                  const tvm::ffi::TensorView mix_hi, const tvm::ffi::TensorView mix_lo, const tvm::ffi::TensorView sq,
                  const int64_t ctas) {
    using namespace sglang::host;
    auto M = SymbolicSize{"rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, kK}).with_strides({kK, 1}).with_dtype<sglang::bf16_t>().with_device(dev).verify(x);
    TensorMatcher({kN, kK}).with_strides({kK, 1}).with_dtype<float>().with_device(dev).verify(hi);
    TensorMatcher({kN, kK}).with_strides({kK, 1}).with_dtype<float>().with_device(dev).verify(lo);
    TensorMatcher({kSplits, M, kN}).with_dtype<float>().with_device(dev).verify(mix_hi);
    TensorMatcher({kSplits, M, kN}).with_dtype<float>().with_device(dev).verify(mix_lo);
    TensorMatcher({kSplits, M}).with_dtype<float>().with_device(dev).verify(sq);
    const int64_t rows = M.unwrap();
    RuntimeCheck(rows >= 32 && rows <= 5 * int64_t(BM), "pnorm: rows ", rows);
    RuntimeCheck(ctas == 1 || ctas == 2 || ctas == 4 || ctas == 8 || ctas == 16, "pnorm: ctas ", ctas);
    for (const auto* t : {&mix_hi, &mix_lo}) {
      RuntimeCheck(t->stride(0) == rows * kN && t->stride(1) == kN && t->stride(2) == 1, "pnorm: mix layout");
    }
    RuntimeCheck(sq.stride(0) == rows && sq.stride(1) == 1, "pnorm: sq layout");
    for (const void* p : {x.data_ptr(), hi.data_ptr(), lo.data_ptr(), mix_hi.data_ptr(), mix_lo.data_ptr()}) {
      RuntimeCheck(reinterpret_cast<uintptr_t>(p) % 16 == 0, "pnorm: 16-byte alignment");
    }
    const uint32_t m = uint32_t(rows), nrb = (m + BM - 1) / BM;
    auto mh = static_cast<float*>(mix_hi.data_ptr());
    auto ml = static_cast<float*>(mix_lo.data_ptr());
    auto s = static_cast<float*>(sq.data_ptr());
    const DLDevice d = dev.unwrap();
    switch (nrb) {
      case 1: launch<1>(x.data_ptr(), hi.data_ptr(), lo.data_ptr(), mh, ml, s, m, uint32_t(ctas), d); break;
      case 2: launch<2>(x.data_ptr(), hi.data_ptr(), lo.data_ptr(), mh, ml, s, m, uint32_t(ctas), d); break;
      case 3: launch<3>(x.data_ptr(), hi.data_ptr(), lo.data_ptr(), mh, ml, s, m, uint32_t(ctas), d); break;
      case 4: launch<4>(x.data_ptr(), hi.data_ptr(), lo.data_ptr(), mh, ml, s, m, uint32_t(ctas), d); break;
      default: launch<5>(x.data_ptr(), hi.data_ptr(), lo.data_ptr(), mh, ml, s, m, uint32_t(ctas), d); break;
    }
  }
};

}
"""

# king-v6 FFN-PAIR cluster form (king-v2d's fallback; prenorm_pair.FFN_PAIR_BANDS ctas 2): the same kernel in its own
# namespace, launched as one 2-CTA cluster so both CTAs share one TPC (a plain 2-CTA grid can land on two TPCs and block
# MegaMoE's 73rd cluster on every EP rank, VERIFY_dec_mhc M1 correction 2). Device code is SOURCE's, unchanged; only the
# namespace, the ctas check and the launch attribute differ. Built only when a band selects it; SOURCE (and its JIT key)
# is untouched.
CLUSTER_NAME = "mhc/dec_prenorm_pair_cluster.cuh"
CLUSTER_ENTRY = "dec_prenorm_pair_cluster::PrenormPair::run"

_CLUSTER_EDITS = (
    ("namespace dec_prenorm_pair {", "namespace dec_prenorm_pair_cluster {"),
    ('RuntimeCheck(ctas == 1 || ctas == 2 || ctas == 4 || ctas == 8 || ctas == 16, "pnorm: ctas ", ctas);',
     'RuntimeCheck(ctas == 2, "pnorm: cluster ctas ", ctas);'),
    ("sglang::host::LaunchKernel(dim3(ctas), dim3(256), dev, P::BYTES)(k, m, ta, th, tl, mh, ml, sq);",
     "sglang::host::LaunchKernel(dim3(ctas), dim3(256), dev, P::BYTES).enable_cluster(dim3(2, 1, 1))(k, m, ta, th, tl, mh, ml, sq);"),
)


def cluster_source() -> str:
    text = SOURCE
    for old, new in _CLUSTER_EDITS:
        if text.count(old) != 1:
            raise RuntimeError(f"pnorm cluster: edit anchor found {text.count(old)} times: {old[:48]!r}")
        text = text.replace(old, new)
    return text

# king-v7 FFN-PAIR-W / STATS-TAIL text (prenorm_pair.FFN_PAIR_W_ON, prenorm_pair.STATS_TAIL_ON). SOURCE with these edits,
# in its own namespace and JIT module (dec_prenorm_pair_x); SOURCE, CLUSTER_* and their JIT keys are untouched.
# - The kernel gains a template flag kTail. At kTail false the device code is SOURCE's, unchanged (extra parameters are
#   unused). The launcher serves ctas 1 (one CTA) or ctas 2 (one 2-CTA cluster, so both CTAs share one TPC: MegaMoE's
#   73 two-CTA clusters keep the other 73 TPCs, useful_ideas MegaMoE note).
# - kTail true (STATS-TAIL, IDEAS_dec_mhc M2): after the last split's epilogue every CTA joins (bar.sync at ctas 1; the
#   cluster barrier with release / acquire at ctas 2, so the other CTA's split partials are visible), then each thread
#   runs one row's dec_hcsk reduce + Sinkhorn (cuda_sinkhorn.SOURCE reduce_sinkhorn_kernel<., 16, true>) with the same
#   instructions in the same order: per column acc = add(acc, add(hi_s, lo_s)) for s = 0..15 from 0.f, the square sums
#   the same way, then every lane (r, j) of the king's warp is replayed with its own operand order (its butterfly
#   partners are lanes k^2, k^1, k^8, k^4 of the 16 comb values k = 4r + j). Rows are split between the CTAs. Proven at
#   prepare against stock's two DeepGEMM launches + the king's kRows-1 reduce on NaN-filled outputs, bitwise.
# - TpcProbe: a 2-CTA cluster launch that records each CTA's %smid / %cluster_ctarank; prepare requires every cluster on
#   one TPC (sibling SMs 2t, 2t+1) before any ctas-2 band serves.
X_NAME = "mhc/dec_prenorm_pair_x.cuh"
X_NAMESPACE = "dec_prenorm_pair_x"

_X_TAIL_DEVICE = r"""namespace xtail {

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

// One row of dec_hcsk::reduce_sinkhorn_kernel<kRows, 16, true>: the slice sums of its s_sum[rr][c] thread, then lane
// k = 4r + j of its warp 0 for every k (lanes 16..31 only repeat lanes 0..15 and store nothing).
__device__ __forceinline__ void row_stats(const uint32_t shape_m, const uint32_t row, const float* __restrict__ mix_hi,
                                          const float* __restrict__ mix_lo, const float* __restrict__ sqr_sum,
                                          const float* __restrict__ scale, const float* __restrict__ base,
                                          float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb,
                                          const float inv_k, const float rms_eps, const int iters, const float eps) {
  float acc[kN];
#pragma unroll
  for (uint32_t c = 0; c < kN; ++c) acc[c] = 0.f;
  float sq = 0.f;
#pragma unroll 4
  for (uint32_t s = 0; s < kSplits; ++s) {
    const size_t at = (size_t(s) * shape_m + row) * kN;
    const float4* ph = reinterpret_cast<const float4*>(mix_hi + at);
    const float4* pl = reinterpret_cast<const float4*>(mix_lo + at);
#pragma unroll
    for (uint32_t q = 0; q < kN / 4; ++q) {
      const float4 h = __ldcg(ph + q);
      const float4 l = __ldcg(pl + q);
      acc[4 * q + 0] = add_f(acc[4 * q + 0], add_f(h.x, l.x));
      acc[4 * q + 1] = add_f(acc[4 * q + 1], add_f(h.y, l.y));
      acc[4 * q + 2] = add_f(acc[4 * q + 2], add_f(h.z, l.z));
      acc[4 * q + 3] = add_f(acc[4 * q + 3], add_f(h.w, l.w));
    }
    sq = add_f(sq, __ldcg(sqr_sum + size_t(s) * shape_m + row));
  }
  const float log2e = __int_as_float(0x3FB8AA3B);
  const float neg_log2e = __int_as_float(0xBFB8AA3B);
  const float rsq = div_full(1.0f, sqrt_approx_ftz(fma_rn(inv_k, sq, rms_eps)));
  const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    float z = fma_rn(mul_f(acc[j], rsq), s0, base[j]);
    float sig = div_full(1.0f, add_f(ex2_approx(mul_f(z, neg_log2e)), 1.0f));
    const float pre_v = add_f(sig, eps);
    z = fma_rn(mul_f(acc[4 + j], rsq), s1, base[4 + j]);
    sig = div_full(1.0f, add_f(ex2_approx(mul_f(z, neg_log2e)), 1.0f));
    const float post_v = add_f(sig, sig);
    pre[size_t(row) * 4 + j] = pre_v;
    post[size_t(row) * 4 + j] = post_v;
  }
  float c[16], t[16], u[16], e[16];
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) c[k] = fma_rn(mul_f(acc[8 + k], rsq), s2, base[8 + k]);
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) t[k] = max_f(c[k], c[k ^ 2]);
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) u[k] = max_f(t[k], t[k ^ 1]);
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) e[k] = ex2_approx(mul_f(sub_f(c[k], u[k]), log2e));
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) t[k] = add_f(e[k], e[k ^ 2]);
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) u[k] = add_f(t[k], t[k ^ 1]);
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) c[k] = add_f(div_full(e[k], u[k]), eps);
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) t[k] = add_f(c[k], c[k ^ 8]);
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) u[k] = add_f(t[k], t[k ^ 4]);
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) c[k] = div_full(c[k], add_f(u[k], eps));
#pragma unroll 1
  for (int it = 1; it < iters; ++it) {
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k) t[k] = add_f(c[k], c[k ^ 2]);
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k) u[k] = add_f(t[k], t[k ^ 1]);
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k) c[k] = div_full(c[k], add_f(u[k], eps));
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k) t[k] = add_f(c[k], c[k ^ 8]);
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k) u[k] = add_f(t[k], t[k ^ 4]);
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k) c[k] = div_full(c[k], add_f(u[k], eps));
  }
#pragma unroll
  for (uint32_t k = 0; k < 16; ++k) comb[size_t(row) * 16 + k] = c[k];
}

}  // namespace xtail

"""

_X_TAIL_CALL = r"""  __syncthreads();
  if (warp_idx == 2) cute::TMEM::Allocator1Sm().free(0, P::TMEM_COLS);
  if constexpr (kTail) {
    if (G > 1) {
      uint32_t n_cta;
      asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(n_cta));
      if (n_cta != G) __trap();
      asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    const uint32_t per = (shape_m + G - 1) / G;
    const uint32_t row_lo = g * per;
    const uint32_t row_hi = row_lo + per < shape_m ? row_lo + per : shape_m;
    for (uint32_t row = row_lo + threadIdx.x; row < row_hi; row += blockDim.x)
      xtail::row_stats(shape_m, row, mix_hi, mix_lo, sqr_sum, hc_scale, hc_base, pre_out, post_out, comb_out, inv_k,
                       rms_eps, iters, hc_eps);
  }
#endif
}"""

_X_HOST = r"""template <uint32_t kNumRB, bool kTail>
void launch(const void* x, const void* hi, const void* lo, float* mh, float* ml, float* sq, uint32_t m, uint32_t ctas,
            DLDevice dev, const float* scale, const float* base, float* pre, float* post, float* comb, float inv_k,
            float rms_eps, int iters, float eps) {
  using P = Plan<kNumRB>;
  auto k = prenorm_pair_kernel<kNumRB, kTail>;
  static const bool attr = [&] {
    sglang::host::RuntimeCheck(
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(P::BYTES)) == cudaSuccess,
        "pnorm-x: shared memory attribute");
    return true;
  }();
  (void)attr;
  const CUtensorMap ta = encode(x, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, m, BK, BM);
  const CUtensorMap th = encode(hi, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 4, kN, 128 / 4, BN);
  const CUtensorMap tl = encode(lo, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 4, kN, 128 / 4, BN);
  if (ctas == 2) {
    sglang::host::LaunchKernel(dim3(2), dim3(256), dev, P::BYTES).enable_cluster(dim3(2, 1, 1))(
        k, m, ta, th, tl, mh, ml, sq, scale, base, pre, post, comb, inv_k, rms_eps, iters, eps);
  } else {
    sglang::host::LaunchKernel(dim3(1), dim3(256), dev, P::BYTES)(
        k, m, ta, th, tl, mh, ml, sq, scale, base, pre, post, comb, inv_k, rms_eps, iters, eps);
  }
}

template <bool kTail>
void dispatch(const void* x, const void* hi, const void* lo, float* mh, float* ml, float* s, uint32_t m, uint32_t ctas,
              DLDevice d, const float* scale, const float* base, float* pre, float* post, float* comb, float inv_k,
              float rms_eps, int iters, float eps) {
  switch ((m + BM - 1) / BM) {
    case 1: launch<1, kTail>(x, hi, lo, mh, ml, s, m, ctas, d, scale, base, pre, post, comb, inv_k, rms_eps, iters, eps); break;
    case 2: launch<2, kTail>(x, hi, lo, mh, ml, s, m, ctas, d, scale, base, pre, post, comb, inv_k, rms_eps, iters, eps); break;
    case 3: launch<3, kTail>(x, hi, lo, mh, ml, s, m, ctas, d, scale, base, pre, post, comb, inv_k, rms_eps, iters, eps); break;
    case 4: launch<4, kTail>(x, hi, lo, mh, ml, s, m, ctas, d, scale, base, pre, post, comb, inv_k, rms_eps, iters, eps); break;
    default: launch<5, kTail>(x, hi, lo, mh, ml, s, m, ctas, d, scale, base, pre, post, comb, inv_k, rms_eps, iters, eps); break;
  }
}

struct PrenormPair {
  static uint32_t check(const tvm::ffi::TensorView& x, const tvm::ffi::TensorView& hi, const tvm::ffi::TensorView& lo,
                        const tvm::ffi::TensorView& mix_hi, const tvm::ffi::TensorView& mix_lo,
                        const tvm::ffi::TensorView& sq, const int64_t ctas, sglang::host::SymbolicSize& M,
                        sglang::host::SymbolicDevice& dev) {
    using namespace sglang::host;
    TensorMatcher({M, kK}).with_strides({kK, 1}).with_dtype<sglang::bf16_t>().with_device(dev).verify(x);
    TensorMatcher({kN, kK}).with_strides({kK, 1}).with_dtype<float>().with_device(dev).verify(hi);
    TensorMatcher({kN, kK}).with_strides({kK, 1}).with_dtype<float>().with_device(dev).verify(lo);
    TensorMatcher({kSplits, M, kN}).with_dtype<float>().with_device(dev).verify(mix_hi);
    TensorMatcher({kSplits, M, kN}).with_dtype<float>().with_device(dev).verify(mix_lo);
    TensorMatcher({kSplits, M}).with_dtype<float>().with_device(dev).verify(sq);
    const int64_t rows = M.unwrap();
    RuntimeCheck(rows >= 32 && rows <= 5 * int64_t(BM), "pnorm-x: rows ", rows);
    RuntimeCheck(ctas == 1 || ctas == 2, "pnorm-x: ctas ", ctas);
    for (const auto* t : {&mix_hi, &mix_lo}) {
      RuntimeCheck(t->stride(0) == rows * kN && t->stride(1) == kN && t->stride(2) == 1, "pnorm-x: mix layout");
    }
    RuntimeCheck(sq.stride(0) == rows && sq.stride(1) == 1, "pnorm-x: sq layout");
    for (const void* p : {x.data_ptr(), hi.data_ptr(), lo.data_ptr(), mix_hi.data_ptr(), mix_lo.data_ptr()}) {
      RuntimeCheck(reinterpret_cast<uintptr_t>(p) % 16 == 0, "pnorm-x: 16-byte alignment");
    }
    return uint32_t(rows);
  }

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView hi, const tvm::ffi::TensorView lo,
                  const tvm::ffi::TensorView mix_hi, const tvm::ffi::TensorView mix_lo, const tvm::ffi::TensorView sq,
                  const int64_t ctas) {
    using namespace sglang::host;
    auto M = SymbolicSize{"rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    const uint32_t m = check(x, hi, lo, mix_hi, mix_lo, sq, ctas, M, dev);
    dispatch<false>(x.data_ptr(), hi.data_ptr(), lo.data_ptr(), static_cast<float*>(mix_hi.data_ptr()),
                    static_cast<float*>(mix_lo.data_ptr()), static_cast<float*>(sq.data_ptr()), m, uint32_t(ctas),
                    dev.unwrap(), nullptr, nullptr, nullptr, nullptr, nullptr, 0.f, 0.f, 1, 0.f);
  }

  static void run_tail(const tvm::ffi::TensorView x, const tvm::ffi::TensorView hi, const tvm::ffi::TensorView lo,
                       const tvm::ffi::TensorView mix_hi, const tvm::ffi::TensorView mix_lo,
                       const tvm::ffi::TensorView sq, const tvm::ffi::TensorView scale,
                       const tvm::ffi::TensorView base, const tvm::ffi::TensorView pre,
                       const tvm::ffi::TensorView post, const tvm::ffi::TensorView comb, float inv_k, float rms_eps,
                       int64_t iters, float eps, const int64_t ctas) {
    using namespace sglang::host;
    auto M = SymbolicSize{"rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    const uint32_t m = check(x, hi, lo, mix_hi, mix_lo, sq, ctas, M, dev);
    TensorMatcher({3}).with_strides({1}).with_dtype<float>().with_device(dev).verify(scale);
    TensorMatcher({kN}).with_strides({1}).with_dtype<float>().with_device(dev).verify(base);
    TensorMatcher({M, 4}).with_strides({4, 1}).with_dtype<float>().with_device(dev).verify(pre).verify(post);
    TensorMatcher({M, 4, 4}).with_strides({16, 4, 1}).with_dtype<float>().with_device(dev).verify(comb);
    RuntimeCheck(iters >= 1 && iters <= 1000, "pnorm-x: sinkhorn iters ", iters);
    dispatch<true>(x.data_ptr(), hi.data_ptr(), lo.data_ptr(), static_cast<float*>(mix_hi.data_ptr()),
                   static_cast<float*>(mix_lo.data_ptr()), static_cast<float*>(sq.data_ptr()), m, uint32_t(ctas),
                   dev.unwrap(), static_cast<const float*>(scale.data_ptr()), static_cast<const float*>(base.data_ptr()),
                   static_cast<float*>(pre.data_ptr()), static_cast<float*>(post.data_ptr()),
                   static_cast<float*>(comb.data_ptr()), inv_k, rms_eps, int(iters), eps);
  }
};

__global__ void tpc_probe_kernel(int* __restrict__ out, int* __restrict__ counter, const int expected,
                                 const unsigned long long timeout_ns) {
  if (threadIdx.x == 0) {
    uint32_t smid, rank, n_cta;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(n_cta));
    atomicAdd(counter, 1);
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    int seen = 0;
    do {  // bounded: every CTA exits after timeout_ns whether or not the others became resident
      seen = atomicAdd(counter, 0);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (seen < expected && t - t0 < timeout_ns);
    out[4 * blockIdx.x + 0] = int(smid);
    out[4 * blockIdx.x + 1] = int(rank);
    out[4 * blockIdx.x + 2] = int(n_cta);
    out[4 * blockIdx.x + 3] = seen >= expected ? 1 : 0;
  }
}

struct TpcProbe {
  // out: int32 [4 * ctas] (smid, cluster rank, cluster size, all CTAs co-resident); counter: int32 [1], zero.
  static void run(const tvm::ffi::TensorView out, const tvm::ffi::TensorView counter, const int64_t smem,
                  const int64_t timeout_ns) {
    using namespace sglang::host;
    auto n = SymbolicSize{"n"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({n}).with_strides({1}).with_dtype<int32_t>().with_device(dev).verify(out);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(dev).verify(counter);
    const int64_t total = n.unwrap();
    RuntimeCheck(total >= 8 && total % 8 == 0 && total <= 8 * 512, "pnorm-x probe: size ", total);
    RuntimeCheck(smem >= 0 && smem <= 200 * 1024, "pnorm-x probe: smem ", smem);
    RuntimeCheck(timeout_ns > 0 && timeout_ns <= 100000000, "pnorm-x probe: timeout ", timeout_ns);
    static const bool attr = [] {
      RuntimeCheck(cudaFuncSetAttribute(tpc_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024) ==
                       cudaSuccess,
                   "pnorm-x probe: shared memory attribute");
      return true;
    }();
    (void)attr;
    const uint32_t ctas = uint32_t(total / 4);
    LaunchKernel(dim3(ctas), dim3(32), dev.unwrap(), std::size_t(smem))
        .enable_cluster(dim3(2, 1, 1))(tpc_probe_kernel, static_cast<int*>(out.data_ptr()),
                                       static_cast<int*>(counter.data_ptr()), int(ctas),
                                       static_cast<unsigned long long>(timeout_ns));
  }
};

}
"""

_X_EDITS = (
    ("namespace dec_prenorm_pair {", "namespace dec_prenorm_pair_x {"),
    ("template <uint32_t kNumRB>\nstruct Plan {", _X_TAIL_DEVICE + "template <uint32_t kNumRB>\nstruct Plan {"),
    ("template <uint32_t kNumRB>\n__global__ void __launch_bounds__(256, 1)\nprenorm_pair_kernel(",
     "template <uint32_t kNumRB, bool kTail>\n__global__ void __launch_bounds__(256, 1)\nprenorm_pair_kernel("),
    ("          float* __restrict__ mix_hi, float* __restrict__ mix_lo, float* __restrict__ sqr_sum) {",
     "          float* __restrict__ mix_hi, float* __restrict__ mix_lo, float* __restrict__ sqr_sum,\n"
     "          const float* __restrict__ hc_scale, const float* __restrict__ hc_base, float* __restrict__ pre_out,\n"
     "          float* __restrict__ post_out, float* __restrict__ comb_out, const float inv_k, const float rms_eps,\n"
     "          const int iters, const float hc_eps) {"),
    ("  __syncthreads();\n  if (warp_idx == 2) cute::TMEM::Allocator1Sm().free(0, P::TMEM_COLS);\n#endif\n}",
     _X_TAIL_CALL),
)
_X_HOST_ANCHOR = "template <uint32_t kNumRB>\nvoid launch("


def x_source() -> str:
    text = SOURCE
    for old, new in _X_EDITS:
        if text.count(old) != 1:
            raise RuntimeError(f"pnorm-x: edit anchor found {text.count(old)} times: {old[:48]!r}")
        text = text.replace(old, new)
    if text.count(_X_HOST_ANCHOR) != 1:
        raise RuntimeError("pnorm-x: host anchor not found once")
    text = text[:text.index(_X_HOST_ANCHOR)] + _X_HOST
    if "namespace dec_prenorm_pair {" in text or text.count("namespace dec_prenorm_pair_x {") != 1:
        raise RuntimeError("pnorm-x: namespace edit failed")
    return text
