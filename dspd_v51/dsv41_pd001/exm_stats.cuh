// EXM stats kernels (dspd/exm.py; dspd-w33e, SHIPPED OFF; NEVER run on a GPU yet): the side-stream half of one shifted
// mHC boundary: the NEXT layer's (pre, post, comb) from the new residual R' that exm_crit.cuh wrote, bit for bit as
// stock's deep_gemm.mega_mhc computes them (struct/w33_exm/EXM_SPEC.md section 6; route (a) of EXM_FEAS.md 1.3).
// Two launches on the engine's hc_stats_stream, stream-ordered (FIX of struct/w33_exm/REVIEW_exm_runtime.md F1 / F5):
//
// exm_stats_kernel: non-persistent CTAs of 128 threads (one warpgroup: warp q owns TMEM lanes 32q.. = tile rows
// 16q..16q+15) run tasks task = blockIdx.x, + gridDim.x, ... with task = m_block * S + split (stock's task index).
// gridDim = all tasks (BURST) or a few CTAs (TRICKLE). Each task writes its partials to the workspace; no CTA waits on
// another and no CTA runs a Mix. The only blocking instruction is tcgen05.alloc (256 columns) on the CTA's own SM; the
// launch's dynamic shared memory keeps the CTA off SMs that hold a TMEM-using engine kernel, and the columns go back
// right after the CTA's last drain (before its partial stores).
//
// Per task (m-block mb, split s), exactly stock's per-task arithmetic (written fresh; DeepGEMM helpers included):
//   tiles   per k-block kb of KB(s) ascending: fn[n, r * 5120 + kb * 64 + 32 j + k] into the byte image a TMA
//           SWIZZLE_128B box (32, 24, 4) leaves (16-byte chunk k / 4 of row n at chunk (k / 4) ^ (n % 8), every
//           [j][r] sub-tile 1024-byte aligned; impls:224-232, layout:129); R' rows 64 mb .. 64 mb + 63 (rows >= T as
//           +0, stock's TMA zero fill) into a padded shared tile
//   A       atom u = 0..7 of the k-block: lane (q, l) holds rows {16q + l/4, +8}, column pair c = l & 3, i.e. hidden
//           kb * 64 + 8u + 2c, +1, of routes 0..3, stored with cute's SM100_TMEM_STORE_16dp256b4x from the same
//           registers in the same order as stock (impls:460-461,510-515); 4 atoms per half k-block (columns 0..127)
//   hc      the same values feed acc[row][u % 2] = __ffma2_rn(pv, pv, acc) for routes 0..3 (impls:484,501): per (row,
//           parity, column pair) the operand sequence is stock's post worker p's (k-block, its atoms ascending, route)
//   MMA     one thread of warp 0 issues SM100_MMA_TF32_TS<tf32, tf32, f32, 64, 24, K, K> with the idesc of cute's
//           make_runtime_instr_desc, B = make_umma_desc<K, 24, 32, 128>(fn[u * 8 / 32][r], 0, u * 8 % 32), in stock's
//           order (kb, u, route), accumulate = not the task's first instruction (impls:233-285)
//   drain   SM100_TMEM_LOAD_16dp256b2x / 1x, stock's (row, output) map, rows >= T written as 0 (impls:594-625)
//   hc part (L0 + L2) + (L1 + L3) per parity (warp_reduce_sum<4>), then (0 + W0) + W1 (impls:545-546,634-642)
//
// exm_mix_kernel<S>: launched after the stats kernel on the same stream (plain stream order, no PDL: its CTAs must not
// sit resident through the stats kernel), one warp per row, kMixRows rows per CTA; S is a template constant, so the
// split loads unroll as stock's run_mix_task does (epilogue:29-41). Per row (epilogue:16-101): lane-strided hc split
// sum from +0, then warp_reduce_sum<32>; GEMM split sums 0..S-1 from +0 per output lane; rsqrtf(K1); affine K2; sigmoid
// fast_rcp(1 + __expf(-a)) (+ pre eps / x post scale); Sinkhorn on lanes 8..23 with warp_reduce_max<4> / sum<4>, K3,
// column sums xor 4 then xor 24, iters - 1 further row (K5) / column (K6) steps; stores new_prev_mix[t * 4 + o],
// new_post_mix[t * 4 + o], new_comb[t * 16 + 4 i + o]. Rows are independent; no ticket, no inter-CTA order.
#pragma once

#include "exm_common.cuh"

#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace sglang {
namespace exm {

inline constexpr uint32_t kStatsThreads = 128;  // one warpgroup: the M = 64 tile's four TMEM lane quarters
inline constexpr uint32_t kTmemCols = 256;      // A: 4 atoms x 32 columns (0..127); D: 24 columns at 128
inline constexpr uint32_t kAccCol = 128;
inline constexpr uint32_t kFnAtomK = 32;        // floats per 128-byte swizzle row

inline constexpr uint32_t kMixRows = 8;         // mix kernel: one warp per row, 8 rows per CTA
inline constexpr uint32_t kMixThreads = kMixRows * 32;

struct StatsParams {
  const bf16_t* new_residual;  // [T, 4, 5120]: R', written by exm_crit on the main stream before the fork
  const float* fn;             // [24, 20480]
  float* gemm_partials;        // [m-blocks * S][64][24]   (stock's Workspace layout, task-major)
  float* hc_partials;          // [m-blocks * S][64]
  uint32_t num_tokens;
  uint32_t num_splits;
  uint32_t num_m_blocks;
};

struct MixParams {
  const float* gemm_partials;  // written by exm_stats_kernel, the previous launch on this stream
  const float* hc_partials;
  const float* scales;         // [3]
  const float* bases;          // [24]
  float* new_pre;              // [T, 4]
  float* new_post;             // [T, 4]
  float* new_comb;             // [T, 4, 4]
  uint32_t num_tokens;
  uint32_t sinkhorn_iters;
  float hc_norm_eps;
  float hc_pre_eps;
  float hc_post_scale;
  float sinkhorn_eps;
};

struct StatsSmem {
  float fn[2][kRoutes][kOutputs * kFnAtomK];      // 24 KB; [j][r] sub-tiles of 3072 B, 1024-aligned
  uint32_t res[kRoutes][kBlockM][32];             // 32 KB; bf16 pairs, 16-byte chunk u of row m at u ^ (m & 7)
  cutlass::arch::ClusterBarrier mma_bar;          // MMA completion (tcgen05.commit)
  uint32_t tmem_base;
};
inline constexpr uint32_t kStatsSmemUsed = sizeof(StatsSmem) + 1024;  // + the manual 1024-byte alignment

SGL_DEVICE float col_sum(const float v) {  // epilogue:71-76: adjacent rows (xor 4), then the two row pairs (xor 24)
  const float a = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
  return __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 24));
}

// one warp: the Mix of row t (stock's run_mix_task, shifted, epilogue:16-101), S a template constant as in stock
template <uint32_t kNumSplits>
SGL_DEVICE void mix_row(const MixParams& p, const uint32_t t, const uint32_t lane) {
  using deep_gemm::math::fast_rcp;
  const uint32_t row = t % kBlockM;
  const uint32_t first_task = t / kBlockM * kNumSplits;
  float hc = 0.0f;
#pragma unroll
  for (uint32_t k = lane; k < kNumSplits; k += 32)
    hc = __fadd_rn(hc, __ldcg(p.hc_partials + (first_task + k) * kBlockM + row));
  hc = deep_gemm::math::warp_reduce_sum(hc);
  const uint32_t n = min(lane, kOutputs - 1);
  float out = 0.0f;
#pragma unroll
  for (uint32_t k = 0; k < kNumSplits; ++k)
    out = __fadd_rn(out, __ldcg(p.gemm_partials + (first_task + k) * kGemmPartial + row * kOutputs + n));
  out = lane < kOutputs ? out : 0.0f;
  const float rr = rsqrtf(k1_rms_arg(hc, kInvHcHidden, p.hc_norm_eps));
  float mix = 0.0f;
  if (lane < kOutputs) {
    const uint32_t si = lane < kRoutes ? 0u : lane < 2 * kRoutes ? 1u : 2u;
    const float aff = k2_affine(out, rr, p.scales[si], p.bases[lane]);
    if (lane < 2 * kRoutes) {
      const float sig = fast_rcp(__fadd_rn(1.0f, __expf(-aff)));
      mix = lane < kRoutes ? __fadd_rn(sig, p.hc_pre_eps) : __fmul_rn(sig, p.hc_post_scale);
    } else {
      mix = aff;
    }
  }
  __syncwarp();
  // Sinkhorn: comb lanes 8..23 (row i = (lane - 8) / 4, column o = lane % 4); dead lanes stay in their own groups
  const bool comb_lane = lane >= 2 * kRoutes && lane < kOutputs;
  float v = comb_lane ? mix : 0.0f;
  const float row_max = deep_gemm::math::warp_reduce_max<kRoutes>(v);
  v = __expf(__fsub_rn(v, row_max));
  const float row_sum = deep_gemm::math::warp_reduce_sum<kRoutes>(v);
  v = k3_row_norm(v, fast_rcp(row_sum), p.sinkhorn_eps);
  // from here every value is a product a * b (stock's `value *= rcp`): the K5 / K6 sums see its factors
  float a = v, b = fast_rcp(__fadd_rn(col_sum(v), p.sinkhorn_eps));
  v = __fmul_rn(a, b);
#pragma unroll 1
  for (uint32_t it = 1; it < p.sinkhorn_iters; ++it) {
    b = fast_rcp(__fadd_rn(k5_row_sum(a, b, v), p.sinkhorn_eps));
    a = v;
    v = __fmul_rn(a, b);
    b = fast_rcp(__fadd_rn(k6_col_sum(a, b, v), p.sinkhorn_eps));
    a = v;
    v = __fmul_rn(a, b);
  }
  if (lane < kRoutes) p.new_pre[t * kRoutes + lane] = mix;
  else if (lane < 2 * kRoutes) p.new_post[t * kRoutes + (lane & 3u)] = mix;
  else if (comb_lane) p.new_comb[t * kRoutes * kRoutes + lane - 2 * kRoutes] = v;
}

template <uint32_t kNumSplits>
__global__ void __launch_bounds__(kMixThreads, 1) exm_mix_kernel(const __grid_constant__ MixParams p) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t t = blockIdx.x * kMixRows + (threadIdx.x >> 5);
  if (t < p.num_tokens) mix_row<kNumSplits>(p, t, lane);  // warp-uniform: a whole warp takes or skips its row
}

__global__ void __launch_bounds__(kStatsThreads, 1) exm_stats_kernel(const __grid_constant__ StatsParams p) {
  using umma_t = cute::SM100_MMA_TF32_TS<cutlass::tfloat32_t, cutlass::tfloat32_t, float, kBlockM, kOutputs,
                                         cute::UMMA::Major::K, cute::UMMA::Major::K>;
  extern __shared__ uint8_t smem_raw[];
  auto& sm = *reinterpret_cast<StatsSmem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~uintptr_t{1023u});
  const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
  const uint32_t S = p.num_splits, T = p.num_tokens;
  const uint32_t num_tasks = p.num_m_blocks * S;
  const auto idesc = cute::UMMA::make_runtime_instr_desc<cutlass::tfloat32_t, cutlass::tfloat32_t, float, kBlockM,
                                                         kOutputs, cute::UMMA::Major::K, cute::UMMA::Major::K>();

  if (tid == 0) {
    sm.mma_bar.init(1);
    cutlass::arch::fence_barrier_init();
  }
  if (warp == 0) {
    cute::TMEM::Allocator1Sm().allocate(kTmemCols, &sm.tmem_base);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  deep_gemm::ptx::tcgen05_before_thread_sync();
  __syncthreads();
  deep_gemm::ptx::tcgen05_after_thread_sync();
  const uint32_t tmem = sm.tmem_base;  // the allocator's base (DeepGEMM #480 class: never assume column 0)
  uint32_t phase = 0;
  bool freed = false;  // CTA-uniform: the task loop is uniform
  const uint32_t c = lane & 3u;
  const uint32_t rows[2] = {warp * 16 + lane / 4, warp * 16 + lane / 4 + 8};

  for (uint32_t task = blockIdx.x; task < num_tasks; task += gridDim.x) {
    const uint32_t mb = task / S, s = task % S;
    const uint32_t kb0 = split_first_kb(s, S), nkb = split_num_kb(s, S);
    float2 acc[2][2] = {};  // [row][parity]: stock's per-task hc_norm_sqr_sums, one per post worker
    for (uint32_t kbi = 0; kbi < nkb; ++kbi) {
      const uint32_t kb = kb0 + kbi;
      // fn: 2 atoms x 4 routes x 24 rows x 8 chunks of 16 B = 1536 float4, as TMA SWIZZLE_128B places them
      for (uint32_t idx = tid; idx < 2 * kRoutes * kOutputs * 8; idx += kStatsThreads) {
        const uint32_t chunk = idx & 7u, n = (idx >> 3) % kOutputs, jr = (idx >> 3) / kOutputs;
        const uint32_t r = jr & 3u, j = jr >> 2;
        const float4 v = *reinterpret_cast<const float4*>(p.fn + static_cast<uint64_t>(n) * kRoutes * kHidden +
                                                          r * kHidden + kb * kBlockK + j * kFnAtomK + chunk * 4);
        *reinterpret_cast<float4*>(&sm.fn[j][r][n * kFnAtomK + ((chunk ^ (n & 7u)) * 4)]) = v;
      }
      // R': 4 routes x 64 rows x 8 chunks of 8 bf16 = 2048 uint4; rows >= T as +0
      for (uint32_t idx = tid; idx < kRoutes * kBlockM * 8; idx += kStatsThreads) {
        const uint32_t chunk = idx & 7u, m = (idx >> 3) & (kBlockM - 1), r = idx >> 9;
        const uint32_t tr = mb * kBlockM + m;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (tr < T)
          v = *reinterpret_cast<const uint4*>(p.new_residual + (static_cast<uint64_t>(tr) * kRoutes + r) * kHidden +
                                              kb * kBlockK + chunk * 8);
        *reinterpret_cast<uint4*>(&sm.res[r][m][(chunk ^ (m & 7u)) * 4]) = v;
      }
      cutlass::arch::fence_view_async_shared();  // generic-proxy fn stores -> the tensor core's async proxy
      __syncthreads();
      for (uint32_t half = 0; half < 2; ++half) {
#pragma unroll
        for (uint32_t uu = 0; uu < 4; ++uu) {
          const uint32_t u = half * 4 + uu;
          float2 pv[kRoutes][2];  // stock's post_values[route][row]
#pragma unroll
          for (uint32_t k = 0; k < 2; ++k) {
            const uint32_t m = rows[k];
#pragma unroll
            for (uint32_t r = 0; r < kRoutes; ++r) {
              pv[r][k] = __bfloat1622float2(bf2_of(sm.res[r][m][((u ^ (m & 7u)) << 2) | c]));
              acc[k][u & 1u] = __ffma2_rn(pv[r][k], pv[r][k], acc[k][u & 1u]);
            }
          }
          const auto tv = reinterpret_cast<const uint32_t(*)[4]>(pv);
          cute::SM100_TMEM_STORE_16dp256b4x::copy(tv[0][0], tv[0][1], tv[0][2], tv[0][3], tv[1][0], tv[1][1],
                                                  tv[1][2], tv[1][3], tv[2][0], tv[2][1], tv[2][2], tv[2][3],
                                                  tv[3][0], tv[3][1], tv[3][2], tv[3][3], tmem + uu * 32);
        }
        cutlass::arch::fence_view_async_tmem_store();
        deep_gemm::ptx::tcgen05_before_thread_sync();
        __syncthreads();
        if (warp == 0) {
          deep_gemm::ptx::tcgen05_after_thread_sync();
#pragma unroll
          for (uint32_t uu = 0; uu < 4; ++uu) {
            const uint32_t u = half * 4 + uu;
#pragma unroll
            for (uint32_t r = 0; r < kRoutes; ++r) {
              const auto b_desc = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, kOutputs, kFnAtomK, 128>(
                  sm.fn[u * 8 / kFnAtomK][r], 0, u * 8 % kFnAtomK);
              umma_t::fma(tmem + uu * 32 + r * 8, b_desc, tmem + kAccCol, (kbi > 0 || u > 0 || r > 0) ? 1u : 0u,
                          idesc);
            }
          }
          cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&sm.mma_bar));
        }
        sm.mma_bar.wait(phase);
        phase ^= 1u;
        deep_gemm::ptx::tcgen05_after_thread_sync();
      }
    }

    // GEMM partial: drain D (stock's workspace epilogue map), rows >= T as 0
    uint32_t v[12];
    cute::SM100_TMEM_LOAD_16dp256b2x::copy(tmem + kAccCol, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem + kAccCol + 16, v[8], v[9], v[10], v[11]);
    cutlass::arch::fence_view_async_tmem_load();
    deep_gemm::ptx::tcgen05_before_thread_sync();
    if (task + gridDim.x >= num_tasks) {  // the CTA's last task: the columns go back before the partial stores
      __syncthreads();
      if (warp == 0) cute::TMEM::Allocator1Sm().free(tmem, kTmemCols);
      freed = true;
    }
    float* gp = p.gemm_partials + static_cast<uint64_t>(task) * kGemmPartial;
    const bool live0 = mb * kBlockM + rows[0] < T, live1 = mb * kBlockM + rows[1] < T;
#pragma unroll
    for (uint32_t o = 0; o < kOutputs; o += 8) {
      const uint32_t vi = o / 2;
      const float2 a = live0 ? make_float2(__uint_as_float(v[vi]), __uint_as_float(v[vi + 1])) : make_float2(0.f, 0.f);
      const float2 b = live1 ? make_float2(__uint_as_float(v[vi + 2]), __uint_as_float(v[vi + 3]))
                             : make_float2(0.f, 0.f);
      *reinterpret_cast<float2*>(gp + rows[0] * kOutputs + o + c * 2) = a;
      *reinterpret_cast<float2*>(gp + rows[1] * kOutputs + o + c * 2) = b;
    }
    // hc partial per row: per parity (L0 + L2) + (L1 + L3), then (0 + W0) + W1
#pragma unroll
    for (uint32_t k = 0; k < 2; ++k) {
      const float w0 = deep_gemm::math::warp_reduce_sum<4>(__fadd_rn(acc[k][0].x, acc[k][0].y));
      const float w1 = deep_gemm::math::warp_reduce_sum<4>(__fadd_rn(acc[k][1].x, acc[k][1].y));
      if (c == 0) p.hc_partials[static_cast<uint64_t>(task) * kBlockM + rows[k]] = __fadd_rn(__fadd_rn(0.0f, w0), w1);
    }
  }
  if (!freed) {  // a CTA without a task (never launched so: grid <= tasks)
    __syncthreads();
    if (warp == 0) cute::TMEM::Allocator1Sm().free(tmem, kTmemCols);
  }
}

// Host side: run(new_residual, fn, scales, bases, new_pre, new_post, new_comb, gemm_scratch, hc_scratch, S, num_ctas
// (<= 0: one CTA per task), smem_bytes, hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, iters, stream): the
// stats kernel, then the mix kernel of S (40 or 27), both on `stream` (the engine's hc_stats_stream, passed
// explicitly by exm.py).
struct ExmStats {
  static void run(const tvm::ffi::TensorView new_residual, const tvm::ffi::TensorView fn,
                  const tvm::ffi::TensorView scales, const tvm::ffi::TensorView bases,
                  const tvm::ffi::TensorView new_pre, const tvm::ffi::TensorView new_post,
                  const tvm::ffi::TensorView new_comb, const tvm::ffi::TensorView gemm_scratch,
                  const tvm::ffi::TensorView hc_scratch, const int64_t num_splits, const int64_t num_ctas,
                  const int64_t smem_bytes, const double hc_norm_eps, const double hc_pre_eps,
                  const double hc_post_scale, const double sinkhorn_eps, const int64_t sinkhorn_iters,
                  const int64_t stream) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto G = SymbolicSize{"gemm_scratch"};
    auto H = SymbolicSize{"hc_scratch"};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, kRoutes, kHidden}).with_dtype<bf16_t>().with_device(device).verify(new_residual);
    TensorMatcher({kOutputs, kRoutes * kHidden}).with_dtype<float>().with_device(device).verify(fn);
    TensorMatcher({3}).with_dtype<float>().with_device(device).verify(scales);
    TensorMatcher({kOutputs}).with_dtype<float>().with_device(device).verify(bases);
    TensorMatcher({M, kRoutes}).with_dtype<float>().with_device(device).verify(new_pre);
    TensorMatcher({M, kRoutes}).with_dtype<float>().with_device(device).verify(new_post);
    TensorMatcher({M, kRoutes, kRoutes}).with_dtype<float>().with_device(device).verify(new_comb);
    TensorMatcher({G}).with_dtype<float>().with_device(device).verify(gemm_scratch);
    TensorMatcher({H}).with_dtype<float>().with_device(device).verify(hc_scratch);
    const auto rows = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(rows > 0 && (num_splits == 40 || num_splits == 27), "EXM: rows / S (40 and 27 are instantiated)");
    RuntimeCheck(sinkhorn_iters >= 1, "EXM: sinkhorn iterations");
    const uint32_t m_blocks = (rows + kBlockM - 1) / kBlockM;
    const uint64_t tasks = static_cast<uint64_t>(m_blocks) * static_cast<uint64_t>(num_splits);
    RuntimeCheck(G.unwrap() >= static_cast<int64_t>(tasks * kGemmPartial) &&
                     H.unwrap() >= static_cast<int64_t>(tasks * kBlockM),
                 "EXM: workspace too small");
    for (const auto& t : {new_residual, fn, gemm_scratch, hc_scratch})
      RuntimeCheck(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16u == 0, "EXM: 16-byte alignment");
    const uint32_t grid = num_ctas <= 0 ? static_cast<uint32_t>(tasks)
                                        : static_cast<uint32_t>(std::min<uint64_t>(num_ctas, tasks));
    const auto smem = static_cast<uint32_t>(std::max<int64_t>(smem_bytes, kStatsSmemUsed));
    static uint32_t configured = 0;  // one attribute call per size, at the first (eager) call of that size
    if (smem > configured) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(exm_stats_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      configured = smem;
    }
    const auto sp = StatsParams{
        .new_residual = static_cast<const bf16_t*>(new_residual.data_ptr()),
        .fn = static_cast<const float*>(fn.data_ptr()),
        .gemm_partials = static_cast<float*>(gemm_scratch.data_ptr()),
        .hc_partials = static_cast<float*>(hc_scratch.data_ptr()),
        .num_tokens = rows,
        .num_splits = static_cast<uint32_t>(num_splits),
        .num_m_blocks = m_blocks,
    };
    const auto mp = MixParams{
        .gemm_partials = static_cast<const float*>(gemm_scratch.data_ptr()),
        .hc_partials = static_cast<const float*>(hc_scratch.data_ptr()),
        .scales = static_cast<const float*>(scales.data_ptr()),
        .bases = static_cast<const float*>(bases.data_ptr()),
        .new_pre = static_cast<float*>(new_pre.data_ptr()),
        .new_post = static_cast<float*>(new_post.data_ptr()),
        .new_comb = static_cast<float*>(new_comb.data_ptr()),
        .num_tokens = rows,
        .sinkhorn_iters = static_cast<uint32_t>(sinkhorn_iters),
        .hc_norm_eps = static_cast<float>(hc_norm_eps),
        .hc_pre_eps = static_cast<float>(hc_pre_eps),
        .hc_post_scale = static_cast<float>(hc_post_scale),
        .sinkhorn_eps = static_cast<float>(sinkhorn_eps),
    };
    const auto st = reinterpret_cast<cudaStream_t>(stream);
    LaunchKernel(grid, kStatsThreads, st, smem)(exm_stats_kernel, sp);
    const uint32_t mix_grid = (rows + kMixRows - 1) / kMixRows;
    if (num_splits == 40)
      LaunchKernel(mix_grid, kMixThreads, st)(exm_mix_kernel<40>, mp);
    else
      LaunchKernel(mix_grid, kMixThreads, st)(exm_mix_kernel<27>, mp);
  }
};

}  // namespace exm
}  // namespace sglang
