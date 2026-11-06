NAME = "paged_logits_impl.cuh"

SOURCE = r"""#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace paged_logits {

using namespace deep_gemm;
using fp4_t = cutlass::float_e2m1_t;
using Barrier = cutlass::arch::ClusterTransactionBarrier;

constexpr uint32_t kHeads = 32;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kNextN = 6;
constexpr uint32_t kHalf = 3;
constexpr uint32_t kBlockQ = 4;
constexpr uint32_t kPageKV = 128;
constexpr uint32_t kSplitKV = 256;
constexpr uint32_t UMMA_M = 128;
constexpr uint32_t UMMA_N = kHalf * kHeads;
constexpr uint32_t UMMA_K = 64;
constexpr uint32_t kTokenBytes = kHeadDim / 2;
constexpr uint32_t kQHalfBytes = kBlockQ * kHeads * kTokenBytes;
constexpr uint32_t kKVStageBytes = kSplitKV * kTokenBytes;
constexpr uint32_t kNumQStages = 2;
constexpr uint32_t kNumKVStages = 10;
constexpr uint32_t kNumMathThreads = 256;
constexpr uint32_t kNumSpecThreads = 128;
constexpr uint32_t kSFQCol = 4 * UMMA_N;
constexpr uint32_t kSFKVCol = kSFQCol + 16;
constexpr uint32_t kNumTmemCols = 512;
constexpr uint32_t kQTxBytes = 2 * (kQHalfBytes + kBlockQ * kHeads * 4 + kBlockQ * kHeads * 2);
constexpr uint32_t kKVTxBytes = kKVStageBytes + kSplitKV * 4;

struct Smem {
    alignas(1024) fp4_t q[kNumQStages][2][kQHalfBytes];
    alignas(1024) fp4_t kv[kNumKVStages][kKVStageBytes];
    alignas(128) uint32_t sf_q[kNumQStages][2][kBlockQ * kHeads];
    alignas(128) uint32_t sf_q_t[kNumQStages][2][2][kBlockQ * kHeads];
    alignas(128) uint32_t sf_kv[kNumKVStages][kSplitKV];
    alignas(128) __nv_bfloat16 w[kNumQStages][2][kBlockQ * kHeads];
    Barrier full_q[kNumQStages], empty_q[kNumQStages];
    Barrier full_kv[kNumKVStages], empty_kv[kNumKVStages];
    Barrier full_t[4], empty_t[4];
    uint32_t tmem_ptr;
};

struct Walk {
    const uint32_t* lens;
    uint32_t req, split, end_q, end_split, total;
    bool done;

    CUTLASS_DEVICE Walk(const uint32_t* meta, const uint32_t sm, const uint32_t* context_lens, const uint32_t total_q) {
        const uint32_t sq = meta[sm * 2], ss = meta[sm * 2 + 1];
        end_q = meta[sm * 2 + 2], end_split = meta[sm * 2 + 3];
        lens = context_lens, total = total_q;
        req = sq / kNextN, split = ss;
        done = sq >= total_q or (sq == end_q and ss >= end_split);
    }

    CUTLASS_DEVICE bool next(uint32_t& r, uint32_t& s0, uint32_t& s1, uint32_t& num_pages) {
        while (not done) {
            r = req;
            const uint32_t ctx = lens[r * kNextN + kNextN - 1];
            num_pages = (ctx + kPageKV - 1) / kPageKV;
            const bool last = r * kNextN == end_q;
            s0 = split;
            s1 = last ? end_split : (ctx + kSplitKV - 1) / kSplitKV;
            if (last) {
                done = true;
            } else {
                ++ req, split = 0;
                if (req * kNextN >= total or (req * kNextN == end_q and end_split == 0))
                    done = true;
            }
            if (s0 < s1)
                return true;
        }
        return false;
    }
};

CUTLASS_DEVICE void transpose_sf(const uint32_t* src, uint32_t* dst, const uint32_t lane) {
    uint32_t v[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i)
        v[i] = ptx::ld_shared(src + i * 32 + lane);
    __syncwarp();
    ptx::st_shared(dst + lane * 4, v[0], v[1], v[2], v[3]);
}

__global__ void __launch_bounds__(kNumSpecThreads + kNumMathThreads, 1)
kernel(const uint32_t num_q_tokens_total, const uint32_t logits_stride, const uint32_t block_table_stride,
       const uint32_t* context_lens, cutlass::bfloat16_t* logits, const uint32_t* block_table,
       const uint32_t* schedule_meta,
       const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_sf_q,
       const __grid_constant__ CUtensorMap tm_kv, const __grid_constant__ CUtensorMap tm_sf_kv,
       const __grid_constant__ CUtensorMap tm_w) {
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<Smem*>(smem_buffer);
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane = ptx::get_lane_idx();
    const uint32_t sm = blockIdx.x;

    if (warp_idx == 8) {
        cute::prefetch_tma_descriptor(&tm_q);
        cute::prefetch_tma_descriptor(&tm_sf_q);
        cute::prefetch_tma_descriptor(&tm_w);
        cute::prefetch_tma_descriptor(&tm_kv);
        cute::prefetch_tma_descriptor(&tm_sf_kv);
    }
    if (warp_idx == 9 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            smem.full_q[i].init(1);
            smem.empty_q[i].init(kNumMathThreads + 64);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            smem.full_kv[i].init(1);
            smem.empty_kv[i].init(2);
        }
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++ i) {
            smem.full_t[i].init(1);
            smem.empty_t[i].init(128);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();
    if (warp_idx == 10)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr);
    __syncthreads();

    cudaGridDependencySynchronize();

    if (warp_idx == 8) {
        cutlass::arch::warpgroup_reg_dealloc<56>();
        if (cute::elect_one_sync()) {
            Walk walk(schedule_meta, sm, context_lens, num_q_tokens_total);
            RingPipeline<kNumQStages> qp;
            uint32_t r, s0, s1, num_pages;
            while (walk.next(r, s0, s1, num_pages)) {
                CUTE_TIE_DECL(qp.advance(), qs, qph);
                smem.empty_q[qs].wait(qph ^ 1);
                #pragma unroll
                for (uint32_t h = 0; h < 2; ++ h) {
                    const uint32_t tok = r * kNextN + h * kHalf;
                    tma::copy<kHeadDim, kBlockQ * kHeads, 0>(&tm_q, &smem.full_q[qs], smem.q[qs][h], 0, tok * kHeads);
                    tma::copy<kBlockQ * kHeads, 1, 0>(&tm_sf_q, &smem.full_q[qs], smem.sf_q[qs][h], 0, tok);
                    tma::copy<kHeads, kBlockQ, 0>(&tm_w, &smem.full_q[qs], smem.w[qs][h], 0, tok);
                }
                smem.full_q[qs].arrive_and_expect_tx(kQTxBytes);
            }
        }
        __syncwarp();
    } else if (warp_idx == 9) {
        cutlass::arch::warpgroup_reg_dealloc<56>();
        Walk walk(schedule_meta, sm, context_lens, num_q_tokens_total);
        RingPipeline<kNumKVStages> kp;
        uint32_t r, s0, s1, num_pages;
        while (walk.next(r, s0, s1, num_pages)) {
            const uint32_t* row = block_table + r * static_cast<uint64_t>(block_table_stride);
            const auto load = [&](const uint32_t blk) {
                const uint32_t page = blk * 32 + lane;
                return page < num_pages ? row[page] : 0u;
            };
            uint32_t cached_base = s0 * 2 / 32 * 32, cached_coord = load(cached_base / 32);
            uint32_t next_coord = load(cached_base / 32 + 1);
            #pragma unroll 1
            for (uint32_t s = s0; s < s1; ++ s) {
                const uint32_t page_base = s * 2;
                if (page_base + 2 > cached_base + 32) {
                    cached_base = page_base / 32 * 32;
                    cached_coord = next_coord, next_coord = load(cached_base / 32 + 1);
                }
                const int c0 = __shfl_sync(0xffffffff, cached_coord, static_cast<int>(page_base - cached_base));
                const int c1 = __shfl_sync(0xffffffff, cached_coord, static_cast<int>(page_base + 1 - cached_base));
                CUTE_TIE_DECL(kp.advance(), ks, kph);
                smem.empty_kv[ks].wait(kph ^ 1);
                if (cute::elect_one_sync()) {
                    tma::copy<kHeadDim, kPageKV, 0, fp4_t, true>(&tm_kv, &smem.full_kv[ks], smem.kv[ks], 0, 0, 1, c0);
                    tma::copy<kPageKV, 1, 0>(&tm_sf_kv, &smem.full_kv[ks], smem.sf_kv[ks], 0, c0);
                    tma::copy<kHeadDim, kPageKV, 0, fp4_t, true>(&tm_kv, &smem.full_kv[ks],
                                                                 smem.kv[ks] + kPageKV * kTokenBytes, 0, 0, 1, c1);
                    tma::copy<kPageKV, 1, 0>(&tm_sf_kv, &smem.full_kv[ks], smem.sf_kv[ks] + kPageKV, 0, c1);
                    smem.full_kv[ks].arrive_and_expect_tx(kKVTxBytes);
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == 10 or warp_idx == 11) {
        cutlass::arch::warpgroup_reg_dealloc<56>();
        const uint32_t m = warp_idx - 10;
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);
        const auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<fp4_t, fp4_t, float, cutlass::float_ue8m0_t,
                                                                         UMMA_M, UMMA_N, cute::UMMA::Major::K,
                                                                         cute::UMMA::Major::K>();
        Walk walk(schedule_meta, sm, context_lens, num_q_tokens_total);
        RingPipeline<kNumQStages> qp;
        RingPipeline<kNumKVStages> kp;
        uint32_t tphase[2] = {0, 0};
        uint32_t r, s0, s1, num_pages;
        while (walk.next(r, s0, s1, num_pages)) {
            CUTE_TIE_DECL(qp.advance(), qs, qph);
            smem.full_q[qs].wait(qph);
            #pragma unroll
            for (uint32_t h = 0; h < 2; ++ h)
                transpose_sf(smem.sf_q[qs][h], smem.sf_q_t[qs][m][h], lane);
            cutlass::arch::fence_view_async_shared();
            #pragma unroll
            for (uint32_t h = 0; h < 2; ++ h) {
                mma::sm100::replace_smem_desc_addr(sf_desc, smem.sf_q_t[qs][m][h]);
                if (cute::elect_one_sync())
                    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kSFQCol + (m * 2 + h) * 4);
                __syncwarp();
            }
            #pragma unroll 1
            for (uint32_t s = s0; s < s1; ++ s) {
                CUTE_TIE_DECL(kp.advance(), ks, kph);
                smem.full_kv[ks].wait(kph);
                transpose_sf(smem.sf_kv[ks] + m * kPageKV, smem.sf_kv[ks] + m * kPageKV, lane);
                cutlass::arch::fence_view_async_shared();
                mma::sm100::replace_smem_desc_addr(sf_desc, smem.sf_kv[ks] + m * kPageKV);
                if (cute::elect_one_sync())
                    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kSFKVCol + m * 4);
                __syncwarp();
                #pragma unroll
                for (uint32_t h = 0; h < 2; ++ h) {
                    const uint32_t slot = m * 2 + h;
                    smem.empty_t[slot].wait(tphase[h] ^ 1);
                    ptx::tcgen05_after_thread_sync();
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                            const auto rid = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * 2, k * 2);
                            const auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kTokenBytes>(
                                smem.kv[ks], m * UMMA_M, k * UMMA_K);
                            const auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kTokenBytes>(
                                smem.q[qs][h], 0, k * UMMA_K);
                            ptx::SM100_MMA_MXF4_SS::fma(a_desc, b_desc, slot * UMMA_N, k, rid,
                                                        kSFKVCol + m * 4, kSFQCol + slot * 4);
                        }
                        ptx::umma_arrive_no_elect(smem.full_t[slot]);
                    }
                    __syncwarp();
                }
                __syncwarp();
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_kv[ks]));
                tphase[0] ^= 1, tphase[1] ^= 1;
            }
            smem.empty_q[qs].arrive();
        }
    } else if (warp_idx < 8) {
        cutlass::arch::warpgroup_reg_alloc<224>();
        const uint32_t wg = warp_idx / 4;
        const uint32_t t = (warp_idx % 4) * 32 + lane;
        Walk walk(schedule_meta, sm, context_lens, num_q_tokens_total);
        RingPipeline<kNumQStages> qp;
        uint32_t tphase[2] = {0, 0};
        float accum[UMMA_N];
        uint32_t r, s0, s1, num_pages;
        while (walk.next(r, s0, s1, num_pages)) {
            CUTE_TIE_DECL(qp.advance(), qs, qph);
            smem.full_q[qs].wait(qph);
            nv_bfloat162 wt[2 * kHalf][kHeads / 2];
            #pragma unroll
            for (uint32_t h = 0; h < 2; ++ h) {
                const auto wrow = reinterpret_cast<const uint4*>(smem.w[qs][h]);
                #pragma unroll
                for (uint32_t i = 0; i < kHalf; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kHeads / 8; ++ j) {
                        const auto packed = ptx::ld_shared(wrow + i * (kHeads / 8) + j);
                        wt[h * kHalf + i][j * 4 + 0] = *reinterpret_cast<const nv_bfloat162*>(&packed.x);
                        wt[h * kHalf + i][j * 4 + 1] = *reinterpret_cast<const nv_bfloat162*>(&packed.y);
                        wt[h * kHalf + i][j * 4 + 2] = *reinterpret_cast<const nv_bfloat162*>(&packed.z);
                        wt[h * kHalf + i][j * 4 + 3] = *reinterpret_cast<const nv_bfloat162*>(&packed.w);
                    }
                }
            }
            #pragma unroll 1
            for (uint32_t s = s0; s < s1; ++ s) {
                const uint32_t col = s * kSplitKV + wg * UMMA_M + t;
                #pragma unroll
                for (uint32_t h = 0; h < 2; ++ h) {
                    const uint32_t slot = wg * 2 + h;
                    smem.full_t[slot].wait(tphase[h]);
                    ptx::tcgen05_after_thread_sync();
                    ptx::tmem_load_32dp32b<32>(slot * UMMA_N, reinterpret_cast<uint32_t*>(accum));
                    ptx::tmem_load_32dp32b<32>(slot * UMMA_N + 32, reinterpret_cast<uint32_t*>(accum + 32));
                    ptx::tmem_load_32dp32b<32>(slot * UMMA_N + 64, reinterpret_cast<uint32_t*>(accum + 64));
                    cutlass::arch::fence_view_async_tmem_load();
                    ptx::tcgen05_before_thread_sync();
                    smem.empty_t[slot].arrive();
                    tphase[h] ^= 1;
                    #pragma unroll
                    for (uint32_t i = 0; i < kHalf; ++ i) {
                        const auto& w = wt[h * kHalf + i];
                        const float* a = accum + i * kHeads;
                        auto sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                        auto sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                        #pragma unroll
                        for (uint32_t j = 0; j < kHeads; j += 4) {
                            sum_0 = __hfma2(ptx::cvt_relu_bf16x2_f32(make_float2(a[j], a[j + 1])), w[j / 2], sum_0);
                            sum_1 = __hfma2(ptx::cvt_relu_bf16x2_f32(make_float2(a[j + 2], a[j + 3])), w[j / 2 + 1], sum_1);
                        }
                        const auto sum = __hadd2_rn(sum_0, sum_1);
                        const __nv_bfloat16 reduced = __hadd_rn(sum.x, sum.y);
                        const uint32_t row = r * kNextN + h * kHalf + i;
                        logits[row * static_cast<uint64_t>(logits_stride) + col] = static_cast<cutlass::bfloat16_t>(reduced);
                    }
                }
            }
            cutlass::arch::fence_view_async_shared();
            smem.empty_q[qs].arrive();
        }
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

inline void expect(const bool ok, const char* what) {
    if (!ok) TVM_FFI_THROW(ValueError) << "paged_logits: " << what;
}

inline bool is_dtype(const tvm::ffi::TensorView& t, const uint8_t code, const uint8_t bits) {
    return t.dtype().code == code and t.dtype().bits == bits and t.dtype().lanes == 1;
}

inline PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
    static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
    if (fn == nullptr) {
        cudaDriverEntryPointQueryResult q;
        const auto e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&fn),
                                               cudaEnableDefault, &q);
        expect(e == cudaSuccess and q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled not found");
    }
    return fn;
}

inline CUtensorMap make_map(const void* ptr, const CUtensorMapDataType dtype, const uint32_t rank, const uint64_t* dims,
                            const uint64_t* strides, const uint32_t* box, const CUtensorMapSwizzle swizzle) {
    CUtensorMap m;
    cuuint64_t d[3] = {1, 1, 1}, s[2] = {0, 0};
    cuuint32_t b[3] = {1, 1, 1}, e[3] = {1, 1, 1};
    for (uint32_t i = 0; i < rank; ++ i)
        d[i] = dims[i], b[i] = box[i];
    for (uint32_t i = 0; i + 1 < rank; ++ i)
        s[i] = strides[i];
    const auto r = encode_fn()(&m, dtype, rank, const_cast<void*>(ptr), d, s, b, e, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    expect(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
    return m;
}

struct PagedLogits {
    static void run(tvm::ffi::TensorView q_fp, tvm::ffi::TensorView q_sf, tvm::ffi::TensorView kv,
                    tvm::ffi::TensorView w, tvm::ffi::TensorView lens, tvm::ffi::TensorView table,
                    tvm::ffi::TensorView meta, tvm::ffi::TensorView out, int64_t pdl) {
        expect(q_fp.ndim() == 4 and q_fp.size(1) == kNextN and q_fp.size(2) == kHeads and q_fp.size(3) == 64 and
               q_fp.dtype().bits == 8 and q_fp.stride(3) == 1 and q_fp.stride(2) == 64 and q_fp.stride(1) == 64 * kHeads and
               q_fp.stride(0) == 64 * kHeads * kNextN, "q must be [requests, 6, 32, 64] packed fp4, contiguous");
        const int64_t requests = q_fp.size(0), rows = requests * kNextN;
        expect(requests > 0 and rows * kHeads < (int64_t{1} << 31), "request count");
        expect(q_sf.ndim() == 3 and q_sf.size(0) == requests and q_sf.size(1) == kNextN and q_sf.size(2) == kHeads and
               is_dtype(q_sf, kDLInt, 32) and q_sf.stride(2) == 1 and q_sf.stride(1) == kHeads and
               q_sf.stride(0) == kHeads * kNextN, "q scales must be [requests, 6, 32] int32, contiguous");
        expect(kv.ndim() == 4 and kv.size(0) > 0 and kv.size(1) == kPageKV and kv.size(2) == 1 and kv.size(3) == 68 and
               is_dtype(kv, kDLUInt, 8) and kv.stride(3) == 1 and kv.stride(1) == 68 and
               kv.stride(0) >= kPageKV * 68 and kv.stride(0) % 16 == 0, "index K must be [pages, 128, 1, 68] uint8");
        expect(w.ndim() == 2 and w.size(0) == rows and w.size(1) == kHeads and is_dtype(w, kDLBfloat, 16) and
               w.stride(1) == 1 and (w.stride(0) * 2) % 16 == 0, "weights must be [rows, 32] bf16");
        expect(lens.ndim() == 2 and lens.size(0) == requests and lens.size(1) == kNextN and is_dtype(lens, kDLInt, 32) and
               lens.stride(1) == 1 and lens.stride(0) == kNextN, "lens must be [requests, 6] int32, contiguous");
        expect(table.ndim() == 2 and table.size(0) == requests and is_dtype(table, kDLInt, 32) and table.stride(1) == 1 and
               table.stride(0) >= table.size(1), "page table must be [requests, pages] int32");
        expect(meta.ndim() == 2 and meta.size(0) >= 2 and meta.size(1) == 2 and is_dtype(meta, kDLInt, 32) and
               meta.stride(1) == 1 and meta.stride(0) == 2, "schedule must be [sms + 1, 2] int32, contiguous");
        expect(out.ndim() == 2 and out.size(0) == rows and is_dtype(out, kDLBfloat, 16) and out.stride(1) == 1 and
               out.stride(0) == out.size(1) and out.size(1) % kSplitKV == 0 and out.size(1) < (int64_t{1} << 31),
               "logits must be [rows, stride] bf16 with a stride of whole splits");
        const DLDevice dev = q_fp.device();
        for (const auto* t : {&q_sf, &kv, &w, &lens, &table, &meta, &out})
            expect(t->device().device_type == kDLCUDA and t->device().device_id == dev.device_id, "device");
        int current = -1;
        expect(cudaGetDevice(&current) == cudaSuccess and current == dev.device_id, "the current device must be q's");

        const auto page_bytes = static_cast<uint64_t>(kv.stride(0));
        const auto pages = static_cast<uint64_t>(kv.size(0));
        const auto kv_ptr = static_cast<const uint8_t*>(kv.data_ptr());
        CUtensorMap maps[5];
        {
            const uint64_t d[2] = {kHeadDim, static_cast<uint64_t>(rows) * kHeads}, s[1] = {kTokenBytes};
            const uint32_t b[2] = {kHeadDim, kBlockQ * kHeads};
            maps[0] = make_map(q_fp.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, d, s, b, CU_TENSOR_MAP_SWIZZLE_64B);
        }
        {
            const uint64_t d[2] = {kHeads, static_cast<uint64_t>(rows)}, s[1] = {kHeads * 4};
            const uint32_t b[2] = {kHeads, kBlockQ};
            maps[1] = make_map(q_sf.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, 2, d, s, b, CU_TENSOR_MAP_SWIZZLE_NONE);
        }
        {
            const uint64_t d[3] = {kHeadDim, kPageKV, pages}, s[2] = {kTokenBytes, page_bytes};
            const uint32_t b[3] = {kHeadDim, kPageKV, 1};
            maps[2] = make_map(kv_ptr, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3, d, s, b, CU_TENSOR_MAP_SWIZZLE_64B);
        }
        {
            const uint64_t d[2] = {kPageKV, pages}, s[1] = {page_bytes};
            const uint32_t b[2] = {kPageKV, 1};
            maps[3] = make_map(kv_ptr + kPageKV * kTokenBytes, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, d, s, b,
                               CU_TENSOR_MAP_SWIZZLE_NONE);
        }
        {
            const uint64_t d[2] = {kHeads, static_cast<uint64_t>(rows)}, s[1] = {static_cast<uint64_t>(w.stride(0)) * 2};
            const uint32_t b[2] = {kHeads, kBlockQ};
            maps[4] = make_map(w.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d, s, b, CU_TENSOR_MAP_SWIZZLE_NONE);
        }
        const int smem = static_cast<int>(sizeof(Smem));
        static bool attr = false;
        if (not attr) {
            expect(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                   "shared memory attribute");
            attr = true;
        }
        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(static_cast<unsigned>(meta.size(0) - 1), 1, 1);
        config.blockDim = dim3(kNumSpecThreads + kNumMathThreads, 1, 1);
        config.dynamicSmemBytes = smem;
        config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attrs;
        config.numAttrs = pdl != 0 ? 1 : 0;
        const auto e = cudaLaunchKernelEx(&config, kernel, static_cast<uint32_t>(rows), static_cast<uint32_t>(out.size(1)),
                                          static_cast<uint32_t>(table.stride(0)),
                                          static_cast<const uint32_t*>(lens.data_ptr()),
                                          static_cast<cutlass::bfloat16_t*>(out.data_ptr()),
                                          static_cast<const uint32_t*>(table.data_ptr()),
                                          static_cast<const uint32_t*>(meta.data_ptr()),
                                          maps[0], maps[1], maps[2], maps[3], maps[4]);
        expect(e == cudaSuccess, cudaGetErrorString(e));
    }
};

}
"""
