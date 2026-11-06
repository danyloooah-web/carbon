
ENTRY = "v4_entry.cuh"
TAG = "5f56105550c62c37"
WRAPPERS = (('prefill', 'v4::MegaPrefill::run'),)
CUDA_FLAGS = ('--expt-extended-lambda', '--use_fast_math', '-DNDEBUG', '-DFLASH_MLA_IS_BUILD_ON_CUDA', '--diag-suppress=445')

SOURCES = {
    "pattn/kernel.cuh": r"""#pragma once

#include <cuda_fp8.h>
#include <math_constants.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "kernels/utils.h"
#include "helpers.h"
#include "common_subroutine.h"

#include "config.h"

namespace pattn::sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

static constexpr float MAX_INIT_VAL = -1e30;
static constexpr float O_QUANT_CLAMP_MIN_VALUE = 1e-4;

#ifdef KERUTILS_ENABLE_SM103A
static constexpr bool IS_TMEM_LD_WITH_RED_AVAILABLE = true;
#else
static constexpr bool IS_TMEM_LD_WITH_RED_AVAILABLE = false;
#endif

__device__ __forceinline__
float2 apply_rope(const float2 &x, const float &cur_cos, const float &cur_sin) {
    float2 a = {x.x, x.x};
    float2 b = {cur_cos, cur_sin};
    float2 c = {-x.y * cur_sin, +x.y * cur_cos};
    float2 y = ku::float2_fma(a, b, c);
    return y;
}

enum class KVLocation {
    ORIG,
    ORIG_AND_EXTRA,
    EXTRA
};

template<Config CONFIG>
__device__ __forceinline__
void Kernel<CONFIG>::devfunc(const Params &params, const TMAParams &tma_params, const AuxParams &aux_params) {
#ifdef KERUTILS_ENABLE_SM100A
    const uint32_t cta_idx = IS_2CTA ? blockIdx.x % 2 : 0;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const uint32_t idx_in_warpgroup = threadIdx.x % 128;
    const uint32_t lane_idx = threadIdx.x % 32;

    extern __shared__ char smem_buf[];
    SharedMemoryPlan &smem = *reinterpret_cast<SharedMemoryPlan*>(smem_buf);

    if constexpr (IS_2CTA) {
        ku::barrier_cluster_arrive_relaxed();
        ku::barrier_cluster_wait_acquire();
    }

    if (warp_idx == 0 && elect_one_sync()) {

        if constexpr (IS_DECODE) {
            if constexpr (D_BF16 > 0) {
                cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_bf16_part);
            }
        } else {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
        }
    } else if (warp_idx == 1 && elect_one_sync()) {

        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_KV_SLOTS; ++i) {

            smem.bar_kv_slot_full[i].init(IS_DECODE ? 128*CLUSTER_SIZE + 1 : 1);
            smem.bar_kv_slot_empty[i].init(1);
        }
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_INDICES_BUFS; ++i) {
            smem.bar_indices_full[i].init(32);
            smem.bar_indices_empty[i].init(IS_DECODE ? 256 + (FP8_RAW_PIPE ? 1 : 0) : 128);
        }
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_P_BUFS; ++i) {
            smem.bar_tP_full[i].init(1);
            if constexpr (NEED_TP_EMPTY_BAR) {
                smem.bar_tP_empty[i].init(128*CLUSTER_SIZE);
            }
        }
        if constexpr (IS_2CTA && ENABLE_Q_NORM) {
            smem.bar_q_sqr_sum_full.init(128);
        }
        smem.bar_clc_full.init(1);
        smem.bar_clc_empty.init(cta_idx == 1 ? 1 : NUM_WORKING_THREADS);
        smem.bar_tQ_full.init(128*CLUSTER_SIZE);
        smem.bar_tQ_empty.init(1+128);
        smem.bar_tO_full.init(1);
        smem.bar_tO_empty.init(128*CLUSTER_SIZE);
        smem.bar_SO_full.init(128*CLUSTER_SIZE);
        if constexpr (!REUSE_KV_COMPLETION) {
            smem.bar_SO_empty.init(1);
        }
        smem.bar_li_mi_full.init(128);
        smem.bar_li_mi_empty.init(128);
        if constexpr (IS_DECODE) {
            smem.bar_raw_kv_full.init(1);
            if constexpr (FP8_RAW_PIPE) smem.bar_raw_kv_empty[0].init(128);
        }
        fence_barrier_init();
    } else if (warp_idx == 3) {

        AllocatorT().allocate(512, smem.tmem_start_addr.data());
        AllocatorT().release_allocation_lock();
        KU_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_start_addr.data()[0] == 0);
    }

    if constexpr (IS_2CTA) {
        ku::barrier_cluster_arrive_relaxed();
        ku::barrier_cluster_wait_acquire();
    } else {
        __syncthreads();
    }

    struct OuterloopArgs {
        bool is_valid;
        uint32_t s_q_idx;
        uint32_t job_idx_mod_2;
        uint32_t topk_length;
        uint32_t num_kv_blocks;

        uint32_t extra_topk_length;
        uint32_t num_orig_slots;
    };

    auto _make_outer_loop_args = [&](uint32_t job_idx_mod_2, uint32_t cta_x_idx) -> OuterloopArgs {
        uint32_t s_q_idx = cta_x_idx / CLUSTER_SIZE;
        if constexpr (IS_DECODE) {
            uint32_t topk_length = params.topk_length ? (uint32_t)__ldg(params.topk_length + s_q_idx) : (uint32_t)params.topk;
            uint32_t extra_topk_length = params.extra_topk_length ? (uint32_t)__ldg(params.extra_topk_length + s_q_idx) : (uint32_t)params.extra_topk;
            bool have_extra_kv = params.extra_topk > 0;
            uint32_t num_orig_slots, num_kv_blocks;
            if (have_extra_kv) {
                num_orig_slots = (uint32_t)params.topk;
                num_kv_blocks = ku::ceil_div(num_orig_slots + extra_topk_length, (uint32_t)B_TOPK);
            } else {
                num_orig_slots = 0xFFFFFFFFu;
                num_kv_blocks = ku::ceil_div(topk_length, (uint32_t)B_TOPK);
            }
            num_kv_blocks = std::max(num_kv_blocks, 1u);
            return {
                true,
                s_q_idx,
                job_idx_mod_2,
                topk_length,
                num_kv_blocks,
                extra_topk_length,
                num_orig_slots
            };
        } else {
            uint32_t topk_length = params.topk_length ? __ldg(params.topk_length + s_q_idx) : params.topk;
            uint32_t num_kv_blocks = std::max(ku::ceil_div(topk_length, (uint32_t)B_TOPK), 1u);
            return {
                true,
                s_q_idx,
                job_idx_mod_2,
                topk_length,
                num_kv_blocks
            };
        }
    };

    auto run_along_kv_blocks = [&](const OuterloopArgs &cur_args, auto callable) {
        uint32_t num_full_orig_blocks = std::min(cur_args.num_kv_blocks, cur_args.num_orig_slots / B_TOPK);
        bool has_mixed_block = num_full_orig_blocks < cur_args.num_kv_blocks && cur_args.num_orig_slots % B_TOPK != 0;
        CUTE_NO_UNROLL
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_full_orig_blocks; ++kv_block_idx) {
            callable.template operator()<KVLocation::ORIG>(kv_block_idx);
        }
        if (has_mixed_block) {
            callable.template operator()<KVLocation::ORIG_AND_EXTRA>(num_full_orig_blocks);
        }
        CUTE_NO_UNROLL
        for (uint32_t kv_block_idx = num_full_orig_blocks + has_mixed_block; kv_block_idx < cur_args.num_kv_blocks; ++kv_block_idx) {
            callable.template operator()<KVLocation::EXTRA>(kv_block_idx);
        }
    };

    auto get_first_job = [&]() -> OuterloopArgs {
        return _make_outer_loop_args(0, blockIdx.x);
    };
    auto get_next_job = [&](const OuterloopArgs &cur_args) -> OuterloopArgs {
        smem.bar_clc_full.wait(cur_args.job_idx_mod_2);
        ku::CLCResult next_cta0_idx = ku::get_clc_query_response<true>(smem.clc_response_obj);
        arrive_on_cta0_barrier(smem.bar_clc_empty);

        if (!next_cta0_idx.is_valid) {
            return OuterloopArgs {false};
        } else {
            return _make_outer_loop_args(
                cur_args.job_idx_mod_2^1,
                next_cta0_idx.x
            );
        }
    };

    if (warpgroup_idx == 0) {

        cutlass::arch::warpgroup_reg_alloc<184>();

        #pragma nv_diag_suppress 549
        float cached_cos[2], cached_sin[2];
        auto shift_cached_cos_and_cached_sin = [&]() {
            cached_cos[0] = cached_cos[1];
            cached_sin[0] = -cached_sin[1];
        };

        auto load_q_and_save_to_tmem = [&](const OuterloopArgs &cur_job) {
            static constexpr uint32_t NUM_CACHED_BF16_PER_THREAD = H_Q_PER_CTA * D_QK / 128;
            static constexpr uint32_t NUM_BF16_PER_LOAD = 256 / 16;
            bf16 cached_q[NUM_CACHED_BF16_PER_THREAD];
            float q_sqr_sum = 0.0f;

            uint32_t cur_q_position = __ldg(params.token_positions + cur_job.s_q_idx);
            cached_cos[1] = __ldg(params.cos_sin_cache + cur_q_position * D_ROPE + lane_idx);
            cached_sin[1] = __ldg(params.cos_sin_cache + cur_q_position * D_ROPE + D_ROPE / 2 + lane_idx);

            bf16 *q_token_base = params.q + (uint64_t)cur_job.s_q_idx * params.stride_q_s_q;

            static constexpr uint32_t TILE_SIZE = 64;
            CUTE_UNROLL
            for (uint32_t local_tile_idx = 3; local_tile_idx != 0xFFFFFFFF; --local_tile_idx) {
                uint32_t tile_idx =
                    CLUSTER_SIZE == 1 ?
                    local_tile_idx * 2 + (warp_idx / (H_Q_PER_CTA / 32)) :
                    (warp_idx / (H_Q_PER_CTA / 32)) * 4 + local_tile_idx;
                CUTE_UNROLL
                for (uint32_t i = 0; i < TILE_SIZE / NUM_BF16_PER_LOAD; ++i) {
                    static_assert(NUM_MRGEMM_RAILS == 2);
                    uint32_t h_q_idx = cta_idx * H_Q_PER_CTA + idx_in_warpgroup % H_Q_PER_CTA;
                    uint32_t d_q_idx = tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD;
                    KU_LDG_256(
                        q_token_base + h_q_idx * NUM_BF16_PER_LOAD + d_q_idx * H_Q,
                        cached_q + local_tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD,
                        ".nc", "no_allocate", "evict_first", "256B"
                    );
                }

                if (local_tile_idx == 3 && tile_idx == D_VO / TILE_SIZE - 1) {
                    float2 cur_q_sqr_sum = {0.0f, 0.0f};
                    CUTE_UNROLL
                    for (uint32_t j = 0; j < TILE_SIZE; j += 2) {
                        float2 x = __bfloat1622float2(*(nv_bfloat162*)(cached_q+local_tile_idx*TILE_SIZE+j));
                        if constexpr (ENABLE_Q_NORM) {
                            cur_q_sqr_sum = ku::float2_fma(x, x, cur_q_sqr_sum);
                        }
                        float cur_cos = __shfl_sync(0xFFFFFFFF, cached_cos[1], j/2);
                        float cur_sin = __shfl_sync(0xFFFFFFFF, cached_sin[1], j/2);
                        float2 y = apply_rope(x, cur_cos, cur_sin);
                        *(nv_bfloat162*)(cached_q+local_tile_idx*TILE_SIZE+j) = nv_bfloat162{__float2bfloat16_rn(y.x), __float2bfloat16_rn(y.y)};
                    }
                    q_sqr_sum += cur_q_sqr_sum.x + cur_q_sqr_sum.y;
                } else {
                    if constexpr (ENABLE_Q_NORM) {

                        CUTE_UNROLL
                        for (uint32_t i = 0; i < TILE_SIZE; ++i)
                            asm volatile ("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(q_sqr_sum) : "h"(*(uint16_t*)(cached_q+local_tile_idx*TILE_SIZE+i)));
                    }
                }
            }

            if constexpr (ENABLE_Q_NORM) {
                smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup] = q_sqr_sum;
            }

            smem.bar_tQ_empty.wait(cur_job.job_idx_mod_2^1);
            ku::tcgen05_after_thread_sync();

            static constexpr uint32_t NUM_CACHED_UINT32 = NUM_CACHED_BF16_PER_THREAD/2;
            ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32/2>(tmem_cols::Q, cached_q);
            ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32/2>(tmem_cols::Q+NUM_CACHED_UINT32/2, cached_q+NUM_CACHED_UINT32/2*2);
            cutlass::arch::fence_view_async_tmem_store();

            ku::tcgen05_before_thread_sync();
            arrive_on_cta0_barrier(smem.bar_tQ_full);
            if constexpr (IS_2CTA && ENABLE_Q_NORM) {
                smem.bar_q_sqr_sum_full.arrive();
            }
        };
        auto store_o = [&](const OuterloopArgs &cur_job, const bool &is_last_job) {
            smem.bar_li_mi_full.wait(cur_job.job_idx_mod_2);
            float li = 0.0f;
            float mi = smem.rowwise_mi_buf[idx_in_warpgroup % H_Q_PER_CTA];
            if constexpr (FOLD_FACTOR == 2) {
                li = smem.rowwise_li_buf[idx_in_warpgroup] + smem.rowwise_li_buf[idx_in_warpgroup^64];
            } else {
                li = __fadd_rn(
                    __fadd_rn(smem.rowwise_li_buf[idx_in_warpgroup], smem.rowwise_li_buf[idx_in_warpgroup^64]),
                    __fadd_rn(smem.rowwise_li_buf[idx_in_warpgroup^32], smem.rowwise_li_buf[idx_in_warpgroup^96])
                );
            }
            smem.bar_li_mi_empty.arrive();

            if (idx_in_warpgroup < H_Q_PER_CTA) {
                uint32_t global_index = cur_job.s_q_idx * H_Q + cta_idx * H_Q_PER_CTA + idx_in_warpgroup;
                float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                params.lse[global_index] = cur_lse;
            }

            float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + cta_idx * H_Q_PER_CTA + idx_in_warpgroup % H_Q_PER_CTA) * CUDART_L2E_F;
            float output_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li + exp2f(attn_sink - mi));

            smem.bar_tO_full.wait(cur_job.job_idx_mod_2);
            ku::tcgen05_after_thread_sync();

            static constexpr uint32_t MMA_ATOM_N = 256;
            static constexpr uint32_t NUM_MMA_ATOMS = D_VO / MMA_ATOM_N;
            static constexpr uint32_t NUM_O_TMEM_COLS_PER_ATOM = MMA_ATOM_N / FOLD_FACTOR;
            static constexpr uint32_t EPILOGUE_TILE_SIZE = O_QUANT_TILE_SIZE;
            static constexpr uint32_t NUM_EPILOGUE_TILES_PER_ATOM = NUM_O_TMEM_COLS_PER_ATOM / EPILOGUE_TILE_SIZE;
            static_assert(NUM_O_TMEM_COLS_PER_ATOM % EPILOGUE_TILE_SIZE == 0);
            fp8_e4m3 output_fp8[NUM_MMA_ATOMS][NUM_O_TMEM_COLS_PER_ATOM];
            uint8_t output_sf[NUM_MMA_ATOMS][NUM_O_TMEM_COLS_PER_ATOM / O_QUANT_TILE_SIZE];
            static_assert(!H64_GENERIC || (FOLD_FACTOR == 2 && EPILOGUE_TILE_SIZE == 32 && NUM_EPILOGUE_TILES_PER_ATOM >= 2));
            float final_output_cache[EPILOGUE_TILE_SIZE];
            float final_output_reduction;

            CUTE_UNROLL
            for (uint32_t mma_atom_idx = 0; mma_atom_idx < NUM_MMA_ATOMS; mma_atom_idx += 1) {
                CUTE_UNROLL
                for (uint32_t epilogue_tile_idx_in_atom = 0; epilogue_tile_idx_in_atom < NUM_EPILOGUE_TILES_PER_ATOM; ++epilogue_tile_idx_in_atom) {

                    uint32_t tmem_col_base = tmem_cols::O + mma_atom_idx * NUM_O_TMEM_COLS_PER_ATOM + epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE;
                    float output[EPILOGUE_TILE_SIZE];
                    float reduce_result_by_tmem_ld;
                    bool is_cached_final_tile = H64_GENERIC && mma_atom_idx + 1 == NUM_MMA_ATOMS &&
                        epilogue_tile_idx_in_atom + 1 == NUM_EPILOGUE_TILES_PER_ATOM;
                    if (is_cached_final_tile) {
                        CUTE_UNROLL
                        for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; ++j)
                            output[j] = final_output_cache[j];
                        if constexpr (IS_TMEM_LD_WITH_RED_AVAILABLE)
                            reduce_result_by_tmem_ld = final_output_reduction;
                    } else {
                        if constexpr (IS_TMEM_LD_WITH_RED_AVAILABLE) {
                            ku::tmem_ld_red_32dp32bNx<EPILOGUE_TILE_SIZE, true, true, true>(tmem_col_base, output, reduce_result_by_tmem_ld);
                        } else {
                            ku::tmem_ld_32dp32bNx<EPILOGUE_TILE_SIZE>(tmem_col_base, output);
                        }
                        bool drains_output = mma_atom_idx + 1 == NUM_MMA_ATOMS &&
                            epilogue_tile_idx_in_atom + (H64_GENERIC ? 2 : 1) == NUM_EPILOGUE_TILES_PER_ATOM;
                        if constexpr (H64_GENERIC) {
                            if (drains_output) {
                                if constexpr (IS_TMEM_LD_WITH_RED_AVAILABLE) {
                                    ku::tmem_ld_red_32dp32bNx<EPILOGUE_TILE_SIZE, true, true, true>(tmem_col_base + EPILOGUE_TILE_SIZE, final_output_cache, final_output_reduction);
                                } else {
                                    ku::tmem_ld_32dp32bNx<EPILOGUE_TILE_SIZE>(tmem_col_base + EPILOGUE_TILE_SIZE, final_output_cache);
                                }
                            }
                        }

                        cutlass::arch::fence_view_async_tmem_load();
                        if (drains_output) {
                            ku::tcgen05_before_thread_sync();
                            if (!is_last_job) {

                                arrive_on_cta0_barrier(smem.bar_tO_empty);
                            }
                        }
                    }

                    bool should_perform_rope;
                    {
                        static_assert(FOLD_FACTOR == 2);
                        static_assert(D_ROPE % EPILOGUE_TILE_SIZE == 0);
                        should_perform_rope =
                            mma_atom_idx + 1 == NUM_MMA_ATOMS &&
                            epilogue_tile_idx_in_atom >= NUM_EPILOGUE_TILES_PER_ATOM - D_ROPE/EPILOGUE_TILE_SIZE &&
                            warp_idx >= 2;
                        if (should_perform_rope) {
                            CUTE_UNROLL
                            for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; j += 2) {
                                float2 x = *(float2*)(output + j);
                                uint32_t src_lane = j/2 + (epilogue_tile_idx_in_atom + 1 == NUM_EPILOGUE_TILES_PER_ATOM ? EPILOGUE_TILE_SIZE / 2 : 0);
                                float cur_cos = __shfl_sync(0xFFFFFFFF, cached_cos[0], src_lane);
                                float cur_sin = __shfl_sync(0xFFFFFFFF, cached_sin[0], src_lane);
                                float2 y = apply_rope(x, cur_cos, cur_sin);
                                *(float2*)(output + j) = y;
                            }
                        }
                    }

                    float output_abs_max;
                    if (!IS_TMEM_LD_WITH_RED_AVAILABLE || should_perform_rope) {
                        output_abs_max = get_max<EPILOGUE_TILE_SIZE, true>(output) * output_scale;
                    } else {
                        output_abs_max = reduce_result_by_tmem_ld * output_scale;
                    }
                    output_abs_max = max(O_QUANT_CLAMP_MIN_VALUE, output_abs_max);
                    float sf = output_abs_max / 448.0f;
                    uint32_t sf_as_uint32 = *reinterpret_cast<uint32_t*>(&sf);
                    uint32_t exp_sf = (int32_t)((sf_as_uint32-1) >> 23) + (1 - 127);
                    uint32_t sf_inv_as_uint32 = (127 - exp_sf) << 23;
                    float sf_inv = *reinterpret_cast<float*>(&sf_inv_as_uint32);
                    float cur_multiplier = output_scale * sf_inv;
                    float2 cur_multiplier_float2 = float2(cur_multiplier, cur_multiplier);

                    CUTE_UNROLL
                    for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; j += 2) {
                        float2 x = *(float2*)(output + j);
                        x = ku::float2_mul(x, cur_multiplier_float2);
                        *(__nv_fp8x2_storage_t*)(output_fp8[mma_atom_idx] + epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE + j) = __nv_cvt_float2_to_fp8x2(
                            x,
                            __NV_SATFINITE,
                            __nv_fp8_interpretation_t::__NV_E4M3
                        );
                    }
                    output_sf[mma_atom_idx][epilogue_tile_idx_in_atom] = exp_sf + 127;
                }
            }

            uint32_t head_idx = cta_idx*H_Q_PER_CTA + idx_in_warpgroup%H_Q_PER_CTA;
            uint32_t wv_group_idx = head_idx / WV_GROUP_SIZE;
            uint32_t head_idx_in_wv_group = head_idx % WV_GROUP_SIZE;
            CUTE_UNROLL
            for (uint32_t mma_atom_idx = 0; mma_atom_idx < NUM_MMA_ATOMS; mma_atom_idx += 1) {

                static constexpr uint32_t OUTPUT_SAVE_AS_SCALE_GRAN = 32;

                uint32_t head_dim_idx_base =
                    CLUSTER_SIZE == 1 ?
                    mma_atom_idx * MMA_ATOM_N + (warp_idx/(H_Q_PER_CTA/32)) * (MMA_ATOM_N/FOLD_FACTOR) :
                    mma_atom_idx * NUM_O_TMEM_COLS_PER_ATOM + (warp_idx/(H_Q_PER_CTA/32)) * MMA_ATOM_N;
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_O_TMEM_COLS_PER_ATOM / OUTPUT_SAVE_AS_SCALE_GRAN; ++i) {
                    uint32_t sf_block_idx = head_idx_in_wv_group + (head_dim_idx_base/OUTPUT_SAVE_AS_SCALE_GRAN+i) * WV_GROUP_SIZE;
                    *((uint8_t*)(params.out_sf + cur_job.s_q_idx + wv_group_idx*params.stride_out_sf_wv_group + (sf_block_idx/4)*params.stride_out_sf_head_dim) + sf_block_idx%4) = output_sf[mma_atom_idx][i];
                }

                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_O_TMEM_COLS_PER_ATOM; i += 32) {
                    uint32_t head_dim_idx = head_dim_idx_base + i;
                    KU_STG_256(
                        params.out_fp8 + (uint64_t)cur_job.s_q_idx*(H_Q*D_VO) + wv_group_idx*(WV_GROUP_SIZE*D_VO) + head_idx_in_wv_group*32 + head_dim_idx*WV_GROUP_SIZE,
                        output_fp8[mma_atom_idx] + i,
                        "no_allocate",
                        "evict_first"
                    );
                }
            }
        };

        OuterloopArgs cur_job = get_first_job();
        load_q_and_save_to_tmem(cur_job);
        do {
            OuterloopArgs next_job = get_next_job(cur_job);
            shift_cached_cos_and_cached_sin();
            if (next_job.is_valid) {
                load_q_and_save_to_tmem(next_job);
            }
            store_o(cur_job, !next_job.is_valid);
            cur_job = next_job;
        } while (cur_job.is_valid);

        NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
        if (warp_idx == 0) {
            AllocatorT().free(0, 512);
        }
    } else if (warpgroup_idx == 3) {

        cutlass::arch::warpgroup_reg_alloc<128>();

        OuterloopArgs cur_job = get_first_job();
        uint32_t local_warp_idx = warp_idx - 12;
        static_assert(FOLD_FACTOR == 2);
        bf16* sS_base = smem.s + (local_warp_idx >= 2 ? H_Q_PER_CTA * (B_TOPK/2) : 0) + (idx_in_warpgroup%H_Q_PER_CTA) * 8;
        RingBufferState rs;

        uint32_t s_previous_slot = NUM_KV_SLOTS - 1;
        uint32_t s_previous_phase = 1;
        do {

            static constexpr uint32_t NUM_ELEMS_PER_THREAD = B_TOPK * H_Q_PER_CTA / 128;
            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;

            float score_multiplier;
            if constexpr (ENABLE_Q_NORM) {
                if constexpr (IS_2CTA) {
                    smem.bar_q_sqr_sum_full.wait(cur_job.job_idx_mod_2);
                } else {
                    smem.bar_tQ_full.wait(cur_job.job_idx_mod_2);
                }

                if constexpr (H_Q_PER_CTA == 64) {
                    score_multiplier = smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup] + smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup^64];
                } else {
                    score_multiplier = __fadd_rn(
                        __fadd_rn(smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup], smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup^64]),
                        __fadd_rn(smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup^32], smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup^96])
                    );
                }
                score_multiplier = params.sm_scale_div_log2 * rsqrtf(score_multiplier / D_QK + params.rms_norm_eps);
            } else {
                score_multiplier = params.sm_scale_div_log2;
            }

            smem.bar_tQ_empty.arrive();

            CUTE_NO_UNROLL
            for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                auto [p_buf_idx, p_bar_phase] = rs.get<NUM_P_BUFS>();
                uint32_t preloaded_validity = 0;
                if constexpr (H64_GENERIC) {
                    static_assert(NUM_ELEMS_PER_THREAD == 32);
                    smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                    uint32_t mask_address = static_cast<uint32_t>(__cvta_generic_to_shared(
                        (char*)&smem.is_k_valid[indices_buf_idx] + (local_warp_idx >= 2 ? 4 : 0)));
                    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(preloaded_validity) : "r"(mask_address) : "memory");

                }
                smem.bar_tP_full[p_buf_idx].wait(p_bar_phase);
                if constexpr (!H64_GENERIC) {
                    smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                }
                ku::tcgen05_after_thread_sync();

                float p[NUM_ELEMS_PER_THREAD];
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    barrier_ids::WG3_WARP02_SYNC,
                    barrier_ids::WG3_WARP13_SYNC,
                    false,
                    H64_GENERIC
                >(
                    tmem_cols::get_p(p_buf_idx),
                    (char*)&smem.is_k_valid[indices_buf_idx],
                    local_warp_idx,
                    lane_idx,
                    [&]() {
                        if constexpr (NEED_TP_EMPTY_BAR) {
                            arrive_on_cta0_barrier(smem.bar_tP_empty[p_buf_idx]);
                        }
                    },
                    smem.p_exchange_buf,
                    p,
                    preloaded_validity
                );

                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= score_multiplier;

                smem.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(64, barrier_ids::WG3_WARP02_SYNC + (local_warp_idx&1));
                smem.bar_indices_empty[indices_buf_idx].arrive();
                cur_pi_max = max(cur_pi_max, smem.rowwise_max_buf[idx_in_warpgroup^64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                float new_max, scale_for_old;
                if (!should_scale_o) {

                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;

                nv_bfloat16 s[NUM_ELEMS_PER_THREAD];
                float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>((nv_bfloat162*)s, p, score_multiplier, new_max);
                li = fmaf(li, scale_for_old, cur_sum);

                if constexpr (REUSE_KV_COMPLETION) {

                    smem.bar_kv_slot_empty[s_previous_slot].wait(s_previous_phase);
                } else {
                    smem.bar_SO_empty.wait(rs.get<1>().second^1);
                }
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; ++i) {
                    ku::st_shared(sS_base + i*8*H_Q_PER_CTA, *(__int128_t*)(s + i*8));
                }

                if (kv_block_idx > 0 && should_scale_o) {
                    ku::tcgen05_after_thread_sync();
                    rescale_O<D_VO / FOLD_FACTOR, 32, tmem_cols::O>(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }

                fence_view_async_shared();
                ku::tcgen05_before_thread_sync();
                arrive_on_cta0_barrier(smem.bar_SO_full);
                if constexpr (REUSE_KV_COMPLETION) {
                    if (++s_previous_slot == NUM_KV_SLOTS) {
                        s_previous_slot = 0;
                        s_previous_phase ^= 1;
                    }
                }
                rs.update();
            }

            if (real_mi == -CUDART_INF_F) {

                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            smem.bar_li_mi_empty.wait(cur_job.job_idx_mod_2^1);
            static_assert(H_Q_PER_CTA % 32 == 0);
            if (local_warp_idx < H_Q_PER_CTA / 32) {
                if constexpr (!IS_DECODE) {
                    uint32_t global_index = cur_job.s_q_idx * H_Q + cta_idx * H_Q_PER_CTA + idx_in_warpgroup;
                    params.max_logits[global_index] = real_mi * CUDART_LN2_F;
                }
                smem.rowwise_mi_buf[idx_in_warpgroup] = mi;
            }
            smem.rowwise_li_buf[idx_in_warpgroup] = li;
            smem.bar_li_mi_full.arrive();

            cur_job = get_next_job(cur_job);
        } while (cur_job.is_valid);
    } else if (warpgroup_idx == 2) {
        cutlass::arch::warpgroup_reg_dealloc<72>();
        if (warp_idx == MMA_WARP && cta_idx == 0 && elect_one_sync()) {

            auto tiled_mma_qk = TiledMMA_QK{};
            auto tiled_mma_sv = TiledMMA_SV{};
            Tensor tQ = tiled_mma_qk.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_qk, Shape<Int<H_Q_PER_CTA>, Int<D_QK / NUM_MRGEMM_RAILS>>{})
            );
            Tensor tP = partition_fragment_C(tiled_mma_qk, Shape<Int<H_Q_PER_CTA>, Int<B_TOPK * NUM_MRGEMM_RAILS>>{});
            Tensor sS = make_tensor(
                make_smem_ptr(smem.s),
                ku::make_umma_canonical_k_major_layout<H_Q_PER_CTA, B_TOPK, 0>()
            );
            Tensor tO = partition_fragment_C(tiled_mma_sv, Shape<Int<H_Q_PER_CTA>, Int<D_VO>>{});
            tQ.data().get() = tmem_cols::Q;
            tO.data().get() = tmem_cols::O;

            RingBufferState rs_qk, rs_sv;
            auto run_qk_gemm = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                if (kv_block_idx == 0) {
                    smem.bar_tQ_full.wait(job.job_idx_mod_2);
                }
                auto [kv_slot_idx, kv_bar_phase] = rs_qk.get<NUM_KV_SLOTS>();
                Tensor sK = make_tensor(
                    make_smem_ptr(smem.kv_slots[kv_slot_idx]),
                    ku::make_umma_canonical_k_major_layout<B_TOPK / CLUSTER_SIZE * NUM_MRGEMM_RAILS, D_QK / NUM_MRGEMM_RAILS, 128>()
                );

                if constexpr (IS_DECODE && D_BF16 == 0) {

                    smem.bar_kv_slot_full[kv_slot_idx].arrive();
                } else {
                    smem.bar_kv_slot_full[kv_slot_idx].arrive_and_expect_tx(IS_DECODE ? B_TOPK*D_BF16*sizeof(bf16) : B_TOPK*D_QK*sizeof(bf16));
                }
                smem.bar_kv_slot_full[kv_slot_idx].wait(kv_bar_phase);
                auto [p_buf_idx, p_bar_phase] = rs_qk.get<NUM_P_BUFS>();
                if constexpr (NEED_TP_EMPTY_BAR) {
                    smem.bar_tP_empty[p_buf_idx].wait(p_bar_phase ^ 1);
                }
                tP.data().get() = tmem_cols::get_p(p_buf_idx);

                ku::tcgen05_after_thread_sync();
                ku::utcmma_ts(tiled_mma_qk, tQ, sK, tP, true);
                umma_arrive_on_every_cta(smem.bar_tP_full[p_buf_idx]);

                if (kv_block_idx == job.num_kv_blocks-1) {
                    umma_arrive_on_every_cta(smem.bar_tQ_empty);
                }
                rs_qk.update();
            };

            auto run_sv_gemm = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                if (kv_block_idx == 0) {
                    smem.bar_tO_empty.wait(job.job_idx_mod_2^1);
                }
                auto [kv_slot_idx, _] = rs_sv.get<NUM_KV_SLOTS>();
                smem.bar_SO_full.wait(rs_sv.get<1>().second);
                Tensor sV = make_tensor(
                    make_smem_ptr(smem.kv_slots[kv_slot_idx]),
                    ku::make_umma_canonical_mn_major_layout<D_VO / CLUSTER_SIZE, B_TOPK, 128>()
                );
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_sv, sS, sV, tO, kv_block_idx == 0);
                umma_arrive_on_every_cta(smem.bar_kv_slot_empty[kv_slot_idx]);
                if constexpr (!REUSE_KV_COMPLETION) {
                    umma_arrive_on_every_cta(smem.bar_SO_empty);
                }
                if (kv_block_idx == job.num_kv_blocks-1) {
                    umma_arrive_on_every_cta(smem.bar_tO_full);
                }
                rs_sv.update();
            };

            OuterloopArgs cur_job = get_first_job();
            run_qk_gemm(cur_job, 0);
            do {
                CUTE_NO_UNROLL
                for (uint32_t kv_block_idx = 1; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                    run_qk_gemm(cur_job, kv_block_idx);
                    run_sv_gemm(cur_job, kv_block_idx-1);
                }

                OuterloopArgs next_job = get_next_job(cur_job);
                if (next_job.is_valid) {
                    run_qk_gemm(next_job, 0);
                }
                run_sv_gemm(cur_job, cur_job.num_kv_blocks-1);

                cur_job = next_job;
            } while (cur_job.is_valid);
        } else if (warp_idx == CLC_WARP && elect_one_sync()) {

            bool phase = 0;
            while (true) {
                if (cta_idx == 0) {
                    smem.bar_clc_empty.wait(phase^1);
                    ku::issue_clc_query_multicast_cluster_all(smem.bar_clc_full, smem.clc_response_obj);
                }
                smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj));

                smem.bar_clc_full.wait(phase&1);
                ku::CLCResult clc_result = ku::get_clc_query_response<true>(smem.clc_response_obj);
                arrive_on_cta0_barrier(smem.bar_clc_empty);
                if (!clc_result.is_valid)
                    break;

                phase ^= 1;
            }
            if constexpr (IS_2CTA) {
                if (cta_idx == 0) {
                    smem.bar_clc_empty.wait(phase);
                    smem.bar_clc_empty.arrive(1u);
                } else {
                    smem.bar_clc_empty.wait(0);
                }
            }
        } else if (warp_idx == 10) {

            OuterloopArgs cur_job = get_first_job();
            RingBufferState rs;
            static constexpr uint32_t NUM_INDICES_PER_THREAD = B_TOPK / 32;
            static_assert(B_TOPK % 32 == 0);

            do {
                if constexpr (!IS_DECODE && PRED_IDX) {

                    CUTE_NO_UNROLL
                    for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                        smem.bar_indices_empty[indices_buf_idx].wait(indices_bar_phase^1);

                        uint32_t indices[NUM_INDICES_PER_THREAD];
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                            const auto pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                            const auto src = params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos;

                            indices[i] = static_cast<uint32_t>(pos < cur_job.topk_length ? __ldg(src) : -1);
                        }

                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {

                          const auto is_index_valid = indices[i] < static_cast<uint32_t>(params.s_kv);
                          const auto mask = __ballot_sync(0xFFFFFFFF, is_index_valid);
                          if (lane_idx == 0) {
                              *((uint32_t*)smem.is_k_valid[indices_buf_idx] + i) = mask;
                          }
                        }

                        smem.bar_indices_full[indices_buf_idx].arrive();
                        rs.update();
                    }
                } else if constexpr (!IS_DECODE) {
                    auto body = [&]<bool CHECK_TOPK_SUBSCRIPT>() {
                        CUTE_NO_UNROLL
                        for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                            auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                            smem.bar_indices_empty[indices_buf_idx].wait(indices_bar_phase^1);

                            CUTE_UNROLL
                            for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                                uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                                int cur_index;
                                if constexpr (CHECK_TOPK_SUBSCRIPT) {

                                    cur_index = pos < cur_job.topk_length ? __ldg(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos) : -1;
                                } else {

                                    cur_index = __ldg(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos);
                                }
                                bool is_index_valid = (uint32_t)cur_index < (uint32_t)params.s_kv;
                                uint32_t mask = __ballot_sync(0xFFFFFFFF, is_index_valid);
                                if (lane_idx == 0) {
                                    *((uint32_t*)smem.is_k_valid[indices_buf_idx] + i) = mask;
                                }
                            }

                            smem.bar_indices_full[indices_buf_idx].arrive();
                            rs.update();
                        }
                    };
                    if (cur_job.topk_length % B_TOPK == 0 && cur_job.topk_length != 0)
                        body.template operator()<false>();
                    else
                        body.template operator()<true>();
                } else {
                    int *indices_base = params.indices + (int64_t)cur_job.s_q_idx * params.stride_indices_s_q;
                    int *extra_indices_base = params.extra_indices + (int64_t)cur_job.s_q_idx * params.stride_extra_indices_s_q;
                    run_along_kv_blocks(cur_job, [&]<KVLocation LOC>(uint32_t kv_block_idx) {
                        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                        smem.bar_indices_empty[indices_buf_idx].wait(indices_bar_phase^1);

                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                            uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                            bool in_extra;
                            if constexpr (LOC == KVLocation::ORIG) {
                                in_extra = false;
                            } else if constexpr (LOC == KVLocation::EXTRA) {
                                in_extra = true;
                            } else {
                                in_extra = pos >= cur_job.num_orig_slots;
                            }
                            uint32_t local_pos = in_extra ? pos - cur_job.num_orig_slots : pos;
                            uint32_t valid_len = in_extra ? cur_job.extra_topk_length : cur_job.topk_length;

                            int cur_index = local_pos < valid_len ? __ldg((in_extra ? extra_indices_base : indices_base) + local_pos) : -1;
                            bool is_index_valid = cur_index >= 0;

                            int64_t src_block_stride = in_extra ? params.stride_extra_kv_block : params.stride_kv_block;
                            const auto &fast_divmod = in_extra
                                ? aux_params.fast_divmod_extra_page_block_size
                                : aux_params.fast_divmod_page_block_size;
                            uint32_t token_idx = is_index_valid ? (uint32_t)cur_index : 0;
                            int idx_in_block;
                            int block_idx = fast_divmod.divmod(idx_in_block, (int)token_idx);
                            uint32_t row = i * 32 + lane_idx;

                            uint32_t tma_k_stride = in_extra ? ExtraKVFormat::TMA_K_STRIDE : OrigKVFormat::TMA_K_STRIDE;
                            uint32_t num_scales_each_token = in_extra ? ExtraKVFormat::NUM_SCALES_EACH_TOKEN : OrigKVFormat::NUM_SCALES_EACH_TOKEN;
                            smem.decode_tma_coords[indices_buf_idx][row] = is_index_valid
                                ? (src_block_stride / tma_k_stride) * block_idx + idx_in_block
                                : -1;

                            uint32_t page_block_size = in_extra ? params.extra_page_block_size : params.page_block_size;
                            uint8_t *kv_base = (uint8_t*)(in_extra ? params.extra_kv : params.kv);
                            uint8_t *scale_ptr = kv_base + page_block_size * tma_k_stride +
                                (int64_t)block_idx * src_block_stride +
                                idx_in_block * num_scales_each_token +
                                cta_idx * (num_scales_each_token / 2);
                            bool is_fp4_token = HAS_FP4_KV && in_extra;
                            if (!is_fp4_token) {
                                uint8_t *scale_dst = smem.decode_scales[indices_buf_idx] +
                                    row * NUM_SCALES_EACH_TOKEN_PER_CTA;
                                if constexpr (NUM_SCALES_EACH_TOKEN_PER_CTA == 4) {
                                    uint32_t scales;
                                    asm volatile (
                                        "ld.global.nc.L1::no_allocate.b32 %0, [%1];"
                                        : "=r"(scales)
                                        : "l"((uint64_t)scale_ptr)
                                    );
                                    *(uint32_t*)scale_dst = is_index_valid ? scales : 0;
                                } else if constexpr (NUM_SCALES_EACH_TOKEN_PER_CTA == 8) {
                                    uint64_t scales;
                                    asm volatile (
                                        "ld.global.nc.L1::no_allocate.b64 %0, [%1];"
                                        : "=l"(scales)
                                        : "l"((uint64_t)scale_ptr)
                                    );
                                    *(uint64_t*)scale_dst = is_index_valid ? scales : 0;
                                } else {
                                    static_assert(NUM_SCALES_EACH_TOKEN_PER_CTA == 16);
                                    __int128_t scales;
                                    asm volatile (
                                        "ld.global.nc.L1::no_allocate.b128 %0, [%1];"
                                        : "=q"(scales)
                                        : "l"((uint64_t)scale_ptr)
                                    );
                                    *(__int128_t*)scale_dst = is_index_valid ? scales : 0;
                                }
                            } else {
                                smem.decode_scale_ptrs[indices_buf_idx][row] = is_index_valid ? scale_ptr : nullptr;
                            }
                            uint32_t mask = __ballot_sync(0xFFFFFFFF, is_index_valid);
                            if (lane_idx == 0) {
                                *((uint32_t*)smem.is_k_valid[indices_buf_idx] + i) = mask;
                            }
                        }

                        smem.bar_indices_full[indices_buf_idx].arrive();
                        rs.update();
                    });
                }
                cur_job = get_next_job(cur_job);
            } while (cur_job.is_valid);
        } else if (warp_idx == 11) {
            if constexpr (FP8_RAW_PIPE) {
                if (elect_one_sync()) {
                    OuterloopArgs cur_job = get_first_job();
                    RingBufferState rs;
                    do {
                        run_along_kv_blocks(cur_job, [&]<KVLocation LOC>(uint32_t kv_block_idx) {
                            auto [index_slot, index_phase] = rs.get<NUM_INDICES_BUFS>();
                            smem.bar_indices_full[index_slot].wait(index_phase);
                            smem.bar_raw_kv_empty[0].wait(rs.get<1>().second ^ 1);
                            CUTE_UNROLL
                            for (uint32_t row = 0; row < B_TOPK; row += 4) {
                                bool extra = LOC == KVLocation::EXTRA ||
                                    (LOC == KVLocation::ORIG_AND_EXTRA && kv_block_idx * B_TOPK + row >= cur_job.num_orig_slots);
                                int4 coords = *(int4*)(smem.decode_tma_coords[index_slot] + row);
                                if constexpr (HAS_FP4_KV) {

                                    auto tensor_map = extra ? &tma_params.tensor_map_extra_kv_fp4_part_cta0 :
                                        &tma_params.tensor_map_kv_fp8_part_cta0;
                                    ku::tma_gather4(tensor_map, smem.bar_raw_kv_full,
                                        smem.raw_kv + row / 4 * RAW_KV_GROUP_BYTES, 0, coords,
                                        (int64_t)TMA::CacheHintSm90::EVICT_FIRST);
                                } else {
                                    auto tensor_map = extra ? &tma_params.tensor_map_extra_kv_fp8_part_cta0 :
                                        &tma_params.tensor_map_kv_fp8_part_cta0;
                                    ku::tma_gather4(tensor_map, smem.bar_raw_kv_full,
                                        smem.raw_kv + row * RAW_FP8_TOKEN_SMEM_STRIDE_CTA0, 0, coords,
                                        (int64_t)TMA::CacheHintSm90::EVICT_FIRST);
                                }
                            }
                            smem.bar_indices_empty[index_slot].arrive();
                            if constexpr (HAS_FP4_KV) {
                                uint32_t num_orig_rows = LOC == KVLocation::ORIG ? B_TOPK :
                                    (LOC == KVLocation::EXTRA ? 0u : cur_job.num_orig_slots - kv_block_idx * B_TOPK);
                                smem.bar_raw_kv_full.arrive_and_expect_tx(num_orig_rows * OrigKVFormat::RAW_TOKEN_SMEM_STRIDE +
                                    (B_TOPK - num_orig_rows) * ExtraKVFormat::RAW_TOKEN_SMEM_STRIDE);
                            } else {
                                smem.bar_raw_kv_full.arrive_and_expect_tx(B_TOPK * RAW_FP8_TOKEN_SMEM_STRIDE_CTA0);
                            }
                            rs.update();
                        });
                        cur_job = get_next_job(cur_job);
                    } while (cur_job.is_valid);
                }
            }
        }
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_alloc<128>();
        if constexpr (IS_DECODE) {
            if constexpr (HAS_FP4_KV) {

                static_assert(!OrigKVFormat::IS_FP4 && ExtraKVFormat::IS_FP4 && ExtraKVFormat::D_BF16 == 0);
                uint32_t local_warp_idx = warp_idx - 4;
                static constexpr uint32_t NUM_DEQUANT_WARPS = 4, NUM_LANES_PER_ROW = 4;
                static constexpr uint32_t NUM_ROWS_PER_WAVEFRONT = 32 / NUM_LANES_PER_ROW;
                static constexpr uint32_t NUM_TOKENS_PER_THREAD = B_TOPK / (NUM_DEQUANT_WARPS * NUM_ROWS_PER_WAVEFRONT);
                static_assert(B_TOPK % (NUM_DEQUANT_WARPS * NUM_ROWS_PER_WAVEFRONT) == 0);
                static constexpr uint32_t NUM_ROWS_PER_WARP = NUM_TOKENS_PER_THREAD * NUM_ROWS_PER_WAVEFRONT;
                static constexpr uint32_t MAX_CHUNKS_PER_LANE = OrigKVFormat::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;
                const uint32_t row_in_wavefront = lane_idx % NUM_ROWS_PER_WAVEFRONT, idx_in_row = lane_idx / NUM_ROWS_PER_WAVEFRONT;
                const uint32_t chunk_rotation = row_in_wavefront / 4 * 4;
                auto get_row_idx = [&](uint32_t token_idx) {
                    return local_warp_idx * NUM_ROWS_PER_WARP + token_idx * NUM_ROWS_PER_WAVEFRONT + row_in_wavefront;
                };

                auto get_chunk_base = [&]<typename F>(uint32_t half) {
                    constexpr uint32_t NUM_CHUNKS_PER_LANE = F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;
                    static_assert(NUM_CHUNKS_PER_LANE == 2 || NUM_CHUNKS_PER_LANE == 4 || NUM_CHUNKS_PER_LANE == 8);
                    if constexpr (NUM_CHUNKS_PER_LANE <= 4) {
                        return (idx_in_row * NUM_CHUNKS_PER_LANE + chunk_rotation) % F::NUM_CHUNKS_PER_ROW;
                    } else {
                        return idx_in_row * NUM_CHUNKS_PER_LANE + (half * 4 ^ chunk_rotation);
                    }
                };
                auto get_raw_row_offset = [&](uint32_t row) {
                    return row / 4 * RAW_KV_GROUP_BYTES;
                };

                static constexpr uint32_t STS_ATOM_COL_STRIDE_BYTES = B_TOPK * 128;
                static constexpr uint32_t STS_TOKEN_STRIDE_BYTES = NUM_ROWS_PER_WAVEFRONT * 128;
                uint32_t sts_offsets[8];
                {
                    Tensor sKV = make_tensor(make_smem_ptr(smem.kv_slots[0]), ku::make_umma_canonical_k_major_layout<B_TOPK, D_QK / CLUSTER_SIZE, 128>());
                    CUTE_UNROLL
                    for (uint32_t i = 0; i < 8; ++i) {
                        sts_offsets[i] = (uint32_t)((&sKV(get_row_idx(0), i * 8) - smem.kv_slots[0]) * sizeof(bf16));
                    }
                }

                OuterloopArgs cur_job = get_first_job();
                RingBufferState rs;
                do {

                    auto process_block = [&]<uint32_t NUM_FP8_TOKENS>(uint32_t num_orig_rows) {

                        auto for_each_token = [&](auto callable) {
                            cute::for_each(cute::make_int_sequence<NUM_TOKENS_PER_THREAD>{}, [&](auto token_idx) {
                                if constexpr (token_idx < NUM_FP8_TOKENS) {
                                    callable.template operator()<OrigKVFormat>(token_idx);
                                } else {
                                    callable.template operator()<ExtraKVFormat>(token_idx);
                                }
                            });
                        };

                        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                        smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);

                        static constexpr uint32_t FP8_SCALE_BYTES_PER_LANE = OrigKVFormat::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW / 2;
                        static constexpr uint32_t FP4_SCALE_BYTES_PER_LANE = ExtraKVFormat::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW * 2;
                        static_assert(FP8_SCALE_BYTES_PER_LANE <= FP4_SCALE_BYTES_PER_LANE && FP4_SCALE_BYTES_PER_LANE % 4 == 0);
                        uint32_t cached_scales[NUM_TOKENS_PER_THREAD][FP4_SCALE_BYTES_PER_LANE / 4];
                        const uint8_t *scale_ptrs[NUM_TOKENS_PER_THREAD];
                        for_each_token([&]<typename F>(uint32_t token_idx) {
                            uint32_t row = get_row_idx(token_idx);
                            if constexpr (F::IS_FP4) {
                                scale_ptrs[token_idx] = smem.decode_scale_ptrs[indices_buf_idx][row];
                            } else {

                                const uint8_t *scale_src = smem.decode_scales[indices_buf_idx] + row * NUM_SCALES_EACH_TOKEN_PER_CTA;
                                if constexpr (FP8_SCALE_BYTES_PER_LANE == 4) {
                                    uint32_t scales = *(uint32_t*)(scale_src + idx_in_row * 4);
                                    cached_scales[token_idx][0] = chunk_rotation ? __byte_perm(scales, scales, 0x1032) : scales;
                                } else {
                                    static_assert(FP8_SCALE_BYTES_PER_LANE == 2);
                                    cached_scales[token_idx][0] = *(uint16_t*)(scale_src + get_chunk_base.template operator()<F>(0) / 2);
                                }
                            }
                        });

                        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
                        if constexpr (!FP8_RAW_PIPE) smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                        uint8_t *slot_base = (uint8_t*)smem.kv_slots[kv_slot_idx];

                        uint8_t *raw_base = FP8_RAW_PIPE ? (uint8_t*)smem.raw_kv : slot_base;

                        if constexpr (!FP8_RAW_PIPE) cute::for_each(cute::make_int_sequence<NUM_ROWS_PER_WARP / 4>{}, [&](auto i) {

                            uint32_t row_start = local_warp_idx * NUM_ROWS_PER_WARP + i * 4;
                            constexpr bool is_fp4 = i / (NUM_ROWS_PER_WAVEFRONT / 4) >= NUM_FP8_TOKENS;
                            auto tensor_map = is_fp4 ?
                                (cta_idx == 0 ? &tma_params.tensor_map_extra_kv_fp4_part_cta0 : &tma_params.tensor_map_extra_kv_fp4_part_cta1) :
                                (cta_idx == 0 ? &tma_params.tensor_map_kv_fp8_part_cta0 : &tma_params.tensor_map_kv_fp8_part_cta1);
                            if (elect_one_sync()) {
                                int4 coords = *(int4*)(smem.decode_tma_coords[indices_buf_idx] + row_start);
                                ku::tma_gather4(
                                    tensor_map,
                                    smem.bar_raw_kv_full,
                                    slot_base + get_raw_row_offset(row_start),
                                    0,
                                    coords,
                                    (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                                );
                            }
                        });
                        smem.bar_indices_empty[indices_buf_idx].arrive();

                        for_each_token([&]<typename F>(uint32_t token_idx) {
                            if constexpr (F::IS_FP4) {
                                uint64_t scale_addr = (uint64_t)(scale_ptrs[token_idx] + get_chunk_base.template operator()<F>(0) * 2);
                                if constexpr (FP4_SCALE_BYTES_PER_LANE == 8) {
                                    uint64_t scales = 0;
                                    if (scale_ptrs[token_idx] != nullptr) {
                                        asm volatile (
                                            "ld.global.nc.L1::no_allocate.b64 %0, [%1];"
                                            : "=l"(scales)
                                            : "l"(scale_addr)
                                            : "memory"
                                        );
                                    }
                                    *(uint64_t*)cached_scales[token_idx] = scales;
                                } else {
                                    static_assert(FP4_SCALE_BYTES_PER_LANE == 4);
                                    uint32_t scales = 0;
                                    if (scale_ptrs[token_idx] != nullptr) {
                                        asm volatile (
                                            "ld.global.nc.L1::no_allocate.b32 %0, [%1];"
                                            : "=r"(scales)
                                            : "l"(scale_addr)
                                            : "memory"
                                        );
                                    }
                                    cached_scales[token_idx][0] = scales;
                                }
                            }
                        });

                        if (!FP8_RAW_PIPE && idx_in_warpgroup == 0) {
                            smem.bar_raw_kv_full.arrive_and_expect_tx(num_orig_rows * OrigKVFormat::RAW_TOKEN_SMEM_STRIDE + (B_TOPK - num_orig_rows) * ExtraKVFormat::RAW_TOKEN_SMEM_STRIDE);
                        }
                        smem.bar_raw_kv_full.wait(rs.get<1>().second);

                        uint32_t cached_input[NUM_TOKENS_PER_THREAD][MAX_CHUNKS_PER_LANE][4];
                        for_each_token([&]<typename F>(uint32_t token_idx) {
                            uint32_t row = get_row_idx(token_idx);
                            uint8_t *row_base = raw_base + get_raw_row_offset(row) + row % 4 * F::RAW_TOKEN_SMEM_STRIDE;
                            CUTE_UNROLL
                            for (uint32_t g = 0; g < F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW; ++g) {
                                *(__int128_t*)(cached_input[token_idx][g]) = ku::ld_shared(
                                    row_base + get_chunk_base.template operator()<F>(g / 4) * 16 + g % 4 * 16
                                );
                            }
                        });
                        NamedBarrier::arrive_and_wait(128, 7);
                        if constexpr (FP8_RAW_PIPE) {

                            smem.bar_raw_kv_empty[0].arrive();
                            smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                        }

                        for_each_token([&]<typename F>(uint32_t token_idx) {
                            static constexpr uint32_t C_LANE = F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;
                            static constexpr uint32_t NUM_STS_PER_CHUNK = F::CHUNK_ELEMS * sizeof(bf16) / 16;
                            ku::nvbf16x2 fp4_scales[2];
                            CUTE_UNROLL
                            for (uint32_t g = 0; g < C_LANE; ++g) {
                                ku::nvbf16x2 data_bf16x2[F::CHUNK_ELEMS / 2];
                                if constexpr (F::IS_FP4) {
                                    if (g % 2 == 0) {
                                        fp8x4_to_bf16x2x2(cached_scales[token_idx][g / 2], fp4_scales);
                                    }
                                    CUTE_UNROLL
                                    for (uint32_t j = 0; j < 4; ++j) {
                                        fp4x8_to_bf16x2x4(cached_input[token_idx][g][j], data_bf16x2 + j * 4);
                                    }
                                    ku::nvbf16x2 scale_lo = __low2bfloat162(fp4_scales[g % 2]), scale_hi = __high2bfloat162(fp4_scales[g % 2]);
                                    CUTE_UNROLL
                                    for (uint32_t k = 0; k < F::CHUNK_ELEMS / 2; ++k) {
                                        data_bf16x2[k] = __hmul2(data_bf16x2[k], k < F::QUANT_TILE_SIZE / 2 ? scale_lo : scale_hi);
                                    }
                                } else {
                                    __nv_fp8_e8m0 scale = ((__nv_fp8_e8m0*)cached_scales[token_idx])[g * F::CHUNK_ELEMS / F::QUANT_TILE_SIZE];
                                    CUTE_UNROLL
                                    for (uint32_t k = 0; k < F::CHUNK_ELEMS / 2; ++k) {
                                        data_bf16x2[k] = fp8x2_to_bf16x2_with_scale(((ku::nve4m3x2*)cached_input[token_idx][g])[k], scale);
                                    }
                                }

                                static constexpr uint32_t CHUNKS_PER_ATOM_ROW = 128 / (F::CHUNK_ELEMS * 2);
                                uint32_t atom_col = get_chunk_base.template operator()<F>(g / 4) / CHUNKS_PER_ATOM_ROW + g % 4 / CHUNKS_PER_ATOM_ROW;
                                CUTE_UNROLL
                                for (uint32_t j = 0; j < NUM_STS_PER_CHUNK; ++j) {
                                    ku::st_shared(
                                        slot_base + sts_offsets[g % CHUNKS_PER_ATOM_ROW * NUM_STS_PER_CHUNK + j] + atom_col * STS_ATOM_COL_STRIDE_BYTES + token_idx * STS_TOKEN_STRIDE_BYTES,
                                        *(__int128_t*)(data_bf16x2 + j * 4)
                                    );
                                }
                            }
                        });

                        fence_view_async_shared();
                        arrive_on_cta0_barrier(smem.bar_kv_slot_full[kv_slot_idx]);
                        rs.update();
                    };

                    run_along_kv_blocks(cur_job, [&]<KVLocation LOC>(uint32_t kv_block_idx) {
                        if constexpr (LOC == KVLocation::ORIG) {
                            process_block.template operator()<NUM_TOKENS_PER_THREAD>(B_TOPK);
                        } else if constexpr (LOC == KVLocation::EXTRA) {
                            process_block.template operator()<0>(0);
                        } else {

                            uint32_t num_orig_rows = cur_job.num_orig_slots - kv_block_idx * B_TOPK;
                            uint32_t num_fp8_tokens = (uint32_t)std::clamp((int)num_orig_rows - (int)(local_warp_idx * NUM_ROWS_PER_WARP), 0, (int)NUM_ROWS_PER_WARP) / NUM_ROWS_PER_WAVEFRONT;
                            [&]<uint32_t... Ks>(std::integer_sequence<uint32_t, Ks...>) {
                                ((num_fp8_tokens == Ks ? process_block.template operator()<Ks>(num_orig_rows) : void()), ...);
                            }(std::make_integer_sequence<uint32_t, NUM_TOKENS_PER_THREAD + 1>{});
                        }
                    });

                    cur_job = get_next_job(cur_job);
                } while (cur_job.is_valid);
            } else {

                using fp8_e8m0 = __nv_fp8_e8m0;
                uint32_t local_warp_idx = warp_idx - 4;
                static constexpr uint32_t NUM_DEQUANT_WARPS = 4, NUM_ROWS_PER_WARP = 8;
                static constexpr uint32_t GROUP_SIZE = 4;
                static constexpr uint32_t NUM_COLS_PER_GROUP = D_VO / CLUSTER_SIZE / (GROUP_SIZE*16);
                static_assert((D_VO/CLUSTER_SIZE) % (GROUP_SIZE*16) == 0);
                uint32_t group_idx = lane_idx / GROUP_SIZE, idx_in_group = lane_idx % GROUP_SIZE;
                static constexpr uint32_t NUM_CHUNKS_PER_WARP = B_TOPK / (NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP);
                static_assert(B_TOPK % (NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP) == 0);
                static constexpr uint32_t NUM_TOKENS_PER_THREAD = NUM_CHUNKS_PER_WARP;

                auto get_row_idx = [&](uint32_t local_row_idx) {
                    return local_row_idx*NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP + local_warp_idx*NUM_ROWS_PER_WARP + group_idx;
                };
                auto [sts_base_offset_0, sts_base_offset_1] = [&] {
                    Tensor sKV = make_tensor(make_smem_ptr(smem.kv_slots[0]), ku::make_umma_canonical_k_major_layout<B_TOPK, D_QK, 128>());
                    return std::pair<uint32_t, uint32_t> {
                        (uint32_t)(&sKV(local_warp_idx*NUM_ROWS_PER_WARP+group_idx, idx_in_group*16 + 0) - smem.kv_slots[0]),
                        (uint32_t)(&sKV(local_warp_idx*NUM_ROWS_PER_WARP+group_idx, idx_in_group*16 + 8) - smem.kv_slots[0])
                    };
                }();

                uint32_t d_fp8_this_cta = cta_idx == 0 ? D_FP8_CTA0 : D_FP8_CTA1;
                uint32_t d_fp8_this_cta_padded = d_fp8_this_cta + ((cta_idx==0?IS_CTA0_RAW_KV_PADDED:IS_CTA1_RAW_KV_PADDED)?64:0);

                OuterloopArgs cur_job = get_first_job();
                RingBufferState rs;
                do {
                    auto loop_body = [&]<KVLocation LOC>(uint32_t kv_block_idx) {

                        auto is_pos_in_extra = [&](uint32_t pos) -> bool {
                            if constexpr (LOC == KVLocation::ORIG) {
                                return false;
                            } else if constexpr (LOC == KVLocation::EXTRA) {
                                return true;
                            } else {
                                return pos >= cur_job.num_orig_slots;
                            }
                        };

                        static constexpr uint32_t NUM_SCALED_EACH_TOKEN_LOCAL = NUM_SCALES_EACH_TOKEN / CLUSTER_SIZE;
                        fp8_e8m0 cached_scales[NUM_TOKENS_PER_THREAD][NUM_SCALED_EACH_TOKEN_LOCAL];
                        int cached_tma_coord[NUM_TOKENS_PER_THREAD];
                        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                        smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                        CUTE_UNROLL
                        for (uint32_t local_row_idx = 0; local_row_idx < NUM_TOKENS_PER_THREAD; ++local_row_idx) {
                            uint32_t row = get_row_idx(local_row_idx);
                            if constexpr (!FP8_RAW_PIPE) {
                                cached_tma_coord[local_row_idx] = smem.decode_tma_coords[indices_buf_idx][row];
                            }
                            uint8_t *scale_src = smem.decode_scales[indices_buf_idx] +
                                row * NUM_SCALES_EACH_TOKEN_PER_CTA;
                            if constexpr (NUM_SCALED_EACH_TOKEN_LOCAL == 4) {
                                *(uint32_t*)cached_scales[local_row_idx] = *(uint32_t*)scale_src;
                            } else if constexpr (NUM_SCALED_EACH_TOKEN_LOCAL == 8) {
                                *(uint64_t*)cached_scales[local_row_idx] = *(uint64_t*)scale_src;
                            } else {
                                static_assert(NUM_SCALED_EACH_TOKEN_LOCAL == 16);
                                *(__int128_t*)cached_scales[local_row_idx] = *(__int128_t*)scale_src;
                            }
                        }
                        smem.bar_indices_empty[indices_buf_idx].arrive();

                        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
                        if constexpr (!FP8_RAW_PIPE) smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);

                        if constexpr (!FP8_RAW_PIPE) {
                            int4 collected_tma_coords[NUM_TOKENS_PER_THREAD][2];
                            CUTE_UNROLL
                            for (uint32_t local_row_idx = 0; local_row_idx < NUM_TOKENS_PER_THREAD; ++local_row_idx) {
                                CUTE_UNROLL
                                for (uint32_t i = 0; i < 2; ++i) {

                                    uint32_t row_start = local_row_idx*NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP + local_warp_idx*NUM_ROWS_PER_WARP + i*4;
                                    bool group_in_extra = is_pos_in_extra(kv_block_idx*B_TOPK + row_start);
                                    auto tensor_map = group_in_extra ?
                                        (cta_idx == 0 ? &tma_params.tensor_map_extra_kv_fp8_part_cta0 : &tma_params.tensor_map_extra_kv_fp8_part_cta1) :
                                        (cta_idx == 0 ? &tma_params.tensor_map_kv_fp8_part_cta0 : &tma_params.tensor_map_kv_fp8_part_cta1);
                                    int4 coords;
                                    coords.x = i == 0 ? cached_tma_coord[local_row_idx] : __shfl_sync(0xFFFFFFFF, cached_tma_coord[local_row_idx], i*16);
                                    coords.y = __shfl_sync(0xFFFFFFFF, cached_tma_coord[local_row_idx], i*16+4);
                                    coords.z = __shfl_sync(0xFFFFFFFF, cached_tma_coord[local_row_idx], i*16+8);
                                    coords.w = __shfl_sync(0xFFFFFFFF, cached_tma_coord[local_row_idx], i*16+12);
                                    collected_tma_coords[local_row_idx][i] = coords;
                                    if (elect_one_sync()) {
                                        auto smem_ptr = (fp8_e4m3*)smem.kv_slots[kv_slot_idx] + row_start * d_fp8_this_cta_padded;
                                        ku::tma_gather4(
                                            tensor_map,
                                            smem.bar_raw_kv_full,
                                            smem_ptr,
                                            0,
                                            coords,
                                            (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                                        );
                                    }
                                }
                            }

                            if (D_BF16 > 0 && cta_idx+1 == CLUSTER_SIZE && elect_one_sync()) {
                                CUTE_UNROLL
                                for (uint32_t local_row_idx = 0; local_row_idx < NUM_TOKENS_PER_THREAD; ++local_row_idx) {
                                    CUTE_UNROLL
                                    for (uint32_t i = 0; i < 2; ++i) {
                                        uint32_t row_start = local_row_idx*NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP + local_warp_idx*NUM_ROWS_PER_WARP + i*4;

                                        auto tensor_map = is_pos_in_extra(kv_block_idx*B_TOPK + row_start) ? &tma_params.tensor_map_extra_kv_bf16_part : &tma_params.tensor_map_kv_bf16_part;
                                        auto smem_ptr = smem.kv_slots[kv_slot_idx] + (IS_2CTA ? (D_FP8-D_VO/2)/64 : D_FP8/64) * B_TOPK * 64 + row_start * 64;
                                        if constexpr (IS_2CTA) {
                                            ku::tma_gather4_cta_group_2<true>(
                                                tensor_map,
                                                smem.bar_kv_slot_full[kv_slot_idx],
                                                smem_ptr,
                                                0,
                                                collected_tma_coords[local_row_idx][i],
                                                (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                                            );
                                        } else {
                                            ku::tma_gather4(
                                                tensor_map,
                                                smem.bar_kv_slot_full[kv_slot_idx],
                                                smem_ptr,
                                                0,
                                                collected_tma_coords[local_row_idx][i],
                                                (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                                            );
                                        }
                                    }
                                }
                            }

                            if (idx_in_warpgroup == 0) {
                                smem.bar_raw_kv_full.arrive_and_expect_tx(B_TOPK*d_fp8_this_cta_padded*sizeof(fp8_e4m3));
                            }
                        }
                        smem.bar_raw_kv_full.wait(rs.get<1>().second);

                        fp8_e4m3 cached_input[NUM_TOKENS_PER_THREAD][NUM_COLS_PER_GROUP][16];
                        CUTE_UNROLL
                        for (uint32_t local_row_idx = 0; local_row_idx < NUM_TOKENS_PER_THREAD; ++local_row_idx) {
                            uint32_t row = get_row_idx(local_row_idx);
                            for (uint32_t local_col_idx = 0; local_col_idx < NUM_COLS_PER_GROUP; ++local_col_idx) {
                                if (MODEL_TYPE == ModelType::V4 && cta_idx+1 == CLUSTER_SIZE && local_col_idx+1 == NUM_COLS_PER_GROUP) {

                                    continue;
                                }
                                *(__int128_t*)(cached_input[local_row_idx][local_col_idx]) = ku::ld_shared(
                                    (FP8_RAW_PIPE ? (fp8_e4m3*)smem.raw_kv : (fp8_e4m3*)smem.kv_slots[kv_slot_idx]) +
                                    row*d_fp8_this_cta_padded +
                                    local_col_idx*GROUP_SIZE*16 +
                                    idx_in_group*16
                                );
                            }
                        }
                        NamedBarrier::arrive_and_wait(128, 7);
                        if constexpr (FP8_RAW_PIPE) {

                            smem.bar_raw_kv_empty[0].arrive();
                            smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                        }

                        CUTE_UNROLL
                        for (uint32_t local_row_idx = 0; local_row_idx < NUM_TOKENS_PER_THREAD; ++local_row_idx) {
                            CUTE_UNROLL
                            for (uint32_t local_col_idx = 0; local_col_idx < NUM_COLS_PER_GROUP; ++local_col_idx) {
                                if (MODEL_TYPE == ModelType::V4 && cta_idx+1 == CLUSTER_SIZE && local_col_idx+1 == NUM_COLS_PER_GROUP) {

                                    continue;
                                }
                                ku::nve4m3x2 data_fp8x2[8];
                                ku::nvbf16x2 data_bf16x2[8];
                                *(__int128_t*)data_fp8x2 = *(__int128_t*)(cached_input[local_row_idx][local_col_idx]);
                                static_assert(KV_QUANT_TILE_SIZE == 64 || KV_QUANT_TILE_SIZE == 32);
                                uint32_t scale_idx = KV_QUANT_TILE_SIZE == 64 ? local_col_idx : local_col_idx*2 + (idx_in_group >= GROUP_SIZE/2);
                                CUTE_UNROLL
                                for (uint32_t j = 0; j < 8; ++j) {
                                    data_bf16x2[j] = fp8x2_to_bf16x2_with_scale(data_fp8x2[j], cached_scales[local_row_idx][scale_idx]);
                                }
                                ku::st_shared(
                                    smem.kv_slots[kv_slot_idx] + sts_base_offset_0 + local_row_idx*(NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP*64) + local_col_idx*(B_TOPK*GROUP_SIZE*16),
                                    *(__int128_t*)(data_bf16x2 + 0)
                                );
                                ku::st_shared(
                                    smem.kv_slots[kv_slot_idx] + sts_base_offset_1 + local_row_idx*(NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP*64) + local_col_idx*(B_TOPK*GROUP_SIZE*16),
                                    *(__int128_t*)(data_bf16x2 + 4)
                                );
                            }
                        }

                        fence_view_async_shared();
                        arrive_on_cta0_barrier(smem.bar_kv_slot_full[kv_slot_idx]);
                        rs.update();
                    };

                    run_along_kv_blocks(cur_job, loop_body);

                    cur_job = get_next_job(cur_job);
                } while (cur_job.is_valid);
            }
        } else {

            if (elect_one_sync()) {
                OuterloopArgs cur_job = get_first_job();
                RingBufferState rs;
                uint32_t local_warp_idx = warp_idx - 4;
                do {
                    for (uint32_t i = 0; i < cur_job.num_kv_blocks; ++i) {
                        static_assert(B_TOPK % (4*4) == 0);
                        static constexpr uint32_t NUM_ROW_PER_WARP = B_TOPK / 4;
                        int4 topk_idxs[NUM_ROW_PER_WARP / 4];
                        CUTE_UNROLL
                        for (uint32_t local_row = 0; local_row < NUM_ROW_PER_WARP / 4; local_row += 1) {
                            uint32_t row = local_row * 4 * 4 + local_warp_idx * 4;
                            uint32_t pos = i * B_TOPK + row;

                            topk_idxs[local_row] = pos < cur_job.topk_length
                                ? __ldg((int4*)(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos))
                                : int4{-1, -1, -1, -1};
                        }

                        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
                        smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                        CUTE_UNROLL
                        for (uint32_t local_row = 0; local_row < NUM_ROW_PER_WARP / 4; local_row += 1) {
                            uint32_t row = local_row * 4 * 4 + local_warp_idx * 4;
                            CUTE_UNROLL
                            for (uint32_t tile_idx = 0; tile_idx < (CLUSTER_SIZE == 2 ? (D_QK/2/64) : (D_QK/64)); ++tile_idx) {

                                if constexpr (CLUSTER_SIZE == 1) {
                                    ku::tma_gather4(
                                        &tma_params.tensor_map_kv,
                                        smem.bar_kv_slot_full[kv_slot_idx],
                                        smem.kv_slots[kv_slot_idx] + tile_idx * B_TOPK * 64 + row * 64,
                                        tile_idx * 64,
                                        topk_idxs[local_row],
                                        (int64_t)TMA::CacheHintSm90::EVICT_LAST
                                    );
                                } else {
                                    ku::tma_gather4_cta_group_2<true>(
                                        &tma_params.tensor_map_kv,
                                        smem.bar_kv_slot_full[kv_slot_idx],
                                        smem.kv_slots[kv_slot_idx] + tile_idx * B_TOPK * 64 + row * 64,
                                        tile_idx * 64 + cta_idx * (D_QK/2),
                                        topk_idxs[local_row],
                                        (int64_t)TMA::CacheHintSm90::EVICT_LAST
                                    );
                                }
                            }
                        }
                        rs.update();
                    }
                    cur_job = get_next_job(cur_job);
                } while (cur_job.is_valid);
            }
        }
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

template<typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
fwd_kernel(__grid_constant__ const typename Kernel::Params params, __grid_constant__ const typename Kernel::TMAParams tma_params, __grid_constant__ const typename Kernel::AuxParams aux_params) {
    Kernel::devfunc(params, tma_params, aux_params);
}

template<Config CONFIG>
void Kernel<CONFIG>::run(const Params &params) {
    KU_ASSERT(params.h_q == H_Q);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_VO);
    KU_ASSERT(params.stride_indices_s_q*sizeof(int) % 32 == 0, "indices.stride(0) must be 32B aligned, got %d elements (%d bytes)", params.stride_indices_s_q, (int)(params.stride_indices_s_q*sizeof(int)));

    KU_ASSERT(params.enable_q_norm == ENABLE_Q_NORM);
    KU_ASSERT(params.is_rope_neox_style == false && params.rope_dim == 64);
    KU_ASSERT(params.wv_group_size == WV_GROUP_SIZE);
    KU_ASSERT(params.num_per_channels == O_QUANT_TILE_SIZE);
    KU_ASSERT(params.use_tma_aligned_col_major_sf == true && params.round_sf == true && params.use_packed_ue8m0 == true);

    TMAParams tma_params = {};
    if constexpr (IS_DECODE) {
        KU_ASSERT(params.b == 1, "Only batch size 1 is supported for the fused decoding kernel");
        KU_ASSERT(params.model_type == MODEL_TYPE && params.extra_model_type == EXTRA_MODEL_TYPE);
        if (params.extra_topk > 0) {

            KU_ASSERT(params.topk % (HAS_FP4_KV ? 8 : 4) == 0, "topk (%d) must be a multiple of %d when the extra KV cache is used", params.topk, HAS_FP4_KV ? 8 : 4);
        }
        auto make_fp4_kv_tensor_map = [](bool is_extra, void *kv_ptr, int num_blocks, int64_t block_stride_bytes, int row_stride_bytes) -> std::pair<CUtensorMap, CUtensorMap> {
            using F = ExtraKVFormat;
            KU_ASSERT((int64_t)kv_ptr % 16 == 0, "The base address of %skv (%p) must be 16B aligned", is_extra?"extra_":"", kv_ptr);
            KU_ASSERT(row_stride_bytes == (int)F::BYTES_PER_TOKEN, "%skv_cache.stride(-2) (%d) must be %d, i.e. each page block in the KV cache must be contiguous", is_extra?"extra_":"", row_stride_bytes, (int)F::BYTES_PER_TOKEN);
            KU_ASSERT(block_stride_bytes % F::TMA_K_STRIDE == 0, "%skv_cache.stride(0) (%ld) must be a multiple of %d. Padding might be necessary", is_extra?"extra_":"", block_stride_bytes, (int)F::TMA_K_STRIDE);
            KU_ASSERT((uint64_t)num_blocks * (uint64_t)(block_stride_bytes / F::TMA_K_STRIDE) <= INT32_MAX, "%skv: too many rows for the int32 TMA coordinates", is_extra?"extra_":"");

            static_assert(F::RAW_TOKEN_DATA_BYTES % 4 == 0);
            auto make_tensor_map_for_cta = [&](uint32_t cta_idx) {
                return ku::make_tensor_map(
                    {(uint64_t)F::RAW_TOKEN_DATA_BYTES/4, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / F::TMA_K_STRIDE)},
                    {(uint64_t)F::TMA_K_STRIDE},
                    {F::RAW_TOKEN_SMEM_STRIDE/4, 1},
                    (uint8_t*)kv_ptr + cta_idx * F::RAW_TOKEN_DATA_BYTES,
                    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
                );
            };
            CUtensorMap fp4_part_tensor_map_cta1 = {};
            if constexpr (IS_2CTA) {
                fp4_part_tensor_map_cta1 = make_tensor_map_for_cta(1);
            }
            return {make_tensor_map_for_cta(0), fp4_part_tensor_map_cta1};
        };
        auto make_kv_tensor_map = [](bool is_extra, void *kv_ptr, int num_blocks, int64_t block_stride_bytes, int row_stride_bytes) -> std::tuple<CUtensorMap, CUtensorMap, CUtensorMap> {
            KU_ASSERT((int64_t)kv_ptr % 16 == 0, "The base address of %skv (%p) must be 16B aligned", is_extra?"extra_":"", kv_ptr);
            KU_ASSERT(row_stride_bytes == (int)KV_CACHE_BYTES_PER_TOKEN, "%skv_cache.stride(-2) (%d) must be %d, i.e. each page block in the KV cache must be contiguous", is_extra?"extra_":"", row_stride_bytes, (int)KV_CACHE_BYTES_PER_TOKEN);
            KU_ASSERT(block_stride_bytes % TMA_K_STRIDE == 0, "%skv_cache.stride(0) (%ld) must be a multiple of %d. Padding might be necessary", is_extra?"extra_":"", block_stride_bytes, (int)TMA_K_STRIDE);
            KU_ASSERT((uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TMA_K_STRIDE) <= INT32_MAX, "%skv: too many rows for the int32 TMA coordinates", is_extra?"extra_":"");
            static_assert(D_FP8_CTA0%4 == 0);
            auto fp8_part_tensor_map_cta0 = ku::make_tensor_map(
                {(uint64_t)D_FP8_CTA0/4, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TMA_K_STRIDE)},
                {(uint64_t)TMA_K_STRIDE},
                {RAW_FP8_TOKEN_SMEM_STRIDE_CTA0/4, 1},
                (uint8_t*)kv_ptr,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
            );
            CUtensorMap fp8_part_tensor_map_cta1 = {};
            if constexpr (IS_2CTA) {
                static_assert(D_FP8_CTA1%4 == 0);
                fp8_part_tensor_map_cta1 = ku::make_tensor_map(
                    {(uint64_t)D_FP8_CTA1/4, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TMA_K_STRIDE)},
                    {(uint64_t)TMA_K_STRIDE},
                    {RAW_FP8_TOKEN_SMEM_STRIDE_CTA1/4, 1},
                    (uint8_t*)kv_ptr + D_FP8_CTA0,
                    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
                );
            }
            CUtensorMap bf16_part_tensor_map = {};
            if constexpr (D_BF16 > 0) {
                bf16_part_tensor_map = ku::make_tensor_map(
                    {(uint64_t)D_BF16, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TMA_K_STRIDE)},
                    {(uint64_t)TMA_K_STRIDE},
                    {D_BF16, 1},
                    (uint8_t*)kv_ptr + D_FP8,
                    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                );
            }
            return {fp8_part_tensor_map_cta0, fp8_part_tensor_map_cta1, bf16_part_tensor_map};
        };
        {
            auto [fp8_part_cta0, fp8_part_cta1, bf16_part] = make_kv_tensor_map(false, params.kv, params.num_blocks, params.stride_kv_block, params.stride_kv_row);
            tma_params.tensor_map_kv_fp8_part_cta0 = fp8_part_cta0;
            tma_params.tensor_map_kv_fp8_part_cta1 = fp8_part_cta1;
            tma_params.tensor_map_kv_bf16_part = bf16_part;
        }
        if (params.extra_topk > 0) {
            if constexpr (HAS_FP4_KV) {
                auto [fp4_part_cta0, fp4_part_cta1] = make_fp4_kv_tensor_map(true, params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block, params.stride_extra_kv_row);
                tma_params.tensor_map_extra_kv_fp4_part_cta0 = fp4_part_cta0;
                tma_params.tensor_map_extra_kv_fp4_part_cta1 = fp4_part_cta1;
            } else {
                auto [fp8_part_cta0, fp8_part_cta1, bf16_part] = make_kv_tensor_map(true, params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block, params.stride_extra_kv_row);
                tma_params.tensor_map_extra_kv_fp8_part_cta0 = fp8_part_cta0;
                tma_params.tensor_map_extra_kv_fp8_part_cta1 = fp8_part_cta1;
                tma_params.tensor_map_extra_kv_bf16_part = bf16_part;
            }
        }
    } else {
        tma_params.tensor_map_kv = ku::make_tensor_map(
            {(uint64_t)D_QK, (uint64_t)params.s_kv},
            {(uint64_t)params.stride_kv_s_kv * sizeof(bf16)},
            {64, 1},
            params.kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    auto aux_params = AuxParams {
    };
    if constexpr (IS_DECODE) {
        aux_params.fast_divmod_page_block_size = cutlass::FastDivmod(params.page_block_size);
        aux_params.fast_divmod_extra_page_block_size = cutlass::FastDivmod(params.extra_kv != nullptr ? params.extra_page_block_size : 1);
    }

    auto kernel = &fwd_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3(params.s_q * CLUSTER_SIZE, 1, 1),
        dim3(NUM_THREADS, 1, 1),
        dim3(CLUSTER_SIZE, 1, 1),
        smem_size,
        params.stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params, aux_params
    ));
}

template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params) {

    using KernelType = Kernel<CONFIG>;
    KernelType::run(params);
}

}
""",
    "pattn/config.h": r"""#pragma once

#include <cutlass/fast_math.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "kernels/defines.h"
#include "kernels/params.h"
#include "kernels/kv_cache_format.h"

#include "kernel.h"

namespace pattn::sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

using namespace cute;

template<Config CONFIG>
struct Kernel {

static constexpr SparseAttnFwdMode FWD_MODE = CONFIG.FWD_MODE;
static constexpr ModelType MODEL_TYPE = CONFIG.MODEL_TYPE;
static constexpr ModelType EXTRA_MODEL_TYPE = CONFIG.EXTRA_MODEL_TYPE;
static constexpr uint32_t H_Q = CONFIG.H_Q;

using Params = ParamT<FWD_MODE>;

static_assert(FWD_MODE == SparseAttnFwdMode::Prefill || FWD_MODE == SparseAttnFwdMode::Decode);
static_assert(H_Q == 64 || H_Q == 128);

static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;

static constexpr uint32_t D_QK = 512;
static constexpr uint32_t D_VO = 512;
static constexpr uint32_t O_QUANT_TILE_SIZE = 32;
static constexpr uint32_t D_ROPE = 64;
static constexpr uint32_t D_NOPE = D_QK - D_ROPE;
static constexpr uint32_t WV_GROUP_SIZE = 8;
static constexpr bool ENABLE_Q_NORM = CONFIG.ENABLE_Q_NORM;
static constexpr bool H64_GENERIC = CONFIG.H64_GENERIC;
static_assert(!H64_GENERIC || (H_Q == 64 && !ENABLE_Q_NORM));
static constexpr bool FP8_RAW_PIPE = CONFIG.RAW_PIPE;

static_assert(!FP8_RAW_PIPE || (IS_DECODE && H_Q == 64 && !ENABLE_Q_NORM && MODEL_TYPE == ModelType::V41 &&
    (EXTRA_MODEL_TYPE == ModelType::V41 || EXTRA_MODEL_TYPE == ModelType::V41_FP4)));
static_assert(!CONFIG.MMA11 || !FP8_RAW_PIPE);
static constexpr uint32_t MMA_WARP = FP8_RAW_PIPE ? 9 : (CONFIG.MMA11 ? 11 : 8);
static constexpr uint32_t CLC_WARP = FP8_RAW_PIPE ? 8 : 9;
static constexpr bool PRED_IDX = CONFIG.PRED_IDX;
static_assert(!PRED_IDX || !IS_DECODE);

static constexpr uint32_t CLUSTER_SIZE = ku::ceil_div((uint32_t)H_Q, 64u);
static constexpr uint32_t IS_2CTA = CLUSTER_SIZE == 2;

static constexpr uint32_t B_TOPK = CLUSTER_SIZE == 2 ? 96 : 64;
static constexpr uint32_t H_Q_PER_CTA = 64;

template<ModelType MT>
struct KVFormat {
    using Base = KVCacheFormat<MT>;
    static constexpr bool IS_FP4 = Base::IS_FP4;
    static constexpr uint32_t D_FP4 = Base::D_FP4;
    static constexpr uint32_t D_FP8 = Base::D_FP8;
    static constexpr uint32_t D_BF16 = Base::D_BF16;
    static constexpr uint32_t QUANT_TILE_SIZE = Base::QUANT_TILE_SIZE;
    static constexpr uint32_t NUM_SCALES_EACH_TOKEN = Base::NUM_SCALES_EACH_TOKEN;
    static constexpr uint32_t TMA_K_STRIDE = Base::TMA_K_STRIDE;
    static constexpr uint32_t BYTES_PER_TOKEN = Base::BYTES_PER_TOKEN;

    static constexpr uint32_t RAW_TOKEN_DATA_BYTES = Base::QUANT_BYTES / CLUSTER_SIZE;
    static constexpr uint32_t RAW_TOKEN_SMEM_STRIDE = RAW_TOKEN_DATA_BYTES + 16;
    static constexpr uint32_t NUM_CHUNKS_PER_ROW = RAW_TOKEN_DATA_BYTES / 16;
    static constexpr uint32_t CHUNK_ELEMS = Base::IS_FP4 ? 32 : 16;
};
using OrigKVFormat = KVFormat<MODEL_TYPE>;
using ExtraKVFormat = KVFormat<EXTRA_MODEL_TYPE>;
static_assert(is_valid_kv_format_pair(MODEL_TYPE, EXTRA_MODEL_TYPE));
static constexpr bool HAS_FP4_KV = ExtraKVFormat::IS_FP4;

static constexpr uint32_t D_FP8 = OrigKVFormat::D_FP8;
static constexpr uint32_t D_BF16 = OrigKVFormat::D_BF16;
static constexpr uint32_t KV_QUANT_TILE_SIZE = OrigKVFormat::QUANT_TILE_SIZE;
static constexpr uint32_t NUM_SCALES_EACH_TOKEN = OrigKVFormat::NUM_SCALES_EACH_TOKEN;
static constexpr uint32_t NUM_SCALES_EACH_TOKEN_PER_CTA = NUM_SCALES_EACH_TOKEN / CLUSTER_SIZE;
static constexpr uint32_t TMA_K_STRIDE = OrigKVFormat::TMA_K_STRIDE;
static constexpr uint32_t KV_CACHE_BYTES_PER_TOKEN = OrigKVFormat::BYTES_PER_TOKEN;

static constexpr uint32_t RAW_KV_GROUP_BYTES = ku::ceil_div(4 * std::max(OrigKVFormat::RAW_TOKEN_SMEM_STRIDE, ExtraKVFormat::RAW_TOKEN_SMEM_STRIDE), 128u) * 128;
static_assert(!HAS_FP4_KV || B_TOPK / 4 * RAW_KV_GROUP_BYTES <= B_TOPK * D_QK / CLUSTER_SIZE * sizeof(bf16));

static constexpr uint32_t D_FP8_CTA0 = CLUSTER_SIZE == 1 ? D_FP8 : D_VO/2;
static constexpr uint32_t D_FP8_CTA1 = D_FP8 - D_FP8_CTA0;
static constexpr bool IS_CTA0_RAW_KV_PADDED = D_FP8_CTA0 % 128 == 0;
static constexpr bool IS_CTA1_RAW_KV_PADDED = D_FP8_CTA1 % 128 == 0;

static constexpr uint32_t RAW_FP8_TOKEN_SMEM_STRIDE_CTA0 = HAS_FP4_KV ? OrigKVFormat::RAW_TOKEN_SMEM_STRIDE : D_FP8_CTA0 + (IS_CTA0_RAW_KV_PADDED ? 64 : 0);
static constexpr uint32_t RAW_FP8_TOKEN_SMEM_STRIDE_CTA1 = HAS_FP4_KV ? OrigKVFormat::RAW_TOKEN_SMEM_STRIDE : D_FP8_CTA1 + (IS_CTA1_RAW_KV_PADDED ? 64 : 0);

static constexpr uint32_t NUM_THREADS = 512;

static_assert(!FP8_RAW_PIPE || (CLUSTER_SIZE == 1 && D_FP8 == 512 && D_BF16 == 0));
static constexpr uint32_t NUM_WORKING_THREADS = (
    CLUSTER_SIZE == 1 ? (
        IS_DECODE ?
        128 + 128 + (1+1+32) + 128 :
        128 + 128 + (1+1+32) + 4
    ) : (
        IS_DECODE ?
        128*2 + 128*2 + (1+2+32*2) + 128*2 :
        128*2 + 128*2 + (1+2+32*2) + 4*2
    )) + (FP8_RAW_PIPE ? 1 : 0);

static constexpr uint32_t FOLD_FACTOR = 128 / H_Q_PER_CTA;

static_assert(FOLD_FACTOR == 2 || FOLD_FACTOR == 4);
static_assert(H_Q_PER_CTA == 32 || H_Q_PER_CTA == 64);
static constexpr uint32_t NUM_MRGEMM_RAILS = 2;
static constexpr uint32_t NUM_P_ELEMS_PER_THREAD = H_Q_PER_CTA * B_TOPK / 128;

static constexpr uint32_t NUM_KV_SLOTS = FP8_RAW_PIPE ? 2 : 3;
static constexpr bool REUSE_KV_COMPLETION = CONFIG.REUSE_KV;
static_assert(!REUSE_KV_COMPLETION || (CLUSTER_SIZE == 1 && NUM_KV_SLOTS == 3));
static constexpr uint32_t NUM_INDICES_BUFS = 4;
static constexpr uint32_t NUM_P_BUFS = CLUSTER_SIZE == 2 ? 1 : 2;
static constexpr uint32_t NEED_TP_EMPTY_BAR = NUM_P_BUFS == 1;

struct tmem_cols {
    static constexpr uint32_t O = 0;
    static constexpr uint32_t Q = O + D_VO / FOLD_FACTOR;
    static constexpr uint32_t P_0 = Q + D_QK / NUM_MRGEMM_RAILS / 2;
    static constexpr uint32_t P_1 = P_0 + B_TOPK*NUM_MRGEMM_RAILS/FOLD_FACTOR;

    static constexpr uint32_t get_p(const uint32_t &p_buf_idx) {
        if constexpr (NUM_P_BUFS == 1) {
            return P_0;
        } else if constexpr (NUM_P_BUFS == 2) {
            return p_buf_idx ? P_1 : P_0;
        } else {
            static_assert(NUM_P_BUFS == 1 || NUM_P_BUFS == 2);
        }
    }
    static_assert(get_p(NUM_P_BUFS-1) + B_TOPK*NUM_MRGEMM_RAILS/FOLD_FACTOR <= 512);
};

using MMAAtom_QK = cute::conditional_t<
    CLUSTER_SIZE == 2,
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, H_Q, B_TOPK * NUM_MRGEMM_RAILS, UMMA::Major::K, UMMA::Major::K>,
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, H_Q, B_TOPK * NUM_MRGEMM_RAILS, UMMA::Major::K, UMMA::Major::K>
>;
using TiledMMA_QK = decltype(make_tiled_mma(MMAAtom_QK{}));
using TiledMMA_SV = cute::conditional_t<
    CLUSTER_SIZE == 2,
    decltype(make_tiled_mma(
        SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, 128, 256, UMMA::Major::K, UMMA::Major::MN>{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}
    )),
    decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, H_Q, 256, UMMA::Major::K, UMMA::Major::MN>{}))
>;

struct SharedMemoryPlan {
    CUTE_ALIGNAS(1024) bf16 kv_slots[NUM_KV_SLOTS][B_TOPK * D_QK / CLUSTER_SIZE];
    CUTE_ALIGNAS(128) uint8_t raw_kv[FP8_RAW_PIPE ? (HAS_FP4_KV ? B_TOPK / 4 * RAW_KV_GROUP_BYTES : B_TOPK * RAW_FP8_TOKEN_SMEM_STRIDE_CTA0) : 0];
    CUTE_ALIGNAS(1024) bf16 s[H_Q_PER_CTA * B_TOPK];
    CUTE_ALIGNAS(1024) float p_exchange_buf[4][32*NUM_P_ELEMS_PER_THREAD];
    CUTE_ALIGNAS(1024) uint8_t is_k_valid[NUM_INDICES_BUFS][ku::find_next_power_of_2(B_TOPK/8)];

    CUTE_ALIGNAS(16) int decode_tma_coords[IS_DECODE ? NUM_INDICES_BUFS : 0][B_TOPK];
    uint8_t decode_scales[IS_DECODE ? NUM_INDICES_BUFS : 0][B_TOPK * NUM_SCALES_EACH_TOKEN_PER_CTA];

    const uint8_t *decode_scale_ptrs[(IS_DECODE && HAS_FP4_KV) ? NUM_INDICES_BUFS : 0][B_TOPK];
    float q_sqr_sum_buf[ENABLE_Q_NORM ? 2 : 0][128];
    float rowwise_max_buf[128];
    float rowwise_mi_buf[H_Q_PER_CTA];
    float rowwise_li_buf[128];

    transac_bar_t bar_kv_slot_full[NUM_KV_SLOTS], bar_kv_slot_empty[NUM_KV_SLOTS];
    transac_bar_t bar_indices_full[NUM_INDICES_BUFS], bar_indices_empty[NUM_INDICES_BUFS];
    transac_bar_t bar_tQ_empty, bar_tQ_full;
    transac_bar_t bar_q_sqr_sum_full;
    transac_bar_t bar_tO_empty, bar_tO_full;
    transac_bar_t bar_tP_full[NUM_P_BUFS], bar_tP_empty[NEED_TP_EMPTY_BAR ? NUM_P_BUFS : 0];
    transac_bar_t bar_SO_full, bar_SO_empty;
    transac_bar_t bar_clc_full, bar_clc_empty;
    transac_bar_t bar_li_mi_full, bar_li_mi_empty;
    transac_bar_t bar_raw_kv_full;
    transac_bar_t bar_raw_kv_empty[FP8_RAW_PIPE ? 1 : 0];

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};
static_assert(sizeof(SharedMemoryPlan) <= 227 * 1024);

struct TMAParams {

    CUtensorMap tensor_map_kv;

    CUtensorMap tensor_map_kv_fp8_part_cta0;
    CUtensorMap tensor_map_kv_fp8_part_cta1;
    CUtensorMap tensor_map_extra_kv_fp8_part_cta0;
    CUtensorMap tensor_map_extra_kv_fp8_part_cta1;
    CUtensorMap tensor_map_kv_bf16_part;
    CUtensorMap tensor_map_extra_kv_bf16_part;
    CUtensorMap tensor_map_extra_kv_fp4_part_cta0;
    CUtensorMap tensor_map_extra_kv_fp4_part_cta1;
};

using AllocatorT = std::conditional_t<IS_2CTA, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

struct AuxParams {
    cutlass::FastDivmod fast_divmod_page_block_size;
    cutlass::FastDivmod fast_divmod_extra_page_block_size;
};

static __device__ __forceinline__ void umma_arrive_on_every_cta(transac_bar_t &bar) {

    if constexpr (IS_2CTA) {
        ku::umma_arrive_multicast_2x1SM_noelect(bar, 1|2);
    } else {
        ku::umma_arrive_noelect(bar);
    }
}
static __device__ __forceinline__ void umma_arrive_on_cta0(transac_bar_t &bar) {

    if constexpr (IS_2CTA) {
        ku::umma_arrive_2x1SM_noelect(bar);
    } else {
        ku::umma_arrive_noelect(bar);
    }
}
static __device__ __forceinline__ void arrive_on_cta0_barrier(transac_bar_t &bar) {
    if constexpr (IS_2CTA) {
        bar.arrive(0u);
    } else {
        bar.arrive();
    }
}

struct barrier_ids {
    static constexpr int WG0_SYNC = 0;
    static constexpr int WG3_SYNC = 1;
    static constexpr int WG3_WARP02_SYNC = 2;
    static constexpr int WG3_WARP13_SYNC = 3;
};

static __device__ __forceinline__ void
devfunc(const Params &params, const TMAParams &tma_params, const AuxParams &aux_params);

static void run(const Params &params);

};

}
""",
    "pattn/kernel.h": r"""#pragma once

#include <cutlass/float8.h>

#include "kernels/params.h"

namespace pattn::sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

using fp8_e4m3 = cutlass::float_e4m3_t;

struct Config {
    SparseAttnFwdMode FWD_MODE;
    ModelType MODEL_TYPE;
    ModelType EXTRA_MODEL_TYPE;
    uint32_t H_Q;
    bool ENABLE_Q_NORM;

    bool H64_GENERIC = false;

    bool REUSE_KV = false;
    bool RAW_PIPE = false;
    bool MMA11 = false;
    bool PRED_IDX = false;
};

template<typename Base>
struct ParamsTemplate : Base {

    bool enable_q_norm;
    float rms_norm_eps;

    uint32_t* __restrict__ token_positions;
    bool is_rope_neox_style;
    uint32_t rope_dim;
    float* __restrict__ cos_sin_cache;

    uint32_t n_wv_group;
    uint32_t wv_group_size;
    uint32_t num_per_channels;
    bool use_tma_aligned_col_major_sf;
    bool round_sf;
    bool use_packed_ue8m0;

    fp8_e4m3* __restrict__ out_fp8;
    uint32_t* __restrict__ out_sf;
    uint32_t stride_out_sf_wv_group, stride_out_sf_head_dim;
};

template<SparseAttnFwdMode FWD_MODE>
using ParamT = std::conditional_t<is_decode_v<FWD_MODE>, ParamsTemplate<SparseAttnDecodeParams>, ParamsTemplate<SparseAttnFwdParams>>;

template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params);

}
""",
    "pattn/helpers.h": r"""#pragma once

#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

namespace pattn::sm100 {

using namespace cute;

struct bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};

CUTE_DEVICE
int int4_max(int4 t) {
    return max(max(t.x, t.y), max(t.z, t.w));
}

CUTE_DEVICE
int int4_min(int4 t) {
    return min(min(t.x, t.y), min(t.z, t.w));
}

struct int32x8_t {
    int a0, a1, a2, a3, a4, a5, a6, a7;
};

struct float8 {
    float2 a01, a23, a45, a67;
};

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define SM100_HAS_NATIVE_BF16X2_CVT 1
#else
#define SM100_HAS_NATIVE_BF16X2_CVT 0
#endif

CUTE_DEVICE
ku::nvbf16x2 f16x2_to_bf16x2(uint32_t f16x2) {
    return __float22bfloat162_rn(__half22float2(*(__half2*)&f16x2));
}

CUTE_DEVICE
ku::nvbf16x2 fp8x2_to_bf16x2(ku::nve4m3x2 data) {
    uint32_t out;
#if SM100_HAS_NATIVE_BF16X2_CVT
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;"
        : "=r"(out)
        : "h"(*(uint16_t*)&data));
#else
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
        : "=r"(out)
        : "h"(*(uint16_t*)&data));
    const ku::nvbf16x2 bf16 = f16x2_to_bf16x2(out);
    out = *(uint32_t*)&bf16;
#endif
    return *(ku::nvbf16x2*)&out;
}

CUTE_DEVICE
ku::nvbf16x2 ue8m0_to_bf16x2(__nv_fp8_e8m0 scale_e8m0) {
    uint16_t packed = (uint16_t)(*(uint8_t*)&scale_e8m0) * 0x0101;
    uint32_t out;
    asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;"
        : "=r"(out)
        : "h"(packed));
    return *(ku::nvbf16x2*)&out;
}

CUTE_DEVICE
ku::nvbf16x2 fp8x2_to_bf16x2_with_scale(ku::nve4m3x2 data, __nv_fp8_e8m0 scale_e8m0) {
    return __hmul2(fp8x2_to_bf16x2(data), ue8m0_to_bf16x2(scale_e8m0));
}

CUTE_DEVICE
void fp4x8_to_bf16x2x4(uint32_t packed, ku::nvbf16x2 *out) {
    uint32_t o0, o1, o2, o3;
#if SM100_HAS_NATIVE_BF16X2_CVT
    asm(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
        : "r"(packed)
    );
    out[0] = *(ku::nvbf16x2*)&o0;
    out[1] = *(ku::nvbf16x2*)&o1;
    out[2] = *(ku::nvbf16x2*)&o2;
    out[3] = *(ku::nvbf16x2*)&o3;
#else
    asm(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
        : "r"(packed)
    );
    const ku::nvbf16x2 b0 = f16x2_to_bf16x2(o0), b1 = f16x2_to_bf16x2(o1);
    const ku::nvbf16x2 b2 = f16x2_to_bf16x2(o2), b3 = f16x2_to_bf16x2(o3);
    out[0] = b0; out[1] = b1; out[2] = b2; out[3] = b3;
#endif
}

CUTE_DEVICE
void fp8x4_to_bf16x2x2(uint32_t packed, ku::nvbf16x2 *out) {
    uint32_t o0, o1;
#if SM100_HAS_NATIVE_BF16X2_CVT
    asm(
        "{\n"
        ".reg .b16 h0, h1;\n"
        "mov.b32 {h0, h1}, %2;\n"
        "cvt.rn.bf16x2.e4m3x2 %0, h0;\n"
        "cvt.rn.bf16x2.e4m3x2 %1, h1;\n"
        "}\n"
        : "=r"(o0), "=r"(o1)
        : "r"(packed)
    );
    out[0] = *(ku::nvbf16x2*)&o0;
    out[1] = *(ku::nvbf16x2*)&o1;
#else
    asm(
        "{\n"
        ".reg .b16 h0, h1;\n"
        "mov.b32 {h0, h1}, %2;\n"
        "cvt.rn.f16x2.e4m3x2 %0, h0;\n"
        "cvt.rn.f16x2.e4m3x2 %1, h1;\n"
        "}\n"
        : "=r"(o0), "=r"(o1)
        : "r"(packed)
    );
    const ku::nvbf16x2 b0 = f16x2_to_bf16x2(o0), b1 = f16x2_to_bf16x2(o1);
    out[0] = b0; out[1] = b1;
#endif
}

CUTE_DEVICE
ku::nvbf16x2 e4m3x2_to_bf16x2(uint16_t packed) {
    return fp8x2_to_bf16x2(*(ku::nve4m3x2*)&packed);
}

}
""",
    "pattn/common_subroutine.h": r"""#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

namespace pattn::sm100 {

CUTE_DEVICE
char load_indices_and_generate_mask(
    int lane_idx,
    int* gIndices,
    int s_kv,
    int abs_pos_start,
    int topk_length
) {
    int indices[8];
    KU_LDG_256(
        gIndices + lane_idx*8,
        indices,
        ".nc",
        "evict_first",
        "evict_normal",
        "256B"
    );
    auto is_valid = [&](int rel_pos_in_lane, int index) -> char {
        int abs_pos = abs_pos_start + lane_idx*8 + rel_pos_in_lane;
        return index >= 0 && index < s_kv && abs_pos < topk_length;
    };
    char is_ks_valid_mask = \
        is_valid(7, indices[7]) << 7 |
        is_valid(6, indices[6]) << 6 |
        is_valid(5, indices[5]) << 5 |
        is_valid(4, indices[4]) << 4 |
        is_valid(3, indices[3]) << 3 |
        is_valid(2, indices[2]) << 2 |
        is_valid(1, indices[1]) << 1 |
        is_valid(0, indices[0]) << 0;
    return is_ks_valid_mask;
}

template<
    int NUM_ELEMS_PER_THREAD,
    int BARRIER_WARP02_SYNC_ID,
    int BARRIER_WARP13_SYNC_ID,
    bool STORE_BACK_P,
    bool MASK_PREFETCHED = false
>
CUTE_DEVICE
void retrieve_mask_and_reduce_p(
    uint32_t p_tmem_col_start,
    char* k_validness_base,
    int local_warp_idx,
    int lane_idx,
    auto slot_bar_P_empty_arrival,
    float p_exchange_buf[4][32*NUM_ELEMS_PER_THREAD],
    float p[NUM_ELEMS_PER_THREAD],
    uint32_t preloaded_validity = 0
) {
    using namespace cute;
    using cutlass::arch::NamedBarrier;
    static_assert(BARRIER_WARP13_SYNC_ID == BARRIER_WARP02_SYNC_ID+1);

    float p_peer[NUM_ELEMS_PER_THREAD];
    auto load_op = []<int N>(uint32_t ts, void* d) {
        ku::tmem_ld_32dp32bNx<N>(ts, d);
    };
    if (local_warp_idx < 2) {
        ku::tmem_ld_st_decomposed<NUM_ELEMS_PER_THREAD, 1>(p_tmem_col_start, p, load_op);
        ku::tmem_ld_st_decomposed<NUM_ELEMS_PER_THREAD, 1>(p_tmem_col_start + NUM_ELEMS_PER_THREAD, p_peer, load_op);
    } else {
        ku::tmem_ld_st_decomposed<NUM_ELEMS_PER_THREAD, 1>(p_tmem_col_start, p_peer, load_op);
        ku::tmem_ld_st_decomposed<NUM_ELEMS_PER_THREAD, 1>(p_tmem_col_start + NUM_ELEMS_PER_THREAD, p, load_op);
    }
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();
    slot_bar_P_empty_arrival();

    static_assert(NUM_ELEMS_PER_THREAD == 48 || NUM_ELEMS_PER_THREAD == 32 || NUM_ELEMS_PER_THREAD == 16);
    using is_k_valid_mask_t = \
        cute::conditional_t<NUM_ELEMS_PER_THREAD == 16, uint16_t,
        cute::conditional_t<NUM_ELEMS_PER_THREAD == 32, uint32_t,
        cute::conditional_t<NUM_ELEMS_PER_THREAD == 48, uint64_t,
        decltype([](){})
    >>>;
    int k_validness_offset = local_warp_idx>=2 ? NUM_ELEMS_PER_THREAD/8 : 0;
    uint8_t is_k_valid_masks[NUM_ELEMS_PER_THREAD / 8];
    if constexpr (NUM_ELEMS_PER_THREAD == 16) {
        *(uint16_t*)is_k_valid_masks = *(uint16_t*)(k_validness_base + k_validness_offset);
    } else if constexpr (NUM_ELEMS_PER_THREAD == 32) {
        if constexpr (MASK_PREFETCHED) {
            __builtin_memcpy(is_k_valid_masks, &preloaded_validity, sizeof(preloaded_validity));
        } else {
            *(uint32_t*)is_k_valid_masks = *(uint32_t*)(k_validness_base + k_validness_offset);
        }
    } else if constexpr (NUM_ELEMS_PER_THREAD == 48) {
        *(uint16_t*)(is_k_valid_masks + 0) = *(uint16_t*)(k_validness_base + k_validness_offset + 0);
        *(uint16_t*)(is_k_valid_masks + 2) = *(uint16_t*)(k_validness_base + k_validness_offset + 2);
        *(uint16_t*)(is_k_valid_masks + 4) = *(uint16_t*)(k_validness_base + k_validness_offset + 4);
    }
    bool needs_masking = true;
    if constexpr (MASK_PREFETCHED) {
        static_assert(NUM_ELEMS_PER_THREAD == 32);
        needs_masking = preloaded_validity != 0xFFFFFFFFu;
    }

    if (needs_masking) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 1) {
            if (!(is_k_valid_masks[i/8] >> (i%8) & 1))
                p[i] = -CUDART_INF_F;
        }
    }

    {

        CUTE_UNROLL
        for (int i = 0; i < NUM_ELEMS_PER_THREAD/4; ++i) {
            ku::st_shared(&p_exchange_buf[local_warp_idx^2][i*32*4 + lane_idx*4], *(float4*)(p_peer + i*4));
        }
        NamedBarrier::arrive_and_wait(64, BARRIER_WARP02_SYNC_ID + (local_warp_idx&1));
        CUTE_UNROLL
        for (int i = 0; i < NUM_ELEMS_PER_THREAD/4; ++i) {
            float2 t[2];
            *(float4*)t = *(float4*)(&p_exchange_buf[local_warp_idx][i*32*4 + lane_idx*4]);
            float2* cur_p = (float2*)(p + i*4);
            cur_p[0] = ku::float2_add(cur_p[0], t[0]);
            cur_p[1] = ku::float2_add(cur_p[1], t[1]);
        }
    }

    if constexpr (STORE_BACK_P) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_ELEMS_PER_THREAD/4; ++i) {
            ku::st_shared(&p_exchange_buf[local_warp_idx][i*32*4 + lane_idx*4], *(float4*)(p+i*4));
        }
    }
}

template<
    int NUM_COLS,
    int CHUNK_SIZE,
    int TMEM_COL_START
>
CUTE_DEVICE
void rescale_O(
    float scale_factor
) {
    float2 scale_factor_float2 = {scale_factor, scale_factor};
    float2 o[CHUNK_SIZE/2];

    CUTE_UNROLL
    for (int chunk_idx = 0; chunk_idx < NUM_COLS/CHUNK_SIZE; ++chunk_idx) {

        ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(TMEM_COL_START + chunk_idx*CHUNK_SIZE, o);
        cutlass::arch::fence_view_async_tmem_load();

        for (int i = 0; i < CHUNK_SIZE/2; ++i) {
            o[i] = ku::float2_mul(o[i], scale_factor_float2);
        }

        ku::tmem_st_32dp32bNx<CHUNK_SIZE>(TMEM_COL_START + chunk_idx*CHUNK_SIZE, o);
        cutlass::arch::fence_view_async_tmem_store();
    }
}

template<int NUM_ELEMS_PER_THREAD, bool DO_ABS = false>
CUTE_DEVICE
float get_max(
    float p[NUM_ELEMS_PER_THREAD]
) {

    float local_max0 = -CUDART_INF_F;
    float local_max1 = -CUDART_INF_F;
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; ++i) {
        float t = p[i];
        if constexpr (DO_ABS)
            t = fabsf(t);
        local_max0 = max(local_max0, t);
    }
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD-NUM_ELEMS_PER_THREAD/2; ++i) {
        float t = p[i+NUM_ELEMS_PER_THREAD/2];
        if constexpr (DO_ABS)
            t = fabsf(t);
        local_max1 = max(local_max1, t);
    }
    return max(local_max0, local_max1);
}

template<int NUM_ELEMS_PER_THREAD>
CUTE_DEVICE
float get_s_from_p(
    nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2],
    float p[NUM_ELEMS_PER_THREAD],
    float scale,
    float new_max
) {
    float2 cur_sum = float2 {0.0f, 0.0f};
    float2 neg_new_max_float2 = float2 {-new_max, -new_max};
    float2 scale_float2 = float2 {scale, scale};
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; i += 1) {
        float2 d = ku::float2_fma(float2{p[i*2], p[i*2+1]}, scale_float2, neg_new_max_float2);
        d.x = exp2f(d.x);
        d.y = exp2f(d.y);
        cur_sum = ku::float2_add(cur_sum, d);
        s[i] = __float22bfloat162_rn(d);
    }
    return cur_sum.x + cur_sum.y;
}

}
""",
    "kernels/defines.h": r"""#pragma once

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>

using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::NamedBarrier;

struct int32x8_t {
    int a0, a1, a2, a3, a4, a5, a6, a7;
};

struct float8 {
    float2 a01, a23, a45, a67;
};

struct bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};
""",
    "kernels/kv_cache_format.h": r"""#pragma once

#include "params.h"

template<ModelType MT>
struct KVCacheFormat {
    static constexpr ModelType MODEL_TYPE = MT;
    static constexpr bool IS_V32 = MT == ModelType::V32 || MT == ModelType::V32_NO_ROPE;
    static constexpr bool IS_FP4 = MT == ModelType::V41_FP4;
    static constexpr int D_QK = MT == ModelType::V32 ? 576 : 512;
    static constexpr int D_ROPE = MT == ModelType::V32_NO_ROPE ? 0 : 64;
    static constexpr int D_NOPE = D_QK - D_ROPE;
    static constexpr int D_FP4 = IS_FP4 ? D_QK : 0;
    static constexpr int D_FP8 = IS_FP4 ? 0 : (MT == ModelType::V41 ? D_QK : D_NOPE);
    static constexpr int D_BF16 = D_QK - D_FP4 - D_FP8;
    static constexpr int QUANT_TILE_SIZE = IS_FP4 ? 16 : (MT == ModelType::V41 ? 32 : (MT == ModelType::V4 ? 64 : 128));
    static constexpr int NUM_SCALES_EACH_TOKEN = (IS_V32 ? D_NOPE : D_QK) / QUANT_TILE_SIZE;
    static constexpr int SCALE_BYTES = IS_V32 ? 4 : 1;
    static constexpr int SCALE_SMEM_BYTES = IS_V32 ? 2 : 1;
    static constexpr int QUANT_BYTES = D_FP4 / 2 + D_FP8;

    static constexpr int TMA_K_STRIDE = QUANT_BYTES + (IS_V32 ? NUM_SCALES_EACH_TOKEN * SCALE_BYTES : 0) + 2 * D_BF16;

    static constexpr int BYTES_PER_TOKEN = TMA_K_STRIDE + (IS_V32 ? 0 : NUM_SCALES_EACH_TOKEN * SCALE_BYTES);
};

constexpr int kv_cache_bytes_per_token(ModelType mt) {
    switch (mt) {
        case ModelType::V32: return KVCacheFormat<ModelType::V32>::BYTES_PER_TOKEN;
        case ModelType::V4: return KVCacheFormat<ModelType::V4>::BYTES_PER_TOKEN;
        case ModelType::V41: return KVCacheFormat<ModelType::V41>::BYTES_PER_TOKEN;
        case ModelType::V41_FP4: return KVCacheFormat<ModelType::V41_FP4>::BYTES_PER_TOKEN;
        case ModelType::V32_NO_ROPE: return KVCacheFormat<ModelType::V32_NO_ROPE>::BYTES_PER_TOKEN;
    }
    return 0;
}

constexpr bool is_valid_kv_format_pair(ModelType kv, ModelType extra_kv) {
    return extra_kv == kv || (kv == ModelType::V41 && extra_kv == ModelType::V41_FP4);
}

template<ModelType KV, ModelType EXTRA_KV = KV>
struct KVFormatPair {
    static_assert(is_valid_kv_format_pair(KV, EXTRA_KV));
    static constexpr ModelType kv = KV, extra_kv = EXTRA_KV;
};

template<typename... Pairs>
struct KVFormatPairs {};
""",
    "kernels/params.h": r"""#pragma once

#include "cutlass/bfloat16.h"

enum class ModelType {
    V32,
    V4,
    V41,
    V41_FP4,
    V32_NO_ROPE
};

struct __align__(4*8) DecodingSchedMeta {
    int begin_req_idx, end_req_idx;
    int begin_block_idx, end_block_idx;
    int begin_split_idx;
    int is_first_req_splitted, is_last_req_splitted;
    int _pad[1];
};
static constexpr int DecodingSchedMetaSize = sizeof(DecodingSchedMeta);

struct DenseAttnDecodeParams {
    using index_t = int64_t;

    int b;
    int s_q;
    int q_seq_per_hk;
    int d, d_v;
    int h_q, h_k;
    int num_blocks;
    int q_head_per_hk;
    bool is_causal;
    float scale_softmax, scale_softmax_log2;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ o_ptr;
    float *__restrict__ softmax_lse_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int *__restrict__ seqlens_k_ptr;

    DecodingSchedMeta *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;

    int total_num_splits;
    float *__restrict__ softmax_lseaccum_ptr;
    float *__restrict__ oaccum_ptr;

    cudaStream_t stream;
};

struct SparseAttnDecodeParams {
    int b, s_q;
    int h_q, h_kv;
    int d_qk, d_v;
    float sm_scale, sm_scale_div_log2;
    int num_blocks, page_block_size, topk;
    ModelType model_type;
    ModelType extra_model_type;

    cutlass::bfloat16_t* __restrict__ q;
    cutlass::bfloat16_t* __restrict__ kv;
    int* __restrict__ indices;
    int* __restrict__ topk_length;
    float* __restrict__ attn_sink;

    float* __restrict__ lse;
    cutlass::bfloat16_t* __restrict__ out;

    int extra_num_blocks, extra_page_block_size, extra_topk;
    cutlass::bfloat16_t* __restrict__ extra_kv;
    int* __restrict__ extra_indices;
    int* __restrict__ extra_topk_length;

    int stride_q_b, stride_q_s_q, stride_q_h_q;
    int stride_kv_block, stride_kv_row;
    int stride_indices_b, stride_indices_s_q;
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_extra_kv_block, stride_extra_kv_row;
    int stride_extra_indices_b, stride_extra_indices_s_q;

    cudaStream_t stream;

    bool enable_split_kv;

    float* __restrict__ lse_accum;
    float* __restrict__ o_accum;
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
    int* __restrict__ num_splits_ptr;
    int num_sm_parts;
};

struct CombineParams {
    int b, s_q, h_q, d_v;

    float* __restrict__ lse;
    void* __restrict__ out;
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;

    float* __restrict__ lse_accum;
    float* __restrict__ o_accum;
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;

    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
    int* __restrict__ num_splits_ptr;
    int num_sm_parts;

    float* attn_sink;

    cudaStream_t stream;
};

struct GetDecodeSchedMetaParams {
    int b;
    int s_q;
    int block_size_n;
    int fixed_overhead_num_blocks;

    int topk, extra_topk;
    int *__restrict__ topk_length, *__restrict__ extra_topk_length;

    int *__restrict__ seqlens_k_ptr;

    DecodingSchedMeta *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int num_sm_parts;

    cudaStream_t stream;
};

struct SparseAttnFwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale, sm_scale_div_log2;

    cutlass::bfloat16_t* __restrict__ q;
    cutlass::bfloat16_t* __restrict__ kv;
    int* __restrict__ indices;
    float* __restrict__ attn_sink;
    int* __restrict__ topk_length;

    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_indices_s_q; int stride_indices_h_kv;

    cutlass::bfloat16_t* __restrict__ out;
    float* __restrict__ max_logits;
    float* __restrict__ lse;

    int num_sm;

    bool stats_in_log2 = false;
    cudaStream_t stream;
};

enum class SparseAttnFwdMode {
    Prefill,
    Decode,
    DecodeWithSplitKV,
};

template<SparseAttnFwdMode FWD_MODE>
inline constexpr bool is_decode_v = std::bool_constant<FWD_MODE == SparseAttnFwdMode::Decode || FWD_MODE == SparseAttnFwdMode::DecodeWithSplitKV>::value;

template<SparseAttnFwdMode FWD_MODE>
using SparseFwdArgT = std::conditional_t<is_decode_v<FWD_MODE>, SparseAttnDecodeParams, SparseAttnFwdParams>;
""",
    "kernels/utils.h": r"""#pragma once

#include <cstdint>

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                              \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

#define FLASH_ASSERT(cond)                                                                                \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                 \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define FLASH_DEVICE_ASSERT(cond)                                                                         \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                          \
            asm("trap;");                                                                                 \
        }                                                                                                 \
    } while(0)

#define println(fmt, ...) { print(fmt, ##__VA_ARGS__); print("\n"); }

template<typename T>
__inline__ __host__ __device__ T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

#ifndef TRAP_ONLY_DEVICE_ASSERT
#define TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

#ifndef TRAP_ONLY_DEVICE_ASSERT
#define TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

struct RingBufferState {
    uint32_t cur_block_idx = 0u;

    __device__ __forceinline__
    void update() {
        cur_block_idx += 1;
    }

    template<uint32_t NUM_STAGES>
    __device__ __forceinline__
    std::pair<uint32_t, bool> get() const {
        uint32_t stage_idx = cur_block_idx % NUM_STAGES;
        bool phase = (cur_block_idx / NUM_STAGES) & 1;
        return {stage_idx, phase};
    }

    __device__ __forceinline__
    RingBufferState offset_by(const int offset) const {

        uint32_t new_block_idx = static_cast<uint32_t>(static_cast<int>(cur_block_idx) + offset);
        RingBufferState new_state;
        new_state.cur_block_idx = new_block_idx;
        return new_state;
    }
};
""",
    "kerutils_include/kerutils/kerutils.cuh": r"""#pragma once

#include "kerutils/common/common.h"

#include "host/host.h"

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include "device/device.cuh"
#endif
""",
    "kerutils_include/kerutils/common/common.h": r"""#pragma once

namespace kerutils {}

#define KU_PRINTLN(fmt, ...) { cute::print(fmt, ##__VA_ARGS__); print("\n"); }

namespace ku = kerutils;

#ifdef __CUDACC__
#define KERUTILS_IS_BUILD_ON_CUDA
#endif

#if __has_include("kernel_operator.h")
#define KERUTILS_IS_BUILD_ON_ASCEND
#endif

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#ifdef KERUTILS_IS_BUILD_ON_ASCEND
#error "KERUTILS_IS_BUILD_ON_CUDA and KERUTILS_IS_BUILD_ON_ASCEND is defined at the same time!"
#endif
#endif

#ifndef KERUTILS_IS_BUILD_ON_CUDA
#ifndef KERUTILS_IS_BUILD_ON_ASCEND
#error "Neither KERUTILS_IS_BUILD_ON_CUDA and KERUTILS_IS_BUILD_ON_ASCEND is defined!"
#endif
#endif

namespace kerutils {

template<typename T>
inline constexpr T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

template<typename T>
inline constexpr T ceil(const T &a, const T &b) {
    return (a + b - 1) / b * b;
}

}
""",
    "kerutils_include/kerutils/host/host.h": r"""#pragma once

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <sstream>
#include <vector>

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>
#endif

#ifdef KERUTILS_IS_BUILD_ON_ASCEND
#include <acl/acl.h>
#endif

#include "kerutils/common/common.h"

namespace kerutils {

class KUException final : public std::exception {
    std::string message = {};

public:
    template<typename... Args>
    explicit KUException(const char *name, const char* file, const int line, Args&&... args) {
        std::ostringstream oss;

        oss << name << " error (" << file << ":" << line << "): ";
        (oss << ... << args);
        message = oss.str();
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#define THROW_KU_EXCEPTION(name, ...) \
    throw kerutils::KUException(name, __FILE__, __LINE__, __VA_ARGS__)

#define KU_ASSERT(cond, ...)                                                                     \
    do {                                                                                         \
        if (not (cond)) {                                                                        \
            char _ku_buf[1024];                                                                  \
            int _ku_len = snprintf(_ku_buf, sizeof(_ku_buf),                                    \
                                   "Assertion `%s` failed (%s:%d)", #cond, __FILE__, __LINE__); \
            __VA_OPT__(                                                                          \
                _ku_len += snprintf(_ku_buf + _ku_len, sizeof(_ku_buf) - _ku_len,              \
                                    ": " __VA_ARGS__);                                          \
            )                                                                                    \
            fprintf(stderr, "%s\n", _ku_buf);                                                    \
            THROW_KU_EXCEPTION("Assertion", _ku_buf);                                            \
        }                                                                                        \
    } while(0)

#ifdef KERUTILS_IS_BUILD_ON_CUDA

#define KU_CUDA_CHECK(call)                                                                                   \
do {                                                                                                          \
    cudaError_t status_ = (call);                                                                             \
    if (status_ != cudaSuccess) {                                                                             \
        char _ku_buf[1024];                                                                                   \
        snprintf(_ku_buf, sizeof(_ku_buf), "CUDA error (%s:%d): %s", __FILE__, __LINE__, cudaGetErrorString(status_)); \
        fprintf(stderr, "%s\n", _ku_buf);                                                                    \
        THROW_KU_EXCEPTION("CUDA", _ku_buf);                                                                  \
    }                                                                                                         \
} while(0)

#define KU_CUTLASS_CHECK(call)                                                                                   \
do {                                                                                                             \
    cutlass::Status status_ = (call);                                                                            \
    if (status_ != cutlass::Status::kSuccess) {                                                                 \
        char _ku_buf[1024];                                                                                      \
        snprintf(_ku_buf, sizeof(_ku_buf), "CUTLASS error (%s:%d): %d", __FILE__, __LINE__, static_cast<int>(status_)); \
        fprintf(stderr, "%s\n", _ku_buf);                                                                       \
        THROW_KU_EXCEPTION("CUTLASS", _ku_buf);                                                                 \
    }                                                                                                            \
} while(0)

#define KU_CHECK_KERNEL_LAUNCH() KU_CUDA_CHECK(cudaGetLastError())

template<typename T, T LOWER_BOUND = 1>
inline __host__ __device__ constexpr T find_next_power_of_2(const T& x) {
    if (x <= LOWER_BOUND)
        return LOWER_BOUND;
    return find_next_power_of_2<T, LOWER_BOUND*2>(x);
}

static inline CUtensorMap make_tensor_map(
    const std::vector<uint64_t> &size,
    const std::vector<uint64_t> &strides,
    const std::vector<uint32_t> &box_size,
    void* global_ptr,
    CUtensorMapDataType data_type,
    CUtensorMapSwizzle swizzle_mode,
    CUtensorMapL2promotion l2_promotion,
    CUtensorMapInterleave interleave_mode = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapFloatOOBfill oob_fill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    const std::vector<uint32_t> &element_strides_ = {}
) {
    int dim = size.size();
    KU_ASSERT(dim >= 1);

    std::vector<uint32_t> element_strides;
    if (element_strides_.empty()) {
        for (int i = 0; i < dim; ++i)
            element_strides.push_back(1);
    } else {
        element_strides = element_strides_;
    }
    KU_ASSERT(strides.size() == (uint32_t)dim-1 && box_size.size() == (uint32_t)dim && element_strides.size() == (uint32_t)dim);

    auto call_cuTensorMapEncodeTiled = [&]<typename... Args>(Args... args) {
        cudaDriverEntryPointQueryResult cuda_status;
        void* pfn = nullptr;
#if (__CUDACC_VER_MAJOR__ > 12)
        KU_CUDA_CHECK(cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled",
            &pfn, 12000,
            cudaEnableDefault,
            &cuda_status));
#else
        KU_CUDA_CHECK(cudaGetDriverEntryPoint(
            "cuTensorMapEncodeTiled",
            &pfn,
            cudaEnableDefault,
            &cuda_status));
#endif
        if (cuda_status != cudaDriverEntryPointSuccess) {
            KU_ASSERT(false, "Failed to load `cuTensorMapEncodeTiled`. cuda_status = %d", cuda_status);
        }
        return reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(pfn)(args...); \
    };

    CUtensorMap result;
    CUresult ret_code = call_cuTensorMapEncodeTiled(
        &result,
        data_type,
        dim,
        global_ptr,
        size.data(),
        strides.data(),
        box_size.data(),
        element_strides.data(),
        interleave_mode,
        swizzle_mode,
        l2_promotion,
        oob_fill
    );
    if (ret_code != CUresult::CUDA_SUCCESS) {
        auto print_vector = [&](auto t, const char* fmt, const char end='\n') {
            for (auto elem : t) {
                printf(fmt, elem);
            }
            printf("%c", end);
        };
        fprintf(stderr, "Failed to create tensormap\n");
        fprintf(stderr, "Dim: %d\n", dim);
        printf("size: "); print_vector(size, "%lu ");
        printf("strides: "); print_vector(strides, "%lu ");
        printf("box_size: "); print_vector(box_size, "%u ");
        printf("element_strides: "); print_vector(element_strides, "%u ");
        printf("global ptr: 0x%lx\n", (int64_t)global_ptr);
        printf("data_type: %d\n", (int)data_type);
        printf("swizzle_mode: %d\n", (int)swizzle_mode);
        printf("l2_promotion: %d\n", (int)l2_promotion);
        printf("interleave_mode: %d\n", (int)interleave_mode);
        printf("oob_fill: %d\n", (int)oob_fill);
        KU_ASSERT(false);
    }
    return result;
}

template<typename T>
static inline std::vector<uint64_t> make_stride_helper(const std::vector<T> &strides_in_elems, size_t elem_size) {
    std::vector<uint64_t> res;
    for (auto stride : strides_in_elems) {
        res.push_back(((uint64_t)stride) * elem_size);
    }
    return res;
}

struct KernelLaunchConfig {
    dim3 grid;
    dim3 block;
    size_t dynamic_smem;
    cudaStream_t stream;
    dim3 cluster{1, 1, 1};
    bool use_pdl{false};
    bool cooperative{false};
};

namespace detail {

template <class Arg>
void* kernel_arg_ptr(Arg&& arg) {
    return const_cast<void*>(reinterpret_cast<const void*>(std::addressof(arg)));
}

}

template<typename KernelFunc, typename... Args>
void launch_kernel(const KernelLaunchConfig &cfg, KernelFunc kernel, Args&&... args) {
    if (cfg.dynamic_smem > 0) {
        KU_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(cfg.dynamic_smem)));
    }

    const bool is_cluster     = !(cfg.cluster.x == 1 && cfg.cluster.y == 1 && cfg.cluster.z == 1);
    const bool need_kernel_ex = is_cluster || cfg.use_pdl;

    if (is_cluster && cfg.cooperative) {
        KU_ASSERT(false, "cluster and cooperative launch are mutually exclusive");
    }

    if (cfg.cooperative) {
        void* kernel_args[sizeof...(Args) > 0 ? sizeof...(Args) : 1] = {};
        if constexpr (sizeof...(Args) > 0) {
            size_t i = 0;
            ((kernel_args[i++] = detail::kernel_arg_ptr(std::forward<Args>(args))), ...);
        }
        KU_CUDA_CHECK(cudaLaunchCooperativeKernel(
            kernel, cfg.grid, cfg.block,
            sizeof...(Args) > 0 ? kernel_args : nullptr,
            static_cast<unsigned int>(cfg.dynamic_smem), cfg.stream));
    } else if (need_kernel_ex) {
        if (is_cluster) {
            KU_ASSERT(cfg.grid.x % cfg.cluster.x == 0 &&
                      cfg.grid.y % cfg.cluster.y == 0 &&
                      cfg.grid.z % cfg.cluster.z == 0);
            if (cfg.cluster.x * cfg.cluster.y * cfg.cluster.z > 8) {
                KU_CUDA_CHECK(cudaFuncSetAttribute(
                    reinterpret_cast<const void*>(kernel),
                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            }
        }

        const unsigned int num_attrs = is_cluster ? 2 : 1;
        cudaLaunchAttribute attrs[2];
        if (is_cluster) {
            attrs[0].id = cudaLaunchAttributeClusterDimension;
            attrs[0].val.clusterDim = {cfg.cluster.x, cfg.cluster.y, cfg.cluster.z};
        }
        attrs[num_attrs - 1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[num_attrs - 1].val.programmaticStreamSerializationAllowed = cfg.use_pdl ? 1 : 0;

        cudaLaunchConfig_t config = {
            {cfg.grid.x, cfg.grid.y, cfg.grid.z},
            {cfg.block.x, cfg.block.y, cfg.block.z},
            cfg.dynamic_smem,
            cfg.stream,
            attrs,
            num_attrs
        };
        KU_CUDA_CHECK(cudaLaunchKernelEx(
            &config, kernel, std::forward<Args>(args)...));
    } else {
        kernel<<<cfg.grid, cfg.block, cfg.dynamic_smem, cfg.stream>>>(
            std::forward<Args>(args)...);
    }

    KU_CHECK_KERNEL_LAUNCH();
}

#endif

#ifdef KERUTILS_IS_BUILD_ON_ASCEND

#define KU_ACLRT_CHECK(call)                                                                                   \
do {                                                                                                            \
    aclError status_ = (call);                                                                                  \
    if (status_ != ACL_SUCCESS) {                                                                               \
        char _ku_buf[1024];                                                                                     \
        snprintf(_ku_buf, sizeof(_ku_buf), "ACLRT error (%s:%d): %s (code %d)", __FILE__, __LINE__,            \
                 aclGetRecentErrMsg(), status_);                                                                \
        fprintf(stderr, "%s\n", _ku_buf);                                                                       \
        THROW_KU_EXCEPTION("ACLRT", _ku_buf);                                                                   \
    }                                                                                                            \
} while(0)

#endif

}""",
    "kerutils_include/kerutils/device/device.cuh": r"""#pragma once

#include "kerutils/common/common.h"

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include "cuda/common.h"
#include "cuda/sm80/intrinsics.cuh"
#include "cuda/sm80/helpers.cuh"
#include "cuda/sm90/intrinsics.cuh"
#include "cuda/sm90/helpers.cuh"
#include "cuda/sm100/intrinsics.cuh"
#include "cuda/sm100/helpers.cuh"
#include "cuda/sm100/gemm.cuh"
#include "cuda/sm100/tma_cta_group2_nosplit.cuh"
#endif
""",
    "kerutils_include/kerutils/device/cuda/common.h": r"""#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>
#include <cute/config.hpp>

namespace kerutils {

enum class CacheHint {
    EVICT_FIRST,
    EVICT_NORMAL,
    EVICT_LAST,
    EVICT_UNCHANGED,
    NO_ALLOCATE
};

enum class PrefetchSize {
    B64,
    B128,
    B256
};

using nvbf16 = __nv_bfloat16;
using nvbf16x2 = __nv_bfloat162;
using nve4m3 = __nv_fp8_e4m3;
using nve4m3x2 = __nv_fp8x2_e4m3;
using nve4m3x4 = __nv_fp8x4_e4m3;

using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define KERUTILS_ENABLE_SM80
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
static_assert(false, "kerutils doesn't support SM architectures below SM80");
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
#define KERUTILS_ENABLE_SM90
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000))
#define KERUTILS_ENABLE_SM90A
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
#define KERUTILS_ENABLE_SM100
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
#define KERUTILS_ENABLE_SM100A
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1030 && __CUDA_ARCH__ < 1200))
#define KERUTILS_ENABLE_SM103A
#endif

#if (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
#define KERUTILS_ENABLE_SM80
#define KERUTILS_ENABLE_SM90
#define KERUTILS_ENABLE_SM90A
#define KERUTILS_ENABLE_SM100
#define KERUTILS_ENABLE_SM100A
#define KERUTILS_ENABLE_SM103A
#endif
""",
    "kerutils_include/kerutils/device/cuda/sm80/helpers.cuh": r"""#pragma once

#include "kerutils/device/cuda/common.h"
#include "kerutils/device/cuda/sm80/intrinsics.cuh"

namespace kerutils {

CUTE_DEVICE
uint32_t get_sm_id_with_range_check(uint32_t num_physical_sms) {
    uint32_t sm_id = get_sm_id();
    if (!(sm_id < num_physical_sms)) {
        trap();
    }
    return sm_id;
}

#ifndef KU_TRAP_ONLY_DEVICE_ASSERT
#define KU_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

CUTE_DEVICE
float2 float2float2(const float &x) {
    return float2 {x, x};
}

CUTE_DEVICE
void st_shared(void* ptr, __int128_t val) {
    asm volatile("st.shared.b128 [%0], %1;" :: "r"(cute::cast_smem_ptr_to_uint(ptr)), "q"(val));
}

CUTE_DEVICE
void st_shared(void* ptr, float4 val) {
    st_shared(ptr, *(__int128_t*)&val);
}

CUTE_DEVICE
__int128_t ld_shared(const void* ptr) {
    __int128_t val;
    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(val) : "r"(cute::cast_smem_ptr_to_uint(ptr)));
    return val;
}

CUTE_DEVICE
float4 ld_shared_float4(const void* ptr) {
    __int128_t temp = ld_shared(ptr);
    return *(float4*)&temp;
}

}
""",
    "kerutils_include/kerutils/device/cuda/sm80/intrinsics.cuh": r"""#pragma once

#include "kerutils/device/cuda/common.h"

namespace kerutils {

template<PrefetchSize PREFETCH_SIZE=PrefetchSize::B128>
CUTE_DEVICE
void cp_async_cacheglobal(const void* src, void* dst, bool pred=true) {
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
    if constexpr (PREFETCH_SIZE == PrefetchSize::B64) {
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16, %2;\n"
            :: "r"(dst_addr),
               "l"(src),
               "r"(pred?16:0));
    } else if constexpr (PREFETCH_SIZE == PrefetchSize::B128) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
            :: "r"(dst_addr),
               "l"(src),
               "r"(pred?16:0));
    } else if constexpr (PREFETCH_SIZE == PrefetchSize::B256) {
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n"
            :: "r"(dst_addr),
               "l"(src),
               "r"(pred?16:0));
    } else {
        static_assert(PREFETCH_SIZE == PrefetchSize::B64 ||
                      PREFETCH_SIZE == PrefetchSize::B128 ||
                      PREFETCH_SIZE == PrefetchSize::B256,
                      "Unsupported prefetch size for cp_async_cacheglobal.");
    }
}

template<CacheHint PRIMARY_PRIORITY, CacheHint SECONDARY_PRIORITY>
CUTE_DEVICE
int64_t create_fraction_based_cache_policy(float fraction = 1.0f) {
    int64_t result;
    #define EMIT(PRIMARY_PRIORITY_STR, SECONDARY_PRIORITY_STR) \
        asm volatile( \
            "createpolicy.fractional.L2::" PRIMARY_PRIORITY_STR ".L2::" SECONDARY_PRIORITY_STR ".b64 %0, %1;\n" \
            : "=l"(result) \
            : "f"(fraction) \
        );
    #define EMIT2(PRIMARY_PRIORITY_STR) \
        { \
            if constexpr (SECONDARY_PRIORITY == CacheHint::EVICT_FIRST) { \
                EMIT(PRIMARY_PRIORITY_STR, "evict_first") \
            } else if constexpr (SECONDARY_PRIORITY == CacheHint::EVICT_UNCHANGED) { \
                EMIT(PRIMARY_PRIORITY_STR, "evict_unchanged") \
            } else { \
                static_assert(SECONDARY_PRIORITY == CacheHint::EVICT_FIRST || \
                            SECONDARY_PRIORITY == CacheHint::EVICT_UNCHANGED, \
                            "Unsupported secondary cache hint for create_fraction_based_cache_policy."); \
            } \
        }
    if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_FIRST) {
        EMIT2("evict_first");
    } else if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_NORMAL) {
        EMIT2("evict_normal");
    } else if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_LAST) {
        EMIT2("evict_last");
    } else if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_UNCHANGED) {
        EMIT2("evict_unchanged");
    } else {
        static_assert(PRIMARY_PRIORITY == CacheHint::EVICT_FIRST ||
                      PRIMARY_PRIORITY == CacheHint::EVICT_NORMAL ||
                      PRIMARY_PRIORITY == CacheHint::EVICT_LAST ||
                      PRIMARY_PRIORITY == CacheHint::EVICT_UNCHANGED,
                      "Unsupported primary cache hint for create_fraction_based_cache_policy.");
    }
    #undef EMIT
    #undef EMIT2
    return result;
}

template<CacheHint CACHE_HINT>
CUTE_DEVICE
constexpr int64_t create_simple_cache_policy() {
    if constexpr (CACHE_HINT == CacheHint::EVICT_FIRST) {
        return 0x12F0000000000000;
    } else if constexpr (CACHE_HINT == CacheHint::EVICT_NORMAL) {
        return 0x1000000000000000;
    } else if constexpr (CACHE_HINT == CacheHint::EVICT_LAST) {
        return 0x14F0000000000000;
    } else {
        static_assert(CACHE_HINT == CacheHint::EVICT_FIRST ||
                      CACHE_HINT == CacheHint::EVICT_NORMAL ||
                      CACHE_HINT == CacheHint::EVICT_LAST,
                      "Unsupported cache hint for create_simple_cache_policy.");
    }
}

CUTE_DEVICE
void atomicadd_f32_with_policy_and_pred(void* global_addr, const float &data, int64_t cache_policy, uint32_t pred = true) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, %3, 1;\n\t"
        "@p red.relaxed.gpu.global.add.L2::cache_hint.f32 [%1], %0, %2; \n\t"
        "}"
        :
        : "f"(data),
          "l"((int64_t)global_addr), "l"(cache_policy), "r"(pred)
    );
}

CUTE_DEVICE
uint32_t get_sm_id() {
    uint32_t ret;
    asm volatile("mov.u32 %0, %%smid;\n" : "=r"(ret));
    return ret;
}

CUTE_DEVICE
void trap() {
    asm volatile("trap;\n");
}

#define KU_LDG_128(global_addr, result, NC_STR, L1_CACHE_HINT_STR, L2_PREFETCH_SIZE_STR) \
    { \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>, "`global_addr` must be a pointer"); \
        static_assert(std::is_pointer_v<decltype(result)> || std::is_array_v<decltype(result)>, "`result` must be a pointer"); \
        uint64_t* result_as_uint64_ptr = (uint64_t*)(result); \
        asm volatile( \
            "ld.global" NC_STR ".L1::" L1_CACHE_HINT_STR ".L2::" L2_PREFETCH_SIZE_STR ".v2.u64 {%0, %1}, [%2];\n" \
            : "=l"(result_as_uint64_ptr[0]), "=l"(result_as_uint64_ptr[1]) \
            : "l"(global_addr) \
        ); \
    }

#define KU_STG_128(global_addr, src, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR) \
    { \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>, "`global_addr` must be a pointer"); \
        static_assert(std::is_pointer_v<decltype(src)> || std::is_array_v<decltype(src)>, "`src` must be a pointer"); \
        uint64_t const* src_as_uint64_ptr = (uint64_t const*)(src); \
        asm volatile( \
            "st.global.L1::" L1_CACHE_HINT_STR L2_CACHE_HINT_STR ".v2.u64 [%0], {%1, %2};\n" \
            : \
            : "l"(global_addr), "l"(src_as_uint64_ptr[0]), "l"(src_as_uint64_ptr[1]) \
        ); \
    }

}
""",
    "kerutils_include/kerutils/device/cuda/sm90/helpers.cuh": r"""#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/cuda/common.h"

namespace kerutils {

template<
    typename TMA,
    typename Tensor0,
    typename Tensor1
>
CUTE_DEVICE
void launch_tma_copy(
    const TMA &tma_copy,
    Tensor0 src,
    Tensor1 dst,
    transac_bar_t &bar,
    const cute::TMA::CacheHintSm90 &cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL
) {
    auto thr_tma = tma_copy.get_slice(cute::_0{});
    cute::copy(
        tma_copy.with(reinterpret_cast<typename transac_bar_t::ValueType&>(bar), 0, cache_hint),
        thr_tma.partition_S(src),
        thr_tma.partition_D(dst)
    );
}

CUTE_DEVICE
int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
    int row_idx = (idx_in_warpgroup/32)*16 + local_row_idx*8 + (idx_in_warpgroup%32/4);
    return row_idx;
}

CUTE_DEVICE
int get_AorC_col_idx(int local_elem_idx, int idx_in_warpgroup) {
    int col_idx = 8*(local_elem_idx/4) + (idx_in_warpgroup%4)*2 + (local_elem_idx&1);
    return col_idx;
}

template <bool commit=true, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTE_DEVICE
void wgmma(TiledMma &tiled_mma, Tensor0 const &tCrA, Tensor1 const &tCrB, Tensor2 &tCrC, bool zero_init) {
    using namespace cute;
    constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;

    if constexpr (Is_RS) { cute::warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    tiled_mma.accumulate_ = zero_init ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;

    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    if constexpr (commit) {
        warpgroup_commit_batch();
    }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTE_DEVICE
void wgmma_ss(bool clear_accum, TiledMma tiled_mma, Tensor0 const &sA, Tensor1 const &sB, Tensor2 &rC_frag, int idx_in_warpgroup) {
    using namespace cute;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor sA_frag = thr_mma.partition_fragment_A(sA);
    Tensor sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));

    warpgroup_fence_operand(rC_frag);
    warpgroup_arrive();
    tiled_mma.accumulate_ = clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_fence_operand(rC_frag);
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTE_DEVICE
void wgmma_rs(bool clear_accum, TiledMma tiled_mma, Tensor0 rA_frag, Tensor1 const &sB, Tensor2 &rC_frag, int idx_in_warpgroup) {
    using namespace cute;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(rA_frag) == size<2>(sB_frag));

    warpgroup_fence_operand(const_cast<Tensor0 &>(rA_frag));
    warpgroup_fence_operand(rC_frag);
    warpgroup_arrive();
    tiled_mma.accumulate_ = clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(rA_frag); ++k) {
        cute::gemm(tiled_mma, rA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_fence_operand(rC_frag);
    warpgroup_fence_operand(const_cast<Tensor0 &>(rA_frag));
}

}
""",
    "kerutils_include/kerutils/device/cuda/sm90/intrinsics.cuh": r"""#pragma once

#include "kerutils/device/cuda/common.h"

namespace kerutils {

template<typename T>
CUTE_DEVICE
static void st_async(void* dst_ptr, const T& data, transac_bar_t &mbar) {
    static_assert(sizeof(T) == 16, "Data type must be 16 bytes (128 bits) for st_async.");
    longlong2 data_long2 = *reinterpret_cast<const longlong2*>(&data);
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 [%0], {%1, %2}, [%3]; \n"
        :
        : "r"(dst_addr), "l"(data_long2.x), "l"(data_long2.y), "r"(mbar_addr)
    );
}

static constexpr int PEER_ADDR_MASK = 16777216;

template<typename T>
CUTE_DEVICE
T* get_peer_addr(const T* p) {
    return (T*)((int64_t)(p) ^ PEER_ADDR_MASK);
}

template<typename T>
CUTE_DEVICE
T* get_cta0_addr(const T* p) {
    constexpr int CTA0_ADDR_MASK = 0xFEFFFFFF;
    return (T*)((int64_t)(p) & CTA0_ADDR_MASK);
}

CUTE_DEVICE
void tma_bulk_reduce_add(void const* src_ptr, void* dst_ptr, int32_t store_bytes) {
    uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(src_ptr);
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
                     :
                     : "l"(dst_ptr), "r"(smem_int_ptr), "r"(store_bytes)
                     : "memory");
}

CUTE_DEVICE
void barrier_cluster_arrive_release() {
    asm volatile("barrier.cluster.arrive.release;" : : : "memory");
}

CUTE_DEVICE
void barrier_cluster_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed;" : : :);
}

CUTE_DEVICE
void barrier_cluster_wait_acquire() {
    asm volatile("barrier.cluster.wait.acquire;" : : : "memory");
}

CUTE_DEVICE
void mbarrier_arrive_relaxed_cluster(transac_bar_t &mbar) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.relaxed.cluster.shared::cta.b64 _, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr));
}

CUTE_DEVICE
void atomicadd_f32x4_with_policy_and_pred(void* global_addr, const float4 &data, int64_t cache_policy, uint32_t pred = true) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, %6, 1;\n\t"
        "@p red.relaxed.gpu.global.add.L2::cache_hint.v4.f32 [%4], {%0, %1, %2, %3}, %5; \n\t"
        "}"
        :
        : "f"(data.x), "f"(data.y), "f"(data.z), "f"(data.w),
          "l"((int64_t)global_addr), "l"(cache_policy), "r"(pred)
    );
}

CUTE_DEVICE
void cp_async_bulk_shared_cta_to_shared_cluster(void* dst_ptr, const void* src_ptr, int32_t load_bytes, transac_bar_t &mbar) {
    uint32_t dst_smem_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
    uint32_t src_smem_addr = cute::cast_smem_ptr_to_uint(src_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3]; \n"
        :
        : "r"(dst_smem_addr), "r"(src_smem_addr), "r"(load_bytes), "r"(mbar_addr)
    );
}

}
""",
    "kerutils_include/kerutils/device/cuda/sm100/helpers.cuh": r"""#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/cuda/common.h"

namespace kerutils {

template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ss(
    TiledMMA &tiled_mma,
    TensorA sA,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{});
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));
    static_assert(size<1>(sA_frag) == size<1>(tC_frag));
    static_assert(size<1>(sB_frag) == size<2>(tC_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            sA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ts(
    TiledMMA &tiled_mma,
    TensorA tA_frag,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{});
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(tA_frag) == size<2>(sB_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            tA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template<int MN, int K, int SWIZZLE, typename T = bf16>
static constexpr auto make_umma_canonical_k_major_layout() {
    using namespace cute;
    using base_atom_type = \
        std::conditional_t<SWIZZLE == 0 || SWIZZLE == 16,
            UMMA::Layout_K_INTER_Atom<T>,
            std::conditional_t<SWIZZLE == 32,
                UMMA::Layout_K_SW32_Atom<T>,
                std::conditional_t<SWIZZLE == 64,
                    UMMA::Layout_K_SW64_Atom<T>,
                    std::conditional_t<SWIZZLE == 128,
                        UMMA::Layout_K_SW128_Atom<T>,
                        void
                    >
                >
            >
        >;
    static_assert(!std::is_same_v<base_atom_type, void>, "Invalid SWIZZLE value");
    return coalesce(tile_to_shape(
        base_atom_type{},
        Shape<Int<MN>, Int<K>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{});
}

template<int MN, int K, int SWIZZLE, typename T = bf16>
static constexpr auto make_umma_canonical_mn_major_layout() {
    using namespace cute;
    using base_atom_type = \
        std::conditional_t<SWIZZLE == 0 || SWIZZLE == 16,
            UMMA::Layout_MN_INTER_Atom<T>,
            std::conditional_t<SWIZZLE == 32,
                UMMA::Layout_MN_SW32_Atom<T>,
                std::conditional_t<SWIZZLE == 64,
                    UMMA::Layout_MN_SW64_Atom<T>,
                    std::conditional_t<SWIZZLE == 128,
                        UMMA::Layout_MN_SW128_Atom<T>,
                        void
                    >
                >
            >
        >;
    static_assert(!std::is_same_v<base_atom_type, void>, "Invalid SWIZZLE value");
    return coalesce(tile_to_shape(
        base_atom_type{},
        Shape<Int<MN>, Int<K>>{},
        Step<_2, _1>{}
    ), Shape<_1, _1>{});
}

template<cute::UMMA::Major MAJOR, int MN, int K, int SWIZZLE, typename T = bf16>
auto make_umma_canonical_layout() {
    if constexpr (MAJOR == cute::UMMA::Major::K) {
        return make_umma_canonical_k_major_layout<MN, K, SWIZZLE, T>();
    } else {
        return make_umma_canonical_mn_major_layout<MN, K, SWIZZLE, T>();
    }
}

}
""",
    "kerutils_include/kerutils/device/cuda/sm100/intrinsics.cuh": r"""#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include "kerutils/device/cuda/common.h"

namespace kerutils {

CUTE_DEVICE
void tma_gather4(const void* desc_ptr, transac_bar_t &mbar_ptr, void* smem_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "r"(mbar_addr), "l"(cache_hint)
        : "memory"
    );
}

CUTE_DEVICE
void tma_gather4_prefetch(const void* desc_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [%0, {%1, %2, %3, %4, %5}], %6;\n"
        :
        : "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "l"(cache_hint)
    );
}

template<bool USE_CTA0_MBAR = false>
CUTE_DEVICE void tma_gather4_cta_group_2(const void* desc_ptr, transac_bar_t &mbar_ptr, void* smem_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar_ptr);
    if constexpr (USE_CTA0_MBAR) {
        mbar_addr &= cute::Sm100MmaPeerBitMask;
    }
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "r"(mbar_addr), "l"(cache_hint)
        : "memory"
    );
}

CUTE_DEVICE
float2 float2_add(const float2 &a, const float2 &b) {
    float2 c;
    asm volatile(
        "add.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b))
    );
    return c;
}

CUTE_DEVICE
float2 float2_mul(const float2 &a, const float2 &b) {
    float2 c;
    asm volatile(
        "mul.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE
float2 float2_fma(const float2 &a, const float2 &b, const float2 &c) {

    float2 d;
    asm volatile(
        "fma.rn.f32x2 %0, %1, %2, %3;\n"
        : "=l"(reinterpret_cast<uint64_t&>(d))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)),
          "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

CUTE_DEVICE
float2 float2_neg(const float2 &a) {
    float2 t = {-1.0f, -1.0f};
    return float2_mul(a, t);
}

CUTE_DEVICE
void st_bulk(void* dst_ptr, int64_t size) {
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
    asm volatile (
        "st.bulk.weak.shared::cta [%0], %1, 0;\n"
        :
        : "r"(dst_addr), "l"(size)
        : "memory"
    );
}

struct CUTE_ALIGNAS(16) CLCResponseObj {

    char opaque[16];
};

struct CLCResult {
    int is_valid;
    int x, y, z;
};

CUTE_DEVICE
void issue_clc_query(transac_bar_t &bar, CLCResponseObj &response_obj) {
    uint32_t response_addr = cute::cast_smem_ptr_to_uint(response_obj.opaque);
    uint32_t mbarrier_addr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n"
        :
        : "r"(response_addr), "r"(mbarrier_addr)
    );
}

CUTE_DEVICE
void issue_clc_query_multicast_cluster_all(transac_bar_t &bar, CLCResponseObj &response_obj) {
    uint32_t response_addr = cute::cast_smem_ptr_to_uint(response_obj.opaque);
    uint32_t mbarrier_addr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n"
        :
        : "r"(response_addr), "r"(mbarrier_addr)
    );
}

template<bool USE_LD_ACQUIRE>
CUTE_DEVICE
CLCResult get_clc_query_response(CLCResponseObj &response_obj) {
    uint32_t response_addr = cute::cast_smem_ptr_to_uint(&response_obj);
    CLCResult result;
    #define EMIT_ASM(LD_MODIFIER)                                                                   \
        asm volatile(                                                                               \
            "{\n"                                                                                   \
            ".reg .pred p1;\n\t"                                                                    \
            ".reg .b128 clc_result;\n\t"                                                            \
            "ld" LD_MODIFIER ".shared.b128 clc_result, [%4];\n\t"                                   \
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"           \
            "selp.u32 %3, 1, 0, p1;\n\t"                                                            \
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_result;\n\t" \
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_result;\n\t" \
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_result;\n\t" \
            "}\n"                                                                                   \
            : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.is_valid)                 \
            : "r"(response_addr)                                                                    \
            : "memory"                                                                              \
        );
    if constexpr (USE_LD_ACQUIRE) {
        EMIT_ASM(".acquire.cta");
    } else {
        EMIT_ASM("");
    }
    return result;
}

#define KU_LDG_256(global_addr, result, NC_STR, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR, L2_PREFETCH_SIZE_STR) \
    { \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>, "`global_addr` must be a pointer"); \
        static_assert(std::is_pointer_v<decltype(result)> || std::is_array_v<decltype(result)>, "`result` must be a pointer"); \
        uint64_t* result_as_uint64_ptr = (uint64_t*)(result); \
        asm volatile( \
            "ld.global" NC_STR ".L1::" L1_CACHE_HINT_STR ".L2::" L2_CACHE_HINT_STR ".L2::" L2_PREFETCH_SIZE_STR ".v4.u64 {%0, %1, %2, %3}, [%4];\n" \
            : "=l"(result_as_uint64_ptr[0]), "=l"(result_as_uint64_ptr[1]), \
            "=l"(result_as_uint64_ptr[2]), "=l"(result_as_uint64_ptr[3]) \
            : "l"(global_addr) \
        ); \
    }

#define KU_STG_256(global_addr, src, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR) \
    { \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>, "`global_addr` must be a pointer"); \
        static_assert(std::is_pointer_v<decltype(src)> || std::is_array_v<decltype(src)>, "`src` must be a pointer"); \
        uint64_t const* src_as_uint64_ptr = (uint64_t const*)(src); \
        asm volatile( \
            "st.global.L1::" L1_CACHE_HINT_STR ".L2::" L2_CACHE_HINT_STR ".v4.u64 [%0], {%1, %2, %3, %4};\n" \
            : \
            : "l"(global_addr), "l"(src_as_uint64_ptr[0]), "l"(src_as_uint64_ptr[1]), \
            "l"(src_as_uint64_ptr[2]), "l"(src_as_uint64_ptr[3]) \
        ); \
    }

}

namespace kerutils {

CUTE_DEVICE
void umma_arrive_noelect(transac_bar_t &bar) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        :
        :"r"(bar_intptr)
    );
}

CUTE_DEVICE
void umma_arrive_multicast_noelect(transac_bar_t &bar, uint16_t cta_mask) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :
        :"r"(bar_intptr), "h"(cta_mask)
    );
}

CUTE_DEVICE
void umma_arrive_2x1SM_noelect(transac_bar_t &bar) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        :
        :"r"(bar_intptr)
    );
}

CUTE_DEVICE
void umma_arrive_multicast_2x1SM_noelect(transac_bar_t &bar, uint16_t cta_mask) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :
        :"r"(bar_intptr), "h"(cta_mask)
    );
}

__device__ __forceinline__ void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

template <int kNumElements>
__device__ __forceinline__
void tmem_ld_32dp32bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 || kNumElements == 16 || kNumElements == 32 || kNumElements == 64 || kNumElements == 128, "Invalid kNumElements");

#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumElements == 1) {
            cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 2) {
            cute::SM100_TMEM_LOAD_32dp32b2x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 4) {
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 8) {
            cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 16) {
            cute::SM100_TMEM_LOAD_32dp32b16x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 32) {
            cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 64) {
            cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 128) {
            cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_start, data[Is]...);
        }
    }(cute::make_index_sequence<kNumElements>{});
#endif
}

template <int kNumElements, bool USE_MAX, bool USE_ABS, bool USE_NAN>
__device__ __forceinline__
void tmem_ld_red_32dp32bNx(uint32_t tmem_start, void* data_, float& redval) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumElements == 2 || kNumElements == 4 || kNumElements == 8 ||
                  kNumElements == 16 || kNumElements == 32 || kNumElements == 64 ||
                  kNumElements == 128,
                  "Invalid kNumElements for tcgen05.ld.red (must be power of 2, at least 2)");
#ifndef __VSCODE_IDE__
    static constexpr char s_max[] = ".max";
    static constexpr char s_min[] = ".min";
    static constexpr char s_abs[] = ".abs";
    static constexpr char s_nan[] = ".NaN";
    static constexpr char s_empty[] = "";

    if constexpr (kNumElements == 2) {

        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x2%4%5%6.f32"
            " {%0, %1}, %2, [%3];\n"
            : "=r"(data[0]), "=r"(data[1]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 4) {

        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x4%6%7%8.f32"
            " {%0, %1, %2, %3}, %4, [%5];\n"
            : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 8) {

        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x8%10%11%12.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7}, %8, [%9];\n"
            : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]),
              "=r"(data[4]), "=r"(data[5]), "=r"(data[6]), "=r"(data[7]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 16) {

        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x16%18%19%20.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15}, %16, [%17];\n"
            : "=r"(data[0]),  "=r"(data[1]),  "=r"(data[2]),  "=r"(data[3]),
              "=r"(data[4]),  "=r"(data[5]),  "=r"(data[6]),  "=r"(data[7]),
              "=r"(data[8]),  "=r"(data[9]),  "=r"(data[10]), "=r"(data[11]),
              "=r"(data[12]), "=r"(data[13]), "=r"(data[14]), "=r"(data[15]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 32) {

        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x32%34%35%36.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15,"
            " %16, %17, %18, %19,"
            " %20, %21, %22, %23,"
            " %24, %25, %26, %27,"
            " %28, %29, %30, %31}, %32, [%33];\n"
            : "=r"(data[0]),  "=r"(data[1]),  "=r"(data[2]),  "=r"(data[3]),
              "=r"(data[4]),  "=r"(data[5]),  "=r"(data[6]),  "=r"(data[7]),
              "=r"(data[8]),  "=r"(data[9]),  "=r"(data[10]), "=r"(data[11]),
              "=r"(data[12]), "=r"(data[13]), "=r"(data[14]), "=r"(data[15]),
              "=r"(data[16]), "=r"(data[17]), "=r"(data[18]), "=r"(data[19]),
              "=r"(data[20]), "=r"(data[21]), "=r"(data[22]), "=r"(data[23]),
              "=r"(data[24]), "=r"(data[25]), "=r"(data[26]), "=r"(data[27]),
              "=r"(data[28]), "=r"(data[29]), "=r"(data[30]), "=r"(data[31]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 64) {

        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x64%66%67%68.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15,"
            " %16, %17, %18, %19,"
            " %20, %21, %22, %23,"
            " %24, %25, %26, %27,"
            " %28, %29, %30, %31,"
            " %32, %33, %34, %35,"
            " %36, %37, %38, %39,"
            " %40, %41, %42, %43,"
            " %44, %45, %46, %47,"
            " %48, %49, %50, %51,"
            " %52, %53, %54, %55,"
            " %56, %57, %58, %59,"
            " %60, %61, %62, %63}, %64, [%65];\n"
            : "=r"(data[0]),  "=r"(data[1]),  "=r"(data[2]),  "=r"(data[3]),
              "=r"(data[4]),  "=r"(data[5]),  "=r"(data[6]),  "=r"(data[7]),
              "=r"(data[8]),  "=r"(data[9]),  "=r"(data[10]), "=r"(data[11]),
              "=r"(data[12]), "=r"(data[13]), "=r"(data[14]), "=r"(data[15]),
              "=r"(data[16]), "=r"(data[17]), "=r"(data[18]), "=r"(data[19]),
              "=r"(data[20]), "=r"(data[21]), "=r"(data[22]), "=r"(data[23]),
              "=r"(data[24]), "=r"(data[25]), "=r"(data[26]), "=r"(data[27]),
              "=r"(data[28]), "=r"(data[29]), "=r"(data[30]), "=r"(data[31]),
              "=r"(data[32]), "=r"(data[33]), "=r"(data[34]), "=r"(data[35]),
              "=r"(data[36]), "=r"(data[37]), "=r"(data[38]), "=r"(data[39]),
              "=r"(data[40]), "=r"(data[41]), "=r"(data[42]), "=r"(data[43]),
              "=r"(data[44]), "=r"(data[45]), "=r"(data[46]), "=r"(data[47]),
              "=r"(data[48]), "=r"(data[49]), "=r"(data[50]), "=r"(data[51]),
              "=r"(data[52]), "=r"(data[53]), "=r"(data[54]), "=r"(data[55]),
              "=r"(data[56]), "=r"(data[57]), "=r"(data[58]), "=r"(data[59]),
              "=r"(data[60]), "=r"(data[61]), "=r"(data[62]), "=r"(data[63]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 128) {

        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x128%130%131%132.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15,"
            " %16, %17, %18, %19,"
            " %20, %21, %22, %23,"
            " %24, %25, %26, %27,"
            " %28, %29, %30, %31,"
            " %32, %33, %34, %35,"
            " %36, %37, %38, %39,"
            " %40, %41, %42, %43,"
            " %44, %45, %46, %47,"
            " %48, %49, %50, %51,"
            " %52, %53, %54, %55,"
            " %56, %57, %58, %59,"
            " %60, %61, %62, %63,"
            " %64, %65, %66, %67,"
            " %68, %69, %70, %71,"
            " %72, %73, %74, %75,"
            " %76, %77, %78, %79,"
            " %80, %81, %82, %83,"
            " %84, %85, %86, %87,"
            " %88, %89, %90, %91,"
            " %92, %93, %94, %95,"
            " %96, %97, %98, %99,"
            " %100, %101, %102, %103,"
            " %104, %105, %106, %107,"
            " %108, %109, %110, %111,"
            " %112, %113, %114, %115,"
            " %116, %117, %118, %119,"
            " %120, %121, %122, %123,"
            " %124, %125, %126, %127}, %128, [%129];\n"
            : "=r"(data[0]),   "=r"(data[1]),   "=r"(data[2]),   "=r"(data[3]),
              "=r"(data[4]),   "=r"(data[5]),   "=r"(data[6]),   "=r"(data[7]),
              "=r"(data[8]),   "=r"(data[9]),   "=r"(data[10]),  "=r"(data[11]),
              "=r"(data[12]),  "=r"(data[13]),  "=r"(data[14]),  "=r"(data[15]),
              "=r"(data[16]),  "=r"(data[17]),  "=r"(data[18]),  "=r"(data[19]),
              "=r"(data[20]),  "=r"(data[21]),  "=r"(data[22]),  "=r"(data[23]),
              "=r"(data[24]),  "=r"(data[25]),  "=r"(data[26]),  "=r"(data[27]),
              "=r"(data[28]),  "=r"(data[29]),  "=r"(data[30]),  "=r"(data[31]),
              "=r"(data[32]),  "=r"(data[33]),  "=r"(data[34]),  "=r"(data[35]),
              "=r"(data[36]),  "=r"(data[37]),  "=r"(data[38]),  "=r"(data[39]),
              "=r"(data[40]),  "=r"(data[41]),  "=r"(data[42]),  "=r"(data[43]),
              "=r"(data[44]),  "=r"(data[45]),  "=r"(data[46]),  "=r"(data[47]),
              "=r"(data[48]),  "=r"(data[49]),  "=r"(data[50]),  "=r"(data[51]),
              "=r"(data[52]),  "=r"(data[53]),  "=r"(data[54]),  "=r"(data[55]),
              "=r"(data[56]),  "=r"(data[57]),  "=r"(data[58]),  "=r"(data[59]),
              "=r"(data[60]),  "=r"(data[61]),  "=r"(data[62]),  "=r"(data[63]),
              "=r"(data[64]),  "=r"(data[65]),  "=r"(data[66]),  "=r"(data[67]),
              "=r"(data[68]),  "=r"(data[69]),  "=r"(data[70]),  "=r"(data[71]),
              "=r"(data[72]),  "=r"(data[73]),  "=r"(data[74]),  "=r"(data[75]),
              "=r"(data[76]),  "=r"(data[77]),  "=r"(data[78]),  "=r"(data[79]),
              "=r"(data[80]),  "=r"(data[81]),  "=r"(data[82]),  "=r"(data[83]),
              "=r"(data[84]),  "=r"(data[85]),  "=r"(data[86]),  "=r"(data[87]),
              "=r"(data[88]),  "=r"(data[89]),  "=r"(data[90]),  "=r"(data[91]),
              "=r"(data[92]),  "=r"(data[93]),  "=r"(data[94]),  "=r"(data[95]),
              "=r"(data[96]),  "=r"(data[97]),  "=r"(data[98]),  "=r"(data[99]),
              "=r"(data[100]), "=r"(data[101]), "=r"(data[102]), "=r"(data[103]),
              "=r"(data[104]), "=r"(data[105]), "=r"(data[106]), "=r"(data[107]),
              "=r"(data[108]), "=r"(data[109]), "=r"(data[110]), "=r"(data[111]),
              "=r"(data[112]), "=r"(data[113]), "=r"(data[114]), "=r"(data[115]),
              "=r"(data[116]), "=r"(data[117]), "=r"(data[118]), "=r"(data[119]),
              "=r"(data[120]), "=r"(data[121]), "=r"(data[122]), "=r"(data[123]),
              "=r"(data[124]), "=r"(data[125]), "=r"(data[126]), "=r"(data[127]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    }
#endif
}

template <int kNumReplications>
__device__ __forceinline__
void tmem_ld_16dp128bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumReplications == 1 || kNumReplications == 2 || kNumReplications == 4 || kNumReplications == 8 || kNumReplications == 16 || kNumReplications == 32 || kNumReplications == 64, "Invalid kNumReplications");
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumReplications == 1) {
            cute::SM100_TMEM_LOAD_16dp128b1x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 2) {
            cute::SM100_TMEM_LOAD_16dp128b2x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 4) {
            cute::SM100_TMEM_LOAD_16dp128b4x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 8) {
            cute::SM100_TMEM_LOAD_16dp128b8x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 16) {
            cute::SM100_TMEM_LOAD_16dp128b16x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 32) {
            cute::SM100_TMEM_LOAD_16dp128b32x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 64) {
            cute::SM100_TMEM_LOAD_16dp128b64x::copy(tmem_start, data[Is]...);
        }
    }(cute::make_index_sequence<kNumReplications*2>{});
#endif
}

template <int kNumReplications>
__device__ __forceinline__
void tmem_ld_16dp256bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumReplications == 1 || kNumReplications == 2 || kNumReplications == 4 || kNumReplications == 8 || kNumReplications == 16 || kNumReplications == 32, "Invalid kNumReplications");
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumReplications == 1) {
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 2) {
            cute::SM100_TMEM_LOAD_16dp256b2x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 4) {
            cute::SM100_TMEM_LOAD_16dp256b4x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 8) {
            cute::SM100_TMEM_LOAD_16dp256b8x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 16) {
            cute::SM100_TMEM_LOAD_16dp256b16x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 32) {
            cute::SM100_TMEM_LOAD_16dp256b32x::copy(tmem_start, data[Is]...);
        }
    }(cute::make_index_sequence<kNumReplications*4>{});
#endif
}

template <int kNumElements>
__device__ __forceinline__
void tmem_st_32dp32bNx(uint32_t tmem_start, void const* data_) {
    uint32_t const* data = (uint32_t const*)data_;
    static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 || kNumElements == 16 || kNumElements == 32 || kNumElements == 64 || kNumElements == 128, "Invalid kNumElements");
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumElements == 1) {
            cute::SM100_TMEM_STORE_32dp32b1x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 2) {
            cute::SM100_TMEM_STORE_32dp32b2x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 4) {
            cute::SM100_TMEM_STORE_32dp32b4x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 8) {
            cute::SM100_TMEM_STORE_32dp32b8x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 16) {
            cute::SM100_TMEM_STORE_32dp32b16x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 32) {
            cute::SM100_TMEM_STORE_32dp32b32x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 64) {
            cute::SM100_TMEM_STORE_32dp32b64x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 128) {
            cute::SM100_TMEM_STORE_32dp32b128x::copy(data[Is]..., tmem_start);
        }
    }(cute::make_index_sequence<kNumElements>{});
#endif
}

template<int kNumReplications, int kNumElemsPerUnit, typename F>
__device__ __forceinline__
void tmem_ld_st_decomposed(uint32_t tmem_start, void* data, F&& tmem_ld_st_fn) {
    if constexpr (kNumReplications == 0) {

        return;
    } else {

        constexpr int kHighBit = 1 << (31 - __builtin_clz(kNumReplications));
        tmem_ld_st_fn.template operator()<kHighBit>(tmem_start, data);

        constexpr int kRemaining = kNumReplications - kHighBit;
        if constexpr (kRemaining > 0) {
            tmem_ld_st_decomposed<kRemaining, kNumElemsPerUnit>(
                tmem_start + kHighBit * kNumElemsPerUnit,
                (void*)((uint32_t*)data + kHighBit * kNumElemsPerUnit),
                tmem_ld_st_fn
            );
        }
    }
}

}
""",
    "kerutils_include/kerutils/device/cuda/sm100/gemm.cuh": r"""#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/cuda/common.h"

namespace cute {

namespace UMMA {

template <class ValueType, class StorageType, int FOLD_DEGREE>
struct tmem_frg_elastic : tmem_frg_base {
  static_assert(sizeof_bits_v<ValueType> <= sizeof_bits_v<StorageType>, "TMEM MMA allocations require StorageType big enough for ValueType.");

  template <class TmemShape>
  CUTE_HOST_DEVICE constexpr static auto
  make(TmemShape const& tmem_shape)
  {
    CUTE_STATIC_ASSERT_V(size(tmem_shape)*Int<int(sizeof_bits_v<StorageType>)>{} <= Int<128 * 512 * 32>{},
                        "Requesting more TMEM than is available.");
    CUTE_STATIC_ASSERT_V(rank<0>(tmem_shape) == Int<2>{}, "Expected post-partitioned shape ((M_MMA,N_MMA),...).");
    constexpr int R     = decltype(rank(tmem_shape))::value;
    constexpr int M_MMA = decltype(size<0,0>(tmem_shape))::value;
    constexpr int N_MMA = decltype(size<0,1>(tmem_shape))::value;

    using COL_ADDR = C<sizeof_bits<StorageType>::value / sizeof_bits<ValueType>::value>;
    Layout tmem_restride = Layout<Shape <               _128,   _16384>,
                                  Stride<TMEM::DP<ValueType>, COL_ADDR>>{};
    Layout tmem_atom = Layout<Shape<Int<M_MMA>, Shape<Int<N_MMA/FOLD_DEGREE>, Int<FOLD_DEGREE>>>,
                              Stride<_1, Stride<_128, Int<128/FOLD_DEGREE>> >>{};
    constexpr int tile_stride = (128/M_MMA) / FOLD_DEGREE;
    Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1, R>(tmem_shape),
      compact_col_major(take<1,R>(tmem_shape),Int<tile_stride>{})));

    return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));
  }
};

}

template <class ValueType, class StorageType, int FOLD_DEGREE>
struct MakeTensor<UMMA::tmem_frg_elastic<ValueType, StorageType, FOLD_DEGREE>> {
  template <class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Shape const& tmem_shape) {
    return UMMA::tmem_frg_elastic<ValueType, StorageType, FOLD_DEGREE>::make(shape(tmem_shape));
  }
};

template <class ValueType, int FOLD_DEGREE>
struct MakeTensor<UMMA::tmem_frg_elastic<ValueType, uint32_t, FOLD_DEGREE>> {
  template <class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Shape const& tmem_shape) {
    return UMMA::tmem_frg_elastic<ValueType, uint32_t, FOLD_DEGREE>::make(shape(tmem_shape));
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          int FOLD_DEGREE_A = 1,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_TS_NOELECT
{
  static_assert(M == 32 || M == 64 || M == 128, "SM100_MMA_F16BF16_WS_TS_NOELECT M-mode size should be 32, 64 or 128 for 1 CTA cluster MMA.");
  static_assert(N == 64 || N == 128 || N == 256,
                "SM100_MMA_F16BF16_WS_TS_NOELECT N-mode size should be 64, 128 or 256");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC));
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          int FOLD_DEGREE_A,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_WS_TS_NOELECT<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                FOLD_DEGREE_A,
                                a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_WS_TS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::tmem_frg_elastic<a_type, a_type, FOLD_DEGREE_A>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_elastic<c_type, uint32_t, 128/M>;

  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_WS_TS_NOELECT<a_type, b_type, c_type,
                  M, N,
                  a_major, b_major,
                  FOLD_DEGREE_A,
                  a_neg, b_neg>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_SS_NOELECT
{
  static_assert(M == 32 || M == 64 || M == 128, "SM100_MMA_F16BF16_WS_SS_NOELECT M-mode size should be 32, 64 or 128 for 1 CTA cluster MMA.");
  static_assert(N == 64 || N == 128 || N == 256,
                "SM100_MMA_F16BF16_WS_SS_NOELECT N-mode size should be 64, 128 or 256");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC));
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_WS_SS_NOELECT<a_type, b_type, c_type,
                                M, N, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_WS_SS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_elastic<c_type, uint32_t, 128/M>;

  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_WS_SS_NOELECT<a_type, b_type, c_type,
                  M, N, a_major, b_major,
                  a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_2x1SM_TS_NOELECT
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F16BF16_2x1SM_TS_NOELECT M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_F16BF16_2x1SM_TS_NOELECT N-mode size should be a multiple of 16 between 16 and 256.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_F16BF16_2x1SM_TS_NOELECT A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
        "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_2x1SM_TS_NOELECT without CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_TS_NOELECT<a_type, b_type, c_type,
                                     M, N,
                                     a_major, b_major,
                                     a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_2x1SM_TS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::tmem_frg_2sm<a_type, a_type, UMMA::TmemAllocMode::Duplicated>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<a_type, b_type, c_type,
                       M, N,
                       a_major, b_major,
                       a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_2x1SM_SS_NOELECT
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F16BF16_2x1SM_SS_NOELECT M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_F16BF16_2x1SM_SS_NOELECT N-mode size should be a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
        "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_2x1SM_SS_NOELECT without CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS_NOELECT<a_type, b_type, c_type,
                                     M, N, a_major, b_major,
                                     a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_2x1SM_SS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<a_type, b_type, c_type,
                       M, N,
                       a_major, b_major,
                       a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_TS_NOELECT
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16_TS_NOELECT M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0)  && (8 <= N)  && (N <= 256),
                "SM100_MMA_F16BF16_TS_NOELECT N-mode size should be a multiple of 8 between 8 and 256 for M=64");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_F16BF16_TS_NOELECT A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_TS_NOELECT<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_TS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_TS_NOELECT<a_type, b_type, c_type,
                  M, N,
                  a_major, b_major,
                  a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS_NOELECT
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16_SS_NOELECT M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0)  && (8 <= N)  && (N <= 256),
                "SM100_MMA_F16BF16_SS_NOELECT N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_SS_NOELECT<a_type, b_type, c_type,
                                M, N, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_SS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_SS_NOELECT<a_type, b_type, c_type,
                  M, N, a_major, b_major,
                  a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

}
""",
    "kerutils_include/kerutils/device/cuda/sm100/tma_cta_group2_nosplit.cuh": r"""#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/cuda/common.h"

namespace cute {

struct SM100_TMA_2SM_LOAD_1D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy([[maybe_unused]] void const* desc_ptr, [[maybe_unused]] uint64_t* mbar_ptr, [[maybe_unused]] uint64_t cache_hint,
       [[maybe_unused]] void      * smem_ptr,
       [[maybe_unused]] int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);

    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_2D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy([[maybe_unused]] void const* desc_ptr, [[maybe_unused]] uint64_t* mbar_ptr, [[maybe_unused]] uint64_t cache_hint,
       [[maybe_unused]] void      * smem_ptr,
       [[maybe_unused]] int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);

    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_3D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy([[maybe_unused]] void const* desc_ptr, [[maybe_unused]] uint64_t* mbar_ptr, [[maybe_unused]] uint64_t cache_hint,
       [[maybe_unused]] void      * smem_ptr,
       [[maybe_unused]] int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);

    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_4D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);

    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_5D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);

    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0)
  {
    return SM100_TMA_2SM_LOAD_1D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    return SM100_TMA_2SM_LOAD_2D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
    return SM100_TMA_2SM_LOAD_3D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
    return SM100_TMA_2SM_LOAD_4D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    return SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
  using PREFETCH = typename SM90_TMA_LOAD::PREFETCH;
};
struct SM100_TMA_2SM_LOAD_NOSPLIT_OP : SM100_TMA_2SM_LOAD_NOSPLIT {};

template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;

  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;

  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;

  using RefLayout = SrcLayout;

  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
  with(
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {

    return {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }

  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
  with(
    TmaDescriptor const* new_tma_desc,
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {

    return {new_tma_desc, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template <class NumBitsPerTMA>
struct Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
  : TMA_LOAD_Unpack<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
{
  using ThrID     = Layout<_1>;

  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;

  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;

  using RefLayout = SrcLayout;

  tuple<
  TmaDescriptor const*,
  uint64_t*,
  uint64_t
  > const opargs_;
  CUTE_HOST_DEVICE
  Copy_Traits(TmaDescriptor const* desc, uint64_t* mbar, uint64_t cache)
    : opargs_(desc, mbar, cache) {}
};

}
""",
    "v4_entry.cuh": r"""#pragma once

#include <cstdint>
#include <exception>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include "pattn/kernel.h"
#include "pattn/kernel.cuh"

namespace v4 {

namespace ca = pattn::sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;

inline constexpr ca::Config kPrefillGrp =
    ca::Config{SparseAttnFwdMode::Prefill, ModelType::V4, ModelType::V4, 64, false, true, true, false, false, true};

inline void expect(bool ok, const char *what) {
    if (!ok) TVM_FFI_THROW(ValueError) << "v4 mega prefill: " << what;
}

inline bool is_dtype(const tvm::ffi::TensorView &t, uint8_t code, uint8_t bits) {
    return t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1;
}

inline bool fits_int(int64_t v) { return v >= 0 && v <= INT32_MAX; }

struct MegaPrefill {
    static void run(tvm::ffi::TensorView q, tvm::ffi::TensorView kv, tvm::ffi::TensorView indices,
                    tvm::ffi::TensorView topk_length, tvm::ffi::TensorView attn_sink, tvm::ffi::TensorView positions,
                    tvm::ffi::TensorView cos_sin_cache, tvm::ffi::TensorView out_fp8, tvm::ffi::TensorView out_sf,
                    tvm::ffi::TensorView max_logits, tvm::ffi::TensorView lse, double sm_scale, double rms_norm_eps) {
        constexpr float LOG_2_E = 1.44269504f;
        const int64_t s_q = q.size(0);
        expect(q.ndim() == 3 && s_q > 0 && q.size(1) == 64 && q.size(2) == 512 && q.stride(2) == 1 &&
               q.stride(1) == 512 && is_dtype(q, kDLBfloat, 16), "q must be [s_q > 0, 64, 512] bf16, contiguous within a token");
        expect(kv.ndim() == 3 && kv.size(0) > 0 && kv.size(1) == 1 && kv.size(2) == 512 && kv.stride(2) == 1 &&
               is_dtype(kv, kDLBfloat, 16), "kv must be the [s_kv > 0, 1, 512] bf16 workspace");
        expect(indices.ndim() == 3 && indices.size(0) == s_q && indices.size(1) == 1 && indices.stride(2) == 1 &&
               is_dtype(indices, kDLInt, 32), "indices must be [s_q, 1, topk] int32");
        for (const auto *t : {&topk_length, &positions}) {
            expect(t->ndim() == 1 && t->size(0) == s_q && t->stride(0) == 1 && is_dtype(*t, kDLInt, 32),
                   "topk_length / positions must be [s_q] int32 contiguous");
        }
        expect(attn_sink.ndim() == 1 && attn_sink.size(0) == 64 && attn_sink.stride(0) == 1 && is_dtype(attn_sink, kDLFloat, 32),
               "attn_sink must be [64] fp32");
        expect(cos_sin_cache.ndim() == 2 && cos_sin_cache.size(1) == 64 && cos_sin_cache.stride(1) == 1 &&
               cos_sin_cache.stride(0) == 64 && is_dtype(cos_sin_cache, kDLFloat, 32), "cos_sin_cache must be [*, 64] fp32 contiguous");
        expect(out_fp8.ndim() == 3 && out_fp8.size(0) == s_q && out_fp8.size(1) == 8 && out_fp8.size(2) == 4096 &&
               out_fp8.stride(2) == 1 && out_fp8.stride(1) == 4096 && out_fp8.stride(0) == 8 * 4096 && out_fp8.dtype().bits == 8,
               "out_fp8 must be [s_q, 8, 4096] contiguous");
        expect(out_sf.ndim() == 3 && out_sf.size(0) == s_q && out_sf.size(1) == 8 && out_sf.size(2) == 32 && out_sf.stride(0) == 1 &&
               is_dtype(out_sf, kDLInt, 32), "out_sf must be the legacy API's [s_q, 8, 32] int32 view (stride(0) == 1)");
        for (const auto *t : {&max_logits, &lse}) {
            expect(t->ndim() == 2 && t->size(0) == s_q && t->size(1) == 64 && t->stride(1) == 1 && t->stride(0) == 64 &&
                   is_dtype(*t, kDLFloat, 32), "max_logits / lse must be [s_q, 64] fp32 contiguous");
        }
        for (int64_t stride : {q.stride(0), kv.stride(0), kv.stride(1), indices.stride(0), indices.stride(1),
                               out_sf.stride(1), out_sf.stride(2)}) {
            expect(fits_int(stride), "a stride exceeds the int32 range");
        }
        const DLDevice dev = q.device();
        int current = -1, num_sms = 0;
        expect(cudaGetDevice(&current) == cudaSuccess && current == dev.device_id,
               "the current CUDA device must be q's (the legacy API runs under q's device guard)");
        expect(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev.device_id) == cudaSuccess,
               "cannot read the SM count");

        ca::ParamT<SparseAttnFwdMode::Prefill> p{};
        p.s_q = (int)s_q; p.s_kv = (int)kv.size(0); p.h_q = 64; p.h_kv = 1; p.d_qk = 512; p.d_v = 512;
        p.topk = (int)indices.size(2);
        p.sm_scale = (float)sm_scale; p.sm_scale_div_log2 = (float)(sm_scale * LOG_2_E);
        p.q = (cutlass::bfloat16_t *)q.data_ptr();
        p.kv = (cutlass::bfloat16_t *)kv.data_ptr();
        p.indices = (int *)indices.data_ptr();
        p.attn_sink = (float *)attn_sink.data_ptr();
        p.topk_length = (int *)topk_length.data_ptr();
        p.stride_q_s_q = (int)q.stride(0); p.stride_q_h_q = (int)q.stride(1);
        p.stride_kv_s_kv = (int)kv.stride(0); p.stride_kv_h_kv = (int)kv.stride(1);
        p.stride_indices_s_q = (int)indices.stride(0); p.stride_indices_h_kv = (int)indices.stride(1);
        p.out = nullptr;
        p.max_logits = (float *)max_logits.data_ptr();
        p.lse = (float *)lse.data_ptr();
        p.num_sm = num_sms;
        p.stats_in_log2 = false;
        p.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
        p.enable_q_norm = false;
        p.rms_norm_eps = (float)rms_norm_eps;
        p.token_positions = (uint32_t *)positions.data_ptr();
        p.is_rope_neox_style = false;
        p.rope_dim = 64;
        p.cos_sin_cache = (float *)cos_sin_cache.data_ptr();
        p.n_wv_group = 8;
        p.wv_group_size = 8;
        p.num_per_channels = 32;
        p.use_tma_aligned_col_major_sf = true;
        p.round_sf = true;
        p.use_packed_ue8m0 = true;
        p.out_fp8 = (cutlass::float_e4m3_t *)out_fp8.data_ptr();
        p.out_sf = (uint32_t *)out_sf.data_ptr();
        p.stride_out_sf_wv_group = (uint32_t)out_sf.stride(1);
        p.stride_out_sf_head_dim = (uint32_t)out_sf.stride(2);
        try {
            ca::run_fused_norm_rope_attn_rope_cast_fwd_kernel<kPrefillGrp>(p);
        } catch (const std::exception &e) {
            TVM_FFI_THROW(RuntimeError) << "v4 mega prefill: " << e.what();
        }
    }
};

}
""",
}
