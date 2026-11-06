ENTRY = "dattn_main.cuh"

SOURCE = r"""
#pragma once

#include <cstdint>
#include <exception>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include "pattn/kernel.h"
#include "pattn/kernel.cuh"

namespace dattn_entry {

namespace ca = pattn::sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;

inline constexpr ca::Config kDecodeFp4 =
    ca::Config{SparseAttnFwdMode::Decode, ModelType::V41, ModelType::V41_FP4, 64, false, true, false, true, false, false};

inline void expect(bool ok, const char *what) {
    if (!ok) TVM_FFI_THROW(ValueError) << "dattn_entry mega decode: " << what;
}

inline bool is_dtype(const tvm::ffi::TensorView &t, uint8_t code, uint8_t bits) {
    return t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1;
}

struct MegaDecodeFp4 {
    static void run(tvm::ffi::TensorView q, tvm::ffi::TensorView kv, tvm::ffi::TensorView indices,
                    tvm::ffi::TensorView topk_length, tvm::ffi::TensorView extra_kv, tvm::ffi::TensorView extra_indices,
                    tvm::ffi::TensorView extra_topk_length, tvm::ffi::TensorView attn_sink, tvm::ffi::TensorView positions,
                    tvm::ffi::TensorView cos_sin_cache, tvm::ffi::TensorView out_fp8, tvm::ffi::TensorView out_sf,
                    tvm::ffi::TensorView lse, double sm_scale) {
        constexpr float LOG_2_E = 1.44269504f;
        const int64_t s_q = q.size(0);
        expect(q.ndim() == 3 && q.size(1) == 64 && q.size(2) == 512 && q.stride(2) == 1 && q.stride(1) == 512 &&
               is_dtype(q, kDLBfloat, 16), "q must be [s_q, 64, 512] bf16, contiguous within a token");
        expect(kv.ndim() == 4 && kv.size(2) == 1 && kv.size(3) == 528 && kv.stride(1) == 528 && kv.stride(3) == 1 &&
               kv.dtype().bits == 8, "kv must be the V4.1 fp8 paged cache [blocks, page, 1, 528]");
        expect(extra_kv.ndim() == 4 && extra_kv.size(2) == 1 && extra_kv.size(3) == 288 && extra_kv.stride(1) == 288 &&
               extra_kv.stride(3) == 1 && extra_kv.dtype().bits == 8, "extra_kv must be the V4.1 fp4 paged cache [blocks, page, 1, 288]");
        expect(indices.ndim() == 2 && indices.size(0) == s_q && indices.stride(1) == 1 && is_dtype(indices, kDLInt, 32),
               "indices must be [s_q, topk] int32");
        expect(extra_indices.ndim() == 2 && extra_indices.size(0) == s_q && extra_indices.stride(1) == 1 &&
               is_dtype(extra_indices, kDLInt, 32), "extra_indices must be [s_q, extra_topk] int32");
        for (const auto *t : {&topk_length, &extra_topk_length, &positions}) {
            expect(t->ndim() == 1 && t->size(0) == s_q && t->stride(0) == 1 && is_dtype(*t, kDLInt, 32),
                   "topk_length / extra_topk_length / positions must be [s_q] int32 contiguous");
        }
        expect(attn_sink.ndim() == 1 && attn_sink.size(0) == 64 && attn_sink.stride(0) == 1 && is_dtype(attn_sink, kDLFloat, 32),
               "attn_sink must be [64] fp32");
        expect(cos_sin_cache.ndim() == 2 && cos_sin_cache.size(1) == 64 && cos_sin_cache.stride(1) == 1 &&
               cos_sin_cache.stride(0) == 64 && is_dtype(cos_sin_cache, kDLFloat, 32), "cos_sin_cache must be [*, 64] fp32 contiguous");
        expect(out_fp8.ndim() == 3 && out_fp8.size(0) == s_q && out_fp8.size(1) == 8 && out_fp8.size(2) == 4096 &&
               out_fp8.stride(2) == 1 && out_fp8.stride(1) == 4096 && out_fp8.dtype().bits == 8, "out_fp8 must be [s_q, 8, 4096] contiguous");
        expect(out_sf.ndim() == 3 && out_sf.size(0) == s_q && out_sf.size(1) == 8 && out_sf.size(2) == 32 && out_sf.stride(0) == 1 &&
               is_dtype(out_sf, kDLInt, 32), "out_sf must be the legacy API's [s_q, 8, 32] int32 view (stride(0) == 1)");
        expect(lse.ndim() == 2 && lse.size(0) == s_q && lse.size(1) == 64 && lse.stride(1) == 1 && is_dtype(lse, kDLFloat, 32),
               "lse must be [s_q, 64] fp32");

        ca::ParamT<SparseAttnFwdMode::Decode> p{};
        p.b = 1; p.s_q = (int)s_q; p.h_q = 64; p.h_kv = 1; p.d_qk = 512; p.d_v = 512;
        p.sm_scale = (float)sm_scale; p.sm_scale_div_log2 = (float)(sm_scale * LOG_2_E);
        p.num_blocks = (int)kv.size(0); p.page_block_size = (int)kv.size(1); p.topk = (int)indices.size(1);
        p.model_type = ModelType::V41;
        p.extra_model_type = ModelType::V41_FP4;
        p.q = (cutlass::bfloat16_t *)q.data_ptr();
        p.kv = (cutlass::bfloat16_t *)kv.data_ptr();
        p.indices = (int *)indices.data_ptr();
        p.topk_length = (int *)topk_length.data_ptr();
        p.attn_sink = (float *)attn_sink.data_ptr();
        p.lse = (float *)lse.data_ptr();
        p.out = nullptr;
        p.extra_num_blocks = (int)extra_kv.size(0); p.extra_page_block_size = (int)extra_kv.size(1);
        p.extra_topk = (int)extra_indices.size(1);
        p.extra_kv = (cutlass::bfloat16_t *)extra_kv.data_ptr();
        p.extra_indices = (int *)extra_indices.data_ptr();
        p.extra_topk_length = (int *)extra_topk_length.data_ptr();
        p.stride_q_s_q = (int)q.stride(0); p.stride_q_h_q = 512;
        p.stride_kv_block = (int)kv.stride(0); p.stride_kv_row = 528;
        p.stride_indices_s_q = (int)indices.stride(0);
        p.stride_lse_s_q = (int)lse.stride(0);
        p.stride_extra_kv_block = (int)extra_kv.stride(0); p.stride_extra_kv_row = 288;
        p.stride_extra_indices_s_q = (int)extra_indices.stride(0);
        p.enable_split_kv = false;
        p.enable_q_norm = false;
        p.rms_norm_eps = 0.0f;
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
        const DLDevice dev = q.device();
        p.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
        try {
            ca::run_fused_norm_rope_attn_rope_cast_fwd_kernel<kDecodeFp4>(p);
        } catch (const std::exception &e) {
            TVM_FFI_THROW(RuntimeError) << "dattn_entry mega decode: " << e.what();
        }
    }
};

}
"""
