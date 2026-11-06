
from __future__ import annotations

import torch

from bundle_v9 import rehome_transport as T

SOURCE_LAYERS = (2, 8, 14)


def check_layer(i: int, attn) -> None:
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import use_dsv4_q8kv8_sparse_prefill
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
    from sglang.srt.runtime_context import get_exec

    expected_ratio = 0 if i < 2 else 2
    has_src = attn.compressor is not None or attn.indexer is not None
    expected_src = i in SOURCE_LAYERS
    prefill_backend = getattr(get_exec().kernel, "dsv4_prefill_backend", "auto")
    bad = (
        not (attn.accepts_deepgemm_mxfp8_input() and attn.use_mega_attention and attn.wo_b.deepgemm_mxfp8_input),
        attn.compress_ratio != expected_ratio,
        has_src != expected_src,
        attn.q_lora_rank != T.QLORA or attn.attn_mqa.v_head_dim != T.KV_DIM or attn.head_dim != T.KV_DIM,
        envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get(),
        not envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get(),
        use_dsv4_q8kv8_sparse_prefill(prefill_backend),
        is_unified_kv_triton(),
    )
    if any(bad):
        raise RuntimeError(f"v9-rehome: layer {i} attention failed static checks {[k for k, b in enumerate(bad) if b]}")


def _mega_prefill_woa_wob(attn, attn_backend, q, kv, indices_2d, positions_i32, lens, defer_all_reduce: bool):
    from sglang.srt.layers.attention.deepseek_v4_mega_backend import pack_query
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.models.deepseek_v4 import _fp8_wo_a_einsum

    kernel_num_heads = 64 if attn.n_local_heads <= 64 else 128
    if not attn.mega_wq_b_permuted:
        q = pack_query(q, kernel_num_heads)
    attn_sink = attn._local_attn_sink(kernel_num_heads)
    data, scale = attn_backend._forward_mega_prefill(
        q=q,
        kv=kv,
        indices=indices_2d.unsqueeze(1),
        positions=positions_i32,
        cos_sin_cache=attn.mega_cos_sin_cache,
        attn_sink=attn_sink,
        topk_length=lens,
    )
    groups = attn.n_local_groups
    rows = data.shape[0]
    o_fp8, o_s = data[:, :groups], scale[:, :groups]
    if not attn._mega_mxfp8_output(rows):
        raise RuntimeError(f"v9-rehome: _mega_mxfp8_output({rows}) is False")
    y, y_sf = mhc.empty_deepgemm_mxfp8_pair(rows, groups * attn.o_lora_rank, data.device)
    _fp8_wo_a_einsum(o_fp8, o_s, *attn.mega_wo_a, recipe=(1, 1, 32), out=(y.view(rows, groups, -1), y_sf))
    return attn._project_wo_b((y, y_sf), defer_all_reduce)


def owner_attention(i: int, attn, x_kept, x_quant_kept, positions, fb, n: int, kept: int, out_moves, owner_kv):
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged_dsv4
    from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool

    is_src = i in SOURCE_LAYERS
    qkv_kept = attn.wqkv_a(x_quant_kept if x_quant_kept is not None else x_kept)[0]
    kva_all = T.view("kva", n)
    kva_all[:kept].copy_(qkv_kept[:, T.QLORA:])
    qlora_kept, q_for_wqb_kept = attn._normalize_q_lora(qkv_kept[:, :T.QLORA])
    if is_src:
        qlora_all = T.view("qlora", n)
        qlora_all[:kept].copy_(qlora_kept)
    q_kept = attn._compute_q_b(q_for_wqb_kept, positions[:kept], None)
    del qkv_kept, qlora_kept, q_for_wqb_kept

    for mv in out_moves:
        T.wait(mv.helper, T.XQ)

    attn_backend = get_attn_backend()
    kv_pool = get_token_to_kv_pool()
    if kv_pool.request_window is not None or kv_pool.uniform_fp8 or attn_backend.pre_kv_store_hook is not None:
        raise RuntimeError("v9-rehome: unsupported request_window / uniform_fp8 / pre_kv_store_hook on prefill")

    kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
        layer_id=attn.layer_id,
        swa_loc=attn_backend.get_swa_out_cache_loc(fb),
        kv=kva_all,
        kv_weight=attn.kv_norm.weight.data,
        eps=attn.eps,
        freqs_cis=attn.freqs_cis,
        positions=positions,
    )
    if is_src:
        attn_backend.forward_low_ratio_sources(
            layer=attn,
            x=T.view("x", n),
            q_lora=qlora_all,
            positions=positions,
            forward_batch=fb,
        )

    core_meta = attn_backend.forward_metadata.core_attn_metadata
    if attn_backend._mega_paged_prefill(
        fb, num_qo_tokens=n, layer_id=attn.layer_id, compress_ratio=attn.compress_ratio, num_heads=attn.n_local_heads
    ) or not attn_backend._use_sparse_prefill(fb, num_qo_tokens=n):
        raise RuntimeError(f"v9-rehome: layer {i} owner n={n} did not take _forward_prefill_sparse")

    cache = attn_backend.forward_metadata.sparse_prefill_cache
    if cache is None:
        cache = attn_backend._build_sparse_prefill_chunk_cache(fb, core_meta, num_qo_tokens=n)
        attn_backend.forward_metadata.sparse_prefill_cache = cache
    n_swa = int(cache.swa_token_ids.shape[0])
    if cache.num_qo_tokens != n or cache.swa_indices is not None or n_swa != owner_kv.n_swa:
        raise RuntimeError(
            f"v9-rehome: layer {i} cache mismatch (qo={cache.num_qo_tokens}/{n}, "
            f"swa_idx={cache.swa_indices is not None}, n_swa={n_swa}/{owner_kv.n_swa})"
        )

    compressed_slice = None
    extra_k_cache = None
    extra_page_size = None
    flat_token_ids = None
    n_compressed = 0
    if attn.compress_ratio == 0:
        workspace = attn_backend.sparse_prefill_kv_buffer.get(n_swa)
        combined_indices = cache.c0_combined_indices
        combined_lens = cache.c0_combined_lens
        swa_slice = workspace
    else:
        extra_page_size = kv_pool.get_extra_key_page_size(attn.layer_id)
        extra_k_cache = kv_pool.get_extra_key_buffer(attn.layer_id)
        flat_token_ids, combined_indices, combined_lens = cache.layer_inputs(
            attn.compress_ratio, core_meta, extra_page_size
        )
        n_compressed = int(flat_token_ids.shape[0])
        if n_compressed != owner_kv.n_compressed:
            raise RuntimeError(
                f"v9-rehome: layer {i} n_compressed={n_compressed} != expected {owner_kv.n_compressed}"
            )
        workspace, compressed_filled = attn_backend.sparse_prefill_kv_buffer.get_with_head(
            n_compressed + n_swa,
            head_source=(extra_k_cache.data_ptr(), flat_token_ids.data_ptr(), n_compressed),
        )
        if not compressed_filled:
            compressed_slice = workspace[:n_compressed]
        swa_slice = workspace[n_compressed:]

    if compressed_slice is not None:
        dequantize_k_cache_paged_dsv4(
            extra_k_cache,
            flat_token_ids,
            page_size=extra_page_size,
            out=compressed_slice,
            layout=kv_pool.get_extra_key_layout(attn.layer_id),
        )
    dequantize_k_cache_paged_dsv4(
        kv_pool.get_swa_key_buffer_radix(attn.layer_id),
        cache.swa_token_ids,
        page_size=cache.swa_page_size,
        out=swa_slice,
        layout=kv_pool.get_swa_key_layout(),
    )

    copies = []
    if n_swa > owner_kv.s_off_min:
        copies.append(("kvs", swa_slice[owner_kv.s_off_min:n_swa], 0))
    if is_src and n_compressed > owner_kv.c_off_min:
        copies.append(("kvc", workspace[owner_kv.c_off_min:n_compressed], 0))
    if i == 0:
        if combined_indices.shape[1] != T.IDX0_W or core_meta.positions.dtype != torch.int32:
            raise RuntimeError(
                f"v9-rehome: L0 idx0 width {combined_indices.shape[1]} or pos dtype {core_meta.positions.dtype}"
            )
        copies.append(("idx0", combined_indices[kept:n], 0))
        copies.append(("ilen", combined_lens[kept:n], 0))
        copies.append(("ipos", core_meta.positions[kept:n], 0))
    elif is_src:
        if combined_indices.shape[1] != T.IDX2_W:
            raise RuntimeError(f"v9-rehome: layer {i} idx2 width {combined_indices.shape[1]} != {T.IDX2_W}")
        copies.append(("idx2", combined_indices[kept:n], 0))
        copies.append(("ilen", combined_lens[kept:n], 0))
    T.publish_kvr(copies, [mv.helper for mv in out_moves])

    return _mega_prefill_woa_wob(
        attn,
        attn_backend,
        q_kept,
        workspace,
        combined_indices[:kept],
        core_meta.positions[:kept],
        combined_lens[:kept],
        defer_all_reduce=True,
    )


def helper_pre(i: int, attn, x_m: torch.Tensor, pair_m, mv) -> torch.Tensor:
    qkv_m = attn.wqkv_a(pair_m if pair_m is not None else x_m)[0]
    kva_m = qkv_m[:, T.QLORA:].contiguous()
    if i in SOURCE_LAYERS:
        qlora_m, q_for_wqb_m = attn._normalize_q_lora(qkv_m[:, :T.QLORA])
        T.push(mv.owner, "x", x_m, row0=mv.lo)
        T.push(mv.owner, "qlora", qlora_m, row0=mv.lo)
        T.push(mv.owner, "kva", kva_m, row0=mv.lo)
        T.signal(mv.owner, T.XQ)
    else:
        T.push(mv.owner, "kva", kva_m, row0=mv.lo)
        T.signal(mv.owner, T.XQ)
        _, q_for_wqb_m = attn._normalize_q_lora(qkv_m[:, :T.QLORA])
    q_m, _ = attn.wq_b(q_for_wqb_m)
    return q_m.view(-1, attn.n_local_heads, attn.head_dim)


def helper_post(i: int, attn, q_m: torch.Tensor, mv, hkv):
    from sglang.srt.model_executor.forward_context import get_attn_backend

    T.wait(mv.owner, T.KVR)
    m = mv.hi - mv.lo
    ws_swa_0 = 0 if i < 2 else hkv.n_compressed
    ws_len = hkv.n_swa if i < 2 else (hkv.n_compressed + hkv.n_swa)
    kv_buf = T.kv_local(ws_len)
    if hkv.s_hi > hkv.s_off:
        kv_buf[ws_swa_0 + hkv.s_off : ws_swa_0 + hkv.s_hi].copy_(
            T.peer(mv.owner, "kvs", hkv.s_hi - hkv.s_off, hkv.s_off - hkv.s_off_min)
        )
    if i in SOURCE_LAYERS and hkv.c_hi > hkv.c_off:
        kv_buf[hkv.c_off : hkv.c_hi].copy_(
            T.peer(mv.owner, "kvc", hkv.c_hi - hkv.c_off, hkv.c_off - hkv.c_off_min)
        )
    if i == 0:
        T.pos_local(m).copy_(T.peer(mv.owner, "ipos", m, hkv.rel0))
        T.idx0_local(m).copy_(T.peer(mv.owner, "idx0", m, hkv.rel0))
        T.len_local(m).copy_(T.peer(mv.owner, "ilen", m, hkv.rel0))
    elif i in SOURCE_LAYERS:
        T.idx2_local(m).copy_(T.peer(mv.owner, "idx2", m, hkv.rel0))
        T.len_local(m).copy_(T.peer(mv.owner, "ilen", m, hkv.rel0))
    idx_buf = T.idx0_local(m) if i < 2 else T.idx2_local(m)
    return _mega_prefill_woa_wob(
        attn,
        get_attn_backend(),
        q_m,
        kv_buf,
        idx_buf,
        T.pos_local(m),
        T.len_local(m),
        defer_all_reduce=True,
    )
