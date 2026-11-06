
from __future__ import annotations

import logging
import time
import traceback

import torch
import torch.distributed as dist

from bundle_v9 import rehome_plan, rehome_transport as T, moe_front, dense_gemm, attn_boundary
from bundle_v9.placement_table import SLOTS

logger = logging.getLogger(__name__)

ROUTER_LO = rehome_plan.MIN_BLOCK
ROUTER_HI = rehome_plan.CAP
BOUNDARY_MS = (
    4096, 4097, 4457, 5120, 6144, 7168, 8192, 9728, 9856,
    12288, 14720, 14736, 16384, 18048, 22911, 22912, 22943, 22944, 32767, 32768,
)


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(f"v9-rehome STRICT probe failed: {msg}")


def _seeded_bf16(rows: int, cols: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    out = torch.empty((rows, cols), dtype=torch.bfloat16, device=device)
    for s in range(0, rows, 8192):
        m = min(8192, rows - s)
        chunk = torch.randn((m, cols), generator=gen, device=device, dtype=torch.float32)
        scale = 0.25 + 1.75 * torch.rand((m, 1), generator=gen, device=device, dtype=torch.float32)
        out[s : s + m] = (chunk * scale).to(torch.bfloat16)
    return out


def _shared_rows_idx(total: int, block_m: int, device: torch.device) -> torch.Tensor:
    r = torch.arange(total, device=device, dtype=torch.int64)
    m = r % block_m
    page = ((block_m + 127) // 128) * 128
    return (r // block_m) * page + (m // 128) * 128 + (m % 32) * 4 + (m % 128) // 32


def _probe_moe_front_split(module, device: torch.device) -> int:
    from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

    moe = module.layers[0].mlp
    cfg = moe.topk.topk_config
    scale = 1.0 if moe.routed_scaling_factor_fused_in_topk else float(moe.routed_scaling_factor)
    renorm = bool(cfg.renormalize and cfg.top_k > 1)
    img_id = int(moe.config.image_token_id)
    padded = 4096
    hidden, experts, top_k = moe_front.HIDDEN, moe_front.EXPERTS, moe_front.TOP_K
    sf_cols = hidden // 128
    x_full = _seeded_bf16(padded, hidden, 20261101, device)
    gen = torch.Generator(device=device).manual_seed(20261102)
    logits_full = torch.randn((padded, experts), generator=gen, device=device, dtype=torch.float32)
    ids_full = torch.randint(0, 65536, (padded,), generator=gen, device=device, dtype=torch.int64)
    ids_full[::37] = img_id
    bias = moe.gate.e_score_correction_bias
    bias_alt = moe.gate.e_score_correction_bias_vl

    checks = 0
    for block_m in (192, 240):
        s_rows = ((padded + block_m - 1) // block_m) * (((block_m + 127) // 128) * 128)

        def _alloc(fill_byte: int):
            bx = torch.full((padded, hidden), fill_byte, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
            bxsf = torch.full((padded, sf_cols), fill_byte, dtype=torch.int32, device=device)
            bidx = torch.full((padded, top_k), -99, dtype=torch.int64, device=device)
            bw = torch.full((padded, top_k), -9.0, dtype=torch.float32, device=device)
            ssf_storage = torch.full((sf_cols, s_rows), fill_byte, dtype=torch.int32, device=device)
            return bx, bxsf, bidx, bw, ssf_storage.t()

        for own, m in ((0, 1024), (513, 1024), (513, 1025), (1, 2047), (1024, 2048)):
            tot = own + m
            rx, rxsf, ridx, rw, rssf = _alloc(0x5A)
            moe_front.moe_front(
                logits_full[:tot], bias, x_full[:tot], rx, rxsf, ridx, rw,
                bias_alt=bias_alt, input_ids=ids_full[:tot], image_token_id=img_id,
                num_token_non_padded=None, renormalize=renorm, renormalize_epsilon=_RENORMALIZE_SUM_EPSILON,
                routed_scaling_factor=scale, shared_x_sf=rssf, shared_block_m=block_m,
                use_pdl=False, row_offset=0, pad=True,
            )
            tx, txsf, tidx, tw, tssf = _alloc(0xA5)
            bound_own = torch.tensor([own], dtype=torch.int32, device=device)
            if own > 0:
                moe_front.moe_front(
                    logits_full[:own], bias, x_full[:own], tx, txsf, tidx, tw,
                    bias_alt=bias_alt, input_ids=ids_full[:own], image_token_id=img_id,
                    num_token_non_padded=bound_own, renormalize=renorm, renormalize_epsilon=_RENORMALIZE_SUM_EPSILON,
                    routed_scaling_factor=scale, shared_x_sf=tssf, shared_block_m=block_m,
                    use_pdl=False, row_offset=0, pad=False,
                )
            moe_front.moe_front(
                logits_full[own:tot], bias, x_full[own:tot], tx, txsf, tidx, tw,
                bias_alt=bias_alt, input_ids=ids_full[own:tot], image_token_id=img_id,
                num_token_non_padded=None, renormalize=renorm, renormalize_epsilon=_RENORMALIZE_SUM_EPSILON,
                routed_scaling_factor=scale, shared_x_sf=tssf, shared_block_m=block_m,
                use_pdl=False, row_offset=own, pad=True,
            )
            sidx = _shared_rows_idx(tot, block_m, device)
            tag = f"MOE_FRONT block_m={block_m} own={own} m={m}"
            _require(_bits_equal(rx[:tot], tx[:tot]), f"{tag} buf.x")
            _require(_bits_equal(rxsf[:tot], txsf[:tot]), f"{tag} buf.x_sf")
            _require(_bits_equal(ridx, tidx), f"{tag} buf.topk_idx")
            _require(_bits_equal(rw, tw), f"{tag} buf.topk_weights")
            _require(_bits_equal(rssf.index_select(0, sidx), tssf.index_select(0, sidx)), f"{tag} shared_x_sf")
            checks += 1
    return checks


def _probe_seams_and_projections(module, device: torch.device, sk_stats) -> int:
    from sglang.kernels.ops.attention import flash_mla_interface
    from sglang.kernels.ops.attention.dsv4 import fused_k_norm_rope_flashmla
    from sglang.srt.layers.engram import engram_gate
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import select_dsv4_kv_layout
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.models.deepseek_v4 import _fp8_wo_a_einsum

    checks = 0
    kv_lay, _ = select_dsv4_kv_layout()
    layer = module.layers[2]
    hc = layer.ffn_hc
    norm = layer.input_layernorm
    attn = layer.self_attn
    wqkv_a = attn.wqkv_a
    wq_b = attn.wq_b
    wo_b = attn.wo_b

    for n, m in ((8192, 4096), (32767, 9856), (32768, 9856)):
        kept = n - m
        x_in = _seeded_bf16(n, 5120, 3100 + n, device)
        res = _seeded_bf16(n, 4 * 5120, 3200 + n, device).view(n, 4, 5120)
        gen = torch.Generator(device=device).manual_seed(3300 + n)
        pre = torch.rand((n, 4), generator=gen, device=device, dtype=torch.float32)
        post = torch.rand((n, 4), generator=gen, device=device, dtype=torch.float32)
        comb = torch.full((n, 4, 4), 0.25, device=device, dtype=torch.float32)

        rfp8, rsf = mhc.empty_deepgemm_mxfp8_pair(n, 5120, device)
        rupd, rnorm, (rpre, rpost, rcomb) = attn_boundary.exact_boundary(
            x_in, res, pre, post, comb, layer.attn_hc.fn, layer.attn_hc.scale, layer.attn_hc.base,
            norm.weight, hc.cfg.rms_eps, hc.cfg.eps, norm.variance_epsilon, hc.cfg.sinkhorn_iters,
            y_fp8=rfp8, y_gemm_sf=rsf,
        )
        rqkv = wqkv_a((rfp8, rsf))[0]
        rqlora, rq_for_wqb = attn._normalize_q_lora(rqkv[:, : T.QLORA])
        rq = wq_b(rq_for_wqb)[0]

        x_buf = T.view("x", n)
        qlora_buf = T.view("qlora", n)
        kva_buf = T.view("kva", n)
        kfp8, ksf = mhc.empty_deepgemm_mxfp8_pair(kept, 5120, device)
        attn_boundary._REHOME_Y_BF16[:] = [x_buf[:kept]]
        try:
            kupd, knorm, (kpre, kpost, kcomb) = attn_boundary.exact_boundary(
                x_in[:kept], res[:kept], pre[:kept], post[:kept], comb[:kept],
                layer.attn_hc.fn, layer.attn_hc.scale, layer.attn_hc.base,
                norm.weight, hc.cfg.rms_eps, hc.cfg.eps, norm.variance_epsilon, hc.cfg.sinkhorn_iters,
                y_fp8=kfp8, y_gemm_sf=ksf,
            )
        finally:
            left_y = len(attn_boundary._REHOME_Y_BF16)
            attn_boundary._REHOME_Y_BF16.clear()
        _require(left_y == 0 and knorm.data_ptr() == x_buf[:kept].data_ptr(), f"mmhc in-place not consumed n={n}")
        kqkv = wqkv_a((kfp8, ksf))[0]
        kva_buf[:kept].copy_(kqkv[:, T.QLORA:])
        kqlora, kq_for_wqb = attn._normalize_q_lora(kqkv[:, : T.QLORA])
        qlora_buf[:kept].copy_(kqlora)
        kq = wq_b(kq_for_wqb)[0]

        mfp8, msf = mhc.empty_deepgemm_mxfp8_pair(m, 5120, device)
        mupd, mnorm, (mpre, mpost, mcomb) = attn_boundary.exact_boundary(
            x_in[kept:], res[kept:], pre[kept:], post[kept:], comb[kept:],
            layer.attn_hc.fn, layer.attn_hc.scale, layer.attn_hc.base,
            norm.weight, hc.cfg.rms_eps, hc.cfg.eps, norm.variance_epsilon, hc.cfg.sinkhorn_iters,
            y_fp8=mfp8, y_gemm_sf=msf,
        )
        mqkv = wqkv_a((mfp8, msf))[0]
        mqlora, mq_for_wqb = attn._normalize_q_lora(mqkv[:, : T.QLORA])
        mq = wq_b(mq_for_wqb)[0]
        x_buf[kept:].copy_(mnorm)
        qlora_buf[kept:].copy_(mqlora)
        kva_buf[kept:].copy_(mqkv[:, T.QLORA:].contiguous())

        tag = f"mmhc+qkv+wqb n={n} kept={kept} m={m}"
        _require(_bits_equal(rupd, torch.cat([kupd, mupd], dim=0)), f"{tag} updated")
        _require(_bits_equal(rnorm, x_buf), f"{tag} y_bf16")
        _require(_bits_equal(rqlora, qlora_buf), f"{tag} qlora")
        _require(_bits_equal(rqkv[:, T.QLORA:], kva_buf), f"{tag} kva")
        _require(_bits_equal(rq, torch.cat([kq, mq], dim=0)), f"{tag} wq_b")
        _require(_bits_equal(rpre, torch.cat([kpre, mpre], dim=0)), f"{tag} next_pre")
        _require(_bits_equal(rpost, torch.cat([kpost, mpost], dim=0)), f"{tag} next_post")
        _require(_bits_equal(rcomb, torch.cat([kcomb, mcomb], dim=0)), f"{tag} next_comb")

        if n == 8192:
            pos_s = torch.arange(512, dtype=torch.int64, device=device)
            loc_s = torch.arange(512, dtype=torch.int32, device=device)
            page_bytes = kv_lay.page_bytes(256)
            c_ref = torch.zeros((2, page_bytes), dtype=torch.uint8, device=device)
            c_tst = torch.zeros((2, page_bytes), dtype=torch.uint8, device=device)
            fused_k_norm_rope_flashmla(
                kv=rqkv[:512, T.QLORA:], kv_weight=attn.kv_norm.weight.data, eps=attn.eps,
                freqs_cis=attn.freqs_cis, positions=pos_s, out_loc=loc_s, kvcache=c_ref,
                page_size=256, layout=kv_lay, q=None,
            )
            fused_k_norm_rope_flashmla(
                kv=kva_buf[:512], kv_weight=attn.kv_norm.weight.data, eps=attn.eps,
                freqs_cis=attn.freqs_cis, positions=pos_s, out_loc=loc_s, kvcache=c_tst,
                page_size=256, layout=kv_lay, q=None,
            )
            _require(_bits_equal(c_ref, c_tst), "fused_k_norm_rope_flashmla contiguous kva vs strided slice")
        del x_in, res, pre, post, comb, rfp8, rsf, rupd, rnorm, rqkv, rqlora, rq, kfp8, ksf, kupd, kqkv, kq, mfp8, msf, mupd, mnorm, mqkv, mq
        checks += 1

    groups, orank = attn.n_local_groups, attn.o_lora_rank
    kernel_heads = 64 if attn.n_local_heads <= 64 else 128
    attn_sink = attn._local_attn_sink(kernel_heads)
    use_legacy = not kv_lay.is_interleaved
    for n, m in ((32767, 9856), (32768, 9856)):
        kept = n - m
        q_all = _seeded_bf16(n * kernel_heads, T.KV_DIM, 4100 + n, device).view(n, kernel_heads, T.KV_DIM)
        kv_ws = _seeded_bf16(1024, T.KV_DIM, 4200 + n, device).view(1024, 1, T.KV_DIM)
        kv_sub = torch.full_like(kv_ws, -7.25)
        kv_sub[384:1024].copy_(kv_ws[384:1024])
        idx_all = torch.full((n, T.IDX0_W), -1, dtype=torch.int32, device=device)
        col_96 = torch.arange(96, device=device, dtype=torch.int32).unsqueeze(0)
        idx_all[:kept, :96] = (torch.arange(kept, device=device, dtype=torch.int32).unsqueeze(1) * 7 + col_96) % 384
        idx_all[kept:, :96] = 384 + (torch.arange(m, device=device, dtype=torch.int32).unsqueeze(1) * 11 + col_96) % 640
        len_all = 1 + (torch.arange(n, device=device, dtype=torch.int32) * 13) % 96
        pos_all = torch.arange(n, device=device, dtype=torch.int32) % 512

        r_data, r_scale, _, _ = flash_mla_interface.flash_mla_mega_prefill(
            q_all, kv_ws, idx_all.unsqueeze(1), pos_all, attn.mega_cos_sin_cache, attn.softmax_scale,
            attn_sink=attn_sink, topk_length=len_all, use_legacy=use_legacy,
        )
        k_data, k_scale, _, _ = flash_mla_interface.flash_mla_mega_prefill(
            q_all[:kept], kv_ws, idx_all[:kept].unsqueeze(1), pos_all[:kept], attn.mega_cos_sin_cache,
            attn.softmax_scale, attn_sink=attn_sink, topk_length=len_all[:kept], use_legacy=use_legacy,
        )
        m_data, m_scale, _, _ = flash_mla_interface.flash_mla_mega_prefill(
            q_all[kept:], kv_sub, idx_all[kept:].unsqueeze(1), pos_all[kept:], attn.mega_cos_sin_cache,
            attn.softmax_scale, attn_sink=attn_sink, topk_length=len_all[kept:], use_legacy=use_legacy,
        )
        o_fp8, o_s = r_data[:, :groups], r_scale[:, :groups]
        ko_fp8, ko_s = k_data[:, :groups], k_scale[:, :groups]
        mo_fp8, mo_s = m_data[:, :groups], m_scale[:, :groups]
        _require(_bits_equal(o_fp8, torch.cat([ko_fp8, mo_fp8], dim=0)), f"mega_prefill fp8 split n={n} kept={kept}")
        _require(_bits_equal(o_s, torch.cat([ko_s, mo_s], dim=0)), f"mega_prefill sf split n={n} kept={kept}")

        ry, rsf = mhc.empty_deepgemm_mxfp8_pair(n, groups * orank, device)
        _fp8_wo_a_einsum(o_fp8, o_s, *attn.mega_wo_a, recipe=(1, 1, 32), out=(ry.view(n, groups, orank), rsf))
        ref_wo = wo_b((ry, rsf), skip_all_reduce=False)[0]

        ky, ksf = mhc.empty_deepgemm_mxfp8_pair(kept, groups * orank, device)
        _fp8_wo_a_einsum(ko_fp8, ko_s, *attn.mega_wo_a, recipe=(1, 1, 32), out=(ky.view(kept, groups, orank), ksf))
        kept_wo = wo_b((ky, ksf), skip_all_reduce=False)[0]

        my, msf = mhc.empty_deepgemm_mxfp8_pair(m, groups * orank, device)
        _fp8_wo_a_einsum(mo_fp8, mo_s, *attn.mega_wo_a, recipe=(1, 1, 32), out=(my.view(m, groups, orank), msf))
        moved_wo = wo_b((my, msf), skip_all_reduce=False)[0]

        _require(_bits_equal(ry, torch.cat([ky, my], dim=0)), f"wo_a fp8 split n={n} kept={kept} m={m}")
        _require(_bits_equal(rsf, torch.cat([ksf, msf], dim=0)), f"wo_a sf split n={n} kept={kept} m={m}")
        _require(_bits_equal(ref_wo, torch.cat([kept_wo, moved_wo], dim=0)), f"wo_a+wo_b split n={n} kept={kept} m={m}")
        del (
            q_all, kv_ws, kv_sub, idx_all, len_all, pos_all, r_data, r_scale, k_data, k_scale, m_data, m_scale,
            o_fp8, o_s, ko_fp8, ko_s, mo_fp8, mo_s, ry, rsf, ref_wo, ky, ksf, kept_wo, my, msf, moved_wo,
        )
        checks += 1

    stock_stats = mhc._mix_stats_impl
    mhc._mix_stats_impl = sk_stats
    try:
        for lid in (0, 1, 14, 20):
            ahc = module.layers[lid].attn_hc
            res = _seeded_bf16(16384, 4 * 5120, 5100 + lid, device).view(16384, 4, 5120)
            pre_in = None
            if lid > 0:
                gen = torch.Generator(device=device).manual_seed(5200 + lid)
                pre_in = torch.rand((16384, 4), generator=gen, device=device, dtype=torch.float32)
            st_full = mhc.HcState(res.clone(), pre_in)
            ref_sx = mhc.combine(ahc, st_full)
            st_k = mhc.HcState(res[:12288].clone(), None if pre_in is None else pre_in[:12288].clone())
            st_m = mhc.HcState(res[12288:].clone(), None if pre_in is None else pre_in[12288:].clone())
            sx_k = mhc.combine(ahc, st_k)
            sx_m = mhc.combine(ahc, st_m)
            _require(_bits_equal(ref_sx, torch.cat([sx_k, sx_m], dim=0)), f"mhc.combine L{lid}")
            stats_full = mhc.mix_stats(ahc, res)
            stats_k = mhc.mix_stats(ahc, res[:12288])
            stats_m = mhc.mix_stats(ahc, res[12288:])
            for idx in range(3):
                _require(
                    _bits_equal(stats_full[idx], torch.cat([stats_k[idx], stats_m[idx]], dim=0)),
                    f"mhc.mix_stats[{idx}] L{lid}",
                )
            del res, pre_in, st_full, ref_sx, st_k, st_m, sx_k, sx_m, stats_full, stats_k, stats_m
            checks += 1
    finally:
        mhc._mix_stats_impl = stock_stats

    engram = module.layers[1].engram
    n, k = 8192, 4096
    x_eng = _seeded_bf16(n, 4 * 5120, 6101, device).view(n, 4, 5120)
    emb_in = _seeded_bf16(n, engram.wkv.input_size, 6102, device)
    ids = torch.arange(n, device=device, dtype=torch.int64)
    if engram.image_token_id is not None:
        ids[::29] = int(engram.image_token_id)
    img_arg = ids if engram.image_token_id is not None else None
    kv_full, _ = engram.wkv(emb_in)
    ref_eg = engram_gate(
        x_eng, kv_full, engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value,
        img_arg, engram.image_token_id,
    )
    eg_0 = engram_gate(
        x_eng[:k], kv_full[:k], engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value,
        img_arg[:k] if img_arg is not None else None, engram.image_token_id,
    )
    eg_1 = engram_gate(
        x_eng[k:], kv_full[k:], engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value,
        img_arg[k:] if img_arg is not None else None, engram.image_token_id,
    )
    kv_0, _ = engram.wkv(emb_in[:k])
    kv_1, _ = engram.wkv(emb_in[k:])
    _require(_bits_equal(kv_full, torch.cat([kv_0, kv_1], dim=0)), "engram.wkv split")
    _require(_bits_equal(ref_eg, torch.cat([eg_0, eg_1], dim=0)), "engram_gate split")
    del x_eng, emb_in, ids, kv_full, ref_eg, eg_0, eg_1, kv_0, kv_1
    checks += 1
    return checks


def _placed_gate_weight(moe) -> torch.Tensor:
    perm = SLOTS.get(moe.layer_id)
    w = moe.gate.weight
    if perm is None:
        return w
    idx = torch.tensor(perm, dtype=torch.int64, device=w.device)
    return w.index_select(0, idx).contiguous()


def _probe_router_exhaustive(module, device: torch.device, rank: int) -> int:
    from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

    x = _seeded_bf16(ROUTER_HI, 5120, 20261029, device)
    shard = ((0, False), (6, True), (13, False), (19, True))[rank]
    lid_full, use_placed = shard
    moe_full = module.layers[lid_full].mlp
    w_full = _placed_gate_weight(moe_full) if use_placed else moe_full.gate.weight
    ref = linear_bf16_fp32(x, w_full)
    ref_i32 = ref.view(torch.int32)
    ms = list(range(ROUTER_LO, ROUTER_HI + 1))
    bad_pre = torch.zeros(len(ms), dtype=torch.bool, device=device)
    bad_suf = torch.zeros(len(ms), dtype=torch.bool, device=device)
    for i, m in enumerate(ms):
        bad_pre[i] = (linear_bf16_fp32(x[:m], w_full).view(torch.int32) != ref_i32[:m]).any()
        if m % 64 == 0:
            bad_suf[i] = (
                linear_bf16_fp32(x[ROUTER_HI - m :], w_full).view(torch.int32) != ref_i32[ROUTER_HI - m :]
            ).any()
    torch.cuda.synchronize(device)
    pre_bad = [ms[i] for i in torch.nonzero(bad_pre).flatten().tolist()]
    suf_bad = [ms[i] for i in torch.nonzero(bad_suf).flatten().tolist()]
    _require(
        not pre_bad and not suf_bad,
        f"router GEMM L{lid_full} (placed={use_placed}) mismatch pre={pre_bad[:8]} suf={suf_bad[:8]}",
    )
    checks = len(ms) + sum(1 for m in ms if m % 64 == 0)

    for lid in range(rank * 5, (rank + 1) * 5):
        moe = module.layers[lid].mlp
        for placed, w in ((False, moe.gate.weight), (True, _placed_gate_weight(moe))):
            r_all = linear_bf16_fp32(x, w).view(torch.int32)
            for m in BOUNDARY_MS:
                _require(
                    bool(torch.equal(linear_bf16_fp32(x[:m], w).view(torch.int32), r_all[:m])),
                    f"router boundary L{lid} placed={placed} prefix M={m}",
                )
                _require(
                    bool(torch.equal(linear_bf16_fp32(x[ROUTER_HI - m :], w).view(torch.int32), r_all[ROUTER_HI - m :])),
                    f"router boundary L{lid} placed={placed} suffix M={m}",
                )
                checks += 2
    del x, ref, ref_i32
    return checks


@torch.no_grad()
def run(module, sk_stats) -> None:
    lk = T.link()
    device, rank, group = lk.device, lk.rank, lk.group
    t0 = time.perf_counter()
    ok, err = 1, ""
    mf_checks = seam_checks = router_checks = 0
    try:
        with torch.cuda.device(device):
            mf_checks = _probe_moe_front_split(module, device)
            with dense_gemm.probe_routes():
                seam_checks = _probe_seams_and_projections(module, device, sk_stats)
            router_checks = _probe_router_exhaustive(module, device, rank)
            lk.buffer.zero_()
            torch.cuda.synchronize(device)
    except BaseException as exc:
        tail = " <- ".join(line.strip() for line in traceback.format_exc().strip().splitlines()[-3:])
        ok, err = 0, f"rank {rank}: {type(exc).__name__}: {exc} ({tail})"
    gathered = [None] * group.world_size
    dist.all_gather_object(gathered, (ok, err), group=group.cpu_group)
    failures = [msg for flag, msg in gathered if not flag]
    if failures:
        raise RuntimeError("bundle-v9 STRICT prepare probe failed: " + " | ".join(failures))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    group.barrier()
    elapsed = time.perf_counter() - t0
    if rank == 0:
        logger.warning(
            "[b7] bundle-v9 STRICT probes ready in %.2fs (role=%s, moe_front=%d, seams=%d, router=%d/rank)",
            elapsed, lk.role, mf_checks, seam_checks, router_checks,
        )
