
from __future__ import annotations

import copy
import logging
from types import MethodType

logger = logging.getLogger(__name__)

LAYERS = 40
_MHC_STOCK: list = []
_SK: list = []


def _view(module):
    view = copy.copy(module)
    view._parameters = module._parameters.copy()
    view._buffers = module._buffers.copy()
    view._modules = module._modules.copy()
    return view


def mega_domain_gaps(moe) -> list:
    import torch
    from sglang.srt.layers.moe.mega_moe import _mega_moe_mma_type
    from sglang.srt.layers.moe.topk import routing_simulation_enabled
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.models.deepseek_common.utils import _device_sm

    from bundle_v9 import moe_front

    config = moe.topk.topk_config
    checks = (
        ("moe_a2a_backend megamoe", get_moe_a2a_backend().is_megamoe()),
        ("SM 10.x", _device_sm // 10 == 10),
        ("fp8xfp4 MegaMoE experts", _mega_moe_mma_type(moe.experts) == "fp8xfp4"),
        ("not hash-routed", not moe.is_hash),
        ("VL routing bias", moe.gate.e_score_correction_bias_vl is not None),
        ("fp32 routing bias", moe.gate.e_score_correction_bias.dtype == torch.float32),
        ("sqrtsoftplus", config.scoring_func == "sqrtsoftplus"),
        ("no shared top-k slots", config.num_shared_topk_slots == 0),
        (f"top-k {moe_front.TOP_K}", config.top_k == moe_front.TOP_K),
        ("routed scale applied after top-k", not config.apply_routed_scaling_factor_on_output),
        ("no fused shared experts", moe.num_fused_shared_experts == 0),
        ("no routing simulation", not routing_simulation_enabled()),
    )
    return [name for name, ok in checks if not ok]


def _moe_view(moe):
    from bundle_v9 import moe_forward

    view = _view(moe)
    view.forward = MethodType(moe_forward.forward, view)
    return view


def _check_parallel_and_attn_domain(module) -> int:
    from sglang.srt.runtime_context import get_exec, get_parallel

    parallel = get_parallel()
    backend = get_exec().kernel.dsv4_attn_backend
    if parallel.tp_size != 4 or parallel.attn_tp_size not in (1, 2, 4):
        raise RuntimeError(
            f"bundle-v9 requires tp_size=4 and attn_tp_size in (1,2,4), "
            f"got tp={parallel.tp_size}, attn_tp={parallel.attn_tp_size}"
        )
    if backend not in ("flashmla", "flashmla_mega"):
        raise RuntimeError(f"bundle-v9 requires flashmla or flashmla_mega backend, got {backend}")
    config = module.config
    if (config.num_attention_heads, config.o_groups) != (64, 8) or not module.hc_pre_from_prev_sublayer:
        raise RuntimeError("bundle-v9 requires DeepSeek-V4.1-Flash-1017 boundary layers (64 heads, 8 o_groups)")
    return int(parallel.tp_rank)


def _mhc_stock_post_fusion():
    if not _MHC_STOCK:
        from sglang.srt.models import deepseek_v4_mhc as stock_mhc

        fn = stock_mhc._post_fusion
        if fn.__module__ != "sglang.srt.models.deepseek_v4_mhc" or fn.__qualname__ != "_post_fusion":
            raise RuntimeError(
                f"bundle-v9 MHC: deepseek_v4_mhc._post_fusion is not stock ({fn.__module__}.{fn.__qualname__})"
            )
        _MHC_STOCK.append(fn)
    return _MHC_STOCK[0]


def _sk_mix_stats_impl():
    if not _SK:
        from sglang.srt.models import deepseek_v4_mhc as stock_mhc

        from bundle_v9 import mhc_stats

        fn = stock_mhc._mix_stats_impl
        if fn.__module__ != "sglang.srt.models.deepseek_v4_mhc" or fn.__qualname__ != "_mix_stats_impl":
            raise RuntimeError(
                f"bundle-v9 SK: deepseek_v4_mhc._mix_stats_impl is not stock ({fn.__module__}.{fn.__qualname__})"
            )
        _SK.extend((fn, mhc_stats.make_mix_stats_impl(fn)))
    return _SK[1]


def _prepare_p(module):
    import torch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, DeepseekV4Model
    from sglang.srt.runtime_context import get_disagg

    from bundle_v9 import (
        attn_boundary,
        col_exchange,
        dense_gemm,
        early_layers,
        peer_read,
        exact_topk,
        index_hoist,
        moe_front,
        moe_prefill,
        peer_lookup,
        placement,
        prefill_attn,
        prefill_fuse,
        prefill_gemm,
        rehome_block,
        rehome_probe,
        rehome_transport,
        replay_counts,
        exchange_fuse,
        wtile,
    )

    if type(module) is not DeepseekV4Model:
        raise TypeError(f"bundle-v9 requires DeepseekV4Model, got {type(module).__name__}")
    config = module.config
    facts = (
        config.model_type,
        config.num_hidden_layers,
        config.hidden_size,
        config.n_routed_experts,
        config.num_experts_per_tok,
    )
    if facts != ("deepseek_v41", LAYERS, 5120, 384, 6):
        raise RuntimeError(f"bundle-v9 requires DeepSeek-V4.1-Flash-1017, got {facts}")
    if len(module.layers) != LAYERS:
        raise RuntimeError(f"bundle-v9 requires all {LAYERS} layers in one stage, got {len(module.layers)}")

    tp_rank = _check_parallel_and_attn_domain(module)
    role = str(get_disagg().disaggregation_mode)
    for index, layer in enumerate(module.layers):
        if type(layer) is not DeepseekV4DecoderLayer or type(layer.mlp) is not DeepseekV2MoE:
            raise TypeError(f"bundle-v9: layer {index} is {type(layer).__name__} / {type(layer.mlp).__name__}")
        gaps = mega_domain_gaps(layer.mlp)
        if gaps:
            raise RuntimeError(f"bundle-v9 fused MoE front does not serve layer {index}: missing {gaps}")
        if not hasattr(layer, "attn_boundary"):
            raise TypeError("bundle-v9 requires boundary decoder layers")

    view = _view(module)
    view.forward = MethodType(type(module).forward, view)
    _mhc_stock_post_fusion()
    sk_stats = _sk_mix_stats_impl()
    device = module.layers[0].self_attn.wq_b.weight.device

    frontfix_checks = moe_front.probe_frontfix(module, device)
    mmhc_checks = attn_boundary.prepare(module, device)
    peer_on = peer_lookup.prepare(module, role)
    dense_active, dense_checks = dense_gemm.prepare(module, role)
    ihx_layers = index_hoist.prepare(module, role)
    exact_topk.prepare(module)
    wtile.prepare(module, role)  # king-v6 WTILE; never raises, stock tiles on any failure

    layers = []
    for layer in module.layers:
        layer_view = _view(layer)
        layer_view.mlp = _moe_view(layer.mlp)
        if layer.self_attn.layer_id in ihx_layers:
            layer_view.self_attn = index_hoist.attn_view(layer.self_attn, _view)
        if layer.engram is not None:
            if peer_on and role == "decode":
                layer_view.engram = peer_lookup.engram_view(layer.engram, _view)
            else:
                layer_view.engram = col_exchange.engram_view(layer.engram, _view, module)
        layers.append(layer_view)

    view.layers = torch.nn.ModuleList(layers)
    prefill_gemm.prepare(module)
    prefill_fuse.prepare(module)
    exchange_fuse.prepare(module)
    prefill_attn.prepare(module)
    moe_prefill.prepare(module)
    peer_read.prepare(module)
    replay_counts.prepare()
    early_layers.prepare(view, module, sk_stats)

    rehome_on = False
    if rehome_transport.prepare(module):
        rehome_on = rehome_block.prepare(view, module)
        if rehome_on:
            rehome_probe.run(module, sk_stats)
    peer_lookup.prepare_rh(view, module, role, rehome_on)
    placement._pending.clear()
    placement.defer(module)
    view._v4_rehome_on = rehome_on
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if tp_rank == 0:
        logger.warning(
            "[b7] bundle-v9 prepared: role=%s, layers=%d, frontfix=%d, mmhc=%d, mhc_sched=%s, peer_lookup=%s, dense=%d(%d probes), "
            "ihx=%d, prefill=pgemm+pfuse+xfuse+pattn+moe+topk+counts+lean, rehome=%s, place=deferred",
            role,
            LAYERS,
            frontfix_checks,
            mmhc_checks,
            "cake+dg" if attn_boundary.mhc_sched._DG else "cake",
            "on" if peer_on else "cols",
            dense_active,
            dense_checks,
            len(ihx_layers),
            "on" if rehome_on else "off(decode)",
        )
    return view


def _forward_p(view, *args, **kwargs):
    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring as stock_scoring
    from sglang.srt.models import deepseek_v4_mhc as stock_mhc

    from bundle_v9 import (
        attn_boundary,
        dense_gemm,
        exact_topk,
        mhc_floor,
        moe_prefill,
        placement,
        prefill_attn,
        prefill_fuse,
        prefill_gemm,
        replay_counts,
        wtile,
    )

    stock_post = _mhc_stock_post_fusion()
    sk_stats = _sk_mix_stats_impl()
    stock_stats = _SK[0]
    batch = kwargs["forward_batch"] if "forward_batch" in kwargs else (args[2] if len(args) > 2 else None)
    stock_mhc._post_fusion = mhc_floor._post_fusion
    stock_mhc._mix_stats_impl = sk_stats
    stock_scoring.topk_ragged = exact_topk.topk_ragged
    try:
        with (
            attn_boundary.routed(view, batch),
            dense_gemm.routed(batch),
            prefill_gemm.routed(view, batch),
            prefill_fuse.routed(view, batch),
            prefill_attn.routed(view, batch),
            wtile.routed(view, batch),
            moe_prefill.routed(view, batch),
            replay_counts.routed(batch),
        ):
            out = view.forward(*args, **kwargs)
    finally:
        stock_mhc._post_fusion = stock_post
        stock_mhc._mix_stats_impl = stock_stats
        stock_scoring.topk_ragged = exact_topk.stock_topk_ragged()
    if placement._pending:
        placement.settle()
    return out


_ROLE = []


def prepare_model(module):
    from sglang.srt.runtime_context import get_disagg

    if str(get_disagg().disaggregation_mode) == "decode":
        from bundle_v9.dec import slots as dec

        view = dec.prepare_model(module)
        _ROLE[:] = [dec.forward_model]
        return view
    view = _prepare_p(module)
    _ROLE[:] = [_forward_p]
    return view


def forward_model(view, *args, **kwargs):
    return _ROLE[0](view, *args, **kwargs)
