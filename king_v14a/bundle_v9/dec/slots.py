from __future__ import annotations

import copy
import logging
from types import MethodType

logger = logging.getLogger(__name__)

LAYERS = 40


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

    from bundle_v9.dec import moe_front

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
    from bundle_v9.dec import moe_forward

    view = _view(moe)
    view.forward = MethodType(moe_forward.forward, view)
    return view


def prepare_model(module):
    import torch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, DeepseekV4Model

    if type(module) is not DeepseekV4Model:
        raise TypeError(f"bundle-v9 requires DeepseekV4Model, got {type(module).__name__}")
    config = module.config
    facts = (config.model_type, config.num_hidden_layers, config.hidden_size, config.n_routed_experts,
             config.num_experts_per_tok)
    if facts != ("deepseek_v41", LAYERS, 5120, 384, 6):
        raise RuntimeError(f"bundle-v9 requires DeepSeek-V4.1-Flash, got {facts}")
    if len(module.layers) != LAYERS:
        raise RuntimeError(f"bundle-v9 requires all {LAYERS} layers in one stage, got {len(module.layers)}")
    _check_index_reuse_domain(module)
    from bundle_v9.dec.index_reuse import IndexReuse

    view = _view(module)
    view.forward = MethodType(type(module).forward, view)
    view._index_reuse = IndexReuse()
    _mhc_stock_post_fusion()
    _sk_mix_stats_impl()
    from bundle_v9.dec import exact_topk

    exact_topk.prepare(module)
    layers = []
    for index, layer in enumerate(module.layers):
        if type(layer) is not DeepseekV4DecoderLayer or type(layer.mlp) is not DeepseekV2MoE:
            raise TypeError(f"bundle-v9: layer {index} is {type(layer).__name__} / {type(layer.mlp).__name__}")
        gaps = mega_domain_gaps(layer.mlp)
        if gaps:
            raise RuntimeError(f"bundle-v9 MOE_FRONT does not serve layer {index}: missing {gaps}")
        if not hasattr(layer, "attn_boundary"):
            raise TypeError("bundle-v9 INDEX_REUSE requires the current-main CUDA boundary decoder layer")
        layer_view = _view(layer)
        layer_view.mlp = _moe_view(layer.mlp)
        from bundle_v9.dec import row_balance

        if index < row_balance.LAYERS and row_balance.enabled():
            layer_view.self_attn = row_balance.attn_view(layer.self_attn)
        if layer.engram is not None:
            from bundle_v9.dec import col_exchange

            layer_view.engram = col_exchange.engram_view(layer.engram, _view, module)
        layers.append(layer_view)
    view.layers = torch.nn.ModuleList(layers)
    from bundle_v9.dec import attn_boundary

    attn_boundary.prepare(module)
    from bundle_v9.dec import prefill_gemm

    prefill_gemm.prepare(module)
    from bundle_v9.dec import prefill_fuse

    prefill_fuse.prepare(module)
    from bundle_v9.dec import exchange_fuse

    exchange_fuse.prepare(module)
    from bundle_v9.dec import mhc_prefetch

    mhc_prefetch.prepare(module)
    from bundle_v9.dec import decode_gemm

    decode_gemm.prepare(module)
    from bundle_v9.dec import prefill_attn

    prefill_attn.prepare(module)
    from bundle_v9.dec import moe_prefill

    moe_prefill.prepare(module)
    from bundle_v9.dec import moe_decode

    moe_decode.prepare(module)
    from bundle_v9.dec import prenorm_pair

    prenorm_pair.prepare(module)
    from bundle_v9.dec import decode_attn

    decode_attn.prepare(module)
    from bundle_v9.dec import vattn

    vattn.install()  # king-v1e VATTN (dspd-v59 route D); never raises, off on any failure
    from bundle_v9.dec import peer_lookup

    peer_lookup.prepare(view, module)
    from bundle_v9.dec import peer_read

    peer_read.prepare(module)
    from bundle_v9.dec import index_streams

    index_streams.prepare(module)
    from bundle_v9.dec import qnorm_dec

    qnorm_dec.prepare(module)  # king-v2e QNORM-DEC; never raises, stock q-norm on any failure
    from bundle_v9.dec import replay_counts

    replay_counts.prepare()
    from bundle_v9.dec import early_layers

    early_layers.prepare(view, module, _sk_mix_stats_impl())
    from bundle_v9.dec import row_balance

    row_balance.prepare(module)
    from bundle_v9.dec import placement

    placement.defer(module)
    from bundle_v9.dec import rootfork

    rootfork.prepare(view, module)  # king-v13s ROOT-FORK + EKV-EARLY; never raises, king_v10b's path on any failure
    logger.warning("bundle-v9 MOE_FRONT serves all %d MoE layers; PLACEMENT pending", LAYERS)
    return view


def forward_model(view, *args, **kwargs):
    from bundle_v9.dec import placement

    from sglang.srt.models import deepseek_v4_mhc as stock_mhc

    from bundle_v9.dec import mhc_floor

    stock = _mhc_stock_post_fusion()
    sk_stats = _sk_mix_stats_impl()
    stock_stats = _SK[0]
    view._index_reuse.clear()
    stock_mhc._post_fusion = mhc_floor._post_fusion
    stock_mhc._mix_stats_impl = sk_stats
    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring as stock_scoring

    from bundle_v9.dec import exact_topk

    stock_scoring.topk_ragged = exact_topk.topk_ragged
    from bundle_v9.dec import attn_boundary

    batch = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2] if len(args) > 2 else None
    from bundle_v9.dec import rootfork

    try:
        # king-v13s ROOT-FORK / EKV-EARLY: None outside target-verify graph captures; inside the try whose finally
        # restores the mHC / top-k swaps above (fixer, review F9). Host-only: nothing is enqueued before the fork.
        plan = rootfork.entry(batch)
        from bundle_v9.dec import prefill_gemm

        from bundle_v9.dec import replay_counts

        from bundle_v9.dec import prefill_fuse

        from bundle_v9.dec import prefill_attn

        from bundle_v9.dec import row_balance

        from bundle_v9.dec import moe_prefill

        from bundle_v9.dec import moe_decode

        from bundle_v9.dec import decode_attn

        from bundle_v9.dec import index_streams

        from bundle_v9.dec import decode_gemm

        from bundle_v9.dec import vattn

        with attn_boundary.routed(view, batch), decode_gemm.routed(batch), prefill_gemm.routed(view, batch), prefill_fuse.routed(view, batch), prefill_attn.routed(view, batch), row_balance.routed(view, batch), moe_prefill.routed(view, batch), moe_decode.routed(view, batch), replay_counts.routed(batch), index_streams.routed(batch), decode_attn.routed(), vattn.routed(batch), rootfork.routed(plan):
            out = view.forward(*args, **kwargs)
    finally:
        stock_mhc._post_fusion = stock
        stock_mhc._mix_stats_impl = stock_stats
        stock_scoring.topk_ragged = exact_topk.stock_topk_ragged()
        view._index_reuse.clear()
    placement.settle()
    return out


def _prepare_with_reuse(view, stock, x, positions, forward_batch, backend, *args, **kwargs):
    from bundle_v9.dec.index_reuse import backend_view

    return stock(x, positions, forward_batch, backend_view(backend, view._index_reuse), *args, **kwargs)


def _forward_prepare(view, *args, **kwargs):
    return _prepare_with_reuse(view, view._stock_prepare, *args, **kwargs)


def _forward_prepare_multi_stream(view, *args, **kwargs):
    return _prepare_with_reuse(view, view._stock_prepare_multi, *args, **kwargs)


def _forward_prepare_low_ratio_multi_stream(view, *args, **kwargs):
    return _prepare_with_reuse(view, view._stock_prepare_low_ratio, *args, **kwargs)


def _check_index_reuse_domain(module) -> None:
    from sglang.srt.runtime_context import get_exec, get_parallel

    parallel, backend = get_parallel(), get_exec().kernel.dsv4_attn_backend
    if parallel.tp_size != 4 or parallel.attn_tp_size not in (1, 2, 4):
        raise RuntimeError(f"bundle-v9 INDEX_REUSE requires a four-GPU engine, got tp={parallel.tp_size}, "
                           f"attention_tp={parallel.attn_tp_size}")
    if backend not in ("flashmla", "flashmla_mega"):
        raise RuntimeError(f"bundle-v9 INDEX_REUSE requires a PD-compatible attention backend, got {backend}")
    config = module.config
    if (config.num_attention_heads, config.o_groups) != (64, 8) or not module.hc_pre_from_prev_sublayer:
        raise RuntimeError("bundle-v9 INDEX_REUSE requires DeepSeek-V4.1-Flash-1017's current-main boundary layers")


def _attention_view(attention, reuse):
    from sglang.srt.models.deepseek_v4 import MQALayer

    if type(attention) is not MQALayer or not hasattr(attention, "use_mega_attention"):
        raise TypeError("bundle-v9 INDEX_REUSE requires the current-main MQALayer")
    view = _view(attention)
    cls = type(attention)
    view._stock_prepare = MethodType(cls._forward_prepare, view)
    view._stock_prepare_multi = MethodType(cls._forward_prepare_multi_stream, view)
    view._stock_prepare_low_ratio = MethodType(cls._forward_prepare_low_ratio_multi_stream, view)
    view._forward_prepare = MethodType(_forward_prepare, view)
    view._forward_prepare_multi_stream = MethodType(_forward_prepare_multi_stream, view)
    view._forward_prepare_low_ratio_multi_stream = MethodType(_forward_prepare_low_ratio_multi_stream, view)
    view.forward = MethodType(cls.forward, view)
    view._index_reuse = reuse
    return view


_MHC_STOCK = []


def _mhc_stock_post_fusion():
    if not _MHC_STOCK:
        from sglang.srt.models import deepseek_v4_mhc as stock_mhc

        fn = stock_mhc._post_fusion
        if fn.__module__ != "sglang.srt.models.deepseek_v4_mhc" or fn.__qualname__ != "_post_fusion":
            raise RuntimeError(f"bundle-v9 MHC: deepseek_v4_mhc._post_fusion is not stock's ({fn.__module__}."
                               f"{fn.__qualname__})")
        _MHC_STOCK.append(fn)
    return _MHC_STOCK[0]


_SK = []


def _sk_mix_stats_impl():
    if not _SK:
        from sglang.srt.models import deepseek_v4_mhc as stock_mhc

        from bundle_v9.dec import mhc_stats

        fn = stock_mhc._mix_stats_impl
        if fn.__module__ != "sglang.srt.models.deepseek_v4_mhc" or fn.__qualname__ != "_mix_stats_impl":
            raise RuntimeError(f"bundle-v9 SK: deepseek_v4_mhc._mix_stats_impl is not stock's ({fn.__module__}."
                               f"{fn.__qualname__})")
        _SK.extend((fn, mhc_stats.make_mix_stats_impl(fn)))
    return _SK[1]
