"""dsv41-pd001 model slot: private views of the upstream-main DeepSeek-V4.1 model whose MoE layers run MEGA_FRONT,
with dsv41-pd002 PLACEMENT (placement.py: experts and router rows relabeled across the EP ranks, bitwise stock).

Views share stock's parameters and buffers but own their child registries, so binding the candidate never edits the
stock model the audit and the reference use. Only each MoE view's `forward` is pd001's (moe_forward.py): it reaches
dsv41_pd001.mega_forward, where one launch replaces stock's router, routed-scale multiply and MegaMoE pre-dispatch."""

from __future__ import annotations

import copy
import logging
from types import MethodType

logger = logging.getLogger(__name__)

LAYERS = 40


def _view(module):
    """A shallow copy of a module with its own parameter, buffer and child registries."""
    view = copy.copy(module)
    view._parameters = module._parameters.copy()
    view._buffers = module._buffers.copy()
    view._modules = module._modules.copy()
    return view


def mega_domain_gaps(moe) -> list:
    """The static facts that keep an MoE layer out of the MEGA_FRONT domain (empty = served; rows and tensor shapes
    are checked per call by mega_front.covered)."""
    import torch
    from sglang.srt.layers.moe.mega_moe import _mega_moe_mma_type
    from sglang.srt.layers.moe.topk import routing_simulation_enabled
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.models.deepseek_common.utils import _device_sm

    from dsv41_pd001 import mega_front

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
        (f"top-k {mega_front.TOP_K}", config.top_k == mega_front.TOP_K),
        ("routed scale applied after top-k", not config.apply_routed_scaling_factor_on_output),
        ("no fused shared experts", moe.num_fused_shared_experts == 0),
        ("no routing simulation", not routing_simulation_enabled()),
    )
    return [name for name, ok in checks if not ok]


def _moe_view(moe):
    from dsv41_pd001 import moe_forward

    view = _view(moe)
    view.forward = MethodType(moe_forward.forward, view)
    return view


def prepare_model(module):
    """Build the model view; raise unless every one of the 40 MoE layers is in the MEGA_FRONT domain."""
    import torch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, DeepseekV4Model

    if type(module) is not DeepseekV4Model:
        raise TypeError(f"dsv41-pd001 requires DeepseekV4Model, got {type(module).__name__}")
    config = module.config
    facts = (config.model_type, config.num_hidden_layers, config.hidden_size, config.n_routed_experts,
             config.num_experts_per_tok)
    if facts != ("deepseek_v41", LAYERS, 5120, 384, 6):
        raise RuntimeError(f"dsv41-pd001 requires DeepSeek-V4.1-Flash, got {facts}")
    if len(module.layers) != LAYERS:
        raise RuntimeError(f"dsv41-pd001 requires all {LAYERS} layers in one stage, got {len(module.layers)}")
    _check_index_reuse_domain(module)  # dsv41-pd003 INDEX_REUSE
    from dsv41pd_c02.index_reuse import IndexReuse

    view = _view(module)
    view.forward = MethodType(type(module).forward, view)
    view._index_reuse = IndexReuse()
    _mhc_stock_post_fusion()  # dsv41-pd004 MHC: the genuine stock function, captured before any swap
    _sk_mix_stats_impl()  # dsv41-pd006 SK: stock's _mix_stats_impl captured likewise, the drop-in bound to it
    from dsv41_pd001 import topk_jit

    topk_jit.prepare(module)  # dsv41-pd033 TOPK: stock's topk_ragged and DeepSelect dispatcher captured, kernel built
    layers = []
    for index, layer in enumerate(module.layers):
        if type(layer) is not DeepseekV4DecoderLayer or type(layer.mlp) is not DeepseekV2MoE:
            raise TypeError(f"dsv41-pd001: layer {index} is {type(layer).__name__} / {type(layer.mlp).__name__}")
        gaps = mega_domain_gaps(layer.mlp)
        if gaps:
            raise RuntimeError(f"dsv41-pd001 MEGA_FRONT does not serve layer {index}: missing {gaps}")
        if not hasattr(layer, "attn_boundary"):
            raise TypeError("dsv41-pd003 INDEX_REUSE requires the current-main CUDA boundary decoder layer")
        layer_view = _view(layer)
        layer_view.mlp = _moe_view(layer.mlp)
        # dsv41-pd027: INDEX_REUSE off; the layer view keeps stock's attention module
        if layer.engram is not None:  # dsv41-pd034 ENGRAM_COLS (supersedes pd010 ENGRAM_A2A): Engram layers 1 and 14
            from dsv41_pd001 import engram_cols

            layer_view.engram = engram_cols.engram_view(layer.engram, _view, module)
        layers.append(layer_view)
    view.layers = torch.nn.ModuleList(layers)
    from dsv41_pd001 import v16_levers

    v16_levers.prepare(module)  # dsv41-pd022: dsv41pd.v16's mega mHC programs and pinned wq_b tactics, built now
    from dsv41_pd001 import prefill_gemm

    prefill_gemm.prepare(module)  # dsv41-pd046 PGEMM v3: wo_b registered, CUTLASS prefill tactics warmed
    from dsv41_pd001 import skfuse

    skfuse.prepare(module)  # dsv41-pd052 SKFUSE: stock's four seams captured, shapes checked, kernels built
    from dsv41_pd001 import skfuse2

    skfuse2.prepare(module)  # dsv41-pd054 SKFUSE2: Engram exchange facts checked, kernels built
    from dsv41_pd001 import fmlap

    fmlap.prepare(module)  # dsv41-pd055 FMLAP: stock's flash_mla_mega_prefill captured, kernel built
    from dsv41_pd001 import m462

    m462.prepare(module)  # dsv41-pd065 M462: the image MegaMoE captured, #462 host built, kernel compiled
    from dsv41_pd001 import edec

    edec.prepare(view, module)  # dsv41-pd071 EDEC: peers' shards mapped, bitwise self-test, decode lookups without collectives
    from dsv41_pd001 import epeer

    epeer.prepare(module)  # dsv41-pd068 EPEER: kernel built; shards mapped by the first served forward
    from dsv41_pd001 import indexer_streams

    indexer_streams.prepare(module)  # dsv41pd.v41b IDXSTREAM: ratio 1 / 2 indexers registered, side stream made
    from dsv41_pd001 import dp_counts

    dp_counts.prepare()  # dsv41-pd048 DPCNT: stock's scattered_dp_counts captured, rank layout checked
    from dsv41_pd001 import exm

    exm.install(module)  # dspd-v51 EXM: before LEAN's captures; decode role builds and self-tests here, null defers
    from dsv41_pd001 import lean_layers

    lean_layers.prepare(view, module, _sk_mix_stats_impl())  # dsv41-pd032 LEAN: early-layer forwards and graphs
    from dsv41_pd001 import placement

    placement.defer(module)  # dsv41-pd002 PLACEMENT: validated now, placed by the first eager forward's settle()
    logger.warning("dsv41-pd001 MEGA_FRONT serves all %d MoE layers; PLACEMENT pending", LAYERS)
    return view


def forward_model(view, *args, **kwargs):
    """Run the upstream model forward over the candidate's private layer views."""
    from dsv41_pd001 import exm
    from dsv41_pd001 import placement

    batch = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2] if len(args) > 2 else None
    exm.vote(batch)  # dspd-v51 EXM: capture vote, first and outside the try (no rank-local raise can skip the gloo MIN)
    from sglang.srt.models import deepseek_v4_mhc as stock_mhc

    from dsv41_pd001 import post_fusion_floor

    stock = _mhc_stock_post_fusion()
    sk_stats = _sk_mix_stats_impl()
    stock_stats = _SK[0]
    view._index_reuse.clear()  # dsv41-pd003 INDEX_REUSE: no packed segment crosses a forward or a replay
    stock_mhc._post_fusion = post_fusion_floor._post_fusion  # dsv41-pd004 MHC: this forward's boundaries only
    stock_mhc._mix_stats_impl = sk_stats  # dsv41-pd006 SK: this forward's stats only
    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring as stock_scoring

    from dsv41_pd001 import topk_jit

    stock_scoring.topk_ragged = topk_jit.topk_ragged  # dsv41-pd033 TOPK: this forward's dense prefill top-k only
    from dsv41_pd001 import v16_levers

    try:
        from dsv41_pd001 import prefill_gemm

        from dsv41_pd001 import dp_counts  # dsv41-pd048 DPCNT: the replay-tail DP counts from this batch

        from dsv41_pd001 import skfuse  # dsv41-pd052 SKFUSE: small-kernel replacements, this forward only

        from dsv41_pd001 import fmlap  # dsv41-pd055 FMLAP: the prefill attention core, this forward only

        from dsv41_pd001 import m462  # dsv41-pd065 M462: #462's prefill MegaMoE pipeline, this forward only

        from dsv41_pd001 import indexer_streams  # dsv41pd.v41b IDXSTREAM: decode / verify indexer projections

        with v16_levers.routed(view, batch), prefill_gemm.routed(view, batch), skfuse.routed(view, batch), fmlap.routed(view, batch), m462.routed(view, batch), dp_counts.routed(batch), indexer_streams.routed(batch):  # dsv41-pd022 + dsv41-pd046 PGEMM v3 + dsv41pd.v42
            exm_token, exm_swap = None, False  # dspd-v51 EXM
            try:
                exm_token = exm.begin(batch)  # decode / verify forwards (None off)
                exm_swap = exm.swap()  # the floor's name, chained to exact_boundary; undone before routed restores
                out = view.forward(*args, **kwargs)
            finally:
                exm.unswap(exm_swap)
                exm.end(exm_token)  # joins pending stats; never raises
    finally:
        stock_mhc._post_fusion = stock
        stock_mhc._mix_stats_impl = stock_stats
        stock_scoring.topk_ragged = topk_jit.stock_topk_ragged()  # dsv41-pd033 TOPK
        view._index_reuse.clear()
    placement.settle()  # dsv41-pd002 PLACEMENT: once, at the end of the first eager (non-capturing) forward
    return out


# ---- dsv41-pd003 INDEX_REUSE: dsv41pd-c02's attention view (generator/pd1017c02, co-worker; exact) -------------------
def _prepare_with_reuse(view, stock, x, positions, forward_batch, backend, *args, **kwargs):
    """Run a stock prepare against a local facade, never a replaced global backend."""
    from dsv41pd_c02.index_reuse import backend_view

    return stock(x, positions, forward_batch, backend_view(backend, view._index_reuse), *args, **kwargs)


def _forward_prepare(view, *args, **kwargs):
    return _prepare_with_reuse(view, view._stock_prepare, *args, **kwargs)


def _forward_prepare_multi_stream(view, *args, **kwargs):
    return _prepare_with_reuse(view, view._stock_prepare_multi, *args, **kwargs)


def _forward_prepare_low_ratio_multi_stream(view, *args, **kwargs):
    return _prepare_with_reuse(view, view._stock_prepare_low_ratio, *args, **kwargs)


def _check_index_reuse_domain(module) -> None:
    """c02's domain: a four-GPU engine, attention TP1/2/4, flashmla or flashmla_mega, 1017 shapes."""
    from sglang.srt.runtime_context import get_exec, get_parallel

    parallel, backend = get_parallel(), get_exec().kernel.dsv4_attn_backend
    if parallel.tp_size != 4 or parallel.attn_tp_size not in (1, 2, 4):
        raise RuntimeError(f"dsv41-pd003 INDEX_REUSE requires a four-GPU engine, got tp={parallel.tp_size}, "
                           f"attention_tp={parallel.attn_tp_size}")
    if backend not in ("flashmla", "flashmla_mega"):
        raise RuntimeError(f"dsv41-pd003 INDEX_REUSE requires a PD-compatible attention backend, got {backend}")
    config = module.config
    if (config.num_attention_heads, config.o_groups) != (64, 8) or not module.hc_pre_from_prev_sublayer:
        raise RuntimeError("dsv41-pd003 INDEX_REUSE requires DeepSeek-V4.1-Flash-1017's current-main boundary layers")


def _attention_view(attention, reuse):
    from sglang.srt.models.deepseek_v4 import MQALayer

    if type(attention) is not MQALayer or not hasattr(attention, "use_mega_attention"):
        raise TypeError("dsv41-pd003 INDEX_REUSE requires the current-main MQALayer")
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


# ---- dsv41-pd004 MHC: decode-sized mHC boundaries skip Mega-mHC (post_fusion_floor.py) ------------------------------
_MHC_STOCK = []


def _mhc_stock_post_fusion():
    """The sealed deepseek_v4_mhc._post_fusion, captured once (at prepare, before any candidate forward swapped it)."""
    if not _MHC_STOCK:
        from sglang.srt.models import deepseek_v4_mhc as stock_mhc

        fn = stock_mhc._post_fusion
        if fn.__module__ != "sglang.srt.models.deepseek_v4_mhc" or fn.__qualname__ != "_post_fusion":
            raise RuntimeError(f"dsv41-pd004 MHC: deepseek_v4_mhc._post_fusion is not stock's ({fn.__module__}."
                               f"{fn.__qualname__})")
        _MHC_STOCK.append(fn)
    return _MHC_STOCK[0]


# ---- dsv41-pd006 SK: the mHC stats' reduce on pd001's hc_sinkhorn kernel, via load_jit (mhc_stats_jit.py) ---------
_SK = []


def _sk_mix_stats_impl():
    """[stock _mix_stats_impl, drop-in bound to it]: captured once, at prepare, before any candidate forward swapped it."""
    if not _SK:
        from sglang.srt.models import deepseek_v4_mhc as stock_mhc

        from dsv41_pd001 import mhc_stats_jit

        fn = stock_mhc._mix_stats_impl
        if fn.__module__ != "sglang.srt.models.deepseek_v4_mhc" or fn.__qualname__ != "_mix_stats_impl":
            raise RuntimeError(f"dsv41-pd006 SK: deepseek_v4_mhc._mix_stats_impl is not stock's ({fn.__module__}."
                               f"{fn.__qualname__})")
        _SK.extend((fn, mhc_stats_jit.make_mix_stats_impl(fn)))
    return _SK[1]
