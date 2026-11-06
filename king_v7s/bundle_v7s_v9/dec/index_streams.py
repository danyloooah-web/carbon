from __future__ import annotations

import contextlib
from functools import partial

import torch
import sglang.kernels.ops.layernorm.rmsnorm_mxfp8 as rmsnorm_mxfp8_mod
from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import index_q_rope_pack_weights
from sglang.srt.layers.attention.dsv4 import indexer as indexer_mod
from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk, scoring, sparse_table
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput, accepts_mxfp8_swizzled_input
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_utils.capture_mode import skip_low_ratio_indexer

from bundle_v9.dec import glue_pdl, paged_logits, prenorm_pair, paged_topk, qnorm_dec

_STOCK_NORM = rmsnorm_mxfp8_mod.rmsnorm_mxfp8
_STOCK_PROJECTIONS = scoring._index_q_and_weights
_SIDE: list = []
_ATTN: list = []
_BY_NORM: dict = {}
_SERVED: set = set()
_PENDING: dict = {}


def _rmsnorm_mxfp8(x, weight, eps):
    served = qnorm_dec.norm(x, weight, eps)  # king-v2e QNORM-DEC; None = stock, as in the king
    y, y_q, y_sf = served if served is not None else _STOCK_NORM(x, weight, eps)
    prenorm_pair.mark(weight)
    entry = _BY_NORM.get(weight.data_ptr())
    if entry is not None and not skip_low_ratio_indexer(entry[1]):
        indexer = entry[0]
        current = torch.cuda.current_stream()
        pending = _PENDING.setdefault(id(indexer), {})
        pending["e"] = current.record_event()
        side = _SIDE[1]
        side.wait_stream(current)
        with torch.cuda.stream(side):
            q, _ = indexer.wq_b(Mxfp8SwizzledInput(y_q, y_sf))
        current.wait_stream(side)
        pending["q"] = (y, q, y_q, y_sf)
    return y, y_q, y_sf


def _sources(stock, *, layer, x, q_lora, positions, forward_batch, run_compressor=True, run_indexer=True):
    indexer = layer.indexer
    if (indexer is not None and id(indexer) in _SERVED and x.shape[0] > 0
            and not skip_low_ratio_indexer(layer.compress_ratio)):
        entry = _PENDING.setdefault(id(indexer), {})
        if "w" not in entry and q_lora is None:
            side = _SIDE[0]
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                entry["w"] = (x, indexer.head_weights_raw(x))
        elif "w" not in entry and run_indexer:
            side = _SIDE[0]
            side.wait_event(entry["e"])
            with torch.cuda.stream(side):
                entry["w"] = (x, indexer.head_weights_raw(x))
    return stock(layer=layer, x=x, q_lora=q_lora, positions=positions, forward_batch=forward_batch,
                 run_compressor=run_compressor, run_indexer=run_indexer)


def _index_q_and_weights(*, indexer, x, q_lora, freqs_cis, positions):
    entry = _PENDING.pop(id(indexer), {})
    w, q = entry.get("w"), entry.get("q")
    if w is None or q is None or w[0] is not x or q[0] is not q_lora:
        raise RuntimeError("bundle-v9 IDXSTREAM: an indexer scored rows whose wq_b / weights_proj were not started "
                           "for its own x and q_lora in this forward")
    torch.cuda.current_stream().wait_stream(_SIDE[0])
    torch.cuda.current_stream().wait_stream(_SIDE[1])
    rows, heads = x.shape[0], indexer.n_local_heads
    args = (q[1].view(rows, heads, indexer.index_head_dim), torch.view_as_real(freqs_cis).flatten(-2), positions, w[1],
            indexer.head_weight_scale)
    # king-v6 IDX-W16: bf16 head weights from the PDL copy of the king's index_q replica (None = stock's fp32 weights).
    # Every consumer converts them to bf16 (PAGED-LOGITS, stock paged / sparse logits), which is then a no-op.
    served = glue_pdl.index_q(*args)
    q_fp4, q_sf, weights = served if served is not None else index_q_rope_pack_weights(*args)
    return q_fp4.view(rows, heads, 64), q_sf.view(rows, heads), weights


@contextlib.contextmanager
def routed(forward_batch):
    mode = getattr(forward_batch, "forward_mode", None)
    if not _SIDE or mode is None or not (mode.is_decode() or mode.is_target_verify()):
        yield
        return
    backend = get_attn_backend()
    stock_sources = backend.forward_low_ratio_sources
    rmsnorm_mxfp8_mod.rmsnorm_mxfp8 = _rmsnorm_mxfp8
    scoring._index_q_and_weights = _index_q_and_weights
    backend.forward_low_ratio_sources = partial(_sources, stock_sources)
    if paged_topk.enabled():
        indexer_mod.topk_transform_paged_bf16 = paged_topk.topk_transform_paged_bf16
        sparse_table.topk_transform_paged_bf16 = paged_topk.topk_transform_paged_bf16
    if paged_logits.enabled():
        full_topk.deep_gemm_fp4_paged_mqa_logits = paged_logits.deep_gemm_fp4_paged_mqa_logits
        sparse_table.deep_gemm_fp4_paged_mqa_logits = paged_logits.deep_gemm_fp4_paged_mqa_logits
    own = [attn.alt_streams[-1] for attn in _ATTN]
    for attn in _ATTN:
        attn.alt_streams[-1] = _SIDE[2]
    try:
        yield
        if _PENDING:
            raise RuntimeError(f"bundle-v9 IDXSTREAM: indexer projections started for {len(_PENDING)} layers that "
                               "never scored in this forward")
    finally:
        for attn, stream in zip(_ATTN, own):
            attn.alt_streams[-1] = stream
        rmsnorm_mxfp8_mod.rmsnorm_mxfp8 = _STOCK_NORM
        scoring._index_q_and_weights = _STOCK_PROJECTIONS
        del backend.forward_low_ratio_sources
        if paged_topk.enabled():
            indexer_mod.topk_transform_paged_bf16 = sparse_table.topk_transform_paged_bf16 = (
                paged_topk.stock_topk_transform_paged_bf16())
        if paged_logits.enabled():
            full_topk.deep_gemm_fp4_paged_mqa_logits = sparse_table.deep_gemm_fp4_paged_mqa_logits = (
                paged_logits.stock_deep_gemm_fp4_paged_mqa_logits())
        _PENDING.clear()


def prepare(model) -> None:
    if _STOCK_NORM is not rmsnorm_mxfp8_mod.rmsnorm_mxfp8 or _STOCK_PROJECTIONS is not scoring._index_q_and_weights:
        raise RuntimeError("bundle-v9 IDXSTREAM: rmsnorm_mxfp8 or _index_q_and_weights was swapped before prepare")
    served, device = [], None
    for layer in model.layers:
        attn = layer.self_attn
        if attn.indexer is None or attn.compress_ratio not in (1, 2):
            continue
        if attn.alt_streams is None or len(attn.alt_streams) != 3:
            raise TypeError(f"bundle-v9 IDXSTREAM: layer {attn.layer_id} has no three-stream low-ratio forward")
        _BY_NORM[attn.q_norm.weight.data_ptr()] = (attn.indexer, attn.compress_ratio)
        _SERVED.add(id(attn.indexer))
        _ATTN.append(attn)
        served.append(attn.layer_id)
        device = attn.q_norm.weight.device
    if not served:
        raise TypeError("bundle-v9 IDXSTREAM serves DeepSeek-V4.1's ratio 1 / 2 indexers; this stack has none")
    _prove_idxq(model)
    glue_pdl.prepare_idx([attn.indexer for attn in _ATTN])  # king-v6 IDX-W16; never raises, stock index_q on failure
    paged_topk.prepare(model)
    paged_logits.prepare(model)
    if not _SIDE:
        top = torch.cuda.Stream.priority_range()[1]
        _SIDE.extend([torch.cuda.Stream(device=device), torch.cuda.Stream(device=device, priority=top),
                      torch.cuda.Stream(device=device, priority=top)])
        own = {s.cuda_stream for attn in _ATTN for s in attn.alt_streams}
        if len({s.cuda_stream for s in _SIDE}) != 3 or own & {s.cuda_stream for s in _SIDE}:
            raise RuntimeError("bundle-v9 IDXSTREAM: the side streams are not three new streams")
    print(f"bundle-v9 IDXSTREAM: indexer streams on layers {served}, priority "
          f"{[s.priority for s in _SIDE]}", flush=True)


def _prove_idxq(model) -> None:
    gen = None
    for layer in model.layers:
        attn = layer.self_attn
        if attn.indexer is None or attn.compress_ratio not in (1, 2):
            continue
        linear, norm = attn.indexer.wq_b, attn.q_norm
        if not accepts_mxfp8_swizzled_input(linear):
            raise TypeError(f"bundle-v9 IDXQ: layer {attn.layer_id}'s indexer wq_b does not take the swizzled pair")
        device = norm.weight.device
        gen = gen or torch.Generator(device=device).manual_seed(20261103)
        for rows in (6, 48, 96, 216, 288):
            x = (torch.randn((rows, norm.weight.shape[0]), device=device, generator=gen, dtype=torch.float32) * 3).to(torch.bfloat16)
            y, y_q, y_sf = _STOCK_NORM(x, norm.weight, norm.variance_epsilon)
            whole, _ = linear(y)
            pair, _ = linear(Mxfp8SwizzledInput(y_q, y_sf))
            if not torch.equal(whole.view(torch.int16), pair.view(torch.int16)):
                raise RuntimeError(f"bundle-v9 IDXQ: layer {attn.layer_id} differs from stock's wq_b at {rows} rows")
