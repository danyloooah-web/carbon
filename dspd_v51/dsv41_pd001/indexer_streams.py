"""dsv41pd.v41b IDXSTREAM (dsv41pd.v29's indexer streams, on dsv41-pd070bf): in decode and verify forwards an
index-source layer's two indexer projections, wq_b on the normed q_lora and weights_proj on x, start on a side stream
as soon as their inputs exist instead of queueing behind the layer's compressor, and the indexer's scoring takes their
outputs from there.

Stock (_forward_prepare_low_ratio_multi_stream) gives a ratio 1 / 2 layer one sources stream: the compressor from x's
arrival, then, once q_lora is ready, the indexer, whose _index_q_and_weights launches wq_b, weights_proj and the
rope / FP4 pack ahead of the paged logits and top-k; attention waits for all of it (pdprofD-v16: +32 to +125 us a
source layer over a plain one, 2.7-4% of a decode step). Here:
- the compressor pass, on a stream that has waited for x, forks the side stream and starts weights_proj(x) there;
- q_norm's rmsnorm_mxfp8, q_lora's producer, forks it again and starts wq_b(q_lora) there;
- a layer without a compressor pass starts weights_proj(x) on its sources stream at the indexer pass, beside the side
  stream's wq_b;
- scoring's _index_q_and_weights waits for the side stream and runs stock's rope / FP4 pack on those outputs.
The kernels, their inputs and their row counts are stock's (decode and verify hand the indexer x and q_lora unsliced);
only the streams change, so every output byte is stock's. The side stream rejoins through the sources stream, which
stock joins before attention, and nothing the side stream wrote is reused before that join: its next allocation
comes from a later layer's fork, ordered after it. Extend forwards (pd052 SKFUSE swaps rmsnorm_mxfp8 and the sources
pass there) never enter this scope.
"""

from __future__ import annotations

import contextlib
from functools import partial

import torch
import sglang.kernels.ops.layernorm.rmsnorm_mxfp8 as rmsnorm_mxfp8_mod
from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import index_q_rope_pack_weights
from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_utils.capture_mode import skip_low_ratio_indexer

_STOCK_NORM = rmsnorm_mxfp8_mod.rmsnorm_mxfp8
_STOCK_PROJECTIONS = scoring._index_q_and_weights
_SIDE: list = []  # the side stream, made at prepare, before any capture
_BY_NORM: dict = {}  # q_norm weight data_ptr -> (the layer's indexer, its compress ratio)
_SERVED: set = set()  # id() of every registered indexer
# within one forward: id(indexer) -> {"w": (x, weights_proj(x)), "q": (q_lora, wq_b(q_lora))}
_PENDING: dict = {}


def _rmsnorm_mxfp8(x, weight, eps):
    """rmsnorm_mxfp8's arguments and result; an index-source layer's q_norm also starts its indexer's wq_b on the side
    stream, on the normed rows the indexer is handed later."""
    y, y_q, y_sf = _STOCK_NORM(x, weight, eps)
    entry = _BY_NORM.get(weight.data_ptr())
    if entry is not None and not skip_low_ratio_indexer(entry[1]):
        indexer = entry[0]
        side = _SIDE[0]
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            q, _ = indexer.wq_b(y)
        _PENDING.setdefault(id(indexer), {})["q"] = (y, q)
    return y, y_q, y_sf


def _sources(stock, *, layer, x, q_lora, positions, forward_batch, run_compressor=True, run_indexer=True):
    """forward_low_ratio_sources' keyword arguments after stock's bound method; an index-source layer's weights_proj(x)
    starts on the side stream at its compressor pass, or on this stream at its indexer pass when it had none."""
    indexer = layer.indexer
    if (indexer is not None and id(indexer) in _SERVED and x.shape[0] > 0
            and not skip_low_ratio_indexer(layer.compress_ratio)):
        entry = _PENDING.setdefault(id(indexer), {})
        if "w" not in entry and q_lora is None:  # the compressor pass: this stream has waited for x
            side = _SIDE[0]
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                entry["w"] = (x, indexer.head_weights_raw(x))
        elif "w" not in entry and run_indexer:
            entry["w"] = (x, indexer.head_weights_raw(x))
    return stock(layer=layer, x=x, q_lora=q_lora, positions=positions, forward_batch=forward_batch,
                 run_compressor=run_compressor, run_indexer=run_indexer)


def _index_q_and_weights(*, indexer, x, q_lora, freqs_cis, positions):
    """scoring._index_q_and_weights' arguments and result: stock's rope / FP4 pack on the projections the side stream
    and the sources pass started for this x and q_lora."""
    entry = _PENDING.pop(id(indexer), {})
    w, q = entry.get("w"), entry.get("q")
    if w is None or q is None or w[0] is not x or q[0] is not q_lora:
        raise RuntimeError("dsv41pd.v41b IDXSTREAM: an indexer scored rows whose wq_b / weights_proj were not started "
                           "for its own x and q_lora in this forward")
    torch.cuda.current_stream().wait_stream(_SIDE[0])
    rows, heads = x.shape[0], indexer.n_local_heads
    q_fp4, q_sf, weights = index_q_rope_pack_weights(
        q[1].view(rows, heads, indexer.index_head_dim),
        torch.view_as_real(freqs_cis).flatten(-2),
        positions,
        w[1],
        indexer.head_weight_scale,
    )
    return q_fp4.view(rows, heads, 64), q_sf.view(rows, heads), weights


@contextlib.contextmanager
def routed(forward_batch):
    """The indexer streams for one decode or verify forward, put back after it; every other forward runs stock's."""
    mode = getattr(forward_batch, "forward_mode", None)
    if not _SIDE or mode is None or not (mode.is_decode() or mode.is_target_verify()):
        yield
        return
    backend = get_attn_backend()
    stock_sources = backend.forward_low_ratio_sources
    rmsnorm_mxfp8_mod.rmsnorm_mxfp8 = _rmsnorm_mxfp8
    scoring._index_q_and_weights = _index_q_and_weights
    backend.forward_low_ratio_sources = partial(_sources, stock_sources)
    try:
        yield
        if _PENDING:
            raise RuntimeError(f"dsv41pd.v41b IDXSTREAM: indexer projections started for {len(_PENDING)} layers that "
                               "never scored in this forward")
    finally:
        rmsnorm_mxfp8_mod.rmsnorm_mxfp8 = _STOCK_NORM
        scoring._index_q_and_weights = _STOCK_PROJECTIONS
        del backend.forward_low_ratio_sources
        _PENDING.clear()


def prepare(model) -> None:
    """Every decoder layer's ratio 1 / 2 indexer registered under its q_norm weight, and the side stream made, before
    any graph capture."""
    if _STOCK_NORM is not rmsnorm_mxfp8_mod.rmsnorm_mxfp8 or _STOCK_PROJECTIONS is not scoring._index_q_and_weights:
        raise RuntimeError("dsv41pd.v41b IDXSTREAM: rmsnorm_mxfp8 or _index_q_and_weights was swapped before prepare")
    served, device = [], None
    for layer in model.layers:
        attn = layer.self_attn
        if attn.indexer is None or attn.compress_ratio not in (1, 2):
            continue
        _BY_NORM[attn.q_norm.weight.data_ptr()] = (attn.indexer, attn.compress_ratio)
        _SERVED.add(id(attn.indexer))
        served.append(attn.layer_id)
        device = attn.q_norm.weight.device
    if not served:
        raise TypeError("dsv41pd.v41b IDXSTREAM serves DeepSeek-V4.1's ratio 1 / 2 indexers; this stack has none")
    if not _SIDE:
        _SIDE.append(torch.cuda.Stream(device=device))
    print(f"dsv41pd.v41b IDXSTREAM: indexer streams on layers {served}", flush=True)
