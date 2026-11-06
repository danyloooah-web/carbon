
from __future__ import annotations

import contextlib
import logging
from types import MethodType

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

PROBE_ROWS = (1, 35, 36, 216, 217, 240, 264, 288)
DRAFT = 6
PROBE_SEED = 20261031
PINNED = (("deepseek_v4", "MQALayer._forward_prepare_low_ratio_multi_stream", "bc455a7c3e96a5e1"),)
_STOCK: list = []


def _same_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a is b or (a.data_ptr(), a.shape, a.stride(), a.dtype) == (b.data_ptr(), b.shape, b.stride(), b.dtype)


def _bits_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)
    return a.shape == b.shape and bool(torch.equal(a, b))


def _head_weights(stream, indexer, x: torch.Tensor):
    with torch.cuda.stream(stream):
        return indexer.head_weights_raw(x), stream.record_event()


def _query(stream, indexer, qin, freqs_cis, positions, w_raw, w_ready):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import index_q_rope_pack_weights

    rows, heads = qin[0].shape[0], indexer.n_local_heads
    with torch.cuda.stream(stream):
        q, _ = indexer.wq_b(qin)
        stream.wait_event(w_ready)
        q_fp4, q_sf, weights = index_q_rope_pack_weights(
            q.view(rows, heads, indexer.index_head_dim), torch.view_as_real(freqs_cis).flatten(-2), positions, w_raw,
            indexer.head_weight_scale)
        ready = stream.record_event()
    return (q_fp4.view(rows, heads, 64), q_sf.view(rows, heads), weights), ready


def _indexer_positions(attn_backend, positions: torch.Tensor) -> torch.Tensor:
    hoisted = getattr(attn_backend.forward_metadata, "low_ratio_pos_i64", None)
    if hoisted is None:
        return positions
    if hoisted.shape[0] < positions.shape[0]:
        raise RuntimeError(f"IHX: {hoisted.shape[0]} hoisted indexer rows < {positions.shape[0]} decode/verify rows")
    return hoisted[: positions.shape[0]]


@contextlib.contextmanager
def _handoff(layer_id: int, expect: tuple, packed: tuple, ready):
    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

    taken = []

    def hoisted(*, indexer, x, q_lora, freqs_cis, positions):
        if taken or any(a is not b for a, b in zip((indexer, x, q_lora, freqs_cis), expect)) or not _same_view(
                positions, expect[4]):
            raise RuntimeError(f"IHX L{layer_id}: the indexer asked for a query the hoist did not compute")
        taken.append(True)
        torch.cuda.current_stream().wait_event(ready)
        return packed

    if scoring._index_q_and_weights is not _STOCK[1]:
        raise RuntimeError("IHX: scoring._index_q_and_weights is not stock's")
    scoring._index_q_and_weights = hoisted
    try:
        yield
    finally:
        scoring._index_q_and_weights = _STOCK[1]
    if not taken:
        raise RuntimeError(f"IHX L{layer_id}: the indexer did not take the hoisted query")


def _prepare_hoisted(self, x, positions, forward_batch, attn_backend, q_out=None, x_quant=None):
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    from sglang.srt.model_executor.runner_utils.capture_mode import skip_low_ratio_indexer

    if skip_low_ratio_indexer(self.compress_ratio):
        return _STOCK[0](self, x, positions, forward_batch, attn_backend, q_out, x_quant=x_quant)
    pos = _indexer_positions(attn_backend, positions)
    current_stream = torch.cuda.current_stream()
    stream_kv, stream_q, stream_sources = self.alt_streams[0], self.alt_streams[1], self.alt_streams[-1]
    x_linear = x_quant if x_quant is not None else x
    if self.compressor is not None:
        stream_sources.wait_stream(current_stream)
    qkv_a = None
    if self.fuse_wqa_wkv:
        qkv_a, _ = self.wqkv_a(x_linear)
    if self.compressor is not None:
        with torch.cuda.stream(stream_sources):
            attn_backend.forward_low_ratio_sources(
                layer=self, x=x, q_lora=None, positions=positions, forward_batch=forward_batch, run_indexer=False)
    stream_kv.wait_stream(current_stream)
    q_lora, q_for_wqb = self._compute_q_a(x_linear, qkv_a=qkv_a)
    if not isinstance(q_for_wqb, Mxfp8SwizzledInput):
        raise RuntimeError(f"IHX L{self.layer_id}: q_lora came without rmsnorm_mxfp8's (q, s)")
    with torch.cuda.stream(stream_kv):
        self._compute_kv_to_cache(x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a)
    w_raw, w_ready = _head_weights(stream_kv, self.indexer, x)
    stream_q.wait_stream(current_stream)
    packed, ready = _query(stream_q, self.indexer, q_for_wqb, self.freqs_cis, pos, w_raw, w_ready)
    stream_sources.wait_stream(current_stream)
    q = self._compute_q_b(q_for_wqb, positions, q_out)
    with torch.cuda.stream(stream_sources), _handoff(
            self.layer_id, (self.indexer, x, q_lora, self.freqs_cis, pos), packed, ready):
        attn_backend.forward_low_ratio_sources(
            layer=self, x=x, q_lora=q_lora, positions=positions, forward_batch=forward_batch, run_compressor=False)
    current_stream.wait_stream(stream_kv)
    current_stream.wait_stream(stream_q)
    current_stream.wait_stream(stream_sources)
    return q


def attn_view(attn, view_fn):
    view = view_fn(attn)
    view._forward_prepare_low_ratio_multi_stream = MethodType(_prepare_hoisted, view)
    return view


def _seeded(rows: int, k: int, g: torch.Generator, device) -> torch.Tensor:
    x = torch.randn(rows, k, generator=g, device=device, dtype=torch.float32)
    row = torch.pow(2.0, (torch.arange(rows, device=device) * 5 % 9 - 4).float())
    grp = torch.pow(2.0, (torch.arange(k // 32, device=device) * 3 % 7 - 3).float()).repeat_interleave(32)
    return (x * row[:, None] * grp[None, :]).to(torch.bfloat16)


def _scores(packs: tuple, ratio: int, topk: int, g: torch.Generator, device) -> list:
    from sglang.kernels.ops.attention.dsv4.index_logits import deep_gemm_fp4_paged_mqa_logits
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import DeepGEMMDecodeData

    rows, heads = packs[0][0].shape[0], packs[0][0].shape[1]
    lens = (torch.randint(1024, 2048 - DRAFT, (rows // DRAFT, 1), generator=g, device=device)
            + torch.arange(DRAFT, device=device)).view(-1).to(torch.int32)
    table = torch.randint(0, 128, (rows // DRAFT, 32), generator=g, device=device, dtype=torch.int32)
    k_cache = torch.randint(112, 144, (128, 64 * 68), generator=g, device=device, dtype=torch.uint8)
    k_cache = k_cache.view(128, 64, 1, 68)
    meta = PagedIndexerMetadata(
        page_size=256, compressed_page_size=64, page_table=table.repeat_interleave(DRAFT, 0), compressed_seq_lens=lens,
        use_topk_v2=False, compress_ratio=ratio, rows_per_request=DRAFT)
    valid = torch.arange(meta.max_compressed_seq_len, device=device)[None, :] < lens[:, None]
    out = []
    for q_fp4, q_sf, weights in packs:
        data = DeepGEMMDecodeData(q_fp4=q_fp4.view(rows, 1, heads, 64), q_sf=q_sf.view(rows, 1, heads),
                                  weights=weights, k_cache=k_cache)
        q, lens_v, tbl = data.paged_logits_args(meta)
        logits = deep_gemm_fp4_paged_mqa_logits(q, k_cache, weights, lens_v, tbl, meta.deep_gemm_metadata,
                                                meta.max_compressed_seq_len)
        out += [logits.masked_fill(~valid, 0), logits.float().masked_fill(~valid, float("-inf")).topk(topk).indices]
    return out


@torch.no_grad()
def _probe_layer(attn, rows_list, seed: int) -> list:
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    idx, dev, cur = attn.indexer, attn.freqs_cis.device, torch.cuda.current_stream()
    s_kv, s_q, s_src = attn.alt_streams[0], attn.alt_streams[1], attn.alt_streams[-1]
    for rows in rows_list:
        g = torch.Generator(device=dev).manual_seed(seed + 7919 * attn.layer_id + rows)
        x, q_a = _seeded(rows, idx.weights_proj.weight.shape[1], g, dev), _seeded(rows, attn.q_lora_rank, g, dev)
        pos = torch.randint(0, attn.freqs_cis.shape[0], (rows,), generator=g, device=dev)
        y, qin = attn._normalize_q_lora(q_a)
        if not isinstance(qin, Mxfp8SwizzledInput):
            return [f"L{attn.layer_id} r={rows}: _normalize_q_lora gave {type(qin).__name__}"]
        q_ref, s_ref = flashinfer_mxfp8_quantize(y, is_sf_swizzled_layout=True, alignment=32)
        for s in (s_src, s_kv, s_q):
            s.wait_stream(cur)
        with torch.cuda.stream(s_src):
            w_ref, b_ref = idx.head_weights_raw(x), idx.wq_b(y)[0]
            ref = _STOCK[1](indexer=idx, x=x, q_lora=y, freqs_cis=attn.freqs_cis, positions=pos)
        w_raw, w_ready = _head_weights(s_kv, idx, x)
        got, _ = _query(s_q, idx, qin, attn.freqs_cis, pos, w_raw, w_ready)
        with torch.cuda.stream(s_q):
            b_got = idx.wq_b(qin)[0]
        for s in (s_src, s_kv, s_q):
            cur.wait_stream(s)
        pairs = [("q", q_ref, qin[0]), ("sf", s_ref, qin[1]), ("wq_b", b_ref, b_got), ("head_w", w_ref, w_raw),
                 ("q_fp4", ref[0], got[0]), ("q_sf", ref[1], got[1]), ("weights", ref[2], got[2])]
        if rows % DRAFT == 0:
            l_ref, i_ref, l_got, i_got = _scores((ref, got), attn.compress_ratio, idx.index_topk, g, dev)
            pairs += [("logits", l_ref, l_got), ("topk_ids", i_ref, i_got)]
        bad = [f"L{attn.layer_id} r={rows}:{name}" for name, a, b in pairs if not _bits_eq(a, b)]
        if bad:
            return bad
    return []


@torch.no_grad()
def prepare(module, role: str) -> frozenset:
    if role not in ("decode", "null"):
        return frozenset()
    import sglang.srt.models.deepseek_v4 as deepseek_v4
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
    from sglang.srt.layers.quantization.mxfp8_input import accepts_mxfp8_swizzled_input
    from sglang.srt.runtime_context import get_parallel

    from bundle_v9 import dense_gemm, tactic_pins

    tactic_pins._verify_segments("IHX", {"deepseek_v4": deepseek_v4}, PINNED)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("IHX prepare requires an eager CUDA stream")
    stock = (deepseek_v4.MQALayer._forward_prepare_low_ratio_multi_stream, scoring._index_q_and_weights)
    if [f.__qualname__ for f in stock] != ["MQALayer._forward_prepare_low_ratio_multi_stream", "_index_q_and_weights"]:
        raise RuntimeError(f"IHX: stock functions are not loaded as stock ({[f.__qualname__ for f in stock]})")
    _STOCK[:] = stock
    attns = [layer.self_attn for layer in module.layers if getattr(layer.self_attn, "indexer", None) is not None]
    for attn in attns:
        streams = attn.alt_streams
        if (type(attn) is not deepseek_v4.MQALayer or type(attn.indexer) is not DeepseekV41Indexer
                or attn.compress_ratio not in (1, 2) or streams is None or len({id(s) for s in streams[:3]}) != 3
                or not accepts_mxfp8_swizzled_input(attn.indexer.wq_b)):
            raise RuntimeError(f"IHX: layer {attn.layer_id} is not a low-ratio multi-stream indexer layer IHX serves")
    if not attns:
        raise RuntimeError("IHX: the model has no indexer layer")
    bad: list = []
    try:
        with dense_gemm.probe_routes():
            for attn in dict.fromkeys((attns[0], attns[-1])):
                bad += _probe_layer(attn, PROBE_ROWS, PROBE_SEED)
        torch.cuda.synchronize()
    except Exception as exc:
        bad.append(f"{type(exc).__name__}: {exc}")
    parallel = get_parallel()
    gathered = [None] * parallel.tp_group.world_size
    dist.all_gather_object(gathered, bad[:4], group=parallel.tp_group.cpu_group)
    failures = [f"rank {r}: {', '.join(errs)}" for r, errs in enumerate(gathered) if errs]
    if failures:
        raise RuntimeError(f"IHX STRICT byte-probe failed ({' | '.join(failures)})")
    if int(parallel.tp_rank) == 0:
        logger.warning("[b7] IHX hoisted layers %s; probed L%d/L%d x %d rows",
                       [a.layer_id for a in attns], attns[0].layer_id, attns[-1].layer_id, len(PROBE_ROWS))
    return frozenset(attn.layer_id for attn in attns)
