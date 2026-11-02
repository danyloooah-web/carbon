"""dsv41-pd034 ENGRAM_COLS: the Engram DP lookup exchanges only each shard's own hash columns. Bitwise stock.

Stock (sglang.srt.layers.engram.EngramEmbedding._dp_sharded_lookup under DP attention, SUM_LEN padding): every rank
gathers all ranks' hash ids, looks all 24 hash columns of every row up in its own table shard (zero where another
shard owns the row), all-reduces that [rows, 24 x 256] bf16 buffer and keeps its own slice. dsv41-pd010 ENGRAM_A2A
(superseded here) replaced the all-reduce by an all-to-all of each rank's rows plus a four-way sum on balanced forwards.

The shards are column-aligned: `shard_range` gives rank r exactly the subtables of hash columns 6r..6r+5 (24 columns
over 4 ranks), and a column-c hash id is `hash % prime_c + bounds[c]` (EngramHasher), always inside column c's
subtable (`check_domain` verifies both from the live model). So the only non-zero values rank r contributes are its
own six columns. Here every rank gathers just those columns for every row (stock's gather kernel, a quarter of its
writes), sends rank d the rows of rank d (`all_to_all_single`, split sizes from the host-side token counts, no device
sync) and places source s's block at columns 6s..6s+5 of its own rows. A placed value is the owner's value; stock's sum
of it with three zeros only turns -0.0 into +0.0, which the placement reproduces by adding +0.0 (fp32 opmath).
Pad rows past the rank's count read zero, as after stock's dp_scatter.

Measured on pd027 (box 204.12.176.156, 2026-10-28), per Engram layer and twice a forward: cold-round prefill (32K
rows a rank, trace p46) gather 1.8 ms + all-to-all 3.0 ms + sum 0.9 ms; steady-state heavy forwards (one rank at 32K
rows, stock's all-reduce, trace p43) gather 0.6 ms + all-reduce 1.5 ms.

The choice reads only facts every DP rank shares (all-gathered token counts, the batch-wide extend flag, the padding
mode, capture), never the rank's own forward mode or row count: pd010 r1 tested the rank-local is_extend(), an idle
rank took stock's all-reduce while the others entered the all-to-all, and the PD prefill engine hung (f4_1_pd011, box
204.12.176.156, 2026-10-27 23:06 UTC). An idle rank (forward mode IDLE) enters the all-to-all with zero rows. Forwards
below MIN_ROWS rows across the group (host-bound), MAX_LEN, gatherv, decode and capturing forwards run stock's method.

dsv41-pd052 SKFUSE: the placement runs skfuse.engram_place (EPLACE, 16-byte vectors; stock's strided torch.add was
~0.6 ms a cold layer), and where PGEMM routes the wkv for this rank's rows the Engram view's forward (engram_forward)
takes the exchange's received blocks straight to the wkv's MXFP8 pair (EQUANT: placement fused with stock's
flashinfer_mxfp8_quantize) and calls PGEMM's own GEMM on it, so the [rows, 6144] bf16 buffer is never written. Both
are stock's bytes (skfuse.py); the collectives are the same on every rank whichever local path a rank takes.

dsv41-pd054 SKFUSE2 (skfuse2.py):
  IDGATHER  the exchange's id gather is the TP group's all_gatherv of each rank's ids (one NCCL all-gather, or a group
            of broadcasts when the counts differ) instead of dp_gather_replicate's all-reduce of a zero-padded buffer:
            the exchange's domain has sum(counts) == the buffer's rows, so both fill every row with its owner's ids.
  EQ8       where PGEMM routes the Engram wkv for EVERY rank's count (all-gathered facts, so every rank decides alike
            and every rank is then in engram_forward), the exchange moves FP8: each sender quantizes its own columns'
            rows into MXFP8 codes + scale bytes (a 32-element group lies in one source's 1536 columns, so the
            receiver's group math runs on the sender, bit for bit), fused with stock's gather of those rows (EQ8G), and
            the receiver places the bytes into the wkv GEMM's codes and swizzled scales. The same one all_to_all_single
            (u8 rows of 1584 bytes instead of bf16 rows of 1536 values: half the bytes).
"""

from __future__ import annotations

from types import MethodType

from dsv41_pd001 import edec as _pd071_edec  # dsv41-pd071 EDEC

from dsv41_pd001 import epeer as _pd068_epeer  # dsv41-pd068 EPEER

MIN_ROWS = 8192  # all-gathered rows; smaller prefill forwards are host-bound, where stock's one all-reduce is cheaper
WORLD = 4

_STOCK = []


def _stock_lookup():
    """EngramEmbedding._dp_sharded_lookup as loaded, captured once before any view is bound."""
    if not _STOCK:
        from sglang.srt.layers.engram import EngramEmbedding

        fn = EngramEmbedding._dp_sharded_lookup
        if fn.__module__ != "sglang.srt.layers.engram" or fn.__qualname__ != "EngramEmbedding._dp_sharded_lookup":
            raise RuntimeError(f"dsv41-pd034 ENGRAM_COLS: EngramEmbedding._dp_sharded_lookup is not stock's "
                               f"({fn.__module__}.{fn.__qualname__})")
        _STOCK.append(fn)
    return _STOCK[0]


def check_domain(engram, model) -> tuple:
    """The static facts the column exchange relies on, as this rank's owned column range [c0, c1); raise otherwise."""
    from sglang.srt.layers.engram import Engram, EngramEmbedding
    from sglang.srt.runtime_context import get_parallel

    if type(engram) is not Engram or type(engram.embed) is not EngramEmbedding:
        raise TypeError(f"dsv41-pd034 ENGRAM_COLS requires stock Engram / EngramEmbedding, got "
                        f"{type(engram).__name__} / {type(engram.embed).__name__}")
    embed = engram.embed
    if embed.peer is not None or embed.host_table is not None or embed.staged_by_scheduler:
        raise RuntimeError("dsv41-pd034 ENGRAM_COLS requires device-resident shards (no peer, host table or staging)")
    parallel = get_parallel()
    facts = (parallel.tp_size, parallel.attn_dp_size, parallel.attn_tp_size, parallel.attn_cp_size)
    if facts != (WORLD, WORLD, 1, 1) or embed.tp_size != WORLD or parallel.tp_rank != parallel.attn_dp_rank:
        raise RuntimeError(f"dsv41-pd034 ENGRAM_COLS requires TP4 = DP-attention 4 (attention TP 1, no CP) with TP rank "
                           f"= DP rank, got (tp, dp, attn_tp, attn_cp) = {facts}")
    layout, hasher = model.engram_layout, model.engram_hasher
    bounds = layout.column_bounds(engram.layer_hash_index)
    cols = len(bounds) - 1
    if cols % WORLD or embed.dim != layout.head_dim:
        raise RuntimeError(f"dsv41-pd034 ENGRAM_COLS requires whole columns a rank, got {cols} columns of {embed.dim}")
    per = cols // WORLD
    c0, c1 = parallel.tp_rank * per, (parallel.tp_rank + 1) * per
    if (embed.row_start, embed.row_start + embed.rows) != (bounds[c0], bounds[c1]):
        raise RuntimeError(f"dsv41-pd034 ENGRAM_COLS: rank {parallel.tp_rank} holds rows "
                           f"[{embed.row_start}, {embed.row_start + embed.rows}), not columns {c0}..{c1 - 1}")
    offsets = [int(v) for v in hasher.offsets[engram.layer_hash_index].tolist()]
    primes = [int(p) for per_ngram in hasher.primes[engram.layer_hash_index].tolist() for p in per_ngram]
    if offsets != list(bounds[:-1]) or primes != [bounds[c + 1] - bounds[c] for c in range(cols)]:
        raise RuntimeError("dsv41-pd034 ENGRAM_COLS: the hasher's column offsets / primes are not the table's bounds")
    return c0, c1


def _cols_counts(forward_batch, rows: int):
    """Per-rank row counts when this forward takes the column exchange, else None (stock's method runs).

    Every input is identical on all DP ranks, so all four enter the same collective (module docstring: the r1 hang).
    """
    import torch

    from sglang.srt.layers.dp_attention import is_dp_gatherv_active

    if forward_batch is None or torch.cuda.is_current_stream_capturing():
        return None
    padding = forward_batch.dp_padding_mode
    if not forward_batch.is_extend_in_batch or padding is None or not padding.is_sum_len() or is_dp_gatherv_active():
        return None
    counts = forward_batch.global_num_tokens_cpu
    if counts is None or len(counts) != WORLD:
        return None
    counts = [int(c) for c in counts]
    if sum(counts) != rows or rows < MIN_ROWS:
        return None
    return counts


def _fp8_counts(engram, counts) -> bool:
    """EQ8's domain: PGEMM routes the Engram wkv for every rank's count. Only all-gathered counts enter, so every
    rank decides alike; each rank's rows are then >= its count, so every rank is served by engram_forward."""
    return engram is not None and all(_wkv_route(engram, c) is not None for c in counts)


def _exchange(self, indices, forward_batch, engram=None):
    """The column exchange's collectives for this forward: (received, mine, n_local, width, fp8), or None, before any
    collective, when the forward is outside the domain (stock's method then runs). received is [WORLD * mine, width]
    bf16, or with fp8 (EQ8, engram_forward only) [WORLD * mine, 1584] u8 rows of MXFP8 codes + scale bytes."""
    import torch
    import torch.distributed as dist

    from sglang.kernels.ops.embeddings.engram_gather import engram_gather
    from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len
    from sglang.srt.runtime_context import get_parallel

    from dsv41_pd001 import skfuse2

    rows = get_global_dp_buffer_len()
    counts = _cols_counts(forward_batch, rows)
    if counts is None:
        return None
    parallel = get_parallel()
    mine = counts[parallel.attn_dp_rank]
    n_local = indices.shape[0]
    if n_local < mine:
        raise RuntimeError(f"dsv41-pd034 ENGRAM_COLS: rank {parallel.attn_dp_rank} holds {n_local} rows for its DP "
                           f"count {mine}")
    # dsv41-pd054 IDGATHER: every rank's ids by the TP group's all_gatherv (stock: dp_gather_replicate's all-reduce)
    ids_global = skfuse2.id_gather(indices, counts, rows, forward_batch)
    c0, c1 = self._pd034_cols
    width = (c1 - c0) * self.dim
    own = ids_global[:, c0:c1].contiguous()
    fp8 = _fp8_counts(engram, counts)
    if fp8:  # dsv41-pd054 EQ8 + EQ8G: the gathered rows leave as MXFP8 codes + scale bytes
        values = skfuse2.gather_pack(self, own)
    else:
        values = torch.empty((rows, width), dtype=torch.bfloat16, device=indices.device)
        engram_gather(*self._table_ptrs(), own.view(-1), values.view(-1, self.dim), self.dim, FP8_BLOCK_SIZE,
                      row_lo=self.row_start, row_hi=self.row_start + self.rows)
    received = values.new_empty((WORLD * mine, values.shape[1]))
    dist.all_to_all_single(received, values, [mine] * WORLD, counts, group=parallel.tp_group.device_group)
    if fp8 and skfuse2.CHECK:
        skfuse2.check_eq8(self, own, received, mine, n_local, counts, parallel.tp_group.device_group)
    return received, mine, n_local, width, fp8


def dp_sharded_lookup(self, indices, forward_batch):
    """Stock's lookup with the column exchange (bound on a view of the EngramEmbedding)."""
    peer = _pd071_edec.lookup(self, indices, forward_batch)  # dsv41-pd071 EDEC: decode rows read in place
    if peer is not None:
        return peer
    from dsv41_pd001 import skfuse

    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len

    counts = _cols_counts(forward_batch, get_global_dp_buffer_len())
    if counts is not None and _fp8_counts(getattr(self, "_pd054_engram", None), counts):
        # every rank's count is routed, so every rank should be in engram_forward's FP8 exchange: a caller mix would
        # pair a bf16 all-to-all here with u8 ones elsewhere (dsv41-pd054 EQ8). Fail loudly, before any collective.
        raise RuntimeError("dsv41-pd054 EQ8: the bf16 lookup ran where every rank's count takes the FP8 exchange")
    peer = _pd068_epeer.lookup_bf16(self, indices, forward_batch)  # dsv41-pd068 EPEER: no collective
    if peer is not None:
        return peer.view(*indices.shape, self.dim)
    exchanged = _exchange(self, indices, forward_batch)
    if exchanged is None:
        return _stock_lookup()(self, indices, forward_batch)
    received, mine, n_local, width, _ = exchanged
    local = received.new_empty((n_local, WORLD * width))
    # source s holds columns s*per..(s+1)*per; + 0.0 turns -0.0 into +0.0 as stock's sum with three zeros does
    skfuse.engram_place(received, local, mine)  # dsv41-pd052 EPLACE: stock's torch.add(..., 0.0) placement, bit for bit
    if n_local > mine:
        local[mine:].zero_()  # stock's dp_scatter zero-fills the local rows and copies this rank's count
    return local.view(*indices.shape, self.dim)


_STOCK_FORWARD = []


def _stock_forward():
    """Engram.forward as loaded, captured once before any view is bound."""
    if not _STOCK_FORWARD:
        from sglang.srt.layers.engram import Engram

        fn = Engram.forward
        if fn.__module__ != "sglang.srt.layers.engram" or fn.__qualname__ != "Engram.forward":
            raise RuntimeError(f"dsv41-pd052 EQUANT: Engram.forward is not stock's ({fn.__module__}.{fn.__qualname__})")
        _STOCK_FORWARD.append(fn)
    return _STOCK_FORWARD[0]


def _wkv_route(engram, rows: int):
    """PGEMM's (registered wkv, route) for this rank's rows, or None when PGEMM does not serve the call."""
    from dsv41_pd001 import prefill_gemm

    entry = prefill_gemm._REG.get(engram.wkv.weight.data_ptr())
    if entry is None or rows <= 0 or not prefill_gemm._STOCK_LINEAR:
        return None
    route = prefill_gemm._route(entry, rows)
    return None if route is None else (entry, route)


def engram_forward(self, x, hash_ids, forward_batch=None, *, input_ids, cp_all_tokens=False):
    """Stock's Engram.forward (bound on the Engram view); in the column exchange's domain with PGEMM serving the wkv
    for this rank's rows, the exchange feeds EQUANT's MXFP8 pair to PGEMM's GEMM (stock quantizes the bf16 placement
    then runs the same GEMM). Every other call is stock's forward, which reaches dp_sharded_lookup above."""
    from sglang.srt.layers.engram import engram_gate

    from dsv41_pd001 import prefill_gemm, skfuse

    stock = _stock_forward()
    if cp_all_tokens or forward_batch is None or forward_batch.engram_staged_rows is not None:
        return stock(self, x, hash_ids, forward_batch, input_ids=input_ids, cp_all_tokens=cp_all_tokens)
    served = _wkv_route(self, x.shape[0])
    if served is None:
        return stock(self, x, hash_ids, forward_batch, input_ids=input_ids, cp_all_tokens=cp_all_tokens)
    peer = _pd068_epeer.lookup(self, hash_ids, forward_batch, x.shape[0])  # dsv41-pd068 EPEER: no collective
    if peer is not None:
        entry, route = served
        kv = prefill_gemm._gemm(*route, *peer, entry[1], entry[2]).view(x.shape[0], entry[1].shape[0])
        image_ids = input_ids if self.image_token_id is not None else None
        return engram_gate(x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value, image_ids,
                           self.image_token_id)
    exchanged = _exchange(self.embed, hash_ids, forward_batch, engram=self)
    if exchanged is None:  # outside the exchange's domain, decided before any collective: stock's whole forward
        return stock(self, x, hash_ids, forward_batch, input_ids=input_ids, cp_all_tokens=cp_all_tokens)
    received, mine, n_local, width, fp8 = exchanged
    if WORLD * width != hash_ids.shape[-1] * self.embed.dim or n_local != x.shape[0]:
        raise RuntimeError(f"dsv41-pd052 EQUANT: exchanged width {WORLD * width} / rows {n_local} do not match the "
                           f"layer's {hash_ids.shape[-1] * self.embed.dim} / {x.shape[0]}")
    if fp8:
        from dsv41_pd001 import skfuse2

        q, s = skfuse2.place(received, mine, n_local)  # dsv41-pd054 EQ8: the received codes and scale bytes, placed
    else:
        q, s = skfuse.engram_quant(received, mine, n_local)
    entry, route = served
    kv = prefill_gemm._gemm(*route, q, s, entry[1], entry[2]).view(n_local, entry[1].shape[0])
    image_ids = input_ids if self.image_token_id is not None else None
    return engram_gate(x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value, image_ids, self.image_token_id)


def engram_view(engram, view, model):
    """A view of an Engram whose embedding view exchanges own columns; parameters and tables stay shared."""
    _stock_lookup()
    cols = check_domain(engram, model)
    engram_v = view(engram)
    embed_v = view(engram.embed)
    embed_v._pd034_cols = cols
    embed_v._dp_sharded_lookup = MethodType(dp_sharded_lookup, embed_v)
    engram_v.embed = embed_v
    _stock_forward()
    engram_v.forward = MethodType(engram_forward, engram_v)  # dsv41-pd052 EQUANT
    embed_v._pd054_engram = engram_v  # dsv41-pd054 EQ8: dp_sharded_lookup checks the FP8 domain against its wkv
    _pd068_epeer.register(engram_v, model)  # dsv41-pd068 EPEER: the view and every rank's shard rows
    return engram_v
