
from __future__ import annotations

from types import MethodType

from bundle_v9 import peer_read as _peer_read

MIN_ROWS = 8192
WORLD = 4

_STOCK = []


def _stock_lookup():
    if not _STOCK:
        from sglang.srt.layers.engram import EngramEmbedding

        fn = EngramEmbedding._dp_sharded_lookup
        if fn.__module__ != "sglang.srt.layers.engram" or fn.__qualname__ != "EngramEmbedding._dp_sharded_lookup":
            raise RuntimeError(f"v9 COL-EXCHANGE: EngramEmbedding._dp_sharded_lookup is not stock's "
                               f"({fn.__module__}.{fn.__qualname__})")
        _STOCK.append(fn)
    return _STOCK[0]


def check_domain(engram, model) -> tuple:
    from sglang.srt.layers.engram import Engram, EngramEmbedding
    from sglang.srt.runtime_context import get_parallel

    if type(engram) is not Engram or type(engram.embed) is not EngramEmbedding:
        raise TypeError(f"v9 COL-EXCHANGE requires stock Engram / EngramEmbedding, got "
                        f"{type(engram).__name__} / {type(engram.embed).__name__}")
    embed = engram.embed
    if embed.peer is not None or embed.host_table is not None or embed.staged_by_scheduler:
        raise RuntimeError("v9 COL-EXCHANGE requires device-resident shards (no peer, host table or staging)")
    parallel = get_parallel()
    facts = (parallel.tp_size, parallel.attn_dp_size, parallel.attn_tp_size, parallel.attn_cp_size)
    if facts != (WORLD, WORLD, 1, 1) or embed.tp_size != WORLD or parallel.tp_rank != parallel.attn_dp_rank:
        raise RuntimeError(f"v9 COL-EXCHANGE requires TP4 = DP-attention 4 (attention TP 1, no CP) with TP rank "
                           f"= DP rank, got (tp, dp, attn_tp, attn_cp) = {facts}")
    layout, hasher = model.engram_layout, model.engram_hasher
    bounds = layout.column_bounds(engram.layer_hash_index)
    cols = len(bounds) - 1
    if cols % WORLD or embed.dim != layout.head_dim:
        raise RuntimeError(f"v9 COL-EXCHANGE requires whole columns a rank, got {cols} columns of {embed.dim}")
    per = cols // WORLD
    c0, c1 = parallel.tp_rank * per, (parallel.tp_rank + 1) * per
    if (embed.row_start, embed.row_start + embed.rows) != (bounds[c0], bounds[c1]):
        raise RuntimeError(f"v9 COL-EXCHANGE: rank {parallel.tp_rank} holds rows "
                           f"[{embed.row_start}, {embed.row_start + embed.rows}), not columns {c0}..{c1 - 1}")
    offsets = [int(v) for v in hasher.offsets[engram.layer_hash_index].tolist()]
    primes = [int(p) for per_ngram in hasher.primes[engram.layer_hash_index].tolist() for p in per_ngram]
    if offsets != list(bounds[:-1]) or primes != [bounds[c + 1] - bounds[c] for c in range(cols)]:
        raise RuntimeError("v9 COL-EXCHANGE: the hasher's column offsets / primes are not the table's bounds")
    return c0, c1


def _cols_counts(forward_batch, rows: int, min_rows: int = MIN_ROWS):
    """The DP counts when every rank takes a column path, else None. Each input is the same on every rank (global
    counts, extend-in-batch, padding mode, capture), so all ranks decide alike; king-v6 PEER-SMALL passes a lower
    ``min_rows`` for the collective-free bf16 PEER-READ lookup only (peer_read.lookup_bf16)."""
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
    if sum(counts) != rows or rows < min_rows:
        return None
    return counts


def _fp8_counts(engram, counts) -> bool:
    return engram is not None and all(_wkv_route(engram, c) is not None for c in counts)


def _exchange(self, indices, forward_batch, engram=None):
    import torch
    import torch.distributed as dist

    from sglang.kernels.ops.embeddings.engram_gather import engram_gather
    from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len
    from sglang.srt.runtime_context import get_parallel

    from bundle_v9 import exchange_fuse

    rows = get_global_dp_buffer_len()
    counts = _cols_counts(forward_batch, rows)
    if counts is None:
        return None
    parallel = get_parallel()
    mine = counts[parallel.attn_dp_rank]
    n_local = indices.shape[0]
    if n_local < mine:
        raise RuntimeError(f"v9 COL-EXCHANGE: rank {parallel.attn_dp_rank} holds {n_local} rows for its DP "
                           f"count {mine}")
    ids_global = exchange_fuse.id_gather(indices, counts, rows, forward_batch)
    c0, c1 = self._v4_cols
    width = (c1 - c0) * self.dim
    own = ids_global[:, c0:c1].contiguous()
    fp8 = _fp8_counts(engram, counts)
    if fp8:
        values = exchange_fuse.gather_pack(self, own)
    else:
        values = torch.empty((rows, width), dtype=torch.bfloat16, device=indices.device)
        engram_gather(*self._table_ptrs(), own.view(-1), values.view(-1, self.dim), self.dim, FP8_BLOCK_SIZE,
                      row_lo=self.row_start, row_hi=self.row_start + self.rows)
    received = values.new_empty((WORLD * mine, values.shape[1]))
    dist.all_to_all_single(received, values, [mine] * WORLD, counts, group=parallel.tp_group.device_group)
    if fp8 and exchange_fuse.CHECK:
        exchange_fuse.check_eq8(self, own, received, mine, n_local, counts, parallel.tp_group.device_group)
    return received, mine, n_local, width, fp8


def dp_sharded_lookup(self, indices, forward_batch):
    from bundle_v9 import prefill_fuse

    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len

    counts = _cols_counts(forward_batch, get_global_dp_buffer_len())
    if counts is not None and _fp8_counts(getattr(self, "_v4_engram", None), counts):
        raise RuntimeError("v9 EQ8: the bf16 lookup ran where every rank's count takes the FP8 exchange")
    peer = _peer_read.lookup_bf16(self, indices, forward_batch)
    if peer is not None:
        return peer.view(*indices.shape, self.dim)
    exchanged = _exchange(self, indices, forward_batch)
    if exchanged is None:
        return _stock_lookup()(self, indices, forward_batch)
    received, mine, n_local, width, _ = exchanged
    local = received.new_empty((n_local, WORLD * width))
    prefill_fuse.engram_place(received, local, mine)
    if n_local > mine:
        local[mine:].zero_()
    return local.view(*indices.shape, self.dim)


_STOCK_FORWARD = []


def _stock_forward():
    if not _STOCK_FORWARD:
        from sglang.srt.layers.engram import Engram

        fn = Engram.forward
        if fn.__module__ != "sglang.srt.layers.engram" or fn.__qualname__ != "Engram.forward":
            raise RuntimeError(f"v9 XQUANT: Engram.forward is not stock's ({fn.__module__}.{fn.__qualname__})")
        _STOCK_FORWARD.append(fn)
    return _STOCK_FORWARD[0]


def _wkv_route(engram, rows: int):
    from bundle_v9 import prefill_gemm

    entry = prefill_gemm._REG.get(engram.wkv.weight.data_ptr())
    if entry is None or rows <= 0 or not prefill_gemm._STOCK_LINEAR:
        return None
    route = prefill_gemm._route(entry, rows)
    return None if route is None else (entry, route)


def engram_forward(self, x, hash_ids, forward_batch=None, *, input_ids, cp_all_tokens=False):
    from sglang.srt.layers.engram import engram_gate

    from bundle_v9 import prefill_gemm, prefill_fuse

    stock = _stock_forward()
    if cp_all_tokens or forward_batch is None or forward_batch.engram_staged_rows is not None:
        return stock(self, x, hash_ids, forward_batch, input_ids=input_ids, cp_all_tokens=cp_all_tokens)
    served = _wkv_route(self, x.shape[0])
    if served is None:
        return stock(self, x, hash_ids, forward_batch, input_ids=input_ids, cp_all_tokens=cp_all_tokens)
    peer = _peer_read.lookup(self, hash_ids, forward_batch, x.shape[0])
    if peer is not None:
        entry, route = served
        kv = prefill_gemm._gemm(*route, *peer, entry[1], entry[2]).view(x.shape[0], entry[1].shape[0])
        image_ids = input_ids if self.image_token_id is not None else None
        return engram_gate(x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value, image_ids,
                           self.image_token_id)
    exchanged = _exchange(self.embed, hash_ids, forward_batch, engram=self)
    if exchanged is None:
        return stock(self, x, hash_ids, forward_batch, input_ids=input_ids, cp_all_tokens=cp_all_tokens)
    received, mine, n_local, width, fp8 = exchanged
    if WORLD * width != hash_ids.shape[-1] * self.embed.dim or n_local != x.shape[0]:
        raise RuntimeError(f"v9 XQUANT: exchanged width {WORLD * width} / rows {n_local} do not match the "
                           f"layer's {hash_ids.shape[-1] * self.embed.dim} / {x.shape[0]}")
    if fp8:
        from bundle_v9 import exchange_fuse

        q, s = exchange_fuse.place(received, mine, n_local)
    else:
        q, s = prefill_fuse.engram_quant(received, mine, n_local)
    entry, route = served
    kv = prefill_gemm._gemm(*route, q, s, entry[1], entry[2]).view(n_local, entry[1].shape[0])
    image_ids = input_ids if self.image_token_id is not None else None
    return engram_gate(x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value, image_ids, self.image_token_id)


def engram_view(engram, view, model):
    _stock_lookup()
    cols = check_domain(engram, model)
    engram_v = view(engram)
    embed_v = view(engram.embed)
    embed_v._v4_cols = cols
    embed_v._dp_sharded_lookup = MethodType(dp_sharded_lookup, embed_v)
    engram_v.embed = embed_v
    _stock_forward()
    engram_v.forward = MethodType(engram_forward, engram_v)
    embed_v._v4_engram = engram_v
    _peer_read.register(engram_v, model)
    return engram_v
