
from __future__ import annotations

from types import MethodType

_STOCK = []


def _stock_lookup():
    if not _STOCK:
        from sglang.srt.layers.engram import EngramEmbedding

        fn = EngramEmbedding._dp_sharded_lookup
        if fn.__module__ != "sglang.srt.layers.engram" or fn.__qualname__ != "EngramEmbedding._dp_sharded_lookup":
            raise RuntimeError(f"v9 ROW-EXCHANGE: EngramEmbedding._dp_sharded_lookup is not stock's "
                               f"({fn.__module__}.{fn.__qualname__})")
        _STOCK.append(fn)
    return _STOCK[0]


def check_domain(engram) -> None:
    from sglang.srt.layers.engram import Engram, EngramEmbedding
    from sglang.srt.runtime_context import get_parallel

    if type(engram) is not Engram or type(engram.embed) is not EngramEmbedding:
        raise TypeError(f"v9 ROW-EXCHANGE requires stock Engram / EngramEmbedding, got "
                        f"{type(engram).__name__} / {type(engram.embed).__name__}")
    embed = engram.embed
    if embed.peer is not None or embed.host_table is not None:
        raise RuntimeError("v9 ROW-EXCHANGE requires device-resident shards (no peer or host table)")
    parallel = get_parallel()
    facts = (parallel.tp_size, parallel.attn_dp_size, parallel.attn_tp_size, parallel.attn_cp_size)
    if facts != (4, 4, 1, 1) or embed.tp_size != 4 or parallel.tp_rank != parallel.attn_dp_rank:
        raise RuntimeError(f"v9 ROW-EXCHANGE requires TP4 = DP-attention 4 (attention TP 1, no CP) with TP rank "
                           f"= DP rank, got (tp, dp, attn_tp, attn_cp) = {facts}")


def _a2a_counts(forward_batch, rows: int):
    counts = _a2a_shared(forward_batch, rows)
    if counts is None or 2 * max(counts) >= rows:
        return None
    return counts


def _a2a_shared(forward_batch, rows: int):
    import torch

    from sglang.srt.layers.dp_attention import is_dp_gatherv_active

    if forward_batch is None or torch.cuda.is_current_stream_capturing():
        return None
    padding = forward_batch.dp_padding_mode
    if not forward_batch.is_extend_in_batch or padding is None or not padding.is_sum_len() or is_dp_gatherv_active():
        return None
    counts = forward_batch.global_num_tokens_cpu
    if counts is None or len(counts) != 4:
        return None
    counts = [int(c) for c in counts]
    if sum(counts) != rows:
        return None
    return counts


def dp_sharded_lookup(self, indices, forward_batch):
    import torch
    import torch.distributed as dist

    from sglang.srt.layers.dp_attention import dp_gather_replicate, get_global_dp_buffer_len
    from sglang.srt.runtime_context import get_parallel

    rows = get_global_dp_buffer_len()
    counts = _a2a_counts(forward_batch, rows)
    if counts is None:
        return _stock_lookup()(self, indices, forward_batch)
    ids_global = torch.empty((rows, *indices.shape[1:]), dtype=indices.dtype, device=indices.device)
    dp_gather_replicate(ids_global, indices.clone(), forward_batch)
    values = self._owned_rows(ids_global).view(rows, -1)
    parallel = get_parallel()
    mine = counts[parallel.attn_dp_rank]
    received = values.new_empty((len(counts) * mine, values.shape[1]))
    dist.all_to_all_single(received, values, [mine] * len(counts), counts, group=parallel.tp_group.device_group)
    local = received.view(len(counts), mine, values.shape[1]).sum(dim=0)
    n_local = indices.shape[0]
    if n_local != mine:
        if n_local < mine:
            raise RuntimeError(f"v9 ROW-EXCHANGE: rank {parallel.attn_dp_rank} holds {n_local} rows for "
                               f"its DP count {mine}")
        padded = local.new_zeros((n_local, local.shape[1]))
        padded[:mine] = local
        local = padded
    return local.view(*indices.shape, self.dim)


def engram_view(engram, view):
    _stock_lookup()
    check_domain(engram)
    engram_v = view(engram)
    embed_v = view(engram.embed)
    embed_v._dp_sharded_lookup = MethodType(dp_sharded_lookup, embed_v)
    engram_v.embed = embed_v
    return engram_v


def rehome_spans(plan) -> list:
    starts = [sum(plan.counts[:r]) for r in range(len(plan.counts))]
    spans = []
    for r in range(len(plan.counts)):
        spans.append((starts[r], starts[r] + plan.kept(r)))
        at = plan.kept(r)
        for mv in sorted(plan.incoming(r), key=lambda mv: mv.at):
            if mv.at != at:
                raise RuntimeError(f"v9-rehome ENGRAM: rank {r} receives rows at {mv.at}, expected {at}")
            spans.append((starts[mv.owner] + mv.lo, starts[mv.owner] + mv.hi))
            at += mv.hi - mv.lo
    return spans


def rehome_lookup(embed, indices, forward_batch, plan, rank: int):
    import torch

    from sglang.srt.layers.dp_attention import dp_gather_replicate, get_global_dp_buffer_len

    rows = get_global_dp_buffer_len()
    counts = _a2a_shared(forward_batch, rows)
    if counts is None or tuple(counts) != tuple(plan.counts) or indices.shape[0] != plan.counts[rank]:
        raise RuntimeError(f"v9-rehome ENGRAM: shared counts {counts} / local rows {indices.shape[0]} do not match the "
                           f"plan {plan.counts} on rank {rank}")
    ids_global = torch.empty((rows, *indices.shape[1:]), dtype=indices.dtype, device=indices.device)
    dp_gather_replicate(ids_global, indices.clone(), forward_batch)
    values, tl = rehome_gather(embed, ids_global, plan)
    return rehome_reduce(embed, values, tl, rank).view(tl[rank], *indices.shape[1:], embed.dim)


# The king's value stages, split out of rehome_lookup unchanged so that peer_lookup.prepare_rh proves PEER-RH
# against this very code: the dense gather of every global row, then the all-reduce or all-to-all + 4-way sum.
def rehome_gather(embed, ids_global, plan):
    import torch

    rows = ids_global.shape[0]
    values = embed._owned_rows(torch.cat([ids_global[a:b] for a, b in rehome_spans(plan)])).view(rows, -1)
    return values, [plan.rows(r) for r in range(len(plan.counts))]


def rehome_reduce(embed, values, tl, rank: int):
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    rows, mine = values.shape[0], tl[rank]
    if 2 * max(tl) >= rows:
        start = sum(tl[:rank])
        return embed._reduce_owned_rows(values)[start:start + mine]
    received = values.new_empty((len(tl) * mine, values.shape[1]))
    dist.all_to_all_single(received, values, [mine] * len(tl), tl, group=get_parallel().tp_group.device_group)
    return received.view(len(tl), mine, values.shape[1]).sum(dim=0)
