from __future__ import annotations

import sys
import time

import torch
import torch.distributed as dist

from bundle_v9.dec.placement_table import LOCAL, RANKS, SLOTS

_REQUIRED = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv",
             "mega_w13_weight_scale_utccp", "mega_w2_weight_scale_utccp")

_pending: list = []
_placed: set = set()


def _ep_group():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().moe_ep_group


def _slice_span(t: torch.Tensor) -> int:
    return 1 + sum((s - 1) * st for s, st in zip(t.shape[1:], t.stride()[1:]) if s > 0)


def expert_tensors(experts, n: int, where: str) -> dict:
    tensors = dict(experts.named_parameters(recurse=False))
    tensors.update(experts.named_buffers(recurse=False))
    per_local = {k: t for k, t in tensors.items() if t.dim() >= 1 and t.shape[0] == LOCAL}
    per_global = sorted(k for k, t in tensors.items() if t.dim() >= 1 and t.shape[0] == n and k not in per_local)
    missing = [k for k in _REQUIRED if k not in per_local]
    storages = {}
    for k, t in per_local.items():
        storages.setdefault(t.untyped_storage().data_ptr(), []).append(k)
    shared = [v for v in storages.values() if len(v) > 1]
    tangled = [k for k, t in per_local.items() if t.stride(0) < _slice_span(t)]
    if per_global or missing or shared or tangled:
        raise RuntimeError(f"bundle-v9 PLACEMENT: {where} experts: global-sized {per_global}, missing per-expert "
                           f"{missing}, shared storage {shared}, overlapping expert slices {tangled}")
    return per_local


def _table() -> dict:
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "decode":
        from bundle_v9.dec.decode_placement import SLOTS_DECODE

        return SLOTS_DECODE
    return SLOTS


def check(model) -> list:
    group = _ep_group()
    rank, world = group.rank_in_group, group.world_size
    if world != RANKS:
        raise RuntimeError(f"bundle-v9 PLACEMENT needs EP{RANKS}, got EP{world}")
    plan = []
    table = _table()
    for layer in model.layers:
        moe = layer.mlp
        slots = table.get(getattr(moe, "layer_id", None))
        if slots is None:
            continue
        experts, gate = moe.experts, moe.gate
        n = gate.weight.shape[0]
        if (n != LOCAL * RANKS or experts.num_local_experts != LOCAL or experts.moe_ep_rank != rank
                or sorted(slots) != list(range(n))):
            raise RuntimeError(f"bundle-v9 PLACEMENT: layer {moe.layer_id} needs {LOCAL * RANKS} experts as "
                               f"{RANKS}x{LOCAL} (gate {n}, local {experts.num_local_experts}, ep rank "
                               f"{experts.moe_ep_rank} vs {rank})")
        per_local = expert_tensors(experts, n, f"layer {moe.layer_id}")
        for t in (gate.weight, gate.e_score_correction_bias, gate.e_score_correction_bias_vl):
            if t is None or t.shape[0] != n:
                raise RuntimeError(f"bundle-v9 PLACEMENT: layer {moe.layer_id} gate tensor shape mismatch")
        plan.append((moe, slots, per_local))
    return plan


def defer(model) -> None:
    check(model)
    _pending.append(model)


def settle() -> None:
    if _pending and not torch.cuda.is_current_stream_capturing():
        start = time.monotonic()
        placed = place(_pending.pop())
        print(f"bundle-v9 PLACEMENT: {placed} layers placed in {time.monotonic() - start:.1f} s", file=sys.stderr, flush=True)


def exchange_plan(slots, rank: int, world: int, local: int):
    send = [[e % local for j, e in enumerate(slots[d * local:(d + 1) * local]) if e // local == rank
             and e != d * local + j] for d in range(world)]
    recv = [[j for j, e in enumerate(slots[rank * local:(rank + 1) * local]) if e // local == s
             and e != rank * local + j] for s in range(world)]
    return send, recv


_SAME_WIDTH_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _move(t: torch.Tensor, send, recv, group) -> None:
    raw = t.data.view(_SAME_WIDTH_INT[t.element_size()])
    dev = raw.device
    src = torch.tensor([i for part in send for i in part], dtype=torch.long, device=dev)
    dst = torch.tensor([j for part in recv for j in part], dtype=torch.long, device=dev)
    out = raw.index_select(0, src).contiguous()
    got = torch.empty((len(dst),) + tuple(raw.shape[1:]), dtype=raw.dtype, device=dev)
    dist.all_to_all_single(got.view(torch.uint8), out.view(torch.uint8), output_split_sizes=[len(p) for p in recv],
                           input_split_sizes=[len(p) for p in send], group=group)
    raw.index_copy_(0, dst, got)


def place(model) -> int:
    group = _ep_group()
    rank, world = group.rank_in_group, group.world_size
    placed = 0
    with torch.no_grad():
        for moe, slots, per_local in check(model):
            if id(moe) in _placed:
                continue
            send, recv = exchange_plan(slots, rank, world, LOCAL)
            for name in sorted(per_local):
                _move(per_local[name], send, recv, group.device_group)
            perm = torch.tensor(slots, dtype=torch.long, device=moe.gate.weight.device)
            for t in (moe.gate.weight, moe.gate.e_score_correction_bias, moe.gate.e_score_correction_bias_vl):
                t.copy_(t.index_select(0, perm))
            _placed.add(id(moe))
            placed += 1
    torch.cuda.synchronize()
    dist.barrier(group=group.device_group)
    return placed
