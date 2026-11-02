"""dsv41-pd002 PLACEMENT: each MoE layer's 384 routed experts placed across the four EP ranks by placement_table.SLOTS,
fitted on prefill routing recorded under the policy-20 replay (generator/pd002/placement/fit_placement.py), instead of
stock's experts r*96..r*96+95 on rank r.

A prefill MegaMoE call ends with its busiest rank, and popular experts are not spread evenly by expert id: on the
recorded big prefill forwards the busiest rank carried 1.29x the mean routed + shared work, 1.05x under the fitted
table (held-out forwards; ~5% of the experts change rank). Decode streams every local expert either way.

The router emits slot ids directly: the gate's weight rows and both score-correction biases (text and vision) are
permuted with the experts, so the logits, the sqrtsoftplus scores, the biased top-k selection and the routing weights
are stock's, only named by slot; MegaMoE's combine sums each token's slots in top-k order on its home rank, so the
output is bitwise stock. Every per-expert tensor of the experts module moves with its expert, in place (captured
graphs and MegaMoE's weight tuples keep their addresses), over one all-to-all on the EP group per tensor. The audit's
stock reference runs on the same permuted module: the same function, relabeled.

cacheon prepares a node inside its first dispatched call, after that call's stock reference ran on the module as
loaded (E34, 2026-10-11: placing in prepare cost the first audited call 25% of its rows). prepare therefore only
validates and records; settle() places at the end of the first eager (non-capturing) forward, so the first call runs
both arms on the loaded layout and every later call both on the placed one."""

from __future__ import annotations

import torch
import torch.distributed as dist

from dsv41_pd001.placement_table import LOCAL, RANKS, SLOTS

# The sealed fp8xfp4 MegaMoE experts module's per-expert tensors (mega_moe.build_mega_moe_experts_weights): the
# interleaved w13 and its SF, w2 and its SF, and the UTCCP SF copies registered for EPLB. Each must move.
_REQUIRED = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv",
             "mega_w13_weight_scale_utccp", "mega_w2_weight_scale_utccp")

_pending: list = []
_placed: set = set()


def _ep_group():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().moe_ep_group


def _slice_span(t: torch.Tensor) -> int:
    """Elements one dim-0 slice spans in storage (inner dims' extent)."""
    return 1 + sum((s - 1) * st for s, st in zip(t.shape[1:], t.stride()[1:]) if s > 0)


def expert_tensors(experts, n: int, where: str) -> dict:
    """The experts module's per-expert tensors, validated for an in-place slice move: every _REQUIRED name present, no
    global-sized (id-indexed) tensor, each tensor its own storage, and expert slices disjoint in storage (dim 0
    outermost). DeepGEMM's MN-major scale layouts are non-contiguous; that is fine, overlap or sharing is not."""
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
        raise RuntimeError(f"dsv41-pd002 PLACEMENT: {where} experts: global-sized {per_global}, missing per-expert "
                           f"{missing}, shared storage {shared}, overlapping expert slices {tangled}")
    return per_local


def check(model) -> list:
    """Validate every layer the table names; returns [(moe, slots)]. Raises on any mismatch (no partial placement)."""
    group = _ep_group()
    rank, world = group.rank_in_group, group.world_size
    if world != RANKS:
        raise RuntimeError(f"dsv41-pd002 PLACEMENT needs EP{RANKS}, got EP{world}")
    plan = []
    for layer in model.layers:
        moe = layer.mlp
        slots = SLOTS.get(getattr(moe, "layer_id", None))
        if slots is None:
            continue
        experts, gate = moe.experts, moe.gate
        n = gate.weight.shape[0]
        if (n != LOCAL * RANKS or experts.num_local_experts != LOCAL or experts.moe_ep_rank != rank
                or sorted(slots) != list(range(n))):
            raise RuntimeError(f"dsv41-pd002 PLACEMENT: layer {moe.layer_id} needs {LOCAL * RANKS} experts as "
                               f"{RANKS}x{LOCAL} (gate {n}, local {experts.num_local_experts}, ep rank "
                               f"{experts.moe_ep_rank} vs {rank})")
        per_local = expert_tensors(experts, n, f"layer {moe.layer_id}")
        for t in (gate.weight, gate.e_score_correction_bias, gate.e_score_correction_bias_vl):
            if t is None or t.shape[0] != n:
                raise RuntimeError(f"dsv41-pd002 PLACEMENT: layer {moe.layer_id} gate tensor shape mismatch")
        plan.append((moe, slots, per_local))
    return plan


def defer(model) -> None:
    """prepare: validate now, place at the end of the first eager forward (settle)."""
    check(model)
    _pending.append(model)


def settle() -> None:
    """The model forward's last step: place once, in the first eager (non-capturing) call."""
    if _pending and not torch.cuda.is_current_stream_capturing():
        place(_pending.pop())


def exchange_plan(slots, rank: int, world: int, local: int):
    """Only the slots whose expert changes: send[d] = my local indices (stock layout, expert e at rank e // local, index
    e % local) that rank d's changed slots take, in d's slot order; recv[s] = my changed local slots that rank s fills,
    in my slot order. Kept experts sit at their stock slot (fit_placement swaps slots) and do not move."""
    send = [[e % local for j, e in enumerate(slots[d * local:(d + 1) * local]) if e // local == rank
             and e != d * local + j] for d in range(world)]
    recv = [[j for j, e in enumerate(slots[rank * local:(rank + 1) * local]) if e // local == s
             and e != rank * local + j] for s in range(world)]
    return send, recv


_SAME_WIDTH_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _move(t: torch.Tensor, send, recv, group) -> None:
    """Move whole expert slices of t in place. A same-width integer view keeps t's strides (index_select / index_copy_
    have no FP8 kernels; DeepGEMM's scale tensors are strided); NCCL carries the contiguous slices as bytes."""
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
    """Move every tabled layer's experts and permute its gate; returns the number of layers placed."""
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
                t.copy_(t.index_select(0, perm))  # in place on the parameter (version bump for TopKConfig caches)
            _placed.add(id(moe))
            placed += 1
    torch.cuda.synchronize()
    dist.barrier(group=group.device_group)
    return placed
