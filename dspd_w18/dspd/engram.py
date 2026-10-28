"""model.layers.*.engram (layers 1 and 14), levers EGP and N3: Engram lookups without the DP collectives.

Under DP attention the Engram table is sharded over the TP ranks, so the engine's lookup gathers every DP rank's hash
ids, reads the rows this rank's shard owns (zero elsewhere) and sums over the ranks: an id all-gather plus a
reduce-scatter of [ranks x tokens, 24 x 256] bf16 rows per call, in decode graphs (MAX_LEN padding) and eager prefill.

EGP: the engine already has a lookup that needs none of that, for tables it allocates in exportable memory
(SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS): every shard mapped into every rank, and `engram_gather_peer` reads each id from
the shard that owns it, so a rank looks up its own tokens only. This op maps the shards the engine did allocate: at a
module's first call (eager, every rank in lockstep) each rank exports its fp8 rows and e8m0 scales through CUDA IPC,
opens its peers' over NVLink, and builds the kernel's per-shard address table. The rows are the stock bytes and the
kernel's arithmetic is the stock dequantization, so the lookup is bit-identical (zeros as +0, as the zero-fill + sum
leaves them); a synthetic lookup over every shard is compared bitwise with the engine's sharded lookup before use and
the ranks agree on the verdict (MIN over the TP group). Once on, every call on every rank takes it (no per-call
fallback that could split a collective); decode graphs capture the gather kernel instead of the two collectives.

N3 (fallback when the mapping is refused): eager SUM_LEN prefill lookups by all_gatherv + reduce_scatterv over the
per-rank token counts instead of whole-buffer collectives (bit-identical, first use checked and agreed)."""
import torch
import torch.distributed as dist

from dspd import ekq, mxt
from dspd.common import pins_match, say
from sglang.kernels.ops.embeddings import engram_gather as EGK
from sglang.srt.layers import dp_attention as DPA
from sglang.srt.layers import engram as E
from sglang.srt.runtime_context import get_parallel

ENABLED = True
PEER = True
CHECK_ROWS = 512
PINS = {"_dp_sharded_lookup": "af2f31382bda6ce8", "EngramEmbedding.forward": "68f791628981ea39",
        "Engram.forward": "68cd442bb74b9e82", "Engram.embed_rows": "2b4088ed4ba1496c"}
PEER_PINS = {"engram_gather_peer": "c0df68dbac7865b8", "_engram_gather_peer_kernel": "b9038cf0b84cae9f",
             "_owned_rows": "47211cf1e657646c"}
_g = {"pins": None, "peer_pins": None, "verdict": None}


class _State:
    __slots__ = ("module", "stock", "peer", "tried")

    def __init__(self, module):
        self.module = module
        self.stock = type(module).forward
        self.peer = None
        self.tried = False


def prepare(module):
    mxt.install()
    return _State(module)


def _pins_ok():
    if _g["pins"] is None:
        _g["pins"] = pins_match("engram", PINS, {"_dp_sharded_lookup": E.EngramEmbedding._dp_sharded_lookup,
                                                 "EngramEmbedding.forward": E.EngramEmbedding.forward,
                                                 "Engram.forward": E.Engram.forward,
                                                 "Engram.embed_rows": E.Engram.embed_rows})
    return _g["pins"]


def _peer_pins_ok():
    if _g["peer_pins"] is None:
        _g["peer_pins"] = pins_match("engram-peer", PEER_PINS, {
            "engram_gather_peer": EGK.engram_gather_peer,
            "_engram_gather_peer_kernel": getattr(EGK._engram_gather_peer_kernel, "fn", None),
            "_owned_rows": E.EngramEmbedding._owned_rows})
    return _g["peer_pins"]


def _agree(ok, device=None):
    """MIN of a flag over the TP group on its gloo (CPU) group: a torch NCCL device-group collective would build a
    communicator the engine never uses (~0.95 GB per GPU, measured), out of the headroom the KV pool left."""
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=get_parallel().tp_group.cpu_group)
    return bool(flag.item())


# ---- EGP -------------------------------------------------------------------------------------------------------------

def _peer_static(engram):
    """Static, rank-uniform conditions for mapping this module's shards (config and load layout only)."""
    embed = engram.embed
    par = get_parallel()
    return (ENABLED and PEER and embed.peer is None and not embed._shared and embed.host_table is None
            and not embed.staged_by_scheduler and embed.tp_size > 1 and par.tp_size == embed.tp_size
            and par.attn_dp_size > 1 and par.attn_tp_size == 1 and getattr(par, "attn_cp_size", 1) == 1
            and embed.weight.is_cuda and embed.weight.dtype == torch.float8_e4m3fn and embed.weight.is_contiguous()
            and embed.scale.is_contiguous() and embed.dim % E.FP8_BLOCK_SIZE == 0)


def _open(handle, device_index):
    """A peer's storage mapped into this process on this rank's device (lazy peer access over NVLink)."""
    _, h, size, offset, rc_handle, rc_offset, ev_handle, _ = handle
    return torch.UntypedStorage._new_shared_cuda(device_index, h, size, offset, rc_handle, rc_offset, ev_handle, False)


def _connect(engram, cols):
    """Collective over the TP group: every rank runs every collective here whatever its local outcome (its local
    eligibility, pins, export, mapping and check all travel as data), so a local refusal can only refuse everywhere.
    Returns the address table, or None when any rank could not map or verify."""
    embed = engram.embed
    group = get_parallel().tp_group
    rank, world = group.rank_in_group, group.world_size
    device = torch.device("cuda", torch.cuda.current_device())
    mine, err = None, ""
    try:
        if not (_peer_static(engram) and _pins_ok() and _peer_pins_ok()):
            err = "not eligible here (layout or engine pins)"
        else:
            device = embed.weight.device
            mine = (embed.row_start, embed.rows, embed.weight.untyped_storage()._share_cuda_(),
                    embed.weight.storage_offset(), embed.scale.untyped_storage()._share_cuda_(),
                    embed.scale.storage_offset())
    except Exception as e:  # noqa: BLE001
        mine, err = None, f"export: {type(e).__name__}: {e}"
    meta = [None] * world
    dist.all_gather_object(meta, mine, group=group.cpu_group)
    table, keep = None, []
    if all(m is not None for m in meta):
        try:
            n_scales = embed.dim // E.FP8_BLOCK_SIZE
            starts, w_bias, s_bias = [], [], []
            for r, m in enumerate(meta):
                if r != rank and not torch.cuda.can_device_access_peer(device, m[2][0]):
                    raise RuntimeError(f"no peer access from cuda:{device.index} to cuda:{m[2][0]}")
            for r, (start, rows, wh, w_off, sh, s_off) in enumerate(meta):
                if r == rank:
                    wp, sp = embed.weight.data_ptr(), embed.scale.data_ptr()
                else:
                    ws, ss = _open(wh, device.index), _open(sh, device.index)
                    keep += [ws, ss]
                    wp, sp = ws.data_ptr() + w_off, ss.data_ptr() + s_off  # 1-byte elements
                starts.append(start)
                w_bias.append(wp - start * embed.dim)
                s_bias.append(sp - start * n_scales)
            if starts[0] != 0 or starts != sorted(starts):
                raise RuntimeError(f"unordered shard starts {starts}")
            padded = 1 << max(0, (world - 1).bit_length())
            for _ in range(world, padded):
                starts.append(torch.iinfo(torch.int64).max)
                w_bias.append(0)
                s_bias.append(0)
            total = meta[-1][0] + meta[-1][1]
            table = {"w_bias": torch.tensor(w_bias, dtype=torch.int64, device=device),
                     "s_bias": torch.tensor(s_bias, dtype=torch.int64, device=device),
                     "starts": torch.tensor(starts, dtype=torch.int64, device=device),
                     "keep": keep, "total": total, "meta": [(m[0], m[1]) for m in meta]}
        except Exception as e:  # noqa: BLE001
            err = f"map: {type(e).__name__}: {e}"
            table = None
    mapped = _agree(table is not None, device)
    ok = mapped and _verify(engram, table, cols)
    agreed = mapped and _agree(ok, device)
    say("engram", f"EGP layer {engram.layer_hash_index}: peer shards {'mapped and verified' if agreed else 'REFUSED'}"
                  f" (this rank {'ok' if ok else 'failed ' + err}; shards {table['meta'] if table else '-'})"
                  + ("" if agreed else ": engine lookup (N3 in eager prefill)"))
    return table if agreed else None


def _peer_lookup(embed, table, indices):
    flat = indices if indices.stride(-1) == 1 else indices.contiguous()
    out = torch.empty(*flat.shape, embed.dim, dtype=torch.bfloat16, device=indices.device)
    EGK.engram_gather_peer(flat, out, table["w_bias"], table["s_bias"], table["starts"], embed.dim, E.FP8_BLOCK_SIZE)
    return out


def _bits_sums(t, mask, weight):
    """Exact integer fingerprint of the bf16 bit patterns of ``t`` where ``mask``: sum, position-weighted sum and sum
    of squares of the int16 views (int64 arithmetic, no overflow at the check's size)."""
    b = t.view(torch.int16).to(torch.int64) * mask
    return [int(b.sum().item()), int((b * weight).sum().item()), int((b * b).sum().item())]


def _verify(engram, table, cols):
    """Synthetic lookup (same ids on every rank: every shard's first and last rows plus seeded random rows) through the
    peer table vs the engine's sharded lookup, compared bit for bit without moving rows: each rank fingerprints what
    the peer kernel read for every shard's ids, each owner fingerprints its own rows for those ids (the engine's
    `_owned_rows`, zeros as +0 as its zero-fill + sum leaves them), and the fingerprints are exchanged on the gloo
    group. Collective: a local failure still joins the exchange and reports False."""
    embed = engram.embed
    group = get_parallel().tp_group
    rank, world = group.rank_in_group, group.world_size
    device = embed.weight.device
    mine = None
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(20261027 + engram.layer_hash_index)
        ids = torch.randint(0, table["total"], (CHECK_ROWS, cols), device=device, generator=gen, dtype=torch.int64)
        edges = []
        for start, rows in table["meta"]:
            edges += [start, start + rows - 1]
        edge = torch.tensor(edges, dtype=torch.int64, device=device)
        ids[0, : min(cols, edge.numel())] = edge[: min(cols, edge.numel())]
        starts = torch.tensor([m[0] for m in table["meta"]], dtype=torch.int64, device=device)
        owner = (torch.searchsorted(starts, ids, right=True) - 1).unsqueeze(-1)
        got = _peer_lookup(embed, table, ids)
        ref = embed._owned_rows(ids).contiguous() + 0.0   # -0 -> +0, as the sum over shards leaves it
        n = got.numel()
        weight = (torch.arange(n, device=device, dtype=torch.int64) % 1000003 + 1).view(got.shape)
        mine = {"got": [_bits_sums(got, (owner == r), weight) for r in range(world)],
                "ref": _bits_sums(ref, (owner == rank), weight),
                "unowned_zero": bool(((ref.view(torch.int16) != 0) & (owner != rank)).sum().item() == 0)}
        del ids, got, ref, weight, owner
    except Exception as e:  # noqa: BLE001
        say("engram", f"EGP check could not run here ({type(e).__name__}: {e})")
        mine = None
    allv = [None] * world
    dist.all_gather_object(allv, mine, group=group.cpu_group)
    torch.cuda.empty_cache()
    if any(v is None or not v["unowned_zero"] for v in allv):
        return False
    return all(allv[rank]["got"][r] == allv[r]["ref"] for r in range(world))


# ---- N3 (fallback) ---------------------------------------------------------------------------------------------------

def _sizes(engram, hash_ids, forward_batch, cp_all_tokens):
    """Per-DP-rank token counts when this call takes the variable-length path, else None."""
    if not ENABLED or _g["verdict"] is False or forward_batch is None or cp_all_tokens:
        return None
    padding = forward_batch.dp_padding_mode
    if padding is None or not padding.is_sum_len() or getattr(forward_batch, "engram_staged_rows", None) is not None:
        return None
    embed = engram.embed
    if embed.peer is not None or embed._shared or embed.tp_size <= 1 or E.is_in_breakable_cuda_graph():
        return None
    par = get_parallel()
    if not (par.attn_dp_size > 1 and par.attn_tp_size == 1 and par.tp_size == par.attn_dp_size
            and getattr(par, "attn_cp_size", 1) == 1):
        return None
    if torch.cuda.is_current_stream_capturing() or not _pins_ok():
        return None
    sizes = [int(v) for v in DPA.get_dp_global_num_tokens()]
    total = sum(sizes)
    # global facts only (every rank must take the same branch: the two paths issue different collectives)
    if len(sizes) != par.tp_size or total == 0 or DPA.get_global_dp_buffer_len() < total:
        return None
    if hash_ids.shape[0] != sizes[par.attn_dp_rank]:
        raise RuntimeError(f"dspd engram: local rows {hash_ids.shape[0]} != this rank's DP count {sizes}")
    return sizes


def _lookup(embed, indices, sizes):
    group = get_parallel().tp_group
    total = sum(sizes)
    ids = indices.new_empty((total, *indices.shape[1:]))
    group.all_gatherv(indices.contiguous(), sizes=sizes, output=ids)
    values = embed._owned_rows(ids).view(total, -1)
    local = values.new_empty((indices.shape[0], values.shape[1]))
    group.reduce_scatterv(values, output=local, sizes=sizes)
    return local.view(*indices.shape, embed.dim)


def _gate(engram, x, emb, input_ids):
    flat = emb.flatten(-2)
    image_ids = input_ids if engram.image_token_id is not None else None
    out = ekq.out(engram, x, flat, image_ids)   # EKQ / EKQ-L on large eager prefill calls (None = stock projection)
    if out is not None:
        return out
    with mxt.scope():                      # MXT-E: the wkv projection takes MXT's checked CuTe-DSL tactics
        kv, _ = engram.wkv(flat)
    return E.engram_gate(x, kv, engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value, image_ids,
                         engram.image_token_id)


def forward(state, x, hash_ids, forward_batch=None, *, input_ids, cp_all_tokens=False):
    engram = state.module
    if not state.tried and not torch.cuda.is_current_stream_capturing():
        # first eager call of this module: the same call on every rank (lookups are lockstep under DP attention)
        state.tried = True
        if ENABLED and PEER:
            state.peer = _connect(engram, hash_ids.shape[-1])
        ekq.warm(engram)                   # EKQ weights + kernel compile before any scored window (rank-local, no collective)
    if state.peer is not None and not cp_all_tokens:
        emb = _peer_lookup(engram.embed, state.peer, hash_ids)
        say("engram", "EGP peer-shard lookup on" + (" (graph capture)" if torch.cuda.is_current_stream_capturing()
                                                     else ""))
        if x.shape[0] == 0:
            return x
        return _gate(engram, x, emb, input_ids)
    sizes = _sizes(engram, hash_ids, forward_batch, cp_all_tokens)
    if sizes is None:
        return state.stock(engram, x, hash_ids, forward_batch, input_ids=input_ids, cp_all_tokens=cp_all_tokens)
    emb = _lookup(engram.embed, hash_ids, sizes)
    if _g["verdict"] is None:
        ref = engram.embed(hash_ids, forward_batch, cp_all_tokens=False)
        same = ref.shape == emb.shape and torch.equal(ref, emb)
        _g["verdict"] = _agree(same, emb.device)
        say("engram", f"N3 first-use check {'OK' if _g['verdict'] else 'FAILED'} at {sum(sizes)} global rows "
                      f"(local {hash_ids.shape[0]}, this rank {'equal' if same else 'DIFFERENT'})"
                      + ("" if _g["verdict"] else ": lever off on every rank, stock lookup from now on"))
        if not _g["verdict"]:
            emb = ref
    else:
        say("engram", "N3 reduce-scatterv lookup on (eager SUM_LEN extend)")
    if x.shape[0] == 0:
        return x
    return _gate(engram, x, emb, input_ids)
