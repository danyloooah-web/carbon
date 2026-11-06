from __future__ import annotations

import contextlib
import copy
import logging
import sys
from functools import partial
from types import MethodType
from typing import NamedTuple, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

WORLD, LAYERS = 4, 20
ALIGN = 128
OWNER_MIN, MOVE_MIN = 8192, 2048
RECV_CAP = 24704
Q_K, Q_SF, O_N, KV_DIM = 1280, 40, 5120, 512
WIDTHS = (128, 640)
HEAD_CAP, SWA_CAP = 1_310_720, 65_536
CH_Q, CH_KV, CH_RET, CH_PROBE = 0, 1, 2, 3
TIMEOUT_MS = 60_000
SWA_WINDOW = 128

STAY_NS, WORK_NS, WIRE_NS, FIX_NS, MOE_NS = 115, 165, 20, 150_000, 57
PRE_NS, WQB_NS = 48, 29
ATTN_NS = WORK_NS - WQB_NS
SAVE_MIN_PCT = 5
TRACE = False


class Block(NamedTuple):
    src: int
    dst: int
    first: int
    last: int


def _ceil_align(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def _ready(rows: int, moved: int) -> int:
    return PRE_NS * rows + WQB_NS * (rows - moved)


def _sizes(rows: int, helpers: Sequence[int]) -> list:
    cap = rows - MOVE_MIN
    post = STAY_NS - PRE_NS
    own = [(STAY_NS + WORK_NS) * m for m in helpers]

    def takes(level, moved):
        start = _ready(rows, moved)
        return [max(0, (level - max(o, start) - FIX_NS - post * m) // (ATTN_NS + WIRE_NS)) for o, m in zip(own, helpers)]

    def total(level):
        moved = min(sum(takes(level, 0)), cap)
        return min(sum(takes(level, moved)), cap)

    def owner(moved):
        return _ready(rows, moved) + ATTN_NS * (rows - moved) + WIRE_NS * moved + FIX_NS * len(helpers) + post * rows

    lo, hi = 0, owner(0)
    while lo < hi:
        level = (lo + hi) // 2
        if owner(total(level)) <= level:
            hi = level
        else:
            lo = level + 1
    want = total(lo)
    lo, hi = 0, lo
    while lo < hi:
        level = (lo + hi) // 2
        if total(level) >= want:
            hi = level
        else:
            lo = level + 1
    sizes = [min(k // ALIGN * ALIGN, RECV_CAP) for k in takes(lo, want)]
    extra = sum(sizes) - cap
    for i in range(len(sizes) - 1, -1, -1):
        if extra <= 0:
            break
        cut = min(sizes[i], -(-extra // ALIGN) * ALIGN)
        sizes[i] -= cut
        extra -= cut
    return [k if k >= MOVE_MIN else 0 for k in sizes]


def _blocks_of(src: int, rows: int, dsts: Sequence[int], counts: Sequence[int]) -> Tuple[Block, ...]:
    sizes = _sizes(rows, [counts[d] for d in dsts])
    moved = sum(sizes)
    if not moved:
        return ()
    kept = _ceil_align(rows - moved)
    if kept < MOVE_MIN:
        return ()
    out, first = [], kept
    live = [(d, k) for d, k in zip(dsts, sizes) if k]
    for i, (dst, k) in enumerate(live):
        last = rows if i == len(live) - 1 else first + k
        if last - first < MOVE_MIN:
            return ()
        out.append(Block(src, dst, first, last))
        first = last
    return tuple(out)


def _makespan(counts: Sequence[int], blocks: Sequence[Block]) -> int:
    cost = [(STAY_NS + WORK_NS) * c for c in counts]
    moved = {}
    for b in blocks:
        moved[b.src] = moved.get(b.src, 0) + b.last - b.first
    for src, k in moved.items():
        n = sum(1 for b in blocks if b.src == src)
        cost[src] = _ready(counts[src], k) + ATTN_NS * (counts[src] - k) + WIRE_NS * k + FIX_NS * n + (STAY_NS - PRE_NS) * counts[src]
    for b in blocks:
        k = b.last - b.first
        start = max((STAY_NS + WORK_NS) * counts[b.dst], _ready(counts[b.src], moved[b.src]))
        cost[b.dst] = start + (ATTN_NS + WIRE_NS) * k + FIX_NS + (STAY_NS - PRE_NS) * counts[b.dst]
    return max(cost)


def plan(counts: Sequence[int]) -> Tuple[Block, ...]:
    if len(counts) != WORLD:
        return ()
    order = sorted(range(WORLD), key=lambda r: (-counts[r], r))
    top, second, third, last = order
    if counts[top] < OWNER_MIN:
        return ()
    options = [_blocks_of(top, counts[top], (last,), counts),
               _blocks_of(top, counts[top], (last, third), counts),
               _blocks_of(top, counts[top], (last, third, second), counts)]
    if counts[second] >= OWNER_MIN:
        options.append(_blocks_of(top, counts[top], (last,), counts) + _blocks_of(second, counts[second], (third,), counts))
    base = _makespan(counts, ())
    best, score = (), base
    for option in options:
        if option:
            value = _makespan(counts, option)
            if value < score or (value == score and len(option) < len(best)):
                best, score = option, value
    layer = base + MOE_NS * sum(counts)
    return best if (base - score) * 100 >= SAVE_MIN_PCT * layer else ()


class _Forward:

    __slots__ = ("out", "into", "n", "kept", "src_kept", "active", "sent_head", "width")

    def __init__(self, blocks: Sequence[Block], rank: int, counts: Sequence[int]):
        self.out = tuple(b for b in blocks if b.src == rank)
        into = [b for b in blocks if b.dst == rank]
        if len(into) > 1 or (into and self.out):
            raise RuntimeError(f"bundle-v9 ROW_BALANCE: rank {rank} holds an impossible plan {blocks}")
        self.into = into[0] if into else None
        self.n = counts[rank]
        self.kept = min(b.first for b in self.out) if self.out else self.n
        self.src_kept = min(b.first for b in blocks if b.src == self.into.src) if self.into else 0
        self.active = False
        self.sent_head = None
        self.width = 0


_FWD: list = [None]
_LINK: list = []
_INNER: list = [None]
_R_MIN: list = [1]
_STOCK_WO_B: dict = {}


class _Link:

    def __init__(self, group, rank: int, device):
        import torch.distributed._symmetric_memory as symm_mem

        spans, total = [], 0
        for name, nbytes in (("q", RECV_CAP * Q_K), ("qsf", RECV_CAP * Q_SF), ("idx", RECV_CAP * WIDTHS[-1] * 4),
                             ("len", RECV_CAP * 4), ("pos", RECV_CAP * 4), ("ret", 3 * RECV_CAP * O_N * 2),
                             ("kv", (HEAD_CAP + SWA_CAP) * KV_DIM * 2)):
            spans.append((name, total, nbytes))
            total = (total + nbytes + 255) // 256 * 256
        self.spans = {name: (off, nbytes) for name, off, nbytes in spans}
        self.group, self.rank, self.device = group, rank, device
        self.buffer = symm_mem.empty(total, dtype=torch.uint8, device=device)
        self.handle = symm_mem.rendezvous(self.buffer, group=group.device_group)
        self.total = total
        self.remotes = [self.handle.get_buffer(r, (total,), torch.uint8, 0) for r in range(WORLD)]

    def _span(self, name: str, first: int, nbytes: int) -> Tuple[int, int]:
        off, cap = self.spans[name]
        if first + nbytes > cap:
            raise RuntimeError(f"bundle-v9 ROW_BALANCE: {first + nbytes} bytes past inbox {name} ({cap})")
        return off + first, nbytes

    def local(self, name: str, first: int, nbytes: int) -> torch.Tensor:
        off, nbytes = self._span(name, first, nbytes)
        return self.buffer[off:off + nbytes]

    def peer(self, rank: int, name: str, first: int, nbytes: int) -> torch.Tensor:
        off, nbytes = self._span(name, first, nbytes)
        return self.remotes[rank][off:off + nbytes]


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


def _send(link: _Link, dst: int, name: str, src: torch.Tensor, first: int = 0) -> None:
    raw = _bytes(src.contiguous())
    link.peer(dst, name, first, raw.numel()).copy_(raw, non_blocking=True)


def _view(module):
    view = copy.copy(module)
    view._parameters = module._parameters.copy()
    view._buffers = module._buffers.copy()
    view._modules = module._modules.copy()
    return view


def _wq_b_forward(stock, input_, *args, **kwargs):
    fwd = _FWD[0]
    if fwd is None or not fwd.active:
        return stock(input_, *args, **kwargs)
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    if type(input_) is not Mxfp8SwizzledInput:
        raise RuntimeError(f"bundle-v9 ROW_BALANCE: wq_b received {type(input_).__name__}, not the swizzled MXFP8 pair")
    q8, sf = input_
    if q8.shape[0] != fwd.n or q8.dim() != 2 or q8.shape[1] != Q_K:
        raise RuntimeError(f"bundle-v9 ROW_BALANCE: wq_b input {tuple(q8.shape)} on a rank holding {fwd.n} rows")
    link = _LINK[0]
    for b in fwd.out:
        _send(link, b.dst, "q", q8[b.first:b.last])
        _send(link, b.dst, "qsf", sf[b.first * Q_SF:_ceil_align(b.last) * Q_SF])
        link.handle.put_signal(b.dst, CH_Q, TIMEOUT_MS)
    return stock(Mxfp8SwizzledInput(q8[:fwd.kept], sf[:fwd.kept * Q_SF]), *args, **kwargs)


def _split_prefill(q, kv, indices, positions, cos_sin_cache, softmax_scale, *, attn_sink=None, topk_length=None,
                   enable_q_norm=False, rms_norm_eps=0.0, use_legacy=True):
    inner = _INNER[0]
    fwd = _FWD[0]
    if fwd is None or not fwd.active:
        return inner(q, kv, indices, positions, cos_sin_cache, softmax_scale, attn_sink=attn_sink,
                     topk_length=topk_length, enable_q_norm=enable_q_norm, rms_norm_eps=rms_norm_eps,
                     use_legacy=use_legacy)
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    metadata = backend.forward_metadata
    all_positions = metadata.core_attn_metadata.positions
    if (q.shape[0] != fwd.kept or indices.shape[0] < fwd.n or indices.shape[2] != fwd.width or kv.dtype != torch.bfloat16
            or indices.dtype != torch.int32 or topk_length.dtype != torch.int32 or all_positions.dtype != torch.int32):
        raise RuntimeError(f"bundle-v9 ROW_BALANCE: attention q {tuple(q.shape)} indices {tuple(indices.shape)} {indices.dtype} "
                           f"lengths {topk_length.dtype} positions {all_positions.dtype} kv {kv.dtype}, holding {fwd.n} "
                           f"rows, keeping {fwd.kept}, expecting width {fwd.width}")
    swa = metadata.sparse_prefill_cache.swa_token_ids.shape[0]
    total = kv.shape[0]
    head = total - swa
    if head > HEAD_CAP or swa > SWA_CAP:
        raise RuntimeError(f"bundle-v9 ROW_BALANCE: workspace head {head} / {HEAD_CAP}, SWA {swa} / {SWA_CAP} past the gathered bound")
    source = backend.sparse_prefill_kv_buffer._head_source
    send_head = head > 0 and (source is None or source != fwd.sent_head)
    link = _LINK[0]
    for b in fwd.out:
        _send(link, b.dst, "idx", indices[b.first:b.last, 0])
        _send(link, b.dst, "len", topk_length[b.first:b.last])
        _send(link, b.dst, "pos", all_positions[b.first:b.last])
        if send_head:
            _send(link, b.dst, "kv", kv[:head])
        _send(link, b.dst, "kv", kv[head:total], head * KV_DIM * 2)
        link.handle.put_signal(b.dst, CH_KV, TIMEOUT_MS)
    fwd.sent_head = source if head > 0 else None
    return inner(q, kv, indices[:fwd.kept], positions, cos_sin_cache, softmax_scale, attn_sink=attn_sink,
                 topk_length=topk_length[:fwd.kept], enable_q_norm=enable_q_norm, rms_norm_eps=rms_norm_eps,
                 use_legacy=use_legacy)


def _wo_b_forward(stock, input_, *args, **kwargs):
    fwd = _FWD[0]
    if fwd is None or not fwd.active:
        return stock(input_, *args, **kwargs)
    kept, bias = stock(input_, *args, **kwargs)
    if kept.shape != (fwd.kept, O_N) or kept.dtype != torch.bfloat16:
        raise RuntimeError(f"bundle-v9 ROW_BALANCE: wo_b returned {tuple(kept.shape)} {kept.dtype} for {fwd.kept} kept rows")
    o = kept.new_empty((fwd.n, kept.shape[1]))
    o[:fwd.kept].copy_(kept, non_blocking=True)
    link = _LINK[0]
    for b in fwd.out:
        link.handle.wait_signal(b.dst, CH_RET, TIMEOUT_MS)
    moved = fwd.n - fwd.kept
    o[fwd.kept:].copy_(link.local("ret", 0, moved * O_N * 2).view(torch.bfloat16).reshape(moved, O_N), non_blocking=True)
    return o, bias


def _guest(view, fwd) -> None:
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.models import deepseek_v4 as model
    from sglang.srt.models import deepseek_v4_mhc as mhc

    from bundle_v9.dec import prefill_attn, prefill_gemm

    block = fwd.into
    rows = block.last - block.first
    link = _LINK[0]
    link.handle.wait_signal(block.src, CH_Q, TIMEOUT_MS)
    entry = prefill_gemm._REG[view.wq_b.weight.data_ptr()]
    route = prefill_gemm._route(entry, rows)
    if route is None:
        raise RuntimeError(f"bundle-v9 ROW_BALANCE: no wq_b route at {rows} rows")
    q8 = link.local("q", 0, rows * Q_K).view(torch.float8_e4m3fn).reshape(rows, Q_K)
    sf = link.local("qsf", 0, _ceil_align(rows) * Q_SF)
    q = prefill_gemm._gemm(*route, q8, sf, entry[1], entry[2]).view(rows, view.n_local_heads, view.head_dim)
    link.handle.wait_signal(block.src, CH_KV, TIMEOUT_MS)
    width = _width(view)
    indices = link.local("idx", 0, rows * width * 4).view(torch.int32).reshape(rows, 1, width)
    lengths = link.local("len", 0, rows * 4).view(torch.int32)
    positions = link.local("pos", 0, rows * 4).view(torch.int32)
    kv = link.local("kv", 0, (HEAD_CAP + SWA_CAP) * KV_DIM * 2).view(torch.bfloat16).reshape(-1, 1, KV_DIM)
    backend = get_attn_backend()
    was_on = prefill_attn._ON[0]
    prefill_attn._ON[0] = True
    try:
        data, scale, _, _ = prefill_attn.mega_prefill(q, kv, indices, positions, view.mega_cos_sin_cache, backend.softmax_scale,
                                               attn_sink=view._local_attn_sink(view.n_local_heads), topk_length=lengths,
                                               use_legacy=not backend._swa_interleaved)
    finally:
        prefill_attn._ON[0] = was_on
    groups = view.n_local_groups
    y, y_sf = mhc.empty_deepgemm_mxfp8_pair(rows, groups * view.o_lora_rank, q.device)
    model._fp8_wo_a_einsum(data[:, :groups], scale[:, :groups], *view.mega_wo_a, recipe=(1, 1, 32),
                           out=(y.view(rows, groups, -1), y_sf))
    o, _ = _STOCK_WO_B[view.layer_id]((y, y_sf), skip_all_reduce=False)
    _send(link, block.src, "ret", o, (block.first - fwd.src_kept) * O_N * 2)
    link.handle.put_signal(block.src, CH_RET, TIMEOUT_MS)


def _width(attention) -> int:
    return WIDTHS[0] if attention.compress_ratio == 0 else WIDTHS[1]


def _attn_forward(self, x, positions, forward_batch, x_quant=None, defer_all_reduce=False):
    fwd = _FWD[0]
    cls = type(self)
    if fwd is None:
        return cls.forward(self, x, positions, forward_batch, x_quant, defer_all_reduce)
    if fwd.out:
        if x.shape[0] != fwd.n:
            raise RuntimeError(f"bundle-v9 ROW_BALANCE: attention x holds {x.shape[0]} rows, the gathered count is {fwd.n}")
        _ensure_chunk_cache(forward_batch, fwd.n)
        fwd.width = _width(self)
        fwd.active = True
        try:
            return cls.forward(self, x, positions, forward_batch, x_quant, defer_all_reduce)
        finally:
            fwd.active = False
    o = cls.forward(self, x, positions, forward_batch, x_quant, defer_all_reduce)
    if fwd.into is not None:
        _guest(self, fwd)
    return o


def _ensure_chunk_cache(batch, rows: int) -> None:
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    metadata = backend.forward_metadata
    if metadata.sparse_prefill_cache is None:
        metadata.sparse_prefill_cache = backend._build_sparse_prefill_chunk_cache(
            batch, metadata.core_attn_metadata, num_qo_tokens=rows)


def attn_view(attention):
    view = _view(attention)
    for name, seam in (("wq_b", _wq_b_forward), ("wo_b", _wo_b_forward)):
        stock = attention._modules[name]
        linear = _view(stock)
        linear.forward = partial(seam, stock)
        view._modules[name] = linear
    _STOCK_WO_B[attention.layer_id] = attention._modules["wo_b"]
    view.forward = MethodType(_attn_forward, view)
    return view


def _counts(batch) -> Optional[list]:
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.dp_attention import is_dp_gatherv_active
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import is_in_breakable_cuda_graph
    from sglang.srt.runtime_context import get_forward

    padding = batch.dp_padding_mode
    if (not batch.is_extend_in_batch or padding is None or not padding.is_sum_len() or is_dp_gatherv_active()
            or is_in_breakable_cuda_graph() or get_forward().sp_active or is_cp_active(batch)):
        return None
    counts = batch.global_num_tokens_cpu
    if counts is None or len(counts) != WORLD:
        return None
    return [int(c) for c in counts]


def _workspace_fits(batch, fwd) -> bool:
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    head = swa = 0
    if fwd.out:
        seq, ext = batch.seq_lens_cpu.tolist(), batch.extend_seq_lens_cpu
        head = sum(int(s) // _R_MIN[0] for s in seq)
        swa = sum(min(int(s), int(e) + SWA_WINDOW - 1) for s, e in zip(seq, ext))
    group = get_parallel().tp_group
    local = torch.tensor([head, swa], dtype=torch.int64)
    every = [torch.empty_like(local) for _ in range(WORLD)]
    dist.all_gather(every, local, group=group.cpu_group)
    return all(int(t[0]) <= HEAD_CAP and int(t[1]) <= SWA_CAP for t in every)


def _begin(batch) -> Optional[_Forward]:
    if not _LINK or batch is None or torch.cuda.is_current_stream_capturing():
        return None
    counts = _counts(batch)
    if counts is None:
        return None
    blocks = plan(counts)
    if not blocks:
        return None
    from sglang.srt.runtime_context import get_parallel

    mode = batch.forward_mode
    if not (mode.is_extend_without_speculative() or mode.is_idle()):
        raise RuntimeError(f"bundle-v9 ROW_BALANCE: a planned forward in mode {mode}")
    fwd = _Forward(blocks, get_parallel().attn_dp_rank, counts)
    fits = _workspace_fits(batch, fwd)
    if TRACE:
        print(f"bundle-v9 ROW_BALANCE rank {get_parallel().attn_dp_rank} counts {counts} blocks {[tuple(b) for b in blocks]} fits {fits}",
              file=sys.stderr, flush=True)
    return fwd if fits else None


@contextlib.contextmanager
def routed(view, batch):
    fwd = _begin(batch)
    if fwd is None:
        yield
        return
    from sglang.kernels.ops.attention import flash_mla_interface

    _FWD[0] = fwd
    inner = flash_mla_interface.flash_mla_mega_prefill
    if fwd.out:
        _INNER[0] = inner
        flash_mla_interface.flash_mla_mega_prefill = _split_prefill
    try:
        yield
    finally:
        _FWD[0] = None
        if fwd.out:
            flash_mla_interface.flash_mla_mega_prefill = inner
            _INNER[0] = None


def enabled() -> bool:
    from sglang.srt.runtime_context import get_disagg

    return get_disagg().disaggregation_mode != "decode"


def _check_layer(attn) -> None:
    if not (attn.use_mega_attention and attn.mega_wq_b_permuted and attn.attn_tp_size == 1 and attn.n_local_heads == 64
            and attn.head_dim == KV_DIM and attn.n_local_groups * attn.o_lora_rank == 8192
            and attn.wo_b.deepgemm_mxfp8_input and attn.wq_b.weight.shape == (32768, Q_K)):
        raise RuntimeError(f"bundle-v9 ROW_BALANCE: layer {attn.layer_id} attention is not the MegaAttention shape it was built for")


def _probe_wire(link: _Link) -> None:
    rank = link.rank
    nxt, prv = (rank + 1) % WORLD, (rank - 1) % WORLD
    gen = torch.Generator(device=link.device).manual_seed(20261103)
    every = [torch.randint(0, 256, (1 << 16,), dtype=torch.uint8, device=link.device, generator=gen) for _ in range(WORLD)]
    _send(link, nxt, "q", every[rank])
    link.handle.put_signal(nxt, CH_PROBE, TIMEOUT_MS)
    link.handle.wait_signal(prv, CH_PROBE, TIMEOUT_MS)
    if not torch.equal(link.local("q", 0, every[prv].numel()), every[prv]):
        raise RuntimeError(f"bundle-v9 ROW_BALANCE: rank {rank} read other bytes than rank {prv} sent")


def _differs(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape != b.shape or not torch.equal(_bytes(a.contiguous()), _bytes(b.contiguous()))


def _probe(attn, device) -> None:
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    from sglang.srt.models import deepseek_v4 as model
    from sglang.srt.models import deepseek_v4_mhc as mhc

    from bundle_v9.dec import prefill_attn, prefill_gemm

    class _Extend:
        class forward_mode:
            is_extend = staticmethod(lambda: True)

    def wq_b(q8, sf):
        return attn.wq_b(Mxfp8SwizzledInput(q8, sf))[0]

    def core(q, kv, idx, lens, pos):
        return prefill_attn.mega_prefill(q, kv, idx, pos, attn.mega_cos_sin_cache, 1.0, attn_sink=attn._local_attn_sink(64),
                                  topk_length=lens, use_legacy=True)[:2]

    def wo(data, scale):
        rows, groups = data.shape[0], attn.n_local_groups
        y, y_sf = mhc.empty_deepgemm_mxfp8_pair(rows, groups * attn.o_lora_rank, device)
        model._fp8_wo_a_einsum(data[:, :groups], scale[:, :groups], *attn.mega_wo_a, recipe=(1, 1, 32),
                               out=(y.view(rows, groups, -1), y_sf))
        return attn.wo_b((y, y_sf), skip_all_reduce=False)[0]

    gen = torch.Generator(device=device).manual_seed(20261103)
    was_on = prefill_attn._ON[0]
    prefill_attn._ON[0] = True
    try:
        with prefill_gemm.routed(None, _Extend):
            for rows, kept in ((8192, 2048), (16384, 8192), (16384, 12288), (32768, 16384)):
                q8 = (torch.randn((rows, Q_K), device=device, generator=gen) * 4).to(torch.float8_e4m3fn)
                sf = torch.randint(120, 134, (_ceil_align(rows) * Q_SF,), dtype=torch.uint8, device=device, generator=gen)
                whole = wq_b(q8, sf)
                head, tail = wq_b(q8[:kept], sf[:kept * Q_SF]), wq_b(q8[kept:], sf[kept * Q_SF:])
                if _differs(whole[:kept], head) or _differs(whole[kept:], tail):
                    raise RuntimeError(f"bundle-v9 ROW_BALANCE: wq_b differs split at {kept} of {rows} rows")
                s_kv = rows + SWA_WINDOW
                kv = torch.randn((s_kv, 1, KV_DIM), device=device, generator=gen, dtype=torch.float32).to(torch.bfloat16)
                width = WIDTHS[0]
                pos = torch.arange(rows, dtype=torch.int32, device=device)
                lens = torch.full((rows,), width, dtype=torch.int32, device=device)
                idx = (pos[:, None, None] + torch.arange(width, dtype=torch.int32, device=device)[None, None, :])
                q = whole.view(rows, 64, KV_DIM)
                data, scale = core(q, kv, idx, lens, pos)
                d1, s1 = core(q[:kept], kv, idx[:kept], lens[:kept], pos[:kept])
                d2, s2 = core(q[kept:], kv, idx[kept:], lens[kept:], pos[kept:])
                if _differs(data[:kept], d1) or _differs(data[kept:], d2) or _differs(scale[:kept], s1) or _differs(scale[kept:], s2):
                    raise RuntimeError(f"bundle-v9 ROW_BALANCE: the attention core differs split at {kept} of {rows} rows")
                o = wo(data, scale)
                if _differs(o[:kept], wo(d1, s1)) or _differs(o[kept:], wo(d2, s2)):
                    raise RuntimeError(f"bundle-v9 ROW_BALANCE: wo_a + wo_b differ split at {kept} of {rows} rows")
                del q8, sf, whole, head, tail, kv, idx, q, data, scale, d1, s1, d2, s2, o
    finally:
        prefill_attn._ON[0] = was_on
    torch.cuda.synchronize(device)


def prepare(module) -> None:
    from sglang.srt.runtime_context import get_parallel

    if not enabled():
        logger.warning("bundle-v9 ROW_BALANCE: PD decode engine, no row balancing")
        return
    if _LINK or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("bundle-v9 ROW_BALANCE: prepare ran twice or inside a CUDA graph capture")
    parallel = get_parallel()
    group = parallel.tp_group
    if group.world_size != WORLD or parallel.attn_dp_rank != group.rank_in_group:
        raise RuntimeError(f"bundle-v9 ROW_BALANCE requires DP rank = TP rank over {WORLD} ranks, got {parallel.attn_dp_rank} / "
                           f"{group.rank_in_group} of {group.world_size}")
    if module.late_layer_start != LAYERS:
        raise RuntimeError(f"bundle-v9 ROW_BALANCE requires the replay tail at layer {LAYERS}, got {module.late_layer_start}")
    ratios = set()
    for layer in module.layers[:LAYERS]:
        _check_layer(layer.self_attn)
        ratios.add(layer.self_attn.compress_ratio)
    ratios.discard(0)
    _R_MIN[0] = min(ratios) if ratios else 1
    if 128 in ratios:
        raise RuntimeError("bundle-v9 ROW_BALANCE: a c128 layer's indices are wider than the inbox")
    device = module.layers[0].self_attn.wq_b.weight.device
    link = _Link(group, group.rank_in_group, device)
    _probe_wire(link)
    _probe(module.layers[0].self_attn, device)
    torch.cuda.synchronize(device)
    _LINK.append(link)
    logger.warning("bundle-v9 ROW_BALANCE: inboxes %.0f MiB a rank, compress ratios %s", link.total / 2**20, sorted(ratios))
