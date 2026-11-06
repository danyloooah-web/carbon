
from __future__ import annotations

from typing import Optional, Sequence

import torch

RANKS = 4
CAP = 32768
HELP = 16384
TS = 2048
HID, QLORA, KVA, KV_DIM, MULT = 5120, 1280, 512, 512, 4
IDX0_W, IDX2_W = 128, 640
KVC_CAP = 393_216
KVS_CAP = 65_536
KVL_CAP = KVC_CAP + KVS_CAP
XQ, KVR, IDS, RET, PROBE = range(5)
TIMEOUT_MS = 60_000
_ALIGN = 256

_REGIONS = [
    ("x", CAP, (HID,), torch.bfloat16),
    ("qlora", CAP, (QLORA,), torch.bfloat16),
    ("kva", CAP, (KVA,), torch.bfloat16),
    ("kvc", KVC_CAP, (1, KV_DIM), torch.bfloat16),
    ("kvs", KVS_CAP, (1, KV_DIM), torch.bfloat16),
    ("idx0", CAP, (IDX0_W,), torch.int32),
    ("idx2", CAP, (IDX2_W,), torch.int32),
    ("ilen", CAP, (), torch.int32),
    ("ipos", CAP, (), torch.int32),
    ("ids", HELP, (), torch.int64),
    ("tidx", TS, (), torch.int64),
    ("probe", 1 << 20, (), torch.uint8),
]
for _slot in (0, 1, 2):
    _REGIONS += [
        (f"ts{_slot}_streams", TS, (MULT, HID), torch.bfloat16),
        (f"ts{_slot}_input", TS, (HID,), torch.bfloat16),
        (f"ts{_slot}_pre", TS, (MULT,), torch.float32),
        (f"ts{_slot}_s_pre", TS, (MULT,), torch.float32),
        (f"ts{_slot}_s_post", TS, (MULT,), torch.float32),
        (f"ts{_slot}_s_comb", TS, (MULT, MULT), torch.float32),
    ]

_LINK: list = []


def _layout():
    out, off = {}, 0
    for name, rows, shape, dtype in _REGIONS:
        nbytes = rows * torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        out[name] = (off, rows, shape, dtype, nbytes)
        off += -(-nbytes // _ALIGN) * _ALIGN
    return out, off


class Link:

    def __init__(self, group, rank: int, device: torch.device, role: str = "prefill"):
        import torch.distributed._symmetric_memory as symm_mem

        self.regions, total = _layout()
        self.rank, self.device, self.group, self.role = rank, device, group, role
        self.buffer = symm_mem.empty(total, dtype=torch.int8, device=device)
        self.handle = symm_mem.rendezvous(self.buffer, group=group.device_group)
        self.buffer.zero_()
        self._kv_local: Optional[torch.Tensor] = None
        self._idx0_local = torch.full((HELP, IDX0_W), -1, dtype=torch.int32, device=device)
        self._idx2_local = torch.full((HELP, IDX2_W), -1, dtype=torch.int32, device=device)
        self._len_local = torch.zeros((HELP,), dtype=torch.int32, device=device)
        self._pos_local = torch.zeros((HELP,), dtype=torch.int32, device=device)
        group.barrier()
        torch.cuda.synchronize(device)
        self.side = torch.cuda.Stream(device=device, priority=-1)

    def _span(self, name: str, rows: Optional[int], row0: int):
        off, cap, shape, dtype, nbytes = self.regions[name]
        rows = cap - row0 if rows is None else rows
        if row0 < 0 or rows < 0 or row0 + rows > cap:
            raise RuntimeError(f"v9-rehome transport: {name} rows [{row0}, {row0 + rows}) outside [0, {cap})")
        row_bytes = nbytes // cap
        return off + row0 * row_bytes, rows * row_bytes, (rows, *shape), dtype

    def local(self, name: str, rows: Optional[int] = None, row0: int = 0) -> torch.Tensor:
        start, nbytes, shape, dtype = self._span(name, rows, row0)
        return self.buffer[start:start + nbytes].view(torch.uint8).view(dtype).view(shape)

    def peer(self, rank: int, name: str, rows: Optional[int] = None, row0: int = 0) -> torch.Tensor:
        if rank == self.rank:
            return self.local(name, rows, row0)
        start, nbytes, shape, dtype = self._span(name, rows, row0)
        return self.handle.get_buffer(rank, (nbytes,), torch.uint8, start).view(dtype).view(shape)

    def kv_local(self, ws_len: int) -> torch.Tensor:
        if ws_len < 0:
            raise RuntimeError(f"v9-rehome transport: negative kv_local ws_len={ws_len}")
        if ws_len == 0:
            return torch.empty((0, 1, KV_DIM), dtype=torch.bfloat16, device=self.device)
        if self._kv_local is None or ws_len > self._kv_local.shape[0]:
            cap = -(-ws_len // 16384) * 16384
            self._kv_local = None
            self._kv_local = torch.zeros((cap, 1, KV_DIM), dtype=torch.bfloat16, device=self.device)
        return self._kv_local[:ws_len]


def prepare(module) -> bool:
    from sglang.srt.runtime_context import get_disagg, get_parallel

    role = str(get_disagg().disaggregation_mode)
    if role == "decode":
        return False
    if _LINK:
        return True
    if role not in ("prefill", "null"):
        raise RuntimeError(f"v9-rehome transport: unsupported disaggregation_mode {role!r}")
    from sglang.srt.environ import envs

    parallel = get_parallel()
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    if (parallel.tp_size, parallel.attn_dp_size, parallel.attn_tp_size, cap) != (RANKS, RANKS, 1, CAP):
        raise RuntimeError(f"v9-rehome transport needs TP4 = DP-attention 4 and a {CAP}-row MegaMoE cap, got tp "
                           f"{parallel.tp_size} dp {parallel.attn_dp_size} attn_tp {parallel.attn_tp_size} cap {cap}")
    device = next(module.parameters()).device
    link = Link(parallel.tp_group, int(parallel.attn_dp_rank), device, role=role)
    _probe(link)
    _LINK.append(link)
    return True


def _probe(link: Link) -> None:
    nxt, prv = (link.rank + 1) % RANKS, (link.rank - 1) % RANKS
    n = link.regions["probe"][1]
    pattern = lambda r: (torch.arange(n, device=link.device, dtype=torch.int64) * (2 * r + 3) % 251).to(torch.uint8)
    kvc_pat = lambda r: (torch.arange(16 * KV_DIM, device=link.device, dtype=torch.int32).view(16, 1, KV_DIM) + 17 * r).to(torch.bfloat16)
    with torch.cuda.device(link.device):
        _LINK.append(link)
        try:
            push(nxt, "probe", pattern(link.rank))
            signal(nxt, PROBE)
            wait(prv, PROBE)
            join()
            ok_push = bool(torch.equal(link.local("probe"), pattern(prv)))
            src_kvc = kvc_pat(link.rank)
            publish_kvr([("kvc", src_kvc, 0)], (nxt,))
            wait(prv, KVR)
            dst_kv = link.kv_local(64)
            dst_kv[32:48].copy_(link.peer(prv, "kvc", 16, 0))
            join()
            ok_pull = bool(torch.equal(dst_kv[32:48], kvc_pat(prv)))
            torch.cuda.synchronize(link.device)
            link.group.barrier()
            dst_kv[32:48].zero_()
            link._kv_local = None
            link.local("kvc", 16, 0).zero_()
            link.local("probe").zero_()
            torch.cuda.synchronize(link.device)
            link.group.barrier()
            ok = ok_push and ok_pull
        finally:
            _LINK.clear()
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=link.group.cpu_group)
    if not int(flag.item()):
        raise RuntimeError(f"v9-rehome transport: ring probe mismatch (rank {link.rank} ok={ok})")


def active() -> bool:
    return bool(_LINK)


def link() -> Link:
    return _LINK[0]


def view(name: str, rows: Optional[int] = None, row0: int = 0) -> torch.Tensor:
    return _LINK[0].local(name, rows, row0)


def peer(rank: int, name: str, rows: Optional[int] = None, row0: int = 0) -> torch.Tensor:
    return _LINK[0].peer(rank, name, rows, row0)


def kv_local(ws_len: int) -> torch.Tensor:
    return _LINK[0].kv_local(ws_len)


def idx0_local(rows: int) -> torch.Tensor:
    return _LINK[0]._idx0_local[:rows]


def idx2_local(rows: int) -> torch.Tensor:
    return _LINK[0]._idx2_local[:rows]


def len_local(rows: int) -> torch.Tensor:
    return _LINK[0]._len_local[:rows]


def pos_local(rows: int) -> torch.Tensor:
    return _LINK[0]._pos_local[:rows]


def push(dst: int, name: str, src: torch.Tensor, row0: int = 0) -> None:
    lk = _LINK[0]
    if not src.is_contiguous() or src.dtype != lk.regions[name][3]:
        raise RuntimeError(f"v9-rehome transport: push of {src.dtype} contiguous={src.is_contiguous()} into {name}")
    rows = src.shape[0] if lk.regions[name][2] else src.numel()
    dst_view = lk.peer(dst, name, rows, row0)
    ready = torch.cuda.current_stream(lk.device).record_event()
    lk.side.wait_event(ready)
    with torch.cuda.stream(lk.side):
        dst_view.copy_(src.view(dst_view.shape))
    src.record_stream(lk.side)


def publish_kvr(copies: Sequence[tuple], helpers: Sequence[int]) -> None:
    lk = _LINK[0]
    ready = torch.cuda.current_stream(lk.device).record_event()
    lk.side.wait_event(ready)
    with torch.cuda.stream(lk.side):
        for name, src, row0 in copies:
            if not src.is_contiguous() or src.dtype != lk.regions[name][3]:
                raise RuntimeError(
                    f"v9-rehome transport: publish {name} got {src.dtype} contiguous={src.is_contiguous()}"
                )
            rows = src.shape[0] if lk.regions[name][2] else src.numel()
            if rows > 0:
                dst_view = lk.local(name, rows, row0)
                dst_view.copy_(src.view(dst_view.shape))
            src.record_stream(lk.side)
        for h in helpers:
            lk.handle.put_signal(int(h), KVR, TIMEOUT_MS)


def signal(dst: int, channel: int) -> None:
    lk = _LINK[0]
    with torch.cuda.stream(lk.side):
        lk.handle.put_signal(dst, channel, TIMEOUT_MS)


def wait(src: int, channel: int) -> None:
    _LINK[0].handle.wait_signal(src, channel, TIMEOUT_MS)


def join() -> None:
    lk = _LINK[0]
    torch.cuda.current_stream(lk.device).wait_stream(lk.side)
