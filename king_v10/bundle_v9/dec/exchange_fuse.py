from __future__ import annotations

import os
import sys
from functools import lru_cache

import torch

CHECK = False
CHECK_CALLS = 4
PREFIX = "bundle-v9-check" if CHECK else "bundle-v9"
WORLD, WIDTH, PACKED, DIM, BLOCK = 4, 1536, 1584, 256, 32

_CALLS = {"IDGATHER": 0, "EQ8": 0}


def _write(path: str, data: bytes) -> None:
    staging = f"{path}.{os.getpid()}.tmp"
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(staging, path)


@lru_cache(maxsize=None)
def _header() -> str:
    import hashlib

    from sglang.kernels.jit.utils.compile.cache import cache_root

    from bundle_v9.dec import cuda_exchange_fuse

    data = cuda_exchange_fuse.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-exchange-fuse",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_exchange_fuse.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    from bundle_v9.dec import cuda_exchange_fuse

    return load_jit("dec_exchange_fuse", cuda_exchange_fuse.TAG, cuda_files=[_header()],
                    cuda_wrappers=list(cuda_exchange_fuse.WRAPPERS))


def _due(name: str) -> bool:
    _CALLS[name] += 1
    return _CALLS[name] <= CHECK_CALLS


def _differ(a: torch.Tensor, b: torch.Tensor) -> int:
    a, b = a.contiguous().view(torch.uint8).view(-1), b.contiguous().view(torch.uint8).view(-1)
    return -1 if a.numel() != b.numel() else int((a != b).sum())


def _report(name: str, rows: int, diffs: list, ours_us: float, stock_us: float) -> None:
    ok = all(d == 0 for d in diffs)
    print(f"{PREFIX} {name} gpu{torch.cuda.current_device()} rows {rows}: {'OK' if ok else 'MISMATCH'} "
          f"bytes-differ {diffs} ours {ours_us:.1f} us stock {stock_us:.1f} us", file=sys.stderr, flush=True)


def _timed(fn):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    out = fn()
    b.record()
    torch.cuda.synchronize()
    return out, a.elapsed_time(b) * 1e3


def id_gather(indices: torch.Tensor, counts: list, rows: int, forward_batch) -> torch.Tensor:
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    out = torch.empty((rows, *indices.shape[1:]), dtype=indices.dtype, device=indices.device)
    mine = counts[group.rank_in_group]

    def ours():
        group.all_gatherv(indices[:mine].contiguous(), sizes=list(counts), output=out)
        return out

    if not (CHECK and _due("IDGATHER")):
        return ours()
    from sglang.srt.layers.dp_attention import dp_gather_replicate

    ref = torch.empty_like(out)
    _, ours_us = _timed(ours)
    _, stock_us = _timed(lambda: dp_gather_replicate(ref, indices.clone(), forward_batch))
    _report("IDGATHER", rows, [_differ(out, ref)], ours_us, stock_us)
    return out


def gather_pack(embed, own: torch.Tensor) -> torch.Tensor:
    packed = torch.empty((own.shape[0], PACKED), dtype=torch.uint8, device=own.device)
    module().q8_gather_pack(embed.weight.view(torch.uint8), embed.scale.view(torch.uint8), own, packed,
                            int(embed.row_start), int(embed.row_start + embed.rows))
    return packed


def place(received: torch.Tensor, mine: int, n_local: int):
    q = torch.empty((n_local, WORLD * WIDTH), dtype=torch.float8_e4m3fn, device=received.device)
    s = torch.empty(((n_local + 127) // 128) * 128 * (WORLD * WIDTH // 32), dtype=torch.uint8, device=received.device)
    module().q8_place(received, q, s)
    return q, s


def check_eq8(embed, own, received, mine: int, n_local: int, counts: list, group) -> None:
    if not _due("EQ8"):
        return
    import torch.distributed as dist

    from sglang.kernels.ops.embeddings.engram_gather import engram_gather

    from bundle_v9.dec import prefill_fuse

    def gather():
        values = torch.empty((own.shape[0], WIDTH), dtype=torch.bfloat16, device=own.device)
        engram_gather(*embed._table_ptrs(), own.view(-1), values.view(-1, embed.dim), embed.dim, BLOCK,
                      row_lo=embed.row_start, row_hi=embed.row_start + embed.rows)
        return values

    values, gather_us = _timed(gather)
    recv = values.new_empty((WORLD * mine, WIDTH))
    dist.all_to_all_single(recv, values, [mine] * WORLD, counts, group=group)
    (rq, rs), quant_us = _timed(lambda: prefill_fuse.engram_quant(recv, mine, n_local))
    (q, s), place_us = _timed(lambda: place(received, mine, n_local))
    _, pack_us = _timed(lambda: gather_pack(embed, own))
    _report("EQ8", n_local, [_differ(q, rq), _differ(s, rs)], pack_us + place_us, gather_us + quant_us)


def prepare(model) -> None:
    from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
    from sglang.srt.runtime_context import get_parallel

    config = model.config
    facts = (config.engram_head_dim, FP8_BLOCK_SIZE,
             (config.engram_max_ngram_size - 1) * config.engram_n_heads // WORLD * config.engram_head_dim)
    if facts != (DIM, BLOCK, WIDTH):
        raise RuntimeError(f"bundle-v9 EXCHANGE_FUSE requires (Engram head dim, FP8 block, exchange width) = "
                           f"{(DIM, BLOCK, WIDTH)}, got {facts}")
    group = get_parallel().tp_group
    if group.world_size != WORLD or group.pynccl_comm is None or not group.pynccl_comm.available:
        raise RuntimeError("bundle-v9 IDGATHER requires the 4-rank TP group's pynccl communicator (all_gatherv)")
    module()
    if CHECK:
        print(f"{PREFIX} prepared: IDGATHER, EQ8 (check {CHECK_CALLS} calls a lever)", file=sys.stderr, flush=True)
