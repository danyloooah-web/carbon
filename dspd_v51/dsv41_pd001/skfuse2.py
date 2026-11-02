"""dsv41-pd054 SKFUSE2: round 2 of the bitwise-exact kernel work in the cold 32K-row prefill forward
(cuda_skfuse2.py = generator/pd054/kernel/skfuse2.cuh; benches in generator/pd054/bench).

The Engram column exchange (engram_cols.py: pd034 ENGRAM_COLS + pd052 EQUANT) calls these:
  IDGATHER  id_gather: every rank's hash ids by the TP group's all_gatherv (one NCCL all-gather, or a group of
            broadcasts when the counts differ) instead of dp_gather_replicate's all-reduce of a zero-padded
            [sum(counts), 24] int64 buffer. Only inside the exchange's domain, whose facts are all-gathered
            (sum(counts) == the buffer's rows), so every rank enters the same collective and every row holds its
            owner's ids either way.
  EQ8 / EQ8G  gather_pack: a sender's own-column rows gathered (stock's _engram_gather_kernel arithmetic) and quantized
            in the same kernel into the MXFP8 codes + scale bytes the receiver's FlashInfer quantize would make (a
            32-element group lies in one source's 1536 columns); place: the received bytes into the wkv GEMM's
            [n_local, 6144] codes and 128x4-swizzled scales. engram_cols takes this path only where PGEMM routes the
            wkv for every rank's count (all-gathered), so every rank moves the same u8 all-to-all.
Every output is the bf16 path's bytes (pd052: stock's gather, the bf16 all-to-all, EQUANT). CHECK (the twin, never
submitted) also runs the replaced path on the first CHECK_CALLS exchanges of each lever, counted alike on every rank so
the twin's extra collectives line up, and prints OK / MISMATCH under "dsv41-pd054chk"."""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import torch

CHECK = False  # gen_pd054 check: the twin's comparisons (measurement only, never submit)
CHECK_CALLS = 4  # per lever, counted over the process's exchanges (every rank advances the counters alike)
PREFIX = "dsv41-pd054chk" if CHECK else "dsv41-pd054"
WORLD, WIDTH, PACKED, DIM, BLOCK = 4, 1536, 1584, 256, 32

_CALLS = {"IDGATHER": 0, "EQ8": 0}


# ---- the JIT module --------------------------------------------------------------------------------------------------
def _write(path: str, data: bytes) -> None:
    """Publish ``data`` at ``path`` by rename, so a reader never sees a partial file (as skfuse._write)."""
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
    """cuda_skfuse2.SOURCE under SGLang's JIT cache root, content-addressed, written on first use."""
    import hashlib

    from sglang.kernels.jit.utils.compile.cache import cache_root

    from dsv41_pd001 import cuda_skfuse2

    data = cuda_skfuse2.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "cacheon-dsv41-pd054",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_skfuse2.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    from dsv41_pd001 import cuda_skfuse2

    return load_jit("pd054_skfuse2", cuda_skfuse2.TAG, cuda_files=[_header()],
                    cuda_wrappers=list(cuda_skfuse2.WRAPPERS))


# ---- the twin --------------------------------------------------------------------------------------------------------
def _due(name: str) -> bool:
    """Whether the twin checks this call: the first CHECK_CALLS calls of the lever (the same calls on every rank)."""
    _CALLS[name] += 1
    return _CALLS[name] <= CHECK_CALLS


def _differ(a: torch.Tensor, b: torch.Tensor) -> int:
    """Differing bytes of two tensors (-1 when their byte sizes differ)."""
    a, b = a.contiguous().view(torch.uint8).view(-1), b.contiguous().view(torch.uint8).view(-1)
    return -1 if a.numel() != b.numel() else int((a != b).sum())


def _report(name: str, rows: int, diffs: list, ours_us: float, stock_us: float) -> None:
    ok = all(d == 0 for d in diffs)
    print(f"{PREFIX} {name} gpu{torch.cuda.current_device()} rows {rows}: {'OK' if ok else 'MISMATCH'} "
          f"bytes-differ {diffs} ours {ours_us:.1f} us stock {stock_us:.1f} us", file=sys.stderr, flush=True)


def _timed(fn):
    """(result, us) of fn on the current stream, by events with a host sync (twin only)."""
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    out = fn()
    b.record()
    torch.cuda.synchronize()
    return out, a.elapsed_time(b) * 1e3


# ---- IDGATHER --------------------------------------------------------------------------------------------------------
def id_gather(indices: torch.Tensor, counts: list, rows: int, forward_batch) -> torch.Tensor:
    """[rows, C] every rank's hash ids, rank r's at sum(counts[:r]) (the exchange's domain: sum(counts) == rows), by
    the TP group's all_gatherv; stock gathers the same rows by an all-reduce of a zero-padded buffer."""
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
    _, stock_us = _timed(lambda: dp_gather_replicate(ref, indices.clone(), forward_batch))  # every rank, same order
    _report("IDGATHER", rows, [_differ(out, ref)], ours_us, stock_us)
    return out


# ---- EQ8 / EQ8G ------------------------------------------------------------------------------------------------------
def gather_pack(embed, own: torch.Tensor) -> torch.Tensor:
    """[rows, 1584] u8: the MXFP8 codes and scale bytes of stock's gather of ``own`` ([rows, 6] table ids) from the
    rank's shard (embed: the EngramEmbedding view; device shard, shard-local rows)."""
    packed = torch.empty((own.shape[0], PACKED), dtype=torch.uint8, device=own.device)
    module().q8_gather_pack(embed.weight.view(torch.uint8), embed.scale.view(torch.uint8), own, packed,
                            int(embed.row_start), int(embed.row_start + embed.rows))
    return packed


def place(received: torch.Tensor, mine: int, n_local: int):
    """The wkv GEMM's MXFP8 pair from the received [WORLD * mine, 1584] bytes: codes [n_local, 6144] and the swizzled
    scale bytes (rows mine..n_local zero, padding rows' scales zero), as pd052's engram_quant returns them."""
    q = torch.empty((n_local, WORLD * WIDTH), dtype=torch.float8_e4m3fn, device=received.device)
    s = torch.empty(((n_local + 127) // 128) * 128 * (WORLD * WIDTH // 32), dtype=torch.uint8, device=received.device)
    module().q8_place(received, q, s)
    return q, s


def check_eq8(embed, own, received, mine: int, n_local: int, counts: list, group) -> None:
    """Twin: the bf16 path for the same rows (stock's gather, a bf16 all-to-all on every rank alike, pd052's EQUANT)
    against EQ8's pair; the twin's only extra collective."""
    if not _due("EQ8"):
        return
    import torch.distributed as dist

    from sglang.kernels.ops.embeddings.engram_gather import engram_gather

    from dsv41_pd001 import skfuse

    def gather():
        values = torch.empty((own.shape[0], WIDTH), dtype=torch.bfloat16, device=own.device)
        engram_gather(*embed._table_ptrs(), own.view(-1), values.view(-1, embed.dim), embed.dim, BLOCK,
                      row_lo=embed.row_start, row_hi=embed.row_start + embed.rows)
        return values

    values, gather_us = _timed(gather)
    recv = values.new_empty((WORLD * mine, WIDTH))
    dist.all_to_all_single(recv, values, [mine] * WORLD, counts, group=group)  # untimed: every rank, same order
    (rq, rs), quant_us = _timed(lambda: skfuse.engram_quant(recv, mine, n_local))
    (q, s), place_us = _timed(lambda: place(received, mine, n_local))
    _, pack_us = _timed(lambda: gather_pack(embed, own))
    # times without the all-to-alls: ours = gather_pack + place, stock = stock's gather + EQUANT
    _report("EQ8", n_local, [_differ(q, rq), _differ(s, rs)], pack_us + place_us, gather_us + quant_us)


# ---- binding ---------------------------------------------------------------------------------------------------------
def prepare(model) -> None:
    """Check the facts the kernels assume (Engram rows of 256 in 32-element e8m0 blocks, the 4 x 1536 exchange, the TP
    group's pynccl for all_gatherv) and build the kernels now (a build failure fails prepare)."""
    from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
    from sglang.srt.runtime_context import get_parallel

    config = model.config
    facts = (config.engram_head_dim, FP8_BLOCK_SIZE,
             (config.engram_max_ngram_size - 1) * config.engram_n_heads // WORLD * config.engram_head_dim)
    if facts != (DIM, BLOCK, WIDTH):
        raise RuntimeError(f"dsv41-pd054 SKFUSE2 requires (Engram head dim, FP8 block, exchange width) = "
                           f"{(DIM, BLOCK, WIDTH)}, got {facts}")
    group = get_parallel().tp_group
    if group.world_size != WORLD or group.pynccl_comm is None or not group.pynccl_comm.available:
        raise RuntimeError("dsv41-pd054 IDGATHER requires the 4-rank TP group's pynccl communicator (all_gatherv)")
    module()
    if CHECK:
        print(f"{PREFIX} prepared: IDGATHER, EQ8 (check {CHECK_CALLS} calls a lever)", file=sys.stderr, flush=True)
