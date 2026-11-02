"""dsv41-pd033 TOPK: wide DeepSelect normal-kernel calls of scoring.topk_ragged run a bit-identical kernel.

Stock (sealed fork 15338a8e, sglang.srt.layers.attention.dsv4.v41_indexer.scoring.topk_ragged): for bf16 logits it
calls DeepSelect's `topk(logits, out.shape[1], end=lens, output_idx=out, output_idx_offset=offsets,
idx_oob_fill_value=-1)`, whose dispatcher (`deep_select._precompile(...)` called with (rows, vocab)) sends few wide
rows to its cluster kernel and everything else to its normal one-CTA-a-row kernels. The dense prefill top-k of the
ratio-2 index layers 2 / 8 / 14 (FullTopKIndexer._deep_gemm_prefill -> dense_prefill_topk) calls it once per logits
tile: 1.18 + 0.53 ms a tile on the heaviest pd027 forward (trace p43), DeepSelect at 25% of DRAM bandwidth.

Here the calls of at least 49,152 columns that stock's own dispatcher sends to the normal kernel run the pd033 kernel
(bf16, k = 512); its output is DeepSelect's bit for bit, indices and order, fill and NaN rows included (cuda_topk.py
states the contract and how it was verified). Every other call -- narrower logits, fp32 logits, the cluster kernel's
(nondeterministic in stock: generator/pd033 tiepolicy), empty batches -- is stock's topk_ragged unchanged.
slots.forward_model binds `topk_ragged` over scoring.topk_ragged for its own forward only and restores stock's in
`finally`; dense_prefill_topk, defined in scoring.py, looks the name up in that module's globals. Other topk_ragged
importers (sparse_table, dense_blocks) keep their own reference to stock's. That dense_prefill_topk's calls reach the
binding is shown at runtime: generator/pd033's bench drives them through it, and the dsv41-pd033chk twin logs every
served call.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import torch

K = 512  # stock's index_topk (the c2_sparse_raw_indices width): asserted on every served call
CHECK = False  # dsv41-pd033chk only: also run stock on every served call and print the bitwise comparison
VARIANT = (4608, 3, 20)  # the kernel's (candidate capacity, CTAs an SM, speculation x10): generator/pd033 bench

_STOCK = []  # [scoring.topk_ragged, deep_select._precompile] as loaded, captured once at prepare
_CHK = [0, 0, 0.0, 0.0]  # dsv41-pd033chk: served calls, calls with a mismatch, pd033 / stock microseconds


def _stock():
    if not _STOCK:
        from sglang.kernels.ops.attention import deep_select
        from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

        fn = scoring.topk_ragged
        if fn.__module__ != scoring.__name__ or fn.__qualname__ != "topk_ragged":
            raise RuntimeError(f"dsv41-pd033 TOPK: scoring.topk_ragged is not stock's ({fn.__module__}.{fn.__qualname__})")
        pre = deep_select._precompile
        if pre.__module__ != deep_select.__name__ or pre.__qualname__ != "_precompile":
            raise RuntimeError(f"dsv41-pd033 TOPK: deep_select._precompile is not stock's ({pre.__module__}."
                               f"{pre.__qualname__})")
        _STOCK.extend((fn, pre))
    return _STOCK


def stock_topk_ragged():
    """Stock's scoring.topk_ragged, captured at prepare (never read back through the module global)."""
    return _stock()[0]


def _dispatcher():
    """DeepSelect's own dispatcher for topk_ragged's bf16 call (cached by stock: the object stock's topk uses)."""
    return _stock()[1](dtype=torch.bfloat16, topk=K, sorted=False, indices_type=torch.int32, sorted_index=False,
                       return_value=False, page_transform=False)


def stock_uses_normal_kernel(rows: int, vocab: int) -> bool:
    """True when stock's DeepSelect serves a [rows, vocab] bf16 call with its normal kernel (not a cluster kernel)."""
    dispatch = _dispatcher()
    return dispatch(rows, vocab) is dispatch._fallback


def _write(path: str, data: bytes) -> None:
    """Publish ``data`` at ``path`` by rename, so a reader never sees a partial header (as mhc_stats_jit._write)."""
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
    """cuda_topk.SOURCE under SGLang's JIT cache root, content-addressed, written on first use."""
    import hashlib

    from sglang.kernels.jit.utils.compile.cache import cache_root

    from dsv41_pd001 import cuda_topk

    data = cuda_topk.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "cacheon-dsv41-pd033",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_topk.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def module(cap: int = VARIANT[0], blocks: int = VARIANT[1], safety10: int = VARIANT[2]):
    """The load_jit module of one kernel variant (load_jit's default flags for the local GPU)."""
    from sglang.kernels.jit.utils import load_jit

    args = f"{cap}, {blocks}, {safety10}"
    return load_jit(
        "pd033_topk", str(cap), str(blocks), str(safety10),
        cuda_files=[_header()],
        cuda_wrappers=[("run", f"Pd033TopK<{args}>::run"), ("occupancy", f"Pd033TopK<{args}>::occupancy")],
    )


def run(logits: torch.Tensor, lens: torch.Tensor, out: torch.Tensor, offsets, variant=VARIANT) -> None:
    """The pd033 kernel on one normal-kernel call: DeepSelect's arguments, its storage bound, fill -1."""
    storage = logits.untyped_storage().nbytes() - logits.storage_offset() * logits.element_size()
    module(*variant).run(logits, lens, offsets, out, out.shape[1], storage, -1)


def topk_ragged(logits: torch.Tensor, lens: torch.Tensor, *, out: torch.Tensor, offsets: torch.Tensor) -> None:
    """scoring.topk_ragged; DeepSelect's normal-kernel calls of 49,152+ columns run the pd033 kernel (bit for bit)."""
    stock = _stock()[0]
    # Served from 49,152 columns: paired full prefill passes (logits + top-k) measured this kernel faster than stock at
    # 49,152, 81,920 and 114,688 columns (generator/pd033 bench). Narrower calls (no gain shown there) stay on stock.
    if (
        logits.dtype != torch.bfloat16
        or logits.shape[0] == 0
        or logits.shape[1] < 49152
        or not stock_uses_normal_kernel(*logits.shape)
    ):
        return stock(logits, lens, out=out, offsets=offsets)
    if out.shape[1] != K:
        raise RuntimeError(f"dsv41-pd033 TOPK serves top-k {K}, got {out.shape[1]}")
    if not CHECK:
        run(logits, lens, out, offsets)
        return None
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ev[0].record()
    run(logits, lens, out, offsets)
    ev[1].record()
    _check(stock, logits, lens, out, offsets, ev)
    return None


def _check(stock, logits, lens, out, offsets, ev) -> None:
    """dsv41-pd033chk: stock's call on the same inputs into a scratch tensor, compared bit for bit, and both calls
    timed with CUDA events (host sync). The first call also lists the CUDA kernels a profiled pd033 call launches."""
    from sglang.srt.runtime_context import get_parallel

    rank = get_parallel().attn_dp_rank
    ref = torch.empty_like(out)
    ev[2].record()
    stock(logits, lens, out=ref, offsets=offsets)
    ev[3].record()
    ev[3].synchronize()
    us_pd033, us_stock = ev[0].elapsed_time(ev[1]) * 1e3, ev[2].elapsed_time(ev[3]) * 1e3
    bad = int((out != ref).sum())
    _CHK[0] += 1
    _CHK[1] += int(bad != 0)
    _CHK[2] += us_pd033
    _CHK[3] += us_stock
    if _CHK[0] == 1:
        from torch.profiler import ProfilerActivity, profile

        scratch = torch.empty_like(out)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            run(logits, lens, scratch, offsets)
            torch.cuda.synchronize()
        names = sorted({e.key for e in prof.key_averages() if e.device_type.name == "CUDA"})
        print(f"dsv41-pd033chk TOPK trace rank {rank}: {names}", file=sys.stderr, flush=True)
    if bad or _CHK[0] <= 4 or _CHK[0] % 25 == 0:
        print(f"dsv41-pd033chk TOPK rank {rank} call {_CHK[0]} rows/width {logits.shape[0]}/{logits.shape[1]} "
              f"max_end {int(lens.max())}: {'BITWISE' if bad == 0 else f'MISMATCH {bad}'} pd033 {us_pd033:.1f} us "
              f"stock {us_stock:.1f} us (calls {_CHK[0]}, bad {_CHK[1]}, pd033 {_CHK[2] / 1e3:.2f} ms, stock "
              f"{_CHK[3] / 1e3:.2f} ms)", file=sys.stderr, flush=True)


def prepare(model) -> None:
    """Capture stock, check the served domain and build the kernel now (a build failure fails prepare)."""
    _stock()
    if getattr(model.config, "index_topk", None) != K:
        raise RuntimeError(f"dsv41-pd033 TOPK serves index_topk {K}, got {getattr(model.config, 'index_topk', None)}")
    dispatch = _dispatcher()
    if not hasattr(dispatch, "_fallback"):
        raise RuntimeError("dsv41-pd033 TOPK: DeepSelect's dispatcher has no normal-kernel fallback to compare against")
    module()
