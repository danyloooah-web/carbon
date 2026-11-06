
from __future__ import annotations

import os
import sys
from functools import lru_cache

import torch

K = 512
CHECK = False
VARIANT = (4608, 3, 20)

_STOCK = []
_CHK = [0, 0, 0.0, 0.0]


def _stock():
    if not _STOCK:
        from sglang.kernels.ops.attention import deep_select
        from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

        fn = scoring.topk_ragged
        if fn.__module__ != scoring.__name__ or fn.__qualname__ != "topk_ragged":
            raise RuntimeError(f"v9 TOPK: scoring.topk_ragged is not stock's ({fn.__module__}.{fn.__qualname__})")
        pre = deep_select._precompile
        if pre.__module__ != deep_select.__name__ or pre.__qualname__ != "_precompile":
            raise RuntimeError(f"v9 TOPK: deep_select._precompile is not stock's ({pre.__module__}."
                               f"{pre.__qualname__})")
        _STOCK.extend((fn, pre))
    return _STOCK


def stock_topk_ragged():
    return _stock()[0]


def _dispatcher():
    return _stock()[1](dtype=torch.bfloat16, topk=K, sorted=False, indices_type=torch.int32, sorted_index=False,
                       return_value=False, page_transform=False)


def stock_uses_normal_kernel(rows: int, vocab: int) -> bool:
    dispatch = _dispatcher()
    return dispatch(rows, vocab) is dispatch._fallback


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

    from bundle_v9 import cuda_exact_topk

    data = cuda_exact_topk.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_exact_topk.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def module(cap: int = VARIANT[0], blocks: int = VARIANT[1], safety10: int = VARIANT[2]):
    from sglang.kernels.jit.utils import load_jit

    args = f"{cap}, {blocks}, {safety10}"
    return load_jit(
        "v4_topk", str(cap), str(blocks), str(safety10),
        cuda_files=[_header()],
        cuda_wrappers=[("run", f"V4TopK<{args}>::run"), ("occupancy", f"V4TopK<{args}>::occupancy")],
    )


def run(logits: torch.Tensor, lens: torch.Tensor, out: torch.Tensor, offsets, variant=VARIANT) -> None:
    storage = logits.untyped_storage().nbytes() - logits.storage_offset() * logits.element_size()
    module(*variant).run(logits, lens, offsets, out, out.shape[1], storage, -1)


def topk_ragged(logits: torch.Tensor, lens: torch.Tensor, *, out: torch.Tensor, offsets: torch.Tensor) -> None:
    stock = _stock()[0]
    if (
        logits.dtype != torch.bfloat16
        or logits.shape[0] == 0
        or logits.shape[1] < 49152
        or not stock_uses_normal_kernel(*logits.shape)
    ):
        return stock(logits, lens, out=out, offsets=offsets)
    if out.shape[1] != K:
        raise RuntimeError(f"v9 TOPK serves top-k {K}, got {out.shape[1]}")
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
    from sglang.srt.runtime_context import get_parallel

    rank = get_parallel().attn_dp_rank
    ref = torch.empty_like(out)
    ev[2].record()
    stock(logits, lens, out=ref, offsets=offsets)
    ev[3].record()
    ev[3].synchronize()
    us_v4, us_stock = ev[0].elapsed_time(ev[1]) * 1e3, ev[2].elapsed_time(ev[3]) * 1e3
    bad = int((out != ref).sum())
    _CHK[0] += 1
    _CHK[1] += int(bad != 0)
    _CHK[2] += us_v4
    _CHK[3] += us_stock
    if _CHK[0] == 1:
        from torch.profiler import ProfilerActivity, profile

        scratch = torch.empty_like(out)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            run(logits, lens, scratch, offsets)
            torch.cuda.synchronize()
        names = sorted({e.key for e in prof.key_averages() if e.device_type.name == "CUDA"})
        print(f"v4hk TOPK trace rank {rank}: {names}", file=sys.stderr, flush=True)
    if bad or _CHK[0] <= 4 or _CHK[0] % 25 == 0:
        print(f"v4hk TOPK rank {rank} call {_CHK[0]} rows/width {logits.shape[0]}/{logits.shape[1]} "
              f"max_end {int(lens.max())}: {'BITWISE' if bad == 0 else f'MISMATCH {bad}'} v9 {us_v4:.1f} us "
              f"stock {us_stock:.1f} us (calls {_CHK[0]}, bad {_CHK[1]}, v9 {_CHK[2] / 1e3:.2f} ms, stock "
              f"{_CHK[3] / 1e3:.2f} ms)", file=sys.stderr, flush=True)


def prepare(model) -> None:
    _stock()
    if getattr(model.config, "index_topk", None) != K:
        raise RuntimeError(f"v9 TOPK serves index_topk {K}, got {getattr(model.config, 'index_topk', None)}")
    dispatch = _dispatcher()
    if not hasattr(dispatch, "_fallback"):
        raise RuntimeError("v9 TOPK: DeepSelect's dispatcher has no normal-kernel fallback to compare against")
    module()
