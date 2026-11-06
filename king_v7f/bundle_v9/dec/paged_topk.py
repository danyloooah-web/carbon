from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import torch

from bundle_v9.dec import cuda_paged_topk

K = 512
MAX_COLS = 8320 * 32
_STOCK: list = []
_ON = [False]


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
    from sglang.kernels.jit.utils.compile.cache import cache_root

    data = cuda_paged_topk.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-paged-topk",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_paged_topk.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def _module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit("dec_paged_topk", cuda_files=[_header()], cuda_wrappers=[("run", "PagedTopK::run")])


def _stock():
    if not _STOCK:
        from sglang.kernels.ops.attention import deep_select
        from sglang.srt.layers.attention.dsv4 import indexer

        fn = indexer.topk_transform_paged_bf16
        if fn.__module__ != indexer.__name__ or fn.__qualname__ != "topk_transform_paged_bf16":
            raise RuntimeError(f"bundle-v9 PAGED_TOPK: indexer.topk_transform_paged_bf16 is not stock's ({fn.__module__}."
                               f"{fn.__qualname__})")
        pre = deep_select._precompile
        if pre.__module__ != deep_select.__name__ or pre.__qualname__ != "_precompile":
            raise RuntimeError("bundle-v9 PAGED_TOPK: deep_select._precompile is not stock's")
        _STOCK.extend((fn, pre))
    return _STOCK


def stock_topk_transform_paged_bf16():
    return _stock()[0]


def _normal(rows: int, cols: int) -> bool:
    dispatch = _stock()[1](dtype=torch.bfloat16, topk=K, sorted=False, indices_type=torch.int32, sorted_index=False,
                           return_value=False, page_transform=True)
    return dispatch(rows, cols) is dispatch._fallback


def _serves(logits, out_page_indices, page_size) -> bool:
    rows, cols = logits.shape
    return (logits.dtype == torch.bfloat16 and out_page_indices.shape[-1] == K and 0 < rows and cols <= MAX_COLS
            and page_size > 0 and page_size & (page_size - 1) == 0 and _normal(rows, cols))


def _run(logits, ends, page_table, out, page_size) -> None:
    storage = logits.untyped_storage().nbytes() - logits.storage_offset() * logits.element_size()
    _module().run(logits, ends, page_table, out, page_size, storage, -1)


def topk_transform_paged_bf16(logits, seq_lens, page_table, out_page_indices, page_size) -> None:
    if not _serves(logits, out_page_indices, page_size):
        return stock_topk_transform_paged_bf16()(logits, seq_lens, page_table, out_page_indices, page_size)
    _run(logits, seq_lens.reshape(-1).to(torch.int32), page_table, out_page_indices, page_size)
    return None


def _values(family: str, rows: int, cols: int, gen, device) -> torch.Tensor:
    if family == "normal":
        return torch.randn(rows, cols, device=device, generator=gen).bfloat16()
    if family == "relu":
        x = torch.zeros(rows, cols, device=device)
        for _ in range(4):
            x += torch.relu(torch.randn(rows, cols, device=device, generator=gen) - 1.5) * 0.3
        return x.bfloat16()
    if family == "tail":
        return (torch.rand(rows, cols, device=device, generator=gen) ** 4 * 20).bfloat16()
    if family in ("ties8", "ties2"):
        n = 8 if family == "ties8" else 2
        return torch.randint(0, n, (rows, cols), device=device, generator=gen).float().bfloat16()
    if family == "zero":
        x = torch.zeros(rows, cols, device=device).bfloat16()
        x.view(torch.int16)[:, 1::3] = -32768
        return x
    if family == "neginf":
        x = torch.randn(rows, cols, device=device, generator=gen)
        x[torch.rand(rows, cols, device=device, generator=gen) < 0.995] = float("-inf")
        return x.bfloat16()
    if family == "allneginf":
        return torch.full((rows, cols), float("-inf"), device=device).bfloat16()
    if family == "inf":
        x = torch.randn(rows, cols, device=device, generator=gen)
        x[torch.rand(rows, cols, device=device, generator=gen) < 0.002] = float("inf")
        return x.bfloat16()
    if family == "nanrow":
        x = torch.randn(rows, cols, device=device, generator=gen)
        x[rows // 2, cols // 3] = float("nan")
        return x.bfloat16()
    if family == "halves":
        return (torch.randn(rows, cols, device=device, generator=gen).round() * 0.5).bfloat16()
    if family in ("denorm", "rawbits"):
        if family == "denorm":
            b = torch.randint(0, 32, (rows, cols), device=device, generator=gen, dtype=torch.int32)
            b = ((b & 16) << 11) | (b & 15)
        else:
            b = torch.randint(0, 65536, (rows, cols), device=device, generator=gen, dtype=torch.int32)
            b = torch.where((b & 0x7FFF) > 0x7F80, b ^ 0x4000, b)
        return (b - 65536 * (b >= 32768).int()).to(torch.int16).view(torch.bfloat16)
    raise ValueError(family)


FAMILIES = ("normal", "relu", "tail", "ties8", "ties2", "zero", "neginf", "allneginf", "inf", "nanrow", "halves", "denorm",
            "rawbits")
SHAPES = ((288, 131328, 128), (288, 262656, 128), (240, 131328, 128), (288, 72064, 128), (288, 144128, 256),
          (240, 72064, 128), (192, 98304, 256), (144, 40960, 128), (96, 16896, 128), (288, 16384, 128), (288, 600, 128))


def _ends(rows: int, cols: int, mode: str, gen, device) -> torch.Tensor:
    if mode == "full":
        return torch.full((rows,), cols, device=device, dtype=torch.int32)
    if mode == "upper":
        return torch.randint(cols // 2, cols + 1, (rows,), device=device, generator=gen, dtype=torch.int32)
    if mode == "any":
        return torch.randint(0, cols + 1, (rows,), device=device, generator=gen, dtype=torch.int32)
    edges = [0, 1, 511, 512, 513, 1024, 4095, 4096, 4097, 16383, 16384, 16385, 16896, 16897, 20480, 20481, cols - 1,
             cols, cols - 4096, cols - 4097, cols - 511, cols - 513]
    edges = [e for e in edges if 0 <= e <= cols]
    return torch.tensor([edges[i % len(edges)] for i in range(rows)], device=device, dtype=torch.int32)


def _prove(device) -> int:
    stock = stock_topk_transform_paged_bf16()
    gen = torch.Generator(device=device).manual_seed(20261105)
    cases = 0
    for rows, cols, page_size in SHAPES:
        if not _normal(rows, cols):
            continue
        stride = (cols + 63) // 64 * 64
        pages = (cols + page_size - 1) // page_size
        for family in FAMILIES:
            for mode in ("full", "upper", "any", "edges"):
                block = torch.full((rows, stride), float("nan"), device=device, dtype=torch.bfloat16)
                block[:, :cols] = _values(family, rows, cols, gen, device)
                logits = block[:, :cols]
                ends = _ends(rows, cols, mode, gen, device)
                table = torch.randint(0, 1 << 20, (rows, pages), device=device, generator=gen, dtype=torch.int32)
                want = torch.full((rows, K), -7, device=device, dtype=torch.int32)
                got = torch.full((rows, K), -9, device=device, dtype=torch.int32)
                stock(logits, ends, table, want, page_size)
                _run(logits, ends, table, got, page_size)
                if not torch.equal(want, got):
                    bad = int((want != got).any(dim=1).sum())
                    raise RuntimeError(f"bundle-v9 PAGED_TOPK: {bad} of {rows} rows differ from stock's DeepSelect ({family}, "
                                       f"{cols} columns, page {page_size}, ends {mode})")
                cases += 1
    torch.cuda.synchronize(device)
    return cases


def prepare(model) -> None:
    from sglang.srt.layers.attention.dsv4 import indexer
    from sglang.srt.layers.attention.dsv4.v41_indexer import sparse_table

    stock = stock_topk_transform_paged_bf16()
    if sparse_table.topk_transform_paged_bf16 is not stock or indexer.topk_transform_paged_bf16 is not stock:
        raise RuntimeError("bundle-v9 PAGED_TOPK: a decode paged top-k entry was swapped before prepare")
    device = next(model.parameters()).device
    cases = _prove(device)
    _ON[0] = True
    print(f"bundle-v9 PAGED_TOPK: decode paged top-k on DeepSelect's normal-kernel calls, bytes proven on {cases} cases",
          flush=True)


def enabled() -> bool:
    return _ON[0]
