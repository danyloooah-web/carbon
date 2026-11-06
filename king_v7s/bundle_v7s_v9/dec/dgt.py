"""king-v7 DGT: a launch_dependents trigger in DeepGEMM's sm100 1d1d GEMM for wo_a (IDEAS_dec_rest missed M2,
VERIFY_dec_rest §3; RANKED #20), through a bundle-owned header copy and host path, as the king carries its own
MegaMoE / mega-mHC builds (MOE-PREFILL, `cuda_moe_prefill.SOURCES`).

wo_a is the engine's `deep_gemm.fp8_einsum("bhr,hdr->bhd", (o_fp8, o_s), mega_wo_a, (y, y_sf), recipe=(1, 1, 32))`
(`MQALayer._mega_core` -> `_fp8_wo_a_einsum`), 40 calls a decode step on the main stream. Its successor is the wo_b
scale swizzle, PDL-launched under king-v6 WOBSF-PDL (waits before its first read), then CuTe wo_b. The image's sm100
1d1d kernel waits at entry but never triggers (VERIFY_dec_rest C2: only the sm120 copy has a trigger), so the
swizzle's CTAs can launch only when every wo_a CTA has exited. With DGT each wo_a CTA triggers once its TMA-load warp
has issued its last load: the swizzle's CTAs become resident during wo_a's last MMAs and epilogue and start the moment
wo_a completes. Scheduling only: the dependent's griddepcontrol.wait still waits for wo_a's full completion and
flush, and the kernel text is otherwise the image's (dgt_src.HEADER_EDITS; locally the PTX differs from the image
header's by the one launch_dependents block per entry, same registers, no spills: king_lane/v7/front/cc/dgt).

Host path (dgt_src.py): a torch library `dec_dgt_<TAG>` built from the king's own DeepGEMM host texts (pinned by
sha256 in dgt_src.KING_TEXTS, read from cuda_moe_prefill.SOURCES), the image's b7f3816c sm100_fp8_bmm text, and an
entry that dispatches the einsum as the image does. Its JIT has its own signature and include root (the edited
header ahead of the image's include tree), so neither stock's nor the king's kernels or cache keys change. The
library mirrors stock's runtime: PDL per call; SM count, tensor-core utilisation and the deterministic flag must equal
the values it was proven under, or the call goes to stock.

Serving (`fp8_einsum`, swapped in for `deep_gemm.fp8_einsum` by decode_gemm.routed on decode / target-verify forwards
only): the expression, recipe, FP8 output pair, shapes and dtypes must be wo_a's and the row count one of DGT_ROWS
(the graph buckets proven at prepare); everything else is stock's call. A raise turns DGT off and stock serves.

Prepare (`prepare`, from decode_gemm.prepare on the decode server; never raises): build, then at every DGT_ROWS row
count, on layer 0's mega_wo_a and synthetic activations in the attention's own SF layout ([8, 32, align4(rows)]
viewed [rows, 8, 32]), outputs pre-filled with the same random bytes: (1) eager, ours against stock's fp8_einsum over
the whole storage of y and y_sf; (2) ours again on a different fill against stock's logical bytes (every logical
byte written); (3) a captured graph of [dummy kernel, ours, the served wo_b scale swizzle] replayed against stock's
bytes and the king's swizzle of stock's y_sf (the served launch form with its PDL successor). Any failure: off, one
boot line. First boot compiles one DeepGEMM kernel per distinct config of DGT_ROWS (three by the CPU replica in
king_lane/v7/front/tools/woa_layouts.py) plus the host library; the boot line reports the seconds spent.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import torch

# The lead's switch. True: the triggered wo_a serves at DGT_ROWS where proven. False: stock's wo_a everywhere (no
# build, no compile).
DGT_ON = True
# Graph-bucket rows the triggered wo_a serves (each one proven; one JIT compile per distinct config), in value order:
# 240 is about two thirds of the bench's verify steps, 192 / 288 most of the rest (VERIFY_dec_rest C1).
DGT_ROWS = (240, 192, 288)
# Prepare starts no further bucket after this many seconds (a cold JIT cache compiles the host library and one
# DeepGEMM kernel a config; a warm one none). Rank-local: ranks may serve different buckets (scheduling only).
DGT_BUDGET_S = 150.0

EXPR = "bhr,hdr->bhd"
GROUPS, N, K = 8, 1024, 4096
SEED = 20261108

_ON = [False]       # True inside decode_gemm.routed (decode / target-verify forwards)
_STOCK: list = []
_LIB: list = []
_PROVEN: dict = {}  # rows -> the config the library compiled and proved for them
_RT: dict = {}      # the stock runtime the proofs ran under: key (num_sms, tc_util, deterministic), pdl
_STATE = {"on": False, "why": "not prepared"}
_WARNED: list = []
_SERVED: set = set()


def line() -> str:
    return f"bundle-v9 king-v7 DGT: {'ON' if _STATE['on'] else 'off'} ({_STATE['why']})"


def say() -> None:
    print(line(), file=sys.stderr, flush=True)


def on() -> bool:
    return bool(_STATE["on"] and _LIB and _STOCK and _PROVEN)


def _runtime_off(error: Exception) -> None:
    _STATE["on"] = False
    _PROVEN.clear()
    _STATE["why"] = f"disabled at run time, stock's wo_a from here on: {type(error).__name__}: {str(error)[:300]}"
    if not _WARNED:
        _WARNED.append(True)
        say()


def _runtime_key():
    import deep_gemm

    return int(deep_gemm.get_num_sms()), int(deep_gemm.get_tc_util()), bool(deep_gemm.get_deterministic_algorithms())


def _serves(expr, a, b, d, c, recipe) -> bool:
    if expr != EXPR or c is not None or tuple(recipe) != (1, 1, 32):
        return False
    if not (isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and isinstance(d, (tuple, list))
            and len(a) == 2 and len(b) == 2 and len(d) == 2):
        return False
    (x, xs), (w, ws), (y, ys) = a, b, d
    if not all(isinstance(t, torch.Tensor) for t in (x, xs, w, ws, y, ys)) or x.dim() != 3:
        return False
    rows = x.shape[0]
    if (rows not in _PROVEN or tuple(x.shape[1:]) != (GROUPS, K) or tuple(w.shape) != (GROUPS, N, K)
            or tuple(y.shape) != (rows, GROUPS, N) or tuple(ys.shape) != (rows, GROUPS * N // 128)
            or x.dtype != torch.float8_e4m3fn or w.dtype != torch.float8_e4m3fn or y.dtype != torch.float8_e4m3fn
            or xs.dtype != torch.int32 or ws.dtype != torch.int32 or ys.dtype != torch.int32):
        return False
    if _runtime_key() != _RT["key"]:
        return False
    import deep_gemm

    pdl = bool(deep_gemm.get_pdl())
    if pdl != _RT["pdl"]:
        _LIB[0].set_pdl(pdl)
        _RT["pdl"] = pdl
    return True


def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128)):
    if _ON[0] and on():
        try:
            if _serves(expr, a, b, d, c, recipe):
                (x, xs), (w, ws), (y, ys) = a, b, d
                _LIB[0].wo_a(x, xs, w, ws, y, ys)
                if x.shape[0] not in _SERVED:
                    _SERVED.add(x.shape[0])
                    print(f"bundle-v9 king-v7 DGT: first served wo_a at {x.shape[0]} rows "
                          f"({'captured' if torch.cuda.is_current_stream_capturing() else 'eager'})",
                          file=sys.stderr, flush=True)
                return None
        except Exception as error:
            _runtime_off(error)
    return _STOCK[0](expr, a, b, d, c=c, recipe=recipe)


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


def _read(path: str) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _texts(image_include: str):
    """(files, tag): every file of the library root, from pinned king texts, dgt_src texts and the edited header."""
    from bundle_v9.dec import cuda_moe_prefill, dgt_src

    files = {}
    for rel, sha in dgt_src.KING_TEXTS.items():
        text = cuda_moe_prefill.SOURCES.get(rel)
        if text is None or hashlib.sha256(text.encode()).hexdigest() != sha:
            raise RuntimeError(f"the king's host text {rel} is missing or not the pinned one")
        files[rel] = text
    for rel, text in dgt_src.SOURCES.items():
        if rel in files:
            raise RuntimeError(f"{rel} is both a king text and a DGT text")
        files[rel] = text
    header = _read(os.path.join(image_include, dgt_src.IMAGE_HEADER))
    if hashlib.sha256(header).hexdigest() != dgt_src.IMAGE_HEADER_SHA256:
        raise RuntimeError(f"the image's {dgt_src.IMAGE_HEADER} is not the pinned 0.2.3.post2 text")
    header = header.decode()
    for old, new in dgt_src.HEADER_EDITS:
        if header.count(old) != 1:
            raise RuntimeError(f"a DGT header anchor matches {header.count(old)} times (want 1)")
        header = header.replace(old, new)
    files["include/" + dgt_src.IMAGE_HEADER] = header
    blob = b"".join(rel.encode() + b"\0" + text.encode() + b"\0" for rel, text in sorted(files.items()))
    return files, hashlib.sha256(blob).hexdigest()[:16]


def _root(files: dict, tag: str) -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-dgt", tag)
    for rel, text in files.items():
        path = os.path.join(directory, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = text.encode()
        if not os.path.isfile(path) or _read(path) != data:
            _write(path, data)
    os.makedirs(os.path.join(directory, "build"), exist_ok=True)
    return directory


def _build(root: str, tag: str, image_include: str):
    from torch.utils.cpp_extension import CUDA_HOME
    from torch.utils.cpp_extension import load as build_library

    from bundle_v9.dec import dgt_src

    cuda_home = CUDA_HOME or "/usr/local/cuda"
    includes = [os.path.join(root, "include"), image_include, os.path.join(root, "csrc"),
                os.path.join(root, "build_include"), os.path.join(cuda_home, "include")]
    cccl = os.path.join(cuda_home, "include", "cccl")
    if os.path.isdir(cccl):
        includes.append(cccl)
    build_library(name=f"dec_dgt_{tag}", sources=[os.path.join(root, dgt_src.ENTRY)],
                  extra_cflags=list(dgt_src.CXX_FLAGS),
                  extra_ldflags=[f"-L{os.path.join(cuda_home, 'lib64')}", "-lcudart", "-lc10_cuda", "-ltorch_cuda",
                                 *dgt_src.LD_FLAGS],
                  extra_include_paths=includes, build_directory=os.path.join(root, "build"), with_cuda=False,
                  is_python_module=False, verbose=False)
    lib = torch.ops.dec_dgt
    lib.init(root, image_include)
    return lib


def _storage(t: torch.Tensor) -> torch.Tensor:
    """The whole span a strided tensor covers, as a flat view (pads included)."""
    span = 1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()))
    return torch.as_strided(t, (span,), (1,), t.storage_offset())


def _inputs(rows: int, device, gen):
    aligned = (rows + 3) // 4 * 4
    x = torch.randn((rows, GROUPS, K), device=device, generator=gen) * 2
    x[rows // 2].zero_()
    x[0, 0, :64] = 448.0
    x = x.to(torch.float8_e4m3fn)
    words = torch.randint(118, 136, (GROUPS, K // 128, aligned * 4), device=device, generator=gen,
                          dtype=torch.uint8).view(torch.int32)
    xs = words.transpose(1, 2)[:, :rows].transpose(0, 1)  # the attention's out_sf layout (vattn / decode_attn)
    return x, xs


def _outputs(rows: int, device, fill_y: torch.Tensor, fill_s: torch.Tensor):
    from sglang.srt.models import deepseek_v4_mhc as mhc

    y, ys = mhc.empty_deepgemm_mxfp8_pair(rows, GROUPS * N, device)
    y.view(torch.uint8).view(-1).copy_(fill_y)
    _storage(ys).copy_(fill_s)
    return y, ys


def _swizzle(ys: torch.Tensor):
    from bundle_v9.dec import glue_pdl, prefill_fuse

    out = glue_pdl.wob_scale(ys)  # the served wo_b scale swizzle (king-v6 WOBSF-PDL), or the king's
    return out if out is not None else prefill_fuse.wob_scale(ys)


def _prove(lib, w, ws, rows: int, device, gen) -> None:
    from bundle_v9.dec import prefill_fuse

    from sglang.srt.models import deepseek_v4_mhc as mhc

    x, xs = _inputs(rows, device, gen)
    span = _storage(mhc.empty_deepgemm_mxfp8_scales(rows, GROUPS * N, device)).numel()
    fill_y = torch.randint(0, 256, (rows * GROUPS * N,), device=device, generator=gen, dtype=torch.uint8)
    fill_s = torch.randint(-(1 << 30), 1 << 30, (span,), device=device, generator=gen, dtype=torch.int32)
    scratch = torch.zeros(1 << 16, device=device)

    y_ref, ys_ref = _outputs(rows, device, fill_y, fill_s)
    scratch.add_(1.0)
    _STOCK[0](EXPR, (x, xs), (w, ws), (y_ref.view(rows, GROUPS, N), ys_ref), recipe=(1, 1, 32))
    sw_ref = prefill_fuse.wob_scale(ys_ref)

    y_out, ys_out = _outputs(rows, device, fill_y, fill_s)
    scratch.add_(1.0)
    lib.wo_a(x, xs, w, ws, y_out.view(rows, GROUPS, N), ys_out)
    if not (torch.equal(y_ref.view(torch.uint8), y_out.view(torch.uint8))
            and torch.equal(_storage(ys_ref), _storage(ys_out))):
        raise RuntimeError(f"eager wo_a at {rows} rows differs from stock's fp8_einsum (whole y / y_sf storage)")

    y_alt, ys_alt = _outputs(rows, device, fill_y ^ 0x5A, fill_s ^ 0x2D2D2D2D)
    scratch.add_(1.0)
    lib.wo_a(x, xs, w, ws, y_alt.view(rows, GROUPS, N), ys_alt)
    if not (torch.equal(y_ref.view(torch.uint8), y_alt.view(torch.uint8)) and torch.equal(ys_ref, ys_alt)):
        raise RuntimeError(f"wo_a at {rows} rows leaves part of its logical output unwritten")

    y_g, ys_g = _outputs(rows, device, fill_y, fill_s)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch.add_(1.0)
        lib.wo_a(x, xs, w, ws, y_g.view(rows, GROUPS, N), ys_g)
        sw_g = _swizzle(ys_g)
    graph.replay()
    torch.cuda.synchronize(device)
    ok = (torch.equal(y_ref.view(torch.uint8), y_g.view(torch.uint8)) and torch.equal(_storage(ys_ref), _storage(ys_g))
          and torch.equal(sw_ref, sw_g))
    del graph
    if not ok:
        raise RuntimeError(f"the captured wo_a + scale swizzle chain at {rows} rows differs from stock's")


def prepare(module) -> None:
    """Decode-server prepare; never raises."""
    _STATE.update(on=False, why="not prepared")
    _PROVEN.clear()
    if not DGT_ON:
        _STATE["why"] = "switched off by dgt.DGT_ON; stock's wo_a (no build)"
        say()
        return
    start = time.time()
    try:
        import deep_gemm

        fn = deep_gemm.fp8_einsum
        if fn.__module__ != "deep_gemm" or fn.__qualname__ != "fp8_einsum":
            raise RuntimeError(f"deep_gemm.fp8_einsum is not the image's ({fn.__module__}.{fn.__qualname__})")
        _STOCK[:] = [fn]
        if torch.cuda.get_device_capability() != (10, 3):
            raise RuntimeError(f"built and proven for sm_103 (B300), got {torch.cuda.get_device_capability()}")
        if not DGT_ROWS or any(not 0 < r <= 288 for r in DGT_ROWS):
            raise RuntimeError(f"DGT_ROWS {DGT_ROWS} must be decode rows 1..288")
        wo_a = next((layer.self_attn.mega_wo_a for layer in module.layers
                     if getattr(layer.self_attn, "mega_wo_a", None) is not None), None)
        if wo_a is None:
            raise RuntimeError("no layer has the mega attention's quantized wo_a")
        w, ws = wo_a
        if (tuple(w.shape) != (GROUPS, N, K) or w.dtype != torch.float8_e4m3fn or ws.dtype != torch.int32
                or tuple(ws.shape) != (GROUPS, N, K // 128)):
            raise RuntimeError(f"mega_wo_a is {tuple(w.shape)} {w.dtype} / {tuple(ws.shape)} {ws.dtype}")
        image_include = os.path.join(os.path.dirname(deep_gemm.__file__), "include")
        files, tag = _texts(image_include)
        lib = _build(_root(files, tag), tag, image_include)
        key = _runtime_key()
        pdl = bool(deep_gemm.get_pdl())
        lib.set_num_sms(key[0])
        lib.set_tc_util(key[1])
        lib.set_deterministic(key[2])
        lib.set_pdl(pdl)
        if (int(lib.get_num_sms()), int(lib.get_tc_util()), bool(lib.get_deterministic())) != key:
            raise RuntimeError(f"the library's runtime does not mirror stock's {key}")
        device = w.device
        gen = torch.Generator(device=device).manual_seed(SEED)
        proven, skipped = {}, []
        for rows in DGT_ROWS:
            if proven and time.time() - start > DGT_BUDGET_S:
                skipped.append(rows)
                continue
            config = tuple(int(v) for v in lib.config(GROUPS, rows, N, K))
            _prove(lib, w, ws, rows, device, gen)
            proven[rows] = config
        torch.cuda.synchronize(device)
    except Exception as error:
        _PROVEN.clear()
        _LIB.clear()
        _STATE["why"] = (f"FAIL, stock's wo_a: {type(error).__name__}: {str(error)[:500]} "
                         f"({time.time() - start:.1f} s spent)")
        say()
        return
    _LIB[:] = [lib]
    _RT.update(key=key, pdl=pdl)
    _PROVEN.update(proven)
    _STATE["on"] = True
    configs = "; ".join(f"{rows} rows swap {c[0]} block {c[1]}x{c[2]} cluster {c[3]}x{c[4]} stages {c[5]}"
                        for rows, c in sorted(proven.items()))
    _STATE["why"] = (f"wo_a triggers its dependents after its last TMA load at {len(proven)} graph buckets "
                     f"({configs}); "
                     f"bytes equal to stock's fp8_einsum eager and captured with the served scale swizzle; runtime "
                     f"sms {key[0]} tc_util {key[1]} deterministic {key[2]} pdl {pdl}; library {tag}; "
                     f"{time.time() - start:.1f} s"
                     f"{f'; budget skipped {skipped}' if skipped else ''}")
    say()
