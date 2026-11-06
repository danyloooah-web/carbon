import hashlib
import os
import sys
from functools import lru_cache

import torch

from bundle_v9.dec import cuda_mhc_prefetch

THREADS = 320
PROOF_ROWS = (149, 150, 160, 176, 192, 200, 216, 217, 224, 240, 256, 264, 276, 280, 288)
_ON = [False]
_PICK: list = []  # [module] when an edited HOIST text serves (king-v2e R-L2, king-v6 HOIST-T / -PDL-IN); empty = the king's


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
    return _header_of(cuda_mhc_prefetch.SOURCE.encode())


@lru_cache(maxsize=None)
def _header_of(data: bytes) -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-mhc-prefetch",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_mhc_prefetch.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def _king_module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "dec_mhc_prefetch",
        str(THREADS),
        cuda_files=[_header()],
        cuda_wrappers=[("run", f"MhcHoist<{THREADS}>::run"), ("run_dg", f"MhcHoistDg<{THREADS}>::run")],
    )


@lru_cache(maxsize=None)
def _rl2_module(text: str):
    from sglang.kernels.jit.utils import load_jit

    data = text.encode()
    return load_jit(
        "dec_mhc_prefetch_rl2",
        str(THREADS),
        hashlib.sha256(data).hexdigest()[:16],
        cuda_files=[_header_of(data)],
        cuda_wrappers=[("run", f"MhcHoist<{THREADS}>::run"), ("run_dg", f"MhcHoistDg<{THREADS}>::run")],
    )


@lru_cache(maxsize=None)
def _v6_module(text: str):
    """king-v6: the HOIST text with HOIST-T / HOIST-PDL-IN edits (on top of R-L2's when it is on); its own module name."""
    from sglang.kernels.jit.utils import load_jit

    data = text.encode()
    return load_jit(
        "dec_mhc_prefetch_v6",
        str(THREADS),
        hashlib.sha256(data).hexdigest()[:16],
        cuda_files=[_header_of(data)],
        cuda_wrappers=[("run", f"MhcHoist<{THREADS}>::run"), ("run_dg", f"MhcHoistDg<{THREADS}>::run")],
    )


def _module():
    return _PICK[0] if _PICK else _king_module()


def serves(threads: int) -> bool:
    return _ON[0] and threads == THREADS


def post_combine(y, residual, post, comb, pre, weight, eps, quant):
    m = y.shape[0]
    updated, normalized = torch.empty_like(residual), torch.empty_like(y)
    if quant:
        q = torch.empty((m, 5120), dtype=torch.float8_e4m3fn, device=y.device)
        sf = torch.empty((m + 127) // 128 * 128 * 160, dtype=torch.uint8, device=y.device)
    else:
        q = sf = y.new_empty(0)
    _module().run(y, residual, post, comb, pre, weight, updated, normalized, q, sf, eps, quant)
    if not quant:
        return updated, normalized, None, None
    return updated, normalized, q, sf


def post_combine_dg(y, residual, post, comb, pre, weight, eps, y_fp8, y_sf):
    updated, normalized = torch.empty_like(residual), torch.empty_like(y)
    _module().run_dg(y, residual, post, comb, pre, weight, updated, normalized, y_fp8, y_sf, eps)
    return updated, normalized


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and torch.equal(a.contiguous().view(-1).view(torch.uint8),
                                              b.contiguous().view(-1).view(torch.uint8))


def _prove(device) -> None:
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import mhc_post_combine_norm_decode
    from sglang.srt.models.deepseek_v4_mhc import empty_deepgemm_mxfp8_pair

    from bundle_v9.dec import mhc_floor

    gen = torch.Generator(device=device).manual_seed(20261104)
    for rows in PROOF_ROWS:
        y = (torch.randn((rows, 5120), device=device, generator=gen) * 2).bfloat16()
        residual = torch.randn((rows, 4, 5120), device=device, generator=gen).bfloat16()
        y[rows // 2].zero_()
        residual[rows // 3, 1].fill_(-0.0)
        post = torch.rand((rows, 4), device=device, generator=gen) * 2
        comb = torch.rand((rows, 4, 4), device=device, generator=gen)
        pre = torch.rand((rows, 4), device=device, generator=gen)
        norm = (torch.rand(5120, device=device, generator=gen) + 0.5).bfloat16()
        for quant in (False, True):
            want = mhc_post_combine_norm_decode(y, residual, post, comb, pre, norm, 1e-6, quant=quant, threads=THREADS)
            got = post_combine(y, residual, post, comb, pre, norm, 1e-6, quant)
            if not all(_same(a, b) for a, b in zip(got, want)):
                raise RuntimeError(f"dsv41 HOIST: the boundary at {rows} rows (quant {quant}) differs from stock's")
        want_q, want_sf = empty_deepgemm_mxfp8_pair(rows, 5120, device)
        got_q, got_sf = empty_deepgemm_mxfp8_pair(rows, 5120, device)
        want_sf.fill_(0)
        got_sf.fill_(0)
        want_u, want_n = torch.empty_like(residual), torch.empty_like(y)
        mhc_floor._dg_module(THREADS).run(y, residual, post, comb, pre, norm, want_u, want_n, want_q, want_sf,
                                                  1e-6)
        got_u, got_n = post_combine_dg(y, residual, post, comb, pre, norm, 1e-6, got_q, got_sf)
        if not (_same(got_u, want_u) and _same(got_n, want_n) and _same(got_q, want_q) and _same(got_sf, want_sf)):
            raise RuntimeError(f"dsv41 HOIST: the DeepGEMM-pair boundary at {rows} rows differs from the copy it replaces")
    torch.cuda.synchronize(device)


def prepare(module) -> None:
    from bundle_v9.dec import decode_gemm, mhc_floor

    if not decode_gemm.enabled():
        from bundle_v9.dec import hoist_pdl, rl2

        rl2.mark(False, "HOIST is not prepared on this engine (decode_gemm off)")
        print(rl2.line(), file=sys.stderr, flush=True)
        hoist_pdl.mark_hoist(False, "HOIST is not prepared on this engine (decode_gemm off)")
        hoist_pdl.say_hoist()
        hoist_pdl.mark("SEAM-POST", False, "not prepared on this engine (decode_gemm off)")
        hoist_pdl.say("SEAM-POST")
        return
    device = next(module.parameters()).device
    if mhc_floor._decode_threads(mhc_floor.MEGA_MHC_DECODE_MAX_ROWS, device) != THREADS:
        raise RuntimeError("dsv41 HOIST: the floor's top rows do not take 320-thread boundaries")
    _prepare_variant(device)  # R-L2 + HOIST-T / -PDL-IN; leaves _PICK empty (the king's HOIST) on failure or when off
    if not _PICK:
        _prove(device)
    _ON[0] = True
    print(f"dsv41 HOIST: {THREADS}-thread decode boundaries hoisted, bytes proven at {len(PROOF_ROWS)} row counts",
          flush=True)
    from bundle_v9.dec import hoist_pdl, rl2

    print(rl2.line(), file=sys.stderr, flush=True)
    hoist_pdl.say_hoist()
    hoist_pdl.prepare_seam(device)  # king-v6 SEAM-POST; never raises, stock's split_h post on any failure


def _prepare_variant(device) -> None:
    """The served HOIST text. Candidates in order: the composed text (king-v2e R-L2 when on, then king-v6 HOIST-T /
    HOIST-PDL-IN when on), then the R-L2 text alone (king-v2e's exact module), then none (_PICK empty: the king's text,
    proven by the caller as in the king). A candidate serves only after the king's _prove passes on it. Never raises."""
    from bundle_v9.dec import hoist_pdl, rl2

    _PICK.clear()
    king = cuda_mhc_prefetch.SOURCE
    rl2_text, rl2_why = None, None
    if not rl2.RL2_ON:
        rl2_why = "switched off by rl2.RL2_ON; the king's HOIST text"
    else:
        try:
            rl2_text = rl2.hoist_source(king)
            if rl2_text is None:
                rl2_why = "both sub-switches off; the king's HOIST text"
        except Exception as error:
            rl2_why = f"FAIL, the king's HOIST text instead: {type(error).__name__}: {str(error)[:600]}"
    names, _ = hoist_pdl.hoist_edits()
    v6_text, v6_why = None, None
    if names:
        try:
            v6_text = hoist_pdl.hoist_source(rl2_text if rl2_text is not None else king)
        except Exception as error:
            v6_why = f"FAIL, the king's HOIST trigger and launch: {type(error).__name__}: {str(error)[:600]}"
    candidates = ([("v6", v6_text)] if v6_text is not None else []) + ([("rl2", rl2_text)] if rl2_text is not None else [])
    served, errors = None, []
    for kind, text in candidates:
        try:
            _PICK[:] = [_v6_module(text) if kind == "v6" else _rl2_module(text)]
            _prove(device)
        except Exception as error:
            _PICK.clear()
            errors.append(f"{kind} text: {type(error).__name__}: {str(error)[:400]}")
            continue
        served = kind
        break
    failed = "; ".join(errors)
    if rl2_text is None:
        rl2.mark(False, rl2_why)
    elif served is not None:
        rl2.mark(True, f"HOIST at 149..288 rows: {rl2.parts()}; bytes proven at {len(PROOF_ROWS)} row counts against "
                       f"stock and the king's packed kernel")
    else:
        rl2.mark(False, f"FAIL, the king's HOIST text instead: {failed}")
    if not names:
        hoist_pdl.mark_hoist(False, "")
    elif v6_text is None:
        hoist_pdl.mark_hoist(False, v6_why)
    elif served == "v6":
        hoist_pdl.mark_hoist(True, f"HOIST at 149..288 rows ({', '.join(names)} edits"
                                   f"{' on the R-L2 text' if rl2_text is not None else ''}); bytes proven at "
                                   f"{len(PROOF_ROWS)} row counts against stock and the king's packed kernel")
    else:
        hoist_pdl.mark_hoist(False, f"FAIL, {'the R-L2 text' if served == 'rl2' else 'the king text'} serves: {failed}")
