"""Lever MXT: CuTe-DSL MXFP8 GEMM tactics for the attention node's eager prefill projections.

The engine serves the block-FP8 projections that FlashInfer's CuTe-DSL MXFP8 GEMM runs (wq_b, the indexer's wq_b,
the Engram wkv) through its autotuned CuTe-DSL runner; at prefill sizes the tactic it serves leaves ~30 % on the table
(measured in the engine's own prefill trace, 677 us for wq_b at 16K rows): on the B300 rig, wq_b
([M,1280] x [1280,32768]) takes 180 / 340 / 677 us at M = 4096 / 8192 / 16384 by default and 129 / 236 / 476 us with
the tile ((256, 256), cluster (2, 1), no split-K), whose output is bit-identical (same K order). This module wraps the
engine's module-level FlashInfer MXFP8 entry for the duration of our nodes' own calls only (`scope()`, entered by
dspd/attn.py around the attention node and by dspd/engram.py around the wkv projection; the engine's entry is restored
on exit, so stock, twin and draft paths never see the wrapper); there, for an eager call (not capturing) of a checked
shape with >= 1024 rows, it runs the engine's own CuTe-DSL runner with that tactic (split-1, so bit-identical to any
split-1 tactic the engine serves). The first call of each (N, K, row band) also runs the engine's default and compares the two bit for bit; a
difference turns the band back to the engine's tactic. Every other call is the engine's own path.

GEMMs are rank-local (no collective), so the per-rank first-use verdict cannot desynchronize ranks."""
import contextlib
import contextvars

import torch
from flashinfer.gemm import gemm_base as GB

from dspd.common import pins_match, say
from sglang.srt.layers.quantization import fp8_utils as FU

ENABLED = True
MIN_ROWS = 1024
TACTIC = ((256, 256), (2, 1), False, False, 1)
SHAPES = frozenset({(32768, 1280), (4096, 1280), (25600, 6144), (5120, 6144)})  # (N, K) measured on the rig (25600 x 6144 =
# the Engram wkv projection, MXT-E: default 129 / 245 / 529 us vs 105 / 207 / 397 us at M = 1024 / 2048 / 4096)
# 5120 x 6144 = EKQ's value half of that projection (dspd/ekq.py), first-use checked like the rest
# MXT-D: decode / target-verify graph rows (6 per request at bs 1..48) -> the fastest tactic measured bit-identical
# to the default for that (N, K, rows) on the rig (rig_dec.py, cold weights); rows absent = the engine's default.
# Rows <= 32 are left to the engine: its autotuner may serve FlashInfer's split-K kernel there (MAX_M 32), whose
# fp32 reduction order a split-1 tactic does not reproduce bit for bit.
DECODE = {
    (32768, 1280): {36: ((128, 64), (1, 1), True, False, 1), 42: ((128, 64), (1, 1), True, False, 1), 48: ((128, 128), (1, 4), False, False, 1), 60: ((128, 128), (1, 4), False, False, 1), 72: ((128, 128), (1, 4), False, False, 1), 84: ((128, 128), (1, 4), False, False, 1), 96: ((256, 128), (4, 1), True, False, 1), 120: ((128, 128), (1, 4), False, False, 1), 132: ((256, 192), (2, 1), True, False, 1), 144: ((256, 192), (2, 1), True, False, 1), 156: ((256, 192), (2, 1), True, False, 1), 168: ((256, 192), (2, 1), True, False, 1), 180: ((256, 192), (2, 1), True, False, 1), 192: ((128, 192), (1, 1), True, False, 1), 240: ((256, 256), (2, 2), False, False, 1), 264: ((256, 192), (2, 2), True, False, 1), 288: ((256, 192), (2, 2), True, False, 1)},
    (25600, 6144): {36: ((128, 64), (1, 1), True, False, 1), 42: ((128, 64), (1, 1), True, False, 1), 48: ((128, 64), (1, 1), True, False, 1), 60: ((128, 64), (1, 1), True, False, 1), 72: ((256, 128), (4, 1), True, False, 1), 84: ((256, 128), (4, 1), True, False, 1), 96: ((256, 128), (4, 1), True, False, 1), 108: ((256, 128), (4, 1), True, False, 1), 120: ((256, 128), (4, 1), True, False, 1), 132: ((256, 192), (2, 1), True, False, 1), 144: ((256, 192), (2, 1), True, False, 1), 156: ((256, 192), (2, 1), True, False, 1), 168: ((256, 192), (2, 1), True, False, 1), 180: ((256, 192), (2, 1), True, False, 1), 192: ((256, 192), (2, 1), True, False, 1), 264: ((256, 192), (2, 1), True, False, 1), 288: ((256, 192), (2, 1), True, False, 1)},
    (4096, 1280): {72: ((256, 256), (2, 1), False, False, 1), 84: ((256, 256), (2, 1), False, False, 1), 96: ((128, 128), (1, 4), False, False, 1), 132: ((256, 256), (2, 2), False, False, 1), 180: ((128, 128), (1, 4), False, False, 1), 288: ((256, 256), (2, 1), False, False, 1)}}
PINS = {"flashinfer_mxfp8_blockscaled_linear": "44d063202cb3947e"}   # same in the nightly and the stand-in
context = contextvars.ContextVar("dspd_mxt", default=False)
_g = {"orig": None, "pins": None, "verdict": {}, "runner": {}, "compiled": set(), "depth": 0}


def _runner(device, out_dtype):
    """The engine's CuTe-DSL MXFP8 runner for this device's arch, built once per (device, dtype)."""
    r = _g["runner"].get((device, out_dtype))
    if r is None:
        major, minor = torch.cuda.get_device_capability(device)
        r = _g["runner"][(device, out_dtype)] = GB._cute_dsl_gemm_mxfp8_runner(major, minor, True, out_dtype)
    return r


CHECK_ROWS = 256                                       # two whole 128-row scale tiles


def _slice_check(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, out):
    """Bit equality of `out` with the engine's default on the first rows only (<= CHECK_ROWS). The A scales are 1D
    swizzled per 128-row tile, so the slice is whole tiles and takes the matching prefix of the scale buffer."""
    m = q_input.shape[0]
    rows = m if m <= CHECK_ROWS else CHECK_ROWS
    if rows == m:
        ref = _g["orig"](q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype, use_8x4_sf_layout=False,
                         backend="cute-dsl")
        return bool(torch.equal(out, ref))
    k_tiles = -(-q_input.shape[1] // 32)
    sf_cols = -(-k_tiles // 4) * 4
    if x_scale_u8.dim() != 1 or x_scale_u8.numel() < rows * sf_cols:
        return False                                  # unexpected scale layout: keep the engine's tactic
    ref = _g["orig"](q_input[:rows], weight_t, x_scale_u8[: rows * sf_cols], weight_scale_t, out_dtype=out_dtype,
                     use_8x4_sf_layout=False, backend="cute-dsl")
    return bool(torch.equal(out[:rows], ref))


def _decode(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, m, shape):
    """The tactic for a decode-bucket call, or None (absent row count, not precompiled, or a failed first-use check)."""
    tactic = DECODE.get(shape, {}).get(m)
    if tactic is None or (shape, tactic) not in _g["compiled"]:
        return None
    key = (shape, m, "decode")
    verdict = _g["verdict"].get(key)
    if verdict is False:
        return None
    out = torch.empty((m, shape[0]), dtype=out_dtype, device=q_input.device)
    ws = GB._get_cache_buf("mm_mxfp8_workspace", GB.DEFAULT_WORKSPACE_SIZE, q_input.device)
    _runner(q_input.device, out_dtype)(inputs=[q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, out, ws],
                                       tactic=tactic)
    if verdict is None and not torch.cuda.is_current_stream_capturing():
        same = _slice_check(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, out)
        _g["verdict"][key] = same
        say("mxt", f"MXT-D first-use check {'OK' if same else 'DIFFERENT (engine tactic)'}: N {shape[0]} rows {m}")
        if not same:
            del out
            return _g["orig"](q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                              use_8x4_sf_layout=False, backend="cute-dsl")
    return out


def _mm(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, use_8x4_sf_layout=False, backend="auto"):
    orig = _g["orig"]
    m = q_input.shape[0]
    shape = (weight_t.shape[1], weight_t.shape[0])
    if (backend == "cute-dsl" and not use_8x4_sf_layout and context.get() and m < MIN_ROWS and q_input.dim() == 2
            and shape in DECODE):
        out = _decode(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, m, shape)
        if out is not None:
            return out
    if (backend != "cute-dsl" or use_8x4_sf_layout or not context.get() or m < MIN_ROWS or shape not in SHAPES
            or q_input.dim() != 2 or torch.cuda.is_current_stream_capturing()):
        return orig(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                    use_8x4_sf_layout=use_8x4_sf_layout, backend=backend)
    key = (shape, m.bit_length())
    verdict = _g["verdict"].get(key)
    if verdict is False:
        return orig(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                    use_8x4_sf_layout=use_8x4_sf_layout, backend=backend)
    out = torch.empty((m, shape[0]), dtype=out_dtype, device=q_input.device)
    ws = GB._get_cache_buf("mm_mxfp8_workspace", GB.DEFAULT_WORKSPACE_SIZE, q_input.device)
    _runner(q_input.device, out_dtype)(inputs=[q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, out, ws],
                                       tactic=TACTIC)
    if verdict is None:
        same = _slice_check(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, out)
        _g["verdict"][key] = same
        say("mxt", f"MXT first-use check {'OK' if same else 'DIFFERENT (engine tactic for this band)'}: "
                   f"N {shape[0]} K {shape[1]} rows {m}")
        if not same:
            del out
            return orig(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                        use_8x4_sf_layout=use_8x4_sf_layout, backend=backend)
    return out


def _precompile():
    """Build every decode tactic once on synthetic operands, outside any graph capture (a capture never compiles)."""
    if torch.cuda.is_current_stream_capturing() or not torch.cuda.is_available():
        say("mxt", "MXT-D off: first prepare ran inside a capture")
        return
    from sglang.srt.runtime_context import get_disagg

    dev = torch.device("cuda", torch.cuda.current_device())

    def zeros_mx(rows, k):
        """Zero fp8 operand and its 1D 128x4-swizzled UE8M0 scales, laid out as mxfp8_quantize lays them out."""
        sf = (-(-rows // 128) * 128) * (-(-(k // 32) // 4) * 4)
        return (torch.zeros((rows, k), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn),
                torch.full((sf,), 127, dtype=torch.uint8, device=dev))

    # The prefill tactic: a CuTe-DSL tactic compiles once per process at its first call (~0.45 s measured) and is then
    # reused for every shape and row count, so compiling it here keeps that compile out of the first eager prefill
    # call, which may fall inside a scored window
    try:
        a_mx, a_sf = zeros_mx(MIN_ROWS, 1280)
        w_mx, w_sf = zeros_mx(4096, 1280)
        out = torch.empty((MIN_ROWS, 4096), dtype=torch.bfloat16, device=dev)
        ws = GB._get_cache_buf("mm_mxfp8_workspace", GB.DEFAULT_WORKSPACE_SIZE, dev)
        _runner(dev, torch.bfloat16)(inputs=[a_mx, w_mx.t(), a_sf, w_sf, torch.bfloat16, out, ws], tactic=TACTIC)
        del a_mx, a_sf, w_mx, w_sf, out
        say("mxt", f"MXT prefill tactic {TACTIC} precompiled")
    except Exception as e:  # noqa: BLE001
        say("mxt", f"MXT prefill tactic precompile failed ({type(e).__name__}): compiled at first use instead")
    if get_disagg().disaggregation_mode == "prefill":
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        return                                        # a prefill server captures no decode graphs

    for shape, table in DECODE.items():
        n, k = shape
        w_mx, w_sf = zeros_mx(n, k)
        for tactic in sorted(set(table.values())):
            m = min(r for r, t in table.items() if t == tactic)
            a_mx, a_sf = zeros_mx(m, k)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            ws = GB._get_cache_buf("mm_mxfp8_workspace", GB.DEFAULT_WORKSPACE_SIZE, dev)
            try:
                _runner(dev, torch.bfloat16)(inputs=[a_mx, w_mx.t(), a_sf, w_sf, torch.bfloat16, out, ws], tactic=tactic)
                _g["compiled"].add((shape, tactic))
            except Exception as e:  # noqa: BLE001
                say("mxt", f"MXT-D tactic {tactic} for {shape} unavailable ({type(e).__name__}): engine default there")
        del w_mx, w_sf
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()                          # hand the synthetic operands' memory back to the driver
    say("mxt", f"MXT-D precompiled {len(_g['compiled'])} decode tactics")


def enter():
    """Our node's call begins: MXT's context set and the engine's MXFP8 entry swapped for `_mm` (outermost entry).
    Returns the token `leave` takes, None when MXT is not installed. The attention node calls the pair directly, without
    a generator context manager, because it runs on every eager prefill layer."""
    if _g["orig"] is None:
        return None
    token = context.set(True)
    if _g["depth"] == 0:
        setattr(FU, "flashinfer_mm_mxfp8", _mm)
    _g["depth"] += 1
    return token


def leave(token):
    """Our node's call ends: the engine's entry restored at the outermost exit."""
    if token is None:
        return
    _g["depth"] -= 1
    if _g["depth"] == 0:
        setattr(FU, "flashinfer_mm_mxfp8", _g["orig"])
    context.reset(token)


@contextlib.contextmanager
def scope():
    """`enter` / `leave` around a block."""
    token = enter()
    try:
        yield
    finally:
        leave(token)


def install():
    """Resolve the engine's MXFP8 entry and precompile the decode tactics once (the swap itself is `scope()`'s)."""
    if not ENABLED or _g["orig"] is not None:
        return
    if _g["pins"] is None:
        _g["pins"] = pins_match("mxt", PINS, {"flashinfer_mxfp8_blockscaled_linear": FU.flashinfer_mxfp8_blockscaled_linear})
    if not _g["pins"]:
        return
    _g["orig"] = FU.flashinfer_mm_mxfp8
    _precompile()
    say("mxt", f"MXT installed: prefill CuTe-DSL MXFP8 tactic {TACTIC} from {MIN_ROWS} rows for {sorted(SHAPES)}")
