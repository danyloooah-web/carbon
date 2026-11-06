"""king-v6 glue PDL levers on the decode server: QNORM-PDL, WOBSF-PDL and IDX-W16 (IDEAS_dec_rest DR3, DR4, DR9).

The three kernels are anchored copies of the king's own PFUSE kernels (cuda_glue_pdl.py), compiled as one JIT module
(`dec_glue_pdl`) that holds only the switched-on copies. Each lever is proven at prepare at its served launch form (the
programmatic-serialization attribute on, the served kRows / grid) and publishes only after a full pass; any failure,
including the module build, turns that lever off with one boot line and the king's path serves:
- QNORM-PDL: proven by qnorm_dec (king-v2e's 579-case byte proof against stock's Triton q-norm, run on this kernel);
  if it fails, QNORM-DEC falls back to the king's PFUSE kernel and proves that instead.
- WOBSF-PDL: the swizzled bytes against the king's `wob_scale` at every wo_b row count 1..288, in the served DeepGEMM
  scale layout and a row-major one, output buffers pre-filled with different patterns; then decode_gemm's own whole wo_b
  proof runs with the PDL swizzle serving (a failure there turns the lever off and reruns the king's proof).
- IDX-W16: payload, scale and head weights against stock's `index_q_rope_pack_weights`, every served index layer's own
  head-weight scale, int32 and int64 positions; the bf16 weights must equal both the upper half of stock's fp32 words
  (whose lower half must be zero, i.e. stock's values are exact bf16 values) and torch's own `.to(torch.bfloat16)` of them.
PDL is scheduling only: every copy waits (griddepcontrol.wait) before its first read of data a predecessor may still be
writing, and every dependent in these chains waits before reading (CuTe wq_b / wo_b, the king's PAGED-LOGITS, DeepGEMM).
The bytes therefore cannot depend on the overlap; the prepare proofs show the arithmetic is the reference's.
Capture: each served call allocates what the replaced call allocates (IDX-W16 allocates its head weights as bf16, and the
consumers' bf16 copies disappear), does no host sync, and launches on the current stream. A launch error turns its lever
off for later captures and is logged once.
"""
from __future__ import annotations

import hashlib
import os
import sys

import torch

# The lead's switches (each one lever). True: the PDL copy serves where proven. False: the king's path.
QNORM_PDL_ON = True
# Rows a warp in the PDL q-norm copy. The king's launcher uses 2; rows are independent (one warp each), so this changes
# only the grid (1 doubles the warps at decode rows).
QNORM_PDL_ROWS = 1
WOBSF_PDL_ON = True
IDX_W16_ON = True

WOB_WORDS = 8192 // 128
WOB_MAX_ROWS = 288
IDX_HEADS, IDX_DIM = 32, 128
IDX_FULL_TOKENS = (1, 2, 3, 6, 7, 12, 13, 24, 48, 96, 144, 149, 192, 216, 217, 240, 256, 264, 288)
IDX_LAYER_TOKENS = (6, 240, 288)
IDX_FAMILIES = 3
SEED = 20261106

_MOD: list = []
_ERR: list = []
_STATE: dict = {name: {"on": False, "why": "not prepared"} for name in ("QNORM-PDL", "WOBSF-PDL", "IDX-W16")}
_WOB = [False]
_IDX: set = set()
_WARNED: set = set()


def mark(name: str, on: bool, why: str) -> None:
    _STATE[name]["on"] = bool(on)
    _STATE[name]["why"] = why


def line(name: str) -> str:
    state = _STATE[name]
    return f"bundle-v9 king-v6 {name}: {'ON' if state['on'] else 'off'} ({state['why']})"


def say(name: str) -> None:
    print(line(name), file=sys.stderr, flush=True)


def _runtime_off(name: str, error: Exception) -> None:
    mark(name, False, f"disabled at run time, the king's path from here on: {type(error).__name__}: {str(error)[:300]}")
    if name not in _WARNED:
        _WARNED.add(name)
        say(name)


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


def module():
    """The compiled glue module (switched-on copies only). Built once; a build failure is kept and re-raised."""
    if _MOD:
        return _MOD[0]
    if _ERR:
        raise RuntimeError(f"the glue PDL module failed to build earlier: {_ERR[0]}")
    try:
        from sglang.kernels.jit.utils import load_jit
        from sglang.kernels.jit.utils.compile.cache import cache_root

        from bundle_v9.dec import cuda_glue_pdl

        on = dict(qnorm=QNORM_PDL_ON, wob=WOBSF_PDL_ON, idx=IDX_W16_ON)
        data = cuda_glue_pdl.source(qnorm_rows=QNORM_PDL_ROWS, **on).encode()
        digest = hashlib.sha256(data).hexdigest()[:16]
        directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-glue-pdl", digest)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, cuda_glue_pdl.NAME)
        if not os.path.isfile(path):
            _write(path, data)
        mod = load_jit("dec_glue_pdl", digest, cuda_files=[path], cuda_wrappers=cuda_glue_pdl.wrappers(**on))
    except Exception as error:
        _ERR.append(f"{type(error).__name__}: {str(error)[:400]}")
        raise
    _MOD.append(mod)
    return mod


# ---------------------------------------------------------------- WOBSF-PDL


def wob_scale(input_scale: torch.Tensor):
    """The swizzled wo_b input scale from the PDL copy, or None: call the king's prefill_fuse.wob_scale."""
    if not _WOB[0]:
        return None
    try:
        if not (input_scale.dim() == 2 and input_scale.dtype == torch.int32 and input_scale.is_cuda
                and 0 < input_scale.shape[0] <= WOB_MAX_ROWS and input_scale.shape[1] == WOB_WORDS):
            return None
        rows, words = input_scale.shape
    except Exception:
        return None
    try:
        out = torch.empty(((rows + 127) // 128) * 128 * words * 4, dtype=torch.uint8, device=input_scale.device)
        module().wob_scale(input_scale, out, 0)
    except Exception as error:
        _WOB[0] = False
        _runtime_off("WOBSF-PDL", error)
        return None
    return out


def _wob_layouts(rows: int, device, gen):
    from bundle_v9.dec import decode_gemm

    _, served = decode_gemm._random_pair(rows, 8192, device, gen)
    flat = torch.randint(-(1 << 31), (1 << 31) - 1, (rows, WOB_WORDS), device=device, generator=gen, dtype=torch.int32)
    return (("DeepGEMM scale layout", served), ("row-major", flat))


def prepare_wobsf(device) -> None:
    """Prove the PDL swizzle against the king's; publish on a full pass. Never raises."""
    _WOB[0] = False
    if not WOBSF_PDL_ON:
        mark("WOBSF-PDL", False, "switched off by glue_pdl.WOBSF_PDL_ON; the king's wo_b scale swizzle")
        say("WOBSF-PDL")
        return
    try:
        from bundle_v9.dec import prefill_fuse

        mod, king = module(), prefill_fuse.module()
        gen = torch.Generator(device=device).manual_seed(SEED)
        cases = 0
        for rows in range(1, WOB_MAX_ROWS + 1):
            for name, scale in _wob_layouts(rows, device, gen):
                if tuple(scale.shape) != (rows, WOB_WORDS) or scale.dtype != torch.int32:
                    raise RuntimeError(f"the {name} proof scale at {rows} rows is {tuple(scale.shape)} {scale.dtype}")
                size = ((rows + 127) // 128) * 128 * WOB_WORDS * 4
                want = torch.full((size,), 0x5A, dtype=torch.uint8, device=device)
                got = torch.full((size,), 0xA5, dtype=torch.uint8, device=device)
                king.wob_scale(scale, want, 0)
                mod.wob_scale(scale, got, 0)
                if not torch.equal(want, got):
                    raise RuntimeError(f"the PDL swizzle differs from the king's at {rows} rows ({name})")
                cases += 1
        torch.cuda.synchronize(device)
    except Exception as error:
        mark("WOBSF-PDL", False, f"FAIL, the king's wo_b scale swizzle: {type(error).__name__}: {str(error)[:300]}")
        say("WOBSF-PDL")
        return
    _WOB[0] = True
    mark("WOBSF-PDL", True, f"decode wo_b scale swizzle at 1..{WOB_MAX_ROWS} rows on its PDL copy (wait before the "
                            f"first read, CuTe wo_b released after it); bytes proven against the king's on {cases} cases")


def wobsf_on() -> bool:
    return _WOB[0]


def fail_wobsf(why: str) -> None:
    _WOB[0] = False
    mark("WOBSF-PDL", False, f"FAIL, the king's wo_b scale swizzle: {why[:400]}")
    say("WOBSF-PDL")


def report_wobsf() -> None:
    """The ON line, once decode_gemm's own wo_b proof has passed with the PDL swizzle serving (off lines print earlier)."""
    if _WOB[0]:
        say("WOBSF-PDL")


# ---------------------------------------------------------------- IDX-W16


def index_q(input, freqs, positions, head_weights, weight_scale):
    """(payload, scale, bf16 weights) from the PDL bf16 copy of the king's index_q replica, or None: call stock."""
    if not _IDX:
        return None
    try:
        if not (type(weight_scale) is float and weight_scale in _IDX
                and input.dim() == 3 and tuple(input.shape[1:]) == (IDX_HEADS, IDX_DIM) and input.shape[0] > 0
                and input.dtype == torch.bfloat16 and input.is_contiguous() and input.data_ptr() % 16 == 0
                and freqs.dtype == torch.float32 and freqs.dim() == 2 and freqs.shape[1] == 64
                and freqs.is_contiguous() and freqs.data_ptr() % 16 == 0
                and positions.dim() == 1 and positions.shape[0] == input.shape[0]
                and positions.dtype in (torch.int32, torch.int64) and positions.is_contiguous()
                and head_weights.dtype == torch.bfloat16 and head_weights.is_contiguous()
                and tuple(head_weights.shape) == (input.shape[0], IDX_HEADS)
                and input.device == freqs.device == positions.device == head_weights.device):
            return None
    except Exception:
        return None
    try:
        tokens = input.shape[0]
        rows = tokens * IDX_HEADS
        payload = input.new_empty((rows, IDX_DIM // 2), dtype=torch.int8)
        scale = input.new_empty((rows,), dtype=torch.int32)
        weights = input.new_empty((tokens, IDX_HEADS), dtype=torch.bfloat16)
        module().index_q(input, freqs, positions, payload, scale, head_weights, weights, weight_scale)
    except Exception as error:
        _IDX.clear()
        _runtime_off("IDX-W16", error)
        return None
    return payload, scale, weights


def _idx_case(tokens: int, family: int, pos_dtype, gen, device, table: int):
    x = torch.randn((tokens, IDX_HEADS, IDX_DIM), device=device, generator=gen, dtype=torch.float32)
    w = torch.randn((tokens, IDX_HEADS), device=device, generator=gen, dtype=torch.float32)
    if family == 1:
        x = x * torch.pow(10.0, torch.rand((tokens, IDX_HEADS, 1), device=device, generator=gen) * 8 - 4)
        w = w * torch.pow(10.0, torch.rand((tokens, IDX_HEADS), device=device, generator=gen) * 8 - 4)
    elif family == 2:
        x = x * 3
        x[tokens // 2].zero_()
        x[:, 1, :8].fill_(-0.0)
        w[tokens // 2].zero_()
        if tokens > 1:
            w[tokens - 1, :4] = torch.tensor([-0.0, 1e-30, -3e4, 65504.0], device=device)
    freqs = torch.rand((table, 64), device=device, generator=gen, dtype=torch.float32) * 2 - 1
    positions = torch.randint(0, table, (tokens,), device=device, generator=gen).to(pos_dtype)
    return x.to(torch.bfloat16), freqs, positions, w.to(torch.bfloat16)


def _prove_idx(scales, device) -> int:
    from bundle_v9.dec import index_streams

    stock = index_streams.index_q_rope_pack_weights
    if (stock.__module__ != "sglang.kernels.ops.attention.dsv4.fp4_indexer_rope"
            or stock.__qualname__ != "index_q_rope_pack_weights"):
        raise RuntimeError(f"the reference is not stock's index_q_rope_pack_weights ({stock.__module__}."
                           f"{stock.__qualname__})")
    mod = module()
    gen = torch.Generator(device=device).manual_seed(SEED + 1)
    cases = 0
    for index, weight_scale in enumerate(scales):
        for tokens in (IDX_FULL_TOKENS if index == 0 else IDX_LAYER_TOKENS):
            for family in (range(IDX_FAMILIES) if index == 0 else (0,)):
                for pos_dtype in ((torch.int64, torch.int32) if index == 0 else (torch.int64,)):
                    x, freqs, positions, w = _idx_case(tokens, family, pos_dtype, gen, device, 4096)
                    want_p, want_s, want_w = stock(x, freqs, positions, w, weight_scale)
                    rows = tokens * IDX_HEADS
                    got_p = torch.full((rows, IDX_DIM // 2), 0x5A, dtype=torch.int8, device=device)
                    got_s = torch.full((rows,), 0x5A5A5A5A, dtype=torch.int32, device=device)
                    got_w = torch.full((tokens, IDX_HEADS), float("nan"), dtype=torch.bfloat16, device=device)
                    mod.index_q(x, freqs, positions, got_p, got_s, w, got_w, weight_scale)
                    where = f"{tokens} tokens, family {family}, {pos_dtype}, scale {weight_scale!r}"
                    if want_w.dtype != torch.float32 or tuple(want_w.shape) != (tokens, IDX_HEADS):
                        raise RuntimeError(f"stock's head weights are {want_w.dtype} {tuple(want_w.shape)} ({where})")
                    words = want_w.view(torch.int32)
                    if bool((words & 0xFFFF).ne(0).any()):
                        raise RuntimeError(f"stock's fp32 head weights are not exact bf16 values ({where})")
                    if not (torch.equal(got_p.view(torch.uint8), want_p.view(torch.uint8))
                            and torch.equal(got_s, want_s.view(torch.int32))):
                        raise RuntimeError(f"index_q payload or scale differs from stock ({where})")
                    if not (torch.equal(got_w.view(torch.int16), (words >> 16).to(torch.int16))
                            and torch.equal(got_w.view(torch.int16), want_w.to(torch.bfloat16).view(torch.int16))):
                        raise RuntimeError(f"index_q bf16 head weights differ from stock's fp32 weights ({where})")
                    cases += 1
    torch.cuda.synchronize(device)
    return cases


def prepare_idx(indexers) -> None:
    """Prove IDX-W16 on the served index layers' own head-weight scales; publish on a full pass. Never raises."""
    _IDX.clear()
    if not IDX_W16_ON:
        mark("IDX-W16", False, "switched off by glue_pdl.IDX_W16_ON; stock index_q (fp32 head weights)")
        say("IDX-W16")
        return
    try:
        scales, device = [], None
        for indexer in indexers:
            scale = indexer.head_weight_scale
            if type(scale) is not float:
                raise RuntimeError(f"an index layer's head_weight_scale is {type(scale).__name__}, not float")
            if indexer.n_local_heads != IDX_HEADS or indexer.index_head_dim != IDX_DIM:
                raise RuntimeError(f"an index layer has {indexer.n_local_heads} heads of {indexer.index_head_dim}")
            if scale not in scales:
                scales.append(scale)
            device = next(indexer.parameters()).device
        if not scales:
            raise RuntimeError("no served index layer")
        cases = _prove_idx(scales, device)
    except Exception as error:
        _IDX.clear()
        mark("IDX-W16", False, f"FAIL, stock index_q: {type(error).__name__}: {str(error)[:300]}")
        say("IDX-W16")
        return
    _IDX.update(scales)
    mark("IDX-W16", True, f"decode / verify indexer index_q on the PDL copy of the king's replica writing bf16 head "
                          f"weights (the consumers' bf16 copies become no-ops); bytes proven against stock on {cases} "
                          f"cases over {len(scales)} head-weight scale(s)")
    say("IDX-W16")
