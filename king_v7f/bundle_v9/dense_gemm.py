
from __future__ import annotations

import contextlib
import hashlib
import os

import torch
import torch.distributed as dist

from bundle_v9 import cuda_swizzle, tactic_pins

FLASHINFER_VER = "0.7.2"
PAIRED = ("wo_b",)
PROBE_SEED = 20261031

TACTICS = (
    (("decode", "null"), "wo_b", 5120, 8192, 1, 15, ((128, 8), (1, 1), True, True, 1)),
    (("decode", "null"), "wo_b", 5120, 8192, 16, 30, ((128, 16), (1, 1), True, True, 1)),
    (("decode", "null"), "wo_b", 5120, 8192, 31, 102, ((256, 64), (2, 2), True, True, 1)),
    (("decode", "null"), "wo_b", 5120, 8192, 103, 256, ((256, 128), (2, 4), False, True, 1)),
    (("decode", "null"), "wq_b", 32768, 1280, 1, 24, ((128, 32), (4, 1), True, False, 1)),
    (("decode", "null"), "wq_b", 32768, 1280, 25, 60, ((128, 64), (1, 1), True, False, 1)),
    (("decode", "null"), "wq_b", 32768, 1280, 61, 88, ((128, 128), (1, 4), False, False, 1)),
    (("decode", "null"), "wq_b", 32768, 1280, 89, 108, ((256, 128), (4, 1), True, False, 1)),
    (("decode", "null"), "wq_b", 32768, 1280, 109, 126, ((128, 128), (1, 4), False, False, 1)),
    (("decode", "null"), "wq_b", 32768, 1280, 127, 186, ((256, 192), (2, 1), True, False, 1)),
    (("decode", "null"), "wq_b", 32768, 1280, 187, 256, ((256, 256), (2, 2), False, False, 1)),
    (("decode", "null"), "wq_b", 32768, 1280, 257, 288, ((256, 192), (2, 2), True, False, 1)),
    (("decode", "null"), "wkv", 25600, 6144, 36, 66, ((256, 64), (4, 1), True, False, 1)),
    (("decode", "null"), "wkv", 25600, 6144, 67, 126, ((128, 192), (1, 1), False, False, 1)),
    (("decode", "null"), "wkv", 25600, 6144, 127, 240, ((256, 192), (2, 1), False, False, 1)),
    (("decode", "null"), "wkv", 25600, 6144, 241, 288, ((128, 192), (1, 2), True, False, 1)),
    (("decode", "null"), "idx_wq_b", 4096, 1280, 36, 216, ((128, 64), (1, 1), True, False, 1)),
    (("decode", "null"), "idx_wq_b", 4096, 1280, 253, 288, ((128, 128), (1, 1), False, True, 1)),
)

_STATE: list = []


def _write_atomic(path: str, data: bytes) -> None:
    staging = f"{path}.{os.getpid()}.tmp"
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(staging, path)


def _swizzle_header() -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    data = cuda_swizzle.SOURCE.encode("utf-8")
    folder = os.path.join(
        os.path.abspath(os.fspath(cache_root())),
        "bundle-v9-dense-gemm",
        hashlib.sha256(data).hexdigest()[:16],
    )
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, cuda_swizzle.NAME)
    if not os.path.isfile(path):
        _write_atomic(path, data)
    return path


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _seeded_rows(rows: int, k: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, k, generator=g, device=device, dtype=torch.float32)
    row = torch.pow(2.0, (torch.arange(rows, device=device) * 5 % 15 - 7).float())
    grp = torch.pow(2.0, (torch.arange(k // 32, device=device) * 3 % 11 - 5).float()).repeat_interleave(32)
    return (x * row[:, None] * grp[None, :]).to(torch.bfloat16)


def _resolve_linear(layer, name: str):
    attn = getattr(layer, "self_attn", None)
    if name == "wqkv_a":
        return False, None if attn is None else getattr(attn, "wqkv_a", None), None
    if name == "wo_b":
        return False, None if attn is None else getattr(attn, "wo_b", None), None
    if name == "wq_b":
        return False, None if attn is None else getattr(attn, "wq_b", None), getattr(attn, "q_norm", None)
    if name == "idx_wq_b":
        idx = None if attn is None else getattr(attn, "indexer", None)
        return True, None if idx is None else getattr(idx, "wq_b", None), None
    if name == "wkv":
        eg = getattr(layer, "engram", None)
        return True, None if eg is None else getattr(eg, "wkv", None), None
    raise RuntimeError(f"bundle-v9 DENSE: unknown linear target {name}")


def _check_linear(i: int, name: str, lin, shape: tuple[int, int]) -> None:
    method = getattr(lin, "quant_method", None)
    backend = getattr(method, "mxfp8_dense_backend", None)
    w = getattr(lin, "weight", None)
    if type(method).__name__ != "Fp8LinearMethod" or not getattr(method, "block_fp8_as_mxfp8", False):
        raise TypeError(f"bundle-v9 DENSE layer {i} {name}: {type(method).__name__} is not block-fp8 MXFP8")
    if backend is None or not backend.is_flashinfer_cutedsl() or not getattr(lin, "block_fp8_mxfp8_ready", False):
        raise TypeError(f"bundle-v9 DENSE layer {i} {name}: backend {backend} is not FlashInfer CuTe-DSL MXFP8-ready")
    if w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2 or tuple(w.shape) != shape:
        raise TypeError(f"bundle-v9 DENSE layer {i} {name}: weight {None if w is None else tuple(w.shape)} != {shape}")
    if not isinstance(getattr(lin, "weight_scale_inv_swizzled", None), torch.Tensor):
        raise TypeError(f"bundle-v9 DENSE layer {i} {name}: missing weight_scale_inv_swizzled")
    if name in PAIRED and not (
        getattr(lin, "deepgemm_mxfp8_input", False)
        and w.shape[1] % 128 == 0
        and isinstance(getattr(lin, "weight_scale_inv_deepgemm", None), torch.Tensor)
    ):
        raise TypeError(f"bundle-v9 DENSE layer {i} {name}: missing DeepGEMM MXFP8 pair setup")


class LinearTarget:

    def __init__(self, layer: int, name: str, module, q_norm, entries: list[tuple[int, int, int, tuple]]):
        self.layer, self.name, self.paired = layer, name, name in PAIRED
        self.module, self.q_norm = module, q_norm
        self.weight = module.weight
        self.n, self.k = int(self.weight.shape[0]), int(self.weight.shape[1])
        self.entries = entries

    def match(self, rows: int):
        for entry_id, lo, hi, tactic in self.entries:
            if lo <= rows <= hi:
                return entry_id, tactic
        return None


class _MxtState:

    def __init__(self, *, fp8, fp8u, by_ptr, entries, cute_runner, ws, swizzle_mod):
        self.fp8, self.fp8u, self.by_ptr = fp8, fp8u, by_ptr
        self.entries = entries
        self.cute_runner, self.ws, self.swizzle_mod = cute_runner, ws, swizzle_mod
        self.dg_stock = fp8.deepgemm_w8a8_mxfp8_linear_with_fallback
        self.fi_stock = fp8u.flashinfer_mm_mxfp8
        self.depth = 0
        self.probe_hits: list[int] | None = None

    def _swizzle(self, input_scale: torch.Tensor, rows: int, cols: int, stride: int) -> torch.Tensor:
        tiles = (rows + 127) // 128
        out_u8 = torch.empty((tiles * cols * 512,), dtype=torch.uint8, device=input_scale.device)
        self.swizzle_mod.run(input_scale, out_u8, rows, cols, stride)
        return out_u8

    def _exec(self, a: torch.Tensor, b_t: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor,
              n_out: int, rows: int, tactic: tuple) -> torch.Tensor:
        out = torch.empty((rows, n_out), dtype=torch.bfloat16, device=a.device)
        a_u8 = a_sf if a_sf.dtype == torch.uint8 else a_sf.view(torch.uint8)
        b_u8 = b_sf if b_sf.dtype == torch.uint8 else b_sf.view(torch.uint8)
        self.cute_runner(inputs=[a, b_t, a_u8, b_u8, torch.bfloat16, out, self.ws], tactic=tactic)
        return out

    def deepgemm(self, input, weight, weight_scale, input_scale=None, bias=None, weight_scale_swizzled=None):
        rec = self.by_ptr.get(weight.data_ptr())
        if (
            rec is None or not rec.paired or input_scale is None or bias is not None
            or weight_scale_swizzled is not None or input.ndim != 2
        ):
            return self.dg_stock(
                input, weight, weight_scale, input_scale=input_scale, bias=bias,
                weight_scale_swizzled=weight_scale_swizzled,
            )
        rows, k = int(input.shape[0]), int(input.shape[1])
        hit = rec.match(rows)
        cols, stride = k // 128, (rows + 3) // 4 * 4
        if (
            hit is None or k != rec.k or input.dtype != torch.float8_e4m3fn or not input.is_contiguous()
            or input_scale.dtype != torch.int32 or tuple(input_scale.shape) != (rows, cols)
            or input_scale.stride() != (1, stride) or input_scale.device != input.device
            or (rows >= 1024 and _capturing())
        ):
            return self.dg_stock(
                input, weight, weight_scale, input_scale=input_scale, bias=bias,
                weight_scale_swizzled=weight_scale_swizzled,
            )
        entry_id, tactic = hit
        swizzled_u8 = self._swizzle(input_scale, rows, cols, stride)
        w_sf = rec.module.weight_scale_inv_swizzled
        w_sf_arg = w_sf.t() if w_sf.ndim == 2 else w_sf
        out = self._exec(input, weight.t(), swizzled_u8, w_sf_arg, rec.n, rows, tactic)
        if self.probe_hits is not None:
            self.probe_hits.append(entry_id)
        return out

    def flashinfer(self, q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, use_8x4_sf_layout=False,
                   backend="auto"):
        rec = self.by_ptr.get(weight_t.data_ptr())
        if (
            rec is None or rec.paired or backend != "cute-dsl" or use_8x4_sf_layout or out_dtype != torch.bfloat16
            or q_input.ndim != 2 or q_input.dtype != torch.float8_e4m3fn or not q_input.is_contiguous()
            or tuple(weight_t.shape) != (rec.k, rec.n) or int(q_input.shape[1]) != rec.k
        ):
            return self.fi_stock(
                q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                use_8x4_sf_layout=use_8x4_sf_layout, backend=backend,
            )
        rows = int(q_input.shape[0])
        hit = rec.match(rows)
        if hit is None or (rows >= 1024 and _capturing()):
            return self.fi_stock(
                q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                use_8x4_sf_layout=use_8x4_sf_layout, backend=backend,
            )
        entry_id, tactic = hit
        out = self._exec(q_input, weight_t, x_scale_u8, weight_scale_t, rec.n, rows, tactic)
        if self.probe_hits is not None:
            self.probe_hits.append(entry_id)
        return out


@contextlib.contextmanager
def routed(forward_batch=None):
    if not _STATE:
        raise RuntimeError("bundle-v9 DENSE: routed() entered before prepare()")
    st = _STATE[0]
    mode = getattr(forward_batch, "forward_mode", None)
    if not st.entries or mode is None or not (mode.is_decode() or mode.is_target_verify()):
        yield
        return
    with _installed(st):
        yield


@contextlib.contextmanager
def probe_routes():
    if not _STATE:
        raise RuntimeError("bundle-v9 DENSE: probe_routes() entered before prepare()")
    st = _STATE[0]
    if not st.entries:
        yield
        return
    with _installed(st):
        yield


@contextlib.contextmanager
def _installed(st):
    outer = st.depth == 0
    if outer:
        if (
            st.fp8.deepgemm_w8a8_mxfp8_linear_with_fallback is not st.dg_stock
            or st.fp8u.flashinfer_mm_mxfp8 is not st.fi_stock
        ):
            raise RuntimeError("bundle-v9 DENSE: fp8 / fp8_utils MXFP8 functions changed since prepare")
        st.fp8.deepgemm_w8a8_mxfp8_linear_with_fallback = st.deepgemm
        st.fp8u.flashinfer_mm_mxfp8 = st.flashinfer
    st.depth += 1
    try:
        yield
    finally:
        st.depth -= 1
        if outer:
            st.fp8.deepgemm_w8a8_mxfp8_linear_with_fallback = st.dg_stock
            st.fp8u.flashinfer_mm_mxfp8 = st.fi_stock


def _same_i16(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and bool(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)))
    )


@torch.no_grad()
def prepare(module, role: str) -> tuple[int, int]:
    import flashinfer
    from flashinfer.gemm import gemm_base
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args
    from sglang.kernels.ops.layernorm.rmsnorm_mxfp8 import rmsnorm_mxfp8
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
    from sglang.srt.layers.quantization import fp8 as sgl_fp8
    from sglang.srt.layers.quantization import fp8_utils as sgl_fp8_utils
    from sglang.srt.models.deepseek_v4_mhc import Mxfp8SwizzledInput
    from sglang.srt.runtime_context import get_parallel

    _STATE.clear()
    tactic_pins.verify_mxt()
    if flashinfer.__version__ != FLASHINFER_VER:
        raise RuntimeError(f"bundle-v9 DENSE requires flashinfer {FLASHINFER_VER}, got {flashinfer.__version__}")
    if _capturing():
        raise RuntimeError("bundle-v9 DENSE prepare requires an eager CUDA stream")
    if sgl_fp8.deepgemm_w8a8_mxfp8_linear_with_fallback is not sgl_fp8_utils.deepgemm_w8a8_mxfp8_linear_with_fallback:
        raise RuntimeError("bundle-v9 DENSE: fp8.deepgemm_w8a8_mxfp8_linear_with_fallback is not stock's")

    entries = [(name, n, k, lo, hi, tactic) for roles, name, n, k, lo, hi, tactic in TACTICS if role in roles]
    if not entries:
        _STATE.append(_MxtState(fp8=sgl_fp8, fp8u=sgl_fp8_utils, by_ptr={}, entries=[],
                                cute_runner=None, ws=None, swizzle_mod=None))
        return 0, 0

    grouped: dict[str, list] = {}
    for eid, (name, n, k, lo, hi, tactic) in enumerate(entries):
        grouped.setdefault(name, []).append((eid, n, k, lo, hi, tactic))

    by_ptr: dict[int, LinearTarget] = {}
    by_name: dict[str, list[LinearTarget]] = {}
    dev = None
    for name, mine in sorted(grouped.items()):
        shapes = {(n, k) for _, n, k, *_ in mine}
        if len(shapes) != 1:
            raise RuntimeError(f"bundle-v9 DENSE: inconsistent shapes for {name}: {shapes}")
        shape = next(iter(shapes))
        recs: list[LinearTarget] = []
        for i, layer in enumerate(module.layers):
            optional, lin, q_norm = _resolve_linear(layer, name)
            if lin is None:
                if optional:
                    continue
                raise RuntimeError(f"bundle-v9 DENSE: layer {i} missing {name}")
            if name == "wq_b" and q_norm is None:
                raise RuntimeError(f"bundle-v9 DENSE: layer {i} wq_b missing self_attn.q_norm")
            _check_linear(i, name, lin, shape)
            rec = LinearTarget(i, name, lin, q_norm, [(eid, lo, hi, t) for eid, _, _, lo, hi, t in mine])
            key_ptr = int(rec.weight.data_ptr())
            if int(rec.weight.t().data_ptr()) != key_ptr or key_ptr in by_ptr:
                raise RuntimeError(f"bundle-v9 DENSE: non-unique weight pointer on layer {i} {name}")
            by_ptr[key_ptr] = rec
            recs.append(rec)
            dev = rec.weight.device
        if not recs:
            raise RuntimeError(f"bundle-v9 DENSE: no layer carries {name}")
        by_name[name] = recs

    major, minor = torch.cuda.get_device_capability(dev)
    cute_runner = gemm_base._cute_dsl_gemm_mxfp8_runner(major, minor, True, torch.bfloat16)
    ws = gemm_base._get_cache_buf("mm_mxfp8_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, dev)
    swizzle_mod = None
    if any(r.paired for r in by_ptr.values()):
        cpp = make_cpp_args(bool(is_arch_support_pdl()))
        swizzle_mod = load_jit(
            "v4_swizzle_128x4",
            *cpp,
            cuda_files=[_swizzle_header()],
            cuda_wrappers=[("run", f"sglang::bundle_v9::SwizzleColTo128x4<{cpp}>::run")],
        )

    st = _MxtState(
        fp8=sgl_fp8, fp8u=sgl_fp8_utils, by_ptr=by_ptr, entries=entries,
        cute_runner=cute_runner, ws=ws, swizzle_mod=swizzle_mod,
    )
    _STATE.append(st)

    group = get_parallel().tp_group
    bad: list[str] = []
    checks = 0
    try:
        for entry_id, (name, _, _, lo, hi, _) in enumerate(entries):
            recs = by_name[name]
            sample = [recs[0]] if len(recs) == 1 else [recs[0], recs[-1]]
            if hi <= 512:
                mid = next((r for r in range(lo + 1, hi) if r % 128), None)
                row_points = sorted({lo, hi} | ({mid} if mid is not None else set()))
            else:
                row_points = sorted({lo, min(hi, 2048)})
            for rec in sample:
                for rows in row_points:
                    x = _seeded_rows(rows, rec.k, PROBE_SEED + 7919 * rec.layer + 131 * entry_id + rows, dev)
                    modes = ("swizzled", "bf16") if (name == "wq_b" and rows == row_points[0]) else (
                        ("swizzled",) if name == "wq_b" else (("paired",) if rec.paired else ("bf16",))
                    )
                    for mode in modes:
                        def _make_arg():
                            if mode == "paired":
                                return sglang_per_token_group_quant_fp8(
                                    x, 32, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True
                                )
                            if mode == "swizzled":
                                _, q_fp8, q_sf = rmsnorm_mxfp8(x, rec.q_norm.weight, rec.q_norm.variance_epsilon)
                                return Mxfp8SwizzledInput(q_fp8, q_sf)
                            return x

                        ref = rec.module.quant_method.apply(rec.module, _make_arg())
                        st.probe_hits = []
                        try:
                            with _installed(st):
                                mine = rec.module.quant_method.apply(rec.module, _make_arg())
                            hit_ok = st.probe_hits == [entry_id]
                        finally:
                            st.probe_hits = None
                        checks += 1
                        if not (hit_ok and _same_i16(ref, mine)):
                            why = "unreached" if not hit_ok else "diff"
                            bad.append(f"L{rec.layer}:{name}[{lo}..{hi}] r={rows}({mode}):{why}")
                            break
                    if bad:
                        break
                if bad:
                    break
        if torch.cuda.is_available():
            torch.cuda.synchronize(dev)
    except Exception as exc:
        bad.append(f"{type(exc).__name__}: {exc}")

    gathered = [None] * group.world_size
    dist.all_gather_object(gathered, bad[:4], group=group.cpu_group)
    failures = [f"rank {r}: {', '.join(errs)}" for r, errs in enumerate(gathered) if errs]
    if failures:
        _STATE.clear()
        raise RuntimeError(f"bundle-v9 DENSE STRICT byte-probe failed ({' | '.join(failures)})")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return len(entries), checks
