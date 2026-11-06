
import contextlib
import hashlib
import os
from functools import lru_cache

from flashinfer.experimental.deepgemm_mega_mhc import mega_mhc as mmhc

PROGRAMS = {
    "cake_mega_mhc_557f419314de9efca58e": (
        "127ed78e085901310a9d1e05fd48d6b4465dc8a22461824092926b190778e10c",
        "1f0e2b46c28b03162202dbd706286d1f71c7270ac9d1ef1a9a5fdc5fc0c48270",
    ),
    "cake_mega_mhc_7d69b17877f1cfa74fef": (
        "74fa30ecd1762322c84923e16c2c089a9dfcdc26aab04b46e17a4ee867d019b6",
        "7c25f52364dcfaac84c63c7219aa46b130bff0b3223f2da610cb03c82ea8a35f",
    ),
    "cake_mega_mhc_b9fd6d16636225d37ec9": (
        "e6930155051e4324b8187fb45be3a873dbaaa12d9febf49e1808235fdf71c888",
        "5c2e5c514780983a08bd4ebd3d7013b0e49fdf2b2f62de6ebdb51baa7c0724a2",
    ),
}
PREFETCH = (
    "unsigned int prefetch = kb_2 + 1 == 80 / NUM_SPLITS + ((task_2 % NUM_SPLITS < 80 % NUM_SPLITS) ? 1 : 0) && ",
    "unsigned int prefetch = kb_2 == 0 && ",
)
ATOM = "for (int atom = 0; atom < 8; atom += 2) {"
POLL = "} while (static_cast<unsigned long long>(_acquire_observed - "
DG = (
    "77fe3a79ad04c803fd24364f90e6f3b86b45ba8cbd67097b7ba928550c478f68",
    "c0b569280c9a486f09d92730061ff3a7c0b3244ff61828ad9ccdd7ecd932059d",
)
DG_PREFETCH = (
    "k_block_idx == num_task_k_blocks - 1 and task_idx + kNumSMs < num_tasks;",
    "k_block_idx == 0 and task_idx + kNumSMs < num_tasks;",
)
DG_ATOM = "for (uint32_t umma_k_idx = kPostWorkerIdx; umma_k_idx < BLOCK_K / UMMA_K;"
DG_UNPACK = (
    "        const auto run_post_worker = [&](auto post_worker) {",
    "        const auto unpack_bf16x2 = [](const nv_bfloat162& v) {\n"
    "            const uint32_t u = *reinterpret_cast<const uint32_t*>(&v);\n"
    "            return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));\n"
    "        };\n\n"
    "        const auto run_post_worker = [&](auto post_worker) {",
)
DG_ROWS = 2048
_STOCK: list = []
_BUILT: set = set()
_DG: list = []
_DG_CALLS = [0]


def _read(path: str) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 22)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


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


def _image_source(name: str) -> str:
    from flashinfer.jit import env

    return os.fspath(env.FLASHINFER_CSRC_DIR / mmhc.PROGRAMS[name]["sources"][0].removeprefix("csrc/"))


def _patch(text: str, prefetch=PREFETCH, atom=ATOM, unroll="1") -> str:
    if text.count(prefetch[0]) != 1 or text.count(atom) != 1:
        raise RuntimeError("bundle-v9 MHC-SCHED: the image program does not carry exactly one edit site of each kind")
    lines = text.replace(prefetch[0], prefetch[1]).split("\n")
    at = next(i for i, line in enumerate(lines) if line.strip() == atom)
    if lines[at - 1].strip() != "#pragma unroll":
        raise RuntimeError("bundle-v9 MHC-SCHED: the atom loop is not preceded by its unroll pragma")
    lines[at - 1] = lines[at - 1].replace("#pragma unroll", f"#pragma unroll {unroll}")
    return "\n".join(lines)


def _cake_patch(text: str) -> str:
    lines = _patch(text).split("\n")
    sites = [i for i, line in enumerate(lines) if line.strip().startswith(POLL) and line.endswith(");")]
    if len(sites) != 4:
        raise RuntimeError("bundle-v9 MHC-SCHED: the image program does not carry the measured poll sites")
    for i in sites:
        head, cond = lines[i].split("} while (", 1)
        lines[i] = head + "} while ((" + cond[:-2] + ") && (__nanosleep(256), true));"
    return "\n".join(lines)


def _dg_patch(text: str) -> str:
    text = _patch(text, DG_PREFETCH, DG_ATOM, "2")
    if text.count(DG_UNPACK[0]) != 1 or text.count("__bfloat1622float2(") != 3:
        raise RuntimeError("bundle-v9 MHC-SCHED: the DeepGEMM mega mHC does not carry the measured unpack sites")
    return text.replace(DG_UNPACK[0], DG_UNPACK[1]).replace("__bfloat1622float2(", "unpack_bf16x2(")


@lru_cache(maxsize=None)
def _source(name: str) -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    image_sha, patched_sha = PROGRAMS[name]
    data = _read(_image_source(name))
    if hashlib.sha256(data).hexdigest() != image_sha:
        raise RuntimeError(f"bundle-v9 MHC-SCHED: the image's {name} is not the source the schedule was measured on")
    patched = _cake_patch(data.decode()).encode()
    if hashlib.sha256(patched).hexdigest() != patched_sha:
        raise RuntimeError(f"bundle-v9 MHC-SCHED: the scheduled {name} is not the measured source")
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9", "mhc-sched", patched_sha[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}_kernel.cu")
    if not (os.path.isfile(path) and hashlib.sha256(_read(path)).hexdigest() == patched_sha):
        _write(path, patched)
    return path


@lru_cache(maxsize=None)
def _build(name, arch, num_sms, defines):
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

    binding = mmhc.PROGRAMS[name]["sources"][1]
    values = (*defines, ("NUM_SMS", num_sms))
    spec = gen_jit_spec(
        name="_".join([name, "s" + PROGRAMS[name][1][:8], arch, *(f"{key.lower()}{value}" for key, value in values)]),
        sources=[_source(name), env.FLASHINFER_CSRC_DIR / binding.removeprefix("csrc/")],
        extra_cuda_cflags=[*{"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch], *mmhc.COMPILE_FLAGS,
                           *(f"-D{key}={value}" for key, value in values),
                           "--device-entity-has-hidden-visibility=false"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR / mmhc.GENERATED_ROOT.removeprefix("csrc/"),
                             env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,
    )
    loaded = spec.build_and_load(), str(spec.get_library_path())
    _BUILT.add(name)
    return loaded


def _load(name, arch, num_sms, defines):
    if name in PROGRAMS:
        return _build(name, arch, num_sms, defines)
    return _STOCK[0](name, arch, num_sms, defines)


@contextlib.contextmanager
def scheduled():
    if not _STOCK:
        raise RuntimeError("bundle-v9 MHC-SCHED: used before prepare")
    if mmhc.load_program is _load:
        yield
        return
    if mmhc.load_program is not _STOCK[0]:
        raise RuntimeError("bundle-v9 MHC-SCHED: the cake loader changed since prepare")
    mmhc.load_program = _load
    try:
        yield
    finally:
        mmhc.load_program = _STOCK[0]


def prepare() -> None:
    if _STOCK:
        return
    fn = mmhc.load_program
    if fn.__module__ != mmhc.__name__ or fn.__qualname__ != "load_program":
        raise RuntimeError(f"bundle-v9 MHC-SCHED: the cake loader is not the image's ({fn.__module__}.{fn.__qualname__})")
    for name in PROGRAMS:
        _source(name)
    _STOCK.append(fn)


def verify() -> None:
    missing = set(PROGRAMS) - _BUILT
    if missing:
        raise RuntimeError(f"bundle-v9 MHC-SCHED: programs {sorted(missing)} were not built from the scheduled sources")


def dg_prepare() -> None:
    if _DG:
        return
    import deep_gemm

    from bundle_v9 import moe_prefill

    rel = os.path.join("include", "deep_gemm", "impls", "sm100_mega_mhc.cuh")
    data = _read(os.path.join(os.path.dirname(deep_gemm.__file__), rel))
    if hashlib.sha256(data).hexdigest() != DG[0]:
        raise RuntimeError("bundle-v9 MHC-SCHED: the image's DeepGEMM mega mHC is not the source the schedule was measured on")
    patched = _dg_patch(data.decode()).encode()
    if hashlib.sha256(patched).hexdigest() != DG[1]:
        raise RuntimeError("bundle-v9 MHC-SCHED: the scheduled DeepGEMM mega mHC is not the measured source")
    path = os.path.join(moe_prefill.source_root(), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not (os.path.isfile(path) and hashlib.sha256(_read(path)).hexdigest() == DG[1]):
        _write(path, patched)
    _DG.append(moe_prefill.ops())


def dg_serves(x) -> bool:
    from bundle_v9 import moe_prefill

    return bool(_DG) and x.shape[0] >= DG_ROWS and not moe_prefill._graph_scope()


def dg_boundary(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps, sinkhorn_iters):
    import deep_gemm
    import torch

    lib = _DG[0]
    sms, pdl, det = int(deep_gemm.get_num_sms()), bool(deep_gemm.get_pdl()), bool(deep_gemm.get_deterministic_algorithms())
    if int(lib.get_num_sms()) != sms:
        lib.set_num_sms(sms)
    if bool(lib.get_pdl()) != pdl:
        lib.set_pdl(pdl)
    if bool(lib.get_deterministic()) != det:
        lib.set_deterministic(det)
    rows, hc = x.shape[0], residual.shape[1]
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    next_pre = torch.empty((rows, hc, 1), device=x.device, dtype=torch.float32)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty_like(comb)
    lib.mega_mhc(x, residual, pre.unsqueeze(-1), post.unsqueeze(-1), comb, fn, scale, base, hc, rms_eps, hc_eps, 2.0,
                 hc_eps, sinkhorn_iters, weight, norm_eps, 1.0, updated, next_pre, next_post, next_comb, normalized,
                 None, None, None, None, 0)
    _DG_CALLS[0] += 1
    return updated, normalized, (next_pre.squeeze(-1), next_post.squeeze(-1), next_comb)
