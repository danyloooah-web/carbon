"""dsv41-pd072 VATTN (dspd-v59): the target-verify MegaAttention (fused Q RoPE + sparse attention + O RoPE + FP8
cast) through FlashMLA's 2026.09.30 kernel, ported back to the block-split V4.1 page layout this arena's KV pools
keep, and built here through a runtime this bundle owns ("route D").

Stock: DeepseekV4MegaAttnBackend._forward_mega_decode -> flash_mla_interface.flash_mla_mega_decode(use_legacy=True)
(flashmla_mega forces the block-split KVLayout.V41, so `_swa_interleaved` is False) -> the wheel's frozen pre-09.30
kernel `legacy::...::fwd_kernel<Config{Decode, V41, V41_FP4 | V41, 64, false}>`. The wheel's current kernel (same
arithmetic, bit-identical outputs measured on equal values) is ~12 % faster but reads only the token-interleaved
layout. cuda_vattn.py holds that kernel (FlashMLA 395cb3a, MIT) with the decode KV transport ported to the
block-split pages (raw rows TMA-gathered from a raw-region tensor map, scale rows staged into the KV slot tail by
warp 11's lanes with cp.async.cg after the raw gathers); everything else is upstream's code, byte for byte the 24
files of dspd_w38/dspd/vattn_src.

Measured on the arena image (1x B300, CUDA graphs, cold rotated inputs; struct/verify-attn/NOTES.md): layers 2-39
(fp8 SWA + fp4 c2/c1 extra) 240 rows 40.9 -> 36.0 us, 288 rows 41.2 -> 36.0 us; layers 0-1 (SWA only) 15.6 -> 13.6 /
15.8 -> 13.9 us; rows sweep c1 -0.84 / -2.64 / -2.63 / -4.94 / -5.14 / -5.21 us at 6 / 36 / 120 / 180 / 240 / 288.
out_fp8, out_sf and lse equal to stock bit for bit on every rig case (row buckets 6..288, padded rows, ragged
lengths, -1 holes, padded page strides, graph replay). THE ROUTE-D BUILD BELOW HAS NEVER RUN ON A GPU.

Route D, and why. The kernel's fast conversions need PTX ISA 9.2, i.e. CUDA >= 13.2, so the engine's JIT nvcc 13.0
builds a kernel slower than stock (42.0 vs 40.6 us, which the install speed check turns off), and swapping SGLang's
process-global device_compiler_path around a load_jit build would reconfigure unrelated engine behaviour. Instead the
device unit is compiled by a DeepJIT runtime this bundle creates, whose own `Env` prefix is DSPDVA: it takes its nvcc
from DSPDVA_JIT_NVCC_COMPILER and its cubin cache from DSPDVA_JIT_CACHE_DIR, so no SGLang, DG_ or CUDA_ setting is
WRITTEN. Three kinds of setting are READ: CUDA_HOME / CUDA_PATH (cpp_extension's and DeepJIT's toolkit search), and
the global DJ_JIT_* fallbacks for any DSPDVA_JIT_* key left unset, which is why _JIT_PINS pins every one of them
the included CUDA DeepJIT headers read (eleven), absent-only. The cache-directory key is written in C++ (vattn_host.cpp init, overwrite=0); it is the one process-
environment write besides the Python ones here (lane deviation D13).
v50's M462 sets the precedent: a cpp_extension host library (with_cuda=False) whose DeepJIT compiles
the kernel (m462.ops / prepare_facts). The host library here is dsv41_pd001/cuda_vattn.py's vattn_host.cpp; the
device unit it compiles is vattn_kernel_tu.cuh. Two cubins are built, one per layer class, because DeepJIT's loader
takes a cubin with exactly one kernel.

Exactness. (1) The device code is the rig-measured kernel, bit-equal with stock on every rig case. (2) Route D does
NOT reproduce the rig's binary by construction. It keeps the rig's source, flags, arch (sm_103a) and include order
(load_jit's: SGLang's DEFAULT_INCLUDE, our csrc and kerutils, then the engine's own "cutlass" dependency resolver, so
FlashInfer's CUTLASS / CuTe comes before DeepGEMM's), but it also changes the compiler driver and cache (DeepJIT
instead of load_jit's ninja), adds DeepJIT's fixed flags (--ptxas-options=--register-usage-level=10, host -O3), and
launches through cuLaunchKernelEx itself (mirroring CUTLASS's cluster launch: an explicit 1x1x1 cluster attribute, no
PDL). Every difference those could cause is caught by the install self-test (bitwise against the engine's own
flash_mla_mega_decode on c1 / c2 / SWA at 36 and 240 rows, plus at 36 rows one edge case per class with -1 entries in
both index tensors, holes, a graph-padding row, page strides wider than the page and, with the extra cache, a zero SWA
and a zero extra top-k length, each input first held to _gates; then a speed check) and by the first-use whole-call compare
of every verdict key (out_fp8 and out_sf compared byte for byte on that call's own inputs; a difference or a raise
sets that key to stock for the process); a hang or device fault in the first self-test launch is the one thing
neither catches. (3) P5-V: a verdict is looked up before any launch, so an unchecked kernel can never be recorded into
a CUDA graph. (4) What the first-use compare sees differs by role, and the claim is only as strong as that: on the
DECODE role a key's first non-capturing call is sglang's CUDA-graph warm-up batch, so the compare runs on warm-up
values and that verdict is what the served graphs then carry (the kernel is input-independent by construction, but
this check alone does not witness served data); on the colocated (null) audit role there is no graph capture, so each
key's FIRST call is compared on the audit's live data, and later calls of a verified key are served unchecked, exactly
as on the decode role (the other layers of that class in the same forward, and every later audited forward). A
data-dependent difference on such a later call therefore reaches the audit's grading exactly as it would reach
decode's served graphs. This is deliberate: comparing every null-role call would make the audited bytes differ from
what the decode role serves. The install self-test, which runs on both roles, is the comparison that does not depend
on who calls first.
(5) Attention is rank-local, so a per-rank verdict never desynchronises ranks.

Off (the engine's method serves) on: the prefill role, a pin mismatch, changed MegaAttention fixed parameters, a
device that is not sm_103, an engine JIT target other than the one the kernel was measured for, a missing nvcc 13.4,
a failed text write, host build, device compile or load, a host library whose CUTLASS is not the device tree's, a
build that has taken longer than BUILD_BUDGET_S when its next device compile would start (the time a rank spends
waiting for another rank's host build counts), a self-test input outside the gates, a self-test difference, a failed
speed check, a gate miss (shape, dtype, stride, width), a key whose verdict is False, and any capture that reaches a
key with no verdict. Any raise inside install turns VATTN off instead of raising into prepare. Each build stage's wall
time is logged (one line per process).
The host build is serialised across ranks by cpp_extension itself, as v50's M462 relies on: the image's torch
(2.14.1+cu130) holds `with FileLock(<build directory>/lock)` around the build, an advisory fcntl.flock that the kernel
releases when its holder exits or is killed, so no stale lock can hang a later rank, and a live builder holds the
others for its own build only. That build has no timeout of its own (nor has DeepJIT's nvcc): a wedged compiler would
hold its rank, and the ranks waiting on it, until the engine's own init timeout, as it would for v50's M462. The lock
FILE stays in place after every release (filelock's UnixFileLock), so its existence says nothing about a holder; this
module never tests, stats or polls it (review V59-R5-F1).

Scope: `routed(forward_batch)` around our model node's call swaps the backend class's method for the duration of a
target-verify forward only (prefill, extend, plain decode and the draft model never see it), as v50's levers swap
theirs, and restores it in `finally`.

Sandbox: no ctypes / pathlib / tempfile / importlib / subprocess / open() here; paths come from os.path, the texts are
published by v50's m462._write (os.open plus an atomic rename), the CUTLASS version is read with os.open / os.read,
and the host library is built by torch's cpp_extension, exactly as v50's M462 builds its own. Engine modules are
imported inside install() only, so a disabled VATTN imports nothing from the engine."""
import contextlib
import contextvars
import hashlib
import inspect
import os
import sys
import time

import torch

# king-v1e: dspd-v59 VATTN in the king's decode tree (bundle_v9.dec); W4 serves whatever VATTN declines
ENABLED = True
# king-v5 VATTN-T (dec_rest DR2): the dependent-launch trigger the king's W4 kernel carries
# (dec/cuda_prefill_attn.py fwd_kernel: griddepcontrol.launch_dependents right after Kernel::devfunc), added to
# VATTN's fwd_kernel the same way, so wo_a's DeepGEMM (launched with PDL; it reads nothing before its
# cudaGridDependencySynchronize) lands its CTAs and runs its prologue while the attention drains, as it did on the
# king. Scheduling only: griddepcontrol.wait still waits for this grid's completion and memory flush. VATTN's
# launch itself stays non-PDL. The edit is applied to the published copy of kernel.cuh (cuda_vattn.py is never
# edited), and _signature hashes the published texts, so the trigger build has its own cache directory and cubins.
TRIGGER = True             # False: the published texts are cuda_vattn.SOURCES unchanged (king_v2b's tag and cubins)
_T_FILE = "csrc/core_attn/kernel.cuh"
_T_OLD = ("fwd_kernel(__grid_constant__ const typename Kernel::Params params, __grid_constant__ const typename "
          "Kernel::TMAParams tma_params, __grid_constant__ const typename Kernel::AuxParams aux_params) {\n"
          "    Kernel::devfunc(params, tma_params, aux_params);\n"
          "}\n")
_T_NEW = ("fwd_kernel(__grid_constant__ const typename Kernel::Params params, __grid_constant__ const typename "
          "Kernel::TMAParams tma_params, __grid_constant__ const typename Kernel::AuxParams aux_params) {\n"
          "    Kernel::devfunc(params, tma_params, aux_params);\n"
          "    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n"
          "}\n")
# king-v7 ATTN-PDL-IN (dec_rest DR5): VATTN launched as a PDL secondary. The host launch adds the programmatic-stream-
# serialisation attribute next to the cluster attribute, and the kernel executes griddepcontrol.wait right after its
# CTA-local prologue (barrier init in shared memory, the TMEM allocation, the CTA barrier) and before any thread reads
# or writes global memory. So VATTN's CTAs may become resident and run that prologue while wq_b (CuTe, which triggers
# before its epilogue) drains, and a predecessor that never triggers still releases it at its exit, not at its flush.
# Scheduling only: the wait returns only after every prerequisite grid has completed and flushed its memory, and it is
# a no-op without the attribute. Read-before-wait audit (every warp, every role): NOTES.md (king_lane/v7/attn). Same
# mechanism as VATTN-T: anchored edits of the published copies of kernel.cuh and vattn_host.cpp, each anchor exactly
# once or the lever is off (the texts then carry VATTN-T alone); _signature hashes the published texts, so this build
# has its own tag, host library and cubins, and the install self-test, speed check and first-use compares run on it.
PDL_IN = True              # False: no PDL edit in the published texts (king_v5's tag and cubins with TRIGGER on)
_P_EDITS = (
    ("csrc/core_attn/kernel.cuh",
     "        AllocatorT().release_allocation_lock();\n"
     "        KU_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_start_addr.data()[0] == 0);\n"
     "    }\n"
     "\n"
     "    if constexpr (IS_2CTA) {\n"
     "        ku::barrier_cluster_arrive_relaxed();\n"
     "        ku::barrier_cluster_wait_acquire();\n"
     "    } else {\n"
     "        __syncthreads();\n"
     "    }\n",
     "        AllocatorT().release_allocation_lock();\n"
     "        KU_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_start_addr.data()[0] == 0);\n"
     "    }\n"
     "\n"
     "    if constexpr (IS_2CTA) {\n"
     "        ku::barrier_cluster_arrive_relaxed();\n"
     "        ku::barrier_cluster_wait_acquire();\n"
     "    } else {\n"
     "        __syncthreads();\n"
     "    }\n"
     "\n"
     "    // king-v7 ATTN-PDL-IN: VATTN is launched as a PDL secondary. Everything above is CTA-local (shared-memory\n"
     "    // barrier init, the TMEM allocation, the CTA barrier); no thread has touched global memory yet. Every thread\n"
     "    // waits here for its prerequisite grids' completion and memory flush before any role reads q, the indices,\n"
     "    // the lengths, the positions, the KV caches or writes an output. A no-op without the launch attribute.\n"
     "    asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"),
    ("csrc/vattn_host.cpp",
     "constexpr int kLaunchAttrCount = 1;      // [0] = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION\n",
     "constexpr int kLaunchAttrCount = 1;      // [0] = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION\n"
     "constexpr int kAttrPdl = 1;              // king-v7 ATTN-PDL-IN: [1] = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION\n"),
    ("csrc/vattn_host.cpp",
     "    p.launch_attrs = kLaunchAttrCount;\n",
     "    p.launch_attrs = kLaunchAttrCount + kAttrPdl;  // king-v7 ATTN-PDL-IN: the cluster, then PDL\n"),
    ("csrc/vattn_host.cpp",
     "    p.attr_pdl = 0;\n    p.rc = 0;\n",
     "    p.attr_pdl = kAttrPdl;  // king-v7 ATTN-PDL-IN: the kernel waits (griddepcontrol.wait) before any global access\n"
     "    p.rc = 0;\n"),
    ("csrc/vattn_host.cpp",
     "    CUlaunchAttribute attributes[1] = {};\n",
     "    CUlaunchAttribute attributes[2] = {};  // king-v7 ATTN-PDL-IN: [0] the cluster, [1] programmatic serialisation\n"),
    ("csrc/vattn_host.cpp",
     "    if (plan.rc != 0 || plan.launch_attrs != kLaunchAttrCount || plan.attr_pdl != 0)\n        return -24;\n",
     "    if (plan.rc != 0 || plan.attr_pdl != kAttrPdl || plan.launch_attrs != kLaunchAttrCount + kAttrPdl)\n"
     "        return -24;  // king-v7 ATTN-PDL-IN: exactly the cluster attribute and the PDL attribute\n"),
    ("csrc/vattn_host.cpp",
     "    out.attributes[0].value.clusterDim.z = plan.attr_cluster[2];\n",
     "    out.attributes[0].value.clusterDim.z = plan.attr_cluster[2];\n"
     "    // king-v7 ATTN-PDL-IN: launch_attribute[1], programmatic stream serialisation (numAttrs = plan.launch_attrs = 2)\n"
     "    out.attributes[1].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;\n"
     "    out.attributes[1].value.programmaticStreamSerializationAllowed = 1;\n"),
)
ROLES = ("decode", "null")  # the colocated (null) engine is the audit's: it grades what decode would serve
VERSION = "v3"              # cache marker: v3 is the route-D build (v2 was the load_jit one)
# pip nvidia-cuda-nvcc 13.4 in the image (site-packages/nvidia/cu13); the 13.0 cudart headers next to it need the
# CCCL CTK check off (the kernel's device code uses inline PTX only)
_NVCC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__))), "nvidia", "cu13", "bin",
                     "nvcc")
_NVCC_ENV = "DSPDVA_JIT_NVCC_COMPILER"      # DeepJIT's Env: <prefix>_JIT_NVCC_COMPILER, prefix DSPDVA
# DeepJIT's Env falls back to the GLOBAL DJ_<suffix> key for every suffix our own prefix leaves unset
# (deep_jit/utils/env.hpp Env::get), so a DJ_JIT_CPP_STANDARD, DJ_JIT_CHECK_NO_SPILLS or DJ_JIT_PRINT_LOAD_TIME in the
# image or the validator's environment would change or fail route D's compile or load (an unparsable bool panics in
# get_env). Every switch the CUDA DeepJIT headers this host library includes read through Env (env.get<T>("JIT_...")),
# except JIT_NVCC_COMPILER (set below) and JIT_CACHE_DIR (set in vattn_host.cpp init), is pinned here to DeepJIT's own
# default, absent-only: an operator's own DSPDVA value still wins. The lane's check derives this set from the headers.
_JIT_PINS = {"DSPDVA_JIT_CPP_STANDARD": "20", "DSPDVA_JIT_CHECK_NO_SPILLS": "0",
             "DSPDVA_JIT_CHECK_NO_LOCAL_MEMORY": "0", "DSPDVA_JIT_WITH_LINEINFO": "0", "DSPDVA_JIT_DEBUG": "0",
             "DSPDVA_JIT_DUMP_ASM": "0", "DSPDVA_JIT_DUMP_PTX": "0", "DSPDVA_JIT_DUMP_SASS": "0",
             "DSPDVA_JIT_PTXAS_VERBOSE": "0", "DSPDVA_JIT_PRINT_COMPILER_COMMAND": "0",
             "DSPDVA_JIT_PRINT_LOAD_TIME": "0"}
_CXX_FLAGS = ("-std=c++20", "-O3", "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-Wno-psabi",
              "-Wno-deprecated-declarations")
_LD_FLAGS = ("-lc10_cuda", "-ltorch_cuda")
# DeepseekV4MegaAttnBackend._forward_mega_decode / flash_mla_interface.flash_mla_mega_decode (sg_real 15338a8e) and the
# wheel's fused_norm_rope_attn_rope_cast.decode (sgl-flash-mla 1.0.0) whose legacy kernel VATTN reproduces
PINS = {"_forward_mega_decode": "5aada4ccf9343ca8", "flash_mla_mega_decode": "441862ecfd32811e",
        "mega.decode": "6bbec47f63d9b17c"}
_FIXED = {"is_rope_neox_style": False, "rope_dim": 64, "num_per_channels": 32, "use_tma_aligned_col_major_sf": True,
          "round_sf": True, "use_packed_ue8m0": True}
CAPABILITY = (10, 3)        # B300: the device the kernel was measured and built for
ARCH = "103a"               # the device target load_jit compiles for on this capability; DeepJIT's own default
                            # (the family target "103f") is overridden to it, so route D builds the measured target
DECODE = False              # target-verify forwards only (this arena's target never runs plain decode); exact either way
SPEED_MARGIN = 0.99         # install speed check: VATTN must take <= 99 % of stock at 240 rows (cold inputs)
MAX_WIDTH = 640             # SWA + extra top-k entries per row; the engine's kernel splits KV only past 640
BUILD_BUDGET_S = 120.0      # install gives up (VATTN off) when the host build (with any wait for another rank's
                            # host build) and the device compiles have taken longer than this by the time the next
                            # device compile would start (_within_budget); a warm cache takes seconds
_STUB_HEADER = "elfutils/libdwfl.h"  # what DeepJIT's exception.hpp includes; v50's stub sits at <dir>/<this>
# The install self-test's edge case (review V59-R3-F5), at the smallest shape: page strides wider than the page, by
# the rig's own padding (struct/verify-attn rig2 padded_page_stride: p * bytes_per_token + 2048, bit-equal to stock).
# Both keep every gate of _gates (kv.stride(0) % 512 == 0, ekv.stride(0) % 256 == 0); the lane's check recomputes it.
_EDGE_KV_PAD = 2048         # kv.stride(0) = 256 * 528 + 2048 bytes
_EDGE_EKV_PAD = 2048        # ekv.stride(0) = page * 288 + 2048 bytes
context = contextvars.ContextVar("dspd_vattn", default=False)
_g = {"state": None, "lib": None, "cls": None, "orig": None, "verdict": {}, "depth": 0,
      "aws": None}  # king-v7: the aws module while AWS serves (install), else None
_said = set()


# ------------------------------------------------------------------------------------------------- shared helpers
def say(tag, what):
    """One line per (tag, message) per process on stderr: which path a lever took, for the A/B record (common.say)."""
    key = (tag, what)
    if key not in _said:
        _said.add(key)
        print(f"[dspd {tag}] pid {os.getpid()}: {what}", file=sys.stderr, flush=True)


def _digest(fn):
    """First 16 hex of the sha256 of a function's source, or '' when the source is not readable (common.digest)."""
    try:
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return ""


def pins_match(tag, expected, functions):
    """True when every mirrored engine function still has a source the op was written against (common.pins_match)."""
    got = {name: _digest(fn) for name, fn in functions.items()}
    ok = set(got) == set(expected) and all(
        got[n] in (expected[n] if isinstance(expected[n], (set, frozenset)) else {expected[n]}) for n in got)
    say(tag, "engine pins match" if ok else f"engine pins MISMATCH {got}: lever off, stock forward")
    return ok


# ------------------------------------------------------------------------------------------------- the route-D build
def _texts():
    """(texts, VATTN-T reason, ATTN-PDL-IN reason): cuda_vattn.SOURCES, with VATTN-T's trigger in kernel.cuh's
    fwd_kernel when TRIGGER is on and its anchor occurs exactly once, then (king-v7) ATTN-PDL-IN's wait and launch
    attribute when PDL_IN is on and every one of its anchors occurs exactly once (all or none). A reason is "" when its
    edit is in the texts; otherwise it says why not."""
    from bundle_v9.dec import cuda_vattn

    texts = dict(cuda_vattn.SOURCES)
    if not TRIGGER:
        t_why = "TRIGGER = False"
    else:
        count = texts.get(_T_FILE, "").count(_T_OLD)
        if count != 1:
            t_why = f"its anchor occurs {count} times in {_T_FILE}"
        else:
            texts[_T_FILE] = texts[_T_FILE].replace(_T_OLD, _T_NEW)
            t_why = ""
    if not PDL_IN:
        return texts, t_why, "PDL_IN = False"
    trial = dict(texts)
    for rel, old, new in _P_EDITS:
        count = trial.get(rel, "").count(old)
        if count != 1:
            return texts, t_why, f"an anchor occurs {count} times in {rel}"
        trial[rel] = trial[rel].replace(old, new)
    return trial, t_why, ""


def _signature(ext):
    """One digest over every text of this lever, the target arch, the launch constants, the external include trees
    (both lists are built from them) and the device tree's CUTLASS version: DeepJIT's extra_signature and the name of
    the texts directory and host library, so a changed source, arch, constant, include tree or CUTLASS can never hit a
    cached cubin or a cached host library."""
    from bundle_v9.dec import cuda_vattn

    texts = _texts()[0]  # king-v5 VATTN-T / king-v7 ATTN-PDL-IN: the texts _publish writes (both off: SOURCES)
    h = hashlib.sha256()
    for rel in sorted(texts):
        h.update(rel.encode())
        h.update(hashlib.sha256(texts[rel].encode()).digest())
    h.update(f"{VERSION}|{ARCH}|{list(cuda_vattn.EXPECT)}".encode())
    h.update(f"|{ext['default']}|{ext['cutlass']}|{ext['image']}|{ext['cuda_home']}|{ext['cutlass_version']}".encode())
    return h.hexdigest()[:16]


def _publish(tag):
    """Write every text of this lever under <SGLang JIT cache root>/cacheon-dspd-v59-vattn/<tag>/ and return that
    directory. v50's atomic writer publishes each file, so a reader never sees a partial one; a file already there is
    left alone (its name carries the digest of its content). The elfutils declarations DeepJIT's exception header
    takes are v50's own build stub, used by reference, not copied."""
    from sglang.kernels.jit.utils.compile.cache import cache_root

    from bundle_v9.dec import cuda_moe_prefill as cuda_m462, cuda_vattn, moe_prefill as m462  # king-v1e

    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "cacheon-dspd-v59-vattn", tag)
    texts = _texts()[0]  # king-v5 VATTN-T / king-v7 ATTN-PDL-IN (both off: cuda_vattn.SOURCES as before)
    texts[cuda_vattn.STUB] = cuda_m462.SOURCES[cuda_vattn.STUB]
    for rel, text in texts.items():
        path = os.path.join(directory, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            m462._write(path, text.encode())
    os.makedirs(os.path.join(directory, "build"), exist_ok=True)
    os.makedirs(os.path.join(directory, "dj"), exist_ok=True)
    return directory


def _cutlass_version(dirs):
    """((major, minor, patch), path) of the first cutlass/version.h on `dirs`, the file `#include <cutlass/...>`
    resolves to in that include order. Read with os.open / os.read (this module calls no open())."""
    for directory in dirs:
        path = os.path.join(directory, "cutlass", "version.h")
        if not os.path.isfile(path):
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            chunks = []
            while True:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(fd)
        fields = {}
        for line in b"".join(chunks).decode("utf-8", "replace").splitlines():
            parts = line.split()
            if (len(parts) == 3 and parts[0] == "#define" and parts[2].isdigit()
                    and parts[1] in ("CUTLASS_MAJOR", "CUTLASS_MINOR", "CUTLASS_PATCH")):
                fields.setdefault(parts[1], int(parts[2]))
        if len(fields) != 3:
            raise RuntimeError(f"dspd-v59 VATTN: no CUTLASS_MAJOR / MINOR / PATCH in {path}")
        return (fields["CUTLASS_MAJOR"], fields["CUTLASS_MINOR"], fields["CUTLASS_PATCH"]), path
    raise RuntimeError(f"dspd-v59 VATTN: no cutlass/version.h on {dirs}")


def _external():
    """The include trees outside our texts, resolved read-only exactly as the rig's load_jit build resolved them:
    SGLang's DEFAULT_INCLUDE first and, after our two directories, the engine's own "cutlass" dependency, whose
    resolver lists FlashInfer's CUTLASS / CuTe before DeepGEMM's include tree (load_jit: DEFAULT_INCLUDE +
    extra_include_paths + each dependency's paths). DeepJIT's headers come from DeepGEMM's tree and the CUDA ones from
    the toolkit cpp_extension resolves, as v50's M462 finds them. Nothing is built or written here."""
    import deep_gemm
    from sglang.kernels.jit.utils.compile.paths import DEFAULT_INCLUDE
    from sglang.kernels.jit.utils.deps import get_cutlass_include_paths
    from torch.utils.cpp_extension import CUDA_HOME

    default = [os.path.abspath(p) for p in DEFAULT_INCLUDE]
    cutlass = [os.path.abspath(p) for p in get_cutlass_include_paths()]
    version, version_h = _cutlass_version(default + cutlass)
    return {"default": default, "cutlass": cutlass, "cutlass_version": version, "cutlass_version_h": version_h,
            "image": os.path.abspath(os.path.join(os.path.dirname(deep_gemm.__file__), "include")),
            "cuda_home": CUDA_HOME or "/usr/local/cuda"}


def _includes(root, ext):
    """(host include paths, device include paths). The device list is load_jit's order: DEFAULT_INCLUDE, our csrc and
    kerutils (load_jit's extra_include_paths in the rig build), then the "cutlass" dependency's paths. The host list is
    v50's stub directory and then that same list, so `#include <cutlass/...>` resolves to one CUTLASS on both sides
    (the host's FastDivmod and the device's AuxParams), then DeepGEMM's tree (DeepJIT) and the CUDA toolkit's
    headers."""
    from bundle_v9.dec import cuda_vattn

    if not cuda_vattn.STUB.endswith("/" + _STUB_HEADER):
        raise RuntimeError(f"dspd-v59 VATTN: the build stub {cuda_vattn.STUB} is not <dir>/{_STUB_HEADER}")
    # DeepJIT's exception.hpp says `#include <elfutils/libdwfl.h>`, so the include directory is the one that HOLDS
    # elfutils/ (<root>/build_include, the directory m462.ops passes), not elfutils/ itself
    stub_dir = os.path.join(root, cuda_vattn.STUB[: -len(_STUB_HEADER) - 1])
    csrc = os.path.join(root, "csrc")
    kerutils = os.path.join(csrc, "3rdparty", "kerutils", "include")
    device = [*ext["default"], csrc, kerutils, *ext["cutlass"]]
    host = [stub_dir, *device]
    if ext["image"] not in host:
        host.append(ext["image"])
    host.append(os.path.join(ext["cuda_home"], "include"))
    cccl = os.path.join(ext["cuda_home"], "include", "cccl")
    if os.path.isdir(cccl):
        host.append(cccl)
    return host, device


def _within_budget(start, stages):
    elapsed = time.monotonic() - start
    if elapsed > BUILD_BUDGET_S:
        raise RuntimeError(f"dspd-v59 VATTN: the build has taken {elapsed:.1f} s > {BUILD_BUDGET_S:.0f} s "
                           f"({', '.join(stages)}); not starting the next stage")


def _build():
    """The host library (torch's cpp_extension, one C++ source, no nvcc) and the two cubins its DeepJIT compiles,
    one per layer class. Returns the loaded op namespace. Each stage's wall time is logged; once BUILD_BUDGET_S has
    passed, the next stage is not started and VATTN stays off."""
    from torch.utils.cpp_extension import load as build_library

    from bundle_v9.dec import cuda_vattn

    start = time.monotonic()
    ext = _external()
    signature = _signature(ext)
    tag = f"{VERSION}-{signature}"
    root = _publish(tag)
    host, device = _includes(root, ext)
    say("vattn", f"VATTN build {tag}: device includes {device}; CUTLASS "
                 f"{'.'.join(str(v) for v in ext['cutlass_version'])} from {ext['cutlass_version_h']}")
    # DeepJIT reads its nvcc and every compile switch from our own prefix; an operator's value, if any, wins, and a
    # key we leave unset would fall back to the global DJ_JIT_* one (see _JIT_PINS)
    for key, value in ((_NVCC_ENV, _NVCC), *sorted(_JIT_PINS.items())):
        if key not in os.environ:
            os.environ[key] = value
    # cpp_extension serialises this build across the ranks that share the cache root (review V59-R5-F1): the image's
    # torch 2.14.1+cu130 takes `with FileLock(<build_directory>/lock)` (an fcntl.flock advisory lock) around the
    # build, so one rank builds while the others block in load(); each then takes the lock in turn, re-runs ninja on
    # the built tree (nothing left to do) and loads the library. The kernel drops the lock when its holder exits or is
    # killed, so a dead builder cannot hang the others (the next holder rebuilds what it left); the lock
    # FILE outlives every release, so its existence is not the lock, and nothing here tests it (the pre-wait of FIX
    # rounds 3-4 did, and spun the whole budget on every rank after the first build). A wait for another rank's
    # build counts against BUILD_BUDGET_S, which _within_budget checks before each device compile.
    # with_cuda=False: one C++ source, no nvcc (cpp_extension's CUDA mode would derive arch flags from a visible
    # GPU); the CUDA runtime and torch's CUDA libraries are linked explicitly, as v50's M462 links them
    build_library(name=f"dspd_v59_vattn_{tag.replace('-', '_')}",
                  sources=[os.path.join(root, "csrc", "vattn_host.cpp")], extra_cflags=list(_CXX_FLAGS),
                  extra_ldflags=[f"-L{os.path.join(ext['cuda_home'], 'lib64')}", "-lcudart", *_LD_FLAGS],
                  extra_include_paths=host, build_directory=os.path.join(root, "build"), with_cuda=False,
                  is_python_module=False, verbose=False)
    lib = torch.ops.dspd_v59_vattn
    host_cutlass = tuple(int(v) for v in lib.cutlass_version())
    if host_cutlass != tuple(ext["cutlass_version"]):
        raise RuntimeError(f"dspd-v59 VATTN: the host library was built against CUTLASS {host_cutlass}, the device "
                           f"include tree holds {ext['cutlass_version']}")
    stages = [f"host build {time.monotonic() - start:.1f} s (any wait for another rank's host build included)"]
    lib.init(root, device, os.path.join(root, "dj"), ARCH, signature, list(cuda_vattn.EXPECT))
    for has_extra in (0, 1):
        _within_budget(start, stages)
        began = time.monotonic()
        rc = int(lib.compile_jit(has_extra))
        if rc != 0:
            raise RuntimeError(f"dspd-v59 VATTN: the device unit did not compile for has_extra={has_extra} (rc {rc})")
        stages.append(f"cubin x{has_extra} {time.monotonic() - began:.1f} s")
    say("vattn", f"VATTN build stages: {', '.join(stages)}; {time.monotonic() - start:.1f} s in all "
                 f"(budget {BUILD_BUDGET_S:.0f} s)")
    return lib


# ------------------------------------------------------------------------------------------------- the call
def _alloc(s_q, dev):
    """Stock's outputs: out_fp8 [s_q, 8, 4096] e4m3, out_sf [s_q, 8, 32] int32 (TMA-aligned col-major: stride (1,
    32 * a, a), a = s_q rounded up to 4), lse [s_q, 64] fp32 (unused by the engine)."""
    out = torch.empty((s_q, 8, 4096), dtype=torch.float8_e4m3fn, device=dev)
    a = (s_q + 3) // 4 * 4
    sf = torch.empty((8, 32, a), dtype=torch.int32, device=dev).transpose(1, 2)[:, :s_q, :].transpose(0, 1)
    lse = torch.empty((s_q, 64), dtype=torch.float32, device=dev)
    return out, sf, lse


def _byte(t):
    return t.dtype in (torch.uint8, torch.int8, torch.float8_e4m3fn)


def _gates(q, kv, idx, pos, cs, sink, tl, ekv, eidx, etl):
    """Every shape / dtype / stride the kernel (and stock's legacy kernel) relies on; anything else is stock's call."""
    if not (isinstance(q, torch.Tensor) and q.is_cuda and q.dtype == torch.bfloat16 and q.dim() == 3
            and q.shape[1] == 64 and q.shape[2] == 512 and q.stride(2) == 1 and q.stride(1) == 512 and q.shape[0] > 0):
        return False
    s_q = q.shape[0]
    if not (isinstance(kv, torch.Tensor) and _byte(kv) and kv.dim() == 4 and kv.shape[2] == 1 and kv.shape[3] == 528
            and kv.stride(1) == 528 and kv.stride(3) == 1 and kv.stride(0) % 512 == 0 and kv.data_ptr() % 16 == 0
            and kv.shape[0] * (kv.stride(0) // 512) < 2 ** 31):
        return False
    for t, dt in ((idx, torch.int32), (pos, torch.int32), (tl, torch.int32), (sink, torch.float32),
                  (cs, torch.float32)):
        if not isinstance(t, torch.Tensor) or t.dtype != dt or t.device != q.device:
            return False
    if not (idx.dim() == 2 and idx.shape[0] == s_q and idx.stride(1) == 1 and idx.stride(0) % 8 == 0
            and idx.shape[1] > 0 and tl.dim() == 1 and tl.shape[0] == s_q and tl.stride(0) == 1
            and pos.dim() == 1 and pos.shape[0] == s_q and pos.stride(0) == 1 and sink.dim() == 1
            and sink.shape[0] == 64 and sink.stride(0) == 1 and cs.dim() == 2 and cs.shape[1] == 64
            and cs.stride(1) == 1 and cs.stride(0) == 64):
        return False
    if ekv is None:
        return eidx is None and etl is None and idx.shape[1] <= MAX_WIDTH
    if not (isinstance(ekv, torch.Tensor) and _byte(ekv) and ekv.device == q.device and ekv.dim() == 4
            and ekv.shape[2] == 1 and ekv.shape[3] == 288 and ekv.stride(1) == 288 and ekv.stride(3) == 1
            and ekv.stride(0) % 256 == 0 and ekv.data_ptr() % 16 == 0
            and ekv.shape[0] * (ekv.stride(0) // 256) < 2 ** 31):
        return False
    return (isinstance(eidx, torch.Tensor) and eidx.dtype == torch.int32 and eidx.device == q.device
            and eidx.dim() == 2 and eidx.shape[0] == s_q and eidx.stride(1) == 1 and eidx.stride(0) % 8 == 0
            and eidx.shape[1] > 0 and idx.shape[1] % 8 == 0 and idx.shape[1] + eidx.shape[1] <= MAX_WIDTH
            and isinstance(etl, torch.Tensor) and etl.dtype == torch.int32 and etl.device == q.device
            and etl.dim() == 1 and etl.shape[0] == s_q and etl.stride(0) == 1)


def _key(kind, idx, ekv, eidx, rows):
    """The verdict key: the layer class, the extra cache's page rows (256 / compression ratio, so c1 / c2 / c4 and
    SWA-only never share a verdict), both top-k widths, and the row count. Each key's own first use is checked
    against stock before any capture may record it."""
    return (kind, int(ekv.shape[1]) if ekv is not None else 0, int(idx.shape[1]),
            int(eidx.shape[1]) if eidx is not None else 0, int(rows))


def _launch(q, kv, idx, pos, cs, sink, tl, ekv, eidx, etl, scale):
    """(out_fp8, out_sf) from VATTN, or None when the module refuses the call (rc < 0)."""
    out, sf, lse = _alloc(q.shape[0], q.device)
    has = ekv is not None
    rc = int(_g["lib"].decode(q, kv, idx, tl, sink, ekv if has else tl, eidx if has else tl, etl if has else tl, pos,
                              cs, out, sf, lse, float(scale), 1 if has else 0))
    return (out, sf) if rc == 0 else None


def _alloc_sf(s_q, dev):
    """king-v7 AWS: stock's out_sf layout for s_q rows alone, [s_q, 8, 32] int32 with stride (1, 32 * a, a), a = s_q
    rounded up to 4 (as _alloc builds it). DeepGEMM's fp8_einsum takes an SF whose M stride is align4 of ITS OWN rows
    (check_sf_layout), so each slice of the split call gets its own."""
    a = (s_q + 3) // 4 * 4
    return torch.empty((8, 32, a), dtype=torch.int32, device=dev).transpose(1, 2)[:, :s_q, :].transpose(0, 1)


def _split_launch(q, kv, idx, pos, cs, sink, tl, ekv, eidx, etl, scale, n1):
    """king-v7 AWS: the verify attention as two VATTN launches over row slices, A1 = rows [0, n1) and A2 = rows
    [n1, R), on the current stream, with an event recorded between them (the side stream's wo_a of A1's rows waits on
    it). Rows are independent jobs (no split-KV at width <= MAX_WIDTH), so out_fp8 and lse are row slices of one
    buffer each and every row's bytes are the unsplit call's; out_sf is per slice (_alloc_sf). `full` is an
    unfilled stock-layout SF of all R rows: the engine-visible scale, filled from the two slices only on a fallback.
    Returns that dict; None when nothing was launched (A1 refused); "unsplit" when A1 ran and A2 was refused (the
    caller then serves the unsplit call, which rewrites every row). Layers with the extra cache only."""
    rows = q.shape[0]
    if ekv is None or not 0 < n1 < rows:
        return None
    dev = q.device
    out = torch.empty((rows, 8, 4096), dtype=torch.float8_e4m3fn, device=dev)
    lse = torch.empty((rows, 64), dtype=torch.float32, device=dev)
    full, sf1, sf2 = _alloc_sf(rows, dev), _alloc_sf(n1, dev), _alloc_sf(rows - n1, dev)
    lib = _g["lib"]

    def part(a, b, sf):
        return int(lib.decode(q[a:b], kv, idx[a:b], tl[a:b], sink, ekv, eidx[a:b], etl[a:b], pos[a:b], cs, out[a:b],
                              sf, lse[a:b], float(scale), 1))

    if part(0, n1, sf1) != 0:
        return None
    ev = torch.cuda.Event()
    ev.record()
    if part(n1, rows, sf2) != 0:
        return "unsplit"
    return {"out": out, "full": full, "sf1": sf1, "sf2": sf2, "ev": ev, "n1": n1, "rows": rows}


# ------------------------------------------------------------------------------------------------- the swapped method
def _mega_decode(self, *, q, swa_k_cache, swa_page_indices, positions, cos_sin_cache, attn_sink, swa_topk_lengths,
                 extra_k_cache, extra_indices, extra_topk_lengths):
    orig = _g["orig"]
    kw = dict(q=q, swa_k_cache=swa_k_cache, swa_page_indices=swa_page_indices, positions=positions,
              cos_sin_cache=cos_sin_cache, attn_sink=attn_sink, swa_topk_lengths=swa_topk_lengths,
              extra_k_cache=extra_k_cache, extra_indices=extra_indices, extra_topk_lengths=extra_topk_lengths)
    if (not context.get() or getattr(self, "_swa_interleaved", True) is not False
            or not isinstance(getattr(self, "softmax_scale", None), float)
            or not _gates(q, swa_k_cache, swa_page_indices, positions, cos_sin_cache, attn_sink, swa_topk_lengths,
                          extra_k_cache, extra_indices, extra_topk_lengths)):
        return orig(self, **kw)
    kind = "extra" if extra_k_cache is not None else "swa"
    key = _key(kind, swa_page_indices, extra_k_cache, extra_indices, q.shape[0])
    verdict = _g["verdict"].get(key)
    capturing = torch.cuda.is_current_stream_capturing()
    if verdict is False or (verdict is None and capturing):
        return orig(self, **kw)             # P5-V: never record an unchecked kernel into a graph (before any launch)
    # king-v7 AWS: only inside aws._aws_mega_core's own attention call (armed for exactly this row count), only on a
    # key whose unsplit VATTN verdict is True, and (P5-V again) a split key with no verdict is never captured
    plan = _g["aws"].armed(q.shape[0]) if (_g["aws"] is not None and verdict is True and kind == "extra") else None
    if plan is not None:
        akey = ("aws", plan.n1) + key
        averdict = _g["verdict"].get(akey)
        if averdict is True or (averdict is None and not capturing):
            try:
                res = _split_launch(q, swa_k_cache, swa_page_indices, positions, cos_sin_cache, attn_sink,
                                    swa_topk_lengths, extra_k_cache, extra_indices, extra_topk_lengths,
                                    self.softmax_scale, plan.n1)
            except Exception as e:  # noqa: BLE001
                res = "unsplit"
                say("vattn", f"AWS split launch raised for {akey} ({type(e).__name__}: {str(e)[:200]}): unsplit")
            if res is None or res == "unsplit":
                _g["verdict"][akey] = False
                say("vattn", f"AWS split refused for {akey}: VATTN serves it unsplit")
            else:
                if averdict is None:            # eager: both slices against the engine's kernel on these inputs
                    try:
                        ref = orig(self, **kw)
                        same = (bool(torch.equal(res["out"].view(torch.uint8), ref[0].view(torch.uint8)))
                                and bool(torch.equal(res["sf1"], ref[1][:plan.n1]))
                                and bool(torch.equal(res["sf2"], ref[1][plan.n1:])))
                    except Exception as e:  # noqa: BLE001
                        _g["verdict"][akey] = False
                        say("vattn", f"AWS first-use check raised ({type(e).__name__}): unsplit for {akey}")
                        del res
                        return orig(self, **kw)
                    _g["verdict"][akey] = same
                    say("vattn", f"AWS first-use check {'OK' if same else 'DIFFERENT (stock)'}: {akey}")
                    if not same:
                        del res
                        return ref
                    del ref
                _g["aws"].stash(res)
                return res["out"], res["full"]
    try:
        got = _launch(q, swa_k_cache, swa_page_indices, positions, cos_sin_cache, attn_sink, swa_topk_lengths,
                      extra_k_cache, extra_indices, extra_topk_lengths, self.softmax_scale)
    except Exception as e:  # noqa: BLE001
        got = None
        say("vattn", f"VATTN launch failed for {key} ({type(e).__name__}: {str(e)[:200]}): stock")
    if got is None:
        _g["verdict"][key] = False
        return orig(self, **kw)
    if verdict is None:                     # eager (P5-V above): compare with stock on these inputs
        try:
            ref = orig(self, **kw)
            same = (bool(torch.equal(got[0].view(torch.uint8), ref[0].view(torch.uint8)))
                    and bool(torch.equal(got[1], ref[1])))
        except Exception as e:  # noqa: BLE001  (e.g. OOM in the reference: this key keeps the engine's kernel)
            _g["verdict"][key] = False
            say("vattn", f"VATTN first-use check raised ({type(e).__name__}): stock for {key}")
            del got
            return orig(self, **kw)
        _g["verdict"][key] = same
        say("vattn", f"VATTN first-use check {'OK' if same else 'DIFFERENT (stock)'}: {key}")
        if not same:
            del got
            return ref
        del ref
    return got


# ------------------------------------------------------------------------------------------------- install self-test
def _synth(rows, ratio, extra, dev, seed, edge=False):
    """Block-split V4.1 caches and verify-shaped indices (6 consecutive positions per request). `edge` (review
    V59-R3-F5) adds the data-dependent paths the engine also hands the kernel, so route D's binary meets them here,
    inside install, before any live call: request 0's six SWA rows hold 41..46 valid slots followed by -1 (the
    valid-prefix / -1-suffix shape build_causal_swa_page_indices writes for a short window: slots 64..127, a whole KV
    block, are invalid), row 2 has -1 holes inside its SWA length, the last row is a graph-padding row as
    match_num_queries pads one (SWA length 1 on index 0; extra length 1 on index -1), and both caches are views of wider
    page buffers (kv.stride(0) = 256 * 528 + _EDGE_KV_PAD, ekv.stride(0) = page * 288 + _EDGE_EKV_PAD). With an extra
    cache, row 1 has SWA top-k length 0 (every SWA slot -1), row 3 extra top-k length 0 (every extra slot -1) and row 4
    -1 holes inside its extra length; each of those rows keeps valid slots in the other cache. No row is left with no
    valid slot at all: on an SWA-only layer a zero length would be such a row (a job of zero KV blocks), which the
    engine never serves on a verify forward and the rig never measured (lane deviation D14)."""
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    nreq = (rows + 5) // 6
    ctx = 1500 + (torch.arange(nreq, device=dev) * 37) % 900
    swa_d = (torch.randn((nreq + 1, 256, 512), generator=g, device=dev) * 1.5).to(torch.float8_e4m3fn).view(torch.uint8)
    swa_s = torch.randint(124, 128, (nreq + 1, 256, 16), generator=g, device=dev).to(torch.uint8)
    swa = torch.cat([swa_d.reshape(nreq + 1, -1), swa_s.reshape(nreq + 1, -1)], 1).view(nreq + 1, 256, 1, 528)
    r = torch.arange(rows, device=dev)
    pos = (ctx[r // 6] + r % 6).to(torch.int32)
    w = pos[:, None] - 127 + torch.arange(128, device=dev)[None, :]
    swa_idx = ((r // 6)[:, None] * 256 + w % 256).to(torch.int32).contiguous()
    swa_len = torch.full((rows,), 128, dtype=torch.int32, device=dev)
    out = dict(q=(torch.randn((rows, 64, 512), generator=g, device=dev) * 2).to(torch.bfloat16), kv=swa, idx=swa_idx,
               tl=swa_len, pos=pos, sink=torch.randn((64,), generator=g, device=dev),
               cs=torch.randn((4096, 64), generator=g, device=dev).contiguous(), ekv=None, eidx=None, etl=None)
    if extra:
        page = 256 // ratio
        nblk = (int(ctx.max()) // ratio + 8) // page + 1
        total = nreq * nblk + 1
        ed = torch.randint(0, 256, (total, page, 256), generator=g, device=dev).to(torch.uint8)
        es = (torch.rand((total, page, 32), generator=g, device=dev) * 0.27 + 0.03).to(torch.float8_e4m3fn).view(torch.uint8)
        out["ekv"] = torch.cat([ed.reshape(total, -1), es.reshape(total, -1)], 1).view(total, page, 1, 288)
        ncomp = (ctx[r // 6] // ratio)
        sel = (torch.rand((rows, 512), generator=g, device=dev) * ncomp[:, None]).long()
        out["eidx"] = ((1 + (r // 6)[:, None] * nblk + sel // page) * page + sel % page).to(torch.int32).contiguous()
        lens = torch.full((rows,), 512, dtype=torch.int32, device=dev)
        lens[-1] = 300                                # one ragged row
        out["etl"] = lens
    if edge:
        for j in range(6):                            # request 0: a valid prefix of 41 + j slots, -1 after it
            out["tl"][j] = 41 + j
            out["idx"][j, 41 + j:] = -1
        out["idx"][2, 3::13] = -1                     # holes inside row 2's length (43): slots 3, 16, 29, 42
        out["tl"][-1] = 1                             # a graph-padding row: SWA length 1 on index 0
        out["idx"][-1, :] = 0
        wide = torch.zeros((nreq + 1, 256 * 528 + _EDGE_KV_PAD), dtype=torch.uint8, device=dev)
        wide[:, : 256 * 528] = out["kv"].reshape(nreq + 1, -1)
        out["kv"] = wide[:, : 256 * 528].view(nreq + 1, 256, 1, 528)
        if extra:
            out["tl"][1] = 0                          # no SWA slot at all; its extra slots stay valid
            out["idx"][1, :] = -1
            out["etl"][3] = 0                         # no extra slot at all; its SWA slots stay valid
            out["eidx"][3, :] = -1
            out["eidx"][4, ::7] = -1                  # holes inside row 4's extra length (512)
            out["etl"][-1] = 1                        # the graph-padding row: extra length 1 on index -1
            out["eidx"][-1, :] = -1
            wide = torch.zeros((total, page * 288 + _EDGE_EKV_PAD), dtype=torch.uint8, device=dev)
            wide[:, : page * 288] = out["ekv"].reshape(total, -1)
            out["ekv"] = wide[:, : page * 288].view(total, page, 1, 288)
    return out


def _stock_call(x, scale):
    from sglang.kernels.ops.attention import flash_mla_interface as FI

    data, sf, _ = FI.flash_mla_mega_decode(x["q"], x["kv"], x["idx"], x["pos"], x["cs"], scale, attn_sink=x["sink"],
                                           topk_length=x["tl"], extra_kv=x["ekv"], extra_indices=x["eidx"],
                                           extra_topk_length=x["etl"], use_legacy=True)
    return data, sf


def _self_test(dev):
    scale = 512 ** -0.5
    for rows in (36, 240):                            # the smallest shape first: route D's first launch ever
        for name, ratio, extra in (("c1", 1, True), ("c2", 2, True), ("swa", 1, False)):
            # at the smallest shape each class also runs its edge case (-1 entries, a zero length, padded page
            # strides), right after its all-valid case, so the first launch ever is still the plainest one
            for edge in ((False, True) if rows == 36 else (False,)):
                case = f"{name}{' edge' if edge else ''}, {rows} rows"
                x = _synth(rows, ratio, extra, dev, seed=rows * 7 + ratio + (11 if extra else 0), edge=edge)
                if not _gates(x["q"], x["kv"], x["idx"], x["pos"], x["cs"], x["sink"], x["tl"], x["ekv"], x["eidx"],
                              x["etl"]):
                    return f"self-test input outside the gates ({case})"
                ref = _stock_call(x, scale)
                got = _launch(x["q"], x["kv"], x["idx"], x["pos"], x["cs"], x["sink"], x["tl"], x["ekv"], x["eidx"],
                              x["etl"], scale)
                if got is None or not (torch.equal(got[0].view(torch.uint8), ref[0].view(torch.uint8))
                                       and torch.equal(got[1], ref[1])):
                    return f"self-test DIFFERENT ({case})"
    # speed: three rotated input sets (KV mostly out of L2, as between the layers of a decode step), one graph of 9
    # calls per path. The check exists to catch a wrong toolchain (a 13.0 build is ~3 % slower than stock); the
    # measured margin of the right one is ~11 %
    sets = [_synth(240, 1, True, dev, seed=99 + i) for i in range(3)]
    argsets = [(x["q"], x["kv"], x["idx"], x["pos"], x["cs"], x["sink"], x["tl"], x["ekv"], x["eidx"], x["etl"], scale)
               for x in sets]

    def graph_of(fns, reps=3):
        """A private CUDA graph of every fn, reps times (GPU time only, as in the decode graphs)."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for fn in fns:
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        # thread_local: the engine's other threads (KV transfer, scheduler) keep their CUDA calls legal meanwhile
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            for _ in range(reps):
                for fn in fns:
                    fn()
        return graph, reps * len(fns)

    def timed(gn):
        graph, n = gn
        graph.replay()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1) * 1000 / n

    g_stock = graph_of([(lambda x=x: _stock_call(x, scale)) for x in sets])
    g_ours = graph_of([(lambda a=a: _launch(*a)) for a in argsets])
    t_stock, t_ours = float("inf"), float("inf")
    for _ in range(5):                                # interleaved rounds, best of each
        t_stock = min(t_stock, timed(g_stock))
        t_ours = min(t_ours, timed(g_ours))
    del g_stock, g_ours, sets, argsets
    if t_ours > SPEED_MARGIN * t_stock:
        return f"speed check failed ({t_ours:.1f} us vs stock {t_stock:.1f} us at 240 rows, route D)"
    return f"OK ({t_ours:.1f} us vs stock {t_stock:.1f} us at 240 rows, c1, route D)"


def _pdl_check(dev):
    """king-v7 ATTN-PDL-IN at the served launch form: a private CUDA graph in which each VATTN call's per-row inputs
    (q, both index tensors, both lengths, the positions) are written by the kernels right before it, so its
    programmatic edge is on a producer of what it reads, as wq_b is in the served graphs; three rotated sets at 240
    rows, the graph replayed twice, each output against the engine's kernel byte for byte. Returns "" or the reason.
    (Without the attribute griddepcontrol.wait is a no-op, so this check is only run when the PDL edits are in.)"""
    scale = 512 ** -0.5
    sets = [_synth(240, 1, True, dev, seed=313 + i) for i in range(3)]
    refs = [_stock_call(x, scale) for x in sets]
    names = ("q", "idx", "tl", "pos", "eidx", "etl")
    buf = {k: torch.empty_like(sets[0][k]) for k in names}

    def run():
        outs = []
        for x in sets:
            for k in names:
                buf[k].copy_(x[k])
            outs.append(_launch(buf["q"], x["kv"], buf["idx"], buf["pos"], x["cs"], x["sink"], buf["tl"], x["ekv"],
                                buf["eidx"], buf["etl"], scale))
        return outs

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        outs = run()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize(dev)
    for i, (got, ref) in enumerate(zip(outs, refs)):
        if got is None:
            return f"VATTN refused set {i} inside the graph"
        if not (torch.equal(got[0].view(torch.uint8), ref[0].view(torch.uint8)) and torch.equal(got[1], ref[1])):
            return f"set {i} differs from the engine's kernel after graph replay"
    del graph, outs, refs, sets, buf
    return ""


def install():
    """Pins, build, self-test; once per process. The method swap itself is routed()'s. Never raises into prepare."""
    if _g["state"] is not None:
        return
    _g["state"] = "off"
    try:
        if ENABLED:
            _install()
    except Exception as e:  # noqa: BLE001  (any failure: VATTN off, the engine's kernel serves)
        _g["state"], _g["lib"] = "off", None
        say("vattn", f"VATTN off: install raised ({type(e).__name__}: {str(e)[:300]})")
    say("vattn", _trigger_line())  # king-v5 VATTN-T: one ON / off line per process
    say("vattn", _pdl_line())      # king-v7 ATTN-PDL-IN: one ON / off line per process
    try:                           # king-v7 AWS (bundle_v9/dec/aws.py): never raises; one ON / off line per process
        from bundle_v9.dec import aws

        aws.install()
        _g["aws"] = aws if aws.serving() else None
    except Exception as e:  # noqa: BLE001
        _g["aws"] = None
        say("vattn", f"AWS off: aws.install raised ({type(e).__name__}: {str(e)[:200]})")


def _trigger_line():
    if _g["state"] != "on":
        return ("VATTN-T off: VATTN is off" if ENABLED else "VATTN-T off: VATTN ENABLED = False") + \
            " (W4 or the engine's kernel serves verify attention)"
    try:
        why = _texts()[1]
    except Exception as e:  # noqa: BLE001
        why = f"texts raised ({type(e).__name__})"
    if why:
        return f"VATTN-T off: {why}; VATTN serves without the dependent-launch trigger"
    return ("VATTN-T ON: VATTN's fwd_kernel issues griddepcontrol.launch_dependents after devfunc (as W4), so "
            "wo_a's PDL launch overlaps the attention tail; the self-test above ran on this cubin")


def _pdl_line():
    """king-v7 ATTN-PDL-IN's boot line: ON only when VATTN serves and the PDL edits are in the texts it was built
    from (so the install self-test, the speed check and every first-use compare ran on the PDL launch)."""
    if _g["state"] != "on":
        return ("ATTN-PDL-IN off: VATTN is off" if ENABLED else "ATTN-PDL-IN off: VATTN ENABLED = False") + \
            " (W4 or the engine's kernel serves verify attention, launched without PDL)"
    try:
        why = _texts()[2]
    except Exception as e:  # noqa: BLE001
        why = f"texts raised ({type(e).__name__})"
    if why:
        return f"ATTN-PDL-IN off: {why}; VATTN launches without the programmatic-serialisation attribute"
    return ("ATTN-PDL-IN ON: VATTN launches with the programmatic-stream-serialisation attribute and executes "
            "griddepcontrol.wait after its CTA-local prologue, before any global access, so its CTAs land and run the "
            "barrier / TMEM prologue while wq_b drains; the self-test above ran on this launch")


def _install():
    if torch.cuda.is_current_stream_capturing() or not torch.cuda.is_available():
        say("vattn", "VATTN off: install ran inside a capture or without CUDA")
        return
    from sglang.srt.runtime_context import get_disagg

    try:
        role = get_disagg().disaggregation_mode
    except ValueError:                                # no engine context published (the 1-GPU rig): decode-like
        role = None
    if role is not None and role not in ROLES:
        say("vattn", f"VATTN off on the {role} role")
        return                                        # a prefill server runs no target-verify attention
    try:
        import flash_mla.fused_norm_rope_attn_rope_cast as MEGA
        from sglang.kernels.jit.utils.arch import get_jit_cuda_arch
        from sglang.kernels.ops.attention import flash_mla_interface as FI
        from sglang.srt.layers.attention.deepseek_v4_mega_backend import DeepseekV4MegaAttnBackend as B
    except ImportError as e:
        say("vattn", f"VATTN off: {e}")
        return
    if not pins_match("vattn", PINS, {"_forward_mega_decode": B._forward_mega_decode,
                                      "flash_mla_mega_decode": FI.flash_mla_mega_decode, "mega.decode": MEGA.decode}):
        return
    if getattr(FI, "_MEGA_FIXED_PARAMS", None) != _FIXED:
        say("vattn", "VATTN off: MegaAttention fixed parameters changed")
        return
    if torch.cuda.get_device_capability() != CAPABILITY:
        say("vattn", f"VATTN off: built and measured for sm_{CAPABILITY[0]}{CAPABILITY[1]}, got "
                     f"{torch.cuda.get_device_capability()}")
        return
    target = get_jit_cuda_arch()                      # read only: the target the engine's own JIT compiles for
    if f"{target.major}{target.minor}{target.suffix}" != ARCH:
        say("vattn", f"VATTN off: the engine's JIT target is {target.target_name}, not the measured sm_{ARCH}")
        return
    if not os.path.exists(_NVCC):
        say("vattn", f"VATTN off: missing {_NVCC}")
        return
    dev = torch.device("cuda", torch.cuda.current_device())
    began = time.monotonic()
    try:
        _g["lib"] = _build()
        verdict = _self_test(dev)
        if verdict.startswith("OK") and not _texts()[2]:   # king-v7 ATTN-PDL-IN: the served (graph, PDL) form
            why = _pdl_check(dev)
            verdict = (f"ATTN-PDL-IN graph check failed: {why}" if why
                       else f"{verdict}; ATTN-PDL-IN graph check OK (3 x 240 rows, producer -> VATTN, 2 replays)")
    except Exception as e:  # noqa: BLE001
        say("vattn", f"VATTN off: build / self-test failed ({type(e).__name__}: {str(e)[:300]})")
        _g["lib"] = None
        return
    finally:
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
    if not verdict.startswith("OK"):
        say("vattn", f"VATTN off: {verdict} (install {time.monotonic() - began:.1f} s)")
        _g["lib"] = None
        return
    _g["cls"], _g["orig"] = B, B._forward_mega_decode
    _g["state"] = "on"
    say("vattn", f"VATTN installed: self-test {verdict}; build + self-test {time.monotonic() - began:.1f} s")


# ------------------------------------------------------------------------------------------------- scope
def enter(forward_batch):
    """Our node's call begins: for a target-verify forward, VATTN's context set and the backend class's
    _forward_mega_decode swapped (outermost entry). Returns the token leave() takes (None: nothing to undo)."""
    if _g["state"] != "on" or forward_batch is None:
        return None
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not (mode.is_target_verify() or (DECODE and mode.is_decode())):
        return None
    token = context.set(True)
    if _g["depth"] == 0:
        setattr(_g["cls"], "_forward_mega_decode", _mega_decode)
        if _g["aws"] is not None:
            _g["aws"].enter()  # king-v7 AWS: MQALayer._mega_core swapped for this forward (never raises)
    _g["depth"] += 1
    return token


def leave(token):
    if token is None:
        return
    _g["depth"] -= 1
    if _g["depth"] == 0:
        setattr(_g["cls"], "_forward_mega_decode", _g["orig"])
        if _g["aws"] is not None:
            _g["aws"].leave()  # king-v7 AWS: the engine's MQALayer._mega_core back
    context.reset(token)


@contextlib.contextmanager
def routed(forward_batch):
    """enter / leave around one forward of our model node, restored in `finally` as v50's levers restore theirs."""
    token = enter(forward_batch)
    try:
        yield
    finally:
        leave(token)
