"""Lever EXM (dspd-w33e, on dsv41_pd001 since dspd-v51; ENABLED = True but NEVER RUN ON A GPU: G-EXM-0..8 are owed):
the exact split mHC boundary at the decode / target-verify mHC boundaries, inside the captured graph B.

Stock runs each of the 77 decode / verify mHC boundaries of a step (37 attention entries in the MXFP8 form, 40 FFN
entries in the bf16 form) as deep_gemm.mega_mhc: a memset of its split barriers plus one persistent 148-CTA kernel whose
critical path waits for the next layer's stats (14-16.5 us per call, 1.17-1.29 ms a step). EXM splits the boundary:
  - exm_crit.cuh on the main stream (PDL): the new residual, the normalized row and the MXFP8 pair, one CTA per
    row, no grid barrier, no tensor core; x1's square sums in stock's (split, parity, column pair) chain topology;
  - exm_stats.cuh on the engine's own hc_stats_stream, forked by an event after the critical kernel: the next
    layer's (pre, post, comb) from the new residual: a stats kernel issuing stock's TF32 UMMA stream on stock's
    (m-block, split) tiles (no CTA waits on another), then a stream-ordered mix kernel, one warp per row.
Every output byte is stock's by construction (struct/w33_exm/EXM_SPEC.md: the arithmetic contract; EXM_FEAS.md: the
route; the six FMA-contraction pins of exm_common.cuh follow stock's SASS). The stats are joined before every reader:
every reader of the triplet goes through deepseek_v4_mhc._compute_triplet (swapped here for a join of EVERY pending
fork, then the engine's function), EXM's own boundary entry joins before it serves or declines (no stock mega_mhc / CMX
cake ever runs beside EXM side work: DeepGEMM #478's %gridid barriers), and end() joins what is left.

On dsv41_pd001 (dspd-v51): post_fusion_floor serves decode-like boundaries up to MEGA_MHC_DECODE_MAX_ROWS (216) without
reaching a mega boundary, so in a decode engine's graphs EXM serves the verify buckets above it (240, 264, 288 rows,
27 splits). AX, PSWEEP, MHCF, PDX and CMX named below are dspd_w38's; none is in this bundle.

When ENABLED (decode role and the null role, the colocated audit / check engine, which grades what decode serves):
  install(model)  pins (engine functions, the DeepGEMM / cute headers EXM includes or re-derives, its own .cuh files),
                  the engine's single hc_stats_stream, MegaMoE's SM margin (the FFN-entry form is declined when the
                  F-side stats could hold the TPC MegaMoE's clustered launch needs), one load_jit build, scratch, event
                  pairs created outside any capture, a self-test of both forms at S = 40 and 27 on three layers' REAL
                  parameters with special rows (zero, x300, x1e-4, zero x, subnormal inputs and outputs, subnormal x1,
                  rows inside stock's K4 rounding band), every output bitwise against stock's mhc_mega_boundary, then
                  DeepJIT's own record of how it built stock's mega_mhc (no fast math / FTZ, the same nvcc as load_jit);
                  on the null role only the checks run here, the rest in its first armed eager forward (RT1, below)
  vote(batch)     called first in slots.forward_model, OUTSIDE its try (no rank-local raise can skip it): in a
                  CAPTURE forward of a decode / verify batch every rank of a served role takes part in one gloo MIN over
                  the TP CPU group of (installed, per (form, bucket): verdict and the full key) whatever its own state
  begin(batch)    armed for decode / target-verify forwards; returns the token end() takes
  boundary(...)   (mhc_mega_boundary's signature) serves a key (form, rows, S, stats grid, stats smem): eager -> its
                  first call compares the WHOLE call with stock's on the same inputs (every output, bit views) and
                  serves EXM's bytes on a match; capture -> only the key the vote AGREED on (no work inside a
                  capture that was not checked: the same rows, split count, grid and smem); anything else -> the
                  previous boundary
  swap / unswap   post_fusion_floor.mhc_mega_boundary (the name the floor's _post_fusion calls; outermost inside
                  v16_levers.routed: declined keys fall to v16_levers.exact_boundary, FlashInfer's cake or stock) and
                  deepseek_v4_mhc._compute_triplet (join only; no kernel), both chaining to what was there
Rank uniformity: a captured bucket serves EXM on every rank or on none (the vote). Eager serving uses the rank-local
verdict: neither path holds a collective (local kernels only), so a divergent eager verdict changes only timing, never
collective participation. The MoE-input form (moe_input; MEGA_GATE only, off in the arena) is declined.

While ENABLED is False: install() returns at once, vote() / begin() return None, swap() returns False, unswap(False) /
end(None) are no-ops; the module imports only os / contextvars / hashlib / json / torch / exm_support.

dspd-w37e (struct/w37e/LANE_W37E.md) applies dspd-w33's EXM FLIP GATES here, model.py unchanged:
  RT1  the null role's prepare runs inside the audit's first EXTEND call, beside the audit references, under the
       300-450 s watchdog: there install keeps to the checks (role, pins, digests, deterministic mode, stats stream,
       MegaMoE margin) and begin() runs the build, the self-test and the DeepJIT record check in the first armed EAGER
       decode / verify forward (never inside a capture), each phase timed in its log line; any failure there leaves
       EXM off, nothing pending, and that forward serves the previous boundary. The decode role still builds inside
       prepare (capture warm-up 1, outside the watchdog); its line now carries the record and total seconds too
  RT2  end() never raises into forward_model's finally: a raising join is logged, pending is cleared and the context is
       reset, so every later restore of the finally runs (the device error surfaces at the engine's next CUDA call)
  RT5  swap() installs the boundary / triplet wrappers only in a forward begin() armed: EXM is absent from every
       EXTEND (AX / MHCF / PDX / sweep-step), idle or unbatched forward
  FIX  (struct/w37e/REVIEW.md F1 / F3 / F4) the deferral runs only in a step no DP rank extends: the forward batch's
       is_extend_in_batch (the DP-global MAX) is False, so AX's gate passes on no rank and its handshake / PSWEEP sweep
       never share that call; it logs when it starts, and that forward logs its EXM.begin-to-EXM.end seconds once
       (the GPU gate grades both); a raising log call in end() or in the deferral's failure path is contained"""
import contextvars
import hashlib
import json
import os

import torch

from dsv41_pd001.exm_support import pins_match, say

ENABLED = True
ROLES = ("decode", "null")         # null = the colocated audit / check engine (eager): it grades what decode serves
DEFER_ROLES = ("null",)            # RT1: prepare runs inside a watchdog-bound call; build at the first armed eager one
FORMS = ("A", "F")                 # A: attention entry, MXFP8 pair; F: FFN entry, bf16 only
# the decode role's target-verify graph buckets per rank: 6 x {1..8, 10..32 step 2, 40, 44, 48} (LANE_DEC H7)
BUCKETS = tuple(6 * b for b in (*range(1, 9), *range(10, 33, 2), 40, 44, 48))
MAX_ROWS = 288                     # the largest bucket; S = 40 up to 192 rows, 27 at 193..320
SPLITS = (40, 27)                  # the split counts EXM was derived and checked for (the mix kernel's instances)
HIDDEN, MULT, MIX = 5120, 4, 24
# stats CTAs per form: 0 = one per task (BURST: the stats retire within ~one task + the mix kernel); n = TRICKLE (n
# CTAs; on the F side only while MegaMoE leaves >= 2 n SMs free: each CTA may hold its own TPC)
STATS_GRID = {"A": 0, "F": 0}
STATS_SMEM = 112 * 1024            # dynamic shared memory of a stats CTA (keeps it off SMs of TMEM-using kernels)
BUCKET_ON = None                   # None: every admitted key; else a frozenset of (form, rows) from the PD-pair A/B
STOCK_NVCC = ""                    # G-EXM-0's DeepJIT nvcc version text; "" = not pinned (the flip copy pins it)
SELF_ROWS = (6, 96, 145, 192, 193, 240, 288)
SELF_LAYERS = (2, 3, 21)
K4_BAND = (1e-7, 3e-7, 1e-8, 3e-8, 2e-7, 5e-8)  # self-test scales whose x1 mean square sits in stock's K4 band
EVENTS = 4                         # fork / join event pairs per process (one fork is in flight at a time)
SCRATCH_MBLOCKS = 5                # <= 320 rows
_DIR = os.path.dirname(os.path.abspath(__file__))
CUH = ("exm_common.cuh", "exm_crit.cuh", "exm_stats.cuh")
# EXM's own sources (sha256[:16]); a copy edited without updating these never builds
CUH_SHAS = {"exm_common.cuh": "7d38d05fcfb0eb19", "exm_crit.cuh": "6333d3b462df0f53",
            "exm_stats.cuh": "81a5b37578161225"}
# engine functions EXM mirrors, wraps or relies on (srcpin digests of the engine @15338a8e, struct/w27/tools/srcpin.py)
PINS = {"mhc_mega_boundary": "13c41e5350804e1d", "_post_fusion": "a34c362a74c4d4ec",
        "_compute_triplet": "b3e5a53247eff8b6", "run_attn_post": "77de828299d2aada", "run_moe_post": "62431f4cc899ea63",
        "empty_deepgemm_mxfp8_pair": "78230a1c8044d010", "empty_deepgemm_mxfp8_scales": "66265add96179a29",
        "use_stats_stream": "989b71903bf0b348", "fork_stats_stream": "1bab6d438f971cc6",
        "mega_moe_input_of": "6f4ad8ef85215a4a", "HcState.take_rows": "ee1bc5aee910d078",
        "HcState.with_residual": "110c50962431a3c2", "HcUpdate.post": "50421527eb086367",
        "HcReadCall.read": "7f4b5e13c0c29fe2", "deep_gemm.mega_mhc": "0d4726a09c49f6e4",
        "_mega_moe_max_num_sms": "d3ab97a443dbf0e8", "FullCudaGraphBackend.capture_one": "b11da397ea881b30",
        "device_compiler_path": "64362ea774fe21f7"}
# the installed DeepGEMM headers EXM re-derives (mega_mhc) or includes (sha256[:16]; identical in kernel_pkgs, the
# 0.2.3.post2 research copy and containerd snapshot 536)
DG_PINS = {"deep_gemm/impls/sm100_mega_mhc.cuh": "77fe3a79ad04c803",
           "deep_gemm/epilogue/sm100_mega_mhc.cuh": "5bfb051b4d5d3705",
           "deep_gemm/layout/mega_mhc.cuh": "e044a60196bb8cbb", "deep_gemm/scheduler/mega_mhc.cuh": "bf774cfaa8ce01cd",
           "deep_gemm/common/math.cuh": "809b3d5e162f01b2", "deep_gemm/ptx/utils.cuh": "8a9d5e03afdeb20f",
           "deep_gemm/mma/sm100.cuh": "439c57d8aa4914c0", "deep_gemm/ptx/tcgen05.cuh": "69a3f9a8c6291aa4",
           "deep_gemm/common/compile.cuh": "a13373053608b286", "cute/arch/mma_sm100_umma.hpp": "7ba0cef9f7fe4714",
           "cute/arch/mma_sm100_desc.hpp": "762d488e441c4949", "cute/arch/copy_sm100.hpp": "43f813e1e4541bc6",
           "cute/arch/tmem_allocator_sm100.hpp": "2452d93263bfbdf2", "cutlass/arch/barrier.h": "90716cfd8e3300ca"}
# nvcc's float flags, explicit (DeepJIT's fast_math = false defaults; install checks stock's build record)
CUDA_FLAGS = ["-ftz=false", "-prec-div=true", "-prec-sqrt=true", "-fmad=true", "--expt-extended-lambda"]
# command-line forms that would change stock's float arithmetic (DeepJIT options.hpp:104-108 and nvcc's spellings)
BAD_FLAGS = ("--use_fast_math", "-use_fast_math", "--use-fast-math", "-ftz=true", "--ftz=true", "-fmad=false",
             "--fmad=false")
context = contextvars.ContextVar("dspd_exm", default=False)
# state: None = not installed, False = off, dict = on; verdict: key -> bool (rank-local, eager first use; key = (form,
# rows, S, grid, smem)); agreed: (form, rows) -> the key the last capture vote agreed on; pending: [(join event, main
# stream, refs)]; forms: the forms served (F declined on a short MegaMoE margin); role: the engine's; defer: what the
# checks of a DEFER_ROLES install hand to its first armed eager forward (RT1), None once that forward ran or off
_g = {"state": None, "verdict": {}, "agreed": {}, "pending": [], "prev": None, "prev_triplet": None, "eng": None,
      "role": None, "votes": 0, "ev_next": 0, "forms": FORMS, "margin": None, "defer": None}


def _file_sha(path):
    """sha256[:16] of a file through os-level reads (the sandbox bans the open() builtin)."""
    return hashlib.sha256(_read(path)).hexdigest()[:16]


def _read(path, limit=1 << 26):
    fd, out = os.open(path, os.O_RDONLY), []
    try:
        size = 0
        for chunk in iter(lambda: os.read(fd, 1 << 20), b""):
            out.append(chunk)
            size += len(chunk)
            if size > limit:
                raise OSError(f"{path}: larger than {limit} bytes")
    finally:
        os.close(fd)
    return b"".join(out)


def _now():
    return os.times().elapsed                  # wall-clock seconds (no extra import)


def install(model=None):
    """Once per process (slots.prepare_model). A no-op while ENABLED is False; any failure leaves EXM off and nothing
    raises into prepare. The decode role builds and self-tests here (capture warm-up 1); a DEFER_ROLES role (null)
    stops after the checks and its first armed eager forward finishes the install (RT1: _deferred)."""
    if not ENABLED or _g["state"] is not None:
        return
    _g["state"] = False
    if not _role_ok():                                   # first: vote() reads the role even when install fails later
        say("exm", f"EXM off: role {_g['role']!r} (served roles {ROLES})")
        return
    try:
        pre = _checks(model)
        if pre and _g["role"] in DEFER_ROLES:
            _g["defer"] = pre
            say("exm", f"EXM checks OK (role {_g['role']}, {pre['checks_s']:.1f} s): build, self-test and DeepJIT "
                       f"record check deferred to the first armed eager decode / verify forward (RT1)")
            return
        st = _finish(pre) if pre else False
        if st:
            _g["state"] = st
            say("exm", _installed(st, "in prepare"))
    except Exception as e:  # noqa: BLE001
        say("exm", f"EXM off: install raised ({type(e).__name__}: {str(e)[:200]})")


def _installed(st, where):
    return (f"EXM installed (role {_g['role']}): split mHC at decode / verify boundaries, forms {_g['forms']}, splits "
            f"{SPLITS}, <= {MAX_ROWS} rows, stats grid {STATS_GRID}, smem {STATS_SMEM}, MegaMoE margin "
            f"{_g['margin']} SMs; {st['times']} ({where})")


def _deferred():
    """RT1: the null role's build, self-test and DeepJIT record check, in its first armed EAGER decode / verify forward
    of a step no DP rank extends (begin, before any swap; inside a capture it waits for an eager one). Rank-local and
    safe there: the role is eager and EXM's eager serving uses local kernels only. Never raises: any failure leaves EXM
    off with nothing pending, so this forward serves the previous boundary. On success the state carries this
    forward's start for end()'s one first-forward line (FIX F3)."""
    try:
        if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
            return
        pre, _g["defer"] = _g["defer"], None
        t = _now()
        say("exm", f"EXM deferred install: started (role {_g['role']}) in an armed eager decode / verify forward of a "
                   f"step no DP rank extends")
        if torch.cuda.current_device() != pre["dev_index"]:
            say("exm", f"EXM off: the current device {torch.cuda.current_device()} is not prepare's "
                       f"{pre['dev_index']}")
            return
        st = _finish(pre)
        if st:
            st["first"] = (t, _now() - t)
            _g["state"] = st
            say("exm", _installed(st, f"deferred: {st['first'][1]:.1f} s in the first armed eager decode / verify "
                                      f"forward"))
        else:
            _drop_pending()
            say("exm", f"EXM off (role {_g['role']}): the deferred install refused after {_now() - t:.1f} s; the "
                       f"previous boundary serves")
    except Exception as e:  # noqa: BLE001  (a build / self-test / record raise: off, the previous boundary serves)
        _g["defer"], _g["state"] = None, False
        _drop_pending()
        _tell(f"EXM off: the deferred install raised ({type(e).__name__}: {str(e)[:200]})")


def _tell(what):
    """say() for a failure path that must never raise (FIX F4: a broken stderr inside end() or the deferral's handler
    would otherwise escape into slots.py and skip the restores RT2 protects)."""
    try:
        say("exm", what)
    except Exception:  # noqa: BLE001  (nothing left to tell it to)
        pass


def _drop_pending():
    """Join what a failed self-test left forked, if the device still can; drop it either way."""
    try:
        _join_all()
    except Exception:  # noqa: BLE001  (a broken device: nothing to order behind)
        pass
    finally:
        _g["pending"].clear()


def _checks(model):
    """Everything install verifies before it builds (all a DEFER_ROLES prepare does): the role, no PD multiplexing /
    two-batch overlap, no capture, the engine's own boundary, the engine pins, the DeepGEMM / cute header digests and
    EXM's own .cuh digests, DeepGEMM's split mode, one hc_stats_stream for every layer, MegaMoE's SM margin. Returns
    what _finish needs, or False (EXM off, the reason said)."""
    import deep_gemm

    from sglang.kernels.jit.utils.compile import toolchain as TC
    from sglang.kernels.ops.layernorm import mhc_mega as KM
    from sglang.srt.layers.moe import mega_moe as MM
    from sglang.srt.model_executor.runner_backend import full_cuda_graph_backend as FG
    from sglang.srt.models import deepseek_v4_boundary as BD
    from sglang.srt.models import deepseek_v4_mhc as MHC
    from sglang.srt.runtime_context import get_disagg, get_exec

    from dsv41_pd001 import post_fusion_floor

    t0 = _now()
    dis = get_disagg()
    if dis.disaggregation_mode not in ROLES or dis.enable_pdmux or get_exec().overlap.enable_two_batch_overlap:
        say("exm", f"EXM off: role {dis.disaggregation_mode}, PD multiplexing / two-batch overlap must be off")
        return False
    if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        say("exm", "EXM off: prepare ran inside a capture / no CUDA")
        return False
    if MHC.mhc_mega_boundary is not KM.mhc_mega_boundary or \
            post_fusion_floor.mhc_mega_boundary is not KM.mhc_mega_boundary:
        say("exm", "EXM off: deepseek_v4_mhc's or the floor's mhc_mega_boundary is not sglang's function")
        return False
    fns = {"mhc_mega_boundary": KM.mhc_mega_boundary, "_post_fusion": MHC._post_fusion,
           "_compute_triplet": MHC._compute_triplet, "run_attn_post": MHC.run_attn_post,
           "run_moe_post": MHC.run_moe_post, "empty_deepgemm_mxfp8_pair": MHC.empty_deepgemm_mxfp8_pair,
           "empty_deepgemm_mxfp8_scales": MHC.empty_deepgemm_mxfp8_scales,
           "use_stats_stream": MHC.use_stats_stream, "fork_stats_stream": MHC.fork_stats_stream,
           "mega_moe_input_of": MHC.mega_moe_input_of, "HcState.take_rows": MHC.HcState.take_rows,
           "HcState.with_residual": MHC.HcState.with_residual, "HcUpdate.post": BD.HcUpdate.post,
           "HcReadCall.read": BD.HcReadCall.read, "deep_gemm.mega_mhc": deep_gemm.mega_mhc,
           "_mega_moe_max_num_sms": MM._mega_moe_max_num_sms,
           "FullCudaGraphBackend.capture_one": FG.FullCudaGraphBackend.capture_one,
           "device_compiler_path": TC.device_compiler_path}
    if not pins_match("exm", PINS, fns):
        return False
    dg_inc = os.path.join(os.path.dirname(os.path.abspath(deep_gemm.__file__)), "include")
    got = {rel: (_file_sha(os.path.join(dg_inc, rel)) if os.path.exists(os.path.join(dg_inc, rel)) else "")
           for rel in DG_PINS}
    own = {name: _file_sha(os.path.join(_DIR, name)) for name in CUH}
    if got != DG_PINS or own != CUH_SHAS:
        say("exm", f"EXM off: header / source digests MISMATCH {got} {own}")
        return False
    if deep_gemm.get_deterministic_algorithms():
        say("exm", "EXM off: DeepGEMM deterministic mode (16 splits)")
        return False
    side = getattr(model, "hc_stats_stream", None)
    layers = list(getattr(model, "layers", []))
    if side is None or not layers or any(getattr(lyr, "hc_stats_stream", None) is not side for lyr in layers):
        say("exm", "EXM off: no single engine hc_stats_stream shared by every layer")
        return False
    index = torch.cuda.current_device()
    dev = torch.device("cuda", index)
    physical = torch.cuda.get_device_properties(dev).multi_processor_count
    forms, margin = _served_forms(MM._mega_moe_max_num_sms(), physical)
    _g["forms"], _g["margin"] = forms, margin
    if not forms:
        say("exm", f"EXM off: no form left to serve (MegaMoE margin {margin} SMs)")
        return False
    return {"model": model, "dev": dev, "dev_index": index, "side": side, "dg_inc": dg_inc, "MHC": MHC,
            "floor": post_fusion_floor, "stock": KM.mhc_mega_boundary, "dg": deep_gemm, "TC": TC,
            "checks_s": _now() - t0}


def _finish(pre):
    """The build, the self-test of the served forms at S = 40 / 27 on three layers' REAL parameters and DeepJIT's record
    of stock's mega_mhc build, in dspd-w33's order (decode role: right after _checks, inside prepare; null role: the
    first armed eager decode / verify forward). Returns the state, or False (EXM off, the reason said)."""
    model, dev, forms = pre["model"], pre["dev"], _g["forms"]
    t1 = _now()
    t0 = t1 - pre["checks_s"]
    st = _state(dev, pre["side"], _build(pre["dg_inc"]))
    t2 = _now()
    _g["eng"] = {"MHC": pre["MHC"], "floor": pre["floor"], "stock": pre["stock"], "dg": pre["dg"]}
    for idx, params in _self_params(model, dev):
        for rows in SELF_ROWS:
            for form in forms:
                if _splits(rows) not in SPLITS:
                    say("exm", f"EXM off: {rows} rows do not select {SPLITS} splits here")
                    return False
                if not _self_test(st, rows, form, params, dev):
                    say("exm", f"EXM off: self-test DIFFERENT at {rows} rows, form {form}, layer {idx}")
                    return False
    torch.cuda.synchronize(dev)
    t3 = _now()
    why = _toolchain_verdict(_stock_records(_dj_roots()), pre["TC"].device_compiler_path())
    t4 = _now()
    if why is not None:
        say("exm", f"EXM off: stock's build record: {why}")
        return False
    st["times"] = (f"checks {t1 - t0:.1f} s, build {t2 - t1:.1f} s, self-test {t3 - t2:.1f} s, record "
                   f"{t4 - t3:.1f} s, total {t4 - t0:.1f} s")
    say("exm", f"EXM self-test OK: rows {SELF_ROWS}, forms {forms}, layers {SELF_LAYERS} (real parameters)")
    return st


def _served_forms(max_sms, physical):
    """(forms, margin): MegaMoE launches min(max_sms, current) rounded down to even CTAs as 2-CTA clusters across the
    whole grid (engine srt/layers/moe/mega_moe.py _mega_moe_max_num_sms / _configure_mega_moe_deep_gemm_num_sms). The
    FFN-entry stats run beside it: the F form is served only while that launch leaves one TPC (2 SMs) free, and 2 SMs
    per TRICKLE CTA (REVIEW_exm_runtime F2); None = no clustered launch (pre-SM100), the margin is the whole GPU."""
    used = 0 if max_sms is None else int(max_sms) - int(max_sms) % 2
    margin = int(physical) - used
    grid = int(STATS_GRID["F"])
    ok_f = margin >= 2 and (grid <= 0 or margin >= 2 * grid)
    return tuple(f for f in FORMS if f != "F" or ok_f), margin


def _build(dg_inc):
    """The one JIT module (exm_crit.cuh + exm_stats.cuh) through the engine's load_jit and its default toolchain
    (toolchain.device_compiler_path(): CUDA_HOME / CUDA_PATH / which nvcc / /usr/local/cuda, as DeepJIT resolves it;
    no compiler override). Also what the GPU rigs build (struct/w33_exm/gpu/g_exm_bits.py)."""
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    pdl = make_cpp_args(bool(is_arch_support_pdl()))
    return load_jit("dspd_exm", "".join(CUH_SHAS[n][:8] for n in CUH), *pdl,
                    cuda_files=[os.path.join(_DIR, n) for n in CUH[1:]], extra_include_paths=[_DIR, dg_inc],
                    extra_cuda_cflags=CUDA_FLAGS,
                    cuda_wrappers=[("crit_a", f"sglang::exm::ExmCrit<{pdl}>::run"),
                                   ("crit_f", f"sglang::exm::ExmCrit<{pdl}>::run_f"),
                                   ("stats", "sglang::exm::ExmStats::run")])


def _state(dev, side, mod):
    """Per-process state: the module, the engine's stats stream, event pairs created now (outside any capture) and the
    split workspace (<= SCRATCH_MBLOCKS m-blocks x 40 splits; stream-ordered, no tickets)."""
    n_tasks = SCRATCH_MBLOCKS * max(SPLITS)
    cur = torch.cuda.current_stream(dev)
    events = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(EVENTS)]
    for a, b in events:
        a.record(cur)
        b.record(cur)
    return {"dev": dev.index, "mod": mod, "side": side, "events": events,
            "gemm": torch.empty((n_tasks * 64 * MIX,), dtype=torch.float32, device=dev),
            "hc": torch.empty((n_tasks * 64,), dtype=torch.float32, device=dev)}


def _dj_roots():
    """DeepJIT's disk-cache roots as DiskCache::from_env reads them (deep_jit/cache/disk.hpp:85-105): DG_ (then DJ_)
    JIT_CACHE_DIR as a ':' list, else $HOME/.dj."""
    for name in ("DG_JIT_CACHE_DIR", "DJ_JIT_CACHE_DIR"):
        value = os.getenv(name, "")
        if value:
            return [p for p in value.split(":") if p]
    home = os.getenv("HOME", "")
    return [os.path.join(home, ".dj")] if home else []


def _stock_records(roots):
    """DeepJIT's meta.json of every committed cache entry whose kernel.cu instantiates stock's mega_mhc (written by
    backend.hpp:160-167 beside kernel.cubin); None for such an entry whose record cannot be read."""
    recs = []
    for root in roots:
        cache = os.path.join(root, "cache")
        try:
            names = sorted(os.listdir(cache))
        except OSError:
            continue
        for name in names:
            d = os.path.join(cache, name)
            try:
                if not os.path.exists(os.path.join(d, ".committed")) or b"sm100_mega_mhc" not in _read(
                        os.path.join(d, "kernel.cu")):
                    continue
            except OSError:
                continue
            try:
                recs.append(json.loads(_read(os.path.join(d, "meta.json")).decode("utf-8")))
            except (OSError, ValueError):
                recs.append(None)
    return recs


def _toolchain_verdict(recs, nvcc):
    """None when every record of stock's mega_mhc builds shows no fast math, no FTZ, FMA contraction on, the nvcc that
    load_jit builds EXM with (same resolved path) and, when pinned, STOCK_NVCC's version; else the reason (REVIEW_exm_
    exact F4: the K-pins and the FTZ choice belong to that toolchain)."""
    if not recs:
        return "no DeepJIT record of a mega_mhc build"
    want = os.path.realpath(str(nvcc))
    for r in recs:
        if not isinstance(r, dict):
            return "an unreadable DeepJIT record"
        opts, info = r.get("compiler_options") or {}, r.get("compiler_info") or {}
        cmd = str(r.get("command", "")).split()
        pairs = {(a, b) for a, b in zip(cmd, cmd[1:])}
        if opts.get("use_fast_math") or any(f in cmd for f in BAD_FLAGS) or pairs & {("-ftz", "true"),
                                                                                       ("-fmad", "false")}:
            return "fast math / FTZ / no-FMA flags in stock's build"
        if os.path.realpath(str(info.get("path", ""))) != want:
            return f"stock's nvcc {info.get('path')!r} is not load_jit's {nvcc!r}"
        if STOCK_NVCC and STOCK_NVCC not in str(info.get("version", "")):
            return f"stock's nvcc version is not the pinned {STOCK_NVCC!r}"
    return None


def _self_params(model, dev):
    """(layer, (fn, scale, base, weight, rms_eps, hc_eps, norm_eps, iters)) per form of the self-test layers, from the
    REAL mHC parameters (attention entry: attn_hc; FFN entry: ffn_hc); synthetic ones when a layer has none."""
    out = []
    layers = list(getattr(model, "layers", []))
    for idx in SELF_LAYERS:
        lyr = layers[idx] if idx < len(layers) else None
        for name, hc in (("attn_hc", getattr(lyr, "attn_hc", None)), ("ffn_hc", getattr(lyr, "ffn_hc", None))):
            if hc is not None:
                out.append((f"{idx}.{name}", (hc.fn, hc.scale, hc.base, hc.norm.weight, hc.cfg.rms_eps, hc.cfg.eps,
                                              hc.norm.variance_epsilon, hc.cfg.sinkhorn_iters)))
    if not out:
        gen = torch.Generator(device=dev)
        gen.manual_seed(20261031)
        out.append(("synthetic", (torch.randn((MIX, MULT * HIDDEN), generator=gen, device=dev) * 0.01,
                                  torch.rand(3, generator=gen, device=dev) + 0.5,
                                  torch.randn(MIX, generator=gen, device=dev) * 0.1,
                                  (torch.rand(HIDDEN, generator=gen, device=dev) + 0.5).to(torch.bfloat16),
                                  1e-20, 1e-6, 1e-20, 20)))
    return out


def _subnormal_bf16(shape, gen, dev):
    """bf16 subnormals built from bits (sign, exponent 0, mantissa 1..127): no device flush can touch them."""
    mant = torch.randint(1, 128, shape, generator=gen, device=dev, dtype=torch.int32)
    sign = torch.randint(0, 2, shape, generator=gen, device=dev, dtype=torch.int32)
    return (mant - 32768 * sign).to(torch.int16).view(torch.bfloat16)    # int16 bits 0x8000 | m for a negative one


def _self_rows(rows, dev, seed):
    """The self-test rows: randn plus, by row index, 0 zero, 1 x300, 2 x1e-4, 3 zero x, 4 x and route 0 subnormal,
    5 x and EVERY route subnormal (subnormal outputs: FTZ probe), 6 x1 subnormal before the norm (routes 1e-36, pre
    x1e-3), and every 2nd row from 8 scaled into stock's K4 band (x1 RMS 1e-8..3e-7, where K4 FFMA and FMUL + FADD
    differ; the next layer's hc norm of those rows sits in K1's band too; REVIEW_exm_exact F1 / F6)."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    x = torch.randn((rows, HIDDEN), generator=gen, device=dev)
    res = torch.randn((rows, MULT, HIDDEN), generator=gen, device=dev)
    pre = torch.softmax(torch.randn((rows, MULT), generator=gen, device=dev), dim=-1)
    post = 2.0 * torch.sigmoid(torch.randn((rows, MULT), generator=gen, device=dev))
    comb = torch.softmax(torch.randn((rows, MULT, MULT), generator=gen, device=dev), dim=-1)
    scale = torch.ones((rows,), device=dev)
    scale[1:2] = 300.0
    scale[2:3] = 1e-4
    scale[6:7] = 1e-36
    for t in range(8, rows, 2):
        scale[t] = K4_BAND[(t // 2) % len(K4_BAND)]
    x = (x * scale[:, None]).to(torch.bfloat16)
    res = (res * scale[:, None, None]).to(torch.bfloat16)
    x[0:1] = 0.0
    res[0:1] = 0.0
    x[3:4] = 0.0
    if rows > 4:
        x[4] = _subnormal_bf16((HIDDEN,), gen, dev)
        res[4, 0] = _subnormal_bf16((HIDDEN,), gen, dev)
    if rows > 5:
        x[5] = _subnormal_bf16((HIDDEN,), gen, dev)
        res[5] = _subnormal_bf16((MULT, HIDDEN), gen, dev)
    pre[6:7] *= 1e-3
    return x.contiguous(), res.contiguous(), pre.contiguous(), post.contiguous(), comb.contiguous()


def _self_test(st, rows, form, params, dev):
    """_self_rows with real parameters through EXM's serve path (fork / join included) and stock's boundary; every
    output bitwise."""
    x, res, pre, post, comb = _self_rows(rows, dev, 20261031 + rows)
    fn, scale, base, weight, rms_eps, hc_eps, norm_eps, iters = params
    args = (x, res, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps, iters)
    pair = _g["eng"]["MHC"].empty_deepgemm_mxfp8_pair(rows, HIDDEN, dev) if form == "A" else (None, None)
    return _check(st, form, _splits(rows), args, *pair)[0]


def _splits(rows):
    """DeepGEMM's host split rule (EXM_SPEC 3) on this call's rows and the live SM count; None in deterministic mode."""
    dg = _g["eng"]["dg"]
    if dg.get_deterministic_algorithms():
        return None
    m_blocks = -(-rows // 64)
    longest = -(-80 // min(64, max(16, dg.get_num_sms() // m_blocks)))
    return -(-80 // longest)


def _key(form, rows):
    """(form, rows, S, stats grid, stats smem): everything a served boundary's launch depends on (REVIEW_exm_exact F7,
    runtime F8), or None when EXM does not serve that form / split count here."""
    if form not in _g["forms"]:
        return None
    splits = _splits(rows)
    return None if splits not in SPLITS else (form, int(rows), splits, int(STATS_GRID[form]), int(STATS_SMEM))


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.uint8) if t.element_size() == 1 else \
        t.view(torch.int32)


def _same_all(ref, out, pair_ref, pair_out):
    pairs = [(ref[0], out[0]), (ref[1], out[1]), (ref[2][0], out[2][0]), (ref[2][1], out[2][1]),
             (ref[2][2], out[2][2])]
    if pair_ref is not None:
        pairs += [(pair_ref[0], pair_out[0]), (pair_ref[1], pair_out[1])]
    return all(a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(_bits(a), _bits(b))) for a, b in pairs)


def _check(st, form, splits, args, y_fp8, y_sf):
    """Stock's whole boundary into fresh outputs, then EXM's serve path into the caller's buffers, joined and synced
    on the host, every output compared bit for bit. Returns (same, EXM's result)."""
    MHC = _g["eng"]["MHC"]
    x = args[0]
    _join_all()
    if form == "A":
        f, s = MHC.empty_deepgemm_mxfp8_pair(x.shape[0], HIDDEN, x.device)
        ref = _g["eng"]["stock"](*args, y_fp8=f, y_gemm_sf=s)
        pair_ref = (f, s)
    else:
        ref = _g["eng"]["stock"](*args, y_fp8=None, y_gemm_sf=None, moe_input=None)
        pair_ref = None
    out = _launch(st, form, splits, args, y_fp8, y_sf)
    _join_all()
    torch.cuda.current_stream(x.device).synchronize()
    return _same_all(ref, out, pair_ref, None if pair_ref is None else (y_fp8, y_sf)), out


def _event_pair(st):
    """The next fork / join pair; joins everything first when every pair is still pending."""
    if len(_g["pending"]) >= len(st["events"]):
        _join_all()
    pair = st["events"][_g["ev_next"] % len(st["events"])]
    _g["ev_next"] += 1
    return pair


def _launch(st, form, splits, args, y_fp8, y_sf):
    """The served boundary: critical kernel on the current stream, then the stats + mix kernels forked onto the
    engine's hc_stats_stream; the stats tuple stays pending (its tensors held in `pending`) until a join."""
    x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps, iters = args
    mod, side = st["mod"], st["side"]
    main = torch.cuda.current_stream(x.device)
    rows = x.shape[0]
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    next_pre = torch.empty((rows, MULT, 1), dtype=torch.float32, device=x.device)    # stock's [T, 4, 1] mix outputs
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty_like(comb)
    if form == "A":
        mod.crit_a(x, residual, pre, post, comb, weight, updated, normalized, y_fp8, y_sf, splits, float(norm_eps), 1.0,
                   main.cuda_stream)
    else:
        mod.crit_f(x, residual, pre, post, comb, weight, updated, normalized, splits, float(norm_eps), 1.0,
                   main.cuda_stream)
    fork, join = _event_pair(st)
    fork.record(main)
    side.wait_event(fork)
    refs = (updated, next_pre, next_post, next_comb)
    try:
        mod.stats(updated, fn, scale, base, next_pre.view(rows, MULT), next_post.view(rows, MULT), next_comb,
                  st["gemm"], st["hc"], splits, STATS_GRID[form], STATS_SMEM, float(rms_eps), float(hc_eps), 2.0,
                  float(hc_eps), int(iters), side.cuda_stream)
    finally:                                   # even on a raise: the side stream is joined back (no open fork)
        join.record(side)
        _g["pending"].append((join, main, refs))
    if not torch.cuda.is_current_stream_capturing():
        # eager only: `pending` already keeps every block alive until a join orders its reuse behind the stats; in a
        # capture record_stream would only defer the frees and grow the graph pool (REVIEW_exm_runtime F6)
        for t in refs:
            t.record_stream(side)
    return updated, normalized, (next_pre.squeeze(-1), next_post.squeeze(-1), next_comb)


def _join_all():
    """Make the current stream (and every fork's main stream) wait for every pending stats kernel."""
    if not _g["pending"]:
        return
    cur = torch.cuda.current_stream()
    for join, main, _refs in _g["pending"]:
        cur.wait_event(join)
        if main != cur:
            main.wait_event(join)
    _g["pending"].clear()


def _role_ok():
    """The engine's role, read once (install reads it first; a rank whose install never ran reads it here), so the
    vote below depends on nothing rank-local."""
    if _g["role"] is None:
        try:
            from sglang.srt.runtime_context import get_disagg
            _g["role"] = get_disagg().disaggregation_mode
        except Exception:  # noqa: BLE001  (the same import on every rank of the engine)
            _g["role"] = ""
    return _g["role"] in ROLES


def _vote_keys():
    return [(f, r) for f in FORMS for r in BUCKETS]


def _local_vote(keys):
    """This rank's vote: [installed, smem, -smem] + per (form, bucket) [verdict of its full key, code, -code] with code
    = S + 64 (grid + 1) of the key this rank would capture now (0: none). MIN over ranks then shows agreement on the
    verdict AND on S / grid / smem (min code == max code)."""
    st = _g["state"]
    smem = int(STATS_SMEM)
    out = [1 if st else 0, smem, -smem]
    for f, r in keys:
        key = _key(f, r) if st else None
        code = 0 if key is None else key[2] + 64 * (key[3] + 1)
        out += [1 if key is not None and _g["verdict"].get(key) is True else 0, code, -code]
    return out


def _agreed(got, keys):
    if not (got[0] == 1 and got[1] == -got[2]):
        return {}
    out = {}
    for i, (f, r) in enumerate(keys):
        flag, lo, neg_hi = got[3 + 3 * i:6 + 3 * i]
        if flag == 1 and lo == -neg_hi and lo > 0:
            out[(f, r)] = (f, r, lo % 64, lo // 64 - 1, got[1])
    return out


def _vote():
    """One MIN over the TP group's gloo (CPU) group; every rank of a served role runs it in every capture forward
    whatever its own state, so the collective always pairs up. A local raise still votes (zeros); a failed collective
    serves the previous boundary in this capture."""
    keys = _vote_keys()
    try:
        local = _local_vote(keys)
    except Exception as e:  # noqa: BLE001  (still take part: this rank votes nothing)
        local = [0] * (3 + 3 * len(keys))
        say("exm", f"EXM vote: local table raised ({type(e).__name__}); voting zeros")
    try:
        import torch.distributed as dist

        from sglang.srt.runtime_context import get_parallel
        v = torch.tensor(local, dtype=torch.int64)
        dist.all_reduce(v, op=dist.ReduceOp.MIN, group=get_parallel().tp_group.cpu_group)
        _g["agreed"] = _agreed([int(a) for a in v.tolist()], keys)
        say("exm", f"EXM capture vote: {len(_g['agreed'])} of {len(keys)} (form, bucket) keys agreed")
    except Exception as e:  # noqa: BLE001  (a failed vote serves stock in this capture: local kernels only)
        _g["agreed"] = {}
        say("exm", f"EXM vote failed ({type(e).__name__}: {str(e)[:200]}); stock in this capture")
    _g["votes"] += 1


def _armed(forward_batch):
    mode = getattr(forward_batch, "forward_mode", None) if forward_batch is not None else None
    return mode is not None and bool(mode.is_decode() or mode.is_target_verify())


def vote(forward_batch):
    """slots.forward_model's first call, outside its try (REVIEW_exm_runtime F7): a capture forward of a decode / verify
    batch on a served role votes here, before any lever's rank-local enter / begin could raise."""
    if not ENABLED:
        return None
    try:
        go = _armed(forward_batch) and _role_ok() and torch.cuda.is_available() and \
            torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001  (attribute reads; identical on every rank)
        go = False
    if go:
        _vote()
    return None


def begin(forward_batch):
    """Our model call starts. Armed only for a decode / target-verify forward; on a DEFER_ROLES role the first armed
    eager one of a step no DP rank extends finishes the install first (RT1, _deferred: never raises). FIX F1:
    is_extend_in_batch is the DP-global MAX, so in a step where any rank extends AX's gate passes on every rank (its
    pending handshake / PSWEEP sweep runs in that call): the deferral waits for a step where it is False on every
    rank. Returns the token `end` takes, or None."""
    if not ENABLED or not _armed(forward_batch):
        return None
    if _g["defer"] is not None and not getattr(forward_batch, "is_extend_in_batch", False):
        _deferred()
    if not _g["state"]:
        return None
    return context.set(True)


def end(token):
    """Our model call ends (slots.forward_model's finally): join every fork still pending, then disarm. Never raises
    (RT2): a raising join (a sticky device error, an invalidated capture) is logged and its forks dropped, and the context is
    reset, so the later statements of forward_model's finally all run; the error surfaces at the engine's next CUDA call.
    FIX F3: the forward that ran the deferral logs its EXM.begin-to-EXM.end seconds once (host clock; the call's
    references and the work before EXM.begin are outside it). FIX F4: no log call here can raise."""
    if token is None:
        return
    try:
        _join_all()
    except Exception as e:  # noqa: BLE001
        _tell(f"EXM end: the join raised ({type(e).__name__}: {str(e)[:200]}); its forks dropped")
    finally:
        _g["pending"].clear()
        try:
            context.reset(token)
        except Exception:  # noqa: BLE001  (a token of another context: disarm this one anyway)
            context.set(False)
    st = _g["state"]
    first = st.pop("first", None) if isinstance(st, dict) else None
    if first is not None:
        _tell(f"EXM first armed forward: {_now() - first[0]:.1f} s from begin to end (deferred {first[1]:.1f} s; the "
              f"call's references and the work before begin are not in it)")


def _admit(st, x, residual, pre, post, comb, fn, scale, base, weight, sinkhorn_iters, y_fp8, y_gemm_sf, moe_input):
    """The full key when EXM may serve this boundary, else None."""
    rows = x.shape[0]
    if moe_input is not None or (y_fp8 is None) != (y_gemm_sf is None) or not 0 < rows <= MAX_ROWS:
        return None
    if not (int(sinkhorn_iters) >= 1 and x.device.index == st["dev"]
            and x.dtype == residual.dtype == weight.dtype == torch.bfloat16
            and tuple(x.shape) == (rows, HIDDEN) and tuple(residual.shape) == (rows, MULT, HIDDEN)
            and tuple(pre.shape) == tuple(post.shape) == (rows, MULT) and tuple(comb.shape) == (rows, MULT, MULT)
            and tuple(fn.shape) == (MIX, MULT * HIDDEN) and tuple(scale.shape) == (3,) and tuple(base.shape) == (MIX,)
            and tuple(weight.shape) == (HIDDEN,)
            and pre.dtype == post.dtype == comb.dtype == fn.dtype == scale.dtype == base.dtype == torch.float32
            and all(t.is_contiguous() and t.device == x.device for t in (x, residual, pre, post, comb, fn, scale,
                                                                          base, weight))
            and all(t.data_ptr() % 16 == 0 for t in (x, residual, pre, post, comb, fn, weight))):
        return None
    if y_fp8 is not None and not (y_fp8.dtype == torch.float8_e4m3fn and tuple(y_fp8.shape) == (rows, HIDDEN)
                                  and y_fp8.is_contiguous() and y_fp8.data_ptr() % 16 == 0
                                  and y_gemm_sf.dtype == torch.int32 and tuple(y_gemm_sf.shape) == (rows, HIDDEN // 128)
                                  and y_gemm_sf.stride() == (1, -(-rows // 4) * 4)):
        return None
    form = "A" if y_fp8 is not None else "F"
    if BUCKET_ON is not None and (form, rows) not in BUCKET_ON:
        return None
    return _key(form, rows)


def boundary(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps, sinkhorn_iters,
             y_fp8=None, y_gemm_sf=None, moe_input=None):
    """mhc_mega_boundary's arguments and result (swapped in for our model call by slots.forward_model)."""
    st = _g["state"]
    _join_all()                                 # incoming triplet joined; no stock / cake kernel beside side work
    if context.get() and st:
        try:
            key = _admit(st, x, residual, pre, post, comb, fn, scale, base, weight, sinkhorn_iters, y_fp8, y_gemm_sf,
                         moe_input)
        except Exception:  # noqa: BLE001  (an argument this engine shapes differently: the previous boundary)
            key = None
        if key is not None:
            form, rows, splits = key[0], key[1], key[2]
            args = (x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps, sinkhorn_iters)
            if torch.cuda.is_current_stream_capturing():
                if _g["agreed"].get((form, rows)) == key and _g["verdict"].get(key) is True:
                    try:
                        return _launch(st, form, splits, args, y_fp8, y_gemm_sf)
                    except Exception as e:  # noqa: BLE001  (stock below writes fresh outputs; nothing raises)
                        say("exm", f"EXM capture launch raised ({type(e).__name__}); stock for this boundary")
                        _join_all()
            else:
                verdict = _g["verdict"].get(key)
                try:
                    if verdict is True:
                        return _launch(st, form, splits, args, y_fp8, y_gemm_sf)
                    if verdict is None:
                        same, out = _check(st, form, splits, args, y_fp8, y_gemm_sf)
                        _g["verdict"][key] = same
                        say("exm", f"EXM first-use check {'OK' if same else 'DIFFERENT (stock for this key)'}: "
                                   f"form {form}, {rows} rows, {splits} splits, grid {key[3]}, smem {key[4]} "
                                   f"(whole call)")
                        if same:
                            return out
                except Exception as e:  # noqa: BLE001  (the previous boundary below rewrites every output)
                    _g["verdict"][key] = False
                    say("exm", f"EXM off for form {form} {rows} rows: {type(e).__name__} ({str(e)[:200]})")
                _join_all()
    tail = {"y_fp8": y_fp8, "y_gemm_sf": y_gemm_sf, "moe_input": moe_input}
    return _g["prev"](x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps, sinkhorn_iters,
                      **tail)


def triplet(hc, residual, stats_stream, precomputed=None):
    """_compute_triplet's signature: join every pending EXM fork, then the engine's function (no kernel here)."""
    _join_all()
    return _g["prev_triplet"](hc, residual, stats_stream, precomputed)


def swap():
    """Swap post_fusion_floor.mhc_mega_boundary (outermost, chaining to what v16_levers.routed installed) and
    deepseek_v4_mhc._compute_triplet for ours; returns whether it did. Never while EXM is off, and only in a forward
    begin() armed (RT5: forward_model calls begin before swap; every EXTEND / idle forward keeps the previous
    functions)."""
    if not _g["state"] or not context.get():
        return False
    MHC, floor = _g["eng"]["MHC"], _g["eng"]["floor"]
    if floor.mhc_mega_boundary is boundary or MHC._compute_triplet is triplet:
        return False
    _g["prev"], _g["prev_triplet"] = floor.mhc_mega_boundary, MHC._compute_triplet
    setattr(floor, "mhc_mega_boundary", boundary)
    setattr(MHC, "_compute_triplet", triplet)
    return True


def unswap(did):
    if did:
        MHC, floor = _g["eng"]["MHC"], _g["eng"]["floor"]
        setattr(MHC, "_compute_triplet", _g["prev_triplet"])
        setattr(floor, "mhc_mega_boundary", _g["prev"])
