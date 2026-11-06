"""king-v7 XG-F: the decode MoE front launched with programmatic dependent launch (IDEAS_dec_rest DR8, second half).

The king launches its fused MoE front (`cuda_moe_front.py`, router + top-k + x-quant in one kernel) with
`use_pdl=False` (`moe_route._front`), although the kernel's first instruction is `griddepcontrol.wait`
(`PDLWaitPrimary<true>`, before any load or store). Its predecessor on the MoE stream is the cuBLAS gate GEMM
(nvjet), which triggers its dependents early: on the king's step dump, every PDL-launched kernel after an nvjet kernel
starts 0.4-1.4 us before the nvjet ends, while the non-PDL front starts 0.4-0.8 us (median 0.6) after it
(`critpath/verify/k1_step_dump.txt`, 50 gate -> front pairs). The MoE stream's chain (gate, front, MegaMoE) is joined
by the main stream before the next layer, so the front's launch gap sits on the step's critical path.

Two switches (each one lever):
- XGF_ON: launch the king's front kernel with the programmatic-serialization attribute. Scheduling only: the kernel
  waits for the gate's full completion before its first load or store, so its bytes cannot depend on the overlap.
- XGF_EQ_ON (rider, needs XGF_ON): an anchored copy of the king's v1 front kernel in its own namespace and JIT module
  (`dec_moe_front_eq`) whose x-quant warps run before `griddepcontrol.wait`; the router warps and the pad CTAs keep the
  wait. Served only inside the decode graph's MoE-stream fork (`moe_route.forward_mega_moe`, `fork_mega_moe`), where the
  stream fork is a full graph edge from the main stream's last node (the FFN-entry boundary that writes x) to the gate:
  x is complete and visible before the gate starts, and the front cannot launch before the gate has started. The
  quant warps read only x and write only buf.x / buf.x_sf / the shared-expert scales, which no kernel between that
  edge and this layer's MegaMoE touches (the previous layer's MegaMoE completed before the edge's source ran: the main
  stream joined it). Rows >= moe_front.V2_MIN_ROWS (the v2 kernel) and the LEAN path use the king's text. "Inside the
  fork" is `fork_mega_moe`, which needs get_is_capture_mode(): that is also True during sglang's EAGER warm-up runs
  of each graph size, so EQ runs eagerly there too (safe: eager stream order is a full dependency).
  SHIPS OFF (king-v7 fixer, review F3): the race-freedom above is argued, not measured, and the eager prepare proof
  cannot see a race in the captured fork; a race would silently change decode MoE inputs. Gate before turning it on:
  greedy fixed-prompt token ids with XGF_EQ_ON True against False on the same box must be identical.

Prepare (`prepare`, called from decode_gemm.prepare on the decode server): the served launch form is compared byte for
byte against the king's launch (no PDL, king text) at PROOF_ROWS, on the served template (experts, top-k, hidden,
renorm, scale, shared-SF of layer 0), with and without input_ids / num_token_non_padded, outputs pre-filled with the
same random bytes and compared over their whole storage (buf.x, buf.x_sf, top-k ids and weights, shared-expert scales),
at every shared-expert block_m the decode tiles serve (moe_route._GRAPH_TILES: 16 / 64 under TILES-25, 16 / 32 / 64
under TILES_KING; king-v7 fixer: the proof used 64 alone).
Each launch follows a dummy kernel on the stream so the PDL attribute has a primary. Any failure turns the lever (or
only the EQ rider) off with one boot line; the king's launch serves. Never raises.
Capture: no allocation, no host sync; a launch error turns XG-F off for later calls and the king's launch serves.
"""
from __future__ import annotations

import hashlib
import os
import sys

import torch

# The lead's switches. True: the lever serves where proven. False: the king's launch (use_pdl=False, king text).
XGF_ON = True
XGF_EQ_ON = False   # king-v7 fixer (review F3): off until the greedy token-id A/B (EQ True vs False) is identical

EQ_NAME = "moe/moe_front_eq.cuh"
PROOF_ROWS = (1, 2, 5, 6, 7, 12, 24, 25, 48, 64, 96, 127, 128, 144, 148, 149, 150, 186, 192, 216, 240, 264, 288, 300)
PADDED = 512
SHARED_BLOCK_M = 64
SEED = 20261107

_STATE = {"pdl": False, "eq": False, "why": "not prepared", "eq_why": "not prepared"}
_WARNED: list = []
_PATH: dict = {}


def line() -> str:
    if not _STATE["pdl"]:
        return f"bundle-v9 king-v7 XG-F: off ({_STATE['why']})"
    eq = "EQ ON" if _STATE["eq"] else f"EQ off ({_STATE['eq_why']})"
    return f"bundle-v9 king-v7 XG-F: ON ({_STATE['why']}); {eq}"


def say() -> None:
    print(line(), file=sys.stderr, flush=True)


def launch_form(fork: bool):
    """(use_pdl, early_quant) for one served front call."""
    if not _STATE["pdl"]:
        return False, False
    return True, bool(_STATE["eq"] and fork)


def runtime_off(error: Exception) -> None:
    _STATE["pdl"] = False
    _STATE["eq"] = False
    _STATE["why"] = (f"disabled at run time, the king's launch from here on: {type(error).__name__}: "
                     f"{str(error)[:300]}")
    if not _WARNED:
        _WARNED.append(True)
        say()


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"XG-F EQ anchor '{what}' matches {count} times in the king's front text (want 1)")
    return text.replace(old, new)


def eq_source() -> str:
    """The king's front text with the v1 kernel's x-quant warps ahead of griddepcontrol.wait, in its own namespace."""
    from bundle_v9.dec import cuda_moe_front

    text = cuda_moe_front.SOURCE
    text = _replace_once(text, "namespace sglang::dec_front {", "namespace sglang::dec_front_eq {", "namespace")
    text = _replace_once(
        text,
        "  const uint32_t warp = threadIdx.x / 32;\n"
        "  const uint32_t lane = threadIdx.x % 32;\n"
        "\n"
        "  device::PDLWaitPrimary<true>();\n"
        "\n"
        "  if (blockIdx.x >= p.rows) {\n",
        "  const uint32_t warp = threadIdx.x / 32;\n"
        "  const uint32_t lane = threadIdx.x % 32;\n"
        "\n"
        "  if (blockIdx.x >= p.rows) {\n"
        "    device::PDLWaitPrimary<true>();  // king-v7 XGF-EQ: the pad CTAs keep the wait\n",
        "v1 entry wait")
    text = _replace_once(
        text,
        "    device::PDLTriggerSecondary<true>();\n"
        "    return;\n"
        "  }\n"
        "\n"
        "  int64_t live_rows = p.rows;\n",
        "    device::PDLTriggerSecondary<true>();\n"
        "    return;\n"
        "  }\n"
        "\n"
        "  device::PDLWaitPrimary<true>();  // king-v7 XGF-EQ: the router warps wait before the logits\n"
        "  int64_t live_rows = p.rows;\n",
        "v1 router wait")
    return text


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


def eq_header():
    """(path, digest) of the EQ header in the JIT cache root, written once per content."""
    got = _PATH.get("eq")
    if got is None:
        from sglang.kernels.jit.utils.compile.cache import cache_root

        data = eq_source().encode()
        digest = hashlib.sha256(data).hexdigest()[:16]
        directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-moe-front-eq", digest)
        os.makedirs(os.path.join(directory, os.path.dirname(EQ_NAME)), exist_ok=True)
        path = os.path.join(directory, EQ_NAME)
        if not os.path.isfile(path):
            _write(path, data)
        got = _PATH["eq"] = (path, digest)
    return got


def _facts(moe):
    """The served template flags of the decode front (as moe_route._run_mega_routed and mega_routed_front pass them)."""
    config = moe.topk.topk_config
    renorm = bool(config.renormalize and config.top_k > 1)
    scale = 1.0 if moe.routed_scaling_factor_fused_in_topk else float(moe.routed_scaling_factor)
    shared = moe.mega_shared_l1_weights is not None
    alt = getattr(moe.config, "image_token_id", 0)
    return {"renorm": renorm, "scale": scale, "shared": shared, "alt": int(alt) if alt is not None else 0,
            "top_k": int(config.top_k)}


def _inputs(rows, facts, device, gen, with_ids, with_live, block_m=SHARED_BLOCK_M):
    from bundle_v9.dec import moe_front

    experts, hidden, top_k = moe_front.EXPERTS, moe_front.HIDDEN, facts["top_k"]
    logits = torch.randn((rows, experts), device=device, generator=gen) * 3
    logits[0, :8] = torch.tensor([30.0, -30.0, 0.0, -0.0, 1e-30, 25.0, -1e-30, 19.5], device=device)
    bias = torch.randn(experts, device=device, generator=gen) * 0.1
    bias_alt = torch.randn(experts, device=device, generator=gen) * 0.1
    x = (torch.randn((rows, hidden), device=device, generator=gen) * 3).bfloat16()
    x[rows // 2].zero_()
    x[rows - 1, :64] = 448.0
    ids = None
    if with_ids:
        ids = torch.randint(0, 1000, (rows,), device=device, generator=gen, dtype=torch.int64)
        ids[::4] = facts["alt"]
    live = torch.tensor(max(rows - rows // 7, 1), device=device, dtype=torch.int32) if with_live else None
    padded = max(PADDED, rows)
    shapes = {"buf_x": ((padded, hidden), torch.uint8), "buf_x_sf": ((padded, hidden // 128), torch.int32),
              "idx": ((padded, top_k), torch.int64), "w": ((padded, top_k), torch.int32)}
    base = {name: torch.randint(0, 255, shape, device=device, generator=gen).to(dtype) for name, (shape, dtype)
            in shapes.items()}
    sf_rows = -(-rows // block_m) * (-(-block_m // 128) * 128)
    base["shared"] = torch.randint(0, 1 << 30, (sf_rows * (hidden // 128),), device=device, generator=gen,
                                   dtype=torch.int32) if facts["shared"] else None
    return (logits, bias, bias_alt, x, ids, live), base, sf_rows


def _renorm_eps() -> float:
    from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

    return _RENORMALIZE_SUM_EPSILON


def _run(args, base, sf_rows, facts, use_pdl, early_quant, scratch, block_m=SHARED_BLOCK_M):
    from bundle_v9.dec import moe_front

    logits, bias, bias_alt, x, ids, live = args
    out = {name: (t.clone() if t is not None else None) for name, t in base.items()}
    shared = None
    if out["shared"] is not None:
        shared = torch.as_strided(out["shared"], (sf_rows, moe_front.HIDDEN // 128), (1, sf_rows))
    scratch.add_(1.0)  # a primary kernel on the stream ahead of the (PDL) front launch
    moe_front.moe_front(
        logits, bias, x, out["buf_x"].view(torch.float8_e4m3fn), out["buf_x_sf"], out["idx"],
        out["w"].view(torch.float32),
        bias_alt=bias_alt if ids is not None else None, input_ids=ids, image_token_id=facts["alt"],
        num_token_non_padded=live, renormalize=facts["renorm"], renormalize_epsilon=_renorm_eps(),
        routed_scaling_factor=facts["scale"], shared_x_sf=shared,
        shared_block_m=block_m if shared is not None else 0, use_pdl=use_pdl, early_quant=early_quant)
    return out


def _block_ms() -> tuple:
    """king-v7 fixer: the shared-expert block_m values the decode front is served with. moe_route._run_mega_routed
    passes the tiles' block_m (the second field of moe_route._GRAPH_TILES: 16 / 64 under TILES-25, 16 / 32 / 64 under
    TILES_KING), so the proof covers each of them; 64 (SHARED_BLOCK_M) is always included."""
    from bundle_v9.dec import moe_route

    values = {SHARED_BLOCK_M}
    for _, tiles in moe_route._GRAPH_TILES:
        if tiles:
            values.add(int(str(tiles).split(",")[1]))
    return tuple(sorted(values))


def _prove(facts, device, early_quant, block_ms=(SHARED_BLOCK_M,)) -> int:
    gen = torch.Generator(device=device).manual_seed(SEED + (1 if early_quant else 0))
    scratch = torch.zeros(1 << 16, device=device)
    cases = 0
    for block_m in block_ms:
        for i, rows in enumerate(PROOF_ROWS):
            for with_ids, with_live in ((i % 2 == 0, i % 3 != 1), (i % 2 == 1, i % 3 == 1)):
                args, base, sf_rows = _inputs(rows, facts, device, gen, with_ids, with_live, block_m)
                want = _run(args, base, sf_rows, facts, False, False, scratch, block_m)
                got = _run(args, base, sf_rows, facts, True, early_quant, scratch, block_m)
                for name in want:
                    if want[name] is None:
                        continue
                    if not torch.equal(want[name].view(-1).view(torch.uint8), got[name].view(-1).view(torch.uint8)):
                        raise RuntimeError(f"{name} at {rows} rows (shared block_m {block_m}, input_ids {with_ids}, "
                                           f"live {with_live}) differs from the king's launch")
                cases += 1
    torch.cuda.synchronize(device)
    return cases


def prepare(module) -> None:
    """Decode-server prepare; never raises."""
    _STATE.update(pdl=False, eq=False, why="not prepared", eq_why="not prepared")
    if not XGF_ON:
        _STATE["why"] = "switched off by xgf.XGF_ON; the king's non-PDL front launch"
        say()
        return
    try:
        from bundle_v9.dec import moe_front

        moe = module.layers[0].mlp
        facts = _facts(moe)
        if facts["top_k"] != moe_front.TOP_K:
            raise RuntimeError(f"top-k {facts['top_k']} is not the front's {moe_front.TOP_K}")
        device = next(module.parameters()).device
        block_ms = _block_ms()
        cases = _prove(facts, device, False, block_ms)
    except Exception as error:
        _STATE["why"] = f"FAIL, the king's non-PDL launch: {type(error).__name__}: {str(error)[:400]}"
        say()
        return
    _STATE["pdl"] = True
    _STATE["why"] = (f"front launched with PDL after the gate; bytes equal to the king's launch in {cases} cases at "
                     f"{len(PROOF_ROWS)} row counts x shared block_m {list(block_ms)} (renorm {facts['renorm']}, "
                     f"scale {facts['scale']}, shared-SF {facts['shared']})")
    if not XGF_EQ_ON:
        _STATE["eq_why"] = "switched off by xgf.XGF_EQ_ON"
    else:
        try:
            eq_cases = _prove(facts, device, True, block_ms)
            _STATE["eq"] = True
            _STATE["eq_why"] = f"x-quant warps ahead of the wait inside the graph's MoE-stream fork; {eq_cases} cases"
        except Exception as error:
            _STATE["eq_why"] = f"FAIL, the king's text: {type(error).__name__}: {str(error)[:400]}"
    say()
