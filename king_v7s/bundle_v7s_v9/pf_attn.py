"""king-v7 PATTN-RH (RANKED #40, PRE P2 as corrected by VERIFY_prefill, the PATTN-RH arm): the king's PATTN core
(prefill_attn.py, cuda_prefill_attn.py) also serves the eager sparse-prefill MegaAttention calls of >= ROWS_MIN rows
made while a REHOME step is in flight: the owner's kept rows, the helpers' moved pieces and the bystanders' calls of
layers 0..19. The king stands PATTN down for the whole planned step (prefill_attn.serves: rehome_block.in_flight()),
so on REHOME forwards (about two thirds of the sampled forwards) every layer 0..19 call runs the legacy kernel.

Why the bytes are stock's. PATTN is the king's JIT-built reordered copy of the image's legacy kernel (waits, loads
and barrier arrivals only). Its equality with the legacy kernel is the king's claim, not a construction (a different
cubin); it is what the king serves at >= 8,192 rows on the audited role in its crowned PASS. PATTN launches one
cluster of CTAs per query row (grid = rows x cluster), so a row's out_fp8 / out_sf are a function of its own q,
position, top-k length and indices, the kv rows those select, and the layer's sink, scale and cos/sin table. A call
made in flight differs from a plain one only in which tensors hold those inputs (the owner's sparse-prefill workspace
and combine slices, a helper's transport kv_local / index / length / position buffers). prefill_attn.serves checks
the same shapes, dtypes and strides in flight as outside it, and the REHOME prepare probe (rehome_probe) proves the
legacy kernel split-invariant, so PATTN(piece) = legacy(piece) = legacy(whole call), which is what the REHOME owner
reassembles. The proof below re-checks PATTN = legacy on this rank, for every REHOME layer, before any of it serves.

Proof (prepare; per rank; no collective). For each REHOME layer (0..19), with that layer's own attn_sink, softmax
scale and cos/sin table, PATTN and the legacy kernel run the same synthetic call of PROOF_ROWS rows (>= ROWS_MIN and
not a multiple of 4, so out_sf's padded allocation is exercised) at the layer's index width (128 for the c0 layers
0-1, 640 for the c2 layers): the engine's combine layout ([top-k with -1 holes | SWA window | -1 pad], length =
top-k + window), zero-length padding rows, lengths that are and are not multiples of 64 (the kernel's two per-row
bodies), and kv as an offset view into a larger buffer. out_fp8 bytes and out_sf words must be equal at every layer;
otherwise PATTN-RH is off and the king's stand-down holds.

Serving. Rank-local, as PATTN itself (attention has no collective): a rank whose proof failed keeps the king's
stand-down and its bytes are the same either way. Nothing extra runs in flight: no host sync, no compare, no
allocation beyond PATTN's own outputs (stock allocates the same four). A raise from PATTN inside a flight turns
PATTN-RH off for the rest of the process (one line) and the legacy kernel serves that call; prefill_attn.CHECK's
compare (a host sync, which must not meet REHOME's transport) is never run in flight."""

from __future__ import annotations

import sys
import time

import torch

PATTN_RH = True
ROLES = ("prefill", "null")
LAYERS = 20                 # REHOME re-homes layers 0..19 (rehome_attn.check_layer: ratio 0 for 0-1, 2 for 2-19)
PROOF_ROWS = 8195           # >= prefill_attn.ROWS_MIN; 8195 % 4 != 0 exercises out_sf's padded column-major layout
PROOF_KV = (3000, 1500)     # compressed / SWA rows of the synthetic workspace (the c0 layers read the SWA part only)
PROOF_OFFSET = 3            # kv starts this many rows into its buffer, as a workspace / transport view can

_STATE = {"on": False, "why": "not prepared"}


def on() -> bool:
    return _STATE["on"]


def mark(on_: bool, why: str) -> None:
    _STATE["on"] = bool(on_)
    _STATE["why"] = why


def line() -> str:
    state = "ON" if _STATE["on"] else "off"
    return f"bundle-v9 king-v7 PATTN-RH: {state} ({_STATE['why']}) gpu{torch.cuda.current_device()}"


def fail(error: BaseException, rows: int) -> None:
    """A PATTN raise inside a REHOME flight: off for the rest of the process, one line; the caller serves stock."""
    if _STATE["on"]:
        mark(False, f"FAIL, the king's in-flight stand-down holds from now: a {rows}-row in-flight call raised "
                    f"{type(error).__name__}: {str(error)[:200]}")
        print(line(), file=sys.stderr, flush=True)


def _layout(rows: int, width: int, max_pos: int, n_c: int, n_w: int) -> tuple:
    """(indices [rows, 1, width], lengths [rows], positions [rows]) int32 on the host, in the engine's combine layout:
    top-k min((pos + 1) // 2, 512) entries into the compressed rows with -1 holes kept (c2 only), then the SWA window
    min(pos + 1, 128) entries, then -1 to the width; every 997th row is a zero-length padding row."""
    r = torch.arange(rows, dtype=torch.int64)
    span = max(max_pos - 2048, 1)
    pos = torch.where(r < rows // 2, r % 2048, 2048 + (r * 7919) % span).clamp(max=max_pos - 1)
    pad = (r % 997) == 5
    win = torch.where(pad, torch.zeros_like(pos), (pos + 1).clamp(max=128))
    if width == 640:
        topk = torch.where(pad, torch.zeros_like(pos), ((pos + 1) // 2).clamp(max=512))
        swa0 = n_c
    else:
        topk = torch.zeros_like(pos)
        swa0 = 0
    lens = topk + win
    col = torch.arange(width, dtype=torch.int64).unsqueeze(0)
    top = (r.unsqueeze(1) * 131 + col * 17) % n_c
    top = torch.where(col % 5 == 3, torch.full_like(top, -1), top)
    swa = swa0 + (pos.unsqueeze(1) - win.unsqueeze(1) + 1 + col - topk.unsqueeze(1)) % n_w
    idx = torch.where(col < topk.unsqueeze(1), top,
                      torch.where(col < lens.unsqueeze(1), swa, torch.full_like(swa, -1)))
    return idx.to(torch.int32).unsqueeze(1).contiguous(), lens.to(torch.int32), pos.to(torch.int32)


def _kinds(lens: torch.Tensor) -> tuple:
    zero = int((lens == 0).sum())
    partial = int((lens % 64 != 0).sum())
    return zero, partial, int(lens.numel()) - zero - partial


def _prove(model, stock, run) -> str:
    """Every REHOME layer's PATTN against the legacy kernel on one synthetic call each; raises on any difference."""
    from bundle_v9 import prefill_attn

    if PROOF_ROWS < prefill_attn.ROWS_MIN:
        raise RuntimeError(f"PROOF_ROWS {PROOF_ROWS} is below prefill_attn.ROWS_MIN {prefill_attn.ROWS_MIN}")
    t0 = time.perf_counter()
    layers = list(model.layers)[:LAYERS]
    device = layers[0].self_attn.wq_b.weight.device
    n_c, n_w = PROOF_KV
    gen = torch.Generator(device=device)
    gen.manual_seed(7107)
    q = torch.randn((PROOF_ROWS, prefill_attn.H_Q, prefill_attn.D_QK), dtype=torch.bfloat16, device=device,
                    generator=gen)
    buf = torch.randn((PROOF_OFFSET + n_c + n_w + 2, 1, prefill_attn.D_QK), dtype=torch.bfloat16, device=device,
                      generator=gen)
    inputs, kinds, widths = {}, {}, []
    for index, layer in enumerate(layers):
        attn = layer.self_attn
        ratio = int(attn.compress_ratio)
        if ratio not in (0, 2):
            raise RuntimeError(f"layer {index} has compress ratio {ratio}, not REHOME's 0 / 2")
        width = 128 if ratio == 0 else 640
        cos = attn.mega_cos_sin_cache
        heads = 64 if attn.n_local_heads <= 64 else 128
        sink = attn._local_attn_sink(heads)
        if heads != prefill_attn.H_Q or cos is None or cos.dtype != torch.float32 or cos.dim() != 2 \
                or cos.shape[1] != 64 or not cos.is_contiguous() or sink.dtype != torch.float32 \
                or tuple(sink.shape) != (prefill_attn.H_Q,) or not sink.is_contiguous():
            raise RuntimeError(f"layer {index}'s heads / sink / cos-sin table are not PATTN's served form")
        key = (width, int(cos.shape[0]))
        if key not in inputs:
            idx, lens, pos = _layout(PROOF_ROWS, width, int(cos.shape[0]), n_c, n_w)
            kv = buf[PROOF_OFFSET:PROOF_OFFSET + (n_c + n_w if width == 640 else n_w)]
            inputs[key] = (kv, idx.to(device), lens.to(device), pos.to(device))
            kinds[width] = _kinds(lens)
        kv, idx, lens, pos = inputs[key]
        ref = stock(q, kv, idx, pos, cos, attn.softmax_scale, attn_sink=sink, topk_length=lens, use_legacy=True)
        out = run(q, kv, idx, pos, cos, attn.softmax_scale, sink, lens, 0.0)
        if out[0].shape != ref[0].shape or out[0].dtype != ref[0].dtype or out[1].shape != ref[1].shape \
                or out[1].dtype != ref[1].dtype:
            raise RuntimeError(f"layer {index}: PATTN output {tuple(out[0].shape)} / {tuple(out[1].shape)} is not "
                               f"the legacy kernel's {tuple(ref[0].shape)} / {tuple(ref[1].shape)}")
        d8 = int((out[0].view(torch.uint8) != ref[0].view(torch.uint8)).sum())
        dsf = int((out[1] != ref[1]).sum())
        out = ref = None
        if d8 or dsf:
            raise RuntimeError(f"layer {index} (width {width}): {d8} out_fp8 bytes and {dsf} out_sf words differ "
                               f"from the legacy kernel at {PROOF_ROWS} rows")
        widths.append(width)
    q = buf = inputs = None
    kind = "; ".join(f"width {w}: rows of length 0 / not a multiple of 64 / a nonzero multiple {a}/{b}/{c}"
                     for w, (a, b, c) in sorted(kinds.items()))
    return (f"the king's PATTN also serves eager sparse-prefill calls of >= {prefill_attn.ROWS_MIN} rows during "
            f"REHOME flights (owner, helper and bystander calls of layers 0..{len(layers) - 1}); proven bitwise "
            f"(out_fp8 + out_sf) against the legacy kernel at {PROOF_ROWS} rows on each of {len(widths)} layers' own "
            f"sink / scale / cos-sin table ({widths.count(128)} at width 128, {widths.count(640)} at width 640; "
            f"{kind}; kv an offset view) in {time.perf_counter() - t0:.2f} s")


def prepare(model, role: str, stock, run) -> None:
    """Decide PATTN-RH for this rank; never raises (the king's stand-down holds on any failure)."""
    if not PATTN_RH:
        mark(False, "switched off by pf_attn.PATTN_RH; the king's in-flight stand-down holds")
    elif role not in ROLES:
        mark(False, f"role {role} is not in pf_attn.ROLES {ROLES}; the king's in-flight stand-down holds")
    else:
        try:
            mark(True, _prove(model, stock, run))
        except Exception as error:  # noqa: BLE001  (off; prepare goes on)
            mark(False, f"FAIL, the king's in-flight stand-down holds: {type(error).__name__}: {str(error)[:300]}")
    print(line(), file=sys.stderr, flush=True)
