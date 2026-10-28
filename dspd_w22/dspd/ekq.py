"""Lever EKQ v2: the Engram wkv projection in NVFP4 W4A4 for eager prefill calls, with the per-layer condition of EKQ-L.

wkv maps the [T, 6144] n-gram embedding rows to [T, 4 x 5120 keys | 5120 value] (one key per hyper-connection copy, then
the shared value); the node output is x + gate * value, the keys only steer engram_gate's sigmoid. The per-layer
condition (measured on real requests in the audit engine, 10-28): layer 14's Engram update is ~4-7 % of the node output,
so its whole projection runs NVFP4 (0.57-0.9 % median node rel-L2, 92-100 % of rows <= 2 %); layer 1's is ~51 %, so only
its key half runs NVFP4 (0.03 %) and its value half stays the engine's exact MXFP8 linear on the engine's own rows.

v2 (vs dspd-w19): the GPU saving of v1 (~0.6 ms per Engram call at 16K rows) was eaten on the host: FlashInfer's
nvfp4_quantize / mm_fp4 Python layers (requirement checks, autotuner lookup, tracing) on every call, and w19's per-call
.item() share check (a stream sync per call). v2 calls the compiled pieces directly, as MXT does for MXFP8: the JIT
per-token NVFP4 quantize op and the CuTe-DSL FP4 runner built once, every tactic it can serve compiled at warm-up, so
no call compiles or tunes; nothing reads a device value back. Decode, graphs, small calls and any unexpected layout run the
exact projection. Precision class: tolerance, graded by the eager audit's Engram calls (>= MIN_ROWS rows)."""
import torch
from flashinfer.gemm import gemm_base as GB
from flashinfer.quantization import fp4_quantization as Q
from flashinfer.quantization.fp4_quantization import SfLayout, nvfp4_quantize

from dspd import mxt
from dspd.common import pins_match, say
from sglang.srt.layers import engram as E
from sglang.srt.layers.quantization import fp8_utils as FU

ENABLED = True
# 1024: the eager audit's Engram calls are 1-2K rows, and a lossy path must run where the audit grades it
MIN_ROWS = 1024
FULL_HASH_LAYERS = frozenset({1})          # Engram.layer_hash_index (0 = layer 1, 1 = layer 14) taking the full projection
# The FP4 GEMM runs the runner's own untuned tactic for each row count (rig 10-28, real layer-1 weights: within 1 % of the
# best of 33 tactics at 16K rows for both shapes, and at 4K for the whole projection). That choice varies with the row
# count and every tactic is a separate CuTe-DSL compile (~0.5 s), so warm() compiles each tactic the selector picks for
# MIN_ROWS..MAX_ROWS rows up front.
MAX_ROWS = 131072                          # the prefill engine's chunked_prefill_size: no eager call brings more rows
WARM_ROWS = 32768                          # warm-up rows cap (a 32K-row synthetic call: ~1 GB transient)
HC, DIM, K = 4, 5120, 6144
NK, N = HC * DIM, HC * DIM + DIM
INV_BASE = 1.0 / (448.0 * 6.0)             # per-token NVFP4: amax / (448 * 6) per row
PINS = {"flashinfer_mxfp8_blockscaled_linear": "44d063202cb3947e"}
_g = {"pins": None, "w": {}, "k": None, "compiled": False}


def _pins_ok():
    if _g["pins"] is None:
        _g["pins"] = pins_match("ekq", PINS, {"flashinfer_mxfp8_blockscaled_linear":
                                              FU.flashinfer_mxfp8_blockscaled_linear})
    return _g["pins"]


def _kernels(device):
    """The JIT per-token NVFP4 quantize op and the CuTe-DSL FP4 GEMM runner, resolved once."""
    k = _g["k"]
    if k is None:
        major, minor = torch.cuda.get_device_capability(device)
        k = _g["k"] = {"quant": Q.get_fp4_quantization_module(f"{major * 10 + minor}").nvfp4_quant_and_per_token_scale_sm100,
                       "runner": GB._cute_dsl_gemm_fp4_runner(major, minor, True, torch.bfloat16, True),
                       "layout": SfLayout.layout_128x4.value, "code": Q.nvfp4_4over6_code(None), "minor": minor}
    return k


def _picks(device, n_out):
    """Every compiled kernel an eager call of MIN_ROWS..MAX_ROWS rows can need: the runner's untuned tactic and the L2
    policy it compiles in (both depend on the row count; the runner's own formulas, evaluated here so warm() can compile
    them), each mapped to the fewest rows that are served it."""
    minor = _kernels(device)["minor"]
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    picks = {}
    for m in range(MIN_ROWS, MAX_ROWS + 1, 16):
        t = (GB._select_sm100_mm_fp4_splitk_tactic(m, n_out, K, sms, True, minor)
             or GB._select_sm100_mm_fp4_cute_dsl_tactic(m, n_out, K, sms, 16))
        picks.setdefault((t, GB.mm_fp4_l2_policy(m, t[0], t[2], t[4])), m)
    return picks


def _unswizzle(sf, rows, kb):
    """FlashInfer's 128x4-swizzled 1D UE8M0 buffer -> [rows, kb] bytes."""
    rp, kp = -(-rows // 128) * 128, -(-kb // 4) * 4
    return sf.view(rp // 128, kp // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rp, kp)[:rows, :kb]


def _build(engram, full=None):
    """NVFP4 copy of the key rows (or of all rows for a full layer) + the value rows' views, or None when the projection
    is not the layout EKQ was written for."""
    if full is None:
        full = getattr(engram, "layer_hash_index", None) in FULL_HASH_LAYERS
    lin = engram.wkv
    w = getattr(lin, "weight", None)
    sw = getattr(lin, "weight_scale_inv_swizzled", None)
    kb = K // 32
    tile = (-(-kb // 4) * 4) // 4 * 512                     # bytes of one 128-row tile of the swizzled scales
    if (w is None or sw is None or w.dtype != torch.float8_e4m3fn or tuple(w.shape) != (N, K) or not w.is_contiguous()
            or sw.dim() != 1 or sw.numel() != (N // 128) * tile or not sw.is_contiguous()):
        say("ekq", "EKQ off: unexpected wkv layout")
        return None
    sf_rows = _unswizzle(sw.view(torch.uint8), N, kb)
    nq = N if full else NK
    wk = (w[:nq].float().view(nq, kb, 32) * torch.exp2(sf_rows[:nq].float() - 127.0).unsqueeze(-1)).view(nq, K)
    gsf = (448.0 * 6.0) / wk.abs().max().clamp_min(1e-30)
    wk4, wk4sf = nvfp4_quantize(wk.to(torch.bfloat16), gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    del wk, sf_rows
    torch.cuda.empty_cache()
    return {"wk4t": wk4.t(), "wk4sf": wk4sf, "inv_gsf": (1.0 / gsf).float(), "n": nq, "wv": w[NK:],
            "sv": sw[(NK // 128) * tile:], "full": full}


def _key(st, flat, tactic=-1):
    """[T, n] = flat @ (NVFP4 rows)^T with per-token NVFP4 activations: two launches (quantize, GEMM) + the alpha scale.
    tactic -1 = the runner's untuned choice for this row count."""
    k = _kernels(flat.device)
    a4, a4sf, pt = k["quant"](flat, INV_BASE, None, k["layout"], k["code"])
    out = torch.empty((flat.shape[0], st["n"]), dtype=torch.bfloat16, device=flat.device)
    ws = GB._get_cache_buf("mm_fp4_workspace", GB.DEFAULT_WORKSPACE_SIZE, flat.device)
    k["runner"](inputs=[a4, st["wk4t"], a4sf, st["wk4sf"], pt.view(-1) * st["inv_gsf"], torch.bfloat16, out, 16, True,
                        ws], tactic=tactic)
    return out


def warm(engram):
    """Build EKQ's private weights and compile its kernels at the module's first eager call, which the engine's
    conditioning requests make before any scored window: the quantize op and every FP4 tactic an eager call can be
    served (_picks, both shapes) compile once per process, on synthetic rows here instead of inside a large prefill call.
    Decode servers skip it (their calls never reach MIN_ROWS rows)."""
    if not ENABLED or id(engram) in _g["w"] or torch.cuda.is_current_stream_capturing() or not _pins_ok():
        return
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "decode":
        return
    st = _g["w"][id(engram)] = _build(engram) or False
    if st is False or _g["compiled"]:
        return
    dev = st["wv"].device
    picks = {**_picks(dev, N), **_picks(dev, NK)}
    for (tactic, l2), m in sorted(picks.items(), key=lambda kv: kv[1]):
        if m > WARM_ROWS:
            say("ekq", f"EKQ v2 tactic {tactic} / L2 {l2} first served at {m} rows: compiled at first use")
            continue
        try:
            _key(st, torch.randn(m, K, device=dev, dtype=torch.bfloat16), tactic)
        except Exception as e:  # noqa: BLE001
            say("ekq", f"EKQ v2 tactic {tactic} / L2 {l2} not precompiled ({type(e).__name__}): compiled at first use")
    torch.cuda.synchronize(dev)
    _g["compiled"] = True
    say("ekq", f"EKQ v2 kernels compiled at the first eager call ({len(picks)} FP4 kernels for {MIN_ROWS}-{MAX_ROWS} rows: "
               f"{sorted((m, l2) for (_, l2), m in picks.items())})")


def out(engram, x, flat, image_ids):
    """engram_gate(x, EKQ's wkv projection) for a call EKQ takes, or None (the exact projection)."""
    if (not ENABLED or flat.dim() != 2 or flat.shape[0] < MIN_ROWS or flat.shape[1] != K or flat.dtype != torch.bfloat16
            or not flat.is_contiguous() or torch.cuda.is_current_stream_capturing() or not _pins_ok()):
        return None
    st = _g["w"].get(id(engram))
    if st is None:
        st = _g["w"][id(engram)] = _build(engram) or False
    if st is False:
        return None
    if st["full"]:
        say("ekq", f"EKQ-L Engram hash layer {getattr(engram, 'layer_hash_index', '?')}: full NVFP4 projection on")
        kv = _key(st, flat)
    else:
        say("ekq", "EKQ NVFP4 key projection on")
        key = _key(st, flat)
        with mxt.scope():                                   # the value: the engine's MXFP8 linear, MXT-checked tactic
            value = FU.flashinfer_mxfp8_blockscaled_linear(flat, st["wv"], st["sv"], backend="cute-dsl", pin_tactic=True)
        kv = torch.cat([key, value], dim=1)
    return E.engram_gate(x, kv, engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value, image_ids,
                         engram.image_token_id)

