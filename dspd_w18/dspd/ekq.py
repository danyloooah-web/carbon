"""Lever EKQ: the Engram wkv projection's key half in NVFP4 W4A4 for large eager prefill calls (layers 1 and 14).

wkv maps the [T, 6144] n-gram embedding rows to [T, 4 x 5120 keys | 5120 value] (one key per hyper-connection copy, then
the shared value). The keys are 80 % of the GEMM's FLOPs but only steer the sigmoid gate of engram_gate (a normalized
5120-wide dot with the hidden copy); the node output is x + gate * value. EKQ computes the keys with FlashInfer's
CuTe-DSL NVFP4 GEMM (per-token NVFP4 activations, a private NVFP4 copy of the engine's own key rows, built once from the
FP8 weight and its scales) and the value with the engine's own MXFP8 linear on the engine's own value rows and scale
tiles, so the value columns are the stock bytes. Measured on the B300 rig (real layer-1 weights): the split costs
1326 us vs 2199 us for the stock projection at 16K rows (8K: 618 vs 842; about break-even at 4K, slower at 1K);
the key rows carry ~14 % rel-L2 vs stock, which reaches the node output only through the gate. Precision class:
tolerance, graded by the eager audit's Engram calls of >= MIN_ROWS rows. Decode, graphs, small calls and any unexpected
layout run the stock projection."""
import torch
from flashinfer.gemm import mm_fp4
from flashinfer.quantization.fp4_quantization import SfLayout, nvfp4_quantize

from dspd import mxt
from dspd.common import pins_match, say
from sglang.srt.layers.quantization import fp8_utils as FU

ENABLED = True
# 1024, not the 8K break-even: the eager audit's Engram calls are 1-2K rows, and a lossy path must run where the audit
# grades it (at 1-4K rows EKQ costs <= ~0.08 ms more than stock per call; from 8K it saves 0.2-1.7 ms)
MIN_ROWS = 1024
# EKQ-L, the per-layer condition: how much of the node output x + gate * value the Engram update is decides whether the
# value half goes NVFP4 too. Measured on real requests in the audit engine (10-28, every Engram call 256-16384 rows):
# layer 1's update is ~51 % of the output (an NVFP4 value: 6.4 % median node rel-L2, 0 % of rows <= 2 %), layer 14's
# ~4-7 % (NVFP4 value: 0.57-0.9 % median, 92-100 % of rows <= 2 %; the key-only path: 0.03 %). So layer 14 runs the
# whole projection in NVFP4 (one GEMM, no value GEMM, no concat) and layer 1 only the key half.
FULL_HASH_LAYERS = frozenset({1})          # Engram.layer_hash_index (0 = layer 1, 1 = layer 14) taking the full projection
SHARE_MAX = 0.15                            # first-call check: a full layer whose median update share is above this falls
CHECK_ROWS = 256                            # back to the key half for good (measured on this many rows)
HC, DIM, K = 4, 5120, 6144
NK, N = HC * DIM, HC * DIM + DIM
PINS = {"flashinfer_mxfp8_blockscaled_linear": "44d063202cb3947e"}
_g = {"pins": None, "w": {}}


def _pins_ok():
    if _g["pins"] is None:
        _g["pins"] = pins_match("ekq", PINS, {"flashinfer_mxfp8_blockscaled_linear":
                                              FU.flashinfer_mxfp8_blockscaled_linear})
    return _g["pins"]


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
    return {"wk4": wk4, "wk4sf": wk4sf, "inv_gsf": (1.0 / gsf).float(), "wv": w[NK:],
            "sv": sw[(NK // 128) * tile:], "full": full, "checked": False}


def _key(st, flat):
    a4, a4sf, pt = nvfp4_quantize(flat, 1.0 / (448.0 * 6.0), sfLayout=SfLayout.layout_128x4, do_shuffle=False,
                                  per_token_activation=True)
    return mm_fp4(a4, st["wk4"].t(), a4sf, st["wk4sf"], alpha=(pt.reshape(-1).float() * st["inv_gsf"]).contiguous(),
                  out_dtype=torch.bfloat16, block_size=16, backend="cute-dsl")


def warm(engram):
    """Build EKQ's private key weights and compile its kernels at the module's first eager call, which the engine's
    conditioning requests make before any scored window: the NVFP4 quantize + cute-dsl mm_fp4 compile once per process
    (~1.7 s measured cold), on synthetic rows here instead of inside the first large prefill call. Decode servers skip it
    (their calls never reach MIN_ROWS rows)."""
    if not ENABLED or id(engram) in _g["w"] or torch.cuda.is_current_stream_capturing() or not _pins_ok():
        return
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "decode":
        return
    st = _g["w"][id(engram)] = _build(engram) or False
    if st is False or _g.get("compiled"):
        return
    _key(st, torch.randn(256, K, device=st["wk4"].device, dtype=torch.bfloat16))
    _g["compiled"] = True
    say("ekq", "EKQ kernels compiled at the first eager call")


def _egate(engram, x, kv, image_ids):
    from sglang.srt.layers import engram as E

    return E.engram_gate(x, kv, engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value, image_ids,
                         engram.image_token_id)


def _key_kv(st, flat):
    key = _key(st, flat)
    with mxt.scope():                                       # the value: the engine's MXFP8 linear, MXT-checked tactic
        value = FU.flashinfer_mxfp8_blockscaled_linear(flat, st["wv"], st["sv"], backend="cute-dsl", pin_tactic=True)
    return torch.cat([key, value], dim=1)


def out(engram, x, flat, image_ids):
    """engram_gate(x, EKQ's wkv projection) for a call EKQ takes, or None (the stock projection)."""
    if (not ENABLED or flat.dim() != 2 or flat.shape[0] < MIN_ROWS or flat.shape[1] != K or flat.dtype != torch.bfloat16
            or not flat.is_contiguous() or torch.cuda.is_current_stream_capturing() or not _pins_ok()):
        return None
    st = _g["w"].get(id(engram))
    if st is None:
        st = _g["w"][id(engram)] = _build(engram) or False
    if st is False:
        return None
    if not st["full"]:
        say("ekq", "EKQ NVFP4 key projection on")
        return _egate(engram, x, _key_kv(st, flat), image_ids)
    o = _egate(engram, x, _key(st, flat), image_ids)
    if not st["checked"]:
        st["checked"] = True
        n = min(CHECK_ROWS, o.shape[0])
        d = (o[:n].float() - x[:n].float()).flatten(1).norm(dim=1) / o[:n].float().flatten(1).norm(dim=1).clamp_min(1e-30)
        share = d.median().item()
        if share > SHARE_MAX:
            say("ekq", f"EKQ-L Engram hash layer {getattr(engram, 'layer_hash_index', '?')}: update share {share:.3f} > "
                       f"{SHARE_MAX}, key half only from now on")
            st = _g["w"][id(engram)] = _build(engram, full=False) or False
            if st is False:
                return None
            return _egate(engram, x, _key_kv(st, flat), image_ids)
        say("ekq", f"EKQ-L Engram hash layer {getattr(engram, 'layer_hash_index', '?')}: full NVFP4 projection "
                   f"(update share {share:.3f})")
    return o
