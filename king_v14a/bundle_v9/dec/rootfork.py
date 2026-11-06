"""king-v13s ROOT-FORK and EKV-EARLY: two stream-wiring levers of the decode role's target-verify graph captures.

ROOT-FORK. The engine records its in-graph metadata prep (22 serial, mostly one-CTA kernels, ~54 us) on the capture
stream right before it calls the model, and the model's layer-0 head (embedding, hc_broadcast, Engram hash, rmsnorm +
quant, wqkv_a, q-norm, wq_b; ~44 us) reads none of what the prep writes. While a served graph is captured the model
forward runs on a stream S whose capture dependency set is emptied (cuStreamUpdateCaptureDependencies, driver API), so
its first node is a graph root and the head overlaps the prep. The KV store is the first prep consumer: every
`get_swa_out_cache_loc` read of the forward's first layer makes its stream wait on an event recorded on the capture
stream at the slot's entry, i.e. on the whole prep (the engine joins its own forked draft prep at that same call), and
the layer's end makes S wait on it again before layer 1. The capture stream waits on S before the LM head.

EKV-EARLY. The Engram lookup (PEER, collective-free) and the wkv projection of the layers in EKV_LAYERS read only the
hash ids, the tables and the weights. Right after the hash kernel the lookup (EKV_STAGE "lookup", the default) or the
lookup and the wkv ("wkv") are issued on a side stream E with the same calls the layer would make; the layer waits on
their event and consumes the stash (with "lookup" it then runs stock's own wkv call on the stashed rows, then the gate).
Why "lookup" (fixer, 11-06; GPU traces of the "wkv" build, live_1106/traces/king_v13s_all and king_v13s2_all, 240
rows): the wkv GEMM is a 148-CTA grid with 229 KB of shared memory a CTA, so on E it held every SM inside the root window
and the in-graph prep's multi-CTA kernels stalled behind it (the prep's 481-CTA copy: 3.2 us in king_v10b, 37 us there;
prep wall 55 -> ~119 us). Layer 0's KV store moved +21..+30 us later (all ranks) against -19..-31 us at layer 1, about
a wash. The PEER gather (720 small CTAs, no shared memory, 8-10 us, latency bound) overlaps the prep's small kernels
instead of evicting them, and layer 1 still saves the gather.

Same kernels, same inputs, same launch parameters: only stream and event wiring changes, so every output byte is
v10b's. Each lever has its own switch, engine source pins and a prepare-time probe; any failure leaves it off and the
forward runs king_v10b's path. One boot line per lever per decode rank.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import inspect
import sys
import time
from functools import partial

import torch

ROOT_FORK_ON = True
EKV_EARLY_ON = False  # king-v14a B2 (EKV nets ~0 / +14 us on L1 seams in v13s3)
EKV_LAYERS = (1,)  # Engram layers whose lookup (+ wkv) move to the hash point; 14 is a data-only opt-in (rig first)
# "lookup": only the PEER gather moves (the wkv runs at the layer, as in king_v10b); "wkv": the lookup and the wkv move
# (the first v13s build; measured net-negative, see the docstring). Data only: both stages are proven at prepare.
EKV_STAGE = "lookup"
PROBE_ROWS = (6, 96, 192, 240, 288)
BUDGET_S = 60.0
PREFIX = "bundle-v9 king-v13s"
SAY_LIMIT = 4
# sha256(inspect.getsource(fn))[:16] of live_1106/sg_cacheon_dsv41-pd-tee (the image's text; the AWS / VATTN pins
# computed the same way matched on the live box). v13s/tools/pins.py reproduces them from the text.
ROOT_PINS = {
    "DecodeCudaGraphRunner.capture_one_shape": "3aa5b1c9d2d58958",
    "DecodeCudaGraphRunner._run_in_graph_metadata_prep": "9a5e46663001e41b",
    "DecodeCudaGraphRunner._record_in_graph_metadata_prep_done": "ee264b03fa190457",
    "DeepseekV4AttnBackend.get_swa_out_cache_loc": "b4cdcf75272be2a1",
    "MQALayer._compute_kv_to_cache": "678224fac8995d64",
    "MQALayer._forward_prepare": "e45619be35c3ba0f",
    "MQALayer._forward_prepare_multi_stream": "00d5a176ff2fc84d",
    "DeepseekV4Model.forward": "5a66a844669ee55d",
    "DeepseekV4Model._forward_impl": "d026453719fc934b",
    "DeepseekV4CudaDecoderLayer.forward": "119ec8cf2f4b24b8",
    # fixer (fidelity review F3): the text the no-RAW argument (H2) rests on. What the in-graph prep writes (the Raw ->
    # Full verify upgrade and the in-graph refresh), what layer 0 does before its prepare, and what the head's hash
    # reads; the last is also in EKV_PINS, here so it is pinned with EKV off.
    "DeepseekV4AttnBackend.init_forward_metadata_in_graph": "13a897583ee4aadf",
    "DeepseekV4AttnBackend.make_forward_metadata_from_raw_verify": "019d1f789b7441d9",
    "MQALayer.forward": "b0d19428fed97020",
    "EngramHasher.forward": "cb27b9cf6906c9d7",
}
EKV_PINS = {
    "DeepseekV4Model._forward_layers_hc_pre_from_prev": "13d1c45bade86cac",
    "Engram.forward": "68cd442bb74b9e82",
    "Engram.embed_rows": "2b4088ed4ba1496c",
    "EngramEmbedding.forward": "68f791628981ea39",
    "EngramHasher.forward": "cb27b9cf6906c9d7",
}
_ACTIVE, _EVENT_RECORD = 1, 7  # CU_STREAM_CAPTURE_STATUS_ACTIVE, CU_GRAPH_NODE_TYPE_EVENT_RECORD
_MAX_DEPS = 8  # cuda_rootfork.kMaxDeps
_INFO_WORDS = 4 + 2 * _MAX_DEPS

_CU: dict = {}
_G = {"root": False, "root_why": "not prepared", "ekv": False, "ekv_why": "not prepared", "S": None, "E": None,
      "engrams": [], "forked": 0, "refused": 0, "untaken": 0, "said": 0, "stage": "lookup", "issued": 0, "taken": 0,
      "reported": False, "rank": None, "breakable": None}
_F: dict = {}  # the forward being captured: marker, armed, fired, ekv, stash, taken


def _say(what: str) -> None:
    print(f"{PREFIX} {what}", file=sys.stderr, flush=True)


def _note(what: str) -> None:
    if _G["said"] < SAY_LIMIT:
        _G["said"] += 1
        _say(what)


# ------------------------------------------------------------------------------------------- driver calls (JIT)
def _lib():
    """dec/cuda_rootfork.py's three host-only driver calls, built once (sglang's load_jit, as peer_lookup builds)."""
    if "lib" not in _CU:
        from sglang.kernels.jit.utils import load_jit
        from sglang.kernels.jit.utils.compile.cache import cache_root

        from bundle_v9.dec import cuda_rootfork, peer_lookup

        data = cuda_rootfork.SOURCE.encode()
        directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-rootfork",
                                 hashlib.sha256(data).hexdigest()[:16])
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, cuda_rootfork.NAME)
        if not os.path.isfile(path):
            peer_lookup._write(path, data)
        _CU["lib"] = load_jit("dec_rootfork", cuda_rootfork.TAG, cuda_files=[path],
                              cuda_wrappers=[("capture_info", "DecRootFork::capture_info"),
                                             ("empty_deps", "DecRootFork::empty_deps"),
                                             ("create_stream", "DecRootFork::create_stream")])
    return _CU["lib"]


def _capture_info(stream) -> tuple:
    """(status, capture id, ((dependency node, node type), ...)) of a stream."""
    out = torch.empty(_INFO_WORDS, dtype=torch.int64)
    _lib().capture_info(int(stream.cuda_stream), out)
    w = out.tolist()
    if w[0] != 0:
        raise RuntimeError(f"cuStreamGetCaptureInfo returned {w[0]}")
    if w[3] > _MAX_DEPS:
        raise RuntimeError(f"the stream's capture tail has {w[3]} nodes (more than {_MAX_DEPS} are not read)")
    return w[1], w[2], tuple((w[4 + 2 * i], w[5 + 2 * i]) for i in range(w[3]))


def _at_prep_marker(deps: tuple) -> bool:
    """The capture stream's tail is exactly one external event record node: the engine's in-graph prep marker."""
    return len(deps) == 1 and deps[0][1] == _EVENT_RECORD


def _empty_deps(stream, cap) -> None:
    """Empty a capturing stream's dependency set, so its next node is a graph root; raises unless verified."""
    out = torch.empty(1, dtype=torch.int64)
    _lib().empty_deps(int(stream.cuda_stream), out)
    if int(out[0]) != 0:
        raise RuntimeError(f"cuStreamUpdateCaptureDependencies returned {int(out[0])}")
    status, cid, deps = _capture_info(stream)
    if status != _ACTIVE or cid != _capture_info(cap)[1] or deps:
        raise RuntimeError(f"the forked stream is not a root of the capture (status {status}, {len(deps)} deps)")


# ------------------------------------------------------------------------------------------- per-forward wiring
def entry(batch):
    """At the slot's entry: the plan of this forward, or None outside a served capture (king_v10b's path)."""
    if not (_G["root"] or _G["ekv"]) or batch is None or not torch.cuda.is_current_stream_capturing():
        return None
    breakable = _G.get("breakable")  # fixer (fatal review F4): a breakable (--debug-cuda-graph) capture may end a
    if breakable is None or breakable():  # segment with S unjoined; such captures keep king_v10b's order
        return None
    mode = getattr(batch, "forward_mode", None)
    if mode is None or not mode.is_target_verify():
        return None
    cap = torch.cuda.current_stream()
    plan = {"cap": cap.cuda_stream, "root": False, "ekv": _G["ekv"], "batch": batch}
    if _G["root"]:
        try:
            status, cid, deps = _capture_info(cap)
            why = "" if status == _ACTIVE and _at_prep_marker(deps) else f"{len(deps)} tail nodes, not the marker"
        except Exception as e:  # noqa: BLE001  (a driver query failed: this graph keeps king_v10b's order)
            why = f"{type(e).__name__}: {str(e)[:200]}"
        if why:
            _G["refused"] += 1
            _note(f"ROOT-FORK: capture {_G['refused']} not forked (the slot's entry is not right behind the engine's "
                  f"in-graph prep marker: {why}); king_v10b's order serves it")
        else:
            plan.update(root=True, id=cid, deps=deps)
    return plan


def _hooked_swa_loc(stock, forward_batch):
    if _F.get("armed"):
        torch.cuda.current_stream().wait_event(_F["marker"])
        _F["fired"] += 1
    return stock(forward_batch)


def _root_ok(plan, cap, S):
    """(backend, "") when this capture forks, else (None, why): nothing was enqueued since the slot's entry, S is in
    no capture yet, and the backend's KV-store location read is the pinned stock method."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    try:
        status, cid, deps = _capture_info(cap)
        if cap.cuda_stream != plan["cap"] or cid != plan["id"] or deps != plan["deps"]:
            return None, f"GPU work was enqueued between the slot's entry and the forward ({len(deps)} tail nodes)"
        if _capture_info(S)[0] != 0:
            return None, "the root stream is already in this capture"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    backend = get_attn_backend()
    bound = backend.get_swa_out_cache_loc
    if getattr(bound, "__func__", None) is not _G["swa_fn"] or getattr(bound, "__self__", None) is not backend:
        return None, f"{type(backend).__name__}.get_swa_out_cache_loc is not the pinned stock method"
    return backend, ""


@contextlib.contextmanager
def routed(plan):
    """Innermost around view.forward: fork the forward onto the root stream S and / or arm EKV for this capture.
    The capture stream waits on S (and on any EKV stash no layer took) before anything after the forward."""
    if plan is None:
        yield
        return
    cap, S = torch.cuda.current_stream(), _G["S"]
    backend, joined, forked = None, False, False
    if plan["root"]:
        backend, why = _root_ok(plan, cap, S)
        if backend is None:
            _G["refused"] += 1
            _note(f"ROOT-FORK: capture not forked ({why}); king_v10b's order serves it")
    _F.clear()
    _F.update(ekv=plan["ekv"], stash={}, taken=[], armed=False, fired=0, marker=None)
    counts = (_G["issued"], _G["taken"], _G["untaken"])
    try:
        if backend is not None:
            marker = torch.cuda.Event()
            marker.record(cap)
            S.wait_event(marker)
            joined = True
            try:
                _empty_deps(S, cap)
                forked = True
            except Exception as e:  # noqa: BLE001  (S still waits on the marker; the forward runs on cap)
                _G["refused"] += 1
                _note(f"ROOT-FORK: capture not forked ({type(e).__name__}: {str(e)[:200]}); king_v10b's order")
        if forked:
            _F.update(marker=marker, armed=True)
            stock = type(backend).get_swa_out_cache_loc.__get__(backend)
            backend.get_swa_out_cache_loc = partial(_hooked_swa_loc, stock)
            _G["forked"] += 1
            if _G["forked"] == 1:
                rows = getattr(plan["batch"], "input_ids", None)
                _say(f"ROOT-FORK: first served capture forked ({None if rows is None else rows.shape[0]} rows): the "
                     "model forward starts at the graph root and joins the engine's prep at its first KV store")
            with torch.cuda.stream(S):
                yield
        else:
            yield
    finally:
        if forked:
            del backend.get_swa_out_cache_loc
        for stash in _F.get("stash", {}).values():  # EKV work no layer consumed: joined before the capture ends
            cap.wait_event(stash[3])
            _G["untaken"] += 1
            _note(f"EKV-EARLY: a stash was not consumed by its layer ({_G['untaken']} so far); joined, the layer "
                  "ran king_v10b's path")
        if joined:
            cap.wait_stream(S)
        _F.clear()
        if plan["ekv"] and not _G["reported"]:  # fixer (review F6): EKV's engagement in the first served capture
            _G["reported"] = True
            rows = getattr(plan["batch"], "input_ids", None)
            _say(f"EKV-EARLY: first served capture (rank {_G['rank']}; {None if rows is None else rows.shape[0]} rows, "
                 f"stage {_G['stage']}): early jobs issued {_G['issued'] - counts[0]}, taken by their layer "
                 f"{_G['taken'] - counts[1]}, joined untaken {_G['untaken'] - counts[2]}")


def _layer0_guard(prev, *args, **kwargs):
    """Layer 0 of a forked forward: it must have read the KV-store location (the prep join); S then waits on the
    prep marker once more, so everything from layer 1 on is ordered after the prep whatever stream it forks."""
    out = prev(*args, **kwargs)
    if _F.get("armed"):
        _F["armed"] = False
        if not _F["fired"]:
            raise RuntimeError(f"{PREFIX} ROOT-FORK: layer 0 issued no KV-store location read, so its attention "
                               "would read the in-graph metadata without waiting on the prep")
        torch.cuda.current_stream().wait_event(_F["marker"])
    return out


# ------------------------------------------------------------------------------------------- EKV-EARLY
def _hash_fork(live, input_ids, forward_batch):
    """The model's Engram hash (the live hasher's own forward), then the early lookup + wkv of the EKV layers."""
    hash_ids = live(input_ids, forward_batch)
    if _F.get("ekv") and hash_ids.dim() == 3 and hash_ids.shape[0] > 0:
        _ekv_fork(hash_ids, forward_batch)
    return hash_ids


def _early_ok(engram, rows, forward_batch, global_rows) -> bool:
    """The layer would run stock Engram.forward with the PEER lookup (collective-free) at these rows."""
    from bundle_v9.dec import col_exchange, peer_lookup

    if forward_batch is None or getattr(forward_batch, "engram_staged_rows", None) is not None:
        return False
    if col_exchange._wkv_route(engram, rows) is not None or id(engram.embed) not in peer_lookup._VIEWS:
        return False
    return peer_lookup._served(forward_batch, global_rows, rows)


def _ekv_fork(hash_ids, forward_batch) -> None:
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len

    from bundle_v9.dec import peer_lookup

    rows, global_rows = hash_ids.shape[0], get_global_dp_buffer_len()
    jobs = [engram for engram in _G["engrams"] if _early_ok(engram, rows, forward_batch, global_rows)]
    if not jobs:
        return
    E, wkv = _G["E"], _G["stage"] == "wkv"
    E.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(E):
        for engram in jobs:
            # the calls stock Engram.forward makes: embed_rows -> EngramEmbedding.forward -> _dp_sharded_lookup
            # (col_exchange.dp_sharded_lookup, whose first statement is this lookup), then self.wkv(emb.flatten(-2))
            emb = peer_lookup.lookup(engram.embed, hash_ids[:, engram.layer_hash_index], forward_batch)
            if emb is None:
                raise RuntimeError(f"{PREFIX} EKV-EARLY: PEER_LOOKUP refused a lookup it reported served")
            out = ("kv", engram.wkv(emb.flatten(-2))[0]) if wkv else ("emb", emb)
            done = torch.cuda.Event()
            done.record(E)
            _F["stash"][id(engram)] = (hash_ids, engram.layer_hash_index, out, done, forward_batch, emb)
            _G["issued"] += 1


def ekv_take(engram, x, hash_ids, forward_batch, cp_all_tokens):
    """At the Engram layer (col_exchange.engram_forward): ("emb", rows) or ("kv", kv) of this layer's early work, after
    the current stream waited on its event, or None (the layer then runs king_v10b's path). The stash stays referenced
    until the forward's end."""
    stash = _F.get("stash")
    entry = stash.pop(id(engram), None) if stash else None
    if entry is None:
        return None
    base, index, out, done, batch, _ = entry
    want = base[:, index]
    same = (hash_ids.data_ptr(), tuple(hash_ids.shape), hash_ids.stride(), hash_ids.dtype) == (
        want.data_ptr(), tuple(want.shape), want.stride(), want.dtype)
    if cp_all_tokens or forward_batch is not batch or not same or out[1].shape[0] != x.shape[0]:
        stash[id(engram)] = entry
        return None
    _F["taken"].append(entry)
    torch.cuda.current_stream().wait_event(done)
    _G["taken"] += 1
    return out


# ------------------------------------------------------------------------------------------- prepare
class _Off(Exception):
    pass


def _digest(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


def _pins(expected: dict, fns: dict) -> None:
    got = {name: _digest(fn) for name, fn in fns.items()}
    bad = {name: value for name, value in got.items() if expected.get(name) != value}
    if bad or set(got) != set(expected):
        raise _Off(f"engine pins MISMATCH {bad}")


def _new_stream(device):
    """A non-blocking stream no torch pool hands out (cuStreamCreate), so it is no other lever's or the engine's."""
    out = torch.empty(2, dtype=torch.int64)
    with torch.cuda.device(device):
        torch.cuda.current_stream().synchronize()
        _lib().create_stream(out)
    rc, handle = (int(v) for v in out.tolist())
    if rc != 0 or not handle:
        raise RuntimeError(f"cuStreamCreate returned {rc}")
    return torch.cuda.ExternalStream(handle, device=device)


def _view_module(module):
    import copy

    view = copy.copy(module)
    view._parameters = module._parameters.copy()
    view._buffers = module._buffers.copy()
    view._modules = module._modules.copy()
    return view


def _probe_root(device, S) -> str:
    """The served mechanism on a private graph: a serial 'prep' chain then an external event record on the capture
    stream; a 'head' on S forked to the root that reads only static inputs; a consumer of the prep output behind the
    marker wait; S joined before the tail. Three input rotations, NaN-filled output, each replay of the forked and of
    the unforked graph against eager, bit for bit."""
    gen = torch.Generator(device=device).manual_seed(20261106)
    n = 1 << 16
    src, x = torch.randn(n, device=device, generator=gen), torch.randn(n, device=device, generator=gen)
    prep, out = torch.empty_like(src), torch.empty_like(src)
    ext = torch.cuda.Event(external=True)

    def body(fork):
        cap = torch.cuda.current_stream()
        prep.copy_(src)
        for _ in range(8):
            prep.mul_(1.0009765625).add_(0.5)
        ext.record(cap)
        stream, marker = cap, None
        if fork:
            status, _, deps = _capture_info(cap)
            if status != _ACTIVE or not _at_prep_marker(deps):
                raise RuntimeError(f"probe: the capture tail is not one event record node ({len(deps)} nodes)")
            marker = torch.cuda.Event()
            marker.record(cap)
            S.wait_event(marker)
            try:
                _empty_deps(S, cap)
            except Exception:
                cap.wait_stream(S)  # joined, so the failed probe's capture still ends cleanly
                raise
            stream = S
        with torch.cuda.stream(stream):
            head = (x * 3.0 + 1.0) * x
            if marker is not None:
                torch.cuda.current_stream().wait_event(marker)
            res = head + prep
        if fork:
            cap.wait_stream(S)
        out.copy_(res * 2.0)

    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body(False)
    torch.cuda.current_stream().wait_stream(side)
    graphs = {}
    for fork in (False, True):
        graphs[fork] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[fork], capture_error_mode="thread_local"):
            body(fork)
    for rotation in range(3):
        src.copy_(torch.randn(n, device=device, generator=gen))
        x.copy_(torch.randn(n, device=device, generator=gen))
        p = src.clone()
        for _ in range(8):
            p.mul_(1.0009765625).add_(0.5)
        ref = ((x * 3.0 + 1.0) * x + p) * 2.0
        for fork in (False, True):
            out.fill_(float("nan"))
            graphs[fork].replay()
            torch.cuda.synchronize(device)
            if not torch.equal(out.view(torch.int32), ref.view(torch.int32)):
                raise RuntimeError(f"probe: the {'forked' if fork else 'unforked'} graph differs from eager "
                                   f"(rotation {rotation})")
    del graphs
    torch.cuda.synchronize(device)
    return "probe: forked + unforked private graphs, 3 rotations, NaN-filled output, bitwise vs eager"


def _prepare_root(view, module, device) -> str:
    if not ROOT_FORK_ON:
        raise _Off("switched off by rootfork.ROOT_FORK_ON")
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
    from sglang.srt.layers.engram import EngramHasher
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.models import deepseek_v4 as dv4

    runner, mqa = DecodeCudaGraphRunner, dv4.MQALayer
    _pins(ROOT_PINS, {
        "DecodeCudaGraphRunner.capture_one_shape": runner.capture_one_shape,
        "DecodeCudaGraphRunner._run_in_graph_metadata_prep": runner._run_in_graph_metadata_prep,
        "DecodeCudaGraphRunner._record_in_graph_metadata_prep_done": runner._record_in_graph_metadata_prep_done,
        "DeepseekV4AttnBackend.get_swa_out_cache_loc": DeepseekV4AttnBackend.get_swa_out_cache_loc,
        "MQALayer._compute_kv_to_cache": mqa._compute_kv_to_cache,
        "MQALayer._forward_prepare": mqa._forward_prepare,
        "MQALayer._forward_prepare_multi_stream": mqa._forward_prepare_multi_stream,
        "DeepseekV4Model.forward": dv4.DeepseekV4Model.forward,
        "DeepseekV4Model._forward_impl": dv4.DeepseekV4Model._forward_impl,
        "DeepseekV4CudaDecoderLayer.forward": dv4.DeepseekV4CudaDecoderLayer.forward,
        "DeepseekV4AttnBackend.init_forward_metadata_in_graph": DeepseekV4AttnBackend.init_forward_metadata_in_graph,
        "DeepseekV4AttnBackend.make_forward_metadata_from_raw_verify":
            DeepseekV4AttnBackend.make_forward_metadata_from_raw_verify,
        "MQALayer.forward": mqa.forward,
        "EngramHasher.forward": EngramHasher.forward,
    })
    if is_unified_kv_triton():
        raise _Off("unified-KV store location (not a join point, as the engine's own draft-prep fork rules)")
    layer, layer_view = module.layers[0], view.layers[0]
    attn = layer.self_attn
    if (type(layer) is not dv4.DeepseekV4CudaDecoderLayer or type(attn) is not mqa or attn.compressor is not None
            or attn.indexer is not None or layer.engram is not None or not attn.use_mega_attention):
        raise _Off("layer 0 is not a plain SWA mega-attention MQALayer without Engram")
    if not module.hc_pre_from_prev_sublayer or module.pp_group.world_size != 1 or layer_view is layer:
        raise _Off("the model is not the one-stage boundary-layer stack with a private layer-0 view")
    S = _new_stream(device)
    detail = _probe_root(device, S)
    _G.update(S=S, swa_fn=DeepseekV4AttnBackend.get_swa_out_cache_loc)
    layer_view.forward = partial(_layer0_guard, layer_view.forward)
    return (f"target-verify captures: the model forward on a root-forked stream, joined to the engine's in-graph prep "
            f"at layer 0's first KV-store location read; {len(ROOT_PINS)} engine pins match; {detail}")


def _ekv_once(engram, hash_ids, tables, like=None):
    """(the gathered rows, the wkv output) of one early call; `like` = a reference pair whose blocks are NaN-poisoned
    first, so the next same-size allocations on this stream start from NaN."""
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather_peer

    from bundle_v9.dec import peer_lookup

    ids = hash_ids[:, engram.layer_hash_index]
    flat = ids.reshape(-1, ids.shape[-1])
    if like is not None:  # poison the block the next same-size allocation on this stream takes
        torch.full_like(like[0], float("nan"))
    emb = torch.empty((*flat.shape, peer_lookup.DIM), dtype=torch.bfloat16, device=ids.device)
    engram_gather_peer(flat, emb, *tables, peer_lookup.DIM, peer_lookup.BLOCK)
    if like is not None:
        torch.full_like(like[1], float("nan"))
    kv, _ = engram.wkv(emb.view(*ids.shape, peer_lookup.DIM).flatten(-2))
    return emb, kv


def _probe_ekv(module, device, t0) -> str:
    """The early calls (the served PEER gather and the served wkv route, under decode_gemm's qm_decode as in
    routed()) on E and, concurrently, on a second stream, against serial references kept alive (no aliasing),
    bit for bit, at PROBE_ROWS; hash ids drawn per column inside the layer's bounds, strided as the model slices."""
    from bundle_v9.dec import decode_gemm, peer_lookup

    gen = torch.Generator(device=device).manual_seed(20261107)
    other, E = torch.cuda.Stream(device=device), _G["E"]
    layers = int(module.engram_hasher.primes.shape[0])
    checked = 0
    for engram in _G["engrams"]:
        bounds = module.engram_layout.column_bounds(engram.layer_hash_index)
        tables = peer_lookup._TABLES[id(engram.embed)]
        method = engram.wkv.quant_method
        stock = method.w8a8_mxfp8_linear
        method.w8a8_mxfp8_linear = partial(decode_gemm.qm_decode, stock)
        try:
            for rows in PROBE_ROWS:
                if time.monotonic() - t0 > BUDGET_S:
                    raise RuntimeError(f"probe budget {BUDGET_S} s exhausted at {rows} rows")
                sets = []
                for _ in range(2):
                    ids = torch.zeros((rows, layers, len(bounds) - 1), dtype=torch.int64, device=device)
                    for c in range(len(bounds) - 1):
                        ids[:, engram.layer_hash_index, c] = torch.randint(
                            int(bounds[c]), int(bounds[c + 1]), (rows,), device=device, generator=gen)
                    sets.append(ids)
                refs = [_ekv_once(engram, ids, tables) for ids in sets]
                torch.cuda.synchronize(device)
                cur, outs = torch.cuda.current_stream(), []
                for stream, ids, ref in zip((E, other), sets, refs):
                    stream.wait_stream(cur)
                    with torch.cuda.stream(stream):
                        outs.append(_ekv_once(engram, ids, tables, like=ref))
                for stream in (E, other):
                    cur.wait_stream(stream)
                torch.cuda.synchronize(device)
                for ref, got in zip(refs, outs):
                    for what, a, b in (("gathered rows", ref[0], got[0]), ("kv", ref[1], got[1])):
                        if b.data_ptr() == a.data_ptr() or not torch.equal(b.view(torch.int16), a.view(torch.int16)):
                            raise RuntimeError(f"probe: layer {engram.layer_hash_index}'s early {what} at {rows} rows "
                                               "differ from the serial call")
                checked += 1
        finally:
            method.w8a8_mxfp8_linear = stock
    torch.cuda.synchronize(device)
    return (f"probe: {checked} (layer, rows) cases, gathered rows and kv on E and a second stream concurrent, bitwise "
            f"vs serial")


def _prepare_ekv(view, module, device, t0) -> str:
    if not EKV_EARLY_ON:
        raise _Off("switched off by rootfork.EKV_EARLY_ON")
    if EKV_STAGE not in ("lookup", "wkv"):
        raise _Off(f"rootfork.EKV_STAGE {EKV_STAGE!r} is neither 'lookup' nor 'wkv'")
    from sglang.srt.layers.engram import Engram, EngramEmbedding, EngramHasher
    from sglang.srt.models import deepseek_v4 as dv4
    from sglang.srt.runtime_context import get_parallel

    from bundle_v9.dec import col_exchange, peer_lookup

    _pins(EKV_PINS, {
        "DeepseekV4Model._forward_layers_hc_pre_from_prev": dv4.DeepseekV4Model._forward_layers_hc_pre_from_prev,
        "Engram.forward": Engram.forward,
        "Engram.embed_rows": Engram.embed_rows,
        "EngramEmbedding.forward": EngramEmbedding.forward,
        "EngramHasher.forward": EngramHasher.forward,
    })
    hasher = module.engram_hasher
    if type(hasher) is not EngramHasher or view._modules.get("engram_hasher") is not hasher:
        raise _Off("the model's Engram hasher is not stock's, or the view's is already replaced")
    if not peer_lookup._STATE["on"] or get_parallel().attn_dp_size <= 1:
        raise _Off(f"the PEER lookup does not serve this engine ({peer_lookup._STATE['detail']})")
    engrams = []
    for index in EKV_LAYERS:
        engram = view.layers[index].engram
        embed = getattr(engram, "embed", None)
        if (engram is None or getattr(engram.forward, "__func__", None) is not col_exchange.engram_forward
                or getattr(getattr(embed, "_dp_sharded_lookup", None), "__func__", None)
                is not col_exchange.dp_sharded_lookup):
            raise _Off(f"layer {index} has no COL_EXCHANGE Engram view")
        if (embed.peer is not None or embed._shared or embed.staged_by_scheduler or embed.tp_size <= 1
                or id(embed) not in peer_lookup._VIEWS or id(embed) not in peer_lookup._TABLES):
            raise _Off(f"layer {index}'s Engram lookup is not the PEER-served DP-sharded one")
        engrams.append(engram)
    _G.update(engrams=engrams, E=_new_stream(device), stage=EKV_STAGE)
    detail = _probe_ekv(module, device, t0)
    hasher_view = _view_module(hasher)
    hasher_view.forward = partial(_hash_fork, hasher)
    view.engram_hasher = hasher_view
    what = ("PEER lookup on a side stream right after the hash kernel; the layer waits on it, then runs stock's wkv "
            "and gate" if EKV_STAGE == "lookup" else
            "PEER lookup + wkv on a side stream right after the hash kernel; the gate waits on them")
    return (f"target-verify captures: Engram layers {list(EKV_LAYERS)}, stage {EKV_STAGE}: {what}; "
            f"{len(EKV_PINS)} engine pins match; {detail}")


def prepare(view, module) -> None:
    """Never raises: each lever is ON only when its pins, domain and probe pass; otherwise king_v10b's path."""
    from sglang.srt.runtime_context import get_parallel

    t0 = time.monotonic()
    rank = get_parallel().tp_rank
    _G.update(rank=rank, reported=False)
    device = module.layers[0].self_attn.wq_b.weight.device
    try:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )

        _G["breakable"] = is_in_breakable_cuda_graph
    except Exception:  # noqa: BLE001  (entry() then never forks: both levers report off below)
        _G["breakable"] = None
    for name, key, fn in (("ROOT-FORK", "root", _prepare_root), ("EKV-EARLY", "ekv", _prepare_ekv)):
        try:
            if _G["breakable"] is None:
                raise _Off("the engine's is_in_breakable_cuda_graph is not importable (entry() could not refuse a "
                           "breakable capture)")
            detail = fn(view, module, device) if key == "root" else fn(view, module, device, t0)
            _G.update({key: True, f"{key}_why": detail})
        except Exception as e:  # noqa: BLE001  (any failure: this lever off, king_v10b's path)
            why = str(e) if isinstance(e, _Off) else f"{type(e).__name__}: {str(e)[:300]}"
            _G.update({key: False, f"{key}_why": why})
        _say(f"{name}: {'ON' if _G[key] else 'off'} (rank {rank}; {_G[f'{key}_why']}; "
             f"{time.monotonic() - t0:.1f} s)")
