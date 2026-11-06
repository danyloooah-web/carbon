# bundle-v9

A model-slot composition for the sealed prefill/decode arena of DeepSeek-V4.1-Flash-1017
(8xB300, TP4/EP4, MegaMoE, boundary decoder layers). One registered `model` op builds private
model and layer views that share the stock weights but own their child registries, so binding
never edits the stock model the audit and the reference use. Every lever is prepared before
serving, scoped to the candidate's own forward, and restored on exit; calls outside a lever's
envelope run stock's own path, and a failure inside an active strict route raises rather than
silently substituting a result.

## Decode and verify

- **MOE-FRONT + FRONTFIX** (`moe_front.py`, `cuda_moe_front.py`, `moe_route.py`): each MoE
  layer's routing, routed-scale multiply and MegaMoE pre-dispatch as one launch, with the
  non-split template and isolated decode boundary inspection; byte-checked at prepare. The front
  kernel triggers its dependents on every exit path after its last store, so the MegaMoE launch
  overlaps the front's tail; MegaMoE waits on the front's completion before it reads, so outputs
  are unchanged. Calls of >= 4,096 rows at row offset 0 with padding run the persistent front (one
  CTA per SM, bulk-loaded rows, warp-level top-k); every other call runs the per-row kernel.
  Prepare byte-probes both against stock's three launches at 1..288 and 4,096..32,768 rows with
  shared block sizes 64, 128, 192 and 240.
- **MOE-DECODE** (`moe_prefill.py`, `cuda_moe_prefill.py`): every decode / verify MegaMoE call
  of 1..288 rows, capture included, on the MOE-PREFILL host library with the image's knobs and
  tiles. Kernel-header changes that keep the image kernel's output bytes: both remote load
  rounds of a token's scale words are issued before the stores (384-expert top-6 at BLOCK_M <=
  64); at <= 512 rows the shared expert's L2 tasks run in the dispatch gap instead of after the
  routed tasks; the dispatch sends a per-expert count atomic only for experts with local picks
  (SM 0's push to the peers carries every SM's arrival). The scale-load order applies at
  BLOCK_M <= 64; the shared-expert and count changes apply at every BLOCK_M, including
  MOE-PREFILL's kernel. Decode and colocated engines
  serve it, so the colocated audit compares it to stock. A call of those rows outside the
  prepared facts raises.
- **DENSE** (`dense_gemm.py`, `tactic_pins.py`, `cuda_swizzle.py`): pinned split_k=1 CuTe-DSL
  MXFP8 tactics for the four decode projections (`wo_b` bands with the one-launch PDL swizzle,
  `wq_b` incl. the 257..288 wide band, `engram.wkv`, `indexer.wq_b`), STRICT byte-probed
  against stock at prepare.
- **PEER** (`peer_lookup.py`): collective-free NVLink CUDA IPC peer-shard Engram gather on the
  decode role; the row-exchange module (`row_exchange.py`) serves the re-homing owner's lookup.
- **FLOOR + MMHC + SK** (`mhc_floor.py`, `attn_boundary.py`, `mhc_stats.py`,
  `cuda_sinkhorn.py`, `cuda_mhc_packed.py`): the decode mHC floor with the exact attention
  boundary (full-SM width, one persistent barrier pair per GPU), the FFN boundary form, and the
  sinkhorn stats reduce.
- **INDEX-HOIST** (`index_hoist.py`): the indexer layers' query side hoisted off the sources
  stream.
- **COUNTS** (`replay_counts.py`): the replay-tail DP counts taken from the batch's exchanged
  counts.

## Prefill

- **MHC-SCHED** (`mhc_sched.py`): the mega mHC boundary programs the bundle's own boundary
  calls build (the 16-, 20/27- and 40-split shifted col-scale programs) are compiled from the
  image's sources with three scheduling edits: the store warp issues the next task's coefficient
  TMA at the task's first hidden block instead of its last, the Post workers' atom loop is
  not unrolled, and the four split-barrier polls sleep 256 ns between reads of the counter.
  The edits are applied at prepare to the image file after its SHA-256 matches,
  and the edited bytes must match a fixed SHA-256. The edited loader is installed only around
  the bundle's own boundary calls and restored after each one. Every other program is
  untouched. Prepare byte-probes the edited programs against stock's boundary at 62..4096 rows
  and raises unless all three were built. On a prefill engine DeepGEMM's mega mHC kernel serves
  the re-home steps' bf16-only FFN entries of >= 2,048 rows in eager forwards with three edits:
  the same coefficient-TMA move, its Post atom loop unrolled by two, and the Post workers' bf16
  pairs widened to fp32 by bit placement (`u << 16`, `u & 0xffff0000`, the bits the image's
  conversion produces) instead of a split into halves. The edited header (image SHA-256
  checked, edited bytes fixed) is placed in the
  MOE-PREFILL host library's own include root, and that library carries DeepGEMM's mega mHC host
  path (sgl-deep-gemm v0.2.3.post2, the image's) with the stock runtime's SM count, PDL and
  deterministic settings mirrored at each call. Stock's `deep_gemm._C` is never re-pointed, so
  the reference keeps the image kernel. Prepare byte-probes that route against stock's boundary
  on the first and last layers at 2,050 and 4,094 rows.

- **PLACE** (`placement.py`, `placement_table.py`): layers 0..19's routed experts placed
  across the four EP ranks from a fitted table; settled once on the first eager forward before
  graph capture. Outputs are stock's except where the router's top-k breaks an exact score tie
  by expert id, which the relabeling reorders.
- **MOE-FRONT / tile band** (`moe_route.py`): the fused front serves eager prefill with the MoE
  tile band; decode graph shapes take the image's tiles.
- **PGEMM + CUBLT** (`prefill_gemm.py`): prefill dense GEMM routes for `wq_b` / `wo_b` /
  `wqkv_a` / the indexer's `wq_b` / the Engram `wkv` on swept CuTe / CUTLASS tactics, with
  cuBLASLt (`torch._scaled_mm`, e8m0 block scales on the swizzled bytes) serving the Engram
  `wkv` and the indexer's `wq_b` from 1024 rows and `wqkv_a` in its band; bitwise with the
  tactics they replace and prepare-checked on the registered weights.
- **PATTN** (`prefill_attn.py`, `cuda_prefill_attn.py`): the eager sparse-prefill MegaAttention
  core for >= 8,192 query rows — reordered waits, loads and barrier arrivals only; outputs are
  the image kernel's bytes. On a re-home step PATTN stands down for the whole step and stock's
  attention serves (PATTN-RH below, which lifts that, ships switched off).
- **MOE-PREFILL** (`moe_prefill.py`, `cuda_moe_prefill.py`): the eager prefill MegaMoE on the
  store-block / epilogue / TMEM-prefetch pipeline variant for DP-global forwards of >= 49,152
  tokens (MP-WIDE below, which extends it into proven bands beneath that floor, ships switched
  off); regrouped epilogue
  rows and reordered waits only. A PD decode engine builds the same library for MOE-DECODE.
- **PATTN-RH** (`pf_attn.py`, `prefill_attn.py`; switch `pf_attn.PATTN_RH`, **ships False**;
  prefill and colocated engines): PATTN also serves the eager sparse-prefill calls of >= 8,192
  rows made while a re-home step is in flight, where it otherwise stands down for the whole
  planned step. The flight spans the step's whole layer loop, so this covers the owner's kept
  rows, the helpers' moved pieces and the bystanders' calls of the re-homed layers 0..19, and
  the eager calls of the later layers of that step (which PATTN serves on every other step).
  PATTN launches one cluster per query row, so a row's bytes depend only on that row's inputs and
  the layer's sink, scale and cos/sin table; an in-flight call differs only in which buffers hold
  those inputs, and the shape, dtype and stride checks are the same as outside a flight. Prepare
  proves PATTN against the image's legacy kernel bit for bit on every re-home layer with that
  layer's own sink, scale and cos/sin table (8,195 synthetic rows in the engine's combine layout,
  at both index widths, with zero-length rows and an offset kv view); on any difference or raise
  the stand-down holds on that rank. Rank-local; nothing extra runs in flight (no host sync, no
  compare), and a raise in flight turns PATTN-RH off for the process with the legacy kernel
  serving that call. A development switch, `pf_attn.RH_CHECK` (0 as shipped), compares PATTN
  with the legacy kernel on the first N in-flight calls on their live inputs, keeping only
  device-side difference counts in flight and reading them at the end of the forward, after the
  re-home step has joined. It ships off because its audited-role exactness rests on the
  synthetic proof until that compare has run. Boot line: `bundle-v9 king-v7 PATTN-RH: off (...)`
  (or `ON (...)`).
- **MP-WIDE** (`pf_mpwide.py`, `moe_prefill.py`; switch `pf_mpwide.MP_WIDE`, **ships False**;
  prefill and colocated engines): MOE-PREFILL also serves eager calls below 49,152 DP-global
  tokens wherever
  the image's own heuristic picks BLOCK_M 240. The bands are scanned at prepare through the same
  heuristic call `serves()` makes, not written in. A MegaMoE config is a function of BLOCK_M, the
  knobs and the model facts, and the kernel template carries no token count, so these calls run
  the same prepared binary pair as calls above the floor; `compile_only` at both edges of every
  band must return the prepared config, and the EP group votes (one CPU-group gather that every
  rank reaches). Any failure on any rank keeps the 49,152 floor on every rank. It ships off: the
  new bands are a regime the floor never served, and their byte equality has not yet been
  measured with `moe_prefill.CHECK` (which needs every EP rank busy in each checked forward).
  Boot line: `bundle-v9 king-v7 MP-WIDE: off (...)` (or `ON (...)`).
- **PEER-READ** (`peer_read.py`, `cuda_peer_read.py`): inside the column-exchange domain of the
  Engram DP lookup, each rank reads its own rows' Engram entries directly from the owning
  peers' shards over CUDA IPC and packs them to the receiver's MXFP8 codes and swizzled scale
  bytes in one kernel, so those forwards run no exchange collectives; every DP rank takes the
  same path in the same forwards. The packed bytes are the column exchange's by construction
  (the same gather-pack arithmetic and placement layout). Peers are mapped on the first served
  forward.
- **COL-EXCHANGE + XFUSE** (`col_exchange.py`, `exchange_fuse.py`, `cuda_exchange_fuse.py`):
  the Engram DP lookup exchanges each shard's own hash columns for SUM_LEN forwards of at least
  8,192 rows, with the send side gathered and quantized to the receiver's MXFP8 codes and
  swizzled scale bytes in one kernel; bitwise stock. Smaller, MAX_LEN, gatherv, decode and
  capturing forwards run stock's lookup, except the bf16 lookup under PEER-SMALL below.
- **PEER-SMALL** (`peer_read.py`, `col_exchange.py`; switch `peer_read.PEER_SMALL`): the bf16
  PEER-READ lookup also serves uncaptured SUM_LEN forwards with extend in batch below 8,192
  DP-global rows (from 1 row), on the prefill and colocated engines. Same kernel and bytes as its
  >= 8,192-row use: the owner's value with -0 written as +0, which is what stock's all-reduce of
  the owner's value and three +0 produces in any order. The FP8 place path and the column
  exchange keep the 8,192-row floor. Every input of the decision is global (DP counts, extend in
  batch, padding mode, capture), so all ranks take the peer path or all run stock's collective.
  With PEER-SMALL on, PEER-READ's peer mapping runs at prepare instead of in the first served
  forward, followed by a probe: each rank's peer reads of 16 rows a layer must equal each owner's
  stock `_owned_rows` bitwise. The group decides together by vote; on any failure PEER-SMALL is
  off on every rank and the 8,192-row path maps lazily, as before.
- **WTILE** (`wtile.py`; switch `wtile.WTILE`): in eager EXTEND forwards the dense full top-k of
  the index layers outside the candidate scheme (layers 2, 8 and 14) cuts its index-logits rows
  into tiles of whole SM rounds of DeepGEMM's logits kernel instead of stock's budget-sized
  tiles. Tile offsets stay multiples of the 4-row q block, and a tile changes only when both
  stock's and the new last tile exceed max(SM count, DeepSelect's largest cluster tier), so every
  tile stays on DeepSelect's row-local NORMAL class. Stock's tiles serve the first 4 changed-tile
  calls of a process, whose new-tile top-k is compared bitwise in scratch; a mismatch or a raise
  turns WTILE off and stock's tiles serve. After that, a changed call serves the new tiles only
  once a compare has passed in its class (the sides of every tile height against DeepGEMM's
  rows thresholds, and of the width against its width thresholds); an unproven class serves
  stock's tiles and is compared first.
- **PFUSE** (`prefill_fuse.py`, `cuda_prefill_fuse.py`): bitwise-exact small-kernel
  replacements in the cold prefill forward (q-norm + MXFP8, SWA dequant, token-id gather,
  index-q pack, the wo_b packed-scale permutation, chunk-combine reuse), each inside its own
  domain and stood down while the re-homing owner flies.
- **TOPK** (`exact_topk.py`, `cuda_exact_topk.py`): the exact wide top-k for dense prefill.
- **REHOME** (`rehome_plan.py`, `rehome_transport.py`, `rehome_block.py`, `rehome_loop.py`,
  `rehome_edges.py`, `rehome_attn.py`, `rehome_probe.py`): on a step whose shared plan re-homes
  token-local rows of layers 0..19 from heavy DP ranks to light ones, the owner executes those
  layers with bundle-owned symmetric buffers; the lean and chunk-combine levers stand down for
  those layers on that step (every rank reads the same plan). A heavy rank's tail rows go to one,
  two or three lighter ranks, and the owner keeps at least a quarter of its rows and 4,096 rows; a
  plan outside the transport's block, slot and row limits runs stock's loop on every rank. The
  transport holds 393,216
  compressed and 65,536 SWA workspace slots, and a helper's local copy at most their sum. The
  workspace bounds are exchanged before the step commits, and a step whose owner range exceeds
  them runs stock's loop on every rank.
- **EARLY** (`early_layers.py`): the early layers' lean forwards and captured graphs on the
  shared barrier pair.

## Scope

Each lever is exact inside its envelope (bitwise or byte-checked at prepare) unless it states a
tolerance, and each restore path returns stock's own functions in `finally`. Domain checks in
`slots.py` raise unless all 40 MoE layers are served.

## Decode role

On the PD decode server the slot serves the decode package `bundle_v9/dec` (`dec/slots.py`): the fused MoE front
(`moe_front.py`, `moe_route.py`), decode expert placement (`placement.py`, `decode_placement.py`), the decode MegaMoE
path with sparse send counts and shared-expert scheduling (`moe_decode.py`), re-measured decode tiles, MegaMoE dispatch
on SM 0 alone at <= 96 rows, compressed-layer decode attention (`decode_attn.py`), pinned wq_b tactics, exact CuTe-DSL
decode tactics for wo_b (<= 256 rows) and wqkv_a (<= 216 rows; to 288 rows under WQKV288 below) (`decode_gemm.py`),
collective-free Engram decode lookups
(`peer_lookup.py`), the index layers' decode paged logits and paged top-k (`paged_logits.py`, `paged_topk.py`), and the
indexer side stream and reused MXFP8 pair (`index_streams.py`). Decode boundaries skip Mega-mHC up to 288 rows a rank.
The attention-side mHC stats of 32..320-row calls run both prenorm projections in one kernel (`prenorm_pair.py`). The
compressed-layer attention kernels and the decode boundary kernel issue `griddepcontrol.launch_dependents` after their
last store, so the next DeepGEMM GEMM launches over their tail (the GEMM reads nothing before `griddepcontrol.wait`, so
no byte changes). The 320-thread decode boundaries (149..288 rows a rank, both mHC boundaries, all three output forms)
issue every residual and hidden-row load before their first store (`mhc_prefetch.py`, same arithmetic and reduction
order). The indexer's wq_b and each index-source layer's sources stream run at the device's greatest stream priority,
and at those layers the attention wq_b waits for the indexer's wq_b (`index_streams.py`; same kernels on the same
inputs, scheduling only). The Engram layers' wkv keeps stock's quantize and runs DeepGEMM at 241..288 rows a rank and
the fastest exact CuTe-DSL tactic below (`decode_gemm.py`; no route splits K). Each kernel-changing lever proves its
bytes against stock at prepare and raises on any difference. The prefill server and the colocated audit engine serve the
`bundle_v9` path described above.

Three decode-tree riders sit on top of that path, each behind a constant at the top of its own module and each falling
back to the path above on any failure. EVL (`evl.py`, compiled by `moe_decode.py`) passes the evict-first L2 policy to
the decode MegaMoE's routed-expert weight and scale TMA loads; only the cache-policy operand changes. QNORM-DEC
(`qnorm_dec.py`, called from `index_streams.py`) runs the decode attention q-norm and its MXFP8 pair on the PFUSE q-norm
kernel for the q_norm weights whose bytes it proves against stock's Triton kernel at prepare. R-L2 (`rl2.py`, compiled
by `mhc_prefetch.py`) gives the 320-thread decode boundaries' residual loads the evict-first policy, and is proven by the
boundary's own byte proof before it serves; an evict-last policy on their residual stores is built behind
`RL2_STORE_KEEP` and ships switched off.

Seven more decode-tree riders follow the same rule. Each has a constant at the top of its module, proves its bytes at
prepare at the launch form it serves, prints one boot line (`ON`, or `off` with the reason), and falls back to the path
above on any prepare failure. The new launch paths (FFN-PAIR, RSK-4ROW, the glue copies, SEAM-POST) also switch off
after a launch error at serving time; HOIST-T runs inside the boundary's existing launch path.
- FFN-PAIR (`prenorm_pair.py`, called from `mhc_stats.py`; `FFN_PAIR_ON`, `FFN_PAIR_BANDS`, `_FFN_MAX_ROWS`): the
  FFN-side mHC stats of 32..192-row calls on the model's stats stream run the attention side's pair kernel on one CTA
  (a band table) instead of stock's two DeepGEMM launches. The proof covers every layer's FFN parts, NaN-filled outputs
  and every 64-row block edge of the band. When FFN-PAIR-W below passes its checks, its band table (one CTA to 128
  rows, one 2-CTA cluster at 129..320 rows) replaces this one.
- RSK-4ROW (`mhc_stats.py`, `cuda_sinkhorn.py`, prepared from `prenorm_pair.py`; `RSK4_ON`, `RSK4_ROWS`): the
  residual reduce_sinkhorn of 32..320-row calls launches the same kernel at 4 rows a CTA (a quarter of the grid)
  instead of 1, so the stats chain after FFN-PAIR needs fewer free SMs beside MegaMoE. Every (row, column) slice sum
  and every row's sinkhorn is the same instruction sequence; proven against the 1-row launch at every row count of the
  band and with every layer's attention- and FFN-side constants.
- QNORM-PDL, WOBSF-PDL and IDX-W16 (`glue_pdl.py`, `cuda_glue_pdl.py`; `QNORM_PDL_ON`, `WOBSF_PDL_ON`, `IDX_W16_ON`):
  the decode q-norm, the wo_b scale swizzle and the indexer's index_q run programmatic-dependent-launch copies of the
  PFUSE kernels. Only parameters are read before `griddepcontrol.wait`; every input a predecessor writes is read after
  it with plain loads, and dependents are released after the wait. index_q writes the head weights as bf16 (the bits of
  stock's fp32 values, which are exact bf16 values), so the consumers' bf16 conversion copies nothing. Each is proven
  against stock or the PFUSE kernel it copies; the q-norm falls back to the PFUSE kernel, the others to their old path.
- HOIST-T (`hoist_pdl.py`, composed by `mhc_prefetch.py` on the R-L2 text; `HOIST_T_ON`): the 320-thread decode
  boundary kernel releases its PDL successor in every mode after its weight and coefficient loads and before its x / R
  loads and stores (one trigger a mode). It is proven by the boundary's own byte proof, and a failure leaves the R-L2
  text serving. A PDL-in form (`HOIST_PDL_IN_ON`) is built and ships switched off.
- SEAM-POST (`hoist_pdl.py`, called from `mhc_floor.py`; `SEAM_POST_ON`, `SEAM_POST_ROWS`): the three closed seam posts
  of 149..288-row calls (before the Engram layers 1 and 14, and the final post) run the boundary kernel's update
  computation alone, in split_h's FMA order, instead of stock's Triton split_h kernel; proven against split_h at every
  row count of the band.

In this build EVL (`evl.EVL_ON`) and R-L2 (`rl2.RL2_ON`) ship switched off; their code stays in place behind those
switches, and with them off the decode MegaMoE and the 320-thread boundaries compile from the king's texts (HOIST-T then
composes on that text). The three riders below come from a second line on the same base and keep their own switches.
DPF compiles its prefetch variants on whichever MegaMoE kernel text EVL leaves selected, and both print their own boot
line. PEER-RH reuses PEER_READ's tables when PEER-SMALL has already mapped them at prepare, and maps them itself
otherwise.

**PEER-RH** (`peer_lookup.py`, `row_exchange.py`, `rehome_block.py`, `slots.py`; prefill and colocated engines): on a
REHOME-planned step each rank reads only its post-plan Engram rows, each from the shard that owns it (the image's
`engram_gather_peer` over the PEER_READ IPC mappings), instead of the dense gather of every global row, the all-to-all
(or all-reduce) and the 4-way sum. The id gather and the row order are unchanged. Prepare checks source pins of the
image's peer gather and of the existing path's gather kernel and owner reduce, then scans every rank's whole shard
of both Engram layers for the exact-dequant domain (no e4m3 NaN byte, no e8m0 255 byte, scale bytes 10..240 under every
non-zero weight block), so every looked-up value is +-0 or a normal number and both paths give the same bits; it then
runs both paths on every rank and both Engram layers for four REHOME plans covering both reduce branches and compares
the bf16 bits. The tables are attached only if every rank passes every check; otherwise, and on any step it declines,
the existing path serves.

**DPF** (`moe_decode.py`, `cuda_moe_prefill.py`; decode role only; `DPF`, **ships False**: measured slower at the
dominant 240-row bucket, so the existing MegaMoE configs serve): at 25..288 rows a rank the decode MegaMoE kernel's
weight-load warp issues `cp.async.bulk.prefetch.L2` for the routed L1 weights and UE8M0 scales of the first 72 routed L1
tasks (local experts 0-3, about 50 MB a layer) at kernel entry, before the cluster sync and the dependent-launch wait;
no kernel of a decode step writes the routed weights. A prefetch only warms L2 (no shared memory, barrier or value
depends on it), so the output bytes are unchanged. The prefetch variant of each band is compiled at prepare beside the
existing configs; a compile failure or a weight-layout miss leaves the existing kernel serving every call, and with
DPF off the launches and code strings are the existing ones.

**VATTN-T** (`vattn.py`; decode role only): the verify-attention kernel issues `griddepcontrol.launch_dependents` after
its last store, as the compressed-layer attention kernels do, so wo_a's DeepGEMM launch overlaps the attention tail;
wo_a reads nothing before `griddepcontrol.wait`, so no byte changes.

### king-v7 decode riders

Each rider below has its own constant at the top of its module (named in the bullet), proves its bytes at prepare at
the launch form it serves, prints one boot line (`ON`, or `off` with the reason), and leaves the path above serving on
any prepare failure, and, where it adds a launch path, after a launch error at serving time. All of them are scheduling
or launch-form changes: no rider changes any arithmetic, reduction order or output layout. None of them needs a hook
in `dec/slots.py`: each is reached from a prepare or scope the decode slot already calls (`decode_gemm.prepare` /
`routed`, `prenorm_pair.prepare`, `vattn.install` / `routed`).

- **ATTN-PDL-IN** (`vattn.py`; `PDL_IN`): VATTN launches as a programmatic dependent of the kernel before it (a second
  launch attribute, programmatic stream serialisation, beside the cluster attribute), and every thread executes
  `griddepcontrol.wait` right after the CTA-local prologue (shared-memory barrier init, the TMEM allocation and its
  barrier), before any global, async, bulk or tensor-memory access. Nothing before the wait reads or writes memory a
  predecessor can touch, so VATTN's CTAs land and run that prologue while wq_b drains, with the same bytes. The edits
  are anchored replacements on VATTN's published texts (each anchor must match exactly once), so the build has its own
  tag. After VATTN's self-test, an install check runs a captured graph in which the kernel right before each VATTN call
  writes its inputs (3 input sets x 240 rows, 2 replays) and compares every output with the engine's kernel byte for
  byte. The PDL build registers its ops under its own namespace, so if it raises or fails its self-test, speed check
  or graph check, VATTN-T's build (the texts without these edits) is built and checked in its place in the same
  process and only ATTN-PDL-IN is lost. Boot lines: `ATTN-PDL-IN graph check OK` inside `VATTN installed: self-test
  ...`, then `ATTN-PDL-IN ON: ...` (or `ATTN-PDL-IN: the PDL build failed its install checks (...)`, `VATTN installed
  (VATTN-T's build: the PDL build fell back): ...` and `ATTN-PDL-IN off: <reason>`).
- **AWS** (`aws.py`, scoped through `vattn.py`; `AWS_ON`, `AWS_BUCKETS`): at the target-verify graph buckets of 156,
  168, 180 and 192 rows, on the layers with the extra (compressed) cache, the verify attention runs as two VATTN launches
  (rows 0..131, then the rest), and wo_a for the first 132 rows runs on a side stream on 72 SMs (a scoped DeepGEMM SM
  count, restored in `finally`) while the second launch runs; the main stream keeps only the tail rows' wo_a, then joins.
  Attention rows are independent jobs and wo_a has no split-K and a per-row output scale, so the split is bitwise by
  construction; out_fp8, lse and y are row slices of one buffer, while each slice gets its own scale buffer (DeepGEMM
  requires each scale tensor's stride to match its own rows) copied into the engine's buffer, as the engine's prefill
  path does. Install proves, per bucket, VATTN against the engine's kernel at that row count, then the split chain
  against the unsplit chain for out_fp8, out_sf, y and y_sf, eager and from captured graphs (the served form, side-stream
  fork and join included) on 3 rotated input sets, then requires the split chain to be at least 1 us a layer faster with
  L2 flushed; a bucket that fails stays unsplit. First use of each key compares the split attention to the engine's
  kernel and the split wo_a to the unsplit wo_a on the live inputs; a key with no verdict is never captured split. Any
  raise in the split path joins the side stream, reruns the unsplit wo_a, and turns AWS off for the process; if the
  attention backend ever hands back buffers other than the split call's own, that call reruns the engine's own unsplit
  method and AWS turns off. Install refuses a split row that is not a multiple of 4 (slice alignment). Boot line:
  `[dspd aws] ... AWS ON: target-verify buckets [...] ...; per bucket: ...` (or `AWS off: <reason>`).
- **XG-F and XGF-EQ** (`xgf.py`, `moe_route.py`, `moe_front.py`; `XGF_ON`, `XGF_EQ_ON`): the fused MoE front is
  launched with the PDL attribute after the router gate GEMM. The front kernel's first instruction is
  `griddepcontrol.wait`, so this is scheduling only. With XGF-EQ, inside the decode graph's MoE-stream fork only (where
  the fork's stream wait is a full graph edge from the producer of x to the gate) and only for the per-row kernel
  (< 4,096 rows), a copy of the front kernel lets its x-quant warps run ahead of the wait; the router warps and the pad
  CTAs still wait. XGF-EQ **ships False** (`XGF_EQ_ON`): its ordering argument has not yet been checked against
  decoded token ids. Prepare compares each form with the existing launch over the whole output storage (buffers,
  scales, top-k ids and weights, shared scales) in 48 cases at 24 row counts for every shared-expert block size the
  decode tile table serves, each launch behind a primary kernel. Boot line: `bundle-v9 king-v7 XG-F: ON (...); EQ off
  (...)` (or `off (...)` / `EQ ON`).
- **DGT** (`dgt.py`, `dgt_src.py`, `decode_gemm.py`; `DGT_ON`, `DGT_ROWS`, `DGT_BUDGET_S`): wo_a's
  `deep_gemm.fp8_einsum("bhr,hdr->bhd")` runs through a bundle-owned DeepGEMM host library whose 1d1d GEMM kernel issues
  one `cudaTriggerProgrammaticLaunchCompletion` after the TMA-load warp's last load, so the wo_b scale swizzle's PDL
  launch lands over wo_a's tail. The image header's SHA-256 is checked before the one anchored edit; the host path is
  the bundle's existing DeepGEMM host texts (pinned by SHA-256) plus the image's fp8 BMM host entry, under its own JIT
  signature and include root, so neither stock's nor MOE-DECODE's kernels or cache keys change. It serves only the
  proven graph row counts (192, 240 and 288), with stock's SM count, tensor-core utilisation and deterministic
  settings equal to those it was proven under (so AWS's SM-scoped slices run stock); everything else runs stock's
  `fp8_einsum`. Each row count is proven against stock three ways: eager over the whole y / y_sf storage, a refill
  check, and a captured graph of [primary, wo_a, the served swizzle] (captured thread-locally after one eager run of
  the same chain on a side stream). Prepare stops starting new row counts after
  `DGT_BUDGET_S`; a raise turns DGT off. Boot lines: `bundle-v9 king-v7 DGT: ON (...)`, then `DGT: first served wo_a
  at N rows (captured)`.
- **WQKV288** (`decode_gemm.py`; `WQKV288_ON`, `WQKV288_TACTIC`, `WQKV288_ROWS`): the CuTe-DSL wqkv_a tactic of the
  25..216-row band also serves 217..288 rows. The whole served substitution (the decode boundary emitting the
  FlashInfer-layout MXFP8 pair, plus CuTe wqkv_a) is proven bitwise at every row count 217..288 against the existing
  one (the boundary emitting the DeepGEMM pair, plus stock DeepGEMM): updated residual, normalized rows, fp8 bytes and the
  wqkv_a output. Any failure keeps the 216-row band. Boot line: `bundle-v9 king-v7 WQKV288: ON (...)` or `off (...)`.
- **WOBPEAK-DROP** (`decode_gemm.py`; `WOBPEAK_ON = False`): the wo_b table is the one above (CuTe-DSL with the scale
  swizzle to 256 rows, stock DeepGEMM wo_b at 257..288 rows) with its own proof rows; `WOBPEAK_ON = True` restores the
  earlier 257..288-row CuTe band. Boot line: `bundle-v9 king-v7 WOBPEAK-DROP: ON (...)`.
- **TILES_KING** (`moe_route.py`; `TILES_KING`, **ships True**): a data switch for the decode MegaMoE graph tile table.
  True is the original table (the 25..156 and 157..192-row bands included), measured faster at 156..192 rows than the
  single 25..288-row band that False selects; both tables are image tile configurations that MOE-DECODE prepares and
  serves under its existing proof.
- **FFN-PAIR-W** (`prenorm_pair.py`, `cuda_prenorm_pair.py`; `FFN_PAIR_W_ON`, `FFN_PAIR_W_BANDS`): the FFN-side mHC
  stats run the pair kernel on one CTA up to 128 rows and as one 2-CTA cluster at 129..320 rows (the cluster's device
  code is the pair kernel's, PTX-identical), so the two CTAs share the one TPC the decode MegaMoE's two-CTA clusters
  leave free. Prepare first runs a placement probe (every 2-CTA cluster must land on sibling SMs, and SMs / 2 clusters
  must be resident at once, with a bounded spin), then proves the served forms against stock's two DeepGEMM launches on
  every layer's FFN parts at every 64-row block edge and the 240, 264 and 288-row buckets, NaN-filled, in two input
  families. Any failure leaves FFN-PAIR's own band table serving (with STATS-TAIL below still on unless it is
  switched off too). Boot lines: `FFN-PAIR-W: proof launches start`, then
  `bundle-v9 king-v7 FFN-PAIR-W: ON (...)` or `off (...)`; a hung proof launch prints `FFN-PAIR-W: WATCHDOG`.
- **STATS-TAIL** (`prenorm_pair.py`, `cuda_prenorm_pair.py`, `mhc_stats.py`; `STATS_TAIL_ON`): when FFN-PAIR serves an
  FFN-side call, the pair kernel's tail instance also runs the slice reduce and Sinkhorn after the last split (joined
  by `bar.sync` on one CTA, by the cluster barrier on two), so the stats chain beside MegaMoE is one kernel and the
  separate reduce launch is skipped. The tail replays the existing reduce's instruction sequence lane by lane (the same
  inline-asm float instructions and operand order; nothing ptxas could contract), and a device-side guard traps a
  launch without its cluster. Prepare compares all six outputs, NaN-filled, with stock's two GEMMs plus the existing
  reduce at every served CTA count and every layer's FFN constants. Any failure leaves the separate reduce (RSK-4ROW
  where on). Boot line: `bundle-v9 king-v7 STATS-TAIL: ON (...)` or `off (...)`.
- **ATTN-PAIR-WIDTH** (`prenorm_pair.py`; `ATTN_WIDTH_ON`, ships False; `ATTN_PAIR_BANDS`): a per-band CTA count for
  the attention-side pair (1, 2, 4, 8 or 16; rows outside every band keep 16), with a bitwise proof at each served CTA
  count other than 16. Built for an A/B arm; it prints its `off` line in this build.
