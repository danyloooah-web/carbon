# dspd-v51 = dsv41pd.v42 (dspd_v50) + EXM ON

**Status: never run on a GPU.** EXM has never executed on one, in any bundle. Only static and CPU checks ran. No gain
here is measured. `exm.ENABLED` is True, and any failed install check leaves the bundle running as dspd_v50.

EXM is our exact split decode mHC boundary (dspd_w38/dspd/exm.py, exm_*.cuh; the .cuh are byte-identical). A critical
kernel runs on the main stream and the next layer's stats fork onto the engine's `hc_stats_stream`; every reader joins
through `_compute_triplet`. This bundle serves forms A (attention entry, MXFP8 pair, replacing FlashInfer's cake) and
F (FFN entry, bf16, replacing stock's Mega-mHC), stats BURST.

- Scope: the MHC floor (216 rows) is unchanged and never reaches the mega boundary in a decode graph, so the served
  graph buckets are 240, 264 and 288 rows (27 splits). The audit's eager decode forwards can also reach EXM above the
  floor, or below it in a step where a DP rank extends.
- Wiring (`slots.py`): `exm.vote` first in `forward_model`, outside the try; `exm.begin` and `exm.swap` inside the
  `routed` block; `exm.unswap` and `exm.end` in a finally inside it; `exm.install(module)` in `prepare_model` before LEAN.
- `swap` patches `post_fusion_floor.mhc_mega_boundary` (the name the floor calls) and chains to `exact_boundary`;
  `_compute_triplet` is patched on `deepseek_v4_mhc`.
- Estimate only: the EXM lane put the whole stack at 430-540 us a step vs stock (EXM_FEAS 5.1). That covered all 77
  boundaries at every bucket with no floor and no cake, so it overstates this bundle. No v50-specific estimate exists.
- Known risk (RT3, unmeasured): A-side BURST stats CTAs (112 KiB smem + TMEM) can hold SMs that wq_b's (2,2) cluster
  tactic needs at 240-288 rows. dspd_v52 avoids it.

Owed before any use, in order (G-EXM-0..8, struct/w33_exm/LANE_EXM.md, struct/w37e/LANE_W37E.md; none has run):

1. G-EXM-0 sass: stock's mega_mhc SASS vs EXM's K sites and FTZ. It can change the K pins in exm_common.cuh (then
   CUH_SHAS) and it sets STOCK_NVCC, which ships unpinned ("").
2. selftest and selftest-null on this bundle: install must log `EXM self-test OK` and `EXM installed (role decode)`;
   the null role must log `EXM checks OK`, then the deferred install. Any `EXM off` line means this bundle runs as dspd_v50.
3. G-EXM-1 / -2 bits, G-EXM-3 sanitizers, G-EXM-4 graphs (1000 replays beside MegaMoE; hang watchdog).
4. G-EXM-5 timing, then G-EXM-6 the real engine on 4 GPUs (log grader: both the vote and the first-use checks).
5. G-EXM-7 audit role: deferred install and first armed forward inside the watchdog, on this bundle's prepare cost.
6. G-EXM-8 PD-pair A/B against dspd_v50; `BUCKET_ON` (shipped None) and `STATS_GRID` come from it.

---

# dsv41pd.v42 = dsv41-pd070bf + measured decode levers

Built from the teammate's dsv41-pd070bf (shared hotkey 5ERHBMvH). Each lever below was profiled alone on pd070bf
(one-lever variants dsv41pd.v41b-f) and kept only where its kernels measured a gain; every one writes stock's bytes
and checks that at prepare.

- IDXSTREAM (v41b): index-source layers' indexer projections on their own stream in decode / verify
- EDEC48 (v41f): pd071's collective-free Engram decode lookup up to 48 requests a rank, not 24

---

# dsv41-pd010 (MEGA_FRONT + PLACEMENT + INDEX_REUSE + MHC<=80 + SK + ENGRAM_A2A)

DeepSeek-V4.1-Flash-1017 on the sealed prefill/decode arena config (cacheon 1659817f `examples/arena_inputs/dsv41pd`:
TP4/EP4/DP-attention pair, MegaMoE, MegaAttention, DSpark). One change: every MoE layer's routing (sqrtsoftplus +
bias, VL bias on image tokens, top-6, renormalize), the routed-scale multiply and MegaMoE's pre-dispatch (FP8 + UE8M0
quantization of the expert input, the shared expert's paged scales, top-k ids and weights, the buffer's pad rows) run
as one CUDA launch (`dsv41_pd001/mega_front.py`) instead of stock's three. Every byte of DeepGEMM's MegaMoE buffer
is stock's (tested 1..32768 rows); the MoE output through `deep_gemm.fp8_fp4_mega_moe` is bitwise stock's.

- Consumer: the registered `model` slot. `prepare_model` builds private model / layer / MoE views sharing stock's
  weights; each MoE view's `forward` is a generated copy of the sealed `DeepseekV2MoE.forward` that calls
  `dsv41_pd001.mega_forward.forward_mega_moe`, a generated copy of the sealed `forward_mega_moe` /
  `_run_mega_routed` / `run_mega_routed_experts` edited only where marked `dsv41-pd001 MEGA_FRONT`. The same path
  serves captured decode / verify, eager prefill and the eager audit.
- Domain: megamoe on SM 10.x, fp8xfp4 experts, VL-routed sqrtsoftplus top-6 layers, 1..32768 rows. `prepare_model`
  raises unless all 40 layers are in the domain; calls outside it (0 rows, EPLB, routing simulation, routed-expert
  capture) take stock's own three launches.
- Casualty: on served calls, stock's `vision_topk` launch, the routed-scale multiply launch and
  `mega_moe_pre_dispatch`. pd001 v1's TRT-LLM MXFP4 front is gone (no caller under megamoe).
- Evidence: generator/pd001/mega/NOTES.md (~4 us a MoE layer saved per forward at decode rows; up to 49 us at
  32768-row prefill calls). Not submitted.

## dsv41-pd002 PLACEMENT

Prefill MegaMoE ends with its busiest EP rank, and popular experts are not spread evenly by expert id. On the policy-20
replay's big prefill forwards the busiest rank carried 1.29-1.49x the mean routed + shared work in layers 0-19 (the
layers that see every prefill token; 20-39 run only the bounded-replay tail). `dsv41_pd001/placement_table.py`
(fitted by generator/pd002/placement/fit_placement.py from recorded routing; cross-rank slot swaps from stock's
identity, ~5% of the experts change rank) relabels each of those layers: the gate's weight rows and both
score-correction biases are permuted with the experts, so routing, weights and MegaMoE's per-token combine order are
stock's and the output is bitwise stock (EP4 kernel check on recorded routing, 48/48 calls bitwise, stock MegaMoE
15-27% faster on the four most skewed layers at 32768 rows a rank).

- Consumer: the same `model` slot. `prepare_model` validates and records; `forward_model` places once at the end of
  the first eager (non-capturing) forward (`placement.settle`), moving only the changed experts' slices of every
  per-expert tensor over one all-to-all on the EP group, in place.
- Casualty: none in code; stock's identity expert layout on layers 0-19.

## dsv41-pd003 INDEX_REUSE

The co-worker's dsv41pd-c02 (generator/pd1017c02): prefill index layers that map to one compressed source segment reuse one packed FP4 index-K gather per forward (dsv41pd_c02/index_reuse.py, byte-identical). Exact: the reused tuple is the gather stock would repeat.

## dsv41-pd004 MHC

Decode / verify forwards (no extend, not a prefill graph, multi-stream overlap on) at <= 512 rows a rank run stock's decode post-combine mHC kernel instead of the Mega-mHC boundary (dsv41_pd001/post_fusion_floor.py, a marked copy of the sealed _post_fusion), with a DeepGEMM-packed-scale variant for the wqkv_a consumer; the next triplet runs on the stats stream where stock forks it. Measured -12..-14 us a layer at 8-512 rows (generator/pd002/mhc/NOTES.md). Numerics: tolerance-level (residual 99.998% bitwise, 1-ulp rest; next coefficients max rel 2e-3, closer to fp64 than Mega-mHC). Bound only inside the candidate's forward: forward_model swaps deepseek_v4_mhc._post_fusion and restores stock's in `finally`.

## dsv41-pd006 SK

The decode / verify mHC stats (the stats-stream triplet pd004's MHC floor issues at every decode boundary) keep stock's projection launches and run their reduce + sinkhorn on pd001's hc_sinkhorn kernel (dsv41_pd001/cuda_hcsk.py, built by SGLang's load_jit): every float op is inline PTX reproducing Triton 3.8's lowering of stock's _hc_mix_reduce_sinkhorn_kernel, so the triplet is stock's bit for bit (0 mismatches at 1..384 rows, real and synthetic weights, side-stream graph replay; generator/pd002/mhc/sk_jit). 2.2-2.5 us shorter a call. Bound inside the candidate's forward only.

## dsv41-pd010

The MHC floor serves decode / verify boundaries at <= 80 rows a rank only (16 sessions at the 5-token DSpark verify); larger decode batches run stock's Mega-mHC. ENGRAM_A2A (dsv41_pd001/engram_a2a.py): in a SUM_LEN prefill forward where no DP rank holds half of the rows, the Engram DP lookup sends each rank only its own rows (all_to_all_single, split sizes from the host-side token counts) and sums the four chunks, instead of all-reducing the whole buffer in which exactly one shard owns each row; the addends are stock's with at most one non-zero, so the output is bitwise stock's. Every other forward runs stock's lookup.

## dsv41-pd021 (pd013 without W4)

The MHC floor serves decode / verify mHC boundaries of at most 216 rows a rank (verify buckets of up to 36 requests at 6 rows a request), which covers the replay's tail; the prefill-bound steady state (264-288 rows) keeps stock's Mega-mHC. pd013's W4 decode attention is left out: its loader's `tvm_ffi.load_module` is a dynamic import the arena's intake refuses.

## dsv41-pd022 (dsv41pd.v16's byte-identical levers, dsv41_pd001/v16_levers.py)

FlashInfer's mega mHC runs the attention entry's MXFP8 boundary wherever stock's mega_mhc would (decode above the MHC floor, prefill boundaries of 1-8192 rows whose launch takes every SM), writing stock's bytes. In decode and verify forwards wq_b runs stock's CuTe-DSL MXFP8 GEMM on fixed per-bucket tactics instead of the boot's autotuned pick; no tactic splits K, so the bytes are stock's. Both come from the co-worker's dsv41pd.v16.

## dsv41-pd027

INDEX_REUSE is off: every layer view keeps stock's attention module, so each compressed layer gathers its own packed index K.

## dsv41-pd034 ENGRAM_COLS

Each DP rank's Engram shard holds whole hash columns (6 of 24), so in every SUM_LEN prefill forward of at least 8192 rows across the group the lookup gathers only the rank's own columns for every row, all-to-alls each rank its rows, and places the four column blocks (dsv41_pd001/engram_cols.py). The values are the owners' and -0.0 becomes +0.0 as in stock's sum, so the output is bitwise stock's. It supersedes dsv41-pd010's ENGRAM_A2A, which moved whole rows and served balanced forwards only. Smaller, MAX_LEN, gatherv, decode and capturing forwards run stock's lookup.

## dsv41-pd052 SKFUSE

Bitwise-exact replacements of small kernels of the cold 32K-row prefill forward (dsv41_pd001/skfuse.py; kernels in
cuda_skfuse.py, built by SGLang's load_jit at prepare). Inside the candidate's own plain extend forwards (no speculative
mode, not capturing; swapped by `skfuse.routed` and restored in `finally`):

- QNORM: the q_lora RMSNorm + MXFP8 (stock's Triton rmsnorm_mxfp8) on [rows, 1280] bf16 with 16-byte rows, one warp a
  row computing the Triton program's own summation tree (div.full, rsqrt.approx.ftz, mxfp8_epilogue's math).
- DEQ: the sparse-prefill SWA dequant on V4.1 fp8 caches (V41 / V41_NEW).
- SWAIDS: build_swa_token_ids' id gather (int32 ids, int64 full_to_swa), its torch prologue verbatim.
- IDXQ: the indexer's index_q_rope_pack_weights for 32 heads x 128 (a warp a token).
- CMEMO: SparsePrefillChunkCache.combine_compressed on a layer that reuses its index source's top-k (layers 3-7, 9-13,
  15-19 of a ratio-2 forward) returns the source layer's combine, which wrote the same raw top-k and chunk fields into
  the same buffers; any forward_low_ratio_sources call (a source layer's compressor / indexer: the raw top-k's only
  writers in an eager forward) drops it. An exact elimination (no replacement kernel); stock's inside graph scopes.
- WOBSF (prefill_gemm.dg_prefill): the wo_b route's DeepGEMM-packed scales written straight into FlashInfer's 128x4
  bytes (a word permutation of what .contiguous() + block_scale_interleave produce).
- EPLACE / EQUANT (engram_cols): the column placement in 16-byte vectors; where PGEMM routes the Engram wkv for a
  rank's rows, the placement fused with stock's flashinfer_mxfp8_quantize and the pair fed to PGEMM's own GEMM.

Calls outside each domain (other shapes, dtypes, strides, layouts, capture, other forward modes) run stock's function.
Every output is stock's bit for bit (generator/pd052/bench: every byte of every output, cold-round shapes and edge
values). ENGRAM_PRE (pd051) is not stackable with it: EQUANT takes the lookup out of embed_rows.

## dsv41-pd052

dsv41-pd034 (ENGRAM_COLS) with LEAN v2 (pd032) + exact top-k A2 (pd033) + MEGA_FRONT v2 (pd036) + PGEMM v3 prefill GEMM routes for wq_b / wo_b / wqkv_a / indexer wq_b / Engram wkv (g8c sweep) + MoE tile band (pd046) + TILES + FRONT_PDL decode levers (pd040) + DPCNT replay-tail DP counts from the batch's exchanged counts (pd048) + SKFUSE bitwise small-kernel replacements in the cold prefill forward (pd052), each stacked by its generator's apply() (generator/pd037/gen_pd037.py).

## dsv41-pd054 SKFUSE2

Round 2 of the bitwise-exact kernel work in the cold 32K-row prefill forward (dsv41_pd001/skfuse2.py; kernels in
cuda_skfuse2.py, built by SGLang's load_jit at prepare), inside the Engram column exchange (engram_cols.py):

- IDGATHER: the exchange's id gather is the TP group's all_gatherv of each rank's hash ids (stock: dp_gather_replicate,
  an all-reduce of a zero-padded buffer). Same rows, same one collective on every rank (the exchange's domain is
  decided from all-gathered counts, with sum(counts) == the buffer's rows).
- EQ8 / EQ8G: where PGEMM routes the Engram wkv for every rank's count (all-gathered), each sender gathers its own
  columns' rows and quantizes them in one kernel to the MXFP8 codes + scale bytes the receiver's FlashInfer quantize
  makes (stock's gather arithmetic, then FlashInfer's group math; a 32-element group lies in one source's 1536
  columns); the all-to-all moves those bytes (u8 rows of 1584 instead of bf16 rows of 1536) and the receiver places
  them into the wkv GEMM's codes and swizzled scales. pd052's bf16 path (EQUANT / EPLACE) serves every other forward.

Every output is the replaced path's bytes (generator/pd054/bench: every byte, cold-round shapes, adversarial values).

## dsv41-pd055 FMLAP

The eager sparse-prefill layers' MegaAttention core (dsv41_pd001/fmlap.py; kernel sources in cuda_fmlap.py, built by
SGLang's load_jit at prepare): inside the candidate's own extend forwards (no speculative mode, not capturing, outside
breakable-graph and replay-tail scopes; swapped by `fmlap.routed` and restored in `finally`),
`flash_mla_interface.flash_mla_mega_prefill` runs the image kernel's own source (sgl-project/FlashMLA v1.0.0 csrc_legacy
core_attn) with deepseek-ai/FlashMLA #227's H64 items and shared PV completion and sgl-project/FlashMLA #21's predicated
index loads. They reorder waits, loads and barrier arrivals only: out_fp8, out_sf, max_logits and lse are the image
kernel's bytes (generator/pd055/bench: cold-round 32K-row shapes, ragged, partial and adversarial inputs).

It serves calls with use_legacy, no q norm, >= 8,192 query rows (ROWS_MIN), 64 heads x 512, a bf16 [s_kv, 1, 512]
workspace and 128- or 640-wide int32 index rows (the SWA-only layers 0-1 and the compressed layers 2-19 of an eager
forward); every other call (smaller calls, decode, the paged prefill, graph capture and replay such as the layer 20-39
replay tail) runs stock's function.

## dsv41-pd062 CUBLT

cuBLASLt MXFP8 (`torch._scaled_mm`, e8m0 block scales on FlashInfer's swizzled bytes) serves the Engram wkv and the
indexer's wq_b from 1024 rows and wqkv_a at 1024-24575 rows (dsv41_pd001/prefill_gemm.py ROUTES). Outputs are bitwise
with the CuTe tactics they replace; prepare checks that on the registered weights and raises otherwise. Measured with
generator/pd061 (every MXFP8 backend in the image on the prefill linears).

## dsv41-pd065 M462

The eager prefill forwards' MegaMoE on DeepGEMM #462's FP8xFP4 prefill pipeline items (dsv41_pd001/m462.py; host and
kernel sources in cuda_m462.py, built at prepare by torch's cpp_extension and the library's DeepJIT): inside the
candidate's own extend forwards (no speculative mode, not capturing, outside breakable-graph and replay-tail scopes;
swapped by `m462.routed` and restored in `finally`), `deep_gemm.fp8_fp4_mega_moe` runs the image kernel's own source
(sgl-deep-gemm 0.2.3.post2) with #462's store blocks 24 / 8 (a 6th TMA stage), its epilogue and one TMEM prefetch
stage. They regroup epilogue rows and reorder waits only: y is the image kernel's bytes (generator/pd060 bench, every
rank, 8K-32K rows a rank). At the power cap: +2.0% a call at 12K rows a rank, +1.6% at 16K, +1.2% at 32K.

It serves calls with a DP-global token hint of >= 49,152, the image heuristic's BLOCK_M 240, no DG_MEGA_MOE_TILES
override and the prepared buffer / weight / SM facts; every other call (graph capture and replay, decode, the tile
bands, smaller calls) runs the image function. A PD decode engine builds nothing.

## dsv41-pd068 EPEER

sglang#47033's peer_read on pd034's Engram column exchange (dsv41_pd001/epeer.py, kernels in cuda_epeer.py): in the
exchange's domain (all-gathered counts, so every rank decides alike) each rank reads its own rows' 24 table rows in
place from the owning GPUs; a rank PGEMM routes gets the wkv GEMM's MXFP8 codes and swizzled scales in one kernel
(q8_gather_pack's arithmetic, q8_place's layout), any other rank stock's placed bf16 rows: no id gather, no all-to-all. The shards are mapped once, by the first served forward
(legacy CUDA IPC handles of the existing allocations over the TP CPU group; nothing copied); a non-IPC-capable
allocation or a row range that disagrees with the layout raises.

## dsv41-pd071 EDEC

The Engram layers' decode lookup without collectives (dsv41_pd001/edec.py): in decode / verify steps of at most 24
requests a decode rank (a no-extend batch with MAX_LEN padding and a global DP buffer of at most 4 x 24 x the verify
width rows: facts every DP rank shares, IDLE ranks included), each rank reads its own rows' 24 table rows in place from
the owning GPUs with the fork's own `engram_gather_peer` (the SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS kernel: fp32(e4m3) x
2^(e - 127), one bf16 rounding, -0 -> +0), instead of stock's NCCL all-gather of the hash ids, the owned-row gather and
the NCCL reduce-scatter (45-70 us a layer on the verify step's main stream). The shards are mapped once at prepare
(legacy CUDA IPC handles of the existing allocations over the TP CPU group; nothing copied); prepare then compares
stock's gather + NCCL sum with EDEC on ids spanning every hash column, every byte, and raises on any difference. A
prefill-role engine maps and serves nothing; bigger decode batches, prefills and SUM_LEN forwards run pd065's lookup.
