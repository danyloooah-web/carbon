"""dsv41-pd033 TOPK kernel source: generator/pd033/kernel/pd033_topk.cuh verbatim (written under SGLang's JIT cache root by topk_jit._header). Regenerate, do not edit."""

NAME = "pd033_topk.cuh"

SOURCE = r"""// dsv41-pd033 TOPK (generator/pd033/kernel/pd033_topk.cuh): DeepSelect's normal-kernel bf16 top-k, bit for bit, as
// stock's scoring.topk_ragged calls it (int32 indices, unsorted, no values, no page transform, per-row `end`, optional
// per-row index offset, fill value for empty slots). It serves only calls stock's own dispatcher sends to the normal
// (one CTA a row) kernels; the cluster kernel's calls stay on stock (dsv41_pd001/topk_jit.py).
//
// Stock's output, per row r with end = ends[r], off = offsets[r] (0 without offsets), k = 512:
//   - end <= k: [0, end) + off, then `fill` (DeepSelect's shortcut epilogue);
//   - a NaN in [0, end): off in every slot (`take_action_when_have_nan`, abort off);
//   - otherwise the k largest of [0, end) under IEEE order with -0 == +0 (bf16 setp), the ties at the k-th value
//     filled in DeepSelect's visit order, the indices written in visit order. Visit order (DeepSelect
//     common_parts.cuh): a row of at most 32 segments of 512 columns in column order; a longer row the tail
//     [P * 512, end) first (P = (segments - 1) / 8 * 8), then segment (lid * (0xB559EB75 % P) + 0x22262226) % P for
//     lid = 0 .. P - 1, each segment in column order. Verified against the sealed DeepSelect on 1,165 / 1,165 rows of
//     stock logits (generator/pd033/bench, part tiepolicy) and bit for bit by this kernel's acceptance grid.
//   - DeepSelect pads a longer row's tail window to 4096 slots with -inf placeholders indexed P * 512 + slot, visited
//     right after the tail: columns [end, P * 512 + 4096) act as -inf elements there. They are selected only when the
//     k-th value is -inf (never for DeepGEMM's finite logits); this kernel visits them the same way. (A short row's
//     padding follows every real element, so its real -inf ties always come first; it is not visited here.)
//
// One CTA streams a row once, in visit order: a round is 32 consecutive visit positions, warp w taking positions 4w ..
// 4w + 3 (segments A, B, C, D; lane l holds columns 16l .. 16l + 15 of each, one 32-byte load). The next round's
// loads are issued while this round is still being appended: A and B once their values are staged in shared memory
// (before the round's barrier), C and D once theirs are (after A's and B's appends); the round after next is
// prefetched into L2. A column is a candidate when its key is at least the filter key (set.geu.bf16x2: NaN always
// passes); candidates (column, bf16 bits) are appended in visit order (one CTA barrier a round for the warp offsets)
// into a shared buffer of `kCap` entries. The filter key:
//   - permuted rows (kSafety10 > 0): speculative, the bin where a sample of round 0's permuted segments holds
//     kSafety10 / 10 times its share of the row's k keys. Checked at the end of the pass: with at least k candidates
//     the row's k-th key is at or above it, so no output column was filtered out; otherwise the row is streamed again
//     at the guaranteed key below;
//   - otherwise guaranteed: the bin of round 0's k-th largest key (a subset of the row has at most as many keys at or
//     above it as the row);
//   - after an exact select of the buffer (only when it would overflow), the k-th key + 1: later columns come after
//     the k kept ones in visit order, so they cannot win a tie.
// At the end one stable exact select keeps the k largest keys and, at the k-th key, the earliest in visit order (a
// histogram of the buffer's keys finds the k-th key; one ordered pass compacts into the output); it also flags a NaN
// key.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace pd033_topk {

constexpr int kK = 512;                // stock's index_topk; the host entry requires topk == kK
constexpr int kThreads = 256;          // eight warps
constexpr int kWarps = kThreads / 32;
constexpr int kSeg = 512;              // DeepSelect's segment (NUM_ELEMS_PER_SEG)
constexpr int kLaneCols = kSeg / 32;   // a lane's 16 consecutive columns of a segment: one 32-byte load
constexpr int kWarpSegs = 4;           // a warp's consecutive visit positions a round (segments A, B, C, D)
constexpr int kRoundSegs = kWarpSegs * kWarps;  // 32 visit positions a round
constexpr int kNoPermSegs = 32;        // DeepSelect: rows of <= 32 segments are visited in column order
constexpr int kTailSegs = 8;           // DeepSelect: NUM_TAIL_ELEMS / NUM_ELEMS_PER_SEG
constexpr uint32_t kPermAdd = 0x22262226u;    // DeepSelect PERM_ADD_BASE
constexpr uint32_t kPermPrime = 0xB559EB75u;  // DeepSelect PERM_MUL_PRIME
constexpr uint32_t kNegInf2 = 0xFF80FF80u;    // two bf16 -inf: DeepSelect's tail-window placeholder
constexpr uint32_t kNegInfKey = 0x007Fu;      // key of -inf: the lowest non-NaN key
constexpr uint32_t kPosInfKey = 0xFF80u;      // key of +inf: the highest non-NaN key
constexpr int kFineShift = 4;                 // histogram bin = key >> 4 (eight bins a binade)
constexpr int kFine = 1 << (16 - kFineShift);
constexpr int kCoarseShift = 9;               // coarse bin = 32 fine bins
constexpr int kCoarse = 1 << (16 - kCoarseShift);
constexpr int64_t kMaxVocab = int64_t{1} << 23;  // DeepSelect MAX_VOCAB_SIZE (exclusive)
constexpr unsigned kFull = 0xffffffffu;

struct alignas(8) Entry {
  int32_t col;
  uint32_t key;  // the bf16 bits (a select turns them into order keys while it runs)
};

template <int kCap>
struct alignas(16) Smem {
  Entry ent[kCap];         // candidates (column, bf16 bits), in visit order
  uint32_t fine[kFine];    // counted keys by key >> kFineShift
  uint32_t coarse[kCoarse];
  uint4 stage[kWarps][32][4];    // a lane's segments A and B (then C and D) of the round, for indexed reads
  struct {
    Entry ent[kK];               // a select's output, copied back to [0, kK)
    uint32_t hist[256];
  } sel_buf;
  uint32_t warp_tot[2][kWarps];  // a round's per-warp candidate counts, double buffered by parity
  uint32_t sub[kWarps][16];      // the final select: a warp chunk's keys in the k-th key's bin, by low 4 bits
  uint32_t scan[kWarps];
  uint32_t sel[4];               // select results: high bin, count above it; low bin, count above it
  uint32_t tkey;                 // the filter key (thread 0 raises it; readers may see any older value)
  uint32_t t2;                   // its value as two bf16, clamped to [-inf, +inf] (what the filter compares against)
  uint32_t selected;             // a select ran
};

// IEEE order of bf16 payloads with -0 == +0 (bf16 setp compares them equal) as an unsigned 16-bit key.
__device__ __forceinline__ uint32_t key_of(uint32_t b) {
  b = (b == 0x8000u) ? 0u : b;
  return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}

__device__ __forceinline__ uint32_t bits_of_key(uint32_t k) {
  return (k & 0x8000u) ? (k & 0x7FFFu) : (~k & 0xFFFFu);
}

// Thread 0 only: raise the filter key to `key` (and its bf16 pair: keys below -inf / above +inf filter as -inf / +inf).
template <typename S>
__device__ __forceinline__ void raise_filter(S& s, uint32_t key) {
  if (key <= s.tkey) return;
  s.tkey = key;
  const uint32_t b = bits_of_key(min(max(key, 0x007Fu), 0xFF80u));
  *reinterpret_cast<volatile uint32_t*>(&s.t2) = b | (b << 16);
}

__device__ __forceinline__ bool is_nan_key(uint32_t k) {
  return k > kPosInfKey || k < kNegInfKey;
}

// 0xFFFF in each half where a >= b or unordered (NaN).
__device__ __forceinline__ uint32_t geu_bf16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("set.geu.u32.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ void ldg256(const uint16_t* p, uint4 (&v)[2]) {
  asm("ld.global.nc.L1::no_allocate.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(v[0].x), "=r"(v[0].y), "=r"(v[0].z), "=r"(v[0].w), "=r"(v[1].x), "=r"(v[1].y), "=r"(v[1].z),
        "=r"(v[1].w)
      : "l"(p));
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

// Bit j set where column c0 + j passes (key >= the key of t2's halves, or NaN).
__device__ __forceinline__ uint32_t filter16(const uint4 (&v)[2], uint32_t t2) {
  uint32_t m = 0;
  m |= geu_bf16x2(v[0].x, t2) & 0x00010001u;
  m |= geu_bf16x2(v[0].y, t2) & 0x00040004u;
  m |= geu_bf16x2(v[0].z, t2) & 0x00100010u;
  m |= geu_bf16x2(v[0].w, t2) & 0x00400040u;
  m |= geu_bf16x2(v[1].x, t2) & 0x01000100u;
  m |= geu_bf16x2(v[1].y, t2) & 0x04000400u;
  m |= geu_bf16x2(v[1].z, t2) & 0x10001000u;
  m |= geu_bf16x2(v[1].w, t2) & 0x40004000u;
  return (m & 0xFFFFu) | ((m >> 15) & 0xFFFEu);  // the high element of word i: bit 16 + 2i -> 2i + 1
}

// Count a segment's columns in the fine histogram (`valid`: bit j for column c0 + j). -0 counts one bin low (its
// key is not canonicalized here), which keeps the derived filter key a lower bound.
__device__ __forceinline__ void count16(uint32_t* fine, const uint4 (&v)[2], uint32_t valid) {
  const uint32_t w[8] = {v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[1].z, v[1].w};
  if (valid == 0xFFFFu) {  // every column (all but a row's boundary lanes)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t sgn = (w[i] >> 15) & 0x00010001u;
      const uint32_t k2 = w[i] ^ ((sgn * 0x7FFFu) | 0x80008000u);  // two keys
      atomicAdd(&fine[(k2 & 0xFFFFu) >> kFineShift], 1u);
      atomicAdd(&fine[k2 >> (16 + kFineShift)], 1u);
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t sgn = (w[i] >> 15) & 0x00010001u;
    const uint32_t k2 = w[i] ^ ((sgn * 0x7FFFu) | 0x80008000u);
    if ((valid >> (2 * i)) & 1u) atomicAdd(&fine[(k2 & 0xFFFFu) >> kFineShift], 1u);
    if ((valid >> (2 * i + 1)) & 1u) atomicAdd(&fine[k2 >> (16 + kFineShift)], 1u);
  }
}

// Count 4 of a whole segment's 16 columns of this lane (words 0 and 4: columns 0, 1, 8, 9) in the fine histogram.
__device__ __forceinline__ void count_sample(uint32_t* fine, const uint4 (&v)[2]) {
  const uint32_t w[2] = {v[0].x, v[1].x};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t sgn = (w[i] >> 15) & 0x00010001u;
    const uint32_t k2 = w[i] ^ ((sgn * 0x7FFFu) | 0x80008000u);  // two keys (-0 one bin low: harmless for a guess)
    atomicAdd(&fine[(k2 & 0xFFFFu) >> kFineShift], 1u);
    atomicAdd(&fine[k2 >> (16 + kFineShift)], 1u);
  }
}

struct Plan {
  int end;
  int perm;    // P: permuted segments (0: column order)
  int visits;  // visit positions: 8 + P, or the segment count
  int rounds;
  uint32_t mul;     // (0xB559EB75 % P)
  uint32_t step;    // (32 * mul) % P: a warp's next round, 32 lids on
};

__device__ __forceinline__ Plan make_plan(int end) {
  Plan p;
  p.end = end;
  const int nseg = (end + kSeg - 1) / kSeg;
  if (nseg > kNoPermSegs) {
    p.perm = (nseg - 1) / kTailSegs * kTailSegs;
    p.mul = kPermPrime % static_cast<uint32_t>(p.perm);
    p.step = (static_cast<uint32_t>(kRoundSegs) * p.mul) % static_cast<uint32_t>(p.perm);
    p.visits = kTailSegs + p.perm;
  } else {
    p.perm = 0;
    p.mul = 0;
    p.step = 0;
    p.visits = nseg;
  }
  p.rounds = (p.visits + kRoundSegs - 1) / kRoundSegs;
  return p;
}

// A warp's segments, round by round: visit positions 32q + 4w .. 32q + 4w + 3 (-1: none).
struct Walk {
  uint32_t lid_seg;  // permuted rows: the segment of the warp's next lid-based round's first lid
};

__device__ __forceinline__ Walk make_walk(const Plan& p, int warp) {
  Walk wk;
  wk.lid_seg = 0;
  if (p.perm > 0) {
    const int v = kWarpSegs * warp;  // round 0's first position: the tail window for warps 0, 1
    const uint32_t lid = static_cast<uint32_t>(v < kTailSegs ? kRoundSegs + v - kTailSegs : v - kTailSegs);
    wk.lid_seg = (lid * p.mul + kPermAdd) % static_cast<uint32_t>(p.perm);  // lid * mul < 2^28: no overflow
  }
  return wk;
}

// Round q's four segments of this warp. A round's positions are all tail, all lid-based (consecutive lids: each the
// last's segment + mul, mod P) or all column order, as 8 and 32 are multiples of four.
__device__ __forceinline__ void next_segs(const Plan& p, Walk& wk, int q, int warp, int (&sg)[kWarpSegs]) {
  const int v = q * kRoundSegs + kWarpSegs * warp;
  const uint32_t P = static_cast<uint32_t>(p.perm);
  const bool lid = p.perm > 0 && v >= kTailSegs;
  uint32_t seg = wk.lid_seg;
#pragma unroll
  for (int i = 0; i < kWarpSegs; ++i) {
    int g;
    if (p.perm == 0) {
      g = v + i;
    } else if (!lid) {
      g = p.perm + v + i;  // the tail window
    } else {
      g = static_cast<int>(seg);
      seg += p.mul;
      seg = seg >= P ? seg - P : seg;
    }
    sg[i] = v + i < p.visits ? g : -1;
  }
  if (lid) {
    const uint32_t n = wk.lid_seg + p.step;
    wk.lid_seg = n >= P ? n - P : n;
  }
}

// Round q >= 1's four segments: only permuted rows have such rounds (a column-order row has <= 32 segments, all in
// round 0), and they are past the tail window: consecutive lids.
__device__ __forceinline__ void next_segs_lid(const Plan& p, Walk& wk, int q, int warp, int (&sg)[kWarpSegs]) {
  const int v = q * kRoundSegs + kWarpSegs * warp;
  const uint32_t P = static_cast<uint32_t>(p.perm);
  uint32_t seg = wk.lid_seg;
#pragma unroll
  for (int i = 0; i < kWarpSegs; ++i) {
    sg[i] = v + i < p.visits ? static_cast<int>(seg) : -1;
    seg += p.mul;
    seg = seg >= P ? seg - P : seg;
  }
  const uint32_t n = wk.lid_seg + p.step;
  wk.lid_seg = n >= P ? n - P : n;
}

// A whole segment's 16 columns of this lane (rounds after the first are lid-based); no segment (masked out by valid16)
// reads segment 0 instead, so the load is unconditional.
__device__ __forceinline__ void load_full(const uint16_t* in, int seg, uint32_t lane, uint4 (&v)[2]) {
  ldg256(in + max(seg, 0) * kSeg + static_cast<int>(lane) * kLaneCols, v);
}

// One segment's 16 columns of this lane: -inf past `end` (the tail window's placeholders) and for no segment.
__device__ __forceinline__ void load_seg(const uint16_t* in, int seg, int end, uint32_t lane, uint4 (&v)[2]) {
  const int c0 = seg * kSeg + static_cast<int>(lane) * kLaneCols;
  if (seg < 0 || c0 >= end) {
    v[0] = v[1] = make_uint4(kNegInf2, kNegInf2, kNegInf2, kNegInf2);
    return;
  }
  ldg256(in + c0, v);
  if (c0 + kLaneCols > end) {
    uint32_t* w = reinterpret_cast<uint32_t*>(v);
#pragma unroll
    for (int j = 0; j < kLaneCols; ++j) {
      if (c0 + j >= end) {
        const int i = j >> 1;
        w[i] = (j & 1) ? ((w[i] & 0x0000FFFFu) | 0xFF800000u) : ((w[i] & 0xFFFF0000u) | 0x0000FF80u);
      }
    }
  }
}

// The columns of this lane's 16 that the visit includes: every slot of an existing segment of a permuted row (the
// tail window's placeholders included), the columns before `end` of a column-order row.
__device__ __forceinline__ uint32_t valid16(const Plan& p, int seg, uint32_t lane) {
  if (p.perm > 0) return seg >= 0 ? 0xFFFFu : 0u;
  if (seg < 0) return 0u;
  const int n = p.end - (seg * kSeg + static_cast<int>(lane) * kLaneCols);
  return n >= kLaneCols ? 0xFFFFu : (n <= 0 ? 0u : (1u << n) - 1u);
}

// Warp 0: the highest fine bin b with at least `target` counted keys in bins >= b (0 if fewer are counted), and in
// `above` the count in bins > b.
__device__ __forceinline__ uint32_t query_bin(const uint32_t* fine, const uint32_t* coarse, uint32_t lane,
                                              uint32_t target, uint32_t& above_out) {
  const volatile uint32_t* vc = coarse;
  const volatile uint32_t* vf = fine;
  uint32_t c[4];
  uint32_t local = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    c[j] = vc[lane * 4 + j];
    local += c[j];
  }
  uint32_t suffix = local;  // over lanes >= lane
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_down_sync(kFull, suffix, o);
    if (lane + o < 32) suffix += t;
  }
  uint32_t above = suffix - local, cbin = 0, cabove = 0;
  bool hit = false;  // only the lane whose bins cross target
#pragma unroll
  for (int j = 3; j >= 0; --j) {
    if (above < target && above + c[j] >= target) {
      hit = true;
      cbin = lane * 4 + j;
      cabove = above;
    }
    above += c[j];
  }
  const uint32_t who = __ballot_sync(kFull, hit);
  above_out = 0;
  if (who == 0) return 0u;
  const int src = __ffs(who) - 1;
  cbin = __shfl_sync(kFull, cbin, src);
  cabove = __shfl_sync(kFull, cabove, src);
  uint32_t fs = vf[cbin * 32 + lane];  // suffix over the coarse bin's fine bins >= lane
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_down_sync(kFull, fs, o);
    if (lane + o < 32) fs += t;
  }
  const uint32_t ok = __ballot_sync(kFull, cabove + fs >= target);
  const int h = ok ? 31 - __clz(ok) : 0;
  const uint32_t next = __shfl_sync(kFull, fs, min(h + 1, 31));
  above_out = cabove + (h < 31 ? next : 0u);
  return cbin * 32 + h;
}

__device__ __forceinline__ uint32_t warp_sum(uint32_t v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
  return v;
}

// Warp 0: the bin holding the `target`-th largest (1-based) of the 256 counts, bins scanned high to low. The lane
// that finds it writes the bin to out[0] and the count of the bins above it to out[1].
__device__ __forceinline__ void select_bin(const uint32_t* hist, uint32_t target, uint32_t lane, uint32_t* out) {
  uint32_t c[8];
  uint32_t local = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    c[j] = hist[lane * 8 + j];
    local += c[j];
  }
  uint32_t suffix = local;  // inclusive suffix sum over lanes >= lane
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_down_sync(kFull, suffix, o);
    if (lane + o < 32) suffix += t;
  }
  uint32_t above = suffix - local;
#pragma unroll
  for (int j = 7; j >= 0; --j) {
    if (above < target && above + c[j] >= target) {
      out[0] = lane * 8 + j;
      out[1] = above;
    }
    above += c[j];
  }
}

template <int kCap>
__device__ __forceinline__ void keep_compact(Smem<kCap>& s, int n, uint32_t thr, uint32_t quota, uint32_t lane,
                                          uint32_t warp, int32_t* out, int32_t off);

// Warp w's contiguous chunk [lo, hi) of entries [0, n) for the stable compaction.
__device__ __forceinline__ void warp_chunk(int n, uint32_t warp, int& lo, int& hi) {
  const int wchunk = ((n + kWarps - 1) / kWarps + 31) / 32 * 32;
  lo = min(static_cast<int>(warp) * wchunk, n);
  hi = min(lo + wchunk, n);
}

// The write half of keep_compact: this warp's chunk, given the kept keys above thr (gt_b) and the keys equal to it
// (eq_b) in the warps before.
template <int kCap>
__device__ __forceinline__ void keep_write(Smem<kCap>& s, int wlo, int whi, uint32_t thr, uint32_t quota,
                                        uint32_t gt_b, uint32_t eq_b, uint32_t lane, int32_t* out, int32_t off) {
  const uint32_t lt = (1u << lane) - 1u;
  for (int i0 = wlo; i0 < whi; i0 += 32) {
    const int i = i0 + static_cast<int>(lane);
    const bool in = i < whi;
    const Entry en = in ? s.ent[i] : Entry{0, 0u};
    const bool g = in && en.key > thr;
    const bool e = in && en.key == thr;
    const uint32_t mg = __ballot_sync(kFull, g), me = __ballot_sync(kFull, e);
    const uint32_t my_gt = gt_b + __popc(mg & lt), my_eq = eq_b + __popc(me & lt);
    const bool keep = g || (e && my_eq < quota);
    const uint32_t pos = my_gt + min(my_eq, quota);
    if (keep) {
      if (out != nullptr) {
        out[pos] = en.col + off;
      } else {
        s.sel_buf.ent[pos] = Entry{en.col, bits_of_key(en.key)};  // the buffer holds bits
      }
    }
    gt_b += __popc(mg);
    eq_b += __popc(me);
  }
}

// Keep the kK largest keys of entries [0, n) (n >= kK) and, at the kK-th key, the earliest ties, in their order: in
// [0, kK) of the buffer (out == nullptr) or as out[i] = column + off. Returns the kK-th key, or -1 when an entry holds a
// NaN key (nothing is kept or written then). Every thread calls it with the same n. Two
// 8-bit histograms of the buffer find the kK-th key.
template <int kCap>
__device__ __forceinline__ int select_keep(Smem<kCap>& s, int n, uint32_t lane, uint32_t warp, int32_t* out,
                                        int32_t off) {
  const int tid = static_cast<int>(threadIdx.x);
  __syncthreads();  // the appended entries
  for (int i = tid; i < 256; i += kThreads) s.sel_buf.hist[i] = 0;
  __syncthreads();
  bool nan = false;
  for (int i0 = 0; i0 < n; i0 += kThreads) {  // high byte; candidates share few exponents: warp-aggregate
    const int i = i0 + tid;
    const bool in = i < n;
    uint32_t k = 0x8000u;
    if (in) {  // bits to key, kept for the rest of the select
      k = key_of(s.ent[i].key);
      s.ent[i].key = k;
    }
    nan |= is_nan_key(k);
    const uint32_t bin = in ? (k >> 8) : 0xFFFFFFFFu;
    const uint32_t peers = __match_any_sync(kFull, bin);
    if (in && lane == static_cast<uint32_t>(__ffs(peers) - 1))
      atomicAdd(&s.sel_buf.hist[bin], static_cast<uint32_t>(__popc(peers)));
  }
  if (__syncthreads_or(nan)) return -1;
  if (warp == 0) select_bin(s.sel_buf.hist, kK, lane, s.sel);
  __syncthreads();
  const uint32_t high = s.sel[0];
  const uint32_t need = kK - s.sel[1];  // >= 1 keys wanted from bin `high`
  for (int i = tid; i < 256; i += kThreads) s.sel_buf.hist[i] = 0;
  __syncthreads();
  for (int i0 = 0; i0 < n; i0 += kThreads) {  // low byte within `high`; heavy ties share a bin: warp-aggregate
    const int i = i0 + tid;
    const uint32_t k = i < n ? s.ent[i].key : 0u;
    const bool in = i < n && (k >> 8) == high;
    const uint32_t bin = in ? (k & 0xFFu) : 0xFFFFFFFFu;
    const uint32_t peers = __match_any_sync(kFull, bin);
    if (in && lane == static_cast<uint32_t>(__ffs(peers) - 1))
      atomicAdd(&s.sel_buf.hist[bin], static_cast<uint32_t>(__popc(peers)));
  }
  __syncthreads();
  if (warp == 0) select_bin(s.sel_buf.hist, need, lane, s.sel + 2);
  __syncthreads();
  const uint32_t thr = (high << 8) | s.sel[2];
  const uint32_t quota = need - s.sel[3];  // ties kept: >= 1
  keep_compact(s, n, thr, quota, lane, warp, out, off);
  return static_cast<int>(thr);
}

// The final select (select_keep's contract, out != nullptr): a histogram of the buffer's keys (4096 + 128 bins, one
// pass with few conflicts: candidate keys spread over many bins) names the kK-th key's bin and the count above it, one
// more pass resolves its low bits, and the stable compaction writes the output.
template <int kCap>
__device__ __forceinline__ int hist_select(Smem<kCap>& s, int n, uint32_t lane, uint32_t warp, int32_t* out,
                                           int32_t off) {
  const int tid = static_cast<int>(threadIdx.x);
  {
    uint4* f = reinterpret_cast<uint4*>(s.fine);
    for (int i = tid; i < kFine / 4; i += kThreads) f[i] = make_uint4(0u, 0u, 0u, 0u);
  }
  if (tid < kWarps * 16) (&s.sub[0][0])[tid] = 0;
  __syncthreads();  // also publishes the last appends
  bool nan = false;
  for (int i = tid; i < n; i += kThreads) {  // bits to keys, kept for the rest of the select
    const uint32_t k = key_of(s.ent[i].key);
    s.ent[i].key = k;
    nan |= is_nan_key(k);
    atomicAdd(&s.fine[k >> kFineShift], 1u);
  }
  if (__syncthreads_or(nan)) return -1;
  {
    const uint4* f = reinterpret_cast<const uint4*>(s.fine) + 4 * tid;  // 16 fine bins a thread
    uint32_t sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint4 q = f[j];
      sum += q.x + q.y + q.z + q.w;
    }
    sum += __shfl_xor_sync(kFull, sum, 1);
    if ((tid & 1) == 0) s.coarse[tid >> 1] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    uint32_t above;
    const uint32_t b = query_bin(s.fine, s.coarse, lane, kK, above);
    if (lane == 0) {
      s.sel[0] = b;
      s.sel[1] = above;
    }
  }
  __syncthreads();
  const uint32_t b = s.sel[0], above = s.sel[1];
  // one pass over this warp's chunk: keys above bin b, and bin b's keys by their low bits
  int wlo, whi;
  warp_chunk(n, warp, wlo, whi);
  uint32_t abv = 0;
  for (int i = wlo + static_cast<int>(lane); i < whi; i += 32) {
    const uint32_t k = s.ent[i].key;
    abv += (k >> kFineShift) > b;
    if ((k >> kFineShift) == b) atomicAdd(&s.sub[warp][k & ((1u << kFineShift) - 1u)], 1u);
  }
  abv = warp_sum(abv);
  if (lane == 0) s.scan[warp] = abv;
  __syncthreads();
  if (warp == 0) {  // the low bits: the sub-bin holding the kK-th largest, sub-bins scanned high to low
    uint32_t c = 0;
    if (lane < 16) {
#pragma unroll
      for (int w = 0; w < kWarps; ++w) c += s.sub[w][lane];
    }
    uint32_t suf = c;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) {
      const uint32_t t = __shfl_down_sync(kFull, suf, o);
      if (lane + o < 16) suf += t;
    }
    if (lane < 16 && above + suf - c < kK && above + suf >= kK) {
      s.sel[2] = (b << kFineShift) | lane;
      s.sel[3] = kK - (above + suf - c);  // ties kept: >= 1
    }
  }
  __syncthreads();
  const uint32_t thr = s.sel[2], quota = s.sel[3], low = thr & ((1u << kFineShift) - 1u);
  // the warps before this one: keys above thr (above bin b, or in it above `low`) and keys equal to it. Lane l sums
  // warp l >> 2's sub-bins 4 (l & 3) .. 4 (l & 3) + 3.
  const uint32_t wv = lane >> 2, part = lane & 3u;
  uint32_t g = part == 0 ? s.scan[wv] : 0u, e = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t l = part * 4u + j, c = s.sub[wv][l];
    g += l > low ? c : 0u;
    e += l == low ? c : 0u;
  }
  const bool mine = wv < warp;
  const uint32_t gt_b = __reduce_add_sync(kFull, mine ? g : 0u), eq_b = __reduce_add_sync(kFull, mine ? e : 0u);
  keep_write(s, wlo, whi, thr, quota, gt_b, eq_b, lane, out, off);
  return static_cast<int>(thr);
}

// Stable compaction of entries [0, n): the keys above thr and the first `quota` keys equal to it, in order, into
// [0, kK) of the buffer (out == nullptr) or as out[i] = column + off. Warp w owns a contiguous range; lanes walk it
// 32 entries a step. Every thread calls it after a barrier.
template <int kCap>
__device__ __forceinline__ void keep_compact(Smem<kCap>& s, int n, uint32_t thr, uint32_t quota, uint32_t lane,
                                          uint32_t warp, int32_t* out, int32_t off) {
  const int tid = static_cast<int>(threadIdx.x);
  int wlo, whi;
  warp_chunk(n, warp, wlo, whi);
  uint32_t gt = 0, eq = 0;
  for (int i = wlo + static_cast<int>(lane); i < whi; i += 32) {
    const uint32_t k = s.ent[i].key;
    gt += k > thr;
    eq += k == thr;
  }
  gt = warp_sum(gt);
  eq = warp_sum(eq);
  if (lane == 0) s.scan[warp] = (gt << 16) | eq;  // n <= kCap < 65536 a field
  __syncthreads();
  uint32_t before = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) before += (w < static_cast<int>(warp)) ? s.scan[w] : 0u;
  keep_write(s, wlo, whi, thr, quota, before >> 16, before & 0xFFFFu, lane, out, off);
  if (out != nullptr) return;
  __syncthreads();
  for (int i = tid; i < kK; i += kThreads) s.ent[i] = s.sel_buf.ent[i];
  __syncthreads();
}

// Stage this lane's values of a segment pair in shared memory, for indexed reads while appending (lane-private).
template <int kCap>
__device__ __forceinline__ void stage_pair(Smem<kCap>& s, uint32_t warp, uint32_t lane, const uint4 (&vx)[2],
                                           const uint4 (&vy)[2]) {
  uint4* st = s.stage[warp][lane];
  st[0] = vx[0];
  st[1] = vx[1];
  st[2] = vy[0];
  st[3] = vy[1];
}

// Append this lane's candidates of a staged segment pair (m: bit j < 16 for column cx + j of the first segment, bit
// 16 + j for column cy + j of the second) at px, px + 1, ... and py, py + 1, ... in column order, with their bf16 bits.
template <int kCap>
__device__ __forceinline__ void append_pair(Smem<kCap>& s, uint32_t warp, uint32_t lane, uint32_t m, uint32_t px,
                                            uint32_t py, int cx, int cy) {
  const uint16_t* st = reinterpret_cast<const uint16_t*>(s.stage[warp][lane]);
  const uint32_t py_i = py - __popc(m & 0xFFFFu);  // bits run in order: the first segment's first, so the second's
  const int cy_j = cy - 16;                         // i-th bit (counting all) lands at py_i + i
  uint32_t i = 0;
  while (m) {
    const int j = __ffs(m) - 1;
    m &= m - 1;
    const bool in_y = j >= 16;
    s.ent[(in_y ? py_i : px) + i] = Entry{(in_y ? cy_j : cx) + j, static_cast<uint32_t>(st[j])};
    ++i;
  }
}

// kSafety10 > 0: the speculative filter key is the bin with kSafety10 / 10 times the row's expected share of kK keys
// at or above it in the first round's permuted segments. 0: the first round's guaranteed bound only.
template <int kCap, int kMinBlocks, int kSafety10>
__global__ void __launch_bounds__(kThreads, kMinBlocks) topk_kernel(
    const uint16_t* __restrict__ input,
    int64_t stride_input,
    const int32_t* __restrict__ ends,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ output,
    int64_t stride_output,
    int32_t fill) {
  static_assert(kCap >= kK + kWarpSegs * kSeg && kCap < 65536, "a warp's round must fit after a select");
  static_assert(kSafety10 >= 0, "kSafety10: 0 (no speculation) or the speculation factor x10");
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Smem<kCap>& s = *reinterpret_cast<Smem<kCap>*>(smem_raw);
  const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  const int64_t row = blockIdx.x;
  const int end = ends[row];
  if (end <= kK) {  // DeepSelect's shortcut epilogue
    const int32_t off = offsets != nullptr ? offsets[row] : 0;
    for (int i = static_cast<int>(threadIdx.x); i < kK; i += kThreads) output[row * stride_output + i] = i < end ? i + off : fill;
    return;
  }
  const uint16_t* in = input + row * stride_input;
  const Plan plan = make_plan(end);

  uint4 buf[kWarpSegs][2];  // the round's segments A, B, C, D (32 bytes a lane each)
  uint32_t sab, scd;        // their segment numbers + 1, two a register (0: none)
  Walk walk;
  // Round 0 (the first 32 visit positions) in registers, with its segments; the walk then continues at round 1.
  auto load_round0 = [&]() {
    walk = make_walk(plan, static_cast<int>(warp));
    int sg[kWarpSegs];
    next_segs(plan, walk, 0, static_cast<int>(warp), sg);
#pragma unroll
    for (int i = 0; i < kWarpSegs; ++i) load_seg(in, sg[i], end, lane, buf[i]);
    sab = static_cast<uint32_t>(sg[0] + 1) | (static_cast<uint32_t>(sg[1] + 1) << 16);
    scd = static_cast<uint32_t>(sg[2] + 1) | (static_cast<uint32_t>(sg[3] + 1) << 16);
  };
  auto lo_seg = [](uint32_t p) { return static_cast<int>(p & 0xFFFFu) - 1; };
  auto hi_seg = [](uint32_t p) { return static_cast<int>(p >> 16) - 1; };
  auto count_round = [&]() {
#pragma unroll
    for (int i = 0; i < kWarpSegs; ++i) {
      const uint32_t p = i < 2 ? sab : scd;
      count16(s.fine, buf[i], valid16(plan, (i & 1) ? hi_seg(p) : lo_seg(p), lane));
    }
  };
  auto build_coarse = [&]() {
    const uint4* f = reinterpret_cast<const uint4*>(s.fine) + 4 * threadIdx.x;  // 16 fine bins a thread
    uint32_t sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint4 q = f[j];
      sum += q.x + q.y + q.z + q.w;
    }
    sum += __shfl_xor_sync(kFull, sum, 1);
    if ((threadIdx.x & 1u) == 0) s.coarse[threadIdx.x >> 1] = sum;
  };

  // The filter key. Guaranteed: the bin of the k-th largest key of round 0 (a subset of the row has at most as many
  // keys at or above it as the row) -- the row's first pass without speculation, and the second pass with it.
  // Speculative (kSafety10 > 0, permuted rows): the bin where a sample of round 0's permuted segments (4 of each lane's
  // 16 columns; the tail window, the most recent columns, can run higher than the rest) holds kSafety10 / 10 times its
  // share of the row's kK keys. It is checked at the end of the pass: with at least kK candidates the row's k-th key
  // is at or above it, so no output column was filtered out; otherwise the row is filtered again at the guaranteed key.
  const bool spec = kSafety10 > 0 && plan.perm > 0;            // uniform
  auto bound_from_round0 = [&]() {  // buf holds round 0; every thread; ends with the filter key published
    {
      uint4* f = reinterpret_cast<uint4*>(s.fine);
      for (int i = static_cast<int>(threadIdx.x); i < kFine / 4; i += kThreads) f[i] = make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    count_round();
    __syncthreads();
    build_coarse();
    __syncthreads();
    if (warp == 0) {
      uint32_t above;
      const uint32_t t = query_bin(s.fine, s.coarse, lane, kK, above) << kFineShift;
      if (lane == 0) {
        s.tkey = 0u;
        s.t2 = kNegInf2;
        raise_filter(s, t);
        s.selected = 0u;
      }
    }
    __syncthreads();
  };
  load_round0();
  if (spec) {
    {
      uint4* f = reinterpret_cast<uint4*>(s.fine);
      for (int i = static_cast<int>(threadIdx.x); i < kFine / 4; i += kThreads) f[i] = make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    if (kWarpSegs * static_cast<int>(warp) >= kTailSegs) {  // warps 2 .. 7: round 0's 24 permuted (whole) segments
#pragma unroll
      for (int i = 0; i < kWarpSegs; ++i) {
        const uint32_t p = i < 2 ? sab : scd;
        if (((i & 1) ? hi_seg(p) : lo_seg(p)) >= 0) count_sample(s.fine, buf[i]);
      }
    }
    __syncthreads();
    build_coarse();
    __syncthreads();
    if (warp == 0) {
      // the sample: 4 columns x 32 lanes x 24 segments = 3072 of the row's P * 512 permuted columns
      const uint32_t want = (static_cast<uint32_t>(kSafety10) * 3072u + 10u * static_cast<uint32_t>(plan.perm) - 1u) /
                            (10u * static_cast<uint32_t>(plan.perm));
      uint32_t above;
      const uint32_t g = query_bin(s.fine, s.coarse, lane, want, above) << kFineShift;
      if (lane == 0) {
        s.tkey = 0u;
        s.t2 = kNegInf2;
        raise_filter(s, g);
        s.selected = 0u;
      }
    }
    __syncthreads();
  } else {
    bound_from_round0();
  }

  int count = 0;            // entries in the buffer (uniform)
  bool nan = false;
  auto do_select = [&]() {
    const int thr = select_keep(s, count, lane, warp, nullptr, 0);
    if (thr < 0) {
      nan = true;
      return;
    }
    count = kK;
    // Later columns come after the kK kept ones in visit order: only keys above the kK-th can still win. (Readers
    // that miss this store before the next barrier use the older, lower filter key: still a lower bound.)
    if (threadIdx.x == 0) {
      raise_filter(s, static_cast<uint32_t>(thr) + 1u);
      s.selected = 1u;
    }
  };
#pragma unroll 1
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1) {  // the speculative key was above the row's k-th key: again from round 0, at the guaranteed key
      count = 0;
      __syncthreads();  // every thread is done with the buffer and the stage
      load_round0();
      bound_from_round0();
    }
    for (int r = 0; r < plan.rounds; ++r) {
      const bool more = r + 1 < plan.rounds;  // uniform
      const uint32_t t2 = *reinterpret_cast<const volatile uint32_t*>(&s.t2);
      const uint32_t mab = (filter16(buf[0], t2) & valid16(plan, lo_seg(sab), lane)) |
                           ((filter16(buf[1], t2) & valid16(plan, hi_seg(sab), lane)) << 16);
      const uint32_t mcd = (filter16(buf[2], t2) & valid16(plan, lo_seg(scd), lane)) |
                           ((filter16(buf[3], t2) & valid16(plan, hi_seg(scd), lane)) << 16);
      // A and B to the stage, then the next round's A and B loads (their registers are free now)
      if (mab) stage_pair(s, warp, lane, buf[0], buf[1]);
      uint32_t nab = 0, ncd = 0;  // the next round's segments, packed as sab / scd
      if (more) {
        int nx[kWarpSegs];
        next_segs_lid(plan, walk, r + 1, static_cast<int>(warp), nx);
        load_full(in, nx[0], lane, buf[0]);
        load_full(in, nx[1], lane, buf[1]);
        nab = static_cast<uint32_t>(nx[0] + 1) | (static_cast<uint32_t>(nx[1] + 1) << 16);
        ncd = static_cast<uint32_t>(nx[2] + 1) | (static_cast<uint32_t>(nx[3] + 1) << 16);
        if (r + 2 < plan.rounds) {  // round r + 2's segments into L2: lane l takes line l & 7 of segment l >> 3
          const uint32_t P = static_cast<uint32_t>(plan.perm);
          uint32_t t = walk.lid_seg + (lane >> 3) * plan.mul;  // walk is at round r + 2 now; < 4P
          t = t >= 2u * P ? t - 2u * P : t;
          t = t >= P ? t - P : t;
          if ((r + 2) * kRoundSegs + kWarpSegs * static_cast<int>(warp) + static_cast<int>(lane >> 3) < plan.visits) {
            prefetch_l2(in + t * kSeg + (lane & 7u) * 64u);
          }
        }
      }
      // counts of A | B << 16 and C | D << 16 (<= 512 a field), scanned over the warp's lanes
      const uint32_t yab = __popc(mab & 0xFFFFu) | (__popc(mab >> 16) << 16);
      const uint32_t ycd = __popc(mcd & 0xFFFFu) | (__popc(mcd >> 16) << 16);
      uint32_t x1 = yab, x2 = ycd;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t u1 = __shfl_up_sync(kFull, x1, o), u2 = __shfl_up_sync(kFull, x2, o);
        if (lane >= static_cast<uint32_t>(o)) {
          x1 += u1;
          x2 += u2;
        }
      }
      const uint32_t tot1 = __shfl_sync(kFull, x1, 31), tot2 = __shfl_sync(kFull, x2, 31);
      if (lane == 31) s.warp_tot[r & 1][warp] = (tot1 & 0xFFFFu) + (tot1 >> 16) + (tot2 & 0xFFFFu) + (tot2 >> 16);
      __syncthreads();
      const uint32_t ws = lane < kWarps ? s.warp_tot[r & 1][lane] : 0u;
      const uint32_t total = __reduce_add_sync(kFull, ws);
      const uint32_t before = __reduce_add_sync(kFull, lane < warp ? ws : 0u);
      // A, B appended from the stage; then C, D staged, their next loads issued, and appended
      auto append_all = [&](uint32_t base) {
        const int lcol = static_cast<int>(threadIdx.x & 31u) * kLaneCols;
        const uint32_t xa = x1 - yab, xc = x2 - ycd;  // exclusive: A | B << 16, C | D << 16
        const uint32_t ta = tot1 & 0xFFFFu, tab = ta + (tot1 >> 16), tabc = tab + (tot2 & 0xFFFFu);
        if (mab) {
          append_pair(s, warp, lane, mab, base + (xa & 0xFFFFu), base + ta + (xa >> 16), lo_seg(sab) * kSeg + lcol,
                      hi_seg(sab) * kSeg + lcol);
        }
        if (mcd) stage_pair(s, warp, lane, buf[2], buf[3]);
        if (more) {
          load_full(in, lo_seg(ncd), lane, buf[2]);
          load_full(in, hi_seg(ncd), lane, buf[3]);
        }
        if (mcd) {
          append_pair(s, warp, lane, mcd, base + tab + (xc & 0xFFFFu), base + tabc + (xc >> 16),
                      lo_seg(scd) * kSeg + lcol, hi_seg(scd) * kSeg + lcol);
        }
      };
      if (count + static_cast<int>(total) <= kCap) {
        append_all(static_cast<uint32_t>(count) + before);
        count += static_cast<int>(total);
      } else {
        // The buffer would overflow: select first; if the round still does not fit, append warp by warp, selecting
        // whenever the next warp's candidates would not fit (then count > kK, as a warp appends at most 4 * kSeg).
        if (count > kK) do_select();
        if (!nan && count + static_cast<int>(total) <= kCap) {
          append_all(static_cast<uint32_t>(count) + before);
          count += static_cast<int>(total);
        } else if (!nan) {
#pragma unroll 1
          for (int w = 0; w < kWarps; ++w) {
            const int tw = static_cast<int>(__shfl_sync(kFull, ws, w));
            if (count + tw > kCap) {
              do_select();
              if (nan) break;
            }
            if (static_cast<int>(warp) == w) append_all(static_cast<uint32_t>(count));
            __syncthreads();  // the appends before any select
            count += tw;
          }
        }
      }
      if (nan) break;
      sab = nab;
      scd = ncd;
    }
    if (nan || count >= kK) break;  // uniform; a select keeps kK, so count >= kK after any
  }
  int32_t* out = output + row * stride_output;
  const int32_t off = offsets != nullptr ? offsets[row] : 0;
  if (!nan && hist_select(s, count, lane, warp, out, off) >= 0) return;
  for (int i = static_cast<int>(threadIdx.x); i < kK; i += kThreads) out[i] = off;  // NaN: index 0 + offset (stock)
}

}  // namespace pd033_topk

// input: [B, V] bf16 (row stride a multiple of 128 bytes when B > 1, 128-byte aligned, storage covering the last row
// padded to 128 bytes, as DeepSelect's plan requires); ends: [B] int32; offsets: [B] int32 or None; output: [B, topk]
// int32 with unit column stride and a 32-byte aligned row stride; topk == 512; V < 2^23.
template <int kCap, int kMinBlocks, int kSafety10>
struct Pd033TopK {
  static constexpr auto kernel = pd033_topk::topk_kernel<kCap, kMinBlocks, kSafety10>;
  static constexpr std::size_t smem = sizeof(pd033_topk::Smem<kCap>);

  // Opt in to the dynamic shared memory and prefer the all-shared carveout, so kMinBlocks CTAs fit an SM.
  static void init(int device_id) {
    using namespace sglang::host;
    runtime::init_per_device<Pd033TopK>(device_id, true, [&] {
      runtime::set_max_dynamic_smem(kernel, smem);
      runtime::set_smem_carveout(kernel, cudaSharedmemCarveoutMaxShared);
    });
  }

  // CTAs an SM holds for this variant on the current device (after init).
  static int64_t occupancy() {
    using namespace sglang::host;
    int device_id = 0;
    RuntimeDeviceCheck(cudaGetDevice(&device_id));
    init(device_id);
    return static_cast<int64_t>(runtime::get_blocks_per_sm(kernel, pd033_topk::kThreads, smem));
  }

  static void run(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView ends,
      tvm::ffi::Optional<tvm::ffi::TensorView> offsets,
      tvm::ffi::TensorView output,
      int64_t topk,
      int64_t input_storage_bytes,
      int64_t fill) {
    using namespace sglang::host;
    using namespace pd033_topk;
    auto B = SymbolicSize{"batch_size"};
    auto dev = SymbolicDevice{};
    TensorMatcher({B, -1}).with_strides({-1, 1}).with_device<kDLCUDA>(dev).with_dtype<sglang::bf16_t>().ensure_alignment(128).verify(
        input);
    TensorMatcher({B, kK}).with_strides({-1, 1}).with_device<kDLCUDA>(dev).with_dtype<int32_t>().ensure_alignment(32).verify(
        output);
    TensorMatcher({B}).with_device<kDLCUDA>(dev).with_dtype<int32_t>().verify(ends);
    const int32_t* offsets_ptr = nullptr;
    if (offsets.has_value()) {
      TensorMatcher({B}).with_device<kDLCUDA>(dev).with_dtype<int32_t>().verify(offsets.value());
      offsets_ptr = static_cast<const int32_t*>(offsets.value().data_ptr());
    }
    RuntimeCheck(topk == kK, "dsv41-pd033 TOPK serves topk ", kK, ", got ", topk);
    const int64_t batch = B.unwrap();
    const int64_t vocab = input.size(1);
    RuntimeCheck(vocab < kMaxVocab, "dsv41-pd033 TOPK: vocab must be < 2^23, got ", vocab);
    const int64_t stride = batch > 1 ? input.stride(0) : 64;
    RuntimeCheck(stride * 2 % 128 == 0, "dsv41-pd033 TOPK: input row stride must be a multiple of 128 bytes");
    RuntimeCheck(output.stride(0) * 4 % 32 == 0, "dsv41-pd033 TOPK: output row stride must be 32-byte aligned");
    if (batch == 0) return;
    if (vocab > 0) {
      const int64_t padded_row = (vocab * 2 + 127) / 128 * 128;
      RuntimeCheck((batch - 1) * stride * 2 + padded_row <= input_storage_bytes,
                   "dsv41-pd033 TOPK: input storage must include the last row padded to 128 bytes");
    }
    const auto device = dev.unwrap();
    init(device.device_id);
    LaunchKernel(dim3(static_cast<unsigned>(batch)), kThreads, device, smem)
        .launch(
            kernel,
            static_cast<const uint16_t*>(input.data_ptr()),
            stride,
            static_cast<const int32_t*>(ends.data_ptr()),
            offsets_ptr,
            static_cast<int32_t*>(output.data_ptr()),
            static_cast<int64_t>(output.stride(0)),
            static_cast<int32_t>(fill));
  }
};
"""
