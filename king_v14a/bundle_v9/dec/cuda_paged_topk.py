NAME = "paged_topk_impl.cuh"

SOURCE = r"""#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace paged_topk {

constexpr int kK = 512;
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kSeg = 512;
constexpr int kLaneCols = 16;
constexpr int kNoPermSegs = 32;
constexpr int kTailSegs = 8;
constexpr uint32_t kPermAdd = 0x22262226u;
constexpr uint32_t kPermPrime = 0xB559EB75u;
constexpr int kBins = 4096;
constexpr int kBinsPerThread = kBins / kThreads;
constexpr int kMaxSegs = 528;
constexpr int kFine = 32;
constexpr int kCap = 8192;
constexpr int kDepth = 4;
constexpr int kPairDepth = 2;
constexpr int kMaskWords = 8320;
constexpr unsigned kFull = 0xffffffffu;
constexpr uint32_t kZeroBin = 0x7FFu;
constexpr uint32_t kNegInfBin = 0x007u;
constexpr uint32_t kNegInfKey = 0x007Fu;
constexpr uint32_t kPosInfKey = 0xFF80u;
constexpr uint32_t kNegInfBits = 0xFF80u;

struct alignas(16) Smem {
  uint32_t hist[kBins];
  union {
    uint32_t sub[kMaxSegs][kFine / 2];
    uint32_t col[kCap];
  };
  uint16_t key[kCap];
  uint32_t gtb[kMaxSegs];
  uint32_t seg_start[kMaxSegs];
  uint32_t seg_end[kMaxSegs];
  uint32_t off[kMaxSegs];
  uint32_t eqsel[kMaxSegs];
  uint32_t eqc[kMaxSegs];
  uint32_t sel[kK];
  uint32_t cmask[kMaskWords];
  uint32_t fine[kFine];
  uint32_t warp_a[kWarps];
  uint32_t warp_b[kWarps];
  uint32_t bin, above, binc, tsub, need, ncand, ks, nq;
};

__device__ __forceinline__ uint32_t key_of(uint32_t b) {
  return b == 0x8000u ? 0x8000u : ((b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u));
}

__device__ __forceinline__ bool is_nan_key(uint32_t k) {
  return k > kPosInfKey || k < kNegInfKey;
}

__device__ __forceinline__ uint32_t bits_of(uint32_t k) {
  return k >= 0x8000u ? (k & 0x7FFFu) : (~k & 0xFFFFu);
}

template <int kOp>
__device__ __forceinline__ uint32_t cmp2(uint32_t a, uint32_t b) {
  uint32_t d;
  if constexpr (kOp == 0) {
    asm("set.ge.u32.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  } else if constexpr (kOp == 1) {
    asm("set.gt.u32.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  } else if constexpr (kOp == 2) {
    asm("set.eq.u32.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  } else if constexpr (kOp == 3) {
    asm("set.geu.u32.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  } else {
    asm("set.gtu.u32.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  }
  return d;
}

template <int kOp>
__device__ __forceinline__ uint32_t mask16(const uint32_t (&w)[8], uint32_t t) {
  uint32_t m = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) m |= cmp2<kOp>(w[i], t) & (0x00010001u << (2 * i));
  return (m & 0xFFFFu) | ((m >> 15) & 0xFFFEu);
}

__device__ __forceinline__ void ldg256(const uint16_t* p, uint32_t (&w)[8]) {
  asm volatile("ld.global.nc.L1::no_allocate.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]), "=r"(w[6]), "=r"(w[7])
               : "l"(p));
}

__device__ __forceinline__ uint32_t half(const uint32_t (&w)[8], int j) {
  return (j & 1) ? (w[j >> 1] >> 16) : (w[j >> 1] & 0xFFFFu);
}

struct Row {
  int end;
  int perm;
  int nvis;
  uint32_t mul;
  uint32_t nph;
};

__device__ __forceinline__ Row make_row(int end) {
  Row p;
  p.end = end;
  const int nseg = (end + kSeg - 1) / kSeg;
  if (nseg > kNoPermSegs) {
    p.perm = (nseg - 1) / kTailSegs * kTailSegs;
    p.mul = kPermPrime % static_cast<uint32_t>(p.perm);
    p.nvis = p.perm + kTailSegs;
    p.nph = static_cast<uint32_t>(p.nvis * kSeg - end);
  } else {
    p.perm = 0;
    p.mul = 0;
    p.nvis = nseg;
    p.nph = 0;
  }
  return p;
}

__device__ __forceinline__ int seg_at(const Row& p, int v) {
  if (p.perm == 0) return v;
  if (v < kTailSegs) return p.perm + v;
  return static_cast<int>((static_cast<uint32_t>(v - kTailSegs) * p.mul + kPermAdd) % static_cast<uint32_t>(p.perm));
}

__device__ __forceinline__ uint32_t half_mask(const uint32_t (&w)[8], int c, const Row& p, uint32_t lb2, bool ph_in) {
  if (c + kLaneCols <= p.end) return mask16<3>(w, lb2);
  if (c < p.end) {
    const uint32_t valid = (1u << (p.end - c)) - 1u;
    uint32_t m = mask16<3>(w, lb2) & valid;
    if (p.perm > 0 && ph_in) m |= ~valid & 0xFFFFu;
    return m;
  }
  return p.perm > 0 && ph_in ? 0xFFFFu : 0u;
}

__device__ __forceinline__ bool lane_is_first(uint32_t grp) {
  return (threadIdx.x & 31u) == static_cast<uint32_t>(__ffs(grp) - 1);
}

__device__ __forceinline__ uint32_t cut16(uint32_t m, int c, const Row& p) {
  if (c + kLaneCols <= p.end) return m;
  return c < p.end ? m & ((1u << (p.end - c)) - 1u) : 0u;
}

__device__ __forceinline__ void fine_chunk(Smem& s, const uint32_t (&w)[8], int c, bool any, const Row& p,
                                           uint32_t kbase, uint32_t nf, uint32_t lo2, uint32_t hi2) {
  const uint32_t valid = any ? cut16(0xFFFFu, c, p) : 0u;
  const uint32_t v0 = w[0] & 0xFFFFu;
  const uint32_t eq0 = valid != 0u ? mask16<2>(w, v0 * 0x10001u) & valid : 0u;
  const uint32_t f0 = key_of(v0) - kbase;
  const bool in0 = eq0 != 0u && f0 < nf;
  const uint32_t grp = __match_any_sync(kFull, in0 ? f0 : 0xFFFFFFFFu);
  const uint32_t tot = __reduce_add_sync(grp, in0 ? static_cast<uint32_t>(__popc(eq0)) : 0u);
  if (in0 && lane_is_first(grp)) atomicAdd(&s.fine[f0], tot);
  if (eq0 != valid) {
    const uint32_t rest = mask16<0>(w, lo2) & ~mask16<1>(w, hi2) & valid & ~eq0;
#pragma unroll
    for (int j = 0; j < kLaneCols; ++j)
      if ((rest >> j) & 1u) atomicAdd(&s.fine[key_of(half(w, j)) - kbase], 1u);
  }
}

__device__ __forceinline__ int32_t page_map(uint32_t c, const int32_t* table, uint32_t bits) {
  return (__ldg(table + (c >> bits)) << bits) | static_cast<int32_t>(c & ((1u << bits) - 1u));
}

__device__ __forceinline__ void add_fine(Smem& s, int seg, uint32_t f, uint32_t n) {
  atomicAdd(&s.sub[seg][f >> 1], (f & 1u) ? (n << 16) : n);
  atomicAdd(&s.fine[f], n);
}

__device__ __forceinline__ uint32_t block_exclusive(uint32_t v, uint32_t* scratch, uint32_t lane, uint32_t warp) {
  uint32_t inc = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_up_sync(kFull, inc, o);
    if (lane >= static_cast<uint32_t>(o)) inc += t;
  }
  if (lane == 31) scratch[warp] = inc;
  __syncthreads();
  const uint32_t base = __reduce_add_sync(kFull, lane < warp ? scratch[lane] : 0u);
  __syncthreads();
  return base + inc - v;
}

__device__ __forceinline__ void scan2(const uint32_t (&x)[2], uint32_t (&b)[2], uint32_t* scratch, uint32_t lane,
                                      uint32_t warp) {
  b[0] = block_exclusive(x[0] + x[1], scratch, lane, warp);
  b[1] = b[0] + x[0];
}

__device__ __forceinline__ void count_segment(Smem& s, const uint16_t* in, const Row& p, int c0, const uint32_t (&w)[8]) {
  if (c0 + kLaneCols <= p.end) {
    const uint32_t v2 = (w[0] & 0xFFFFu) * 0x10001u;
    bool tie = true;
#pragma unroll
    for (int i = 0; i < 8; ++i) tie = tie && w[i] == v2;
    if (tie) {
      atomicAdd(&s.hist[(v2 >> 4) & 0xFFFu], static_cast<uint32_t>(kLaneCols));
      return;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      atomicAdd(&s.hist[(w[i] >> 4) & 0xFFFu], 1u);
      atomicAdd(&s.hist[w[i] >> 20], 1u);
    }
  } else if (c0 < p.end) {
#pragma unroll
    for (int j = 0; j < kLaneCols; ++j)
      if (c0 + j < p.end) atomicAdd(&s.hist[half(w, j) >> 4], 1u);
  }
}

__device__ __forceinline__ void find_bin(Smem& s, uint32_t nph, uint32_t lane, uint32_t warp) {
  static_assert(kBinsPerThread == 8, "find_bin reads two uint4 a thread");
  uint32_t local[kBinsPerThread];
  uint32_t tsum = 0;
  const bool pos = threadIdx.x >= static_cast<uint32_t>(kThreads / 2);
  {
    const uint4* h = reinterpret_cast<const uint4*>(s.hist) +
                     (pos ? (threadIdx.x - kThreads / 2) * 2 : (0xFF8u - 8u * threadIdx.x) / 4u);
    const uint4 a = h[0], b = h[1];
    if (pos) {
      local[0] = a.x; local[1] = a.y; local[2] = a.z; local[3] = a.w;
      local[4] = b.x; local[5] = b.y; local[6] = b.z; local[7] = b.w;
    } else {
      local[0] = b.w; local[1] = b.z; local[2] = b.y; local[3] = b.x;
      local[4] = a.w; local[5] = a.z; local[6] = a.y; local[7] = a.x;
    }
  }
#pragma unroll
  for (int i = 0; i < kBinsPerThread; ++i) {
    const uint32_t kb = threadIdx.x * kBinsPerThread + i;
    uint32_t c = local[i];
    if (kb == 0x800u) c = 0u;
    if (kb == kZeroBin) c += s.hist[0x000u];
    if (kb == kNegInfBin) c += nph;
    local[i] = c;
    tsum += c;
  }
  uint32_t suf = tsum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_down_sync(kFull, suf, o);
    if (lane + static_cast<uint32_t>(o) < 32u) suf += t;
  }
  if (lane == 0) s.warp_a[warp] = suf;
  __syncthreads();
  const uint32_t hi = __reduce_add_sync(kFull, lane > warp && lane < static_cast<uint32_t>(kWarps) ? s.warp_a[lane] : 0u);
  const uint32_t incl = suf + hi;
  const uint32_t excl = incl - tsum;
  if (incl >= static_cast<uint32_t>(kK) && excl < static_cast<uint32_t>(kK)) {
    uint32_t acc = excl;
#pragma unroll
    for (int i = kBinsPerThread - 1; i >= 0; --i) {
      if (acc + local[i] >= static_cast<uint32_t>(kK)) {
        s.bin = threadIdx.x * kBinsPerThread + i;
        s.above = acc;
        s.binc = local[i];
        break;
      }
      acc += local[i];
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void find256(Smem& s, int base, uint32_t above_in, uint32_t lane, uint32_t warp) {
  const uint32_t c = threadIdx.x < 256u ? s.hist[base + threadIdx.x] : 0u;
  uint32_t suf = c;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_down_sync(kFull, suf, o);
    if (lane + static_cast<uint32_t>(o) < 32u) suf += t;
  }
  if (lane == 0) s.warp_a[warp] = suf;
  __syncthreads();
  const uint32_t hi =
      above_in + __reduce_add_sync(kFull, lane > warp && lane < static_cast<uint32_t>(kWarps) ? s.warp_a[lane] : 0u);
  const uint32_t incl = suf + hi;
  if (c != 0u && incl >= static_cast<uint32_t>(kK) && incl - c < static_cast<uint32_t>(kK)) {
    s.bin = threadIdx.x;
    s.above = incl - c;
  }
  __syncthreads();
}

__device__ void three_pass(Smem& s, const uint16_t* in, const Row& p, const int32_t* table, uint32_t page_bits,
                           int32_t* out, uint32_t lane, uint32_t warp, int counted) {
  if (counted == 0) {
    for (int i = static_cast<int>(threadIdx.x); i < kBins / 4; i += kThreads)
      reinterpret_cast<uint4*>(s.hist)[i] = make_uint4(0u, 0u, 0u, 0u);
  }
  for (int i = static_cast<int>(threadIdx.x); i < p.nvis * (kFine / 8); i += kThreads)
    reinterpret_cast<uint4*>(&s.sub[0][0])[i] = make_uint4(0u, 0u, 0u, 0u);
  if (threadIdx.x < kFine) s.fine[threadIdx.x] = 0u;
  __syncthreads();

  for (int v = counted + static_cast<int>(warp); v < p.nvis; v += kWarps) {
    const int c0 = seg_at(p, v) * kSeg + static_cast<int>(lane) * kLaneCols;
    if (c0 >= p.end) continue;
    uint32_t w[8];
    ldg256(in + c0, w);
    count_segment(s, in, p, c0, w);
  }
  __syncthreads();

  if (__syncthreads_or(threadIdx.x < 16 && s.hist[threadIdx.x < 8 ? 0x7F8u + threadIdx.x : 0xFF0u + threadIdx.x])) {
    bool nan = false;
    for (int seg = static_cast<int>(warp); seg < p.nvis; seg += kWarps) {
      const int c0 = seg * kSeg + static_cast<int>(lane) * kLaneCols;
      if (c0 >= p.end) continue;
      uint32_t w[8];
      ldg256(in + c0, w);
#pragma unroll
      for (int j = 0; j < kLaneCols; ++j) nan |= c0 + j < p.end && (half(w, j) & 0x7FFFu) > 0x7F80u;
    }
    if (__syncthreads_or(nan)) {
      const int32_t index0 = page_map(0u, table, page_bits);
      for (int i = static_cast<int>(threadIdx.x); i < kK; i += kThreads) out[i] = index0;
      return;
    }
  }

  find_bin(s, p.nph, lane, warp);
  const uint32_t tbin = s.bin, above = s.above;
  const uint32_t key_lo = tbin << 4;
  const uint32_t key_hi = tbin == kZeroBin ? 0x8010u : (tbin + 1u) << 4;
  const bool has_hi = key_hi <= kPosInfKey;
  const uint32_t vhi = (has_hi ? bits_of(key_hi) : 0x7F80u) * 0x10001u;
  const uint32_t vlo = (tbin == kNegInfBin ? kNegInfBits : bits_of(key_lo)) * 0x10001u;

  for (int seg = static_cast<int>(warp); seg < p.nvis; seg += kWarps) {
    const int c0 = seg * kSeg + static_cast<int>(lane) * kLaneCols;
    uint32_t gt = 0;
    if (c0 + kLaneCols <= p.end) {
      uint32_t w[8];
      ldg256(in + c0, w);
      const uint32_t mhi = has_hi ? mask16<0>(w, vhi) : 0u;
      const uint32_t inb = mask16<0>(w, vlo) & ~mhi;
      gt = __popc(mhi);
      if (inb == 0xFFFFu && w[0] == (w[0] & 0xFFFFu) * 0x10001u && w[1] == w[0] && w[2] == w[0] && w[3] == w[0] &&
          w[4] == w[0] && w[5] == w[0] && w[6] == w[0] && w[7] == w[0]) {
        add_fine(s, seg, key_of(w[0] & 0xFFFFu) - key_lo, static_cast<uint32_t>(kLaneCols));
      } else if (inb != 0u) {
#pragma unroll
        for (int j = 0; j < kLaneCols; ++j)
          if ((inb >> j) & 1u) add_fine(s, seg, key_of(half(w, j)) - key_lo, 1u);
      }
    } else if (c0 < p.end) {
      uint32_t w[8];
      ldg256(in + c0, w);
#pragma unroll
      for (int j = 0; j < kLaneCols; ++j) {
        if (c0 + j >= p.end && p.perm == 0) continue;
        const uint32_t k = c0 + j < p.end ? key_of(half(w, j)) : kNegInfKey;
        gt += k >= key_hi ? 1u : 0u;
        if (k >= key_lo && k < key_hi) add_fine(s, seg, k - key_lo, 1u);
      }
    } else if (p.perm > 0 && tbin == kNegInfBin) {
      add_fine(s, seg, kNegInfKey - key_lo, kLaneCols);
    }
    gt = __reduce_add_sync(kFull, gt);
    if (lane == 0) s.gtb[seg] = gt;
  }
  __syncthreads();

  if (warp == 0) {
    const uint32_t f = s.fine[lane];
    uint32_t suf = f;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t t = __shfl_down_sync(kFull, suf, o);
      if (lane + static_cast<uint32_t>(o) < 32u) suf += t;
    }
    if (above + suf >= static_cast<uint32_t>(kK) && above + suf - f < static_cast<uint32_t>(kK)) {
      s.tsub = lane;
      s.need = static_cast<uint32_t>(kK) - (above + suf - f);
    }
  }
  __syncthreads();
  const uint32_t tsub = s.tsub, need = s.need;

  {
    uint32_t gt[2] = {0u, 0u}, eq[2] = {0u, 0u}, eb[2], gx[2], fb[2], take[2];
    int seg[2] = {0, 0};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int v = 2 * static_cast<int>(threadIdx.x) + h;
      if (v < p.nvis) {
        seg[h] = seg_at(p, v);
        gt[h] = s.gtb[seg[h]];
#pragma unroll
        for (int i = 0; i < kFine / 2; ++i) {
          const uint32_t word = s.sub[seg[h]][i];
          const uint32_t lo = word & 0xFFFFu, hi = word >> 16;
          const uint32_t f0 = 2u * static_cast<uint32_t>(i), f1 = f0 + 1u;
          gt[h] += (f0 > tsub ? lo : 0u) + (f1 > tsub ? hi : 0u);
          eq[h] = f0 == tsub ? lo : (f1 == tsub ? hi : eq[h]);
        }
      }
    }
    scan2(eq, eb, s.warp_a, lane, warp);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      take[h] = eb[h] >= need ? 0u : min(eq[h], need - eb[h]);
      gx[h] = gt[h] + take[h];
    }
    scan2(gx, fb, s.warp_b, lane, warp);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (2 * static_cast<int>(threadIdx.x) + h < p.nvis) {
        s.off[seg[h]] = fb[h];
        s.eqsel[seg[h]] = take[h];
      }
    }
    __syncthreads();
  }

  const uint32_t tkey = key_lo + tsub;
  const uint32_t vt = bits_of(tkey) * 0x10001u;
  for (int seg = static_cast<int>(warp); seg < p.nvis; seg += kWarps) {
    const int c0 = seg * kSeg + static_cast<int>(lane) * kLaneCols;
    uint32_t gtm = 0, eqm = 0;
    if (c0 + kLaneCols <= p.end) {
      uint32_t w[8];
      ldg256(in + c0, w);
      gtm = mask16<1>(w, vt);
      eqm = mask16<2>(w, vt);
    } else if (c0 < p.end) {
      uint32_t w[8];
      ldg256(in + c0, w);
#pragma unroll
      for (int j = 0; j < kLaneCols; ++j) {
        if (c0 + j >= p.end && p.perm == 0) continue;
        const uint32_t k = c0 + j < p.end ? key_of(half(w, j)) : kNegInfKey;
        gtm |= (k > tkey ? 1u : 0u) << j;
        eqm |= (k == tkey ? 1u : 0u) << j;
      }
    } else if (p.perm > 0 && tkey == kNegInfKey) {
      eqm = 0xFFFFu;
    }
    if (!__any_sync(kFull, (gtm | eqm) != 0u)) continue;
    const uint32_t neq = __popc(eqm);
    uint32_t eq_rank = neq;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t t = __shfl_up_sync(kFull, eq_rank, o);
      if (lane >= static_cast<uint32_t>(o)) eq_rank += t;
    }
    eq_rank -= neq;
    const uint32_t take = s.eqsel[seg];
    uint32_t selm = gtm;
    if (eq_rank < take) {
      const uint32_t n = take - eq_rank;
      selm |= n >= neq ? eqm : eqm & ((1u << __fns(eqm, 0u, static_cast<int>(n) + 1)) - 1u);
    }
    const uint32_t nsel = __popc(selm);
    uint32_t slot = nsel;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t t = __shfl_up_sync(kFull, slot, o);
      if (lane >= static_cast<uint32_t>(o)) slot += t;
    }
    slot = slot - nsel + s.off[seg];
    while (selm != 0) {
      const uint32_t j = __ffs(selm) - 1u;
      selm &= selm - 1u;
      out[slot++] = page_map(static_cast<uint32_t>(c0) + j, table, page_bits);
    }
  }
}

template <bool kDense>
__device__ __forceinline__ void filter(Smem& s, const uint16_t* in, const Row& p, uint32_t (&wa)[kPairDepth][8],
                                       uint32_t (&wb)[kPairDepth][8], int (&cs)[kPairDepth], int sl, uint32_t lb2,
                                       bool ph_in, uint32_t lane, uint32_t warp, int npairs) {
  for (int base = static_cast<int>(warp); base < npairs; base += kWarps * kPairDepth) {
#pragma unroll
    for (int d = 0; d < kPairDepth; ++d) {
      const int c0 = cs[d];
      uint32_t md = 0u;
      if constexpr (kDense) {
        uint32_t ne = 0u;
        if (c0 >= 0) {
          md = cut16(mask16<4>(wa[d], lb2), c0, p) | (cut16(mask16<4>(wb[d], lb2), c0 + kLaneCols, p) << 16);
          ne = __popc(cut16(mask16<2>(wa[d], lb2), c0, p)) + __popc(cut16(mask16<2>(wb[d], lb2), c0 + kLaneCols, p));
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) ne += __shfl_xor_sync(kFull, ne, o);
        if (c0 >= 0 && (lane & 15u) == 0u) s.eqc[2 * (base + d * kWarps) + static_cast<int>(lane >> 4)] = ne;
      } else if (c0 >= 0) {
        md = half_mask(wa[d], c0, p, lb2, ph_in) | (half_mask(wb[d], c0 + kLaneCols, p, lb2, ph_in) << 16);
      }
      const int nv = 2 * (base + d * kWarps + kWarps * kPairDepth) + static_cast<int>(lane >> 4);
      cs[d] = nv < p.nvis ? seg_at(p, nv) * kSeg + sl : -1;
      if (cs[d] >= 0 && cs[d] < p.end) ldg256(in + cs[d], wa[d]);
      if (cs[d] >= 0 && cs[d] + kLaneCols < p.end) ldg256(in + cs[d] + kLaneCols, wb[d]);
      if (c0 >= 0) s.cmask[c0 >> 5] = md;
    }
  }
}

__device__ __noinline__ bool rebound(Smem& s, const uint16_t* in, const Row& p, uint32_t lane, uint32_t warp) {
  for (int i = static_cast<int>(threadIdx.x); i < kBins / 4; i += kThreads)
    reinterpret_cast<uint4*>(s.hist)[i] = make_uint4(0u, 0u, 0u, 0u);
  __syncthreads();
  const uint32_t nw = static_cast<uint32_t>(p.nvis) * (kSeg / 32);
  const uint32_t per = (nw + kThreads - 1) / kThreads;
  const uint32_t lo = min(threadIdx.x * per, nw), hi = min(lo + per, nw);
  for (uint32_t w = lo; w < hi; ++w) {
    uint32_t m = s.cmask[w];
    if (m == 0u) continue;
    const int c = static_cast<int>(w * 32u);
    uint32_t x[8], y[8];
    if (c < p.end) ldg256(in + c, x);
    if (c + kLaneCols < p.end) ldg256(in + c + kLaneCols, y);
    m &= cut16(0xFFFFu, c, p) | (cut16(0xFFFFu, c + kLaneCols, p) << 16);
#pragma unroll
    for (int j = 0; j < kLaneCols; ++j) {
      if ((m >> j) & 1u) atomicAdd(&s.hist[half(x, j) >> 4], 1u);
      if ((m >> (j + kLaneCols)) & 1u) atomicAdd(&s.hist[half(y, j) >> 4], 1u);
    }
  }
  __syncthreads();
  find_bin(s, 0u, lane, warp);
  if (s.above + s.binc > static_cast<uint32_t>(kCap)) return false;
  const uint32_t sbin = s.bin;
  const uint32_t lb2 = (sbin == kNegInfBin ? kNegInfBits : bits_of(sbin << 4)) * 0x10001u;
  for (uint32_t w = lo; w < hi; ++w) {
    const uint32_t m = s.cmask[w];
    if (m == 0u) continue;
    const int c = static_cast<int>(w * 32u);
    uint32_t x[8], y[8];
    if (c < p.end) ldg256(in + c, x);
    if (c + kLaneCols < p.end) ldg256(in + c + kLaneCols, y);
    s.cmask[w] = m & (cut16(mask16<3>(x, lb2), c, p) | (cut16(mask16<3>(y, lb2), c + kLaneCols, p) << 16));
  }
  for (int i = static_cast<int>(threadIdx.x); i < 512; i += kThreads) s.hist[i] = 0u;
  __syncthreads();
  return true;
}

__global__ void __launch_bounds__(kThreads, 2) paged_topk_kernel(
    const uint16_t* __restrict__ input,
    int64_t stride_input,
    const int32_t* __restrict__ ends,
    const int32_t* __restrict__ page_table,
    int64_t stride_table,
    uint32_t page_bits,
    int32_t* __restrict__ output,
    int64_t stride_output,
    int32_t fill,
    int32_t cols) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  const int64_t r = blockIdx.x;
  const int end = ends[r];
  const int32_t* table = page_table + r * stride_table;
  int32_t* out = output + r * stride_output;
  const uint16_t* in = input + r * stride_input;
  for (int i = static_cast<int>(threadIdx.x); i < kBins / 4; i += kThreads)
    reinterpret_cast<uint4*>(s.hist)[i] = make_uint4(0u, 0u, 0u, 0u);
  if (threadIdx.x == 0) s.ncand = 0u;
  if (threadIdx.x < kFine) s.fine[threadIdx.x] = 0u;
  if (end <= kK) {
    for (int i = static_cast<int>(threadIdx.x); i < kK; i += kThreads)
      out[i] = i < end ? page_map(static_cast<uint32_t>(i), table, page_bits) : fill;
    return;
  }
  const Row p = make_row(end);
  const int npairs = (p.nvis + 1) / 2;

  const int sl = static_cast<int>(lane & 15u) * 2 * kLaneCols;
  uint32_t wa[kPairDepth][8], wb[kPairDepth][8];
  int cs[kPairDepth];
#pragma unroll
  for (int d = 0; d < kPairDepth; ++d) {
    const int v = 2 * (static_cast<int>(warp) + d * kWarps) + static_cast<int>(lane >> 4);
    cs[d] = v < p.nvis ? seg_at(p, v) * kSeg + sl : -1;
    if (cs[d] >= 0 && cs[d] < p.end) ldg256(in + cs[d], wa[d]);
    if (cs[d] >= 0 && cs[d] + kLaneCols < p.end) ldg256(in + cs[d] + kLaneCols, wb[d]);
  }
  __syncthreads();

#pragma unroll
  for (int d = 0; d < kPairDepth; ++d) {
    if (cs[d] >= 0) {
      count_segment(s, in, p, cs[d], wa[d]);
      count_segment(s, in, p, cs[d] + kLaneCols, wb[d]);
    }
  }
  __syncthreads();
  find_bin(s, p.nph, lane, warp);

  const int sampled = min(p.nvis, 2 * kWarps * kPairDepth);
  const uint32_t sbin = s.bin;
  const bool dense = (s.above + s.binc) * static_cast<uint32_t>(p.nvis) >
                     static_cast<uint32_t>(kCap) * static_cast<uint32_t>(sampled);
  if (dense) {
    if (sbin <= kNegInfBin || sbin >= 0xFF8u) {
      three_pass(s, in, p, table, page_bits, out, lane, warp, sampled);
      return;
    }
    const uint32_t kbase = sbin == kZeroBin ? 0x7FF0u : sbin << 4;
    const uint32_t nf = sbin == kZeroBin ? 32u : 16u;
    const uint32_t lo2 = bits_of(kbase) * 0x10001u, hi2 = bits_of(kbase + nf - 1u) * 0x10001u;

#pragma unroll 1
    for (int q = 0; q < 2 * kPairDepth; ++q) {
      const int v = 2 * (static_cast<int>(warp) + (q >> 1) * kWarps) + static_cast<int>(lane >> 4);
      const int c = (v < p.nvis ? seg_at(p, v) * kSeg + sl : -kSeg) + (q & 1) * kLaneCols;
      uint32_t w[8];
      if (c >= 0 && c < p.end) ldg256(in + c, w);
      fine_chunk(s, w, c, c >= 0, p, kbase, nf, lo2, hi2);
    }
    if (threadIdx.x == 0) s.ks = kbase;
    __syncthreads();
    if (warp == 0) {
      const uint32_t c = lane < nf ? s.fine[nf - 1u - lane] : 0u;
      uint32_t inc = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t t = __shfl_up_sync(kFull, inc, o);
        if (lane >= static_cast<uint32_t>(o)) inc += t;
      }
      const uint32_t a = s.above;
      if (lane < nf && a + inc >= static_cast<uint32_t>(kK) && a + inc - c < static_cast<uint32_t>(kK))
        s.ks = kbase + nf - 1u - lane;
    }
    __syncthreads();
#pragma unroll
    for (int d = 0; d < kPairDepth; ++d) {
      const int v = 2 * (static_cast<int>(warp) + d * kWarps) + static_cast<int>(lane >> 4);
      cs[d] = v < p.nvis ? seg_at(p, v) * kSeg + sl : -1;
#pragma unroll
      for (int i = 0; i < 8; ++i) wa[d][i] = wb[d][i] = 0u;
      if (cs[d] >= 0 && cs[d] < p.end) ldg256(in + cs[d], wa[d]);
      if (cs[d] >= 0 && cs[d] + kLaneCols < p.end) ldg256(in + cs[d] + kLaneCols, wb[d]);
    }
  }
  const uint32_t lb = dense ? bits_of(s.ks) : sbin == kNegInfBin ? kNegInfBits : bits_of(sbin << 4);
  const uint32_t lb2 = lb * 0x10001u;
  const bool ph_in = !dense && lb == kNegInfBits;
  for (int i = static_cast<int>(threadIdx.x); i < 512; i += kThreads) s.hist[i] = 0u;

  if (dense)
    filter<true>(s, in, p, wa, wb, cs, sl, lb2, ph_in, lane, warp, npairs);
  else
    filter<false>(s, in, p, wa, wb, cs, sl, lb2, ph_in, lane, warp, npairs);
  __syncthreads();
  if (dense) {
    const uint32_t nw = static_cast<uint32_t>(p.nvis) * (kSeg / 32);
    const uint32_t per = (nw + kThreads - 1) / kThreads;
    const uint32_t lo = min(threadIdx.x * per, nw), hi = min(lo + per, nw);
    uint32_t cnt = 0;
    for (uint32_t w = lo; w < hi; ++w) cnt += __popc(s.cmask[w]);
    if (threadIdx.x == 0) s.nq = 0u;
    const uint32_t slot = block_exclusive(cnt, s.warp_a, lane, warp);
    if (threadIdx.x == kThreads - 1) s.ncand = slot + cnt;
    __syncthreads();
    const uint32_t above = s.ncand;
    if (above < static_cast<uint32_t>(kK)) {
      uint32_t e[2], pre[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int v = 2 * static_cast<int>(threadIdx.x) + h;
        e[h] = v < p.nvis ? s.eqc[v] : 0u;
      }
      scan2(e, pre, s.warp_b, lane, warp);
      const uint32_t tq = static_cast<uint32_t>(kK) - above;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t q = pre[h] < tq ? min(e[h], tq - pre[h]) : 0u;
        if (q > 0u) {
          const uint32_t i = atomicAdd(&s.nq, 1u);
          s.off[i] = 2u * threadIdx.x + static_cast<uint32_t>(h);
          s.eqsel[i] = q;
        }
      }
      if (threadIdx.x == kThreads - 1) s.binc = pre[1] + e[1];
      __syncthreads();
      if (above + s.binc < static_cast<uint32_t>(kK)) {
        three_pass(s, in, p, table, page_bits, out, lane, warp, 0);
        return;
      }
      for (uint32_t i = warp; i < s.nq; i += kWarps) {
        const int c = seg_at(p, static_cast<int>(s.off[i])) * kSeg + static_cast<int>(lane) * kLaneCols;
        const uint32_t qn = s.eqsel[i];
        uint32_t w[8], m = 0u;
        if (c < p.end) {
          ldg256(in + c, w);
          m = cut16(mask16<2>(w, lb2), c, p);
        }
        const uint32_t n = __popc(m);
        uint32_t inc = n;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const uint32_t t = __shfl_up_sync(kFull, inc, o);
          if (lane >= static_cast<uint32_t>(o)) inc += t;
        }
        const uint32_t ex = inc - n;
        uint32_t keep = 0u;
        if (ex < qn) {
          const uint32_t take = qn - ex;
          keep = take >= n ? m : m & ((1u << __fns(m, 0u, static_cast<int>(take) + 1)) - 1u);
        }
        const uint32_t hi16 = __shfl_xor_sync(kFull, keep, 1);
        if ((lane & 1u) == 0u) s.cmask[c >> 5] |= keep | (hi16 << 16);
      }
      __syncthreads();
    }
  }

  for (int pass = 0;; ++pass) {
  {

    const uint32_t nw = static_cast<uint32_t>(p.nvis) * (kSeg / 32);
    const uint32_t nch = (nw + 31u) / 32u;
    for (uint32_t ch = warp; ch < nch; ch += kWarps) {
      const uint32_t w = ch * 32u + lane;
      const uint32_t tot = __reduce_add_sync(kFull, w < nw ? static_cast<uint32_t>(__popc(s.cmask[w])) : 0u);
      if (lane == 0u) s.off[ch] = tot;
    }
    __syncthreads();
    const uint32_t ct = threadIdx.x < nch ? s.off[threadIdx.x] : 0u;
    const uint32_t cb = block_exclusive(ct, s.warp_a, lane, warp);
    if (threadIdx.x == kThreads - 1) s.ncand = cb + ct;
    if (threadIdx.x < nch) s.off[threadIdx.x] = cb;
    __syncthreads();
    for (uint32_t ch = warp; ch < nch; ch += kWarps) {
      const uint32_t w = ch * 32u + lane;
      uint32_t m = w < nw ? s.cmask[w] : 0u;
      const uint32_t n = __popc(m);
      uint32_t inc = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t t = __shfl_up_sync(kFull, inc, o);
        if (lane >= static_cast<uint32_t>(o)) inc += t;
      }
      uint32_t slot = s.off[ch] + inc - n;
      if (w < nw) {
        if ((w & 15u) == 0u) s.seg_start[w >> 4] = slot;
        while (m != 0u) {
          const uint32_t j = __ffs(m) - 1u;
          m &= m - 1u;
          if (slot < static_cast<uint32_t>(kCap)) s.col[slot] = w * 32u + j;
          ++slot;
        }
        if ((w & 15u) == 15u) s.seg_end[w >> 4] = slot;
      }
    }
  }
  __syncthreads();
  if (s.ncand <= static_cast<uint32_t>(kCap)) break;
  if (pass > 0 || ph_in || !rebound(s, in, p, lane, warp)) {
    three_pass(s, in, p, table, page_bits, out, lane, warp, 0);
    return;
  }
  }
  const uint32_t nc = s.ncand;

  bool nan = false;
  for (uint32_t q0 = 0; q0 * kThreads < nc; q0 += 4) {
    uint32_t raw[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t i = threadIdx.x + (q0 + static_cast<uint32_t>(q)) * kThreads;
      if (i < nc) {
        const uint32_t c = s.col[i];
        raw[q] = static_cast<int>(c) < p.end ? __ldg(in + c) : kNegInfBits;
      }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t i = threadIdx.x + (q0 + static_cast<uint32_t>(q)) * kThreads;
      if (i < nc) {
        const uint32_t k = key_of(raw[q]);
        nan |= is_nan_key(k);
        s.key[i] = static_cast<uint16_t>(k);
        atomicAdd(&s.hist[k >> 8], 1u);
      }
    }
  }
  if (__syncthreads_or(nan)) {
    const int32_t index0 = page_map(0u, table, page_bits);
    for (int i = static_cast<int>(threadIdx.x); i < kK; i += kThreads) out[i] = index0;
    return;
  }
  find256(s, 0, 0u, lane, warp);
  const uint32_t hb = s.bin, above1 = s.above;
  for (uint32_t i = threadIdx.x; i < nc; i += kThreads) {
    const uint32_t k = s.key[i];
    if ((k >> 8) == hb) atomicAdd(&s.hist[256u + (k & 0xFFu)], 1u);
  }
  __syncthreads();
  find256(s, 256, above1, lane, warp);
  const uint32_t tkey = (hb << 8) | s.bin;
  const uint32_t need = static_cast<uint32_t>(kK) - s.above;

  const uint32_t per = (nc + kThreads - 1) / kThreads;
  const uint32_t lo = min(threadIdx.x * per, nc), hi = min(lo + per, nc);
  uint32_t mine = 0;
  for (uint32_t i = lo; i < hi; ++i) {
    const uint32_t k = s.key[i];
    mine += k > tkey ? 0x10000u : (k == tkey ? 1u : 0u);
  }
  const uint32_t before = block_exclusive(mine, s.warp_a, lane, warp);
  {
    uint32_t run = before;
    for (uint32_t i = lo; i < hi; ++i) {
      const uint32_t k = s.key[i];
      const uint32_t seg = s.col[i] / kSeg;
      if (i == s.seg_start[seg]) s.gtb[seg] = run;
      run += k > tkey ? 0x10000u : (k == tkey ? 1u : 0u);
      if (i + 1u == s.seg_end[seg]) s.eqc[seg] = run;
    }
  }
  __syncthreads();
  {
    uint32_t ge[2] = {0u, 0u}, gb[2];
    int seg[2] = {0, 0};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int v = 2 * static_cast<int>(threadIdx.x) + h;
      if (v < p.nvis) {
        seg[h] = seg_at(p, v);
        if (s.seg_end[seg[h]] != s.seg_start[seg[h]]) ge[h] = s.eqc[seg[h]] - s.gtb[seg[h]];
      }
    }

    scan2(ge, gb, s.warp_a, lane, warp);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t eq = ge[h] & 0xFFFFu, eq_before = gb[h] & 0xFFFFu;
      if (2 * static_cast<int>(threadIdx.x) + h < p.nvis) {
        s.off[seg[h]] = (gb[h] >> 16) + min(eq_before, need);
        s.eqsel[seg[h]] = eq_before >= need ? 0u : min(eq, need - eq_before);
      }
    }
  }
  __syncthreads();
  {
    uint32_t run = before;
    for (uint32_t i = lo; i < hi; ++i) {
      const uint32_t k = s.key[i];
      const uint32_t c = s.col[i];
      const uint32_t seg = c / kSeg;
      const uint32_t rel = run - s.gtb[seg];
      const uint32_t gr = rel >> 16, er = rel & 0xFFFFu, take = s.eqsel[seg];
      if (k > tkey) {
        s.sel[s.off[seg] + gr + min(er, take)] = c;
        run += 0x10000u;
      } else if (k == tkey) {
        if (er < take) s.sel[s.off[seg] + gr + er] = c;
        run += 1u;
      }
    }
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < kK; i += kThreads) out[i] = page_map(s.sel[i], table, page_bits);
}

}

struct PagedTopK {
  static constexpr auto kernel = paged_topk::paged_topk_kernel;
  static constexpr std::size_t smem = sizeof(paged_topk::Smem);
  static_assert(smem <= 115712, "paged_topk: two CTAs a SM need at most 113 KB of shared memory each");

  static void init(int device_id) {
    using namespace sglang::host;
    runtime::init_per_device<PagedTopK>(device_id, true, [&] {
      runtime::set_max_dynamic_smem(kernel, smem);
      runtime::set_smem_carveout(kernel, cudaSharedmemCarveoutMaxShared);
    });
  }

  static void run(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView ends,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView output,
      int64_t page_size,
      int64_t input_storage_bytes,
      int64_t fill) {
    using namespace sglang::host;
    auto B = SymbolicSize{"batch_size"};
    auto dev = SymbolicDevice{};
    TensorMatcher({B, -1}).with_strides({-1, 1}).with_device<kDLCUDA>(dev).with_dtype<sglang::bf16_t>().ensure_alignment(128).verify(
        input);
    TensorMatcher({B, paged_topk::kK}).with_strides({-1, 1}).with_device<kDLCUDA>(dev).with_dtype<int32_t>().ensure_alignment(32).verify(
        output);
    TensorMatcher({B}).with_device<kDLCUDA>(dev).with_dtype<int32_t>().verify(ends);
    TensorMatcher({B, -1}).with_strides({-1, 1}).with_device<kDLCUDA>(dev).with_dtype<int32_t>().verify(page_table);
    RuntimeCheck(page_size > 0 && (page_size & (page_size - 1)) == 0, "paged_topk: page_size must be a power of 2");
    uint32_t bits = 0;
    while ((int64_t{1} << bits) < page_size) ++bits;
    const int64_t batch = B.unwrap();
    const int64_t vocab = input.size(1);
    RuntimeCheck(vocab <= int64_t{paged_topk::kMaskWords} * 32, "paged_topk: row too wide: ", vocab);
    RuntimeCheck(page_table.size(1) * page_size >= vocab, "paged_topk: the page table must cover the row");
    const int64_t stride = batch > 1 ? input.stride(0) : (vocab + 63) / 64 * 64;
    RuntimeCheck(stride % 16 == 0 && stride >= (vocab + 15) / 16 * 16, "paged_topk: input row stride must cover 16-column loads");
    if (batch == 0) return;
    if (vocab > 0) {
      RuntimeCheck((batch - 1) * stride * 2 + (vocab + 15) / 16 * 32 <= input_storage_bytes,
                   "paged_topk: input storage must include the last row padded to 16 columns");
    }
    const auto device = dev.unwrap();
    init(device.device_id);
    LaunchKernel(dim3(static_cast<unsigned>(batch)), paged_topk::kThreads, device, smem)
        .launch(
            kernel,
            static_cast<const uint16_t*>(input.data_ptr()),
            stride,
            static_cast<const int32_t*>(ends.data_ptr()),
            static_cast<const int32_t*>(page_table.data_ptr()),
            static_cast<int64_t>(page_table.stride(0)),
            bits,
            static_cast<int32_t*>(output.data_ptr()),
            static_cast<int64_t>(output.stride(0)),
            static_cast<int32_t>(fill),
            static_cast<int32_t>(vocab));
  }
};
"""
