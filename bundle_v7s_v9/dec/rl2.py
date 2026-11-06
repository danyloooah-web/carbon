"""king-v2e R-L2: L2 eviction-priority operands on the decode mHC residual stream inside HOIST.

At 149..288 rows every decode / verify hyper-connection boundary runs HOIST (cuda_mhc_prefetch.py, the king's 320-thread
post-combine-norm kernel, modes 0 / 1 / 2). It reads the residual R [rows, 4, 5120] once, writes the next residual R'
once, and R' is next read by the following boundary after the sublayer in between (attention, or MegaMoE streaming
~1.8 GB of expert weights a rank). R-L2 changes only the cache-policy operand of those two accesses:
- the four R loads of each thread (`rh[j][r]`, their last use: the next boundary reads R', not R) become
  `ld.weak.global.L1::no_allocate.L2::cache_hint.v4.b32` with the evict-first policy, the exact instruction form and
  policy value DeepGEMM's own Mega-mHC epilogue uses for its residual re-reads (ptx::ld_evict_first);
- the R' stores (`updated`) become `st.global.L2::cache_hint.v4.b32` with the evict-last policy, so R' is preferred
  over streamed lines until the next boundary reads it. This half ships switched off (RL2_STORE_KEEP, fix F1): with
  EVL serving it adds little, and its retention and crowding behaviour are unverified.
The values loaded and stored, their addresses and their order are the king's, so every output byte is unchanged. The
HOIST byte proof (mhc_prefetch._prove: stock mhc_post_combine_norm_decode at 15 row counts, both quant modes, and the
king's packed DeepGEMM-pair kernel) runs on the R-L2 text before it serves; any failure (anchor, compile, load, proof)
puts back the king's HOIST text, which is then proven as in the king.

Not covered: the 640-thread boundaries at <= 148 rows (stock's mhc_post_combine_norm_decode and the king's packed
kernel). They keep the king's code.
"""
from __future__ import annotations

# The lead's switch. True: HOIST compiles with the R-L2 operands below. False: the king's HOIST text, unchanged.
RL2_ON = False  # king-v6t: off (v2e profile: EVL broke the draft MoE JIT; R-L2 neutral)
# Sub-switches (only read when RL2_ON): the evict-first R loads and the evict-last R' stores.
RL2_LOAD_LAST_USE = True
# king-v2e FIX F1: off by default. EVL already makes the MegaMoE weight stream the first eviction candidate, which is
# what protects R' between boundaries (the M4 verifier: not additive with EVL); whether evict-last retains without a
# persisting-L2 set-aside is unverified; and if it does retain and later accesses do not demote the lines, dead R'
# lines (9.8-11.8 MB a boundary) would crowd MegaMoE's own evict-last activations. True puts the store half back.
RL2_STORE_KEEP = False

_NAMESPACE = "namespace sglang {\nstruct MhcHoistParams {"
_NAMESPACE_RL2 = """namespace sglang {
namespace rl2 {
// king-v2e R-L2: L2 eviction-priority operands on the residual stream; the values moved are unchanged.
constexpr uint64_t kEvictFirst = 0x12F0000000000000ull;  // createpolicy.fractional.L2::evict_first.b64 1.0
constexpr uint64_t kEvictLast = 0x14F0000000000000ull;   // createpolicy.fractional.L2::evict_last.b64 1.0

template <typename V>
__device__ __forceinline__ void load_last_use(V& v, const void* base, int64_t offset) {
  static_assert(sizeof(V) == 16, "R-L2 moves 16-byte vectors");
  const uint4* src = reinterpret_cast<const uint4*>(base) + offset;
  uint32_t* dst = reinterpret_cast<uint32_t*>(v.data());
  asm volatile("ld.weak.global.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
               : "l"(src), "l"(kEvictFirst));
}

template <typename V>
__device__ __forceinline__ void store_keep(const V& v, void* base, int64_t offset) {
  static_assert(sizeof(V) == 16, "R-L2 moves 16-byte vectors");
  uint4* dst = reinterpret_cast<uint4*>(base) + offset;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(v.data());
  asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;"
               :: "l"(dst), "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]), "l"(kEvictLast)
               : "memory");
}
}  // namespace rl2
struct MhcHoistParams {"""
_LOAD = "        rh[j][r].load(p.residual + row * 20480 + r * 5120, tid + j * Threads);\n"
_LOAD_RL2 = "        rl2::load_last_use(rh[j][r], p.residual + row * 20480 + r * 5120, tid + j * Threads);\n"
_STORE = "      updated.store(p.residual_out + row * 20480 + i * 5120, vid);\n"
_STORE_RL2 = "      rl2::store_keep(updated, p.residual_out + row * 20480 + i * 5120, vid);\n"

_STATE = {"on": False, "why": "not prepared"}


def edits() -> tuple:
    out = [(_NAMESPACE, _NAMESPACE_RL2)]
    if RL2_LOAD_LAST_USE:
        out.append((_LOAD, _LOAD_RL2))
    if RL2_STORE_KEEP:
        out.append((_STORE, _STORE_RL2))
    return tuple(out)


def parts() -> str:
    halves = (("R loads evict-first", "R loads", "RL2_LOAD_LAST_USE", RL2_LOAD_LAST_USE),
              ("R' stores evict-last", "R' stores", "RL2_STORE_KEEP", RL2_STORE_KEEP))
    names = [on_name for on_name, _, _, on in halves if on]
    off = [f"{off_name} as in the king, rl2.{switch} is off" for _, off_name, switch, on in halves if not on]
    return ", ".join(names + off) if names else "no operand"


def hoist_source(text: str):
    """The R-L2 HOIST text, or None when no operand is switched on. Raises unless every anchor matches once."""
    chosen = edits()
    if len(chosen) == 1:
        return None
    for old, new in chosen:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"an R-L2 anchor matches {count} times in the HOIST source, not once")
        text = text.replace(old, new)
    return text


def mark(on: bool, why: str) -> None:
    _STATE["on"] = bool(on)
    _STATE["why"] = why


def line() -> str:
    return f"bundle-v9 king-v2e R-L2: {'ON' if _STATE['on'] else 'off'} ({_STATE['why']})"
