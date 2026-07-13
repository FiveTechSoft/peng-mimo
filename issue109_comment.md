# Draft comment for JustVugg/colibri#109 — HONEST VERSION (post after user approval)

---

We tested exactly this in [peng](https://github.com/fivetechsoft/peng-mimo) (the MiMo-V2.5 fork from #49) and have measured results on the full container — mixed verdict, both halves useful.

**TL;DR: int4 containers compress ~28% losslessly (the "int4 is incompressible" intuition doesn't hold — measured), but on our reference box the engine gets ~18% slower streaming compressed, because the decompress CPU has nowhere to hide. Size win is real; speed win is not (here).**

## 1. int4 does compress — full-container measurement

| what | measured |
|---|---|
| full repack, 17 shards, real MiMo-V2.5 311B int4 | **151.8 → 109.0 GiB (71.8%)**, zstd level 1 |
| per-shard range | 70.2–74.8% |
| byte entropy of expert data | 5.89 bits/byte → 73.7% order-0 floor (zstd-1 sits on it; higher levels gain ~nothing) |
| zero bytes / padding | 0.0% — genuine symbol skew, not filler |
| correctness | every frame decompressed + byte-compared vs source; engine output **bit-exact** (tiny oracle TF 31/32 + greedy identical; full-scale TEMP=0 produces identical text and expert counts) |

Why it compresses: quantization removes *precision* redundancy, entropy coding exploits *distribution* redundancy — orthogonal. Trained int4 codes cluster near zero → skewed 4-bit symbols. That's 28% on top of int4, lossless: **effectively 2.87 bits/weight with zero quality cost** — smaller than a lossy int3 container would be.

One caveat that may explain "we've seen zstd do nothing": our tiny *random-init* test containers compress to only ~89–93%. Random weights have no skew. If the zstd check ran on synthetic fixtures, that's the discrepancy. On any real trained shard: `zstd -1 -c out-00004.safetensors | wc -c` is a 5-minute check.

## 2. But streaming compressed is SLOWER on our box — and why that matters for #81

Same-day §37-protocol pairs (RTX 3060, 16 cores, WSL2, 2.75 GB/s NVMe, warm):

| container | tok/s (median of runs) | expert-disk stall |
|---|---|---|
| int4 uncompressed | **0.38** (0.34–0.46) | 7.7–8.2 s / 24 tok |
| int4+zstd | 0.31 (0.30–0.36) | 13.9–15.6 s |

The naive expectation — "cores are idle during I/O waits, decompress hides there for free" — is false once you have a load/compute overlap pipeline: the cores are already full of expert matmul during loads. zstd decompress adds ~1.05 s/token of *new* CPU work (4 GB/token at ~0.95 GB/s/thread) on a box whose host matmul already needs those cores. We tried loader-priority changes and bigger decompress pools; nothing recovers it — it's a capacity problem, not a scheduling problem.

**This directly stress-tests the assumption in #81** (rotation-preconditioned int2: "the inverse transform compute hides in the shadow of the disk wait"). Our measurement says: on a core-limited box with overlap already working, there is no shadow. The rotation dequant would need the same budget check. (int2 still wins on bytes, of course — and int_k + zstd stack: whatever skew survives quantization, the entropy coder takes.)

## 3. Where the compressed container does win

- **Distribution:** 26% less download/storage. We're publishing the compressed container alongside the original.
- **RAM-rich configs:** if experts get pinned to RAM at boot (128 GB box), decompress happens once at load — zero steady-state cost, 43 GB less disk.
- **Core-rich / native-Linux boxes:** without the WSL2 I/O CPU tax there may be spare cores; the sign could flip. Untested — if someone with native Linux + fast NVMe wants to try, the tooling is ready.

## Implementation (~200 lines, engine-agnostic, colibri's st.h lineage)

Standard safetensors structure; each tensor's data region is one zstd-1 frame; per-tensor `"nb"` = uncompressed size; `__metadata__.peng_zstd="1"`. Legacy containers take the old path byte-for-byte.

- repack + verify: [`c/tools/repack_zstd.py`](https://github.com/fivetechsoft/peng-mimo/blob/peng/c/tools/repack_zstd.py)
- reader: `c/st.h` (`znbytes` + thread-local `ZSTD_DCtx`); hot path: `expert_load` in `c/mimo.c` (coalesced 3-frame pread stays one pread; decompress lands in the same slab the QT views point into)
- full write-up incl. the failed speed experiments: findings.md §41

Happy to help port it if there's interest — for colibri the distribution/size win applies as-is; whether streaming wins depends on the host's spare-core budget, and now there's a measured way to predict it.

---
# END DRAFT — post with:
# gh issue comment 109 --repo JustVugg/colibri --body-file <stripped file>
