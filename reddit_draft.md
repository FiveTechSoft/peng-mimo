# Reddit draft — r/LocalLLaMA (post after HF upload completes; user posts manually)

## Title options (pick one)

1. **int4 MoE weights compress 28% losslessly (2.87 bits effective, bit-exact) — but streaming them made inference 18% slower. Full measurements.**
2. int4 is NOT incompressible: 152→109 GB lossless on a 311B MoE. The catch: decompression has nowhere to hide on a busy box.
3. We compressed a 311B model's int4 weights 28% with zero quality loss — and honest numbers on why it didn't make us faster.

(#1 recommended: both hooks in one line, numbers up front.)

## Body

---

**TL;DR:** Quantized int4 weights still have ~26–30% of lossless redundancy (entropy coding, zstd level 1). We repacked Xiaomi MiMo-V2.5 311B from 152 GB → 109 GB, **bit-exact** — the engine reads compressed shards natively and produces identical output. But on our 16-core box, streaming inference got ~18% *slower*, because zstd decompression is real CPU work and the cores were already busy. Both halves measured, tooling open.

## Context

[peng](https://github.com/fivetechsoft/peng-mimo) runs Xiaomi's MiMo-V2.5 (311B MoE, 15B active) on a desktop: dense layers resident in RAM as int4, the 12,032 routed experts streamed from NVMe on demand (fork of [colibri](https://github.com/JustVugg/colibri), which does the same for GLM-5.2 744B). Cold token ≈ 4.7 GB of expert reads → NVMe bandwidth is the whole game.

A DFloat11-style "lossless BF16 compression" writeup made the rounds recently. Not applicable to int4 directly (11.2 bits/weight would be a 2.8× *upgrade* in size), but it raises the right question: **do int4 weights compress at all?** The common intuition — including from the colibri maintainer — is no: "quantization already removed the redundancy."

## Measurement 1: int4 compresses fine

Full container, 17 shards, real trained weights:

| | |
|---|---|
| zstd-1, whole container | **151.8 → 109.0 GiB (71.8%)** |
| byte entropy of expert data | 5.89 bits/byte → 73.7% theoretical floor (zstd-1 sits on it; higher levels gain nothing) |
| padding/zeros | 0.0% — it's genuine symbol skew |
| verification | every frame decompressed + byte-compared; engine output bit-exact (teacher-forcing oracle + identical greedy generations at full scale) |

Why: quantization removes *precision* redundancy; entropy coding exploits *distribution* redundancy. Trained weights cluster near zero → the 4-bit codes are heavily skewed. **Effectively 2.87 bits/weight, lossless** — smaller than a lossy int3 container, at zero quality cost.

(Fun control: random-init test models only compress to ~89–93%. If you've "seen zstd do nothing" on quantized weights, check whether you measured synthetic fixtures.)

## Measurement 2: the speed catch

We taught the engine to read compressed shards directly (decompress in the async loader threads, same slab layout, ~200 lines). Expectation: 25% fewer disk bytes → faster cold tokens, decompress hides in the I/O wait.

Same-day warm benchmark pairs (RTX 3060, 16 cores, WSL2, 2.75 GB/s NVMe):

| container | tok/s (median) | expert-load stall |
|---|---|---|
| int4 plain | **0.38** | ~8 s / 24 tok |
| int4+zstd | 0.31 | ~15 s / 24 tok |

The "decompress hides in the disk-wait shadow" assumption is **false once you have a load/compute overlap pipeline** — the cores are already full of expert matmul during loads. zstd decompress added ~1.05 s/token of new CPU work on a box that had no spare cores. Loader priorities, bigger decompress pools: nothing recovered it. Capacity problem, not scheduling.

This also applies to "dequant compute is free because you're disk-bound" arguments in general (e.g. rotation-preconditioned int2 proposals) — worth budgeting before believing.

## Where the compressed container wins anyway

- **Distribution/storage: 26% smaller.** Both containers are on HF: [plain int4](https://huggingface.co/fivetech/MiMo-V2.5-colibri-peng-int4) (152 GB) and [int4+zstd](https://huggingface.co/fivetech/MiMo-V2.5-colibri-peng-int4-zstd) (109 GB, engine reads it natively; repack tool converts either way with byte-exact verify).
- **RAM-rich boxes** (experts pinned at boot): decompress once at load, zero steady-state cost.
- **Core-rich / native Linux** (no WSL2 I/O CPU tax): the sign may flip — if you have a fat NVMe and spare cores, I'd love to see your numbers.

Everything (including the failed experiments) is in [findings.md §41](https://github.com/fivetechsoft/peng-mimo/blob/peng/findings.md), tooling in `c/tools/repack_zstd.py`.

---

## Notes for posting (not part of body)

- Wait for HF upload to finish and links to 404-check before posting.
- Flair: "Resources" or "Discussion".
- Expected pushback: "why not GGUF/llama.cpp k-quants" → answer: colibri/peng stream experts from disk per token, different regime; k-quants are lossy — this is lossless ON TOP of int4, same idea would stack on k-quants' packed codes.
- Expected question: "native Linux numbers?" → honest answer: untested, invited.
- If asked about GPU decompress: nvcomp zstd + compressed PCIe upload is the roadmap idea; not built.
