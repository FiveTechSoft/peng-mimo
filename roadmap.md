# roadmap — GPU + Corriente speed plan

Findings and experiments on the reference box (RTX 3060 12 GB VRAM + ~23 GB
RAM WSL2, disk ~2.75 GB/s). Complements `findings.md`. The engine runs with
GPU (`COLI_CUDA=1 CUDA_DENSE=1`); the speed dial is **expert hit-rate**,
**honest PROFILE**, and **not paying I/O tax in “other”**.

## Current status (2026-07-12)

### Best measured (full MiMo int4, pin + GPU-first)

| run | setup | hit | disk | attn | matmul | other | tok/s |
|---|---|---|---|---|---|---|---|
| §25 PROMPT | GPU-first + moe_acc | ~60% | 8.9 s | — | ~16 s | — | **0.55** |
| §32 SPEED | byte-strided GEMV + pin | ~60% | 10.6 s | 11 s | **4.9 s** | 20 s | **0.51** |
| §37 TAO + throttle | TRAJ_WARM_EVERY=2, PATHPACK_EVERY=8 | 54% | 14.4 s | 9.1 s | 4.6 s | **5.9 s** | **0.60** ← **best** |
| §38 physical pathpack | disk-order pack + merged fadvise | 58% | 13.8 s | 10.2 s | 5.0 s | 16 s | 0.50 |

**Gate 1.0 tok/s: still FAIL.** Soft ceiling on this box **~0.55–0.65** tok/s.
**Project best remains 0.60 tok/s** (§37). Physical pathpack is landed for sequential
WILLNEED; did not raise the record. Latest big win was cutting hidden **other**.

**Correctness (§39):** SWA ring (§29) corrupted any batch prefill crossing the
window (S ≥ 2, pos ≥ W=128) — long prompts / TF / SCORE were garbage since
`4596b8e`; short-prompt speed numbers stand. Fixed (scratch chunk + ring flush),
tiny oracle back to pre-ring level (TF 31/32, greedy 15/20). SCORE mode +
`SCORE_DUMP` now power the TOPP/TOPK quality matrix (`scripts/bench_topp_ppl.sh`).

### Stack landed (keep)

- CUDA dense eager + **GPU-first** expert tier + `moe_acc` single-stream
- Byte-strided int4 GEMV / fused gate+up
- TRAJ Markov WILLNEED + **persist** `.coli_traj` + boot warm
- **FLOW** pathpack channel thaw (`.coli_pathpack`)
- **ENERGY** channel→VRAM when free VRAM remains
- Expert **bitmaps** (`res_bits` / `pref_bits`)
- **PROFILE-AUX:** `traj_warm | pathpack | persist`
- **TAO=1** wu wei + φ/Fibonacci knobs; `scripts/start_peng.sh`
- Docs: `corriente-peng.md`, `tao.md`, `sacred-geometry.md` + SVGs

### Bench helpers

```bash
# one-shot PROMPT + PROFILE
TAO=1 SPEED=1 PILOT=0 SNAP=~/mimo25_i4 \
  PROMPT='…' NGEN=24 COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 \
  ./c/mimo 64 4 8

# chat / auto env
TAO=1 c/scripts/start_peng.sh chat
c/scripts/run_speed_bench.sh
python3 c/chat_peng.py --bench --fast
```

Watch: `tok/s`, `PROFILE`, **`PROFILE-AUX`**, hit-rate, RSS, GPU expert count.

## Remaining problem

Even at **0.60 tok/s**:

1. **expert-disk ~14 s / 24 tok** — hit ~54% still cold half the time.
2. **attention ~9 s** — next compute pole after matmul was fixed (~5 s).
3. **traj_warm ~3.8 s** (AUX) — still non-trivial; try `TRAJ_WARM_EVERY=3` or off on single-shot.
4. **WSL2 I/O CPU tax** (`findings` §18) — native Linux still on the path to 1.0.

## Next experiments (priority)

### A0. Quality-aware trim (from §40 matrix)

- [x] **Measured (§40): trims are a quality/speed frontier, not a free win.**
  TOPK=6 → 0.36 tok/s (+11% ppl); TOPP=0.7 → 0.44 (+19%); TOPP=0.55 → 0.49 (+70%);
  top-8 → 0.32. Speed ~linear in experts loaded. Keep TOPP=0.55 for SPEED demos;
  recommend TOPP=0.7 (balanced) or TOPK=6 (quality) elsewhere.
- [x] Hybrid `TOPK=6 TOPP=0.7` tested: 0.45 tok/s but +55% ppl (0.7 mass cut on
  already-capped 6 trims deeper) — dominated by plain TOPP=0.7, discarded (§40.3)
- [ ] Layered trim: keep top-8 on first/last ~4 layers, TOPK=5–6 in the middle
- [ ] 128 GB RAM quote: 4×32 GB DDR4-2666 ECC RDIMM (~$250 used, ~$970 new kit)
  + install; kills the disk pole entirely. §37 physics with disk→0:
  attn 9.1 + matmul 4.6 + other 5.9 ≈ 19.6 s / 24 tok ≈ **1.2 tok/s** (gate PASS).
  Stack TOPK=6 (§40), native Linux (§18), DRAFT=2 on warm RAM → ~1.2–1.5 realistic.
  (Old ~0.66 estimate was §29-era kernels: matmul 15.5 s pre-GEMV.)

### A1. Lossless zstd expert pack (§41) — DONE, mixed verdict

- [x] **Size win: 151.8 → 109.0 GiB (71.8%), byte-exact verified, bit-exact engine
  output** (2.87 effective bits/weight, lossless — smaller than int3 would be).
- [x] Repack tool `c/tools/repack_zstd.py` (streaming, atomic, `--verify`,
  `--skip-existing`), format flag `peng_zstd`, reader in `st.h` + `expert_load`.
- [x] **Speed: −18% on this box** (i4z ~0.31 vs i4 ~0.38 same-day median).
  Decompress = ~1.05 s/token new CPU work; OVERLAP already keeps cores busy —
  no shadow to hide it in. `LOADNICE` knob added; doesn't recover. **Default
  container stays uncompressed for streaming here.**
- [ ] Native Linux re-test (no WSL2 I/O CPU tax → spare cores → may flip sign)
- [ ] 128 GB RAM config: compressed container + decompress-once at pin = free win
- [ ] GPU decompress path (nvcomp zstd, −28% PCIe, frees host CPU) — big, pairs
  with the CUDA expert tier

### A2. MiMo-lite: fewer experts without retraining (§42) — explored, naive version rejected

- [x] **Expert redundancy measured (§42.1): no near-duplicate experts** (mid/late
  layers orthogonal, cos < 0.07) — intra-layer merging is out. Early layers share
  a common component (delta-coding candidate, marginal).
- [x] **Usage long-tail measured (§42.2):** only 2.5% of experts unused (3.7 GB);
  Gini 0.615. Keep-192 covers 98.4% of historical calls.
- [x] **EKEEP=n runtime prune landed** (mask by usage rank, zero disk, reversible)
  + SCORE gate: EK192 = prose +7.8% but **code +67%** — usage history is
  prose-biased, pruning removed the code specialists. **Naive frequency pruning
  rejected with data.**
- [ ] Multi-profile prune criterion: union of top-n across `.coli_usage.<profile>`
  heat maps (code/chat/multi) or diverse-corpus usage; re-run EKEEP matrix
- [ ] If a criterion passes screening: broad-corpus validation (10-20k tokens),
  agreement@5/KL, router stats, then physical pruned container (+zstd §41:
  keep-192 ≈ 80 GB)
- [ ] Engine dump extensions for the gate: top-5 ids+probs in SCORE_DUMP,
  per-layer router entropy

### A. Fit more experts (hit-rate) — main lever for 1.0

- [ ] Host **32–64 GB RAM** (architecture-level win)
- [x] **Physical pathpack (runtime):** disk-order of habit experts + merged fadvise (§38)
- [ ] **Physical pathpack repack** of safetensor shards on disk (true sequential layout)
- [ ] int2 experts (`ebits=2`) + quality gate
- [ ] Native Linux vs WSL2 I/O
- [x] SWA ring KV, REPIN→VRAM, autopin 85%, bitmaps, FLOW/ENERGY

### B. Attention / residual compute

- [ ] Confirm all full/SWA attn on CUDA densas (no silent CPU)
- [ ] Tensor-core / better tiled int4 GEMV if matmul rises again
- [ ] Restore honest `cuda-copy` under async streams

### C. AUX / habit I/O (done enough for now)

- [x] PROFILE-AUX timers
- [x] `TRAJ_WARM_EVERY` default 2 on SERVE/SPEED/TAO
- [x] `PATHPACK_EVERY` default 8 on SPEED/TAO
- [ ] Optional: skip traj_warm entirely for `PROMPT` single-shot (`TRAJ=0`)

### D. Product / docs

- [x] `start_peng.sh`, `TAO=1`, Corriente / sacred geometry docs
- [x] MiMo chat template, cancel/health API (earlier F-xx)
- [x] QUALITY vs FAST one-pager in README (§40 frontier table)

### E. Correctness gates (do not regress)

- Tiny / fixture oracle TF + greedy
- Identity DRAFT=0 vs n when MTP used
- Convert atomic + revision
- [ ] **Long-prefill gate (S > sliding_window)** — §39: the tiny oracle (S=32, W=8)
  caught the SWA ring bug only by luck; add a fixture with prefill well past W
- SCORE ppl matrix before/after any routing or attention change (`scripts/bench_topp_ppl.sh`)

## Explicit non-goals (measured dead ends)

- Multi-stream GPU experts + atomic acc (slower on 3060)
- Uncapped TRAJ_K / fadvise storm
- `DRAFT=2` on cold disk
- Forcing ENERGY when VRAM already full after GPU-first pin

## Metrics protocol

| Mode | What tok/s means |
|---|---|
| `PROMPT` + PROFILE | prefill+decode wall for NGEN (use for kernel/disk) |
| SERVE STAT | often decode-focused; **do not mix** with PROMPT |
| Hit-rate | `(hits)/(hits+miss)` experts |
| Gate | mediana ≥ **1.0 tok/s** on a fixed protocol — **not met** |

## Related docs

| Doc | Role |
|-----|------|
| `findings.md` §25–§37 | measurements & change log |
| `docs/corriente-peng.md` | residual-as-river design |
| `docs/tao.md` | wu wei knobs |
| `docs/sacred-geometry.md` | φ / Fibonacci mapping |
| `docs/diagrams/*` | logical / physical / corriente / sacred SVGs |
