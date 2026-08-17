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
- [x] **Multi-profile prune criterion (§42.4): WORKS.** Union of per-domain top-K
  (prose/C/Python/general profiles) — UNION179 = 104 GB expert store (−37%) at
  prose +7.8%, code −1.8% (≤ noise). Naive EK192's +67% code collapse eliminated.
  Tools: `make_domain_corpora.py`, `build_ekeep_mask.py`, engine `EKEEP_MASK`.
- [ ] **MiMo-lite 128 GB play:** 104 GB pruned store fits entirely in RAM →
  disk pole gone → §37 physics ≈ 1.2 tok/s (gate PASS) on the ~$250 upgrade.
  Blocked on finalist validation below.
- [x] **Speed measured: UNION179 mask = median +16% tok/s** (0.51 vs 0.44
  same-day §37 pairs; best 0.55) — cache coverage of the smaller pool, exactly
  in the predicted band. Quality-neutral speedup, stacks with everything.
- [ ] Finalist gates before physical container: broad 10–20k-token corpus,
  agreement@5/KL, router entropy stats, HumanEval subset; then pruned+zstd
  container (~75 GB)
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

- [x] **CUDA_ATTN fused decode attention** (§45/§46): KV resident on device,
  1 sync/layer; opt-in `CUDA_ATTN=1`, non-bit-exact, default off. Real 311B:
  +21% mean per-pair (4/4) with spin, best 0.89 tok/s
- [x] **Spin-sync fix** (§46): `cudaDeviceScheduleSpin` default in `coli_cuda_init`
  (`COLI_CUDA_SYNC=yield|block` override) — yielding syncs cost ~5–8 ms/layer
  under expert-pipeline load; this was the hidden CUDA_ATTN eater
- [x] **311B A/B done on this PC** (§46): container re-downloaded; base 0.60 →
  attn+spin 0.72–0.89 (median ~0.76, host-drift bound). Gate 1.0 still open:
  next poles expert-matmul (~11 s CPU int4) and expert-disk (hit-rate)
- [x] **Re-upload FIXED int4 shard to HF** (§46): surgical qkv patch of
  `out-00001.safetensors` (2026-07-20); guard PASS, ppl 9.97/2.07, Paris OK.
  Pre-20-jul downloads need that shard only. **zstd HF still corrupt** — repack
  from repaired int4 pending
- [ ] **311B greedy sanity + §40 ppl drift matrix for CUDA_ATTN=1** on the
  fixed container
- [ ] Confirm all full/SWA attn on CUDA densas (no silent CPU)
- [ ] Tensor-core / better tiled int4 GEMV if matmul rises again
- [x] **Restore honest `cuda-copy` under async streams** (§47): `COLI_CUDA_PROF=1`
  event-based H2D/D2H bytes+ms, no extra syncs

## CUDA MoE v2 (issue: expert residency, async I/O, fused kernels)

Goal: >1 tok/s without touching weights/quality, evolving toward a hybrid
CPU/NVMe/GPU runtime (GPU owns hidden state; async expert I/O; fused kernels).
Cycle-based; each cycle gated by the profiler data of the previous one.

### Cycle 1 — DONE (§47, 2026-08-16): profiler + low-risk kernels

- [x] **Build fix WSL2 (Ubuntu 26.04, glibc 2.43):** `NVCCFLAGS += -U_GNU_SOURCE`
  (C23 math `noexcept` vs CUDA 12.8 headers). Blocks any CUDA work on this box
- [x] **Event profiler `COLI_CUDA_PROF=1`:** per-site GPU ms (gate_up/gemv/axpy/
  attn), H2D/D2H ops+bytes+ms, sync-wait; drained at existing syncs only
- [x] **`PROF_TRACE=<csv>`** per (forward,layer): tier hits, real `nvme_bytes`,
  kernels, wall; **`PROF_EXPERTS=<csv>`** per-expert freq/tier/disk/ms dump
- [x] **`fused_down_acc_i4/i8`** (down GEMV + weighted acc in one kernel):
  bit-identical (exact-equality test in `test_backend_cuda.cu`); 25→17
  kernels/layer decode; kill-switch `COLI_CUDA_NO_FUSED_DOWN=1`
- [x] **Pinned staging** for MoE activation copies (<1 MB; default ON,
  `COLI_CUDA_PINNED=0` A/B): H2D 16 KB 82→25 µs
- [x] **Kernel-level bench** (`make fused-bench`, `tests/fused_down_bench.cu`):
  300 reps × 8 experts real shapes: 452→388 ms (**−14% MoE layer path**,
  66.8→77.8 GB/s VRAM)
- [x] **B4 scales warp-shuffle / B5 `xg` indexed input: measured, closed as
  non-actionable** in decode S=1 (GEMV is weight-traffic-bound; `xg` is one
  16 KB memcpy vs ~2–4 ms CPU GEMV)
- [ ] **REPLAY e2e baseline + benchmark matrix: BLOCKED** — model snapshot
  (~152 GB) not on this box since the OS reinstall; needs mount/download

### Cycle 2 — next (needs the model mounted)

- [ ] REPLAY baseline with profiler: per-layer CSV audit, find the real pole
  split (disk vs CPU-expert-matmul vs GPU vs sync)
- [ ] C: async expert H2D — upload stream + compute stream + CUDA events +
  residency state machine (UNLOADED→UPLOADING→READY→IN_USE); re-test the
  multi-stream non-goal with pinned buffers (§27 test predates them)
- [ ] D: GPU hidden-state residency — rmsnorm/residual/router on device,
  kill the per-layer D2H/H2D round-trips (~2 syncs + 4 PCIe hops × 48 layers)
- [ ] E: grouped/batched expert GEMV benchmark (grouped GEMV vs grouped GEMM vs
  Tensor Core INT4 at S=1); keep the fastest per workload

### Later cycles

- [ ] F: physical path-pack of shards from coactivation data + predictive
  prefetch P(E_next|E_prev,layer); cold/warm cache bench
- [ ] G: CUDA Graphs over the layer stack; UNION179 ~104 GB full-RAM mode
  (disk-bound vs RAM-bound vs GPU-cache-heavy split); MTP on/off once I/O-bound
  is gone

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

- Tiny / fixture oracle TF + greedy (re-run in §46 with the CUDA=1 binary, flag off)
- Identity DRAFT=0 vs n when MTP used
- Convert atomic + revision
- [ ] **Long-prefill gate (S > sliding_window)** — §39: the tiny oracle (S=32, W=8)
  caught the SWA ring bug only by luck; add a fixture with prefill well past W
- SCORE ppl matrix before/after any routing or attention change (`scripts/bench_topp_ppl.sh`)
- [ ] CUDA_ATTN=1 greedy sanity + ppl drift matrix on the 311B snapshot (§45/§46)
- [ ] **Container qkv scale-grid integrity (`c/tools/verify_mimo_qkv.py`)** — 2026-07-16
  incident: a deployed `mimo25_i4` container was built with the pre-fix per-rank
  scale-grid converter, corrupting ranks 1..NR-1 of every `qkv_proj` (rank 0 intact)
  while `o_proj`/experts stayed bit-identical. Run the guard (fp8 source vs int4
  container, per rank-block) in CI / pre-ship so a stale container can never ship
  again: `python3 c/tools/verify_mimo_qkv.py --int4 <container> --fp8 <fp8_src>`

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
