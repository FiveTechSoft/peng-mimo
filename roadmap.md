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

- [ ] **Switch SPEED default `TOPP=0.55` → `TOPK=6`**: +11% ppl vs +70%, similar
  expert savings. Re-measure tok/s + hit-rate (trims can lower VRAM hits, §29).
- [ ] Layered trim: keep top-8 on first/last ~4 layers, TOPK=5–6 in the middle
- [ ] 128 GB RAM quote: 4×32 GB DDR4-2666 ECC RDIMM (~$250 used, ~$970 new kit)
  + install; kills the disk pole entirely (§29 physics → ~0.66 ceiling, then attn/matmul)

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
- [ ] QUALITY vs FAST one-pager in README (brief)

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
