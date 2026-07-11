# roadmap — GPU (CUDA) acceleration plan

Findings and experiments for GPU offload on the reference box (RTX 3060,
12 GB VRAM + ~23 GB RAM, disk ~2.75 GB/s). Complements `findings.md` §19–§23.
The engine runs with GPU (`COLI_CUDA=1 CUDA_DENSE=1`); the speed dial is expert
hit-rate + bytes read from disk (not dense FLOPs).

## Current status (done + measured)

- **CUDA toolkit** on WSL; build `make mimo CUDA=1 CUDA_ARCH=sm_86`. Gotcha:
  `touch mimo.c backend_cuda.cu` before `make` when editing from Windows.
- **Makefile**: `mimo` links `$(CUDA_OBJ)`; `-g -rdynamic` for Linux SEGV
  backtraces.
- **Eager dense upload** (`cuda_upload_dense_all`): after `model_init` and
  resolving `DRAFT`, uploads densas to VRAM, **frees host copies**, sets
  `gpu_only=1`, shrinks `resident_bytes`. Order: **dense → RAM pin → VRAM
  expert tier**. Measured: **99/99 ok, 4.69 GB host freed in ~3–6 s**.
- **`embed` / `lm_head` never on VRAM** (CPU gather / one matmul per token).
- **MTP densas** only when `DRAFT>0`.
- **Complementary VRAM tier**: slice `r[npin..n)` disjoint from the RAM pin;
  **443 experts / 5.59 GB** with auto budget on this box.
- **Backend**: `*tensor` cache-hit before requiring host `weights` (SEGV fix).
- **`gpu_only` fail-loud**: no silent zeros.
- **`CUDA_EXPERT_GB` auto**: unset + CUDA → `free - 1.5 GB`; `0`=off; `>0`=cap.
- **PROFILE-GPU**: `cuda-copy` is H2D+D2H only (sync after kernel).

### Measured on full MiMo (2026-07-11) — `findings` §23

| run | setup | hit | expert-disk | tok/s / wall | notes |
|---|---|---|---|---|---|
| gpu_ed17 | densas + VRAM=pin duplicate 360 | 39.9% | 50 s / 20 tok | 0.20 / 98 s | pre-complementary |
| complementary | pin 363 + VRAM 443, host densas kept | 52.5% | 29 s | wall ~4× densas-only | pre-eager free |
| §20 NGEN=8 | host densas kept, cap=3 | 27% | thrash | 0.14 | motivated free-host |
| **§23 P0** | eager free densas + complementary + auto GB | **46.3%** | **12.4 s / 24 tok** | **0.50 / 48 s** | chat coherent |

Chat-style SERVE turn also streamed a full Rome sentence after ~26 s load.

Bench helper: `c/scripts/run_chat_bench.sh` (PROMPT + PROFILE).

## Remaining problem

Coverage ~800 distinct experts of 12288. Hit ~46% still leaves half the work
on disk or cold path. After P0, **expert-matmul + attention (~31 s)** outweigh
**expert-disk (12 s)** on this prompt — so both **more hits** and **faster GPU
expert kernels** matter. PCIe copy is only **0.76 s** (honest timer).

WSL2 still burns host CPU on I/O (`findings` §18 overlap).

## Next experiments (priority order)

### 1. Re-measure P0 — DONE ✅
See §23. Commit unblocked on speed/correctness smoke.

### 2. Fuse expert matmul on GPU (P1 speed)
Today 3× (gate/up/down) = 3 H2D + 3 D2H per expert hit. One kernel
`down(silu(gate(x))*up(x))` should cut the 15 s cuda-compute when VRAM-hot.
Later: layer-resident activations; CUDA graphs for S=1.

### 3. Fit more experts (hit-rate)
- `ebits=2` / int2 experts (half the bytes; validate quality).
- REPIN over `gpu_pin` (VRAM ranking frozen to `.coli_usage` today).
- Parallel load of the VRAM tier (today sequential + realloc).

### 4. Residual correctness
- Finding **#18**: post-MTP/v2 ~31/32 TF / 18/20 greedy on tiny int8.
- Identity gate DRAFT=0 vs n on GPU runs.

### 5. Cleanup / product
- [x] Remove `[CUDA-DBG]`.
- [x] Auto `CUDA_EXPERT_GB`.
- [x] Bench scripts under `c/scripts/`.
- [ ] `chat_peng.py`: document `COLI_CUDA=1 CUDA_DENSE=1` opt-in (pass-through env already works).
- [x] Commit after P0 re-measure.

### 6. Note on `ds4` (antirez/DwarfStar)
Confirms VRAM → RAM → NVMe hierarchy; not portable code for MiMo.

## Metrics to watch (always in `PROFILE` / header)

- `tok/s` and **`expert-disk`**.
- **expert hit-rate** (complementary tier + densas freed on host).
- `CUDA complementary VRAM tier: N expert (RAM-pin M)` — N and M **disjoint**.
- **RSS** after eager dense; LRU `cap` per layer.
- PROFILE-GPU: real PCIe `cuda-copy` vs compute.
- VRAM free leftover ~0–1.5 GB headroom after auto expert fill.
