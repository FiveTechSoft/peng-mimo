# roadmap — GPU (CUDA) acceleration plan

Findings and experiments for GPU offload on the reference box (RTX 3060,
12 GB VRAM + ~23 GB RAM, disk ~2.75 GB/s). Complements `findings.md` §19–§24.
The engine runs with GPU (`COLI_CUDA=1 CUDA_DENSE=1`); the speed dial is expert
hit-rate + bytes read from disk (not dense FLOPs).

## Current status (done + measured)

- **CUDA toolkit** on WSL; build `make mimo CUDA=1 CUDA_ARCH=sm_86`. Gotcha:
  `touch mimo.c backend_cuda.cu` before `make` when editing from Windows.
- **Makefile**: `mimo` links `$(CUDA_OBJ)`; `-g -rdynamic` for Linux SEGV
  backtraces.
- **Eager dense upload** + free host; complementary VRAM expert tier (auto GB).
- **Speed pack (§24):** fused SwiGLU, fast int4/int8 GEMV, sticky device-x,
  attention AVX2, SERVE speed defaults, `chat_peng.py --bench/--fast`.
- **cuda-test**: q8/q4/q2/f32 + fused SwiGLU ok on sm_86.

### Measured on full MiMo — `findings` §23–§24

| run | setup | hit | expert-disk | tok/s | notes |
|---|---|---|---|---|---|
| §23 P0 PROMPT | eager free + complementary | 46.3% | 12.4 s / 24 tok | **0.50** | baseline |
| **§24 PROMPT** | + fuse/GEMV/AVX2/headroom | **50.5%** | 11.2 s / 19 tok | **0.43** | no win vs §23 yet |
| §24 SERVE bench | `--bench --fast`, STAT includes prefill | ~43% | — | **0.21 med** | chat template ~33 tok prefill |

Target **1.0 tok/s**: FAIL on this box (PROMPT ~0.43, SERVE-STAT ~0.21).

Bench helpers:

- `c/scripts/run_chat_bench.sh` — PROMPT + PROFILE
- `c/scripts/run_speed_bench.sh` — `chat_peng.py --bench --fast`
- `python3 c/chat_peng.py --bench --fast`

## Remaining problem

Hit ~50% still leaves half the expert work cold. Matmul wall ~16.5 s flat after
fuse (structure ready; throughput not yet above §23). Attention improved
(~15→13 s). Disk ~11 s. Path to 1.0 needs **higher hit** and/or **much faster**
GPU expert path and/or more RAM / native Linux.

WSL2 still burns host CPU on I/O (`findings` §18).

## Next experiments (priority order)

### 1. Re-measure P0 — DONE ✅
See §23.

### 2. Fuse + fast GEMV — DONE ✅ (measured §24)
Landed; PROMPT 0.43 tok/s (not above 0.50). Keep fuse as default path.
- [ ] Restore honest `cuda-copy` timer under async streams.
- [ ] True single-kernel expert MLP (or CUBLAS/int4 tensor core path).

### 3. Fit more experts (hit-rate) — NEXT
- `ebits=2` / int2 experts (half the bytes; validate quality).
- REPIN over `gpu_pin` (VRAM ranking frozen to `.coli_usage` today).
- Parallel load of the VRAM tier (today sequential + realloc).
- SWA ring KV (~1.5 GB host back to expert cap).

### 4. Residual correctness
- Finding **#18**: post-MTP/v2 ~31/32 TF / 18/20 greedy on tiny int8.
- Identity gate DRAFT=0 vs n on GPU runs.

### 5. Product / tooling
- [x] `chat_peng.py`: CUDA/PILOT/REPIN defaults, `--bench`, STAT protocol fix.
- [x] `c/scripts/run_speed_bench.sh`.
- [ ] Document QUALITY vs FAST in README briefly.

### 6. Note on `ds4` (antirez/DwarfStar)
Confirms VRAM → RAM → NVMe hierarchy; not portable code for MiMo.

## Metrics to watch (always in `PROFILE` / header)

- `tok/s` and **`expert-disk`** (PROMPT + PROFILE for kernel work).
- SERVE STAT tok/s = decode / **(prefill+decode)** — do not mix with PROMPT.
- **expert hit-rate**; complementary VRAM N vs RAM-pin M (**disjoint**).
- **RSS** after eager dense; LRU `cap` per layer.
- Gate: mediana ≥ 1.0 on a defined protocol (not yet met).
