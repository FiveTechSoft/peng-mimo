# findings — project discoveries and technical insights

Record of surprises and technical discoveries during the port of colibrí to MiMo-V2.5.
Each entry: the technical finding + an explanation for the uninitiated.

---

## MiMo-V2.5 Architecture

### 1. Official reference code only works with transformers 5.0–5.1

The `modeling_mimo_v2.py` that Xiaomi publishes requires both an API that was born in
transformers 5.0 (`standardize_rope_params`) and another that died in 5.2 (the `input_embeds`
argument to `create_causal_mask`, renamed later). We tested versions 4.57 to 5.13: only
5.0.0 and 5.1.0 satisfy both. We use 5.1.0 in a dedicated venv (`~/mimo-venv`) without
touching a single comma of the official code.

> **In plain terms:** the "instruction manual" for the model can only be read with a very
> specific version of the reference library — neither the one its own tag says (4.57) nor
> the latest. We found the exact window by testing one by one.

### 2. Sliding window attention layers have THEIR OWN attention geometry

It's not just the window: SWA layers use more KV heads (8 vs 4 in the real model; 4 vs 2 in
tiny) and their own head dimensions (`swa_head_dim`, `swa_v_head_dim` — and watch out: if
they're not in the config, `swa_v_head_dim` inherits from `swa_head_dim`, NOT from
`v_head_dim`). Discovered by the tiny model: its SWA `o_proj` is [128,192], not [128,128].

> **In plain terms:** the model alternates two types of "vision": 9 layers see the entire
> conversation and 39 only the last 128 words. Turns out they also wear glasses with
> different prescriptions — if the engine assumes the same for all, the numbers don't add up.

### 3. Sink bias: an "attention sink" only in SWA layers

Each head in sliding window layers has an extra learned logit that participates in the softmax
but whose probability is discarded — a sink that absorbs excess attention. Full layers don't
have it (`add_full_attention_sink_bias: false`).

> **In plain terms:** when the model only sees the last 128 words, it needs a place to "park"
> attention it doesn't want to spend on any of them. It's a stability trick; without implementing
> it, results diverge.

### 4. Values (V) are multiplied by 0.707 before everything

`attention_value_scale: 0.707` (≈ 1/√2) is applied to the V vector right after projection,
before entering the KV cache. We apply it at runtime (negligible cost) so the same code serves
both the f32 oracle and the quantized container.

### 5. Fused QKV and o_proj outside FP8 quantization

The checkpoint saves q, k, and v in a single tensor per layer (`qkv_proj`, "fused_qkv" layout)
— and all `o_proj` are in the `ignored_layers` list: they travel in bf16 while the rest go in
FP8 with 128×128 block scales. The converter will need to treat them differently.

### 6. Config with small gotchas

- `rope_theta` and `partial_rotary_factor` appear DUPLICATED (top-level and inside
  `rope_parameters`); `swa_rope_theta` only top-level.
- `routed_scaling_factor` is null JSON (not absent) → must treat null as 1.0.
- Normalization epsilon is called `layernorm_epsilon` (GLM uses `rms_norm_eps`).
- RoPE is partial (first 64 of 192 dims) and NOT-interleaved (NeoX/rotate_half style),
  with different theta per layer type: 10,000,000 (full) vs 10,000 (SWA).

> **In plain terms:** the config file has several fields with names or formats slightly different
> from other models. Each is a minefield if the engine assumes the "usual" format.

### 6b. C attention nailed validation on the first try

With facts verified in advance (exact order: fused projection → V×0.707 → partial RoPE → cache
→ window p−128+1 → sink in denominator → o_proj), the C implementation passed teacher-forcing
32/32 and greedy 20/20 on the first attempt, without a single debugging iteration. Verified twice
by an independent reviewer.

> **In plain terms:** when you invest time reading the original blueprint with a magnifying glass
> before building, the piece fits on the first try. All the archaeology work on config and modeling
> (findings 1–6) paid off here.

Fragility note recorded: the rope_dim per layer type is derived by integer rescaling from the
full value; if a future config had `swa_head_dim ≠ head_dim` with a non-exact factor, it could
differ by ±1 from the reference calculation. Current configs (tiny and real) are immune; the
oracle would detect it instantly in any case.

### 6c. Quantization in tiny: exact int8, int4 flips one argmax, lossless packing

Full matrix in the oracle tiny: int8 reproduces f32 **token-exact** (32/32, 20/20 — not a single
flip); int4 flips ONE position out of 32 (rest of greedy divergence is autoregressive cascade from
that single flip); int4 packed ≡ int4 unpacked byte-for-byte (lossless packing); entire IDOT kernels
≡ exact dequant to int8; all deterministic. Independently verified by the driver.

> **In plain terms:** compressing weights to half (int8) didn't change a single result; compressing
> to a quarter (int4) changed one of 32 — and on a random-weight model where any noise flips
> decisions at the margin. On the real, trained model, margins are much larger.

### 6d. Cache cap auto-raises silently (benchmark gotcha)

Requesting `cap=2` from CLI doesn't give a cache of 2: the auto-raise from upstream (feature of
2026-07-10) bumps it up to fill the RAM budget. To force the requested cap: `CAP_RAISE=0`. With it,
real LRU eviction maintains exact tokens (20/20) with hit-rate 88%→81% — streaming under pressure
validated.

## Tokenizer

### 7. Modern merges format: single string, not pairs

GLM stores BPE rules as pairs `["Ġ","Ġ"]`; MiMo (tokenizers ≥0.20) stores them as a single string
`"Ġ Ġ"`. The original C parser would have dereferenced NULL. Added support for both formats in
`tok.h` (+ explicit failure if an entry comes without separator — code review finding).

### 8. Digits: one at a time, not three at a time

GLM's pretokenization regex groups numbers in chunks of up to 3 digits (`\p{N}{1,3}`, cl100k style);
MiMo/Qwen takes them one at a time (`\p{N}`). Auto-detected from tokenizer.json. Validated against
the official library with 6 Unicode cases + 4 adversarial ones (emoji ZWJ, 13-digit run…): identical ids.

> **In plain terms:** two models can chop "2026" differently: one as "202"+"6", another as "2"+"0"+"2"+"6".
> If the engine chops differently than the model expects, it understands something else. Pending note:
> detection is simple text search — harden before supporting a third family.

## Converter (FP8 → int4/int8 container)

### 8b. Converter validates against itself: container ≡ runtime quantization

The converter's gate is not "looks similar": the pre-quantized container must produce EXACTLY the same
tokens as the engine quantizing the same weights on the fly — same arithmetic (`np.rint` ≡ `lrintf`),
same per-row scales. Verified on tiny and the 396M fixture, also under cache eviction. Plus: the GLM
path remained byte-identical before/after the refactor with an A/B harness on synthetic shards.

### 8c. Real checkpoint gotchas noted for 316 GB conversion

- MiMo shards reach **34.4 GB** (GLM ~5 GB): the default guardrail `--min-free-gb` (20) is
  INSUFFICIENT — raise it when launching.
- `save_file` is not atomic: a cutoff during final write leaves a truncated output shard that resume
  will skip; if it happens, delete the last `out-*` and restart.
- With ~4 MB/s measured on this line, download alone ≈ 22 h (resumable at any point; vision/audio/MTP
  shards never download thanks to weight_map filter).
- `--io-bits 16` was silently broken upstream (int8 astype overflow); now bits≥16 → explicit f32.
  The `--arch mimo` defaults (dense int8, experts int4, io f32) reproduce the motor's validated
  operating point.

> **In plain terms:** converting 316 GB will take a full day of downloading. All effort in this phase
> went to ensuring that when that day ends, the result is correct on the first try — each pipeline piece
> proves it produces bits identical to the reference before touching the real model.

### 8d. MiMo chat template: ChatML with its own gotchas

- Turns are joined WITHOUT a line break after `<|im_end|>` (Qwen's ChatML carries `\n` — coming from
  there guarantees the error) and without BOS token.
- Default system: "You are MiMo, a helpful AI assistant engineered by Xiaomi."
- Thinking comes ENABLED by default in the official template (`<think>` after `<|im_start|>assistant\n`);
  with long reasoning it can consume the entire token budget before the visible response.
- Stop ids are built BY NAME from the tokenizer (`<|endoftext|>`, `<|im_end|>`), not by id from config
  — the snapshot config only declares one of the two.
- Official `generation_config`: temperature 1.0, top_p 0.95 (and a contradictory `do_sample: false`
  that we ignore).
- Validation without model: `TEMPLATE_DUMP=1` dumps the rendered prompt before tokenizing and a test
  compares it against HF's `apply_chat_template` — 5/5 cases.

> **In plain terms:** each model has its own "conversation protocol" — where the speaker labels go.
> One extra space and the model babbles. We checked against the official tool without needing the model
> downloaded.

### 8e. Download estimate was 15× too pessimistic

With anonymous HF a short test measured ~4 MB/s → we estimated 22 h. Actual download sustains **~59 MB/s**
with 2 streams: 34.4 GB shard in 9.7 min → ~2-4 h total. Lesson: don't estimate bandwidth from a cold
connection and handshake rate-limit.

### 8f. Real checkpoint cuts weight/scale pairs at shard boundaries

First contact with the real FP8 checkpoint: crash. Shard 0 brings 4096 weights but 4095 scales — the
checkpoint writer split a `weight`/`weight_scale_inv` pair right at the file boundary (the scale ended
up in the next shard). No test could see it: our test sources were bf16 (no separate scales). Fix:
resolve the scale via the repo index and fetch only its bytes via HTTP Range (scales are KB). The
downloaded 34 GB shard is conserved and reused.

> **In plain terms:** the model comes in 16 "boxes" and the packer cut a piece in two. Our unpacker
> assumed complete pieces per box. It's EXACTLY the kind of failure the resumable conversion was designed
> for: fix the unpacker and continue where it left off, no re-download needed.

### 11b. Watch out for monitors that find themselves

The converter monitor had two rookie bugs: its `$(...)` expanded one shell layer early (ended up watching
empty strings), and its `pgrep -f convert_fp8` found ITSELF (the pattern was in its own command line) —
reported "ALIVE" with the converter dead. Rules: monitor scripts to file (not inline with quote nesting),
and pgrep patterns that don't appear in the monitor's own command (`pgrep -f 'python.*convert_fp8'`).

## The final bug: the checkpoint that its own code can't read

### 13. FP8 checkpoint qkv comes interleaved by tensor-parallelism ranges

The star finding of the project. With EVERYTHING validated (motor token-exact on two test models, converter
byte-exact against re-derivation, tokenizer exact), the real model generated garbage. Layer-by-layer
bisection (numpy reference built from the same container tensors): the motor computed PERFECT — 6 significant
digits agreement across all 48 layers. The divergence was between what the checkpoint means and what we all
thought it meant.

Xiaomi's official FP8 checkpoint stores each fused `qkv_proj` as **4 concatenated TP-range blocks**
`[Q₁|K₁|V₁|Q₂|K₂|V₂|...]`, not the flat `[Q|K|V]` that the published `modeling_mimo_v2.py` makes `split()`
do. vLLM knows this and de-interleaves in separate code (`_shard_fp8_qkv_proj`). Plus the FP8 block scales
for those tensors go PER-RANGE (108 rows = 4×27, not ceil(13568/128)=106), so the "flat" dequant corrupted
3 of 4 ranges in the 9 full layers (SWA saved by casual divisibility). Forensic proof: FP8 saturation per cell
is perfect (448) under the per-range mesh and rotten under the flat one.

**Why no test caught it**: the oracle tiny and fixture were generated by modeling code itself — flat layout
coherent with itself. Motor and reference shared the same convention; the real checkpoint uses another. Only
the real model, with its 316 GB, could expose the discrepancy.

Fix: de-interleaving on load (`qkv_degroup`, auto-detected by FP8 provenance, with env override) + per-range
scale mesh in converter + in-place patch of 18 affected tensors in container. Result: **"The capital of France is
→ Paris."**

> **In plain terms:** the maker published the furniture with pieces from 4 boxes mixed up and a manual from
> ANOTHER version of the furniture. Its own manual can't assemble it; only a third-party assembler (vLLM)
> documents the real piece order. We had to discover it by measuring which ordering makes the bolts (FP8
> saturation) fit perfectly.

### 14. Infrastructure bug streak on final day

- Linux `pread()` won't read more than 2 GB per call and can come up short without error: the 2.5 GB
  embed/lm_head f32 needed a `pread_full` loop in `st.h`.
- "0 warnings" is not "build ok": a grep for warnings hid a COMPILATION ERROR and we ran 20 minutes with
  an old binary. Rule: check make exit code.
- A scratch `inspect.py` from a subagent shadowed Python stdlib (`script dir enters sys.path first`) and
  broke numpy with an indecipherable error. Rule: never name scratch files like standard modules.
- `pkill -f pattern` kills itself if the pattern appears in its own command line (happened to us THREE times,
  once with the pattern in the restart command). Rule: `patté[r]n` with character class, and kill and restart
  in separate calls.

## Acceleration (v2)

### 15. Free knobs: 2.15× measured (0.20 → 0.43 tok/s)

`TOPP=0.7` (skips low-router-weight experts: −41% of reads) + `DIRECT=1` (O_DIRECT on this NVMe outperforms
buffered) compose almost perfectly. Now they're the chat wrapper defaults.

### 16. Native MTP: lossless proven, but cold disk eats the gain

MiMo's MTP head (1 of 3 chained checkpoint layers, SWA geometry, dense) integrated with its same per-range qkv
convention (mesh [116,32] = 4×29, same signature as SWA layers). 64% acceptance at DRAFT=2 (better than GLM's
39-59%), forwards 31→14, and output **byte-identical** with and without speculation (md5). But on this disk-bound
host each draft position widens the expert union to load per layer and the extra I/O cancels the forwards savings
— exactly the cold-cache phenomenon colibrí documented. Default `DRAFT=0`; opt-in for machines with large RAM/pin.

> **In plain terms:** the trick of "guess several tokens and verify them together" works perfectly
> mathematically, but batch-verifying forces loading experts from ALL guessed positions. With disk as bottleneck,
> you pay in disk what you save in compute. On a machine with more RAM the scales flip.

### 17. Identity gate as sharp oracle

Requiring byte-identical output between DRAFT=0 and DRAFT=n exposed TWO latent numeric bugs that token-exact
validation against oracle couldn't see structurally: int4 kernel choice depended on batch size (batch verification
≠ sequential replay) and expert accumulation in batch-union summed in different order than sequential (floats
don't commute; drift flipped an argmax to token 10). Both fixed; also benefit inherited n-gram speculation.

### 18. Disk↔matmul overlap works — and unmasking the real culprit

The double-buffer pipeline (load-thread pool, consume in original order, bit-exact by construction — md5 identical
in all modes) achieved its mechanical goal: disk and matmul are no longer additive (stall 26 s vs 42 s serialized).
But wall-clock didn't improve: matmul inflates almost 1:1 when loads are in flight. The sum `disk+matmul` stays constant
→ **"disk time" in WSL2 is not device wait, it's CPU burned in the host/VHDX I/O stack**, competing for the same
16 threads as multiplications. Neither priorities nor pool sizes change it.

Consequences: (a) on THIS machine the only remaining lever is read fewer bytes (TOPP, int2, more RAM) or exit WSL2
to native Linux; (b) the pipeline stays enabled by default — on hosts with real I/O via DMA (native NVMe) the
max(disk,matmul) we sought should materialize. The feature is ready for the machine that benefits from it.

> **In plain terms:** we tried having the chef mince while the helper brings ingredients — and discovered
> "bringing ingredients" in WSL also the chef does with the same hands. The overlapped recipe is correct; this
> virtualized kitchen is just tapped out. On native Linux, the same code will perform.

## Environment / tools

### 9. Disk benchmark with zeros = lie

First iobench on zero file: 8.5 GB/s "impossible" (faster than physical bus). Cause: Windows host cache on VHDX.
With random data and cold caches: 2.75 GB/s real (O_DIRECT, 8 threads). Rule: disk benchmarks always with random
data and after WSL restart.

### 10. "Logical" size deceives with OneDrive

37.6 GB "on disk" were mostly online-only placeholders (0 real bytes). Deleting them freed almost nothing. PowerShell's
`Length` measures logical size, not physical.

### 11. WSL: quirks that cost time

- `/tmp` is tmpfs and erases when WSL VM shuts down between commands — no state there across invocations; use `~` or `/mnt/c`.
- `echo EXIT=$?` inside the same `wsl -e bash -c "..."` from PowerShell/Git Bash reports exit code wrong; check from outer shell.
- WSL VHDX grows but doesn't shrink (sparse disabled by default due to corruption risk); compacting requires admin
  (`Optimize-VHD`) or export/import.

### 12. Token-exact validation as method

Every change is accepted only if the C motor reproduces the reference implementation's tokens bit-for-bit
(teacher-forcing 32/32, greedy 20/20). The random-weight tiny model with real architecture is the tool: cheap to
generate, deterministic (same seed → same bytes), and exposes errors a trained model hides (random weights → minimal
margins → any deviation flips argmax).

> **In plain terms:** before downloading 316 GB, we build a scaled mockup of the model with the same pieces and
> check that our copy of the motor produces EXACTLY the same results as the original, number for number. If the mockup
> squares perfectly, the big one will too.

## C Motor: applied optimizations

### 15. Precomputed RoPE: goodbye to ~100k powf/cosf/sinf per token

The motor recalculated RoPE angles for each (position, head, dimension) per token (~100k trig calls/token in MiMo).
Now `rope_build()` precomputes cos/sin tables per position for full and SWA layers up to `max_t`, and `rope_apply()`
only does lookup (same formula, same order). Verified bit-for-bit against direct formula with `c/rope_check.c` (512
comparisons, 0 fails). CPU gain: ~10 ms/token ≈ 0.3% of a 3.3 s token — negligible for speed, but cleaner and
deterministic (and removes a cost that grew with S in prefill).

### 16. I4S knob wired (was documented but not read)

`main()` wasn't reading `I4S` env; now `getenv("I4S")` sets `g_i4s` (threshold S for int4 IDOT activation). Note:
on AVX2 `I4S=1` changes decode rounding (NOT token-exact) → chat only, never for oracle validation.

### 17. PILOT/LOOKA: async prefetch by router lookahead (ported from colibri/glm.c)

While layer L computes, layer L+1's router is applied to current state to predict its top-K experts, and they're
warmed with WILLNEED in a separate I/O thread (ring lock-free 1P/1C). It's purely an I/O hint: doesn't touch
weights or compute, so **doesn't change tokens** (safe to leave enabled). Measured recall on GLM ~71.6% of true
top-8; missing ones are minor stalls.
- `PILOT=1` activates prefetch. `LOOKA=1` (without PILOT) measures MiMo recall and prints it on exit via `atexit`.
  `PILOT_K` adjusts predicted K.
- MiMo's router is identical to GLM's (sigmoid + noaux_tc, n_group=1) → directly portable. In mimo router lives in
  `l->router`/`l->router_bias`; residency queried via `expert_prefetch` (WILLNEED idempotent). Unlike glm, mimo
  does NOT use `st_resident`/`sh_key` checks (its shard API is `m->S` at model level, not per-layer) — ring
  duplicates are harmless.
- Build: `Makefile` adds `-pthread` to LDFLAGS (Linux and macOS) for I/O thread.

**How much it accelerates (honest answer):** PILOT only overlaps I/O with compute; doesn't reduce CPU cost. On the
reference box (64 GB RAM, 2.75 GB/s disk, 15B active = 7.5 GB/token): after warmup the 7.5 GB of active experts
fit in cache → disk idle and decode compute-bound (~3.3 s/token, 0.3 tok/s). There PILOT provides **~0%**. Only helps
when disk really would brake the pipeline:
- cold boot / first token (empty cache): overlaps reading next-layer experts after current-layer compute;
- box with little RAM where active set thrashes (each token needs disk);
- when routing changes and uncached expert must load.

In those disk-bound moments, with 71.6% hit rate PILOT hides ~70% of disk latency (2.7 s/token) behind compute:
that token drops from ~6 s (disk+compute serial) to ~3.3 s → up to **~2x on cold/missed tokens**; ~0% in cached
regime. The big lever remains more RAM (`CAP_RAISE`), `PIN`/`autopin`, or GPU offload (`COLI_CUDA=1`). RoPE (#15)
is even more marginal (~0.3%).

**End-to-end validated (oracle tiny generated with transformers 5.1.0):**
- `LOOKA=1 ./mimo 64 8 8` → **PILOT top-8 recall = 71.1% (135/190)** on MiMo, identical to GLM's 71.6%.
  Router-lookahead transfer wholly to MiMo.
- `PILOT=1` produces **identical tokens** to default mode (same sequence 15/20), confirming prefetch doesn't alter
  output (I/O-only, by design).
- `rope_check.c` already confirmed RoPE bit-for-bit (0 fails).
- **But motor is not perfect token-exact vs THIS oracle:** 31/32 teacher-forcing and 18/20 greedy, EVEN with int8
  experts (no int4 noise). That ~1-position divergence does NOT come from PILOT/RoPE/I4S (proven by PILOT==DEFAULT
  and RoPE bit-exact). On rebasing against recent remote commits (MTP speculative decoding + "v2 acceleration") the
  motor now includes that code: the ~1-position diff probably enters there, or from full-feature oracle SWA+sink+value_scale
  that simpler oracle for 32/32 validation didn't exercise. See finding #18.

### 18. Motor (post-MTP/v2) diverges ~1 position from oracle, even in int8

After rebasing on 4 recent remote commits (MTP speculative decoding, "v2 acceleration: fast defaults 2.15x",
chat_peng, README), motor gives 31/32 teacher-forcing and 18/20 greedy on full-feature tiny oracle (6 layers,
SWA+sink+value_scale, int8 experts). With int4 drops to 15/20 (expected: int4 flips ~1 pos and cascade).
Divergence is NOT from PILOT (identical tokens with/without PILOT) nor RoPE (bit-exact via rope_check.c) nor I4S
(default unchanged in AVX2). Still to locate whether it enters from MTP/v2-acceleration commits or from full-feature
oracle detail (SWA window, sink bias, attention_value_scale 0.707) that simpler 32/32-validation oracle didn't
activate. Not blocking for PILOT, which stays token-safe (WILLNEED only).

## CUDA VRAM offload tier — segfault root cause & fix

### 19. Complementary VRAM tier offloads experts to GPU, but freed host slabs must never be dereferenced

The complementary VRAM tier (see mimo.c `pin_load`) loads the experts the hot-store RAM-pin set did NOT
cover into VRAM, then `expert_cpu_free()` frees their host slab and sets `gpu_only=1`. So a VRAM-only expert's
`qf/q8/q4/s` host pointers are NULL/freed — its weights live ONLY on the GPU. Two code paths could dereference
those freed pointers and segfault:

1. **`backend_cuda.cu` `coli_cuda_tensor_upload`**: the `if(!weights) return -1;` guard ran BEFORE the
   `*tensor` cache-hit check. For a VRAM-only expert `weights` is NULL (freed slab) but `*tensor` already
   holds the valid GPU copy uploaded at pin time. The early `!weights` made `coli_cuda_matmul` fall through
   to the CPU path on freed/NULL pointers → **segfault during the first generated token** (right after the
   prompt was printed, matching the reported symptom).
   **Fix:** reorder so the `*tensor` cache-hit (reuse the device copy without requiring the freed host
   pointers) is checked FIRST; `!weights` now only blocks a *fresh* upload, never a reuse.

2. **`mimo.c` `matmul_qt`**: the `if(w->gpu_only) return;` guard lived INSIDE the `!omp_in_parallel()`
   branch. If a VRAM-only expert were matmul'd inside an OpenMP region the guard was skipped and the CPU path
   ran on the freed slab. **Fix:** pulled the `gpu_only` early-return OUTSIDE the `omp_in_parallel` check so
   it can never touch the freed host slab (defensive for `./glm`; `./mimo`'s expert loop is serial so the
   upload fix #1 is the actual `./mimo` fix, but #2 removes the latent `./glm` crash).

Also added a `SIGSEGV` backtrace handler (`execinfo.h`/`backtrace`) at the top of `main()` and `-g -rdynamic`
to the Makefile CFLAGS/LDFLAGS, so any future host crash prints a symbolized backtrace (no debugger here,
ASan is incompatible with the CUDA runtime).

### 20. Verification — segfault gone, VRAM tier serving experts

`COLI_CUDA=1 CUDA_DENSE=1 CUDA_EXPERT_GB=6 NGEN=8 ./mimo 64 4 8` on RTX 3060 12.9 GB / 20.8 GB RAM:
- **`mimo exited rc=0`** — no segmentation fault; generated real text (`Ecc una frase sulosa e rispos`).
- `[CUDA] complementary VRAM tier: 378 expert, VRAM 4.77 GB` loaded (budget fix: embed+lm_head no longer
  over-counted in `g_cuda_dense_projected`, so the tier gets real budget instead of 0).
- `[CUDA] resident set: 1234 tensor, 11.96 GB VRAM` — GPU near-full (12.9 GB).
- Expert hit-rate **27.2%** (up from 17% baseline). VRAM tier served **565 calls**.
- PROFILO-GPU: router 1.07s | cuda-copy(PCIe) 6.57s | cuda-compute 23.64s (of 30.21s matmul total).

### 21. Remaining bottleneck is HOST-RAM starvation, not GEMM or PCIe

Even with the VRAM tier working, throughput is **0.14 tok/s** because the per-layer expert cache cap is forced
to **3** (`[RAM_GB=20.8 auto] ... cap abbassato 64->3`), causing **658 expert reloads/token** and
`expert-disk 15.16s` of the 56.65s total. Why so little RAM for the cache:
- `densa residente 9.5 GB` is retained in **host** RAM (the dense tensors are uploaded to GPU but the host
  copy is kept), and
- the GPU VRAM is already ~full (11.96 GB), so no more experts can move to VRAM.

Net: only ~2.2 GB is left for the expert LRU → cap=3 → constant disk reloads. The real win is to **free the
host-dense copy after GPU upload** (reclaiming ~9.5 GB for the expert cache → cap would rise to ~20/layer,
dramatically raising hit-rate), or run on a GPU with more VRAM. PCIe copy (6.57s) and cuda-compute (23.64s)
are secondary once the cache stops thrashing.

### 22. P0 hardening (2026-07-11): eager dense, fail-loud gpu_only, auto expert budget

Code review of the complementary-tier work found several correctness/footgun issues that are now fixed
in-tree (before the next full-model re-measure):

1. **Eager dense upload + free host (`cuda_upload_dense_all`)** — densas were still lazy on first
   `matmul_qt`, so `pin_load` saw inflated free VRAM and raced experts against densas. Now after
   `model_init` (and after `g_draft` is resolved): upload every `cuda_eligible` dense tensor, free its
   host buffers, set `gpu_only=1`, and subtract from `resident_bytes` so `cap_for_ram` / AUTOPIN see the
   reclaimed RAM. Order is **dense → expert tier**. `embed` and `lm_head` are unmarked (never VRAM:
   gather path / better spent as expert slots). MTP densas upload only when `DRAFT>0`.

2. **`gpu_only` no longer returns silent zeros** — previous path could `return` without writing `y` if
   CUDA failed or if called under OpenMP (host slab already freed). Now: OpenMP + `gpu_only` →
   fatal exit; CUDA fail with no host copy → fatal exit; if host somehow still present → CPU fallback.
   Prefer loud failure over token corruption.

3. **No double-calloc of `gpu_pin`** — `model_init` allocates `gpu_pin`/`ngpin`; `pin_load` reuses them
   (and frees a previous tier if `pin_load` is called twice) instead of overwriting the pointers.

4. **Expert VRAM budget uses real free** — after eager dense, `remaining = free - 1.5 GB headroom`
   (no more projected-dense arithmetic that double-counted embed/lm_head per device).  
   `CUDA_EXPERT_GB`: unset + `COLI_CUDA=1` → **auto** (`-1`, fill free-headroom); `0` = off; `>0` = cap in GB.

5. **PCIe timer fixed** — `g_copy_sec` used to include kernel time because D2H is synchronous after
   launch. Now: H2D → kernel → `cudaDeviceSynchronize` → D2H; only H2D+D2H accumulate. PROFILE-GPU
   `cuda-copy` vs compute is trustworthy again.

6. **SEGV handler is Linux-only** (`#ifdef __linux__`) so macOS builds do not pull `backtrace`.

### 23. P0 re-measured on full MiMo (2026-07-11) — chat runs, ~0.50 tok/s

Full-model run on the reference box (RTX 3060 12 GB, ~23 GB RAM, WSL2, SNAP
`/root/mimo25_i4`, experts int4 / dense int8, `COLI_CUDA=1 CUDA_DENSE=1`,
`CUDA_EXPERT_GB` auto, `DIRECT=1 TOPP=0.6 THINK=0`).

**PROMPT mode** (`NGEN=24`, *"Write one short sentence about Rome."*):

| metric | value |
|---|---|
| load | 51 s (dense eager 5.8 s + pin 2 s + VRAM tier 9 s) |
| decode | **24 tok in 48.10 s → 0.50 tok/s** |
| hit-rate expert | **46.3%** |
| experts loaded / token | 235 (5.0 / layer of 47; TOPP=0.6 trims top-8) |
| RAM pin | 361 experts (4.6 GB) |
| VRAM complementary | **443 experts (5.59 GB)**, budget auto 5.6 GB |
| VRAM densas | 99 tensors ok, **0 fail**, freed **4.69 GB host** |
| resident set | 1428 tensors, **10.28 GB VRAM** |
| RSS mid-gen | 13.64 GB |
| LRU cap | 7 / layer (was 3 when host densas still resident) |
| expert-disk | 12.36 s |
| expert-matmul | 16.23 s |
| attention | 15.25 s |
| PROFILE-GPU | router 1.28 s · **cuda-copy 0.76 s** · cuda-compute 15.47 s |

Output was coherent English about Rome. CUDA path stable (rc=0, no SEGV).

**SERVE / chat** (same knobs, one turn): model READY in ~26 s after densas already
warm from prior process? / cold load ~25–40 s; streamed a full sentence on Rome.
`chat_peng.py` protocol works with the new binary (`COLI_CUDA=1 CUDA_DENSE=1` in env).

**Vs pre-P0 baselines on this box:**

| era | hit | expert-disk (scale) | tok/s | notes |
|---|---|---|---|---|
| densas GPU, host densas kept, pin≈VRAM duplicate (§20) | ~27–40% | high / cap=3 | 0.14–0.20 | host RAM starved |
| complementary tier, host densas still kept | ~52% (other prompt) | lower | wall ~4× densas-only | pre-eager free |
| **P0 eager+free densas + complementary (§23)** | **46%** | **12 s / 24 tok** | **0.50** | auto expert-GB, PCIe timer honest |

**Reading the profile:** with a warm-ish pin, matmul+attention (~31 s) already
dominate wall more than disk (12 s). Real PCIe copy is only **0.76 s** of the
16 s expert-matmul — so the next speed lever is **kernel/fusion**, not more
memcpy accounting. Hit-rate still ~half: more distinct residents (int2 / REPIN
on `gpu_pin`) remains the coverage lever.

**Still open (speed / correctness):** fuse expert GEMM (1 H2D/D2H per expert);
REPIN for `gpu_pin`; finding #18 oracle ~1-position drift; WSL2 I/O CPU cost.

### 24. Speed pack re-measure (2026-07-11 evening) — fuse + fast GEMV + SERVE bench

**Code landed (same day after §23):**

- `coli_cuda_swiglu`: one H2D of `x`, fused gate+up+silu (int4/int8), down GEMV, one D2H of `y`.
- Fast GEMV: shared-mem `x`, warp-per-output, shuffle reduce; sticky device-`x` (`REUSE_X`) for decode S=1.
- Attention Q·K / V-accumulate AVX2 (+FMA when available).
- SERVE defaults when env unset: `DIRECT=1`, `I4S=1`, `PILOT=1`, `REPIN=32`; VRAM expert headroom 1.5→1.0 GB.
- `chat_peng.py`: `--bench` / `--fast` / `--quality`, STAT protocol fixed (skip blank line after `END`), CUDA/PILOT/REPIN defaults.
- Unit test: `make cuda-test` → `q8/q4/q2/f32 + fused SwiGLU ok`.

**PROMPT mode** (same prompt as §23, `NGEN=24`, `COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 TOPP=0.6 THINK=0 TEMP=0.7`):

| metric | §23 P0 | §24 speed pack |
|---|---|---|
| tok/s (engine) | **0.50** (24 tok / 48.1 s) | **0.43** (19 tok / 44.0 s) |
| hit-rate | 46.3% | **50.5%** |
| VRAM experts | 443 / 5.59 GB | **483 / 6.09 GB** (tighter headroom) |
| RAM pin | 361 | 362 |
| expert-disk | 12.36 s | 11.21 s |
| expert-matmul | 16.23 s | 16.52 s |
| attention | 15.25 s | **12.69 s** |
| cuda-copy (PROFILE) | 0.76 s | 0.00 s† |
| RSS mid-gen | 13.64 GB | 13.66 GB |
| load (wall) | ~51 s | ~53 s |

† Async H2D/D2H no longer isolates PCIe in `g_copy_sec` (timer under-counts). Restore honest copy accounting later.

Output coherent: *"Rome is the Eternal Capital of Italy…"*. Exit 0.

**Reading:** attention improved (~2.5 s). Hit and VRAM expert count up slightly. **tok/s did not beat §23** on this box — matmul wall flat (~16.5 s); fuse/GEMV are correctness-preserving structure work but not yet a clear throughput win at 46–50% hit. Ceiling still hit-rate + disk + remaining CPU attention.

**SERVE / `chat_peng.py --bench --fast`** (1 warmup + 3 runs, `/reset` each, NGEN=24, chat template, target 1.0):

| run | tokens | tok/s (STAT) | hit | wall |
|---|---|---|---|---|
| warmup | 15 | 0.18 | 41% | 84 s |
| meas 1 | 18 | 0.21 | 43% | 86 s |
| meas 2 | 24 | 0.21 | 46% | 117 s |
| meas 3 | 17 | 0.19 | 43% | 88 s |
| **median** | — | **0.21** | **43%** | — |

**GATE 1.0 tok/s: FAIL** (shortfall ~0.79). Load READY ~30 s. Prefill each turn ~33 template tokens; STAT = `prod / (prefill+decode)`, so SERVE numbers are **not** comparable to PROMPT decode-only 0.43–0.50. Use PROMPT + `PROFILE=1` for kernel apples-to-apples; use `--bench` for end-to-end chat UX.

**How to reproduce:**

```bash
# PROMPT profile (findings table)
SNAP=/root/mimo25_i4 COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 TOPP=0.6 THINK=0 \
  NGEN=24 PROFILE=1 PROMPT='Write one short sentence about Rome.' ./c/mimo 64 4 8

# SERVE multi-run gate
SNAP=/root/mimo25_i4 python3 c/chat_peng.py --bench --fast --runs 3 --warmup 1 --ngen 24
# or: c/scripts/run_speed_bench.sh
```

**Still open (speed):** true single-kernel expert MLP; REPIN on `gpu_pin`; int2 experts; SWA ring KV (free ~1.5 GB for cache); honest async PCIe timer; native Linux vs WSL2 I/O; more host RAM. Quality/exactness: finding #18.

### 25. GPU-first hot store + MoE device accumulate (2026-07-11 night) — **0.55 tok/s**

**Code:**

- `coli_cuda_moe_begin/acc/end`: decode S=1 accumulates `y += w*swiglu(x)` on GPU with **one H2D + one D2H per MoE layer** (no per-expert sync/D2H).
- **GPU-first pin**: hottest experts from `.coli_usage` go to **VRAM first**, then next band to RAM pin (was reversed: hottest on CPU, cooler on GPU).
- Tighter RAM budget (92% MemAvailable, page-cache reserve 1.5 GB, `PC_GB=` override) → LRU cap ~11.
- VRAM headroom 0.5 GB → **522 experts / 6.59 GB** on 3060 12 GB.
- CUDA implies `I4S=1` when unset (CPU int4 decode uses IDOT).

**PROMPT** (`NGEN=24`, Rome sentence, `COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 TOPP=0.55 TEMP=0.7`):

| metric | §23 | §24 pack | **§25 GPU-first+acc** |
|---|---|---|---|
| **tok/s** | 0.50 | 0.43 | **0.55** |
| hit-rate | 46% | 50% | **60%** |
| VRAM experts | 443 | 483 | **522** (hottest) |
| VRAM calls / gen | — | 796 | **1310** |
| expert-disk | 12.4 s | 11.2 s | **8.9 s** |
| expert-matmul | 16.2 s | 16.5 s | **16.8 s** |
| attention | 15.3 s | 12.7 s | **13.4 s** |
| LRU cap | 7 | — | **11** |

**Gate 1.0 tok/s: still FAIL.** Best measured on this box: **0.55 tok/s** (~1.8 s/token). Physics on 23 GB + 3060 + WSL2:

```text
~9 s disk + ~17 s matmul + ~13 s attn  over 24 tok  ≈ 0.55 tok/s
To 1.0 need ≈2× less wall (more hit + faster GEMV/attn and/or more RAM / native Linux)
```

**Path to 1.0 (ordered):**

1. **More host RAM (32–64 GB)** — raise pin+LRU, hit → 80%+, cut disk toward 0.
2. **Faster GPU expert GEMV** (tensor-core / better tiling) — matmul is still ~17 s.
3. **SWA ring KV** — reclaim ~1.5 GB for experts.
4. **int2 experts** — half disk+VRAM bytes.
5. **Native Linux** — drop WSL I/O CPU tax (§18).

Reproduce: `TOPP=0.55 COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 NGEN=24 PROFILE=1 PROMPT='…' ./mimo 64 4 8`

### 26. Intelligent anticipatory cache (2026-07-11) — REPIN→VRAM + sticky + PILOT L+2

**Goal:** smarter *ahead-of-time* expert residency without changing tokens.

| Mechanism | Default (SERVE/chat) | Role |
|---|---|---|
| **Sticky PREFETCH** | ON (`PREFETCH=1`) | At layer L start, `WILLNEED` the experts this layer used on the *previous* token (`enr`/`eroute`) — free prior for next token |
| **PILOT** | ON | L+1 router-lookahead (~71% recall) |
| **PILOT_DEPTH** | **2** | Also WILLNEED L+2 (weaker prior, still free I/O) |
| **REPIN** | every 32 tokens | RAM pin swaps (as before) **plus VRAM `gpu_pin` swaps**: coldest GPU expert out, hottest non-resident in (disk→upload→free host), same heat hysteresis as `tier_pick_swap` |

Env overrides: `PREFETCH=0`, `PILOT_DEPTH=1`, `REPIN=0` (off) / `REPIN=n`.

**Does not change math** (I/O + residency only). Expected win: multi-turn / thrashing regimes (higher live hit, less cold disk). Cold single-shot PROMPT already GPU-first may gain little; chat is the target.

Also landed earlier same day: thread-local IDOT quant scratch (colibri #43) — fewer mallocs on CPU expert path.

### 27. Speed sprint (2026-07-11 night) — best **0.57 tok/s**; literature map

**Tried / measured:**

| Idea | Result |
|---|---|
| Multi-stream GPU experts + `atomicAdd` yacc | **Worse** (0.21–0.25 tok/s); contention + launch tax |
| `PILOT_DEPTH=2` on PROMPT | **Worse** (~45–67 s “other”); extra router matmuls unaccounted → now timed in `t_router`; default depth **1** |
| `TOPK=6` + `TOPP=0.5` | fewer experts/token but quality noise; not free win |
| Single-stream moe_acc + GPU-first + TOPP=0.55 **no PILOT** | **0.57 tok/s**, hit 60.5%, disk 8.6 s, matmul 16.4 s, attn 13.0 s |

**Kept (net positive stack):** GPU-first pin, moe_acc single-stream, heat_prefetch_top (WILLNEED hot non-residents at turn/re-pin), REPIN→VRAM, sticky PREFETCH, quant scratch, SERVE PILOT=1 depth 1.

**Literature (relevant, not all portable):**

- [ProMoE](https://arxiv.org/abs/2410.22134) / FineMoE: proactive expert maps → inspired `heat_prefetch_top`
- [DuoServe-MoE](https://arxiv.org/html/2509.07379v1): prefill vs decode schedules (future)
- [Pre-gated MoE](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf): predict experts earlier
- [fMoE](https://arxiv.org/html/2502.05370v1): fine-grained offload
- llama.cpp: dense on GPU, experts CPU — we already invert (hottest experts on VRAM)

**Path still required for 1.0 on this box:** more host RAM and/or faster expert GEMV (tensor cores) and/or int2 experts and/or native Linux. Soft ceiling ~0.6 with current 23 GB + 3060 + WSL2.

### 31. Trajectory bulk WILLNEED (2026-07-12) — predictive expert path

**Idea:** learn which experts co-activate (same layer next token + layer L→L+1), then bulk `posix_fadvise(WILLNEED)` the predicted path so disk misses become warm hits. **I/O only — never changes tokens.**

**Code (`TRAJ=1` default ON in SERVE):**

- `traj_observe_layer` while routing: Markov edges `tok[L][e→e']` and `lay[L][e→e'@L+1]`.
- After each decode step: `traj_commit_prev` + `traj_warm` (sticky + heat + Markov unroll `TRAJ_DEPTH`).
- Turn start: `heat_prefetch_top` also calls `traj_warm`.
- Knobs: `TRAJ=0` off, `TRAJ_K=8`, `TRAJ_DEPTH=2`. `chat_peng` enables TRAJ by default.

```bash
TRAJ=1 TRAJ_K=8 SERVE=1 … ./mimo 64 4 8
# stderr: [TRAJ] bulk expert path WILLNEED on (K=8 depth=2; TRAJ=0 off)
```

**Expected win:** multi-turn / chat (learned routes) — higher hit, less expert-disk. Cold single-shot PROMPT gains little until heat accumulates.

### 30. COLI_PROFILE + mem_watch polish (colibri #71 remainder, 2026-07-12)

**Code:**

- `COLI_PROFILE` / `PENG_PROFILE` → usage file `SNAP/.coli_usage.<sanitized>` so chat vs code heat maps do not share pin history. `chat_peng.py --profile NAME` sets it; `MEMWATCH=1` default in chat env.
- Autopin at full history: **85%** of expert budget (was 50%); override `PIN_FRAC=`.
- `mem_watch_pass`: safer `realloc`, heat_prefetch after pressure shrink; still on every SERVE turn + MORE.

```bash
COLI_PROFILE=chat SERVE=1 … ./mimo 64 4 8
# stderr: [PROFILE] COLI_PROFILE=chat -> /path/.coli_usage.chat
python3 c/chat_peng.py --fast --profile code
```

### 29. SWA KV ring + push toward 1.0 tok/s (2026-07-12) — **gate still FAIL**

**Code (F-11 speed):** SWA/MTP layers allocate `sliding_window` KV rows (ring, `pos % W`); full layers stay linear; RoPE still uses logical `pos`. `kv_bytes_at` matches ring → expert budget sees ~0.2 GB KV@4096 instead of ~1.9 GB. Default `PC_GB` 1.5→1.0.

**Effect at load (this box):**

| | before (§27) | after ring |
|---|---|---|
| KV budget @4096 | ~1.9 GB | **~0.2 GB** |
| LRU cap | 11 | **13** |
| RAM pin experts | ~365 | **~427–430 (5.4 GB)** |
| VRAM experts | 522 | 522 |

**PROMPT** (Rome, NGEN=24, TOPP=0.55, PILOT=0, CUDA stack):

| run | tok/s | hit | disk | matmul | attn |
|---|---|---|---|---|---|
| warm | **0.50** | 59% | 9.7 s | 15.5 s | 12.7 s |
| TOPP=0.45 | 0.43 | 54% | 10.8 s | **24.7 s** | 14.7 s |

Fewer experts (TOPP 0.45) **hurt**: fewer VRAM hits (hot experts from usage miss the thinner top-p set) → more host matmul.

**Physics (honest):** even if disk → 0 on the warm profile (~15.5+12.7+3.6 s / 21 tok) ≈ **0.66 tok/s**. **1.0 needs ~2× faster matmul+attn**, not only hit-rate. Soft ceiling on 23 GB + 3060 + WSL2 remains **~0.55–0.65** without more RAM or faster GEMV/attn.

**Gate 1.0: FAIL** (best still §27 **0.57**). Ring is still correct: frees budget for SERVE/long CTX and multi-turn.

### 28b. Product review F-01/F-02 (2026-07-12)

See [`docs/review-2026-07-11-product.md`](docs/review-2026-07-11-product.md) for the full F-01…F-21 audit (@ `1482e26`) and status table.

- **F-02 fixed:** `mimo_turn_render` uses capacity-safe `snappend`; returns `-1` on overflow; SERVE/TEMPLATE_DUMP reject without OOB read/write. Test: `python3 -m unittest tests.test_template_overflow`.
- **F-01 partial:** `st_init` validates header length vs file size, JSON object shape, offsets in-range, F32/BF16/F16 `nbytes==numel*esz`, duplicate names. U8 packed payloads only require non-empty when `numel>0`. Malformed fixtures in `tests/test_st.c` (fork).
- **F-15 fixed post-audit:** `mimo` links `$(CUDA_OBJ)` in current Makefile.

### 28. colibri #71 — GPU||CPU MoE block + mem_watch (2026-07-12)

Ported from [JustVugg/colibri#71](https://github.com/JustVugg/colibri/pull/71) (open PR: Windows + GPU tiers + dynamic RAM).

**Code (`c/mimo.c`):**

1. **GPU||CPU expert overlap (same 64-expert block)**  
   Phase A: wait load (if any) only for GPU-resident experts, queue `moe_acc` async.  
   Phase B: wait remaining loads + host/SwiGLU path while GPU queue runs until layer `moe_end`.  
   Per-expert waits keep `OVERLAP` load||compute (an all-loads-first draft serialised disk and was reverted).

2. **`mem_watch_pass` (dynamic LRU)**  
   After every SERVE turn and after `\x02MORE`: re-read `MemAvailable`.  
   - free &lt; 3.5 GB → shrink `ecap`, **real** LRU slab free (`eslot_release`)  
   - free &gt; 6.0 GB → grow `ecap` (realloc ecache)  
   - dead band 3.5–6 GB → no-op  
   `MEMWATCH=0` disables. Default ON.

3. **`repin_pass` after full turns** (was MORE-only) so live RAM/VRAM heat chase runs every boundary.

**SERVE smoke** (NGEN=8, one turn):

```text
[RAM] headroom: 6.8 GB free -> cap RAISED 11->12 (MEMWATCH=0 to disable)
[REPIN] RAM layer … / VRAM layer …   # after END, not only MORE
```

**PROMPT** (`TOPP=0.55`, NGEN=24, Rome, CUDA stack): **0.44–0.48 tok/s**, hit ~55–59%, not a clear beat of §27 **0.57**. Expected: single-shot PROMPT is already GPU-first; mem_watch/repin are multi-turn levers. Gate 1.0 still FAIL.

Reproduce:

```bash
# PROMPT
SNAP=/root/mimo25_i4 COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 TOPP=0.55 TEMP=0.7 \
  NGEN=24 PROFILE=1 PROMPT='Write one short sentence about Rome.' ./c/mimo 64 4 8

# SERVE (watch [RAM] / [REPIN] on stderr after END)
SERVE=1 MEMWATCH=1 REPIN=1 … ./c/mimo 64 4 8
```
