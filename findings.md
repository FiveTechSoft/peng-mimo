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

### 43. AVX-512 int4 kernel ported from colibri #95 — matmul −13%, ppl equal-or-better (2026-07-14)

Upstream review (53 new colibri commits) yielded one direct port: the AVX-512
int4→float accumulator (colibri `4b1d0e3`, #95) — 32 weights/iter on two FMA
chains, tree reduction. Our Xeon has AVX-512F/BW (no VNNI, so the existing IDOT
VNNI branches stay dead); the hot single-token host-expert GEMV is `matmul_i4`
(float path, `g_i4s=2` gate) — exactly where this kernel lands.

Measured (real 311B, deterministic SCORE + interleaved §37 speed pairs):

| metric | AVX2 order (BASE) | I4_ACC512=1 |
|---|---|---|
| ppl prose / code | 12.40 / 2.073 | **12.19 (−1.7%) / 2.076 (+0.2%)** |
| expert-matmul (clean runs) | 18.3–18.9 s | **16.2 s (−13%)** |
| tok/s (clean pair) | 0.40–0.43 | 0.48 |
| tiny oracle TF/greedy | 31/32, 15/20 | 31/32, 15/20 (same) |

Tree reduction accumulates less rounding than sequential AVX2 — "more accurate
AND faster", matching upstream's Xeon Silver numbers. Not bit-identical to the
old order, so: **default OFF** (bit-exact gates stay the baseline), **auto-ON
under `SPEED=1`/`TAO=1`** (explicit `I4_ACC512=0/1` always wins). Selftest:
`I4_ACC512_TEST=1 SNAP=x ./mimo`.

Also from the upstream review: colibri's serve heap-overflow (#117) does NOT
affect peng (`mimo.c` sizes the attention score buffer by true `nt` with heap
fallback); their native-Windows port (#131) and `quant_ablation.py` rotation/int3
tooling (#132) are roadmap candidates (§18 path and int3+zstd+EKEEP stacking
respectively).

### 42. Can we make a small MiMo? Expert redundancy, usage, and the EKEEP prune matrix (2026-07-13)

Three linked experiments toward a "MiMo-lite" (fewer experts, no retraining). Net
verdict: **experts are not redundant, and naive usage-based pruning destroys exactly
the domains your usage history under-represents.** The infrastructure for doing it
right (per-domain usage profiles) already exists in the engine.

**42.1 Inter-expert redundancy (`scripts/expert_redundancy.py`)** — dequantized
`gate_proj` of all 256 experts, pairwise cosine on a fixed 131k-coord subsample,
layers 1/24/46 + cross-layer control:

| layer | mean \|cos\| | max pair | pairs > 0.9 |
|---|---|---|---|
| 1 (early) | **0.401** | 0.65 | 0 |
| 24 (mid) | 0.0085 | 0.031 | 0 |
| 46 (late) | 0.0145 | 0.064 | 0 |
| cross-layer control | 0.0027 | 0.012 | — |

No near-duplicates anywhere → **intra-layer expert merging is not viable.** Early
layers share a strong common component (top pair: 34% identical int4 bytes vs 0.4%
random baseline) — delta-coding early layers against a per-layer centroid could
squeeze a few extra points on top of §41's zstd, but only in those few layers.
This shared structure is also part of what zstd is already collecting blindly.

**42.2 Usage distribution (`scripts/usage_analysis.py` on `.coli_usage`, 1.96 M
selections)** — the long tail is fatter than the literature suggests: only 2.5% of
experts (297/12,032, ~3.7 GB) have zero recorded use; per-layer Gini 0.615; top-16
serves ~30-38% (not 80%). Keeping 192/128/96 per layer covers 98.4% / 90.8% /
82.9% of historical calls. Caveat flagged at the time: history ≈ 5,200 tokens of
mostly-prose benchmarks.

**42.3 EKEEP runtime prune + SCORE gate** — new env `EKEEP=n` (mimo.c): at boot,
rank each layer's experts by usage history, mask all but the top n; the router
never selects (and the engine never reads) the rest. Simulates a pruned container
with zero disk writes, fully reversible. Guards: `EKEEP<topk` raised, no-history
warning. No-op regression: without EKEEP, tiny oracle stays bit-exact (TF 31/32,
greedy 15/20). Masked selection applied in `moe()` top-k and PILOT lookahead.

SCORE matrix (§39/§40 harness, same fixed 767+767-token corpus, deterministic —
BASE reproduced §40 to the last decimal):

| config | experts/layer | disk equiv. | ppl prose | Δ | ppl code | Δ |
|---|---|---|---|---|---|---|
| BASE | 256 | 165 GB | 12.40 | — | 2.07 | — |
| EK192 | 192 | 111 GB | 13.36 | **+7.8%** | 3.46 | **+67%** |
| EK128 | 128 | 74 GB | 20.58 | +66% | 10.60 | **+411%** |
| EK96 | 96 | 55 GB | 28.76 | +132% | 23.10 | **+1016%** |

**The headline finding:** degradation is wildly asymmetric. Prose survives EK192
cheaper than TOPK=6 (+7.8% vs +11%) — but code explodes (+67%, worse than the
worst §40 trim) because the usage history is prose-dominated and the pruner
removed the code specialists. The "rarely used expert" is not dead weight; it is
the specialist for whatever your history lacks. Frequency-based pruning measured
as **workload bias made permanent**.

**Design implication:** a viable MiMo-lite needs per-domain coverage, not raw
frequency — e.g. union of top-n across several `.coli_usage.<profile>` heat maps
(code/chat/multilingual), or usage gathered on a deliberately diverse corpus. The
engine already supports profiles (`COLI_PROFILE`). Gate pipeline agreed for any
finalist: broad 10-20k-token corpus → agreement@5/KL if borderline → router-stats
sanity (entropy, per-expert token share) → minimal downstream (HumanEval subset)
before publishing any pruned container.

**42.4 Multi-profile union criterion (in progress)** — machinery landed:

- `scripts/make_domain_corpora.py`: per-domain SCORE requests, deliberately
  **disjoint from the evaluation corpus** (prose = tao/corriente docs not README;
  C = glm.c not olmoe.c; Python = converter tools) — the mask must not be tuned
  on the exam.
- Usage collection: `COLI_PROFILE=collect_<d>` (fresh profile → no contamination)
  + `STATS=<file>` dumps that domain's pure expert heat map after a SCORE pass.
- `scripts/build_ekeep_mask.py`: keep the union of each profile's top-K per layer.
- Engine `EKEEP_MASK=<file>`: explicit keep-list ("layer eid" lines); absent layer
  = all eligible; validates ≥ topk survivors per layer; overrides EKEEP. No-op
  regression: tiny oracle TF 31/32 unchanged.

**Result — the union criterion eliminates the code collapse entirely:**

| config | experts/layer (avg) | disk equiv. | ppl prose | Δ | ppl code | Δ |
|---|---|---|---|---|---|---|
| BASE | 256 | 165 GB | 12.40 | — | 2.073 | — |
| EK192 (naive frequency) | 192 | 111 GB | 13.36 | +7.8% | 3.46 | **+67%** |
| **UNION208** (top-128 × 4 profiles) | 208 | 134 GB | 13.19 | **+6.4%** | 2.068 | **−0.2%** |
| **UNION179** (top-96 × 4 profiles) | 179 | 104 GB | 13.37 | **+7.8%** | 2.036 | **−1.8%** |

Code goes from +67% damage (naive, same prose cost) to *indistinguishable from
BASE* (−0.2% / −1.8% ≤ noise) once each domain contributes its own top-K to the
union. UNION179 gives a **−37% expert store (165 → 104 GB; ~75 GB with §41 zstd)
at prose +7.8% and code intact** — strictly dominating naive EK192 (same prose
cost, 7 GB smaller, code undamaged).

Why the big picture matters: pruning does not reduce experts loaded per token
(router still picks 8), so direct tok/s gain is limited to better cache coverage
of a smaller pool (est. +10–25%, unmeasured). The real prize is the 128 GB RAM
roadmap row: a 104 GB expert store **fits entirely in RAM** — disk pole gone,
§37 physics ≈ 1.2 tok/s, gate 1.0 passed on a ~$250 upgrade.

**Speed (same-day §37 pairs):** UNION179 mask 0.47/0.55 tok/s vs BASE 0.44/0.44
— **median +16%**, inside the predicted +10–25% cache-coverage band. The mask
wins both pairs; best run shows the mechanism (matmul 16.4→14.2 s, disk
7.8→7.2 s: redirected calls land on resident experts more often). Free speed on
top of the disk savings — and unlike TOPP/TOPK trims, quality-neutral (§42.4
table).

Caveats before believing more than screening: same 1.5k-token evaluation corpus
as §40 (the collection corpora are disjoint, but the exam is narrow); pending
gates per the agreed pipeline — broad 10–20k-token corpus, agreement@5/KL,
router stats, minimal downstream — before any physical pruned container ships.

### 41. Lossless zstd on int4 experts: ~25% fewer disk bytes, bit-exact (2026-07-13)

**Origin:** evaluated a DFloat11-style proposal (lossless BF16 exponent compression,
[arXiv 2504.11651](https://arxiv.org/abs/2504.11651)) — 16 → ~11.2 bits/weight via
sign+exponent entropy coding. **Not applicable as-is:** peng experts are already int4;
11.2 bits/weight would be ~2.8× *larger* and ~3× slower on the disk-bound path. But the
underlying question transfers: *do our int4 expert bytes compress losslessly at all?*

**Measured on the real container** (`/root/mimo25_i4`, 128–512 MB samples at multiple
offsets of `out-00004` / `out-00010`, WSL2, 16 cores):

| measurement | value |
|---|---|
| zstd-1 ratio (4 chunks, 2 shards) | **74.2–74.8%** |
| zstd-3 ratio | identical to zstd-1 (±0.02%) |
| zstd-1 ratio, per-expert-size frame (13 MB) | **74.7%** |
| byte entropy of expert data | 5.89 bits/byte → 73.7% order-0 floor |
| zero bytes | 0.0% (no padding artifact) |
| decompress, 1 thread (CLI) | 0.95 GB/s |
| decompress, 8 parallel processes | **3.41 GB/s** aggregate (> 2.75 GB/s NVMe line rate; CLI spawn tax included — in-process libzstd should do better) |

**Why it compresses:** int4 quantized weights cluster near zero → skewed 4-bit symbol
distribution. zstd-1 already sits on the order-0 entropy floor (74.7% vs 73.7% ideal),
so higher levels buy nothing — the redundancy is symbol skew, not repeats. (The tiny
random-init models compressed to only ~89%; real trained weights are the signal.)

**Cold-token math:** 4.7 GB reads → ~3.5 GB compressed. Read 1.71 s → 1.27 s at
2.75 GB/s. Decompress 4.7 GB output at 3.4+ GB/s ≈ 1.38 s, overlappable with reads in
the existing async loader pool (cores are idle while IO-blocked). Pipelined estimate:
**0.6 → ~0.72–0.79 tok/s, lossless, bit-exact** — a §40-frontier shift with zero
quality cost, stackable with TOPK/TOPP trims. Bonus: expert store 165 → ~123 GB.

**Landed (2026-07-13):** zstd-1 frames inside the safetensors container (per-tensor
`"nb"` key + `__metadata__.peng_zstd="1"`), `st.h` decompress-on-read, `expert_load`
coalesced frame pread → decompress into slab. Repack tool `c/tools/repack_zstd.py`
(streaming, atomic, `--verify`). Spec: `docs/superpowers/specs/2026-07-13-zstd-expert-pack-design.md`.

**Measured results (full 311B container):**

- **Size: 151.8 → 109.0 GiB (71.8%), all 17 shards verified byte-exact.** That is
  **2.87 effective bits/weight, lossless** — smaller than a hypothetical int3 (3.0
  bits) with zero quality cost.
- **Bit-exactness at full scale:** TEMP=0 PROMPT produces identical text on both
  containers ("Rome, the Eternal City,"), same experts/token (658.0), same VRAM
  tier behavior. Tiny oracle: TF 31/32, greedy 15/20 — identical to uncompressed,
  across `OVERLAP=0/1` and `DIRECT=1`.
- **Speed: LOSS of ~18% on the reference box.** Same-day warm §37 protocol:
  i4 0.46/0.38/0.34 tok/s (disk 7.7–8.2 s) vs i4z 0.36/0.33/0.31/0.30 (disk
  13.9–15.6 s). `LOADNICE=0/5` (new knob: loader-thread nice) and `OVERLAP_T=8`
  do NOT recover it — `OVERLAP_T=8` is much worse (0.21, contention).

**Why the §41 estimate was wrong:** the "cores are idle while IO-blocked" premise
does not hold — the OVERLAP pipeline already fills cores with expert matmul during
loads. Decompression is ~1.05 s/token of NEW CPU work (4 GB/token at ~0.95 GB/s ×
4 loader threads) on a box whose host matmul already needs ~1.25 s/token of the
same cores. Nothing to hide it behind; priority games don't create capacity. This
also falsifies the "dequant compute hides in the disk-wait shadow" assumption
(relevant upstream: colibri #81 rotation-preconditioned int2 has the same cost model).

**Where the compressed container DOES win:** storage/download (26% less; published:
[`fivetech/MiMo-V2.5-colibri-peng-int4-zstd`](https://huggingface.co/fivetech/MiMo-V2.5-colibri-peng-int4-zstd),
109 GB, all shards verified; `repack_zstd.py --unpack` converts back to plain int4 locally), boxes with spare cores relative to NVMe speed (native Linux without
the WSL2 I/O CPU tax may qualify — untested), and RAM-pin-all configs (128 GB
roadmap: decompress once at pin time, zero steady-state cost, 43 GB less disk).
**Default stays uncompressed for streaming on this box.** Future rescue idea:
upload compressed frames over PCIe (−28% traffic) + GPU decompress (nvcomp zstd)
+ GPU matmul — DFloat11's actual pattern; roadmap-level work.

### 40. Expert-trim quality matrix: fixed TOPK beats adaptive TOPP (2026-07-13)

**Setup:** SCORE mode (teacher-forcing, §39 harness), fixed corpus 767+767 tokens
(prose = README, code = olmoe.c), real MiMo int4. `ppl = exp(−logprob/tokens)`;
`agree` = top-1 argmax agreement vs BASE per position. One run per config
(deterministic: same input, same routing).

| cfg | ~experts/layer | ppl prose | ppl code | Δ prose | Δ code | agree prose | agree code |
|---|---|---|---|---|---|---|---|
| BASE top-8 | 8.0 | **12.40** | **2.07** | — | — | 100% | 100% |
| TOPK=6 | 6.0 | 13.81 | 2.17 | **+11%** | **+5%** | 78.4% | 91.7% |
| TOPK=5 | 5.0 | 15.28 | 2.31 | +23% | +12% | 75.9% | 88.4% |
| TOPP=0.7 | ~7.1 | 14.73 | 2.40 | +19% | +16% | 74.3% | 87.4% |
| TOPP=0.6 | ~5.8 | 17.59 | 2.73 | +42% | +32% | 68.4% | 84.9% |
| TOPP=0.55 | ~5.2 | 21.10 | 3.01 | **+70%** | **+45%** | 62.2% | 82.7% |
| TOPP=0.5 | ~4.7 | 23.35 | 3.30 | +88% | +59% | 58.5% | 79.9% |

**Findings:**

1. **Fixed top-k dominates adaptive top-p at equal expert budget.** TOPK=6 costs
   +11%/+5% ppl; TOPP=0.6 (fewer experts on average: ~5.8) costs 4-6× more. The
   intuition "let easy tokens use fewer experts" is backwards on MiMo's sigmoid
   router: TOPP trims hardest exactly on flat-distribution tokens — the uncertain,
   decision-heavy positions where the tail experts matter most. TOPK removes the
   same 2 lowest-weight experts everywhere, which is far more benign.
2. **`SPEED=1` default `TOPP=0.55` is expensive: +70% prose ppl.** Speed measured
   (2026-07-13, Rome NGEN=24, TAO/SPEED/CUDA stack, warm run of 2):

   | trim | experts loaded/tok (per-layer) | tok/s | Δppl prose |
   |---|---|---|---|
   | none (top-8) | 470 (10.0) | 0.29–0.32 | — |
   | TOPK=6 | 352 (7.5) | **0.36** | +11% |
   | TOPP=0.7 | 280 (6.0) | **0.44** | +19% |
   | TOPP=0.55 | 208 (4.4) | **0.49** | +70% |

   Speed tracks experts-loaded almost linearly (matmul 31s→28s→20s→15s). So the
   trims form a quality/speed frontier, not a free win: TOPK=6 is the quality
   point (+11% ppl, +13% speed), TOPP=0.7 the balanced point (+19% ppl, +38%
   speed), TOPP=0.55 the speed point (+70% ppl, +53% speed). `SPEED=1` keeping
   TOPP=0.55 is defensible for demos; for real use prefer TOPP=0.7 or TOPK=6.
3. **Hybrid `TOPK=6 TOPP=0.7` tested and discarded (2026-07-13):** ppl prose
   19.20 (+55%), code 2.60 (+26%), agree 67.1%/84.6%; speed 0.45 tok/s with
   ~220 experts/tok. Same speed as plain TOPP=0.7 (0.44) but 3× the quality
   hit — the 0.7 mass cut applies to the already-capped 6-expert mass, so it
   trims deeper (≈TP055–TP060 territory). Composition doesn't dominate;
   the three-point frontier above stands.
4. Code degrades less than prose in relative ppl (denser routing consensus), but
   absolute code ppl is so low (2.07) that even +5% is visible in agreement.
5. Reproduce: `scripts/make_score_corpus.py` + `scripts/bench_topp_ppl.sh` +
   `scripts/ppl_report.py`.

### 39. SWA ring corrupted batch prefill — found by oracle, fixed (2026-07-12)

**Symptom:** SCORE mode (teacher-forcing log-likelihood, built for the TOPP/TOPK quality matrix) returned ~uniform logprobs (~10 nats/token, ppl ≈ 25000) and argmax predictions that were pure noise (`。。。`, 0/767 top-1 on real MiMo).

**Isolation (tiny oracle):** same request on `mimo_tiny_i4` vs `ref_mimo.json`:

| gate | before fix | after fix | pre-§29 baseline |
|---|---|---|---|
| TF prefill vs oracle | **3/32** | **31/32** | 31/32 |
| greedy 20 tok | 11/20 | **15/20** | 15/20 |
| SCORE argmax vs oracle | 0/31 | **30/31** | n/a (new) |

Bisect over worktrees: good at `187b452`, broken at `4596b8e` (§29, SWA KV ring).

**Root cause:** `attention()` writes the whole chunk's K/V into the ring **before** the scoring loop. With ring active, position `p+1` writes slot `(p+1) % W`, evicting logical position `p+1−W` — which is still inside position `p`'s window. Any batch with `S ≥ 2` that crosses the window corrupts earlier in-batch positions. Decode (`S=1`) is untouched, and sequences shorter than `W` never wrap — which is why Rome-style short PROMPT runs (9–33 tok « W=128) looked fine and §29–§38 speed numbers stand, while any prefill > 128 tokens (long prompts, SERVE with history, TF/SCORE) silently produced garbage.

**Fix (`c/mimo.c` `attention()`):** when `ring && S>1`, the current chunk's K/V goes to a linear scratch (`Ksc/Vsc`); scoring reads scratch for `t ≥ pos_base` and the ring for history; after scoring, the last `min(W,S)` rows are flushed into the ring. Decode path unchanged.

**New tooling:**

- `SCORE_DUMP=<file>` — per-position argmax dump in SCORE mode (top-1 agreement between configs).
- `scripts/make_score_corpus.py` — fixed 768-tok prose + 768-tok code corpus → SCORE requests.
- `scripts/bench_topp_ppl.sh` — perplexity matrix {BASE, TOPP 0.7/0.6/0.55/0.5, TOPK 6/5} for the expert-trim study.

**Moral:** the token-exact oracle gate caught in minutes what "output looks plausible" missed all day. Long-prefill gate (S > W) should join the validation set — the tiny oracle only covers S=32 with W=8 by luck.

### 38. Physical pathpack (disk-order channels) + merged fadvise (2026-07-12)

**Context:** After §37 (best **0.60 tok/s**), next levers named were *physical pathpack*, *attention*, and the **1.0 tok/s** gate. This entry is the attempt at **runtime physical pathpack** (no full 152 GB shard rewrite).

#### Logical vs physical pathpack

| | Logical (§34) | Physical runtime (§38) | Full shard repack (future) |
|--|---------------|------------------------|----------------------------|
| Order | habit / Markov walk | **disk order** `(fd, off)` of habit experts | rewrite safetensors layout |
| Prefetch | per-expert `fadvise` | **merged ranges** on same fd | sequential `pread` natural |
| Tokens | bit-exact | bit-exact | bit-exact if values unchanged |
| Cost | low | low–medium | high (hours + disk) |

#### Code

- **`expert_disk_key(m, layer, eid)`** — key from `gate_proj.weight` via `st_find` → `(fd << 48) | (off >> 8)`.
- **`pathpack_rebuild`** — among experts with usage heat and/or TRAJ edges, **sort by disk key**; write `g_pp_ord` / `g_pp_pos`. Log: `[FLOW] pathpack rebuilt PHYSICAL …`.
- **Boot:** always rebuild physical after usage/traj load, then `pathpack_save` → `SNAP/.coli_pathpack`.
- **`st_prefetch_phys`** (`st.h`) — sort tensor list by `(fd, off)`, merge spans if same fd and gap ≤ **2 MB**, one `posix_fadvise(WILLNEED)` per merged span.
- **`expert_prefetch` / `expert_prefetch_list`** — collect gate/up/down + `.qs`, then `st_prefetch_phys`.
- **`pathpack_thaw`** — collect ±`FLOW_R` neighbors, one bulk list prefetch (not N separate storms).
- **`ov_submit`** — before loaders run, physical WILLNEED of the miss set (still fills `ws[s]` in original miss-slot order for correct `ov_wait`).
- **`TRAJ_WARM_EVERY=3`** default under `TAO=1` (was 2) to cut AUX further.

#### Smoke (same box: WSL2, ~23 GB RAM, RTX 3060 12 GB, SNAP `/root/mimo25_i4` ext4)

Protocol: `TAO=1 SPEED=1 PILOT=0 TEMP=0 COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 NGEN=24`  
prompt ≈ `Write one short sentence about Rome.` · pin + GPU-first warm.

| | §37 best (throttle) | §38 physical pathpack |
|--|---------------------|------------------------|
| **tok/s** | **0.60** (project best) | **0.50** |
| hit-rate | ~54% | ~58% |
| expert-disk | ~14.4 s | ~13.8 s |
| attention | ~9.1 s | ~10.2 s |
| expert-matmul | ~4.6 s | ~5.0 s |
| other | ~5.9 s | ~16.0 s |
| traj_warm (AUX) | ~3.8 s | ~0.5 s |

**Verdict:** physical packing is **correct engineering** (neighbors ≈ nearby on media; merged WILLNEED). It did **not** set a new throughput record. **Do not claim 0.50 > 0.60.** Project best remains **0.60 tok/s** (§37 / README).

**Why not faster here:** habit experts are still sparse in the shard; merging helps only when tensors are already near. Cold half of MoE work remains disk-bound. Extra bulk prefetch can add syscall/`other` noise. Full **container repack** (rewrite tensors in pathpack order) is still the real sequential layout win.

#### Still needed for 1.0 on this box

1. More host RAM (hit 50% → 80%+).  
2. Optional offline **safetensor repack** by physical pathpack order.  
3. Faster attention path (PROFILE attn ~9–10 s).  
4. Native Linux (drop WSL I/O tax, §18).

**Invariant:** I/O residency and fadvise order only — **no change to logits/tokens**.

### 37. PROFILE-AUX + throttle traj_warm/pathpack (2026-07-12)

**Problem:** NGEN=24 TAO run at 0.40 tok/s had `other ≈ 28 s` (~half wall) unaccounted.

**Code:**

- Model timers: `t_traj_warm`, `t_pathpack`, `t_persist` → `PROFILE-AUX: …`
- `TRAJ_WARM_EVERY` (default **2** on SERVE/SPEED/TAO): traj_warm only every N decode steps.
- `PATHPACK_EVERY` (default **8** SPEED/TAO, else 4): skip full pathpack rebuild on every `usage_save`; `PATHPACK_FORCE=1` forces rebuild.

```bash
# see where former "other" went
… ./mimo 64 4 8
# PROFILE: …
# PROFILE-AUX: traj_warm X | pathpack Y | persist Z
```

**Smoke (TAO=1 SPEED=1 PILOT=0 NGEN=24 Rome, pin+GPU-first, 2026-07-12):**

| | Before throttle | After |
|--|-----------------|-------|
| tok/s | 0.40 | **0.60** |
| other | ~28 s | **5.9 s** |
| traj_warm (AUX) | (hidden) | **3.8 s** |
| pathpack in-gen | every save | **0** (throttled) |

Residual other ~6 s + disk 14 s + attn 9 s still dominate; next lever = attn GPU + disk hit, or `TRAJ_WARM_EVERY=3`.

### 36. TAO=1 — wu wei + sacred geometry (2026-07-12)

Meta-profile: flow with the residual, do not force; harmonic knobs when unset.

- `TAO=1` → SPEED/TRAJ/FLOW on, `ENERGY` auto with **φ fraction (0.618)** of free VRAM.
- Fibonacci: `TRAJ_K=5`, `FLOW_R=3`, `REPIN=55`, traj warm budget 55; **`DRAFT=0`**.
- Docs: `docs/tao.md`, `docs/sacred-geometry.md`, `docs/diagrams/sacred-geometry-mimo.svg`.
- Explicit env always wins. Log: `[TAO] wu wei + sacred proportions (φ, Fibonacci)…`

### 35. ENERGY — channel snow → pure VRAM (2026-07-12)

**Metaphor made real:** free weight potential into GPU compute.

- After pin_load + pathpack: `pathpack_energy_ignite` walks packing **position** (heads of every layer channel first).
- Loads non-resident experts, `qt_cuda_upload`, `expert_cpu_free` → `gpu_only` (same as GPU-first tier).
- Budget: `ENERGY=-1` auto ≈ 85% free VRAM after pin (default with FLOW); `ENERGY=0` off; `ENERGY=n` GB cap.
- Complements heat-based GPU-first (usage) with **channel-continuity** ignition (traj bed).
- Log: `[ENERGY] liberated N channel experts -> VRAM +X GB …`

Does not change tokens — only residency. Soft ceiling still disk/attn when VRAM already full from pin.

### 34. FLOW pathpack channel thaw (2026-07-12) — Corriente in the engine

**Transcendence step:** stop WILLNEED by random expert id; thaw **along the habit channel**.

- `pathpack_rebuild` / `.coli_pathpack`: greedy order from TRAJ edges + usage heat per layer.
- `pathpack_thaw(layer, eid)`: `posix_fadvise` neighbors ±`FLOW_R` (default 2) on that order.
- Wired into sticky PREFETCH and `traj_warm`; rebuild+save with `usage_save`.
- Knobs: `FLOW=1` (default with TRAJ/SERVE/SPEED), `FLOW=0` off, `FLOW_R=2`.
- Analyzer: `python3 c/scripts/path_pack_analyze.py --snap …` also writes `.coli_pathpack`.
- Manifesto/diagram: `docs/corriente-peng.md`, `docs/diagrams/corriente-peng.svg`.

**Still I/O-only (bit-exact tokens).** Physical shard repack remains future work; logical channel thaw is live.

### 33. Expert bitmaps (2026-07-12) — the other map

**Intuition:** lists of slots are the slow story. **Bits** are the real map.

| Bitmap | Shape | Role |
|---|---|---|
| `res_bits` | `n_layers × 4×u64` (256 experts) | resident = VRAM ∪ RAM pin ∪ LRU — O(1) `expert_resident` |
| `pref_bits` | same | already WILLNEED'd this epoch — dedupe PILOT ∪ TRAJ ∪ sticky ∪ block readahead |

- Rebuild `res_bits` after pin_load, LRU promote, REPIN, mem_watch shrink.
- Clear `pref_bits` once per PROMPT / SERVE turn (within a turn: first hint wins).
- `expert_prefetch`: no-op if resident **or** already hinted → kills duplicate `posix_fadvise` storms without shrinking the useful prior.
- Also fixed: residency now includes **LRU** (old `expert_resident` only checked pin/gpu).

```bash
# no new knobs — always on with the model
# stderr still shows [TRAJ] willneed_calls=… (now much closer to unique experts)
```

**Why it fits Vedanta/binary:** presence is 0/1; the continuous scores only *choose* which bits to set; the connected path is OR of masks, not another list scan.

### 32. SPEED unity (2026-07-12) — GEMV denser + lean TRAJ + profile

**Goal:** give peng-mimo a coherent speed body without fadvise storms.

| Change | Why |
|---|---|
| **int4 GEMV / gate+up byte-strided** | one weight load → two nibble MACs; expert-matmul **16.4s → ~4.9s** on NGEN=24 |
| **CUDA_HEADROOM 0.35 GB** (was 0.5) | pack more GPU-first experts on 12 GB; `CUDA_HEADROOM_GB=` |
| **TRAJ budget** | max ~64 `expert_prefetch`/warm (boot 128); top-2 lay edges only — uncapped TRAJ_K=16 caused ~1300 pref/token → `other` >> compute |
| **SPEED=1** | `TOPP=0.55`, TRAJ lean K=6, REPIN=32; optional `TOPK=6` |
| **REPIN I/O → t_edisk** | honest PROFILE (was hiding in `other`) |
| **chat_peng --fast** | sets `SPEED=1` |

```bash
SPEED=1 TRAJ=1 PILOT=0 COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 \
  PROMPT='Write one short sentence about Rome.' NGEN=24 ./mimo 64 4 8
# 2026-07-12: 0.51 tok/s | hit 60% | disk 10.6s | matmul 4.9s | attn 11.1s | other 20s
# GPU-first 534 exp / 6.7 GB + RAM pin ~9 GB from .coli_usage
python3 c/chat_peng.py --fast --profile chat
```

**Soft ceiling still ~0.55–0.65** on 23 GB + 3060 + WSL2; 1.0 needs more RAM / native Linux / faster attn. Matmul is no longer the long pole when experts are warm.

### 31. Trajectory bulk WILLNEED (2026-07-12) — predictive expert path

**Idea:** learn which experts co-activate (same layer next token + layer L→L+1), then bulk `posix_fadvise(WILLNEED)` the predicted path so disk misses become warm hits. **I/O only — never changes tokens.**

**Code (`TRAJ=1` default ON in SERVE):**

- `traj_observe_layer` while routing: Markov edges `tok[L][e→e']` and `lay[L][e→e'@L+1]`.
- After each decode step: `traj_commit_prev` + `traj_warm` (sticky + heat + Markov unroll `TRAJ_DEPTH`).
- Turn start: `heat_prefetch_top` also calls `traj_warm`.
- **Persist:** `SNAP/.coli_traj` or `.coli_traj.<COLI_PROFILE>` (atomic write with `usage_save` each SERVE turn; load at boot).
- Knobs: `TRAJ=0` off, `TRAJ_K=8`, `TRAJ_DEPTH=2`. `chat_peng` enables TRAJ by default.

```bash
TRAJ=1 TRAJ_K=8 COLI_PROFILE=chat SERVE=1 … ./mimo 64 4 8
# stderr: [TRAJ] bulk … on
#         [TRAJ] loaded N edges from …/.coli_traj.chat   (2nd session)
#         [TRAJ] saved N edges -> … (willneed_calls=…)
```

**Expected win:** multi-turn / chat across restarts — higher hit, less expert-disk. Cold single-shot PROMPT gains little until heat accumulates.

**Smoke (2026-07-12, SNAP=/root/mimo25_i4, COLI_PROFILE=smoketest, TRAJ=1, NGEN=3 then NGEN=1):**

- Session 1: planted 4 edges → `[TRAJ] loaded 4` → gen → `[TRAJ] saved 13532` (~215 KB).
- Session 2: `[TRAJ] loaded 13532` → gen → `[TRAJ] saved 14004`. Atomic rename + profile path OK.
- **Boot warm:** after pin+cap, `heat_prefetch_top` + Markov-without-sticky → `[TRAJ] boot warm +1848 hints` before first prefill. Session reload: `loaded 14004` then boot warm then save.

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

**Gate 1.0: FAIL** (best still §27 **0.57**). Ring frees budget for SERVE/long CTX and multi-turn.

> **Correction (§39):** the ring as landed here corrupted **batch prefill** for any chunk crossing the window (S ≥ 2, pos ≥ W). Short-prompt speed runs were unaffected (S « W=128), but long prefill/TF/SCORE produced garbage until the §39 fix (scratch chunk + ring flush).

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

---

### 44. Deployed MiMo-V2.5 container corruption — pre-fix per-rank scale-grid converter (2026-07-16)

**Symptom.** The deployed `mimo25_i4` (and its byte-identical twin `mimo25_i4_v2`,
both built 2026-07-11) produced *coherent-looking but wrong* text on every prompt:
`1+1= → "1+1=Question<think>Let's…"`, `2+2= → "2+2=加倍加倍…"`,
`The capital of France is → "\t-icht die ungen. So."`. Logits were shape-sane
(top-5 `TAB / ， / 。 / \n / 的`) — so this was **corrupt weights**, not an
architecture bug.

**Engine proven correct first (ruled out the motor).**
- Tiny F32 oracle (generated by `make_mimo_oracle.py`): teacher-forcing
  **32/32 positions** match the reference (`mimo_ref`).
- Tiny int4 oracle (same, converted): **31/32** (one benign flip at pos 23).
- Tokenizer (`tok_mimo_cli`) matches the reference tokenizer **exactly**
  (`[16,10,16,28]`, `[785,6722,315,9625,374]`).
- int4 NEON matmul (even→low nibble, odd→high nibble) and `qkv_degroup` (nr=4)
  verified. The tiny oracles are small enough that their qkv has <128 rows/rank,
  so they never exercise the per-rank scale subdivision — which is exactly why
  they passed while the real model failed.

**Root cause: the container's `qkv_proj` was quantized with the pre-fix converter.**
MiMo-V2.5's fused `qkv_proj` concatenates `NR = min(kv_heads_full, kv_heads_swa) = 4`
TP-rank blocks; its fp8 `weight_scale_inv` is emitted **per rank**
(`108` scale rows for a full layer, not `ceil(13568/128)=106`). The old converter
applied a flat `repeat_interleave` grid, which misaligns the scales of ranks 1..3.
The bug signature is unmistakable and localizes the fault precisely:

```
rebuilt-from-fp8  vs  deployed container, per rank-block of qkv_proj (full attn layers)
  L5   rank_max = [0.0000, 0.6966, 0.3701, 0.4136]   # rank 0 bit-identical, ranks 1-3 corrupt
  L11  rank_max = [0.0009, 0.1427, 0.1165, 0.1460]
  L17  rank_max = [0.0007, 0.1134, 0.1253, 0.0947]
```
Rank 0 is exact in every layer (its scale is correct under both grids); ranks
1..3 diverge by up to `0.70` in dequantized space. `o_proj` and **all experts**
(`gate/up/down` for every expert) are **bit-identical** between rebuilt and
deployed → only the fused qkv is affected, exactly as the per-rank grid predicts.
The converter fix itself was already landed (commit `a3a64bc` / `13b95c9,
2026-07-11`); the *deployed artifact* was built before it.

**Fix.** Rebuild the container `qkv_proj` from the fp8 source with the current
converter, leaving the (verified-correct) experts / `o_proj` / norms untouched.
Result `mimo25_i4_fixed` generates correct text:
`"The capital of France is Paris. The capital of Germany is Berlin. The capital
of Italy is Rome. The capital of Spain is Madrid. The"` (CPU, greedy).

**Regression guard — `c/tools/verify_mimo_qkv.py`.** Re-dequantizes the fp8
`qkv_proj` with the (correct) per-rank scale grid and compares, per rank-block,
to the int4 container's dequantized qkv. A correct container matches on **all**
ranks; a corrupt one diverges on ranks 1..NR-1. Catches this class of bug without
running the full model. Verified: `mimo25_i4_fixed` → PASS (rank_max ≈ 0.002 on
all ranks); `mimo25_i4` → FAIL (ranks 1-3).

```bash
python3 c/tools/verify_mimo_qkv.py --int4 <container_dir> --fp8 <fp8_src_dir> \
    [--layers 5,11,17,23] [--threshold 0.05] [--nr 4]
# exit 0 = PASS (all ranks within threshold), 1 = FAIL
```

**Test hardware (GB10 superchip).** Single NVIDIA GB10 (Grace + Blackwell) node:
20× ARMv8 (aarch64), **128 GB LPDDR5X unified CPU↔GPU memory** (121 GiB usable,
~80 GiB free) — *not* 1.8 TB of RAM (that is the NVMe). NVIDIA GB10 GPU,
compute 13.0, **sm_121**, 130.6 GB addressable VRAM (unified). 1.8 TB NVMe SSD,
sequential read **~1.2 GB/s**. Ubuntu 6.17.0 aarch64. The 311 GB int4/int8 model
streams from NVMe — it does **not** fit in the 128 GB unified memory.

**Speed after the fix (same hardware, greedy, 24 tokens, `mimo25_i4_fixed`):**

| config | tok/s | note |
|---|---|---|
| CPU (`COLI_CUDA=0`) | **0.85** | int8 NEON, pages weights from NVMe |
| GPU `CUDA_DENSE=1` (auto expert cache ~30 GB) | 1.88 | GB10 sm_121 |
| GPU `CUDA_EXPERT_GB=80` | 3.14 | expert hit 96% |
| GPU `CUDA_EXPERT_GB=100` | **4.82** | expert hit 98.8%, 51 GB cached in unified VRAM |

Bottleneck at 4.82 tok/s is mixed: `expert-disk` ~1.0 s + `expert-matmul` (GPU)
~1.4 s + `attention` ~1.2 s. Removing the disk pole (fit a pruned/zstd container
in the 128 GB unified memory — see roadmap §A1/A2) is the next lever.
```
