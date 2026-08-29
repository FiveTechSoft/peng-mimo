# dataset/ROADMAP — distillation project

Goal: a small local model (1.5B class) that matches MiMo-V2.5-class
quality **in our programming domain** (C focus, plus Python/Harbour/JS),
trainable on a consumer GPU (RTX 3060 12 GB) and servable by our own
engines (llm_inference.c / llama.cpp).

Guiding principle: the teacher's job is to produce *trajectories*;
the verifier's job is to decide what is true; the student's job is to
reproduce the verified behavior. Volume follows quality, never the
reverse.

## Phase 0 — pipeline skeleton (DONE)

- [x] Seed set: 38 hand-curated prompts (`seeds/*.jsonl`)
- [x] `scripts/zen_generate.py` — generation via OpenCode Zen free API,
      direct calls (no gateway), courteous pacing, fallback chain
      (mimo-v2.5-free → nemotron-3.5-lightning-free → hy3-free)
- [x] `scripts/verify_dataset.py` — per-language verification
      (gcc / py_compile / node --check); only compiling code survives
- [x] `.github/workflows/dataset.yml` — nightly generate → verify →
      commit to `dataset/verified/`
- [x] First batch landed: 32 rows (2026-08-29)

## Phase 1 — seed growth (IN PROGRESS)

- [x] `scripts/oss_instruct.py` — derives tasks from real source code
      (Magicoder OSS-Instruct idea), TASK: marker extraction, dedup
- [x] `.github/workflows/seeds-oss.yml` — weekly seed growth from `c/*.c`
- [ ] Point `--src` at more repos: dreaming (`*.c`, `*.py`), agents
      (`*.prg` — Harbour is a niche with no public datasets; our edge)
- [ ] Seed diversity review: algorithmic tasks vs idioms vs debugging
      vs refactoring; add negative examples (bug + fix pairs) generated
      mechanically (off-by-one, use-after-free, leaks)
- [ ] Target: 2,000+ seeds → ~1,000 verified rows/week sustainable

## Phase 2 — dataset quality hardening

- [ ] Deduplication across batches (normalized prompt + response hash)
- [ ] Beyond syntax: compile+link+run for examples with a `main`,
      optional test harness execution in CI
- [ ] Manual review lane for `skipped` languages (harbour, rust, sql)
- [ ] Response quality filter: length bounds, no leaked reasoning in
      answers, code/prose ratio check
- [ ] Target: 5–10k verified rows with measured pass-rate history

## Phase 3 — reality benchmark (the exam)

- [ ] Collect 50–100 real prompts where local models failed and MiMo
      succeeded (from actual daily use — the spec for everything below)
- [ ] Reference answers from peng (slow but sovereign) and/or Zen
- [ ] Scoring harness (like `bench_topp_ppl.sh`): every candidate model
      and later every student checkpoint gets a comparable score
- [ ] Candidates to score now: Qwen3.8-27B (int4+offload),
      OpenCodeReasoning-Nemotron-1.1-14B, Qwen2.5-Coder-7B
- [ ] Decision gate: if an existing model passes ~90% of the exam,
      skip training and use it (hybrid with peng for the hard tail)

## Phase 4 — training (QLoRA on RTX 3060)

- [ ] SFT format export: verified JSONL → chat template (student-dependent)
- [ ] Student base: Qwen2.5-Coder-1.5B (or current best ≤3B at the time)
- [ ] Unsloth QLoRA config for 12 GB VRAM; 1–2 nights per run
- [ ] Eval before/after against the Phase 3 exam — no promotion without
      a measured win
- [ ] Iterate: more data / better mix / second epoch only if the exam
      says so

## Phase 5 — serving

- [ ] Merge LoRA, convert to GGUF (llama.cpp tooling)
- [ ] Run in llama.cpp first; evaluate port to `llm_inference.c`
      (architecture deltas: QKV biases, tokenizer)
- [ ] Local latency target: ≥40 tok/s on CPU for interactive use
- [ ] Hybrid router (optional): local student first, escalate to
      peng/MiMo when confidence is low

## Phase 6 — publication (optional)

- [ ] Move dataset to a HF dataset repo (fivetech org) once it outgrows
      git (~50 MB) — would be the first verified code dataset distilled
      from MiMo-V2.5
- [ ] Publish the student + a writeup in findings.md style

## Risks and honest constraints

- Zen free can change terms or vanish: fallback chain covers models,
  paid API covers volume (~$100/50M tokens), peng covers sovereignty.
- Free-tier rate limits cap nightly volume; the bottleneck is seeds,
  not generation — Phase 1 matters more than Phase 2.
- A 1.5B student will not match a 311B teacher in general — only in
  the trained domain. The exam (Phase 3) defines that domain; without
  it we are flying blind.
