# dataset/ — programming distillation pipeline

Goal: build a verified programming dataset distilled from MiMo-V2.5
(C focus, plus Python / Harbour / JS / Rust / SQL), to later fine-tune
a small local student (QLoRA on a consumer GPU).

## Pipeline

```
dataset/seeds/*.jsonl          seed prompts (hand-curated + grown)
        │
        ▼  .github/workflows/dataset.yml  (nightly or manual)
scripts/zen_generate.py        calls OpenCode Zen free API directly
        │                      (mimo-v2.5-free, no key, no gateway)
        ▼  candidates-*.jsonl
scripts/verify_dataset.py      per-language check:
        │                      C→gcc, Python→py_compile, JS→node --check
        ▼                      only passing code survives
dataset/verified/prog-<date>.jsonl   committed back by the workflow
```

## Design rules

- **No gateway**: runners call Zen directly (GitHub Actions egress IPs are
  from the Azure pool, different every run). No fixed IP, no CORS issue
  (server-to-server, CORS only applies to browsers).
- **Courteous use**: pacing + exponential backoff in `zen_generate.py`.
  This is a free API — we stay well inside fair use; bulk volume belongs
  to paid APIs, peng handles the local premium slice.
- **Verification is the quality bar**: compiler acceptance, not teacher
  reputation, decides what enters the dataset. Languages without a checker
  (harbour, rust, sql) are kept but flagged `skipped` for manual review.

## Local usage

```bash
# generate 2 candidates (smoke test)
python scripts/zen_generate.py --seeds "dataset/seeds/*.jsonl" \
    --out /tmp/cand.jsonl --limit 2

# verify them
python scripts/verify_dataset.py --in "/tmp/cand.jsonl" --out /tmp/verified.jsonl
```

## Next steps

- grow `dataset/seeds/` (OSS-instruct style: derive prompts from real code
  in this repo and dreaming)
- extend verification beyond syntax: compile+link+run for examples with tests
- merge verified batches into a training-ready SFT format (chat template)
