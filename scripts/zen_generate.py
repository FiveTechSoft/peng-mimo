#!/usr/bin/env python3
"""
zen_generate.py — dataset candidate generation via OpenCode Zen free API.

Calls https://opencode.ai/zen/v1/chat/completions (no API key required)
with courteous pacing and backoff. Designed to run in GitHub Actions
matrix jobs: --batch N --of M selects a strided slice of the seeds.

Input seeds: JSONL {"id", "prompt", ...}
Output:      JSONL {"id", "prompt", "response", "model", "ts"}

Usage:
    python scripts/zen_generate.py --seeds dataset/seeds/c_seed_prompts.jsonl \
        --out candidates.jsonl --batch 0 --of 4
"""

import argparse
import glob
import json
import sys
import time
import urllib.request
import urllib.error

ZEN_URL = "https://opencode.ai/zen/v1/chat/completions"
DEFAULT_MODEL = "mimo-v2.5-free,nemotron-3.5-lightning-free,hy3-free"
MAX_RETRIES = 5


def call_zen(prompt: str, models: list, max_tokens: int, delay: float) -> dict:
    """One completion, rotating across fallback models.

    429/5xx -> exponential backoff, then try next model.
    4xx (model gone / needs auth) -> skip to next model immediately.
    Courteous, not evasive: total attempts stay bounded.
    """
    last_err = None
    for model in models:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode()
        for attempt in range(MAX_RETRIES):
            req = urllib.request.Request(
                ZEN_URL, data=body,
                headers={"Content-Type": "application/json",
                         "User-Agent": "peng-mimo-dataset/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    resp = json.loads(r.read())
                    resp["_model_used"] = model
                    return resp
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (400, 401, 403, 404):
                    print(f"  HTTP {e.code} for {model}, trying next model",
                          file=sys.stderr)
                    break  # model unavailable: rotate
                if e.code == 429 or e.code >= 500:
                    wait = delay * (2 ** attempt) * 5
                    print(f"  HTTP {e.code} ({model}), backoff {wait:.0f}s "
                          f"(attempt {attempt+1}/{MAX_RETRIES})", file=sys.stderr)
                    time.sleep(wait)
                else:
                    raise
    raise RuntimeError(f"all models failed: {models} (last: {last_err})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True, help="JSONL file or glob (e.g. 'dataset/seeds/*.jsonl')")
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="model or comma-separated fallback chain")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--delay", type=float, default=3.0,
                   help="seconds between requests (courtesy pacing)")
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = p.parse_args()

    models = [m.strip() for m in args.model.split(",") if m.strip()]

    seeds = []
    for path in sorted(glob.glob(args.seeds)):
        with open(path, encoding="utf-8") as f:
            seeds.extend(json.loads(line) for line in f if line.strip())
    if not seeds:
        sys.exit(f"no seeds match: {args.seeds}")

    # strided slice for matrix parallelism
    mine = seeds[args.batch::args.of]
    if args.limit:
        mine = mine[:args.limit]
    print(f"batch {args.batch}/{args.of}: {len(mine)} seeds of {len(seeds)} "
          f"| models: {' -> '.join(models)}")

    ok = fail = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, seed in enumerate(mine):
            try:
                resp = call_zen(seed["prompt"], models, args.max_tokens, args.delay)
                msg = resp["choices"][0]["message"]
                content = msg.get("content") or ""
                finish = resp["choices"][0].get("finish_reason", "")
                if not content.strip():
                    # empty completion happens occasionally; one courteous retry
                    time.sleep(args.delay * 2)
                    resp = call_zen(seed["prompt"], models, args.max_tokens, args.delay)
                    msg = resp["choices"][0]["message"]
                    content = msg.get("content") or ""
                    finish = resp["choices"][0].get("finish_reason", "")
                if finish == "length":
                    # truncated output fails verification anyway; retry with more room
                    time.sleep(args.delay * 2)
                    resp = call_zen(seed["prompt"], models,
                                    args.max_tokens * 2, args.delay)
                    msg = resp["choices"][0]["message"]
                    content = msg.get("content") or ""
                    finish = resp["choices"][0].get("finish_reason", "")
                rec = {
                    "id": seed["id"],
                    "prompt": seed["prompt"],
                    "tags": seed.get("tags", []),
                    "response": content,
                    "reasoning": msg.get("reasoning_content") or "",
                    "model": resp.get("_model_used", models[0]),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                ok += 1
                print(f"[{i+1}/{len(mine)}] {seed['id']} ok "
                      f"({len(rec['response'])} chars)")
            except Exception as e:
                fail += 1
                print(f"[{i+1}/{len(mine)}] {seed['id']} FAILED: {e}", file=sys.stderr)
            if i < len(mine) - 1:
                time.sleep(args.delay)

    print(f"done: {ok} ok, {fail} failed -> {args.out}")
    sys.exit(0 if ok > 0 else 1)


if __name__ == "__main__":
    main()
