#!/usr/bin/env python3
"""
oss_instruct.py — OSS-Instruct seed generator.

Takes real source files, extracts fragments, and asks the Zen free API
to invent self-contained programming tasks that such code could solve
(the Magicoder OSS-Instruct idea, adapted to our pipeline).

Output seeds land in dataset/seeds/ and are picked up automatically by
the dataset.yml workflow (it globs dataset/seeds/*.jsonl).

Usage:
    python scripts/oss_instruct.py --src "c/*.c" --out dataset/seeds/oss_c.jsonl \
        --limit 10
"""

import argparse
import glob
import hashlib
import json
import re
import sys
import time
import urllib.request
import urllib.error

ZEN_URL = "https://opencode.ai/zen/v1/chat/completions"
DEFAULT_MODELS = "mimo-v2.5-free,nemotron-3.5-lightning-free,hy3-free"
MAX_RETRIES = 5

LANG_BY_EXT = {".c": "c", ".h": "c", ".py": "python", ".js": "javascript",
               ".prg": "harbour", ".ch": "harbour", ".rs": "rust"}

FRAGMENT_LINES = 60          # lines per fragment
MIN_FRAGMENT_CHARS = 300     # skip tiny fragments
MAX_FILE_BYTES = 200_000     # skip huge files

TASK_PROMPT = """Here is a fragment of real {lang} code from the file "{name}":

```{lang}
{fragment}
```

Invent ONE self-contained programming task (in {lang}) that code like this
fragment could be part of the solution for. Rules:
- The task must be understandable without seeing the fragment.
- Do NOT mention "this code", "the fragment" or the file name.
- Ask for a concrete implementation, not an essay.
- Do NOT include your reasoning or analysis.
- Output the task statement (2-4 sentences) prefixed by the line "TASK:"."""

REASONING_MARKERS = re.compile(
    r"thinking process|\*\*Analy|\*\*Deconstruct|\*\*Identify|"
    r"^\s*[-*]\s+\*\*|step \d|let me think", re.IGNORECASE | re.MULTILINE)


def extract_task(text: str) -> str:
    """Strip leaked chain-of-thought from reasoning models.

    Preferred: content after the "TASK:" marker. Fallback: drop paragraphs
    that look like analysis and keep the rest. Empty -> reject upstream.
    """
    m = re.search(r"TASK:\s*(.+)$", text, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
    else:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        kept = [p for p in paras if not REASONING_MARKERS.search(p)]
        candidate = "\n\n".join(kept).strip()
    # a real task statement shouldn't still smell like reasoning
    if REASONING_MARKERS.search(candidate):
        return ""
    return candidate


def lang_of(path: str) -> str:
    for ext, lang in LANG_BY_EXT.items():
        if path.lower().endswith(ext):
            return lang
    return "c"


def fragments(path: str, lines_per: int):
    """Yield (fragment_text, start_line) chunks of a source file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return
    for i in range(0, len(lines), lines_per):
        frag = "".join(lines[i:i + lines_per]).strip()
        if len(frag) >= MIN_FRAGMENT_CHARS:
            yield frag, i + 1


def seed_id(path: str, start: int) -> str:
    h = hashlib.sha1(f"{path}:{start}".encode()).hexdigest()[:8]
    return f"oss-{h}"


def norm_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", p.lower()).strip()


def call_zen(prompt: str, models: list, max_tokens: int, delay: float) -> str:
    """Completion text with rotation across fallback models (same policy
    as zen_generate.py: backoff on 429/5xx, skip model on 4xx)."""
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
                         "User-Agent": "peng-mimo-seeds/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    resp = json.loads(r.read())
                    return resp["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 403, 404):
                    print(f"  HTTP {e.code} for {model}, next model", file=sys.stderr)
                    break
                if e.code == 429 or e.code >= 500:
                    wait = delay * (2 ** attempt) * 5
                    print(f"  HTTP {e.code} ({model}), backoff {wait:.0f}s", file=sys.stderr)
                    time.sleep(wait)
                else:
                    raise
    raise RuntimeError("all models failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="glob of source files, e.g. 'c/*.c'")
    p.add_argument("--out", required=True, help="output seeds JSONL")
    p.add_argument("--existing", default="dataset/seeds/*.jsonl",
                   help="glob of existing seeds for dedup")
    p.add_argument("--model", default=DEFAULT_MODELS)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--limit", type=int, default=0, help="0 = all fragments")
    args = p.parse_args()

    models = [m.strip() for m in args.model.split(",") if m.strip()]

    # load existing seeds for dedup (ids + normalized prompts)
    seen_ids, seen_prompts = set(), set()
    for path in sorted(glob.glob(args.existing)):
        if path == args.out:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        seen_ids.add(rec.get("id"))
                        seen_prompts.add(norm_prompt(rec.get("prompt", "")))
        except OSError:
            pass

    files = sorted(glob.glob(args.src))
    if not files:
        sys.exit(f"no source files match: {args.src}")

    written = skipped_dup = failed = 0
    count = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for path in files:
            lang = lang_of(path)
            name = path.replace("\\", "/").split("/")[-1]
            for frag, start in fragments(path, FRAGMENT_LINES):
                if args.limit and count >= args.limit:
                    break
                count += 1
                sid = seed_id(path, start)
                if sid in seen_ids:
                    skipped_dup += 1
                    continue
                prompt = TASK_PROMPT.format(lang=lang, name=name, fragment=frag)
                try:
                    raw = call_zen(prompt, models, 2048, args.delay)
                    task = extract_task(raw)
                except Exception as e:
                    failed += 1
                    print(f"[{count}] {sid} FAILED: {e}", file=sys.stderr)
                    continue
                if len(task) < 40:
                    failed += 1
                    print(f"[{count}] {sid} rejected (no clean task)", file=sys.stderr)
                    continue
                if norm_prompt(task) in seen_prompts:
                    skipped_dup += 1
                    continue
                rec = {"id": sid, "prompt": task,
                       "tags": [lang, "oss-instruct"], "source": f"{name}:{start}"}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                seen_ids.add(sid)
                seen_prompts.add(norm_prompt(task))
                written += 1
                print(f"[{count}] {sid} ({lang}, {name}:{start}) -> {task[:70]}...")
                time.sleep(args.delay)

    print(f"done: {written} new seeds, {skipped_dup} dups skipped, {failed} failed -> {args.out}")
    sys.exit(0 if written > 0 or skipped_dup > 0 else 1)


if __name__ == "__main__":
    main()
