#!/usr/bin/env python3
"""
verify_dataset.py — multi-language verification of generated candidates.

Extracts the code block from each response, detects the language
(fence tag or seed tags), and runs the appropriate checker:
  c          -> gcc -fsyntax-only -Wall -std=c11
  python     -> python -m py_compile
  javascript -> node --check   (if node available)
  others     -> marked "skipped" (kept, but flagged unverified)

Only code passing its checker enters the verified dataset.
Skipped-language records are written too, flagged, for manual review.

Usage:
    python scripts/verify_dataset.py --in "candidates*.jsonl" --out verified.jsonl
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

FENCE_RE = re.compile(r"```(\w*)\s*\n(.*?)```", re.DOTALL)
FENCE_OPEN_RE = re.compile(r"```(\w*)\s*\n(.*)$", re.DOTALL)
CODE_HINT_RE = re.compile(r"\b(int|void|char|float|size_t)\s+\w+\s*\(")

LANG_ALIASES = {
    "c": "c", "h": "c",
    "py": "python", "python": "python", "python3": "python",
    "js": "javascript", "javascript": "javascript", "node": "javascript",
    "rs": "rust", "rust": "rust",
    "sql": "sql",
    "prg": "harbour", "harbour": "harbour", "xbase": "harbour",
    "cpp": "cpp", "c++": "cpp", "hpp": "cpp",
}

KNOWN_LANGS = {"c", "cpp", "python", "javascript", "rust", "sql", "harbour"}


def extract_code(response):
    """Return (code, fence_lang). Longest fenced block wins; tolerates a
    truncated response whose closing fence is missing; last resort, raw text."""
    blocks = FENCE_RE.findall(response)
    if blocks:
        lang, code = max(blocks, key=lambda b: len(b[1]))
        return code, LANG_ALIASES.get(lang.lower(), lang.lower() or None)
    m = FENCE_OPEN_RE.search(response)
    if m and ("#include" in m.group(2) or ";" in m.group(2) or "import " in m.group(2)):
        return m.group(2), LANG_ALIASES.get(m.group(1).lower())
    if "#include" in response or CODE_HINT_RE.search(response):
        return response, "c"
    return None, None


def detect_lang(rec, fence_lang):
    if fence_lang in KNOWN_LANGS:
        return fence_lang
    for tag in rec.get("tags", []):
        if tag in KNOWN_LANGS:
            return tag
    return fence_lang or "unknown"


def check_c(code, std="c11"):
    cc = "gcc" if std == "c11" else "g++"
    return run_checker([cc, "-fsyntax-only", "-Wall", f"-std={std if std=='c11' else 'c++17'}"],
                       code, suffix=".c" if std == "c11" else ".cpp")


def check_python(code):
    return run_checker([sys.executable, "-m", "py_compile"], code, suffix=".py",
                       compile_mode=True)


def check_node(code):
    if not shutil.which("node"):
        return None, "node not installed"
    return run_checker(["node", "--check"], code, suffix=".js")


def run_checker(cmd, code, suffix, compile_mode=False):
    """Write code to a temp file and run the checker. compile_mode appends the
    file path to cmd (py_compile style); otherwise the path is already the last arg."""
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        full = cmd + [path] if compile_mode or cmd[-1].startswith("-") else cmd + [path]
        r = subprocess.run(full, capture_output=True, text=True, timeout=60)
        return r.returncode == 0, r.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "checker timeout"
    finally:
        os.unlink(path)


CHECKERS = {"c": lambda c: check_c(c, "c11"), "cpp": lambda c: check_c(c, "c++"),
            "python": check_python, "javascript": check_node}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inputs", required=True, help="glob of candidate JSONL files")
    p.add_argument("--out", required=True)
    p.add_argument("--keep-errors", action="store_true",
                   help="also write rejected records to <out>.rejected.jsonl")
    args = p.parse_args()

    files = sorted(glob.glob(args.inputs))
    if not files:
        sys.exit(f"no files match: {args.inputs}")

    total = no_code = passed = failed = skipped = 0
    rejects = []
    with open(args.out, "w", encoding="utf-8") as out:
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    total += 1
                    rec = json.loads(line)
                    code, fence_lang = extract_code(rec.get("response", ""))
                    if code is None:
                        no_code += 1
                        continue
                    lang = detect_lang(rec, fence_lang)
                    rec["lang"] = lang
                    checker = CHECKERS.get(lang)
                    if checker is None:
                        skipped += 1
                        rec["verify"] = {"status": "skipped", "lang": lang}
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue
                    res = checker(code)
                    if res[0] is None:  # checker unavailable
                        skipped += 1
                        rec["verify"] = {"status": "skipped", "lang": lang,
                                         "reason": res[1]}
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue
                    ok, err = res
                    rec["verify"] = {"status": "pass" if ok else "fail",
                                     "lang": lang, "stderr": err[:2000]}
                    if ok:
                        passed += 1
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    else:
                        failed += 1
                        rejects.append(rec)

    pct = 100.0 * passed / total if total else 0
    print(f"verified: {passed}/{total} pass ({pct:.0f}%) | "
          f"fail: {failed} | skipped-lang: {skipped} | no-code: {no_code}")
    print(f"-> {args.out}")

    if args.keep_errors and rejects:
        rej_path = args.out + ".rejected.jsonl"
        with open(rej_path, "w", encoding="utf-8") as f:
            for r in rejects:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"rejects -> {rej_path}")

    sys.exit(0 if (passed + skipped) > 0 else 1)


if __name__ == "__main__":
    main()
