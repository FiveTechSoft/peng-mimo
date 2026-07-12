#!/usr/bin/env python3
"""Genera requests para SCORE mode (mimo.c run_score) desde texto fijo.

Formato por linea: "<ctxlen> <contlen> <id0> .. <id_{T-1}>"
ctxlen=1 -> se puntuan todas las posiciones 1..T-1 (perplexity de corpus).
Corpus mixto: prosa (README) + codigo (olmoe.c, archivo estable) para ver
degradacion no uniforme entre tokens faciles y de precision.
"""
import sys
from transformers import AutoTokenizer

SNAP = sys.argv[1] if len(sys.argv) > 1 else "/root/mimo25_i4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/score_req.txt"
NTOK = 768  # tokens por request

tok = AutoTokenizer.from_pretrained(SNAP)

prose = open("/mnt/c/peng-mimo/README.md", encoding="utf-8").read()[:8000]
code_lines = open("/mnt/c/peng-mimo/c/olmoe.c", encoding="utf-8").read().splitlines(keepends=True)
code = "".join(code_lines[100:320])

with open(OUT, "w") as f:
    for name, text in (("prose", prose), ("code", code)):
        ids = tok(text, add_special_tokens=False).input_ids[:NTOK]
        f.write(f"1 {len(ids)-1} " + " ".join(map(str, ids)) + "\n")
        print(f"{name}: {len(ids)} tokens", file=sys.stderr)
print(f"wrote {OUT}", file=sys.stderr)
