#!/usr/bin/env python3
"""SCORE requests per-domain for usage collection (§42 multi-profile prune).

Collection corpora are deliberately DISJOINT from the evaluation corpus
(score_req.txt = README + olmoe.c): different files, different domains.
Each output feeds one COLI_PROFILE run whose STATS dump becomes that
domain's expert-usage heat map.
"""
import sys
from transformers import AutoTokenizer

SNAP = sys.argv[1] if len(sys.argv) > 1 else "/root/mimo25_i4"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "/root"
NTOK = 768

tok = AutoTokenizer.from_pretrained(SNAP)

def slurp(path, lo=0, hi=None):
    t = open(path, encoding="utf-8", errors="ignore").read()
    return t[lo:hi] if hi else t[lo:]

DOMAINS = {
    # English tech prose — different docs than the README used for eval
    "prose": slurp("/mnt/c/peng-mimo/docs/tao.md")[:6000]
             + slurp("/mnt/c/peng-mimo/docs/corriente-peng.md")[:6000],
    # C code — glm.c, not the olmoe.c used for eval
    "code_c": "".join(slurp("/mnt/c/peng-mimo/c/glm.c").splitlines(keepends=True)[200:500]),
    # Python — a different language entirely
    "code_py": slurp("/mnt/c/peng-mimo/c/tools/convert_fp8_to_int4.py")[:9000]
               + slurp("/mnt/c/peng-mimo/c/tools/make_mimo_oracle.py")[:6000],
}

for name, text in DOMAINS.items():
    ids = tok(text, add_special_tokens=False).input_ids
    out = f"{OUTDIR}/score_{name}.txt"
    with open(out, "w") as f:
        # two requests of NTOK each = ~1.5k tokens of routing signal per domain
        for a in range(0, min(len(ids), 2 * NTOK), NTOK):
            chunk = ids[a:a + NTOK]
            if len(chunk) < 64:
                break
            f.write(f"1 {len(chunk)-1} " + " ".join(map(str, chunk)) + "\n")
    print(f"{name}: {min(len(ids), 2*NTOK)} tokens -> {out}", file=sys.stderr)
