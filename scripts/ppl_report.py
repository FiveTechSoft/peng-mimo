#!/usr/bin/env python3
"""Tabla final de la matriz TOPP/TOPK: perplexity + acuerdo top-1 vs BASE."""
import math

DIR = "/root/ppl_topp"
CFGS = ["BASE", "TP070", "TP060", "TP055", "TP050", "TK6", "TK5", "HYB"]

base_am = [l.split() for l in open(f"{DIR}/BASE.argmax")]
hdr = ("cfg", "ppl_prosa", "ppl_code", "agree_prosa", "agree_code")
print("%-6s %9s %8s %11s %10s" % hdr)
for c in CFGS:
    lps = []
    for ln in open(f"{DIR}/{c}.lp"):
        p = ln.split()
        if len(p) == 3 and p[1] == "767":
            lps.append(float(p[0]))
    am = [l.split() for l in open(f"{DIR}/{c}.argmax")]
    ag = [100.0 * sum(a == b for a, b in zip(am[i], base_am[i])) / len(base_am[i])
          for i in range(2)]
    print("%-6s %9.2f %8.2f %10.1f%% %9.1f%%" %
          (c, math.exp(-lps[0] / 767), math.exp(-lps[1] / 767), ag[0], ag[1]))
