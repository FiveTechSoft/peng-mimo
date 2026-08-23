#!/usr/bin/env python3
"""Colmena: unique routed experts vs 256 x layers (STATS dump: layer expert count)."""
from __future__ import annotations

import sys
from collections import defaultdict


def load(path: str):
    hits = defaultdict(dict)  # layer -> {eid: count}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            L, e, c = int(p[0]), int(p[1]), int(p[2])
            hits[L][e] = hits[L].get(e, 0) + c
    return hits


def band(hits, lo, hi):
    uniq = sel = 0
    slots = 0
    per = []
    for L in range(lo, hi + 1):
        d = hits.get(L, {})
        u = len(d)
        s = sum(d.values())
        uniq += u
        sel += s
        slots += 256
        per.append((L, u, s))
    return uniq, sel, slots, per


def pareto(hits, lo, hi):
    pairs = []
    for L in range(lo, hi + 1):
        for e, c in hits.get(L, {}).items():
            pairs.append(c)
    pairs.sort(reverse=True)
    tot = sum(pairs) or 1
    out = []
    acc = 0
    for i, c in enumerate(pairs, 1):
        acc += c
        if i in (1, 5, 10) or abs(i / len(pairs) - 0.2) < 1 / max(len(pairs), 1) or i == len(pairs):
            out.append((i, 100.0 * acc / tot))
    # also 10%, 20%, 50% of unique
    n = len(pairs)
    acc = 0
    marks = {}
    for i, c in enumerate(pairs, 1):
        acc += c
        for frac in (0.1, 0.2, 0.5):
            if i == max(1, int(round(frac * n))) or (frac not in marks and i >= frac * n):
                marks[frac] = 100.0 * acc / tot
    return n, marks, tot


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "hive_stats.txt"
    hits = load(path)
    layers = sorted(hits)
    print(f"# HIVE {path} layers_with_hits={layers[:3]}..{layers[-1:] if layers else []}")
    print("# band          uniq   slots  cover%   selections")
    for name, lo, hi in (
        ("L01-L07 early", 1, 7),
        ("L09-L47 late", 9, 47),
        ("L01-L47 all MoE", 1, 47),
    ):
        u, s, slots, _ = band(hits, lo, hi)
        print(f"# {name:16} {u:5d}  {slots:5d}  {100.0*u/slots:5.1f}%   {s}")
        n, marks, tot = pareto(hits, lo, hi)
        print(
            f"#   pareto unique={n}  top10%→{marks.get(0.1,0):.0f}% sel  "
            f"top20%→{marks.get(0.2,0):.0f}%  top50%→{marks.get(0.5,0):.0f}%"
        )
    print("# late per-layer unique/256:")
    _, _, _, per = band(hits, 9, 47)
    for L, u, s in per:
        bar = "#" * (u // 8)
        print(f"#   L{L:02d}  {u:3d}/256  sel={s:4d}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
