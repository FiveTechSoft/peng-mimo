#!/usr/bin/env python3
"""Real Markov predictability of expert routing from .coli_traj (issue #6).

The engine already learns trajectory edges (TRAJ, mimo.c §3118): for each
(layer, expert) it keeps the top-8 successors with counts, both same-layer
next-token ("tok") and layer L -> L+1 same-token ("lay"), persisted in
SNAP/.coli_traj. This tool answers the question the "Grok plan" raises with
real data instead of a synthetic simulation:

    How predictable is expert routing from history alone, and is a learned
    Markov prefetch worth anything on top of PILOT's free 71% router recall?

Metrics (weighted by edge counts):
  - top-K coverage: E[ P(actual successor is in the top-K predicted) ]
    = expected recall of a prefetcher that warms the top-K successors.
  - effective successors (perplexity of the conditional distribution).
  - frequency baseline: coverage of just re-warming the same expert and of
    the per-layer usage top-K (what sticky PREFETCH / PIN already do).

Caveat: the engine stores at most TRAJ_SUC=8 successors per (layer, expert)
with weakest-eviction, so the conditional distributions are truncated;
top-K coverage for K<=8 is (slightly) optimistic, K>8 is not measurable.

Usage: python scripts/traj_analysis.py [/root/mimo25_i4/.coli_traj]
"""
import sys
from collections import defaultdict

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "/root/mimo25_i4/.coli_traj"
KS = (1, 2, 4, 8)


def load(path):
    edges = {"tok": defaultdict(dict), "lay": defaultdict(dict)}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5 or p[0] == "#":
                continue
            kind, l, e0, e1, c = p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])
            edges[kind][(l, e0)][e1] = c
    return edges


def analyze(edge_map, label):
    cov = {k: 0.0 for k in KS}
    tot_weight = 0
    self_cov = 0.0
    perpl = []
    per_layer = defaultdict(lambda: [0.0, 0])   # top-2 coverage, weight
    for (l, e0), succ in edge_map.items():
        counts = np.array(sorted(succ.values(), reverse=True), dtype=np.float64)
        T = counts.sum()
        if T <= 0:
            continue
        p = counts / T
        for k in KS:
            cov[k] += p[:k].sum() * T
        self_cov += succ.get(e0, 0)
        perpl.append(2 ** (-(p * np.log2(p)).sum()))
        per_layer[l][0] += p[:2].sum() * T
        per_layer[l][1] += T
        tot_weight += T
    n = len(edge_map)
    print(f"\n=== {label}: {n} (layer, expert) sources, {tot_weight:,} observed transitions ===")
    for k in KS:
        print(f"  top-{k} coverage (expected prefetch recall): "
              f"{100 * cov[k] / tot_weight:6.2f}%")
    print(f"  self-transition coverage (sticky re-warm):      "
          f"{100 * self_cov / tot_weight:6.2f}%")
    print(f"  effective successors: mean {np.mean(perpl):.2f}, "
          f"p50 {np.percentile(perpl, 50):.2f}, p90 {np.percentile(perpl, 90):.2f}")
    layers = sorted(per_layer)
    sample = layers[:: max(1, len(layers) // 8)]
    print("  per-layer top-2 coverage (sample):",
          ", ".join(f"L{l}={100 * per_layer[l][0] / per_layer[l][1]:.0f}%"
                    for l in sample))
    return {k: cov[k] / tot_weight for k in KS}


def main():
    edges = load(PATH)
    print(f"{PATH}")
    tok = analyze(edges["tok"], "tok edges (same layer, token t -> t+1)")
    lay = analyze(edges["lay"], "lay edges (layer L -> L+1, same token)")
    print("\n=== verdict vs PILOT (router lookahead, measured 71.1% top-8 recall) ===")
    print(f"  Markov lay top-4 coverage: {100 * lay[4]:.1f}%  |  "
          f"tok top-4: {100 * tok[4]:.1f}%")
    print("  PILOT runs the real router one layer ahead; a history-only")
    print("  predictor must beat that recall for free to justify itself.")


if __name__ == "__main__":
    main()
