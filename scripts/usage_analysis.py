#!/usr/bin/env python3
"""Expert-usage distribution from .coli_usage — prune-viability analysis."""
import sys
from collections import defaultdict

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "/root/mimo25_i4/.coli_usage"
N_EXPERTS = 256
EXPERT_MB = 12.6

use = defaultdict(lambda: np.zeros(N_EXPERTS, dtype=np.int64))
with open(PATH) as f:
    for line in f:
        parts = line.split()
        if len(parts) != 3:
            continue
        l, e, c = int(parts[0]), int(parts[1]), int(parts[2])
        use[l][e] += c

layers = sorted(use)
total = sum(int(use[l].sum()) for l in layers)
print(f"{PATH}: {len(layers)} MoE layers, {total:,} total expert selections\n")

# Global coverage curve: per layer, experts sorted by usage desc
keep_opts = [192, 128, 96, 64, 32]
cov = {k: 0 for k in keep_opts}
zero_total = 0
gini_list = []
for l in layers:
    u = np.sort(use[l])[::-1]
    s = u.sum()
    zero_total += int((u == 0).sum())
    for k in keep_opts:
        cov[k] += int(u[:k].sum())
    # Gini coefficient of the usage distribution
    x = np.sort(u).astype(np.float64)
    n = len(x)
    gini = (2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()) if x.sum() else 0
    gini_list.append(gini)

n_layers = len(layers)
print(f"experts with ZERO recorded use: {zero_total} / {n_layers*N_EXPERTS} "
      f"({100*zero_total/(n_layers*N_EXPERTS):.1f}%) "
      f"-> {zero_total*EXPERT_MB/1024:.1f} GB prunable at zero historical cost")
print(f"mean Gini of per-layer usage: {np.mean(gini_list):.3f} "
      f"(0=uniform, 1=one expert does everything)\n")

print(f"{'keep/layer':>10} | {'disk GB':>8} | {'coverage of historical calls':>28}")
for k in keep_opts:
    gb = n_layers * k * EXPERT_MB / 1024
    print(f"{k:>10} | {gb:>8.1f} | {100*cov[k]/total:>27.2f}%")

# Per-layer skew snapshot: top-16 share for a few layers
print("\nper-layer top-16 share (sample):")
for l in layers[:: max(1, n_layers // 8)]:
    u = np.sort(use[l])[::-1]
    s = u.sum()
    print(f"  layer {l:2d}: top-16 = {100*u[:16].sum()/s:5.1f}%  "
          f"top-64 = {100*u[:64].sum()/s:5.1f}%  zero-use = {(u==0).sum():3d}")
