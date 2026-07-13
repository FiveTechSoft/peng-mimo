#!/usr/bin/env python3
"""Union-of-profiles expert keep-mask (§42 multi-profile prune).

Reads N usage histograms ("layer eid count" lines), keeps per layer the union
of each profile's top-K experts, writes "layer eid" keep lines for EKEEP_MASK.
Frequency pruning on ONE workload makes its bias permanent (§42.3: code +67%
at keep-192); the union rule keeps any expert that ANY domain considers hot.
"""
import argparse
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("usage_files", nargs="+")
ap.add_argument("--topk", type=int, default=128, help="top-K per profile per layer")
ap.add_argument("--out", required=True)
ap.add_argument("--n-experts", type=int, default=256)
args = ap.parse_args()

profiles = []
for path in args.usage_files:
    u = defaultdict(lambda: np.zeros(args.n_experts, dtype=np.int64))
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 3:
                u[int(p[0])][int(p[1])] += int(p[2])
    profiles.append((path, u))
    print(f"{path}: {len(u)} layers, {sum(int(x.sum()) for x in u.values()):,} selections")

layers = sorted({l for _, u in profiles for l in u})
kept_total = 0
with open(args.out, "w") as f:
    for L in layers:
        keep = set()
        for _, u in profiles:
            if L in u:
                top = np.argsort(u[L])[::-1][:args.topk]
                keep.update(int(e) for e in top if u[L][e] > 0)
        for e in sorted(keep):
            f.write(f"{L} {e}\n")
        kept_total += len(keep)
print(f"union top-{args.topk} x {len(profiles)} profiles: "
      f"avg {kept_total/len(layers):.1f} experts/layer kept "
      f"({kept_total*12.6/1024:.1f} GB) -> {args.out}")
