#!/usr/bin/env python3
"""PCA of dumped residual streams (RESID_DUMP).

With n samples in D=4096, vectors span at most n-1 dims. Report how concentrated
that span is vs an isotropic null (k_95 ≈ 0.95*(n-1) if energy is flat).
Hypothesis: late layers are a thin sheet if k95 << 0.95*(n-1).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np


def pca_spectrum(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    # n x D, n << D → SVD of X (economy)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = (s * s) / max(X.shape[0] - 1, 1)
    tot = ev.sum()
    if tot <= 0:
        return np.zeros(0)
    return ev / tot


def k_for(cum: np.ndarray, thr: float) -> int:
    idx = np.searchsorted(cum, thr, side="left")
    return int(min(idx + 1, len(cum)))


def main() -> int:
    ddir = sys.argv[1] if len(sys.argv) > 1 else "resid_out"
    meta = os.path.join(ddir, "meta.txt")
    D = 4096
    if os.path.isfile(meta):
        for line in open(meta, encoding="utf-8"):
            if line.startswith("D="):
                D = int(line.split("=", 1)[1])
    print(f"# RESID_PCA dir={ddir} D={D}")
    print("# layer  n   k50  k80  k90  k95  k99  PR   null95  sheet?")
    paths = sorted(glob.glob(os.path.join(ddir, "L*.bin")))
    if not paths:
        print("no L*.bin", file=sys.stderr)
        return 1
    for path in paths:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % D:
            print(f"# skip {path} size={raw.size} not multiple of D={D}", file=sys.stderr)
            continue
        n = raw.size // D
        X = raw.reshape(n, D)
        ev = pca_spectrum(X)
        if ev.size == 0:
            continue
        cum = np.cumsum(ev)
        pr = float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-30))
        nspan = ev.size  # rank <= n-1
        null95 = 0.95 * nspan
        k95 = k_for(cum, 0.95)
        sheet = "YES" if k95 < 0.55 * nspan else ("maybe" if k95 < 0.75 * nspan else "no")
        layer = os.path.splitext(os.path.basename(path))[0]
        print(
            f"{layer:6} {n:4d}  {k_for(cum,0.50):3d}  {k_for(cum,0.80):3d}  "
            f"{k_for(cum,0.90):3d}  {k95:3d}  {k_for(cum,0.99):3d}  "
            f"{pr:5.1f}  {null95:6.1f}  {sheet}"
        )
    print("# PR = participation ratio (1 = one axis, n = isotropic).")
    print("# sheet? YES if k95 < 55% of rank (mass in a thin subspace of the n-span).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
