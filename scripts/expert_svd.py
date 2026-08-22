#!/usr/bin/env python3
"""Effective dimensionality of MiMo int4 experts (issue #4, phase 0).

§42.1 measured pairwise redundancy (no near-duplicates, but early layers share
a strong common component). This tool asks the follow-up question: how many
singular directions does a full 256-expert layer actually span, and what does
that buy in GB/token?

Per layer and projection (gate/up/down, factorized separately — never blindly
concatenated):

  pass 1  load the 256 dequantized experts of one projection (RAM ~ 256 * dim
          * 4 bytes, ~8.6 GB for gate/up/down at 4096x2048), center, and
          eigendecompose the 256x256 Gram matrix. The singular spectrum is
          exact, so the rank-vs-error curve needs no reconstruction.
  pass 2  (only with --functional) rebuild the shared basis V_r (dim x rmax)
          from disk and reconstruct experts to measure *functional* SwiGLU
          error on real residual-stream activations (resid_out/L*.bin),
          not just ||dW||.

Also runs k-means in coefficient space (exact: coefficients are a rotation of
the centered weights) for the prototype+delta variant, and prints the GB/token
impact table at 47 layers x top-8 experts.

Usage:
  python scripts/expert_svd.py --snap /root/mimo25_i4 --layers 1,24,46 \
      --ranks 8,16,32,48,64,96,128 --clusters 8,16,32,64 \
      --functional --act resid_out --out svd_out
  python scripts/expert_svd.py --selftest     # synthetic correlated experts
"""
import argparse
import json
import os
import struct

import numpy as np

PROJS = ("gate_proj", "up_proj", "down_proj")
TOP_K = 8          # experts read per token per layer
N_MOE_LAYERS = 47
NVME_GBS = 2.75    # measured cold-stream ceiling (README/findings)
EXPERT_MB_INT4 = 12.6


# ---------------------------------------------------------------- container io
def load_headers(snap):
    tensors = {}
    for fn in sorted(os.listdir(snap)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(snap, fn)
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
            for name, t in hdr.items():
                if name == "__metadata__":
                    continue
                tensors[name] = (path, 8 + hlen + t["data_offsets"][0],
                                 t["data_offsets"][1] - t["data_offsets"][0],
                                 t.get("shape"))
    return tensors


def read_raw(entry):
    path, off, nb, _ = entry
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(nb)


def dequant_expert(tensors, layer, eid, proj):
    """int4 packed: O rows x I cols, 2 nibbles/byte, w = (nib-8) * row_scale."""
    base = f"model.layers.{layer}.mlp.experts.{eid}.{proj}.weight"
    codes = np.frombuffer(read_raw(tensors[base]), dtype=np.uint8)
    scales = np.frombuffer(read_raw(tensors[base + ".qs"]), dtype=np.float32)
    O = scales.size
    per_row = codes.size // O
    codes = codes.reshape(O, per_row)
    lo = (codes & 0xF).astype(np.int8) - 8
    hi = (codes >> 4).astype(np.int8) - 8
    w = np.empty((O, per_row * 2), dtype=np.float32)
    w[:, 0::2] = lo
    w[:, 1::2] = hi
    w *= scales[:, None]
    return w.ravel()


# ------------------------------------------------------------------ svd passes
def gram_svd(X):
    """Exact singular spectrum of the centered n x d matrix X (n << d) via its
    n x n Gram matrix. Returns mean, singular values (desc), U (n x n)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    G = Xc @ Xc.T
    lam, U = np.linalg.eigh(G)
    order = np.argsort(lam)[::-1]
    lam = np.maximum(lam[order], 0.0)
    return mean, np.sqrt(lam), U[:, order]


def rel_err_curve(lam, ranks):
    """Exact relative Frobenius error of the best rank-r approximation."""
    tot = lam.sum()
    cum = np.cumsum(lam)
    return {r: float(np.sqrt(max(1.0 - cum[min(r, len(lam)) - 1] / tot, 0.0)))
            if tot > 0 else 0.0 for r in ranks}


def energy_k(lam, thr):
    tot = lam.sum()
    if tot <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(lam) / tot, thr) + 1)


def build_basis(tensors, layer, n, proj, mean, U, s, rmax, d):
    """V_r = X_c^T U_r S_r^-1  (d x rmax), accumulated one expert at a time."""
    V = np.zeros((d, rmax), dtype=np.float32)
    scale = U[:, :rmax] / np.maximum(s[:rmax], 1e-12)
    for e in range(n):
        v = dequant_expert(tensors, layer, e, proj)
        V += np.outer(v - mean, scale[e]).astype(np.float32)
    return V


# ------------------------------------------------------------------ functional
def swiglu(x, g, u, d, inter, hidden):
    """gate/up are (inter, hidden); down is (hidden, inter)."""
    a = x @ g.reshape(inter, hidden).T
    b = x @ u.reshape(inter, hidden).T
    h = (a / (1.0 + np.exp(-np.clip(a, -30, 30)))) * b   # SiLU(gate) * up
    return h @ d.reshape(hidden, inter).T


# ------------------------------------------------------------------- reporting
def gb_per_token(bytes_per_expert):
    return N_MOE_LAYERS * TOP_K * bytes_per_expert / 1e9


def kmeans(X, k, seed=42, iters=50):
    """Small dependency-free Lloyd's k-means with k-means++ seeding.
    Returns (labels, centers)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    centers[0] = X[rng.integers(n)]
    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        probs = d2 / max(d2.sum(), 1e-30)
        centers[c] = X[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, np.sum((X - centers[c]) ** 2, axis=1))
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(axis=0)
    return labels, centers


def print_impact(rows):
    print(f"\n=== GB/token impact ({N_MOE_LAYERS} layers x top-{TOP_K}, "
          f"NVMe {NVME_GBS} GB/s) ===")
    print(f"{'config':<28} {'MB/expert':>9} {'GB/token':>8} {'ceiling tok/s':>13}")
    base = gb_per_token(EXPERT_MB_INT4 * 1e6)
    print(f"{'int4 baseline':<28} {EXPERT_MB_INT4:9.2f} {base:8.2f} "
          f"{NVME_GBS / base:13.2f}")
    for name, bpe in rows:
        g = gb_per_token(bpe)
        print(f"{name:<28} {bpe / 1e6:9.2f} {g:8.2f} {NVME_GBS / g:13.2f}")


# --------------------------------------------------------------------- driver
def analyze_layer(tensors, layer, ranks, clusters, functional, act_dir):
    n = 256
    res = {"layer": layer, "projections": {}}
    bases = {}   # proj -> (mean, V, s, U), only when functional

    for proj in PROJS:
        print(f"  layer {layer} {proj}: pass 1 (Gram/SVD)", flush=True)
        X = np.stack([dequant_expert(tensors, layer, e, proj)
                      for e in range(n)])
        d = X.shape[1]
        mean, s, U = gram_svd(X)
        del X
        lam = s * s
        errs = rel_err_curve(lam, ranks)
        k90, k95, k99 = (energy_k(lam, t) for t in (0.90, 0.95, 0.99))
        pres = {"dim": d, "k90": k90, "k95": k95, "k99": k99,
                "rel_frob_err": {str(r): errs[r] for r in ranks},
                "compression": {}}
        for r in ranks:
            streamed = r * 4                      # coeffs per expert (this proj)
            resident = (d + 1) * r * 4 + d * 4    # basis + mean, stays in RAM
            effective = streamed + resident / n   # amortized per expert
            pres["compression"][str(r)] = {
                "streamed_bytes_per_expert": streamed,
                "resident_bytes_basis": resident,
                "effective_bytes_per_expert": effective,
                "ratio_vs_int4_proj": (d // 2) / effective,
            }
        res["projections"][proj] = pres
        print(f"    dim={d} k90={k90} k95={k95} k99={k99}")
        for r in ranks:
            print(f"    rank {r:>3}: rel_frob={errs[r]:.4f}")

        if clusters:
            C = U * s  # exact rotation of the centered weights
            pres["clusters"] = {}
            for k in clusters:
                labels, ctrs = kmeans(C, k)
                err = float(np.linalg.norm(C - ctrs[labels])
                            / (np.linalg.norm(C) + 1e-12))
                amort = k * d * 4 / n
                pres["clusters"][str(k)] = {
                    "rel_frob_err": err,
                    "effective_bytes_per_expert": amort,
                }
                print(f"    k={k:>3}: rel_frob={err:.4f} "
                      f"(prototype-only, {amort / 1e6:.2f} MB/expert amort.)")

        if functional:
            rmax = max(ranks)
            print(f"  layer {layer} {proj}: pass 2 (basis, rmax={rmax})",
                  flush=True)
            bases[proj] = (mean,
                           build_basis(tensors, layer, n, proj,
                                       mean, U, s, rmax, d),
                           s, U)

    if functional:
        base = f"model.layers.{layer}.mlp.experts.0.gate_proj.weight"
        inter = np.frombuffer(read_raw(tensors[base + ".qs"]),
                              dtype=np.float32).size
        hidden = dequant_expert(tensors, layer, 0, "gate_proj").size // inter
        x = None
        if act_dir:
            p = os.path.join(act_dir, f"L{layer:02d}.bin")
            if os.path.isfile(p):
                x = np.fromfile(p, dtype=np.float32).reshape(-1, hidden)
                print(f"  activations: {p} ({x.shape[0]} rows) "
                      f"[residual-stream proxy for MoE input]")
            else:
                print(f"  WARNING: {p} not found, synthetic activations used")
        if x is None:
            rng = np.random.default_rng(42)
            x = rng.standard_normal((96, hidden)).astype(np.float32)
        eids = list(range(0, 256, 16))   # 16 evenly spaced experts
        fres = {}
        for r in ranks:
            errs = []
            for e in eids:
                gt = dequant_expert(tensors, layer, e, "gate_proj")
                ut = dequant_expert(tensors, layer, e, "up_proj")
                dt = dequant_expert(tensors, layer, e, "down_proj")
                ws = {}
                for proj in PROJS:
                    mean, V, s, U = bases[proj]
                    ws[proj] = mean + V[:, :r] @ (U[e, :r] * s[:r])
                y0 = swiglu(x, gt, ut, dt, inter, hidden)
                y1 = swiglu(x, ws["gate_proj"], ws["up_proj"], ws["down_proj"],
                            inter, hidden)
                errs.append(float(np.linalg.norm(y0 - y1) /
                                  (np.linalg.norm(y0) + 1e-12)))
            fres[str(r)] = {"mean": float(np.mean(errs)),
                            "p95": float(np.percentile(errs, 95))}
            print(f"  functional rank {r:>3}: mean rel err "
                  f"{np.mean(errs):.4f} p95 {np.percentile(errs, 95):.4f}")
        res["functional_error"] = fres

    rows = []
    for r in ranks:
        streamed = 3 * r * 4
        resident = sum((res["projections"][p]["dim"] + 1) * r * 4 +
                       res["projections"][p]["dim"] * 4 for p in PROJS)
        rows.append((f"svd rank-{r} (3 proj)", streamed + resident / 256))
    print_impact(rows)
    return res


def selftest():
    """Synthetic correlated experts: shared rank-12 subspace + 12% noise."""
    rng = np.random.default_rng(42)
    n, d, true_rank, noise = 64, 4096, 12, 0.12
    B = rng.standard_normal((true_rank, d)).astype(np.float32) * 0.1
    X = np.stack([np.tensordot(rng.standard_normal(true_rank), B, axes=1) +
                  noise * rng.standard_normal(d) for _ in range(n)])
    mean, s, U = gram_svd(X)
    errs = rel_err_curve(s * s, [4, 8, 12, 16, 24])
    print("=== selftest: synthetic rank-12 + 12% noise ===")
    for r, e in errs.items():
        print(f"  rank {r:>2}: rel_frob={e:.4f}")
    assert errs[16] < 0.35, "rank-16 should recover most of a rank-12 subspace"
    assert errs[4] > errs[16], "error must decrease with rank"
    # clustering error in coefficient space (rotation-invariant):
    C = U * s
    labels, ctrs = kmeans(C, 8)
    cerr = float(np.linalg.norm(C - ctrs[labels]) / np.linalg.norm(C))
    print(f"  k-means k=8: rel_frob={cerr:.4f}")
    # basis reconstruction round-trip (aggregate, matches the curve definition)
    Xc = X - mean
    rmax = 24
    V = Xc.T @ (U[:, :rmax] / np.maximum(s[:rmax], 1e-12))
    e = 0
    r = 16
    rec = mean + V[:, :r] @ (U[e, :r] * s[:r])
    rerr = float(np.linalg.norm(X[e] - rec) / np.linalg.norm(X[e]))
    print(f"  reconstruct expert 0 @ rank 16: rel_err={rerr:.4f}")
    rec_all = (V[:, :r] @ (U[:, :r] * s[:r]).T).T
    aerr = float(np.linalg.norm(Xc - rec_all) / np.linalg.norm(Xc))
    print(f"  aggregate @ rank 16: rel_err={aerr:.4f} (curve={errs[16]:.4f})")
    assert abs(aerr - errs[16]) < 0.02, "reconstruction must match the curve"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default="/root/mimo25_i4")
    ap.add_argument("--layers", default="1,24,46")
    ap.add_argument("--ranks", default="8,16,32,48,64,96,128")
    ap.add_argument("--clusters", default="8,16,32,64")
    ap.add_argument("--functional", action="store_true",
                    help="pass 2: build basis, measure SwiGLU error "
                         "(adds ~ dim * rmax * 4 bytes RAM per projection)")
    ap.add_argument("--act", default="resid_out",
                    help="dir with residual-stream dumps L*.bin")
    ap.add_argument("--out", default="svd_out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    ranks = [int(x) for x in args.ranks.split(",")]
    clusters = [int(x) for x in args.clusters.split(",") if x]
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out, exist_ok=True)
    tensors = load_headers(args.snap)
    print(f"container: {len(tensors)} tensors", flush=True)

    for layer in layers:
        print(f"\n=== layer {layer} ===", flush=True)
        res = analyze_layer(tensors, layer, ranks, clusters,
                            args.functional, args.act)
        path = os.path.join(args.out, f"layer{layer:02d}_svd.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
