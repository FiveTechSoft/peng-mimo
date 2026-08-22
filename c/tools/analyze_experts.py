#!/usr/bin/env python3
"""
analyze_experts.py — Offline analysis tool for issue #4
Shared basis / low-rank / clustering of MoE experts in peng-mimo

Usage examples:
  python3 tools/analyze_experts.py --snap ~/mimo25_i4 --layer 17
  python3 tools/analyze_experts.py --snap ~/mimo25_i4 --layer 5,20,42 --ranks 16,32,48,64 --max-experts 128
"""

import argparse
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open

def dequant_int4(qbytes: np.ndarray, scales: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Dequantize packed int4 (same logic as convert_fp8_to_int4.py)."""
    rb = (cols + 1) // 2
    q = qbytes.reshape(rows, rb)

    v0 = (q.astype(np.int32) & 0x0F) - 8
    v1 = ((q.astype(np.int32) >> 4) & 0x0F) - 8

    out = np.zeros((rows, cols), dtype=np.float32)
    out[:, 0::2] = v0[:, :out[:, 0::2].shape[1]]
    if cols > 1:
        out[:, 1::2] = v1[:, :out[:, 1::2].shape[1]]

    out *= scales[:, None]
    return out

def find_tensor(snap: Path, name: str):
    """Search for a tensor across all out-*.safetensors shards."""
    for f in sorted(snap.glob("out-*.safetensors")):
        with safe_open(str(f), framework="np") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    return None

def load_expert(snap: Path, layer: int, expert_id: int, hidden: int = 4096, inter: int = 2048):
    """
    Load and dequantize one expert (gate_proj, up_proj, down_proj).
    Expected names in the peng int4 container:
      model.layers.{L}.mlp.experts.{E}.gate_proj.weight
      model.layers.{L}.mlp.experts.{E}.up_proj.weight
      model.layers.{L}.mlp.experts.{E}.down_proj.weight
    + corresponding .qs scale tensors
    """
    base = f"model.layers.{layer}.mlp.experts.{expert_id}"

    def load_proj(proj: str):
        w_name = f"{base}.{proj}.weight"
        qs_name = f"{base}.{proj}.weight.qs"

        qbytes = find_tensor(snap, w_name)
        scales = find_tensor(snap, qs_name)

        if qbytes is None:
            # fallback
            qs_name = f"{w_name}.qs"
            qbytes = find_tensor(snap, w_name)
            scales = find_tensor(snap, qs_name)

        if qbytes is None or scales is None:
            raise RuntimeError(f"Tensor not found: {w_name} or its .qs")

        if proj in ("gate_proj", "up_proj"):
            rows, cols = inter, hidden
        else:
            rows, cols = hidden, inter

        return dequant_int4(qbytes, scales, rows, cols)

    gate = load_proj("gate_proj")
    up   = load_proj("up_proj")
    down = load_proj("down_proj")
    return gate, up, down

def flatten_expert(gate, up, down):
    return np.concatenate([gate.ravel(), up.ravel(), down.ravel()])

def analyze_layer(snap: Path, layer: int, ranks: list, out_dir: Path, max_experts: int = 256):
    print(f"\n=== Layer {layer} ===")
    experts = []

    for eid in range(max_experts):
        try:
            g, u, d = load_expert(snap, layer, eid)
            experts.append((g, u, d))
            if (eid + 1) % 32 == 0 or eid == max_experts - 1:
                print(f"  loaded {eid + 1}/{max_experts}")
        except Exception as e:
            print(f"  stopped at expert {eid}: {e}")
            break

    n = len(experts)
    if n == 0:
        print("No experts loaded")
        return

    X = np.stack([flatten_expert(*e) for e in experts]).astype(np.float32)
    print(f"Matrix shape: {X.shape}")

    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    results = {
        "layer": layer,
        "n_experts": n,
        "svd": []
    }

    original_bytes = n * X.shape[1] * 4  # theoretical f32 size

    print(f"{'rank':>6} | {'ratio':>8} | {'frobenius':>10}")
    print("-" * 36)

    for r in ranks:
        X_hat = X_mean + (U[:, :r] * S[:r]) @ Vt[:r]
        err = float(np.linalg.norm(X - X_hat) / (np.linalg.norm(X) + 1e-8))

        shared_bytes = (X.shape[1] + r * X.shape[1] + n * r) * 4
        ratio = original_bytes / max(shared_bytes, 1)

        print(f"{r:6d} | {ratio:8.2f}x | {err:10.4f}")
        results["svd"].append({
            "rank": r,
            "ratio": round(ratio, 3),
            "frobenius": round(err, 5)
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"layer{layer}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        out_dir / f"layer{layer}_svd.npz",
        U=U[:, :max(ranks)],
        S=S[:max(ranks)],
        Vt=Vt[:max(ranks)],
        mean=X_mean
    )

    print(f"Saved results to {out_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Offline expert analysis for peng-mimo issue #4")
    parser.add_argument("--snap", required=True, help="Path to int4 container directory")
    parser.add_argument("--layer", default="17", help="Layer(s) to analyze (e.g. 17 or 5,20,42)")
    parser.add_argument("--ranks", default="8,16,24,32,48,64,96", help="Ranks to evaluate")
    parser.add_argument("--out", default="expert_analysis", help="Output directory")
    parser.add_argument("--max-experts", type=int, default=256, help="Max experts to load per layer")
    args = parser.parse_args()

    snap = Path(args.snap)
    out_dir = Path(args.out)
    ranks = [int(x) for x in args.ranks.split(",")]
    layers = [int(x) for x in args.layer.split(",")]

    for layer in layers:
        analyze_layer(snap, layer, ranks, out_dir, args.max_experts)

    print("\nDone.")

if __name__ == "__main__":
    main()
