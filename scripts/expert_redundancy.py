#!/usr/bin/env python3
"""Measure redundancy between routed experts of a MiMo int4 layer.

Dequantizes gate_proj of every expert (int4 codes * per-row f32 scales),
subsamples a fixed coordinate set, computes the full pairwise cosine matrix.
Controls: same measurement across two layers, cross-layer pairs, and the
identical-int4-code fraction for the most similar pairs.
"""
import json, struct, sys
import numpy as np

SNAP = "/root/mimo25_i4"
LAYERS = [1, 24, 46]          # early / middle / late
PROJ = "gate_proj"
SUB = 131072                  # fixed subsampled coords for cosine (stable estimate)
RNG = np.random.default_rng(42)


def load_headers(snap):
    import os
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


def dequant_expert(tensors, layer, eid):
    """int4 packed: O rows x I cols, 2 nibbles/byte, w = (nib-8) * row_scale."""
    base = f"model.layers.{layer}.mlp.experts.{eid}.{PROJ}.weight"
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


def layer_matrix(tensors, layer, idx):
    vecs = []
    for eid in range(256):
        v = dequant_expert(tensors, layer, eid)[idx]
        n = np.linalg.norm(v)
        vecs.append(v / n if n > 0 else v)
        if eid % 64 == 63:
            print(f"  layer {layer}: {eid+1}/256 experts decoded", flush=True)
    return np.stack(vecs)


def main():
    tensors = load_headers(SNAP)
    probe = dequant_expert(tensors, LAYERS[0], 0)
    idx = np.sort(RNG.choice(probe.size, SUB, replace=False))
    print(f"expert {PROJ} numel={probe.size}, subsample={SUB}", flush=True)

    mats = {}
    for L in LAYERS:
        mats[L] = layer_matrix(tensors, L, idx)
        C = mats[L] @ mats[L].T
        off = C[~np.eye(256, dtype=bool)]
        top = np.argsort(off)[-5:]
        pairs = [(int(i), int(j), float(C[i, j]))
                 for i in range(256) for j in range(i + 1, 256)]
        pairs.sort(key=lambda x: -x[2])
        print(f"\n=== layer {L} ({PROJ}) pairwise cosine, 256 experts ===")
        print(f"  mean |cos| off-diag: {np.abs(off).mean():.4f}")
        print(f"  p50 / p95 / p99 / max: {np.percentile(off,50):.4f} / "
              f"{np.percentile(off,95):.4f} / {np.percentile(off,99):.4f} / "
              f"{off.max():.4f}")
        print(f"  pairs with cos > 0.5: {(off > 0.5).sum() // 2}")
        print(f"  pairs with cos > 0.9: {(off > 0.9).sum() // 2}")
        print("  top-5 pairs:", [(i, j, round(c, 4)) for i, j, c in pairs[:5]])

        i, j, c = pairs[0]
        a = np.frombuffer(read_raw(
            tensors[f"model.layers.{L}.mlp.experts.{i}.{PROJ}.weight"]), np.uint8)
        b = np.frombuffer(read_raw(
            tensors[f"model.layers.{L}.mlp.experts.{j}.{PROJ}.weight"]), np.uint8)
        same = (a == b).mean()
        print(f"  top pair ({i},{j}): identical int4 bytes = {same*100:.2f}% "
              f"(random-pair baseline ~{(1/256)*100:.1f}% bytes, "
              f"~{(1/16)*100:.1f}% per nibble)")

    a, b = LAYERS[0], LAYERS[1]
    X = mats[a][:64] @ mats[b][:64].T
    print(f"\n=== control: cross-layer cosine (layer {a} vs {b}, 64x64) ===")
    print(f"  mean |cos|: {np.abs(X).mean():.4f} | max: {X.max():.4f}")


if __name__ == "__main__":
    main()
