#!/usr/bin/env python3
"""
path_pack_analyze.py — seed of Corriente Peng "path packing"

Reads SNAP/.coli_traj[.profile] and SNAP/.coli_usage[.profile] and emits a
per-layer expert order that clusters co-activated experts (habit channels).

This does NOT rewrite the model. It only proposes a packing order so a future
converter can lay "snow" contiguously along river channels (range readahead).

Bit-exact invariant: reordering storage must preserve tensor values; only
physical locality changes.

Usage:
  python3 path_pack_analyze.py --snap ~/mimo25_i4
  python3 path_pack_analyze.py --snap /root/mimo25_i4 --profile chat --out order.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def usage_path(snap: str, profile: str | None) -> str:
    if profile:
        return os.path.join(snap, f".coli_usage.{profile}")
    return os.path.join(snap, ".coli_usage")


def traj_path(snap: str, profile: str | None) -> str:
    if profile:
        return os.path.join(snap, f".coli_traj.{profile}")
    return os.path.join(snap, ".coli_traj")


def load_usage(path: str) -> Dict[int, Dict[int, int]]:
    """layer -> {expert: count}"""
    out: Dict[int, Dict[int, int]] = defaultdict(dict)
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                L, e, c = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if L < 0 or e < 0 or c <= 0:
                continue
            out[L][e] = out[L].get(e, 0) + c
    return out


def load_traj(path: str) -> Tuple[Dict[int, Dict[int, Dict[int, int]]], Dict[int, Dict[int, Dict[int, int]]]]:
    """tok[L][e0][e1]=c , lay[L][e0][e1]=c"""
    tok: Dict[int, Dict[int, Dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    lay: Dict[int, Dict[int, Dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    if not os.path.isfile(path):
        return tok, lay
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            kind, Ls, e0s, e1s, cs = parts
            try:
                L, e0, e1, c = int(Ls), int(e0s), int(e1s), int(cs)
            except ValueError:
                continue
            if c <= 0 or L < 0 or e0 < 0 or e1 < 0:
                continue
            if kind.startswith("tok"):
                tok[L][e0][e1] = tok[L][e0].get(e1, 0) + c
            elif kind.startswith("lay"):
                lay[L][e0][e1] = lay[L][e0].get(e1, 0) + c
    return tok, lay


def affinity_matrix(
    L: int,
    heat: Dict[int, int],
    tok: Dict[int, Dict[int, int]],
    lay: Dict[int, Dict[int, int]],
) -> Dict[int, Dict[int, float]]:
    """Undirected affinity: heat + symmetric Markov mass."""
    aff: Dict[int, Dict[int, float]] = defaultdict(dict)
    nodes = set(heat.keys())
    for e0, suc in tok.items():
        nodes.add(e0)
        for e1, c in suc.items():
            nodes.add(e1)
            w = float(c)
            aff[e0][e1] = aff[e0].get(e1, 0.0) + w
            aff[e1][e0] = aff[e1].get(e0, 0.0) + w
    for e0, suc in lay.items():
        nodes.add(e0)
        for e1, c in suc.items():
            nodes.add(e1)
            # lay points L -> L+1; still boost co-travel within L for packing of sources
            w = 0.5 * float(c)
            aff[e0][e1] = aff[e0].get(e1, 0.0) + w
            aff[e1][e0] = aff[e1].get(e0, 0.0) + w
    for e, h in heat.items():
        # self mass so isolated hot experts still rank
        aff[e][e] = aff[e].get(e, 0.0) + 0.01 * float(h)
    return aff


def greedy_path_order(aff: Dict[int, Dict[int, float]], heat: Dict[int, int]) -> List[int]:
    """Start at hottest expert; always append strongest affinity neighbor not yet taken."""
    if not heat and not aff:
        return []
    remaining = set(aff.keys()) | set(heat.keys())
    if not remaining:
        return []
    start = max(remaining, key=lambda e: (heat.get(e, 0), e))
    order = [start]
    remaining.remove(start)
    cur = start
    while remaining:
        nbrs = aff.get(cur, {})
        nxt = None
        best = -1.0
        for e, w in nbrs.items():
            if e in remaining and w > best:
                best = w
                nxt = e
        if nxt is None:
            # jump to hottest remaining (new channel)
            nxt = max(remaining, key=lambda e: (heat.get(e, 0), e))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order


def score_locality(order: List[int], aff: Dict[int, Dict[int, float]]) -> float:
    """Sum of affinity between consecutive positions (higher = better packing)."""
    pos = {e: i for i, e in enumerate(order)}
    s = 0.0
    for e0, suc in aff.items():
        i0 = pos.get(e0)
        if i0 is None:
            continue
        for e1, w in suc.items():
            i1 = pos.get(e1)
            if i1 is None or e0 == e1:
                continue
            dist = abs(i0 - i1)
            s += w / (1.0 + dist)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Corriente Peng path-pack order from traj/usage")
    ap.add_argument("--snap", required=True, help="SNAP dir (mimo25_i4)")
    ap.add_argument("--profile", default=None, help="COLI_PROFILE name")
    ap.add_argument("--out", default=None, help="JSON output path (default stdout)")
    ap.add_argument("--top-layers", type=int, default=0, help="if >0 only first N sparse-ish layers by heat")
    args = ap.parse_args()

    snap = os.path.expanduser(args.snap)
    up = usage_path(snap, args.profile)
    tp = traj_path(snap, args.profile)
    usage = load_usage(up)
    tok_all, lay_all = load_traj(tp)

    layers = sorted(set(usage.keys()) | set(tok_all.keys()) | set(lay_all.keys()))
    if not layers:
        print(f"no usage/traj under {snap} (looked for {up}, {tp})", file=sys.stderr)
        return 1

    report = {
        "snap": snap,
        "profile": args.profile,
        "usage_file": up if os.path.isfile(up) else None,
        "traj_file": tp if os.path.isfile(tp) else None,
        "manifest": "docs/corriente-peng.md",
        "invariant": "packing order only; tensor values must stay bit-identical",
        "layers": {},
    }

    # optional: focus on layers with most heat
    if args.top_layers > 0:
        heat_sum = {L: sum(usage.get(L, {}).values()) for L in layers}
        layers = sorted(layers, key=lambda L: heat_sum.get(L, 0), reverse=True)[: args.top_layers]
        layers = sorted(layers)

    total_gain = 0.0
    for L in layers:
        heat = usage.get(L, {})
        tok = tok_all.get(L, {})
        lay = lay_all.get(L, {})
        aff = affinity_matrix(L, heat, tok, lay)
        if not aff and not heat:
            continue
        # baseline: sort by expert id
        nodes = sorted(set(heat.keys()) | set(aff.keys()))
        base_order = nodes
        pack_order = greedy_path_order(aff, heat)
        s_base = score_locality(base_order, aff)
        s_pack = score_locality(pack_order, aff)
        gain = (s_pack / s_base) if s_base > 0 else 1.0
        total_gain += gain
        report["layers"][str(L)] = {
            "n_experts_seen": len(nodes),
            "heat_total": int(sum(heat.values())),
            "traj_tok_edges": sum(len(v) for v in tok.values()),
            "traj_lay_edges": sum(len(v) for v in lay.values()),
            "locality_score_id_order": round(s_base, 3),
            "locality_score_path_order": round(s_pack, 3),
            "locality_ratio": round(gain, 4),
            "order": pack_order,
        }

    nL = max(len(report["layers"]), 1)
    report["mean_locality_ratio"] = round(total_gain / nL, 4)
    report["note"] = (
        "locality_ratio > 1 means path order places co-activated experts closer "
        "than raw id order — candidate win for sequential thaw (readahead)."
    )

    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
        print(
            f"wrote {args.out} · layers={len(report['layers'])} · "
            f"mean_locality_ratio={report['mean_locality_ratio']}",
            file=sys.stderr,
        )
    else:
        print(text)

    # Native engine format for mimo.c pathpack_load (Corriente FLOW)
    pp = os.path.join(snap, f".coli_pathpack.{args.profile}" if args.profile else ".coli_pathpack")
    with open(pp, "w", encoding="utf-8") as f:
        f.write("# coli_pathpack v1\n")
        for L in sorted(report["layers"].keys(), key=int):
            order = report["layers"][L]["order"]
            if not order:
                continue
            f.write(f"{L} {len(order)}")
            for e in order:
                f.write(f" {e}")
            f.write("\n")
    print(f"wrote {pp} (engine FLOW pathpack)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
