#!/usr/bin/env python3
"""verify_mimo_qkv.py — regression guard for the MiMo-V2.5 int4 container.

MiMo-V2.5's fused qkv concatenates NR TP-rank blocks; its fp8 `weight_scale_inv`
is emitted PER RANK (ceil(rows_per_rank/128) scale rows each). A converter built
without the per-rank scale-grid fix dequantizes ranks 1..NR-1 with misaligned
scales; that error survives the int8 requantization and corrupts every
full/SWA layer's qkv while leaving rank 0 intact. The engine is then fed
plausible-looking but wrong weights and produces garbage tokens.

This tool catches exactly that, without needing the full model to run:
  * re-dequantize the fp8 qkv_proj with the (correct) per-rank scale grid,
  * dequantize the int4 container's qkv_proj (int8 * per-row scale),
  * compare the two, per rank-block. A correct container matches on ALL ranks
    (rank 0 and ranks 1..NR-1). A corrupt one diverges on ranks 1..NR-1.

Usage:
  python3 verify_mimo_qkv.py --int4 <container_dir> --fp8 <fp8_src_dir> \
      [--threshold 0.05] [--layers 5,11,17] [--nr 4]

Exit 0 if every checked layer's qkv matches within --threshold on all ranks;
exit 1 if any rank (other than 0, which is always correct) diverges.
"""
import sys, os, re, glob, argparse
import numpy as np
from safetensors import safe_open


def find_int4(d, name):
    for sh in sorted(glob.glob(d + "/out-*.safetensors")):
        with safe_open(sh, framework="np") as f:
            if name in f.keys():
                return f.get_tensor(name)
    return None


def find_fp8(d, name, fetch):
    import torch
    for sh in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(sh, framework="pt") as f:
            if name in f.keys():
                w = f.get_tensor(name).to(torch.float32).numpy()
                sname = name + "_scale_inv"
                try:
                    sc = f.get_tensor(sname).to(torch.float32).numpy()
                except Exception:
                    sc = np.asarray(fetch(sname), dtype=np.float32)
                return w, sc
    return None, None


def dequant_fp8(w, sc, O):
    """Replicate convert_fp8_to_int4.dequant scale-grid logic (per-rank aware)."""
    I = w.shape[1]
    if sc.shape[0] != (O + 127) // 128:
        nr = None
        for cand in range(2, 65):
            if O % cand == 0 and sc.shape[0] % cand == 0 \
               and sc.shape[0] // cand == ((O // cand) + 127) // 128:
                nr = cand
                break
        if nr is None:
            raise ValueError(f"unknown scale grid {tuple(sc.shape)} for {O}x{I}")
        rows_per, srows = O // nr, sc.shape[0] // nr
        blocks = [np.repeat(sc[g * srows:(g + 1) * srows], 128, axis=0)[:rows_per]
                  for g in range(nr)]
        sc = np.concatenate(blocks, axis=0)
        sc = np.repeat(sc, 128, axis=1)[:, :I]
    else:
        sc = np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)[:O, :I]
    return w * sc


def dequant_int8(w, s):
    O = s.shape[0]
    I = w.size // O
    return w.reshape(O, I).astype(np.int8).astype(np.float32) * s.astype(np.float32)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int4", required=True, help="converted int4 container dir")
    ap.add_argument("--fp8", required=True, help="fp8 source dir (model_pp0_epN_shardM.safetensors)")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--layers", default=None, help="comma list; default: all layers in --int4")
    ap.add_argument("--nr", type=int, default=4, help="TP-rank blocks in fused qkv (min kv_full,kv_swa)")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import convert_fp8_to_int4 as C
    fp8_shards = sorted(glob.glob(a.fp8 + "/*.safetensors"))
    fetch = C.make_scale_fetcher(local_shards=fp8_shards)

    if a.layers:
        layers = [int(x) for x in a.layers.split(",")]
    else:
        layers = set()
        for sh in sorted(glob.glob(a.int4 + "/out-*.safetensors")):
            with safe_open(sh, framework="np") as f:
                for k in f.keys():
                    if k.endswith("self_attn.qkv_proj.weight"):
                        m = re.search(r"layers\.(\d+)\.", k)
                        if m:
                            layers.add(int(m.group(1)))
        layers = sorted(layers)

    NR = a.nr
    failed = 0
    for L in layers:
        nm = f"model.layers.{L}.self_attn.qkv_proj.weight"
        w_c = find_int4(a.int4, nm)
        s_c = find_int4(a.int4, nm + ".qs")
        w_f, sc_f = find_fp8(a.fp8, nm, fetch)
        if w_c is None or w_f is None:
            print(f"L{L:2d}: MISSING (int4={w_c is not None} fp8={w_f is not None})")
            failed += 1
            continue
        O = s_c.shape[0]
        Wc = dequant_int8(w_c, s_c)
        Wf = dequant_fp8(w_f, sc_f, O)
        d = np.abs(Wc - Wf)
        rows_per = O // NR
        rank_max = [float(d[r * rows_per:(r + 1) * rows_per].max()) for r in range(NR)]
        bad = [r for r in range(NR) if rank_max[r] > a.threshold]
        status = "OK" if not bad else f"FAIL ranks {bad}"
        if bad:
            failed += 1
        print(f"L{L:2d} qkv O={O:6d} {status} rank_max={[round(x, 4) for x in rank_max]}")
    print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {len(layers)} layers checked, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
