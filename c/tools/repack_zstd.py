#!/usr/bin/env python3
"""Repack a peng safetensors container with per-tensor zstd-1 frames.

Output keeps names/dtypes/shapes and tensor order; data_offsets point to the
compressed frame; each tensor gains "nb" = uncompressed nbytes; __metadata__
gains peng_zstd="1". Engine support: c/st.h (findings §41, spec 2026-07-13).
"""
import argparse, json, os, shutil, struct, sys

import zstandard

LEVEL = 1  # measured at the order-0 entropy floor; higher levels buy nothing (§41)
AUX = ("config.json", "generation_config.json")


def read_header(f):
    hlen = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(hlen)), 8 + hlen


def tensor_items(hdr):
    """Tensor entries in file order (by data_offsets start)."""
    items = [(k, v) for k, v in hdr.items() if k != "__metadata__"]
    items.sort(key=lambda kv: kv[1]["data_offsets"][0])
    return items


def repack_shard(src, dst):
    cctx = zstandard.ZstdCompressor(level=LEVEL)
    with open(src, "rb") as f:
        hdr, data_start = read_header(f)
        new_hdr, frames, off = {}, [], 0
        for name, t in tensor_items(hdr):
            a, b = t["data_offsets"]
            f.seek(data_start + a)
            frame = cctx.compress(f.read(b - a))
            new_hdr[name] = {"dtype": t["dtype"], "shape": t["shape"],
                             "data_offsets": [off, off + len(frame)],
                             "nb": b - a}
            frames.append(frame)
            off += len(frame)
    meta = dict(hdr.get("__metadata__", {}))
    meta["peng_zstd"] = "1"
    new_hdr["__metadata__"] = meta
    hj = json.dumps(new_hdr).encode()
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for frame in frames:
            f.write(frame)
        f.flush(); os.fsync(f.fileno())
    os.rename(tmp, dst)
    return off


def verify_shard(src, dst):
    dctx = zstandard.ZstdDecompressor()
    with open(src, "rb") as fs, open(dst, "rb") as fd:
        sh, ss = read_header(fs)
        dh, ds = read_header(fd)
        for name, t in tensor_items(sh):
            a, b = t["data_offsets"]
            fs.seek(ss + a)
            want = fs.read(b - a)
            za, zb = dh[name]["data_offsets"]
            fd.seek(ds + za)
            try:
                got = dctx.decompress(fd.read(zb - za), max_output_size=dh[name]["nb"])
            except zstandard.ZstdError:
                got = None
            if got != want:
                print(f"VERIFY FAIL: {name} in {os.path.basename(src)}",
                      file=sys.stderr)
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    shards = sorted(x for x in os.listdir(args.indir)
                    if x.endswith(".safetensors"))
    if not shards:
        sys.exit(f"no .safetensors in {args.indir}")
    os.makedirs(args.out, exist_ok=True)
    if not args.verify_only:
        tin = tout = 0
        for i, sh in enumerate(shards):
            src = os.path.join(args.indir, sh)
            dst = os.path.join(args.out, sh)
            out = repack_shard(src, dst)
            tin += os.path.getsize(src); tout += os.path.getsize(dst)
            print(f"[{i+1}/{len(shards)}] {sh}: "
                  f"{out/2**30:.2f} GiB ({100*os.path.getsize(dst)/os.path.getsize(src):.1f}%)",
                  flush=True)
        for aux in AUX:
            p = os.path.join(args.indir, aux)
            if os.path.exists(p):
                shutil.copy2(p, args.out)
        meta = os.path.join(args.indir, "_meta")
        if os.path.isdir(meta):
            shutil.copytree(meta, os.path.join(args.out, "_meta"),
                            dirs_exist_ok=True)
        print(f"total: {tin/2**30:.1f} -> {tout/2**30:.1f} GiB "
              f"({100*tout/tin:.1f}%)")
    if args.verify or args.verify_only:
        for sh in shards:
            if not verify_shard(os.path.join(args.indir, sh),
                                os.path.join(args.out, sh)):
                sys.exit(1)
        print("verify: all frames byte-exact")


if __name__ == "__main__":
    main()
