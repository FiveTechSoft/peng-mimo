#!/usr/bin/env python3
"""
PoC script to build/query a semantic index for prompts and link them to cached KV entries.
Uses hnswlib if installed; otherwise prints guidance.

Usage:
  python3 tools/kv_index_build.py --embeddings embeddings.npy --output index.bin

This is a minimal PoC: it expects precomputed embeddings (numpy .npy) and
will build an HNSW index and serialize it. A companion query mode can load
index.bin and print nearest neighbours for given embeddings.
"""

import argparse
import sys

try:
    import numpy as np
    import hnswlib
except Exception as e:
    print("Required packages missing: numpy and hnswlib. Install with: pip install numpy hnswlib", file=sys.stderr)
    sys.exit(2)


def build(args):
    embs = np.load(args.embeddings)
    dim = embs.shape[1]
    num = embs.shape[0]
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num, ef_construction=200, M=32)
    p.add_items(embs, np.arange(num))
    p.save_index(args.output)
    print(f"Saved index {args.output} (n={num}, dim={dim})")


def query(args):
    q = np.load(args.query)
    p = hnswlib.Index(space='cosine', dim=q.shape[1])
    p.load_index(args.index)
    labels, dist = p.knn_query(q, k=args.k)
    for i in range(len(labels)):
        print(i, labels[i].tolist(), dist[i].tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    b = sub.add_parser('build')
    b.add_argument('--embeddings', required=True)
    b.add_argument('--output', required=True)
    q = sub.add_parser('query')
    q.add_argument('--index', required=True)
    q.add_argument('--query', required=True)
    q.add_argument('--k', type=int, default=4)
    args = parser.parse_args()
    if args.cmd == 'build':
        build(args)
    elif args.cmd == 'query':
        query(args)
    else:
        parser.print_help()
