#!/bin/bash
# T14 gate (b): identity DRAFT=0 vs 2 vs 4 (greedy byte-identical) on the real model.
cd /mnt/c/colibri/c || exit 1
for d in 0 2 4; do
  PROMPT="The capital of France is" NGEN=12 TEMP=0 DRAFT=$d SNAP=~/mimo25_i4 \
    ./mimo 64 4 8 > /tmp/r$d.out 2>/tmp/r$d.err
done
for d in 0 2 4; do
  echo "== DRAFT=$d"
  sed -n 4p /tmp/r$d.out
  grep -E "token in|speculazione|MTP acceptance" /tmp/r$d.out
done
