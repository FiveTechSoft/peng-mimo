#!/usr/bin/env bash
# Measure |SiLU(gate·x)| on routed MiMo experts. Falsifies the neuron-I/O idea
# if dead@0.01 is not ≳ 50%. Requires the int4 SNAP (not WSL-home: put it on NTFS).
set -euo pipefail
SNAP="${SNAP:-/mnt/c/models/mimo25_i4}"
BIN="${BIN:-./c/mimo}"
NGEN="${NGEN:-24}"
if [[ ! -d "$SNAP" ]]; then
  echo "SNAP missing: $SNAP" >&2
  exit 1
fi
export SNAP SILU_HIST=1 COLI_CUDA=0 PILOT=0 DRAFT=0 NGEN
export PROMPT="${PROMPT:-Write one short sentence about Rome. Then another about Berlin.}"
echo "SILU_HIST SNAP=$SNAP NGEN=$NGEN" >&2
exec "$BIN" 64 4 8
