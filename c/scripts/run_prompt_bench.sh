#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
cd "$(dirname "$0")/.."
export SNAP="${SNAP:-/root/mimo25_i4}"
export COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1 TOPP="${TOPP:-0.55}" THINK=0 TEMP=0.7
export NGEN="${NGEN:-24}" PROFILE=1
export PROMPT="${PROMPT:-Write one short sentence about Rome.}"
echo "SNAP=$SNAP TOPP=$TOPP NGEN=$NGEN"
./mimo 64 4 8 2>/tmp/peng_prompt.err | tee /tmp/peng_prompt.out
echo "===== PROFILE / CUDA ====="
grep -E 'tok/s|PROFILE|hit|CUDA expert|VRAM|cap |PIN|tok' /tmp/peng_prompt.err | tail -40
tail -5 /tmp/peng_prompt.out
