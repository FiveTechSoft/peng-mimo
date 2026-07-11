#!/usr/bin/env bash
# One-shot chat-style generation + speed profile on the real MiMo container.
set -euo pipefail
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SNAP="${SNAP:-/root/mimo25_i4}"
OUT="${OUT:-/root/peng_bench.out}"
ERR="${ERR:-/root/peng_bench.err}"
NGEN="${NGEN:-24}"
PROMPT="${PROMPT:-Write one short sentence about Rome.}"

export SNAP
export COLI_CUDA="${COLI_CUDA:-1}"
export CUDA_DENSE="${CUDA_DENSE:-1}"
# CUDA_EXPERT_GB unset => auto in binary
export DIRECT="${DIRECT:-1}"
export TOPP="${TOPP:-0.6}"
export THINK="${THINK:-0}"
export NGEN
export TEMP="${TEMP:-0.7}"
export PROMPT

echo "=== peng chat bench ===" | tee "$ERR"
echo "SNAP=$SNAP NGEN=$NGEN COLI_CUDA=$COLI_CUDA CUDA_DENSE=$CUDA_DENSE TOPP=$TOPP DIRECT=$DIRECT" | tee -a "$ERR"
echo "PROMPT=$PROMPT" | tee -a "$ERR"
echo "START $(date -Is)" | tee -a "$ERR"
free -h | tee -a "$ERR"
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv | tee -a "$ERR"

./mimo 64 4 8 >"$OUT" 2>>"$ERR"
ec=$?
echo "EXIT=$ec END $(date -Is)" | tee -a "$ERR"
echo "===== STDOUT ====="
cat "$OUT"
echo "===== STDERR tail ====="
tail -n 100 "$ERR"
exit "$ec"
