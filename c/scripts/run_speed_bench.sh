#!/usr/bin/env bash
# Full-model speed bench via chat_peng.py --bench --fast
set -euo pipefail
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ -f /root/mimo25_i4/config.json ]; then
  export SNAP=/root/mimo25_i4
elif [ -f "$HOME/mimo25_i4/config.json" ]; then
  export SNAP="$HOME/mimo25_i4"
else
  echo "ERROR: no model at /root/mimo25_i4 or ~/mimo25_i4" >&2
  exit 2
fi

export COLI_CUDA=1 CUDA_DENSE=1 DIRECT=1
ERR="${ERR:-/tmp/peng_bench_err.txt}"
OUT="${OUT:-/tmp/peng_bench_out.txt}"

echo "=== peng speed bench ==="
echo "ROOT=$ROOT SNAP=$SNAP"
date -Is
free -h | head -2
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv 2>/dev/null || true
ls -la c/mimo
echo "START $(date -Is)" | tee "$ERR"

set +e
python3 c/chat_peng.py --bench --fast --runs 3 --warmup 1 --ngen 24 --target 1.0 \
  >"$OUT" 2>>"$ERR"
ec=$?
set -e

echo "EXIT=$ec END $(date -Is)" | tee -a "$ERR"
echo "===== STDOUT ====="
cat "$OUT"
echo "===== STDERR (tail 120) ====="
tail -n 120 "$ERR"
exit "$ec"
