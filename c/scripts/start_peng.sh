#!/usr/bin/env bash
# start_peng.sh — speed-oriented env for MiMo-V2.5 + peng-mimo
#
# Detects RAM / SNAP location / CUDA, exports the measured stack, then either:
#   source scripts/start_peng.sh              # only export env
#   scripts/start_peng.sh chat [--fast] ...   # interactive chat_peng.py
#   scripts/start_peng.sh prompt "text" [N]   # one-shot PROMPT
#   scripts/start_peng.sh env                 # print export lines
#
# Override any knob: SNAP=... PIN_GB=4 PILOT=0 SPEED=1 scripts/start_peng.sh chat
# Tao (wu wei): TAO=1 scripts/start_peng.sh chat
set -euo pipefail

# TAO=1 → Corriente without force (explicit env still wins)
if [[ "${TAO:-0}" == "1" ]]; then
  # Sacred / Fibonacci harmonics (explicit env still wins)
  export SPEED="${SPEED:-1}"
  export TRAJ="${TRAJ:-1}"
  export TRAJ_K="${TRAJ_K:-5}"
  export FLOW="${FLOW:-1}"
  export FLOW_R="${FLOW_R:-3}"
  export ENERGY="${ENERGY:--1}"
  export DRAFT="${DRAFT:-0}"
  export REPIN="${REPIN:-55}"
  export PILOT="${PILOT:-1}"
  export PILOT_DEPTH="${PILOT_DEPTH:-1}"
  export PREFETCH="${PREFETCH:-1}"
  export DIRECT="${DIRECT:-1}"
fi

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$CODE/.." && pwd)"
cd "$CODE"

# ---------- detect host ----------
mem_avail_gb() {
  if [[ -r /proc/meminfo ]]; then
    awk '/MemAvailable:/ {printf "%.1f", $2/1024/1024; exit}' /proc/meminfo
  else
    echo "${RAM_GB:-16}"
  fi
}

MEM_GB="$(mem_avail_gb)"
# round down for integer knobs
MEM_I=${MEM_GB%.*}
[[ -z "$MEM_I" ]] && MEM_I=16

# Prefer native ext4 SNAP; refuse silent /mnt/c for production runs
pick_snap() {
  if [[ -n "${SNAP:-}" ]]; then
    echo "$SNAP"
    return
  fi
  for d in "$HOME/mimo25_i4" /root/mimo25_i4 "$ROOT/mimo25_i4"; do
    if [[ -f "$d/config.json" ]]; then
      echo "$d"
      return
    fi
  done
  echo "$HOME/mimo25_i4"
}

SNAP="$(pick_snap)"
case "$SNAP" in
  /mnt/*)
    echo "[start_peng] WARN: SNAP=$SNAP is under /mnt (Windows 9p) — expert I/O will be very slow." >&2
    echo "            Copy the container to ext4, e.g. /root/mimo25_i4 or ~/mimo25_i4" >&2
    ;;
esac

# CUDA binary?
CUDA_OK=0
if [[ -x "$CODE/mimo" ]] && strings "$CODE/mimo" 2>/dev/null | grep -q COLI_CUDA; then
  CUDA_OK=1
elif command -v nvcc >/dev/null 2>&1 || [[ -e /usr/local/cuda/lib64/libcudart.so ]]; then
  CUDA_OK=1
fi
if [[ "${COLI_CUDA:-}" == "0" ]]; then CUDA_OK=0; fi

# Usage / pin history (autopin inside mimo uses .coli_usage when PIN unset)
USAGE="$SNAP/.coli_usage"
if [[ -n "${COLI_PROFILE:-}" ]]; then
  USAGE="$SNAP/.coli_usage.${COLI_PROFILE}"
fi

# ---------- tier defaults from RAM ----------
# Low RAM: disk-bound → PILOT + aggressive anticipatory I/O, no DRAFT
# Mid: autopin + TRAJ + CUDA stack
# High: room for DRAFT if user opts in; larger pin frac
export SNAP
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 4)}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-passive}"
export DIRECT="${DIRECT:-1}"
export OVERLAP="${OVERLAP:-1}"
export I4S="${I4S:-1}"
export PREFETCH="${PREFETCH:-1}"
export MEMWATCH="${MEMWATCH:-1}"
export TRAJ="${TRAJ:-1}"
export TRAJ_K="${TRAJ_K:-6}"
export TRAJ_DEPTH="${TRAJ_DEPTH:-2}"
export FLOW="${FLOW:-1}"
export FLOW_R="${FLOW_R:-2}"
export ENERGY="${ENERGY:--1}"
export REPIN="${REPIN:-32}"
export THINK="${THINK:-0}"
export DRAFT="${DRAFT:-0}"

if (( MEM_I < 28 )); then
  # ~16–24 GB class (this box): PILOT pays; keep DRAFT off
  export PILOT="${PILOT:-1}"
  export PILOT_DEPTH="${PILOT_DEPTH:-1}"
  export SPEED="${SPEED:-1}"
  export TOPP="${TOPP:-0.55}"
  export TEMP="${TEMP:-0.7}"
  export PIN_FRAC="${PIN_FRAC:-0.90}"
  TIER="low-RAM (${MEM_GB} GB avail) — PILOT+SPEED, no DRAFT"
elif (( MEM_I < 48 )); then
  export PILOT="${PILOT:-1}"
  export PILOT_DEPTH="${PILOT_DEPTH:-1}"
  export SPEED="${SPEED:-1}"
  export TOPP="${TOPP:-0.55}"
  export TEMP="${TEMP:-0.7}"
  export PIN_FRAC="${PIN_FRAC:-0.85}"
  TIER="mid-RAM (${MEM_GB} GB avail) — pin/LRU scale, TRAJ multi-turn"
else
  export PILOT="${PILOT:-1}"
  export PILOT_DEPTH="${PILOT_DEPTH:-1}"
  export SPEED="${SPEED:-0}"
  export TOPP="${TOPP:-0.60}"
  export TEMP="${TEMP:-0.6}"
  export PIN_FRAC="${PIN_FRAC:-0.85}"
  # DRAFT only if user forces it: DRAFT=2 start_peng.sh ...
  TIER="high-RAM (${MEM_GB} GB avail) — warm cache; set DRAFT=2 only if cache is hot"
fi

if [[ "$CUDA_OK" -eq 1 ]]; then
  export COLI_CUDA="${COLI_CUDA:-1}"
  export CUDA_DENSE="${CUDA_DENSE:-1}"
else
  export COLI_CUDA=0
  export CUDA_DENSE=0
fi

# Explicit PIN file if present and PIN not set (else mimo AUTOPIN from .coli_usage)
if [[ -z "${PIN:-}" && -s "$USAGE" ]]; then
  # Leave PIN unset → engine autopin from usage; optional force:
  #   PIN="$USAGE" PIN_GB=6 scripts/start_peng.sh chat
  :
fi

print_banner() {
  echo "========== peng-mimo start =========="
  if [[ "${TAO:-0}" == "1" ]]; then
    echo " TAO=1  wu wei · sacred φ/Fibonacci · flow without force"
  fi
  echo " SNAP=$SNAP"
  echo " tier: $TIER"
  echo " COLI_CUDA=$COLI_CUDA CUDA_DENSE=$CUDA_DENSE"
  echo " PILOT=$PILOT PREFETCH=$PREFETCH TRAJ=$TRAJ FLOW=$FLOW ENERGY=$ENERGY SPEED=$SPEED"
  echo " TOPP=$TOPP TEMP=$TEMP DRAFT=$DRAFT REPIN=$REPIN"
  echo " OMP_NUM_THREADS=$OMP_NUM_THREADS DIRECT=$DIRECT"
  if [[ -s "$USAGE" ]]; then
    echo " usage history: $USAGE ($(wc -l <"$USAGE") lines)"
  else
    echo " usage history: (none yet — first runs train .coli_usage)"
  fi
  echo "====================================="
}

cmd_env() {
  print_banner
  cat <<EOF
export SNAP=$(printf %q "$SNAP")
export OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_WAIT_POLICY=$OMP_WAIT_POLICY
export DIRECT=$DIRECT OVERLAP=$OVERLAP I4S=$I4S
export COLI_CUDA=$COLI_CUDA CUDA_DENSE=$CUDA_DENSE
export PILOT=$PILOT PILOT_DEPTH=$PILOT_DEPTH PREFETCH=$PREFETCH
export TRAJ=$TRAJ TRAJ_K=$TRAJ_K TRAJ_DEPTH=$TRAJ_DEPTH FLOW=$FLOW FLOW_R=$FLOW_R ENERGY=$ENERGY
export SPEED=$SPEED TOPP=$TOPP TEMP=$TEMP DRAFT=$DRAFT
export REPIN=$REPIN MEMWATCH=$MEMWATCH PIN_FRAC=$PIN_FRAC THINK=$THINK
EOF
}

cmd_chat() {
  print_banner
  if [[ ! -x "$CODE/mimo" ]]; then
    echo "[start_peng] building mimo..."
    if [[ "$CUDA_OK" -eq 1 ]]; then make CUDA=1 mimo
    else make mimo
    fi
  fi
  # Prefer --fast on low/mid RAM
  extra=()
  if [[ "${SPEED:-0}" == "1" ]]; then extra+=(--fast); fi
  if [[ -n "${COLI_PROFILE:-}" ]]; then extra+=(--profile "$COLI_PROFILE"); fi
  exec python3 "$CODE/chat_peng.py" "${extra[@]}" "$@"
}

cmd_prompt() {
  local prompt="${1:-Write one short sentence about Rome.}"
  local ngen="${2:-24}"
  print_banner
  if [[ ! -x "$CODE/mimo" ]]; then
    if [[ "$CUDA_OK" -eq 1 ]]; then make CUDA=1 mimo
    else make mimo
    fi
  fi
  export PROMPT="$prompt" NGEN="$ngen"
  exec ./mimo 64 4 8
}

# If sourced: only set env
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  print_banner
  return 0 2>/dev/null || true
fi

mode="${1:-chat}"
shift || true
case "$mode" in
  env)    cmd_env ;;
  chat)   cmd_chat "$@" ;;
  prompt) cmd_prompt "$@" ;;
  help|-h|--help)
    sed -n '2,14p' "$0"
    ;;
  *)
    echo "usage: $0 [chat|prompt|env|help] ..."
    exit 2
    ;;
esac
