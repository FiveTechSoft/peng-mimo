#!/usr/bin/env bash
# One SERVE (chat) turn — same protocol as chat_peng.py
set -euo pipefail
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

cd "$(cd "$(dirname "$0")/.." && pwd)"
export SNAP="${SNAP:-/root/mimo25_i4}"
export COLI_CUDA="${COLI_CUDA:-1}" CUDA_DENSE="${CUDA_DENSE:-1}"
export DIRECT="${DIRECT:-1}" TOPP="${TOPP:-0.6}" THINK="${THINK:-0}"
export NGEN="${NGEN:-32}" TEMP="${TEMP:-0.7}" SERVE=1

PROMPT="${1:-Write one short sentence about Rome.}"
ERR=/root/peng_serve.err
OUT=/root/peng_serve.out
rm -f "$ERR" "$OUT"

echo "=== SERVE turn: $PROMPT ===" | tee "$ERR"
echo "START $(date -Is)" | tee -a "$ERR"

# stdin: one user line then close (EOF ends serve loop after END)
# Ready marker then response streaming then END then STAT
{
  # wait for READY on stdout while capturing stderr to ERR
  ./mimo 64 4 8 2>>"$ERR" | {
    # consume until READY
    buf=""
    while IFS= read -r -n1 c; do
      buf+="$c"
      if [[ "$buf" == *$'\x01\x01READY\x01\x01'* ]]; then break; fi
    done
    # rest of READY line + STAT 0
    read -r _ || true
    read -r _ || true
    echo "READY at $(date -Is)" >>"$ERR"
    # send prompt via a coprocess is hard with pure pipe; use fifo
    true
  }
} 

# Simpler approach: use Python one-shot like chat_peng
python3 - <<'PY' 2>>"$ERR" | tee "$OUT"
import os, subprocess, sys, time
snap = os.environ["SNAP"]
here = os.path.dirname(os.path.abspath("mimo")) or "."
mimo = os.path.join(os.getcwd(), "mimo")
env = dict(os.environ)
env.update(SNAP=snap, SERVE="1", THINK=os.environ.get("THINK","0"),
           NGEN=os.environ.get("NGEN","32"), TOPP=os.environ.get("TOPP","0.6"),
           DIRECT=os.environ.get("DIRECT","1"), TEMP=os.environ.get("TEMP","0.7"),
           COLI_CUDA=os.environ.get("COLI_CUDA","1"),
           CUDA_DENSE=os.environ.get("CUDA_DENSE","1"))
prompt = os.environ.get("CHAT_PROMPT", "Write one short sentence about Rome.")
t0 = time.time()
p = subprocess.Popen([mimo, "64", "4", "8"], cwd=os.getcwd(), env=env,
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
READY = b"\x01\x01READY\x01\x01"
END = b"\x01\x01END\x01\x01"
buf = b""
while READY not in buf:
    c = p.stdout.read(1)
    if not c:
        err = p.stderr.read().decode(errors="replace")
        sys.stderr.write(err)
        sys.exit("engine died during load")
    buf += c
p.stdout.readline(); p.stdout.readline()
load_s = time.time() - t0
print(f"[chat] model ready in {load_s:.1f}s", flush=True)
p.stdin.write(prompt.encode() + b"\n"); p.stdin.flush()
print("peng> ", end="", flush=True)
pend = b""; t1 = time.time()
while True:
    c = p.stdout.read(1)
    if not c:
        sys.exit("engine died mid-turn")
    pend += c
    if pend.endswith(END):
        break
    hold = 0
    for k in range(min(len(END)-1, len(pend)), 0, -1):
        if pend.endswith(END[:k]):
            hold = k; break
    if len(pend) > hold:
        sys.stdout.buffer.write(pend[:len(pend)-hold]); sys.stdout.flush()
        pend = pend[len(pend)-hold:]
stat = p.stdout.readline().decode().split()
gen_s = time.time() - t1
print(flush=True)
if len(stat) >= 4 and stat[0] == "STAT":
    print(f"[chat] STAT: {stat[1]} tokens · {float(stat[2]):.2f} tok/s · hit {stat[3]}% · RSS {stat[4]} GB  (wall gen {gen_s:.1f}s)", flush=True)
else:
    print(f"[chat] raw STAT line: {stat}", flush=True)
# drain stderr summary
err = p.stderr.read().decode(errors="replace")
open("/root/peng_serve.err","a").write(err)
# keep useful tail
for line in err.splitlines():
    if any(k in line for k in ("[CUDA]", "[PIN]", "[RAM", "dense eager", "complementary", "resident set")):
        print(line, file=sys.stderr)
p.stdin.close(); p.terminate()
print(f"[chat] total wall {time.time()-t0:.1f}s", flush=True)
PY

echo "END $(date -Is)" | tee -a "$ERR"
