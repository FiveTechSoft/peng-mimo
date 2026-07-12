#!/bin/bash
# Matriz TOPP/TOPK sobre SCORE mode: perplexity + argmax dump por config.
# Uso (WSL): bash /mnt/c/peng-mimo/scripts/bench_topp_ppl.sh
set -e
cd /mnt/c/peng-mimo/c
REQ=/root/score_req.txt
OUT=/root/ppl_topp
mkdir -p "$OUT"

run(){ # $1=nombre  $2..=env extra
  NAME=$1; shift
  echo "=== $NAME ($*) ==="
  T0=$(date +%s)
  env "$@" SCORE=$REQ SCORE_DUMP=$OUT/$NAME.argmax DIRECT=1 SNAP=/root/mimo25_i4 \
      ./mimo 64 4 8 > $OUT/$NAME.lp 2> $OUT/$NAME.err || { echo "FAIL $NAME"; tail -5 $OUT/$NAME.err; return 1; }
  T1=$(date +%s)
  echo "$NAME $((T1-T0))s: $(cat $OUT/$NAME.lp | tr '\n' ' ')"
}

run BASE  DUMMY=1
run TP070 TOPP=0.7
run TP060 TOPP=0.6
run TP055 TOPP=0.55
run TP050 TOPP=0.5
run TK6   TOPK=6
run TK5   TOPK=5
echo "PPL_BENCH_DONE"
