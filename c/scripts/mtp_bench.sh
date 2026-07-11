#!/bin/bash
# T14 gate (c): 32-token greedy measurement on a chat-formatted prompt.
# uso: mtp_bench.sh <draft> <tag> [env extra, es. "TOPP=0.7 DIRECT=1"]
# Output in ~/gate/b<tag>.out (persistente: /tmp e' tmpfs e sparisce tra invocazioni WSL).
d=${1:?uso: mtp_bench.sh <draft> <tag> [extra-env]}
tag=${2:?serve un tag}
extra=${3:-}
mkdir -p ~/gate
cd /mnt/c/colibri/c || exit 1
P=$'<|im_start|>system\nYou are MiMo, a helpful AI assistant engineered by Xiaomi.<|im_end|><|im_start|>user\nExplain in simple terms why the sky is blue.<|im_end|><|im_start|>assistant\n<think></think>'
env $extra PROMPT="$P" NGEN=32 TEMP=0 DRAFT=$d SNAP=$HOME/mimo25_i4 ./mimo 64 4 8 \
  > ~/gate/b$tag.out 2> ~/gate/b$tag.err
echo "== DRAFT=$d $extra"
grep -E "token in|speculazione|MTP acceptance" ~/gate/b$tag.out
