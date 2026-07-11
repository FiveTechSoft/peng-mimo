#!/bin/bash
# Confronto byte-stretto dei segmenti di generazione (riga 4 fino a '---') tra i run.
# uso: mtp_cmp.sh <tagA> <tagB> [tagC...]  -> estrae ~/gate/b<tag>.out in g<tag>.txt e confronta col primo.
cd ~/gate || exit 1
first=""
for t in "$@"; do
  awk 'NR>=4 && /^---$/{exit} NR>=4' "b$t.out" > "g$t.txt"
  md5sum "g$t.txt"
  if [ -z "$first" ]; then first=$t; continue; fi
  if cmp -s "g$first.txt" "g$t.txt"; then echo "$first vs $t: IDENTICAL"
  else echo "$first vs $t: DIFFER"; diff "g$first.txt" "g$t.txt" | head -8; fi
done
echo "---TEXT($first)---"
cat "g$first.txt"
