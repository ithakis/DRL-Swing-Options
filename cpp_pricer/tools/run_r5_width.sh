#!/usr/bin/env bash
# R5/H-A1 width-shrink equivalence sweep.
# Regenerates the width-64 baseline AND width-48/32 variants under IDENTICAL settings
# (M=2, n_train=3072, threads=8) so the per-seed pairing is clean — width does not
# affect path simulation, so fixed threads ⇒ identical OOS paths across widths.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./build_r1/price_swing
OUT=data/research
mkdir -p "$OUT"
THREADS=8
NTRAIN=3072
NEVAL=65536
SEEDS=$(seq 11 25)

for H in 64 48 32; do
  for s in $SEEDS; do
    $BIN --seed "$s" --n_train "$NTRAIN" --n_eval "$NEVAL" --hidden "$H" \
         --threads "$THREADS" --quiet > "$OUT/r5_h${H}_seed${s}.json" 2>/dev/null
    p=$(grep '"price"' "$OUT/r5_h${H}_seed${s}.json" | grep -oE '[0-9]+\.[0-9]+')
    echo "H=$H seed=$s price=$p"
  done
done
echo "=== R5 width sweep DONE ==="
