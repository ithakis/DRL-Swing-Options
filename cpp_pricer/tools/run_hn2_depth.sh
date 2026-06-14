#!/usr/bin/env bash
# H-N2: depth-2 actor+critic vs the conservative depth-3 baseline (r5_h48_*).
# Matched otherwise: w48, M2 (n_max default 0), n_train 3072, threads 8, seeds 11-25.
# Out: data/research/hn2_d2_seed{s}.json   (compare vs data/research/r5_h48_seed{s}.json)
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./build/price_swing
OUT=data/research
mkdir -p "$OUT"
for s in $(seq 11 25); do
    "$BIN" --seed "$s" --n_train 3072 --hidden 48 \
           --actor_layers 2 --critic_layers 2 \
           --threads 8 --quiet > "$OUT/hn2_d2_seed${s}.json" 2>/dev/null
    p=$(grep '"price"' "$OUT/hn2_d2_seed${s}.json" | grep -oE '[0-9.]+')
    echo "seed $s  depth2  price=$p"
done
echo "H-N2 depth-2 sweep done."
