#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=build_g3i/price_swing
A="--c_cost 0.04 --gamma_cost 2 --hidden 48 --actor_layers 2 --critic_layers 4 --hidden_actor 32 --init_method 0 --threads 4 --quiet"
sweep() { local tag="$1" flags="$2"; for s in $(seq 11 18); do
  $BIN --seed $s $flags $A > "data/explore/bat_${tag}_seed${s}.json" 2>/dev/null; done; echo "[$tag] done"; }
sweep b64lc3   "--batch 64 --lr_c 3e-4" &
sweep b64lc5   "--batch 64 --lr_c 5e-4" & wait
sweep b64lc3la "--batch 64 --lr_c 3e-4 --lr_a 1.5e-4" &
sweep b96lc5   "--batch 96 --lr_c 5e-4" & wait
echo "ALL_BATCH2_DONE"
