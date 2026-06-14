#!/usr/bin/env bash
# Task 2: can the NN be reduced further? actor width/depth + critic width. g2, 8 seeds, t4.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=build_g3i/price_swing
BASE="--c_cost 0.04 --gamma_cost 2 --hidden 48 --actor_layers 2 --critic_layers 4 --hidden_actor 32 --batch 64 --lr_c 5e-4 --threads 4 --quiet"
sweep(){ local tag="$1" extra="$2"; for s in $(seq 11 18); do
  $BIN --seed $s $BASE $extra > "data/explore/rd_${tag}_seed${s}.json" 2>/dev/null; done; echo "[$tag] done"; }
sweep base ""                  &
sweep a16  "--hidden_actor 16" & wait
sweep a24  "--hidden_actor 24" &
sweep a1L  "--actor_layers 1"  & wait
sweep c40  "--hidden_critic 40" &
sweep c32  "--hidden_critic 32" & wait
echo "ALL_REDUCE_DONE"
