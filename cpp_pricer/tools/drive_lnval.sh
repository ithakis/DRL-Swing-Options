#!/usr/bin/env bash
# Validate learn_number winners across 3 regimes, 15 seeds, --threads 2, 4 concurrent.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=build_g3i/price_swing
COMMON="--hidden 48 --actor_layers 2 --critic_layers 4 --hidden_actor 32 --threads 2 --quiet"
declare -A REG=( [nocost]="--c_cost 0" [g1]="--c_cost 0.04 --gamma_cost 1" [g2]="--c_cost 0.04 --gamma_cost 2" )
sweep(){ local tag="$1" rf="$2" flags="$3"; for s in $(seq 11 25); do
  $BIN --seed $s $rf $COMMON $flags > "data/explore/lv_${tag}_seed${s}.json" 2>/dev/null; done; echo "[$tag] done"; }
CFG=()
for r in nocost g1 g2; do
  CFG+=("b128_${r}:${REG[$r]}:--batch 128")
  CFG+=("ln3_${r}:${REG[$r]}:--batch 64 --lr_c 5e-4 --learn_number 3")
  CFG+=("ln4_${r}:${REG[$r]}:--batch 64 --lr_c 5e-4 --learn_number 4")
done
i=0
for c in "${CFG[@]}"; do
  IFS=':' read -r tag rf fl <<< "$c"
  sweep "$tag" "$rf" "$fl" &
  i=$((i+1)); [ $((i%4)) -eq 0 ] && wait
done
wait
echo "ALL_LNVAL_DONE"
