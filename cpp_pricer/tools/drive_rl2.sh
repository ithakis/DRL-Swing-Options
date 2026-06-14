#!/usr/bin/env bash
# Thorough batch-64 RL retune (user skepticism). g2, 8 seeds, --threads 2, 4 CONCURRENT.
# Tests the batch-coupled knobs the first screen skipped: learn_number, learn_every, EMA, joint LR, noise.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=build_g3i/price_swing
ARCH="--c_cost 0.04 --gamma_cost 2 --hidden 48 --actor_layers 2 --critic_layers 4 --hidden_actor 32 --threads 2 --quiet"
B64="--batch 64 --lr_c 5e-4"
sweep(){ local tag="$1" extra="$2"; for s in $(seq 11 18); do
  $BIN --seed $s $ARCH $extra > "data/explore/r2_${tag}_seed${s}.json" 2>/dev/null; done; echo "[$tag] done"; }
# config list: "tag:flags"
CFG=(
 "b64base:$B64"
 "b128ref:--batch 128"
 "ln3:$B64 --learn_number 3"
 "ln4:$B64 --learn_number 4"
 "ln1:$B64 --learn_number 1"
 "lec1:$B64 --learn_every 1"
 "ema9995:$B64 --ema_decay 0.9995"
 "ema9999:$B64 --ema_decay 0.9999"
 "nf20:$B64 --noise_floor 0.20"
 "nf35:$B64 --noise_floor 0.35"
 "wnf4:$B64 --warmup_noise_fraction 0.4"
 "lrA4C6:$B64 --lr_a 4e-4 --lr_c 6e-4"
 "lrA2C4:$B64 --lr_a 2e-4 --lr_c 4e-4"
 "ln4lec1:$B64 --learn_number 4 --learn_every 1"
)
i=0
for c in "${CFG[@]}"; do
  sweep "${c%%:*}" "${c#*:}" &
  i=$((i+1)); [ $((i%4)) -eq 0 ] && wait
done
wait
echo "ALL_RL2_DONE"
