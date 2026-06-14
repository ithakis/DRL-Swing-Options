#!/usr/bin/env bash
# RL-space screen for GELU β3 + actor32 + batch64. g2, 8 seeds, threads 4, 2 concurrent.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=build_g3i/price_swing
BASE="--c_cost 0.04 --gamma_cost 2 --hidden 48 --actor_layers 2 --critic_layers 4 --hidden_actor 32 --batch 64 --lr_c 5e-4 --threads 4 --quiet"
sweep(){ local tag="$1" extra="$2"; for s in $(seq 11 18); do
  $BIN --seed $s $BASE $extra > "data/explore/rl_${tag}_seed${s}.json" 2>/dev/null; done; echo "[$tag] done"; }
# base
sweep base   "" &
sweep lra2   "--lr_a 2e-4" & wait
sweep lra4   "--lr_a 4e-4" &
sweep lrc4   "--lr_c 4e-4" & wait
sweep lrc6   "--lr_c 6e-4" &
sweep ans4   "--adaptive_noise_scale 0.4" & wait
sweep ans8   "--adaptive_noise_scale 0.8" &
sweep sig10  "--noise_sigma0 1.0" & wait
sweep sig16  "--noise_sigma0 1.6" &
sweep tau2   "--tau 0.002" & wait
sweep tau5   "--tau 0.005" &
sweep cw256  "--critic_warmup 256" & wait
sweep cw768  "--critic_warmup 768" & wait
echo "ALL_RL_DONE"
