#!/usr/bin/env bash
# Init-method study at the BEST config (GELU β3 + actor32 + a2c4/w48), focal g2, 15 seeds.
# He arm = reuse reg_gelu_g2. This adds orthogonal(1) + Xavier(2). threads 4, 2 concurrent.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=build_g3i/price_swing
ARCH="--c_cost 0.04 --gamma_cost 2 --hidden 48 --actor_layers 2 --critic_layers 4 --hidden_actor 32 --threads 4 --quiet"
sweep() { local tag="$1" m="$2"; for s in $(seq 11 25); do
  $BIN --seed $s --init_method $m $ARCH > "data/explore/init_${tag}_seed${s}.json" 2>/dev/null; done; echo "[$tag] done"; }
sweep ortho 1 & sweep xavier 2 & wait
echo "ALL_INIT_DONE"
