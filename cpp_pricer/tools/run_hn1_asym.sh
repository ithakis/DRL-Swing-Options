#!/usr/bin/env bash
# H-N1: asymmetric — shallow actor (depth 2), full critic (depth 3). w48/M2/3072, threads 8.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in $(seq 11 25); do
  ./build/price_swing --seed "$s" --n_train 3072 --hidden 48 \
    --actor_layers 2 --critic_layers 3 --threads 8 --quiet \
    > "data/research/hn1_a2c3_seed${s}.json" 2>/dev/null
  echo "seed $s a2c3 price=$(grep '"price"' data/research/hn1_a2c3_seed${s}.json | grep -oE '[0-9.]+')"
done
echo "H-N1 a2c3 sweep done."
