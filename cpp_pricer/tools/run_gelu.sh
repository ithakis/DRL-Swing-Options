#!/usr/bin/env bash
# GELU sweep using the build_gelu binary (a2c4/w48 base, canonical mesh, threads 8).
set -euo pipefail
cd "$(dirname "$0")/.."
for s in $(seq 11 18); do
  build_gelu/price_swing --seed $s --n_train 4096 --hidden 48 --actor_layers 2 --critic_layers 4 \
       --threads 8 --quiet > data/explore/gelu_seed${s}.json 2>/dev/null
  p=$(python3 -c "import json;print(f\"{json.load(open('data/explore/gelu_seed${s}.json'))['price']:.6f}\")")
  echo "[gelu] seed=$s price=$p"
done
echo "ALL_GELU_DONE"
