#!/usr/bin/env bash
# fast sigmoid-GELU sweep (15 seeds) using build_gfast.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in $(seq 11 25); do
  build_gfast/price_swing --seed $s --n_train 4096 --hidden 48 --actor_layers 2 --critic_layers 4 \
       --threads 8 --quiet > data/explore/gfast_seed${s}.json 2>/dev/null
  echo "[gfast] seed=$s $(python3 -c "import json;print(round(json.load(open('data/explore/gfast_seed${s}.json'))['price'],6))")"
done
echo "GFAST_DONE"
