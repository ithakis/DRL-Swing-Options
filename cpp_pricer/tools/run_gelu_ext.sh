#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for s in $(seq 19 25); do
  build_gelu/price_swing --seed $s --n_train 4096 --hidden 48 --actor_layers 2 --critic_layers 4 \
       --threads 8 --quiet > data/explore/gelu_seed${s}.json 2>/dev/null
  echo "[gelu] seed=$s $(python3 -c "import json;print(round(json.load(open('data/explore/gelu_seed${s}.json'))['price'],6))")"
done
echo "GELU_EXT_DONE"
