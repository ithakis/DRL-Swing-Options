#!/usr/bin/env bash
# Sweep the swish gate steepness beta in g(x)=x*sigmoid(beta*x). 8-seed screen.
set -euo pipefail
cd "$(dirname "$0")/.."
build_one() {  # beta tag
  local beta="$1" tag="$2"
  cmake -S . -B "build_$tag" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-DPRICER_GELU_FAST -DGELU_SLOPE=$beta" >/dev/null 2>&1
  cmake --build "build_$tag" --target price_swing -j4 >/dev/null 2>&1
}
run_one() {    # tag binary
  local tag="$1"
  for s in $(seq 11 18); do
    "build_$tag/price_swing" --seed $s --n_train 4096 --hidden 48 --actor_layers 2 --critic_layers 4 \
        --threads 8 --quiet > "data/explore/${tag}_seed${s}.json" 2>/dev/null
  done
  echo "[$tag] done"
}
build_one 1.4 b14; run_one b14
build_one 2.0 b20; run_one b20
build_one 2.5 b25; run_one b25
echo "ALL_BETA_DONE"
