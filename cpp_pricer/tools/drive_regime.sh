#!/usr/bin/env bash
# Cross-regime GELU-vs-SiLU validation. Args: SILU_BIN GELU_BIN [SEED_LO SEED_HI]
# Runs all 3 regimes x {SiLU,GELU} at --threads 4, 2 sweeps concurrent (8 cores).
# Records price + cpu_train (load-independent). Outputs data/explore/reg_<tag>_seed<s>.json
set -euo pipefail
cd "$(dirname "$0")/.."
SILU="$1"; GELU="$2"; LO="${3:-11}"; HI="${4:-25}"
ARCH="--hidden 48 --actor_layers 2 --critic_layers 4 --hidden_actor 32 --threads 4 --quiet"
declare -A REG=( [nocost]="--c_cost 0" [g1]="--c_cost 0.04 --gamma_cost 1" [g2]="--c_cost 0.04 --gamma_cost 2" )
sweep() {  # tag binary regflags
  local tag="$1" bin="$2" rf="$3"
  for s in $(seq "$LO" "$HI"); do
    "$bin" --seed "$s" $rf $ARCH > "data/explore/reg_${tag}_seed${s}.json" 2>/dev/null
  done
  echo "[$tag] done"
}
for r in nocost g1 g2; do
  sweep "silu_${r}" "$SILU" "${REG[$r]}" &
  sweep "gelu_${r}" "$GELU" "${REG[$r]}" &
  wait   # 2 concurrent sweeps (one SiLU + one GELU) per regime
done
echo "ALL_REGIME_DONE"
