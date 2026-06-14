#!/usr/bin/env bash
# FINAL stack vs ORIGINAL, 3 regimes, 15 seeds, threads 4, 2 concurrent.
# ORIG  = SiLU + actor48 + batch128 (build, new code)
# FINAL = GELU β3 + actor32 + batch64 + lr_c 5e-4 (build_g3i)
set -euo pipefail
cd "$(dirname "$0")/.."
declare -A REG=( [nocost]="--c_cost 0" [g1]="--c_cost 0.04 --gamma_cost 1" [g2]="--c_cost 0.04 --gamma_cost 2" )
COMMON="--hidden 48 --actor_layers 2 --critic_layers 4 --threads 4 --quiet"
orig()  { local r="$1"; for s in $(seq 11 25); do ./build/price_swing     --seed $s ${REG[$r]} $COMMON > "data/explore/fin_orig_${r}_seed${s}.json" 2>/dev/null; done; echo "[orig $r] done"; }
final() { local r="$1"; for s in $(seq 11 25); do ./build_g3i/price_swing  --seed $s ${REG[$r]} $COMMON --hidden_actor 32 --batch 64 --lr_c 5e-4 > "data/explore/fin_final_${r}_seed${s}.json" 2>/dev/null; done; echo "[final $r] done"; }
for r in nocost g1 g2; do orig "$r" & final "$r" & wait; done
echo "ALL_FINAL_DONE"
