#!/usr/bin/env bash
# Batch+LR study at BEST config (GELU β3 + actor32). Args: INIT_METHOD [SEED_LO SEED_HI]
# Smaller batch => proportionally faster train (cost ∝ B). Co-tune LR to recover accuracy.
# threads 4, runs pairs concurrently. Focal g2.
set -euo pipefail
cd "$(dirname "$0")/.."
IM="${1:-0}"; LO="${2:-11}"; HI="${3:-18}"
BIN=build_g3i/price_swing
A="--c_cost 0.04 --gamma_cost 2 --hidden 48 --actor_layers 2 --critic_layers 4 --hidden_actor 32 --init_method $IM --threads 4 --quiet"
sweep() { local tag="$1" flags="$2"; for s in $(seq $LO $HI); do
  $BIN --seed $s $flags $A > "data/explore/bat_${tag}_seed${s}.json" 2>/dev/null; done; echo "[$tag] done"; }
# base batch 128 at this config (reference for paired comparison)
sweep b128   "--batch 128" &
sweep b64    "--batch 64"  & wait
sweep b96    "--batch 96"  &
sweep b64lc4 "--batch 64 --lr_c 4e-4" & wait
sweep b64ls  "--batch 64 --lr_a 2.1e-4 --lr_c 4.2e-4" &   # sqrt-scaling x0.71
sweep b64lc8 "--batch 64 --lr_c 8e-4" & wait
echo "ALL_BATCH_DONE"
