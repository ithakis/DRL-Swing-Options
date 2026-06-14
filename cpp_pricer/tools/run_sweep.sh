#!/usr/bin/env bash
# Generic seed sweep. Usage: run_sweep.sh <tag> <extra price_swing flags...>
# Always: w48, n_train 3072, threads 8, seeds 11-25. Out: data/research/<tag>_seed{s}.json
set -euo pipefail
cd "$(dirname "$0")/.."
tag="$1"; shift
for s in $(seq 11 25); do
  ./build/price_swing --seed "$s" --n_train 3072 --hidden 48 --threads 8 "$@" --quiet \
    > "data/research/${tag}_seed${s}.json" 2>/dev/null
done
echo "sweep ${tag} done."
