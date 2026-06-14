#!/usr/bin/env bash
# Parametrized sweep. Usage: run_sweep_n.sh <tag> <n_train> <extra flags...>
set -euo pipefail
cd "$(dirname "$0")/.."
tag="$1"; nt="$2"; shift 2
for s in $(seq 11 25); do
  ./build/price_swing --seed "$s" --n_train "$nt" --hidden 48 --threads 8 "$@" --quiet \
    > "data/research/${tag}_seed${s}.json" 2>/dev/null
done
echo "sweep ${tag} (n_train=${nt}) done."
