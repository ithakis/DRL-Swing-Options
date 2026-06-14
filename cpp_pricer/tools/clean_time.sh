#!/usr/bin/env bash
# clean_time.sh "LABEL:flags" ...  -> median t_train over REPS clean single-instance runs.
# Run ONLY when nothing else is using the CPU (timings are contention-sensitive).
set -euo pipefail
cd "$(dirname "$0")/.."
REPS=3; SEED=11
for spec in "$@"; do
  lab="${spec%%:*}"; flags="${spec#*:}"
  ts=()
  for r in $(seq 1 $REPS); do
    t=$(./build/price_swing --seed $SEED --n_train 4096 --hidden 48 --actor_layers 2 --critic_layers 4 \
         --threads 8 --quiet $flags 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['t_train'])")
    ts+=("$t")
  done
  med=$(printf '%s\n' "${ts[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  echo "$lab  t_train_median=${med}s  (runs: ${ts[*]})"
done
