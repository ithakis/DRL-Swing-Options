#!/usr/bin/env bash
# run_explore.sh LABEL "EXTRA_FLAGS" SEED_LO SEED_HI [N_TRAIN]
# Runs price_swing across CRN seeds at fixed --threads 8 on the canonical exported mesh,
# dumping one JSON per seed to data/explore/<LABEL>_seed<S>.json.
# Baseline arch is a2c4/w48 unless EXTRA_FLAGS overrides it.
set -euo pipefail
LABEL="$1"; EXTRA="$2"; LO="${3:-11}"; HI="${4:-18}"; NTR="${5:-4096}"
BIN=./build/price_swing
OUT=data/explore
mkdir -p "$OUT"
for s in $(seq "$LO" "$HI"); do
  $BIN --seed "$s" --n_train "$NTR" --hidden 48 --actor_layers 2 --critic_layers 4 \
       --threads 8 --quiet $EXTRA > "$OUT/${LABEL}_seed${s}.json" 2>/dev/null
  p=$(python3 -c "import json;print(f\"{json.load(open('$OUT/${LABEL}_seed${s}.json'))['price']:.6f}\")")
  echo "[$LABEL] seed=$s price=$p"
done
echo "[$LABEL] DONE"
