#!/bin/bash
set -e

configs=(
    "SwingOption_20_c0.00_gamma1"
    "SwingOption_20_c0.04_gamma1"
    "SwingOption_20_c0.04_gamma1.5"
    "SwingOption_20_c0.04_gamma2"
)

config_csv=$(IFS=,; echo "${configs[*]}")

python tools/sweep_delta_s0.py \
    --configs "$config_csv" \
    --baseline-config "SwingOption_20_c0.00_gamma1" \
    --output-dir "logs/hedging_s0_sweep" \
    --s0-min 0.50 \
    --s0-max 2.00 \
    --s0-count 31 \
    --gp-grid-count 151 \
    --n-paths-eval 16384 \
    --lsm-train-paths 16384 \
    --eval-seed 999 \
    --lsm-train-seed 998 \
    --eval-batch-size 2048 \
    --max-seeds 3 \
    --preferred-rl-seeds 11,12,13 \
    --lsm-basis chebyshev \
    --lsm-degree 2 \
    --lsm-reg none \
    --lsm-reg-alpha 1e-6 \
    --lsm-n-actions 5

## To activate the correct environment, run:
# cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options && conda activate EP11
# bash delta_sweep.sh