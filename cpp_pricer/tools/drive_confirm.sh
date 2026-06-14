#!/usr/bin/env bash
# 15-seed confirmation: extend baseline + stacked "tuned" combos (a2c4/w48 base, canonical mesh).
set -euo pipefail
cd "$(dirname "$0")/.."
R=tools/run_explore.sh
$R b_a2c4 ""                                                            19 25  # extend base to 11-25
# tunedA = "less time, >= accuracy": narrow fast actor + init nudges (critic stays 48)
$R tunedA "--hidden_actor 32 --init_gain 1.0 --critic_out_init 0.02"    11 25
# tunedB = "max accuracy": + wide critic 64 (time ~neutral vs base after actor shrink)
$R tunedB "--hidden_actor 32 --hidden_critic 64 --init_gain 1.0 --critic_out_init 0.02"  11 25
echo "ALL_CONFIRM_DONE"
