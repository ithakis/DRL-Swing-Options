#!/usr/bin/env bash
# Path-reduction: does the tuned combo at FEWER paths match base a2c4 @4096?
# Direct "less time" test (fewer episodes = less wall-clock). 15 seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
R=tools/run_explore.sh
TUNED="--hidden_actor 32 --init_gain 1.0 --critic_out_init 0.02"
$R tunedA_3072 "$TUNED" 11 25 3072
$R tunedA_3584 "$TUNED" 11 25 3584
echo "ALL_PATHS_DONE"
