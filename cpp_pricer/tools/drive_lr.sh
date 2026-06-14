#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tools/run_explore.sh b_a2c4 ""                       11 18
tools/run_explore.sh lrc9   "--lr_c 9e-4"            11 18
tools/run_explore.sh lrc12  "--lr_c 0.0012"          11 18
tools/run_explore.sh lrac   "--lr_a 5e-4 --lr_c 9e-4" 11 18
echo "ALL_LR_DONE"
