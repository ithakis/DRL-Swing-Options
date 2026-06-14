#!/usr/bin/env bash
# Speed-cadence screen (8 seeds, a2c4/w48 base, canonical mesh, threads 8).
set -euo pipefail
cd "$(dirname "$0")/.."
R=tools/run_explore.sh
$R rt     "--reuse_target 1"                  11 18   # 1 kernel eval/interaction
$R ln1    "--learn_number 1"                  11 18   # halve gradient steps
$R rtln1  "--reuse_target 1 --learn_number 1" 11 18   # both (max speed)
echo "ALL_SPEED_DONE"
