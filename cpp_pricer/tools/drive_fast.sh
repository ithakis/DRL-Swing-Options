#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
R=tools/run_explore.sh
$R wa32   "--hidden_actor 32"                  19 25   # extend wa32 to 11-25
$R a32rt  "--hidden_actor 32 --reuse_target 1" 11 25   # fast tier: narrow actor + reuse
echo "ALL_FAST_DONE"
