#!/usr/bin/env bash
# Width + Init screening sweep (8 seeds, a2c4/w48 base, canonical mesh, threads 8).
set -euo pipefail
cd "$(dirname "$0")/.."
R=tools/run_explore.sh
# --- width: narrow actor (speed) / wide critic (accuracy) ---
$R wa32  "--hidden_actor 32 --hidden_critic 48"  11 18
$R wa24  "--hidden_actor 24 --hidden_critic 48"  11 18
$R wc64  "--hidden_actor 48 --hidden_critic 64"  11 18
# --- init: critic output scale (start Q near true magnitude) ---
$R coi05 "--critic_out_init 0.05"                11 18
$R coi02 "--critic_out_init 0.02"                11 18
# --- init: orthogonal gain bracket (LN largely cancels; confirm) ---
$R gain1 "--init_gain 1.0"                       11 18
echo "ALL_WI_DONE"
