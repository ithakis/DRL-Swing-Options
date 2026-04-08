#!/bin/bash
# Complete the focal robustness study: evaluate all 15 seeds, generate figure, rebuild PDF.
# Run this AFTER the focal training (SwingOption_20_c0.04_gamma2_focal.sh) finishes.
#
# Usage:
#   bash tools/complete_focal_study.sh

set -euo pipefail
cd "$(dirname "$0")/.."

FOCAL_CSV="Jupyter Notebooks/Convex Costs Results 7 focal.csv"
SEEDS="11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
CONFIG="SwingOption_20_c0.04_gamma2"

echo "============================================================"
echo "  Focal Robustness Study — Completion Pipeline"
echo "============================================================"

# ---------------------------------------------------------------
# Step 0: Verify all 15 seed models exist
# ---------------------------------------------------------------
echo ""
echo "--- Step 0: Verifying trained models ---"
missing=0
for seed in 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
    pth="runs/${CONFIG}_${seed}.pth"
    if [ ! -f "$pth" ]; then
        echo "  MISSING: $pth"
        missing=$((missing + 1))
    fi
done
if [ "$missing" -gt 0 ]; then
    echo "ERROR: $missing model(s) missing. Run focal training first."
    exit 1
fi
echo "  All 15 models found."

# ---------------------------------------------------------------
# Step 1: Evaluate all 15 seeds on common test set
# ---------------------------------------------------------------
echo ""
echo "--- Step 1: Evaluating 15 seeds on common test set ---"
python tools/rebuild_results_v7.py \
    --focal_config "$CONFIG" \
    --focal_seeds "$SEEDS" \
    --output "$FOCAL_CSV"

echo ""
echo "  Focal CSV: $FOCAL_CSV"

# ---------------------------------------------------------------
# Step 2: Generate the seed robustness figure
# ---------------------------------------------------------------
echo ""
echo "--- Step 2: Generating seed robustness figure ---"
python tools/generate_seed_robustness_figure.py --csv "$FOCAL_CSV"

echo ""
echo "  Figure: figs/convex_costs_0p04/seed_robustness.pdf"

# ---------------------------------------------------------------
# Step 3: Rebuild the paper PDF
# ---------------------------------------------------------------
echo ""
echo "--- Step 3: Rebuilding paper PDF ---"
./tools/build_latex.sh "$PWD/Paper" DRL_Swing_Options.tex

echo ""
echo "============================================================"
echo "  Done! PDF at: Paper/build/DRL_Swing_Options.pdf"
echo "============================================================"
