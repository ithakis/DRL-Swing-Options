#!/bin/bash
set -e

scripts=(
    # "Convex Cost Experiments/SwingOption_20_c0.01_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma3.sh"
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma3.sh"
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma3.sh"
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma3.sh"
    "Convex Cost Experiments/SwingOption_20_c0.08_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.08_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.08_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.10_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.10_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.10_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.15_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.15_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.15_gamma2.sh"
)

for script in "${scripts[@]}"; do
    echo "Running $script"
    bash "$script"
    echo "Completed $script"
    echo "Cleaning evaluation Parquet files for $script"
    bash "./cleanup_eval_csvs.sh" "$script"
done


## To activate the corect environment, run:
# cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options && conda activate EP11
# bash conv_cost_exps.sh
