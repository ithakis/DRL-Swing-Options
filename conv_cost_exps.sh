#!/bin/bash
set -e

scripts=(
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.08_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.10_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.15_gamma2.sh"
    # New experiments.
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.08_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.10_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.15_gamma1.sh"
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.10_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.15_gamma1.5.sh"
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.08_gamma2.sh"
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma3.sh"
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma3.sh"
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma3.sh"
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma3.sh"
)

for script in "${scripts[@]}"; do
    echo "Running $script"
    bash "$script"
    echo "Completed $script"
done
