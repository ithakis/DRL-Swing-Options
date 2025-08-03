#!/bin/bash

# LSM Swing Option Pricer Runner Script
# Usage: bash runlsm.sh
# 
# This script runs the Least Squares Monte Carlo (LSM) pricer for swing options
# using the same contract parameters as the RL training in run.sh for benchmarking

# To increase accuracy, increase n_paths (computational time scales quadratically)
# To speed up execution, decrease n_paths or n_scenarios

# Monthly Swing Option LSM Pricing with same parameters as run.sh:
args=( 
    # LSM Algorithm Parameters
    -n_paths=2048                  # Number of Monte Carlo paths (increase for accuracy)
    -poly_degree=3                 # Polynomial degree for regression (2-4 recommended)
    -seed=42                       # Random seed for reproducibility
    -n_scenarios=10                # Number of scenarios for strategy simulation
    --plot=1                       # Create plots (set to 0 to disable)
    --verbose=1                    # Print detailed output (set to 0 for quiet)
    
    # Monthly Swing Option Contract Parameters (matching run.sh exactly)
    --strike=1.0              # At-the-money strike
    --maturity=0.0833           # 1 month = 1/12 year
    --n_rights=22               # ~22 trading days in a month
    --q_min=0.0                 # No minimum exercise requirement per day
    --q_max=2.0                 # Max 2 units per day (reasonable daily limit)
    --Q_min=0.0                 # No minimum total exercise requirement
    --Q_max=20.0                # Max 20 units total over the month (10 days worth)
    --risk_free_rate=0.05       # 5% annual risk-free rate
    --min_refraction_days=0     # No refraction period (can exercise daily)
    
    # Market Process Parameters (monthly calibration, exactly matching run.sh)
    --S0=1.0                  # Initial spot price
    --alpha=12.0                # Higher mean reversion for monthly timeframe
    --sigma=1.2                 # Moderate volatility for monthly period
    --beta=150.0                # Jump decay rate
    --lam=6.0                   # Jump intensity (6 jumps per year average)
    --mu_J=0.3                  # Mean jump size (30%)
)
python -m src.lsm_swing_pricer "${args[@]}" -name "LSM_MonthlySwing_1" -seed 1