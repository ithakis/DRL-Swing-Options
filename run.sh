#!/bin/bash

#  nohup "./run.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt

# Monthly Swing Option Baseline:
args=( 
    # Training parameters - IMPROVED PATH GENERATION
    -n_paths=16384 # 4096 or 8192 or 16384 or 32768
    -eval_every=1024
    -n_paths_eval=4096 # Generates exactly 4096 evaluation paths (shared between RL and LSM)
    -munchausen=0
    -nstep=5
    --gamma=0.99
    -learn_every=2
    -iqn=0
    -noise=gauss                  # Use Gaussian noise instead of OU noise
    -epsilon=0.5                   # Default epsilon for exploration noise - gaussian noise
    -epsilon_decay=1.0             # Default epsilon decay rate
    -per=1
    --min_replay_size=10000    # Increase from 1000 to 5000
    --max_replay_size=1000000     # Increase from 1e5 to 1e6
    -t=0.00001
    -bs=64
    -layer_size=128
    -lr_a=1e-05 # from lr_a=3e-4
    -lr_c=2e-4 # from lr_c=3e-4
    --compile=0 # Disable JIT compilation for debugging
    -n_cores=2
    
    ################################################################################
    # # Monthly Swing Option Contract Parameters
    --strike=1.0              # At-the-money strike
    --maturity=0.0833           # 1 month = 1/12 year
    --n_rights=22               # ~22 trading days in a month
    --q_min=0.0                 # No minimum exercise requirement per day
    --q_max=2.0                 # Max 2 units per day (reasonable daily limit)
    --Q_min=0.0                 # No minimum total exercise requirement
    --Q_max=20.0                # Max 20 units total over the month (10 days worth)
    --risk_free_rate=0.05       # 5% annual risk-free rate
    --min_refraction_days=0     # No refraction period (can exercise daily)
    
    # Market Process Parameters (monthly calibration)
    --S0=1.0                  # Initial spot price
    --alpha=12.0                # Higher mean reversion for monthly timeframe
    --sigma=1.2                 # Moderate volatility for monthly period
    --beta=150.0                # Jump decay rate
    --lam=6.0                   # Jump intensity (6 jumps per year average)
    --mu_J=0.3                  # Mean jump size (30%)
)
python run.py "${args[@]}" -name "MonthlySwing24" -seed 1
# python run.py "${args[@]}" -name "MonthlySwing2" -seed 2 &
# python run.py "${args[@]}" -name "MonthlySwing3" -seed 3 &
# python run.py "${args[@]}" -name "MonthlySwing4" -seed 4