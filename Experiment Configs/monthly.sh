#!/bin/bash

#  nohup "./run.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt

# Monthly Swing Option Baseline:
args=( 
    # Training parameters - IMPROVED PATH GENERATION
    -n_paths=32768 # 4096 or 8192 or 16384 or 32768
    -eval_every=1024
    -n_paths_eval=4096 # Generates exactly 4096 evaluation paths (shared between RL and LSM)
    -munchausen=0
    -nstep=5
    -learn_every=2
    -per=1
    -iqn=0
    -noise=gauss                  # Use Gaussian noise instead of OU noise
    -bs=64
    -layer_size=128
    --min_replay_size=5000    # Increase from 1000 to 5000
    -t=5e-3
    -lr_a=2e-4 # from lr_a=3e-4
    -lr_c=2e-4 # from lr_c=3e-4
    --compile=0 # Disable JIT compilation for debugging
    -n_cores=4

    ################################################################################
    # Monthly Swing Option Contract Parameters
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
python run.py "${args[@]}" -name "MonthlySwing1" -seed 1
# python run.py "${args[@]}" -name "Fig10_100_32k" -seed 1
# python run.py "${args[@]}" -name "del" -seed 1
# python run.py "${args[@]}" -name "MonthlySwing_Baseline2" -seed 12 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline3" -seed 13 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline4" -seed 14 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline5" -seed 15 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline6" -seed 16 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline7" -seed 17 
# python run.py "${args[@]}" -name "MonthlySwing_Baseline8" -seed 18
# python run.py "${args[@]}" -name "MonthlySwing_Baseline9" -seed 19
