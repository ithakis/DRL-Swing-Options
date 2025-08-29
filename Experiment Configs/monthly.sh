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
    -n_cores=2

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
    --min_refraction_periods=0     # No refraction period (can exercise consecutive periods)
    
    # Market Process Parameters (monthly calibration)
    --S0=1.0                  # Initial spot price
    --alpha=12.0                # Higher mean reversion for monthly timeframe
    --sigma=1.2                 # Moderate volatility for monthly period
    --beta=150.0                # Jump decay rate
    --lam=6.0                   # Jump intensity (6 jumps per year average)
    --mu_J=0.3                  # Mean jump size (30%)
)
# python run.py "${args[@]}" -name "MonthlySwing1" -seed 1
# python run.py "${args[@]}" -name "MonthlySwing2" -seed 2
# python run.py "${args[@]}" -name "MonthlySwing3" -seed 3
# python run.py "${args[@]}" -name "MonthlySwing4" -seed 4
# python run.py "${args[@]}" -name "MonthlySwing5" -seed 5
# python run.py "${args[@]}" -name "MonthlySwing6" -seed 6
# python run.py "${args[@]}" -name "MonthlySwing7" -seed 7
# python run.py "${args[@]}" -name "MonthlySwing8" -seed 8
# python run.py "${args[@]}" -name "MonthlySwing9" -seed 9

python run.py "${args[@]}" -name "MonthlySwing10" -seed 10 &
python run.py "${args[@]}" -name "MonthlySwing11" -seed 11 &
python run.py "${args[@]}" -name "MonthlySwing12" -seed 12 &
python run.py "${args[@]}" -name "MonthlySwing13" -seed 13
python run.py "${args[@]}" -name "MonthlySwing14" -seed 14 &
python run.py "${args[@]}" -name "MonthlySwing15" -seed 15 &
python run.py "${args[@]}" -name "MonthlySwing16" -seed 16 &
python run.py "${args[@]}" -name "MonthlySwing17" -seed 17 
python run.py "${args[@]}" -name "MonthlySwing18" -seed 18 &
python run.py "${args[@]}" -name "MonthlySwing19" -seed 19 &
python run.py "${args[@]}" -name "MonthlySwing20" -seed 20 &
python run.py "${args[@]}" -name "MonthlySwing21" -seed 21
python run.py "${args[@]}" -name "MonthlySwing22" -seed 22 &
python run.py "${args[@]}" -name "MonthlySwing23" -seed 23 &
python run.py "${args[@]}" -name "MonthlySwing24" -seed 24 &
python run.py "${args[@]}" -name "MonthlySwing25" -seed 25
python run.py "${args[@]}" -name "MonthlySwing26" -seed 26 &
python run.py "${args[@]}" -name "MonthlySwing27" -seed 27 &
python run.py "${args[@]}" -name "MonthlySwing28" -seed 28 &
python run.py "${args[@]}" -name "MonthlySwing29" -seed 29
python run.py "${args[@]}" -name "MonthlySwing30" -seed 30 &
python run.py "${args[@]}" -name "MonthlySwing31" -seed 31 &
python run.py "${args[@]}" -name "MonthlySwing32" -seed 32 &
python run.py "${args[@]}" -name "MonthlySwing33" -seed 33
python run.py "${args[@]}" -name "MonthlySwing34" -seed 34 &
python run.py "${args[@]}" -name "MonthlySwing35" -seed 35 &
python run.py "${args[@]}" -name "MonthlySwing36" -seed 36 &
python run.py "${args[@]}" -name "MonthlySwing37" -seed 37
python run.py "${args[@]}" -name "MonthlySwing38" -seed 38 &
python run.py "${args[@]}" -name "MonthlySwing39" -seed 39 &
python run.py "${args[@]}" -name "MonthlySwing40" -seed 40 &
python run.py "${args[@]}" -name "MonthlySwing41" -seed 41
python run.py "${args[@]}" -name "MonthlySwing42" -seed 42 &
python run.py "${args[@]}" -name "MonthlySwing43" -seed 43 &
python run.py "${args[@]}" -name "MonthlySwing44" -seed 44 &
python run.py "${args[@]}" -name "MonthlySwing45" -seed 45
python run.py "${args[@]}" -name "MonthlySwing46" -seed 46 &
python run.py "${args[@]}" -name "MonthlySwing47" -seed 47 &
python run.py "${args[@]}" -name "MonthlySwing48" -seed 48 &
python run.py "${args[@]}" -name "MonthlySwing49" -seed 49