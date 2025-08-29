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
    -n_cores=4
    
    ################################################################################
    # Anual Swing Option Contract Parameters
    --strike=1.0              # K = 1   (paper works in relative price units)
    --maturity=1.0            # 1-year delivery period
    --n_rights=365            # daily decision opportunities (matches notebook n_steps)
    --q_min=0.0               # bang-bang: 0 or 1 each day
    --q_max=1.0
    --Q_min=0.0
    --Q_max=100.0             # up to 100 call rights
    --risk_free_rate=0.0      # r = 0 in the experiment
    --min_refraction_periods=0
    # ── HHK spike-model parameters (Fig. 1 & 10, matches notebook) ───
    --S0=1.0                  # initial spot (matches strike scale)
    --alpha=7.0               # OU mean-reversion speed
    --sigma=1.4               # OU volatility
    --beta=200.0              # jump decay rate
    --lam=4.0                 # Poisson-jump intensity (4 spikes / yr)
    --mu_J=0.4                # mean jump size
)
python run.py "${args[@]}" -name "Fig10_100_16k" -seed 1
# python run.py "${args[@]}" -name "del" -seed 1
# python run.py "${args[@]}" -name "MonthlySwing_Baseline2" -seed 12 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline3" -seed 13 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline4" -seed 14 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline5" -seed 15 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline6" -seed 16 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline7" -seed 17 
# python run.py "${args[@]}" -name "MonthlySwing_Baseline8" -seed 18
# python run.py "${args[@]}" -name "MonthlySwing_Baseline9" -seed 19
