#!/bin/bash

#  nohup "./run.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt


# Monthly Swing Option Baseline:
args=(
  # RL / training settings (you may shorten n_paths once tuning converges)
  -n_paths=8192 -eval_every=-1 -n_paths_eval=2048 -munchausen=0 -nstep=5
  -learn_every=2 -per=1 -iqn=0 -noise=gauss -bs=64 -layer_size=128
  --min_replay_size=5000 -t=5e-3 -lr_a=2e-4 -lr_c=2e-4 --compile=0
  --use_circular_buffer=1 -n_cores=2

  # swing specifics
  --strike=1.0          # K
  --maturity=1.0        # one calendar year
  --n_rights=365
  --q_min=0.0           # at most one right per day → q_max = 1
  --q_max=1.0
  --Q_min=0.0
  --Q_max=35
  --risk_free_rate=0.0
  --min_refraction_days=1   # one-day spacing between rights

  # spot model
  --S0=1.0
  --alpha=7.0
  --sigma=1.4
  --beta=200.0
  --lam=4.0
  --mu_J=0.4
)
python run.py "${args[@]}" -name "Yearly_Swing_NRights35" -seed 1
# python run.py "${args[@]}" -name "MonthlySwing_Baseline2" -seed 12 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline3" -seed 13 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline4" -seed 14 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline5" -seed 15 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline6" -seed 16 &
# python run.py "${args[@]}" -name "MonthlySwing_Baseline7" -seed 17 
# python run.py "${args[@]}" -name "MonthlySwing_Baseline8" -seed 18
# python run.py "${args[@]}" -name "MonthlySwing_Baseline9" -seed 19

