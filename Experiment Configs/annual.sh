#!/bin/bash

#  nohup "./run.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt

# Monthly Swing Option Baseline:
args=( 
    # Training parameters - IMPROVED PATH GENERATION
    -n_paths=32768              # 32k training episodes for better convergence
    -eval_every=1024
    -n_paths_eval=4096          # Generates exactly 4096 evaluation paths (shared between RL and LSM)
    -munchausen=0               # Keep Munchausen off by default
    -nstep=1                    # Single-step TD targets
    --per_alpha=0.5
    --per_beta_start=0.7
    --per_beta_frames=150000
    --per_priority_floor=1e-6  # Minimum PER priority
    --per_priority_clip_pct=99.5 # Clip PER priorities to percentile (0 disables)
    --gamma=1
    -learn_every=2
    -learn_number=1
    -per=1
    -iqn=0
    -noise=gauss                # Use Gaussian noise instead of OU noise
    --noise_sigma=1.0          # Scale exploration noise (decays with epsilon)
    --noise_anneal_power=1.0   # Exponent tying noise std to epsilon
    -epsilon=0.3
    -epsilon_decay=0.99994
    -bs=64
    -layer_size=64              # 2×64 actor/critic
    --min_replay_size=5000      # Increase from 1000 to 5000
    --max_replay_size=200000
    -t=0.002
    --tau_final=0.002          # Final tau for target schedule (<0 disables)
    --tau_schedule_frac=0.5    # Fraction of training to decay tau toward tau_final
    -lr_a=2e-4                  # from lr_a=3e-4
    -lr_c=2e-4                  # from lr_c=3e-4
    --actor_grad_clip=0
    --critic_grad_clip=0
    --actor_grad_clip_type=none
    --critic_grad_clip_type=none
    --grad_clip_norm_type=2.0
    --weight_decay_actor=5e-5
    --weight_decay_critic=1e-4
    --critic_ema_decay=0.0     # EMA decay for critic eval smoothing (0 disables)
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
