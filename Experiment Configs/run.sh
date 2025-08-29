#!/bin/bash

#  nohup "./run.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt

# Monthly Swing Option Baseline:
args=( 
    # 4096 or 8192 or 16384 or 32768 or 65536
    -n_paths=8192            # Number of Monte Carlo paths (episodes) per run: higher = more stable pricing, slower training
    -eval_every=512         # Frequency (in paths) to run pricing evaluation: lower = more frequent monitoring
    -n_paths_eval=4096       # Number of paths for each evaluation: higher = more accurate price estimate, slower eval
    -munchausen=1            # Munchausen RL: 1=add entropy bonus to reward for better exploration, 0=off
    -nstep=5                 # N-step bootstrapping: how many steps to look ahead for value updates (stability/speed tradeoff)
    --per_alpha=0.5          # PER: prioritization exponent (0=uniform, 1=full prioritization)
    --per_beta_start=0.6     # PER: initial importance-sampling correction (0=off, 1=full correction)
    --per_beta_frames=180000 # PER: frames to anneal beta from start to 1.0 (controls bias correction speed)
    --gamma=0.9998           # Discount factor for future rewards (close to 1 = long-term focus)
    -learn_every=2           # How often to update networks (in steps): lower = more frequent updates
    -learn_number=1          # Number of learning updates per step: higher = more aggressive learning
    -iqn=0                   # Use distributional IQN critic: 1=enable (uncertainty-aware), 0=standard critic
    -noise=ou             # Action noise type: 'gauss' = Gaussian, 'ou' = Ornstein-Uhlenbeck (exploration)
    -epsilon=0.15             # Initial epsilon for exploration noise (probability of random action)
    -epsilon_decay=.99995       # Epsilon decay rate per episode (1.0 = no decay, <1.0 = decaying exploration)
    -per=1                   # Enable Prioritized Experience Replay: 1=on, 0=off
    --min_replay_size=15000  # Minimum buffer size before learning starts (stabilizes early training)
    --max_replay_size=200000 # Maximum replay buffer size (memory/variance tradeoff)
    -t=0.0005               # Soft update factor tau for target networks (lower = more stable, slower adaptation)
    -bs=128                   # Batch size for learning updates (samples per update)
    -layer_size=128          # Number of neurons per hidden layer (network capacity)
    -lr_a=2e-4               # Actor network learning rate (policy update speed)
    -lr_c=1e-4               # Critic network learning rate (value update speed)
    --final_lr_fraction=0.1 # Final LR as fraction of initial (exponential decay: 1.0=no decay, 0.01=decay to 1%)
    --warmup_frac=0.05       # Fraction of episodes for LR warm-up (5% of total episodes)
    --min_lr=5e-5            # Minimum learning rate floor (prevents LR from becoming too small)
    --compile=0              # Use torch.compile for model optimization: 1=on, 0=off (experimental)
    -n_cores=2               # Number of CPU cores to use (parallelism)
    
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
    --min_refraction_periods=0     # No refraction period (can exercise consecutive periods)
    
    # Market Process Parameters (monthly calibration)
    --S0=1.0                  # Initial spot price
    --alpha=12.0                # Higher mean reversion for monthly timeframe
    --sigma=1.2                 # Moderate volatility for monthly period
    --beta=150.0                # Jump decay rate
    --lam=6.0                   # Jump intensity (6 jumps per year average)
    --mu_J=0.3                  # Mean jump size (30%)
)
python run.py "${args[@]}" -name "MonthlySwing32_1" -seed 1 &
python run.py "${args[@]}" -name "MonthlySwing32_2" -seed 2 &
python run.py "${args[@]}" -name "MonthlySwing32_3" -seed 3 &
python run.py "${args[@]}" -name "MonthlySwing32_4" -seed 4

python run.py "${args[@]}" -name "MonthlySwing32_5" -seed 5 &
python run.py "${args[@]}" -name "MonthlySwing32_6" -seed 6 &
python run.py "${args[@]}" -name "MonthlySwing32_7" -seed 7 &
python run.py "${args[@]}" -name "MonthlySwing32_8" -seed 8

python run.py "${args[@]}" -name "MonthlySwing32_9" -seed 9 &
python run.py "${args[@]}" -name "MonthlySwing32_10" -seed 10 &
python run.py "${args[@]}" -name "MonthlySwing32_11" -seed 11 &
python run.py "${args[@]}" -name "MonthlySwing32_12" -seed 12

python run.py "${args[@]}" -name "MonthlySwing32_13" -seed 13 &
python run.py "${args[@]}" -name "MonthlySwing32_14" -seed 14 &
python run.py "${args[@]}" -name "MonthlySwing32_15" -seed 15 &
python run.py "${args[@]}" -name "MonthlySwing32_16" -seed 16


python run.py "${args[@]}" -name "MonthlySwing32_17" -seed 17 &
python run.py "${args[@]}" -name "MonthlySwing32_18" -seed 18 &
python run.py "${args[@]}" -name "MonthlySwing32_19" -seed 19 &
python run.py "${args[@]}" -name "MonthlySwing32_20" -seed 20
