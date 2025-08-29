#!/bin/bash
# Trial: stronger target smoothing + calmer critic + more data per update.
args=(
    # 8192 * 4 = 32768 training episodes total (32k)
    # 8192 * 2 = 16384 training episodes (16k)
    -n_paths=16384              # Training episodes (paths)
    -eval_every=1024           # Evaluation frequency (in episodes)
    -n_paths_eval=4096         # Paths per evaluation (for stable pricing estimate)
    -munchausen=0              # Disable Munchausen RL (no entropy bonus in reward)
    -nstep=1
    --per_alpha=0.5            # PER prioritization exponent
    --per_beta_start=0.7       # PER initial importance-sampling bias correction
    --per_beta_frames=150000   # Anneal beta to 1.0 over 150k transitions
    --gamma=1                  # No need for discounting since reward includes discounting
    -learn_every=2             # Perform learning update every 2 environment steps
    -learn_number=1            # Gradient updates per learning step (1 update per trigger)
    -iqn=0                     # Disable distributional IQN critic (use standard critic)
    -noise=gauss               # Gaussian exploration noise for continuous actions
    -epsilon=0.3               # Initial epsilon-greedy exploration probability (30% random actions)
    -epsilon_decay=0.9999      # Epsilon decay factor per episode (slowly decrease random action rate)
    -per=1                     # Enable Prioritized Experience Replay
    --min_replay_size=10000    # Warm-up buffer size before learning starts (random play)
    --max_replay_size=200000   # Replay buffer capacity (stores up to 200k transitions)
    -t=0.003                   # Target network soft-update rate tau (stronger smoothing)
    -bs=64                     # Batch size for each gradient update
    -layer_size=128            # Hidden layer size for actor/critic networks
    -lr_a=3e-4                 # Actor learning rate (3e-4, constant)
    -lr_c=2e-4                 # Critic learning rate (2e-4, calmer critic)
    --final_lr_fraction=1.0    # Final learning rate as fraction of initial (1.0 => no decay)
    --warmup_frac=0.0          # Fraction of training for learning-rate warmup (0 => no warmup)
    --min_lr=1e-6              # Minimum learning rate (not used since no decay, just a safeguard)
    --compile=0                # Disable torch.compile (for simplicity and compatibility)
    -n_cores=2                 # Number of CPU cores to utilize for parallel processing

    # Swing Option Contract parameters (unchanged from default baseline contract)
    --strike=1.0
    --maturity=0.0833          # ~1 month in years
    --n_rights=22              # 22 exercise opportunities (trading days in a month)
    --q_min=0.0                # Min exercise per decision (no minimum)
    --q_max=2.0                # Max exercise per decision
    --Q_min=0.0                # Global minimum exercise (none)
    --Q_max=44.0               # Global maximum exercise (e.g. 44 units total)
    --risk_free_rate=0.05      # 5% annual risk-free rate
    --min_refraction_days=0    # No refraction period (can exercise in consecutive days)

    # Stochastic process (HHK model) parameters (unchanged from baseline)
    --S0=1.0                   # Initial spot price
    --alpha=12.0               # OU mean-reversion rate
    --sigma=1.2                # OU volatility
    --beta=150.0               # Jump decay rate
    --lam=6.0                  # Jump intensity (6 per year)
    --mu_J=0.3                 # Mean jump size (30% jumps)
)

python run.py "${args[@]}" -name "SwingOption_44_16k_11" -seed 11 &
python run.py "${args[@]}" -name "SwingOption_44_16k_12" -seed 12 &
python run.py "${args[@]}" -name "SwingOption_44_16k_13" -seed 13 &
python run.py "${args[@]}" -name "SwingOption_44_16k_14" -seed 14

# python run.py "${args[@]}" -name "SwingOption2_32k_15" -seed 15 &
# python run.py "${args[@]}" -name "SwingOption2_32k_16" -seed 16 &
# python run.py "${args[@]}" -name "SwingOption2_32k_17" -seed 17 &
# python run.py "${args[@]}" -name "SwingOption2_32k_18" -seed 18