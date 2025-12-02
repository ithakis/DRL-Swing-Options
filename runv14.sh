#!/bin/bash
# Trial: stronger target smoothing + calmer critic + more data per update.
args=(
    # 8192 * 4 = 32768 training episodes total (32k)
    # 8192 * 2 = 16384 training episodes (16k)
    -n_paths=32768              # Training episodes (paths)
    -eval_every=1024           # Evaluation frequency (in episodes)
    -n_paths_eval=8192         # Paths per evaluation (for stable pricing estimate)
    -munchausen=0              # Disable Munchausen RL (no entropy bonus in reward)
    -nstep=1
    --per_alpha=0.5            # PER prioritization exponent
    --per_beta_start=0.75      # PER initial importance-sampling bias correction
    --per_beta_frames=150000   # Anneal beta to 1.0 over 150k transitions
    --per_priority_floor=1e-5  # Minimum PER priority
    --per_priority_clip_pct=99.9 # Clip PER priorities to percentile (0 disables)
    --gamma=1                  # No need for discounting since reward includes discounting
    -learn_every=2             # Perform learning update every 2 environment steps
    -learn_number=1            # Gradient updates per learning step (1 update per trigger)
    -iqn=0                     # Disable distributional IQN critic (use standard critic)
    -noise=gauss               # Gaussian exploration noise for continuous actions
    --noise_sigma=1.0          # Scale exploration noise (decays with epsilon)
    --noise_anneal_power=0.5   # Exponent tying noise std to epsilon
    -epsilon=0.3               # Initial epsilon-greedy exploration probability (30% random actions)
    -epsilon_decay=0.9999      # Epsilon decay factor per episode (sustained exploration)
    -per=1                     # Enable Prioritized Experience Replay
    --min_replay_size=15000    # Warm-up buffer size before learning starts (random play)
    --max_replay_size=200000   # Replay buffer capacity (stores up to 200k transitions)
    -t=0.0032                  # Target network soft-update rate tau (moderate smoothing)
    -bs=64                     # Batch size for each gradient update
    -layer_size=64             # Hidden layer size for actor/critic networks
    -lr_a=3e-4                 # Actor learning rate (3e-4, constant)
    -lr_c=1.6e-4               # Critic learning rate (tempered critic)
    --final_lr_fraction=0.7    # Final learning rate as fraction of initial (decay to 70%)
    --warmup_frac=0.02         # Fraction of training for learning-rate warmup (0 => no warmup)
    --min_lr=1e-6              # Minimum learning rate (safeguard)
    --actor_grad_clip=1.0      # Tighter actor gradient clipping for smoother policy updates
    --critic_grad_clip=2.5     # Allow slightly larger critic updates before clipping
    --actor_grad_clip_type=norm
    --critic_grad_clip_type=norm
    --grad_clip_norm_type=2.0
    --weight_decay_actor=5e-5  # Light L2 regularization on the policy network
    --weight_decay_critic=1.2e-4 # Moderate L2 regularization on the value network
    --critic_ema_decay=0.0     # EMA decay for critic eval smoothing (0 disables)
    --compile=0                # Disable torch.compile (for simplicity and compatibility)
    -n_cores=2                 # Number of CPU cores to utilize for parallel processing

    # Swing Option Contract parameters (unchanged from default baseline contract)
    --strike=1.0
    --maturity=0.0833          # ~1 month in years
    --n_rights=22              # 22 exercise opportunities (trading days in a month)
    --q_min=0.0                # Min exercise per decision (no minimum)
    --q_max=2.0                # Max exercise per decision
    --Q_min=0.0                # Global minimum exercise (none)
    --Q_max=20.0               # Global maximum exercise (e.g. 20 units total)
    --risk_free_rate=0.05      # 5% annual risk-free rate
    --min_refraction_periods=0 # Minimum refraction (cooldown) periods after exercise
    --c_cost=0               # Convex cost coefficient c in r_t = exp(-r*dt)[q_t(S_t-K)^+ - c*q_t^{gamma}] {0.2,0.4,0.6,0.8}
    --gamma_cost=1           # Convex cost exponent gamma for the per-unit exercise cost term {1.5,2,3}

    # LSM benchmark controls (defaults preserve legacy power basis / OLS behavior)
    --lsm_basis=chebyshev        # Polynomial family for LSM regression {power,laguerre,hermite,chebyshev}
    --lsm_degree=100           # Highest polynomial degree to include in the LSM basis
    --lsm_reg=none           # Regularization type for LSM regression {none,ridge,lasso}
    --lsm_reg_alpha=1e-6     # Regularization strength (alpha) for ridge/lasso

    # Stochastic process (HHK model) parameters (unchanged from baseline)
    --S0=1.0                   # Initial spot price
    --alpha=12.0               # OU mean-reversion rate
    --sigma=1.2                # OU volatility
    --beta=150.0               # Jump decay rate
    --lam=6.0                  # Jump intensity (6 per year)
    --mu_J=0.3                 # Mean jump size (30% jumps)
)

# python run.py "${args[@]}" -name "SwingOption_20_11" -seed 11 &
# python run.py "${args[@]}" -name "SwingOption_20_12" -seed 12 &
# python run.py "${args[@]}" -name "SwingOption_20_13" -seed 13 &
# python run.py "${args[@]}" -name "SwingOption_20_14" -seed 14

python run.py "${args[@]}" -name "SwingOption_20_v14_11" -seed 11 &
python run.py "${args[@]}" -name "SwingOption_20_v14_12" -seed 12 &
python run.py "${args[@]}" -name "SwingOption_20_v14_13" -seed 13 &
python run.py "${args[@]}" -name "SwingOption_20_v14_14" -seed 14



## To activate the corect environment, run:
# cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options && conda activate EP11
# bash run.sh
