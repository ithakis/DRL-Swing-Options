#!/bin/bash
# v26: Start near-uniform (PER extremely soft) for ~5–7k, then gently allow PER to matter; keep late-exploitation tweaks and moderate action reg.
args=(
    # 8192 * 4 = 32768 training episodes total (32k)
    # 8192 * 2 = 16384 training episodes (16k)
    -n_paths=32768              # Training episodes (paths)
    -eval_every=1024           # Evaluation frequency (in episodes)
    -n_paths_eval=8192         # Paths per evaluation (for stable pricing estimate)
    -munchausen=0              # Disable Munchausen RL (no entropy bonus in reward)
    -nstep=1
    --per_alpha=0.1            # PER extremely soft to mimic uniform early
    --per_beta_start=1.0       # Full IS correction (uniform effect early)
    --per_beta_frames=120000   # Very slow anneal (keeps beta ~1 through mid-run)
    --per_priority_floor=5e-6  # Minimal floor
    --per_priority_clip_pct=0  # No clipping
    --per_alpha_final=0.4      # Target PER alpha after ramp
    --per_alpha_ramp_start=5000  # Start PER alpha ramp after ~5k episodes
    --per_alpha_ramp_end=15000   # End PER alpha ramp by ~15k episodes
    --per_beta_final=0.8       # Target PER beta after ramp (keeps IS correction moderate)
    --gamma=1                  # No need for discounting since reward includes discounting
    -learn_every=2             # Perform learning update every 2 environment steps
    -learn_number=1            # Gradient updates per learning step (1 update per trigger)
    -iqn=0                     # Disable distributional IQN critic (use standard critic)
    -noise=gauss               # Gaussian exploration noise for continuous actions
    --noise_sigma=1.3          # Strong initial noise scale for early exploration
    --noise_anneal_power=0.55  # Gentle annealing tied to epsilon
    --noise_plateau=3200       # Hold initial noise/epsilon longer to avoid early decay
    --min_action_noise=0.18    # Lower floor for more late exploitation while avoiding collapse
    -epsilon=0.3               # Initial epsilon-greedy exploration probability (30% random actions)
    -epsilon_decay=0.999965    # Slightly faster decay to unlock exploitation later
    -per=1                     # Enable soft PER
    --min_replay_size=18000    # Slightly smaller warm-up since uniform replay
    --max_replay_size=200000   # Replay buffer capacity (stores up to 200k transitions)
    -t=0.0032                  # Target network soft-update rate tau (moderate smoothing)
    -bs=128                    # Batch size for each gradient update
    -layer_size=64             # Hidden layer size for actor/critic networks
    -lr_a=2.2e-4               # Actor learning rate scaled down for larger batch
    -lr_c=1.2e-4               # Critic learning rate scaled down for larger batch
    --final_lr_fraction=0.95   # Keep more learning rate for late-stage adjustment
    --warmup_frac=0.03         # Slightly longer warmup to smooth early updates
    --min_lr=1e-6              # Minimum learning rate (safeguard)
    --actor_grad_clip=1.0      # Tighter actor gradient clipping for smoother policy updates
    --critic_grad_clip=2.5     # Allow slightly larger critic updates before clipping
    --actor_grad_clip_type=norm
    --critic_grad_clip_type=norm
    --grad_clip_norm_type=2.0
    --action_reg_weight=4e-3  # Moderate action L2 early
    --action_reg_cutoff=4000   # Disable action L2 after 4k episodes
    --weight_decay_actor=5e-5  # Light L2 regularization on the policy network
    --weight_decay_critic=1.2e-4 # Moderate L2 regularization on the value network
    --critic_ema_decay=0.0     # EMA decay for critic eval smoothing (0 disables)
    --target_policy_noise=0.15 # Stronger target policy smoothing to temper critic overconfidence
    --target_policy_clip=0.25  # Target policy smoothing noise clip
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
    --c_cost=0.05               # Convex cost coefficient c in r_t = exp(-r*dt)[q_t(S_t-K)^+ - c*q_t^{gamma}] {0.2,0.4,0.6,0.8}
    --gamma_cost=1.5           # Convex cost exponent gamma for the per-unit exercise cost term {1.5,2,3}

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

python run.py "${args[@]}" -name "SwingOption_20_v26_g1.5_11" -seed 11 &
python run.py "${args[@]}" -name "SwingOption_20_v26_g1.5_12" -seed 12 &
python run.py "${args[@]}" -name "SwingOption_20_v26_g1.5_13" -seed 13 &
python run.py "${args[@]}" -name "SwingOption_20_v26_g1.5_14" -seed 14



## To activate the corect environment, run:
# cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options && conda activate EP11
# bash run.sh

# tensorboard --logdir=runs \                                                                           
#   --load_fast=true \
#   --samples_per_plugin=scalars=200 \
#   --reload_interval=30 \
#   --max_reload_threads=4
