#!/bin/bash
# v53 convex cost experiment: c_cost=0.04, gamma_cost=1.5
args=(
    # 8192 * 4 = 32768 training episodes total (32k)
    # 8192 * 2 = 16384 training episodes (16k)
    # 65k eval paths = 65536
    # 128k eval paths = 131072
    # 256k eval paths = 262144
    -n_paths=65536
    -eval_every=1024            # Evaluation frequency (episodes): >0 = periodic (includes initial eval at path 1, plus final if misaligned), -1 = end-only; 0 invalid; no-eval not supported
    -n_paths_eval=262144         # Paths per evaluation (for stable pricing estimate)
    -munchausen=0               # Disable Munchausen RL (no entropy bonus in reward)
    -nstep=1
    --per_alpha=0.1             # PER extremely soft to mimic uniform early
    --per_beta_start=1.0        # Full IS correction (uniform effect early)
    --per_beta_frames=120000    # Very slow anneal (keeps beta ~1 through mid-run)
    --per_priority_floor=5e-6   # Minimal floor
    --per_priority_clip_pct=99.7   # Clip extreme priorities to curb spikes while allowing spread (active)
    --per_alpha_final=0.20         # Softer late PER to reduce bias/variance
    --per_alpha_ramp_start=5000    # Start PER alpha ramp after ~5k episodes
    --per_alpha_ramp_end=25000     # Longer ramp to keep replay closer to uniform through mid-run
    --per_beta_final=0.98          # Mild IS correction to retain some prioritization effect late
    --gamma=1                      # No need for discounting since reward includes discounting
    -learn_every=2                 # Perform learning update every 2 environment steps
    -learn_number=1                # Gradient updates per learning step (1 update per trigger)
    -iqn=0                         # Disable distributional IQN critic (use standard critic)
    -noise_sigma0=1.30             # Initial pre-squash noise std
    -noise_floor=0.26              # Slightly higher floor (v35: 0.24) to avoid mid/late under-exploration
    -noise_plateau=3200            # Episodes to hold initial pre-squash noise before decay
    -per=1                         # Enable soft PER
    --min_replay_size=18000        # Slightly smaller warm-up since uniform replay
    --max_replay_size=200000       # Replay buffer capacity (stores up to 200k transitions)
    -t=0.0032                      # Target network soft-update rate tau (moderate smoothing)
    -bs=128                        # Batch size for each gradient update
    -layer_size=64                 # Hidden layer size for actor/critic networks
    --activation=silu              # Use SiLU activations for hidden layers
    --norm=layernorm               # LayerNorm (revert from RMSNorm)
    --init_method=orthogonal       # Match v43 init (orthogonal + activation gain)
    -lr_a=1.6e-4                   # Lowered peak actor LR (was 2.0e-4 in v40)
    -lr_c=9.0e-5                   # Lowered peak critic LR (was 1.1e-4 in v40)
    --final_lr_fraction=0.20       # Cosine decay to 20% of initial LR by the 40k-episode horizon
    --warmup_episodes=1024         # LR warmup hits full rate by episode 1,024
    --lr_schedule_episodes=40000   # LR schedule horizon (faster decay through 32k)
    --min_lr=1e-6                  # Minimum learning rate (safeguard)
    --actor_grad_clip=1.0          # Tighter actor gradient clipping for smoother policy updates
    --critic_grad_clip=2.5         # Allow slightly larger critic updates before clipping
    --actor_grad_clip_type=norm
    --critic_grad_clip_type=norm
    --grad_clip_norm_type=2.0
    --weight_decay_actor=5e-5      # Light L2 regularization on the policy network
    --weight_decay_critic=1.2e-4   # Moderate L2 regularization on the value network
    --critic_ema_decay=0.0         # EMA decay for critic eval smoothing (0 disables)
    --target_policy_noise=0.15     # Stronger target policy smoothing to temper critic overconfidence
    --target_policy_clip=0.25      # Target policy smoothing noise clip
    --compile=0                    # Disable torch.compile (for simplicity and compatibility)
    -n_cores=4                     # Number of CPU cores to utilize for parallel processing
    --disable_csv_logging=0        # Turn off CSV outputs for this sweep
    --limit_logging_frequency=1    # Throttle per-step TensorBoard logging to shrink files

    # Swing Option Contract parameters (pricing problem definition)
    --strike=1.0                 # Strike price K
    --maturity=0.0833            # Time to maturity in years (~1 month)
    --n_rights=22                # Number of decision dates (exercise opportunities)
    --q_min=0.0                  # Min exercise per decision date
    --q_max=2.0                  # Max exercise per decision date
    --Q_min=0.0                  # Global min total volume over the contract
    --Q_max=20.0                 # Global max total volume over the contract
    --risk_free_rate=0.05        # Annual risk-free rate used for discounting
    --min_refraction_periods=0   # Cooldown periods after an exercise (0 = none)
    --c_cost=0.04                   # Convex exercise cost coefficient (0 disables cost)
    --gamma_cost=1.5               # Convex cost exponent (1 = linear cost in q)

    # LSM benchmark controls (continuation value regression)
    --lsm_basis=chebyshev        # Basis family {power, laguerre, hermite, chebyshev}
    --lsm_degree=7               # Polynomial degree (higher = more flexible regression)
    --lsm_reg=none               # Regularization {none, ridge, lasso}
    --lsm_reg_alpha=1e-6         # Regularization strength (only used if ridge/lasso)

    # Stochastic process (HHK model) parameters (spot dynamics)
    --S0=1.0                     # Initial spot price
    --alpha=12.0                 # Mean-reversion speed (OU)
    --sigma=1.2                  # Diffusion volatility (OU)
    --beta=150.0                 # Jump decay rate (faster decay = shorter jump impact)
    --lam=6.0                    # Jump intensity (expected jumps per year)
    --mu_J=0.3                   # Mean jump size (relative jump magnitude)
)

python run.py "${args[@]}" -name "SwingOption_20_c0.04_gamma1.5_11" -seed 11
python run.py "${args[@]}" -name "SwingOption_20_c0.04_gamma1.5_12" -seed 12
python run.py "${args[@]}" -name "SwingOption_20_c0.04_gamma1.5_13" -seed 13
python run.py "${args[@]}" -name "SwingOption_20_c0.04_gamma1.5_14" -seed 14
python run.py "${args[@]}" -name "SwingOption_20_c0.04_gamma1.5_15" -seed 15

## To activate the correct environment, run:
# cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options && conda activate EP11
# bash SwingOption_20_c0.04_gamma1.5.sh

## TensorBoard launch command:
# tensorboard --logdir=runs \
#   --load_fast=true \
#   --samples_per_plugin=scalars=500 \
#   --reload_interval=30 \
#   --max_reload_threads=4
