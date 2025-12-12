#!/bin/bash
# v37: Cosine LR schedule over 65k-episode horizon (5% warmup, 5% final LR) to stay aggressive through 32k.
args=(
    # 8192 * 4 = 32768 training episodes total (32k)
    # 8192 * 2 = 16384 training episodes (16k)
    # 256k eval paths = 262144
    # 128k eval paths = 131072
    # 65k eval paths = 65536
    -n_paths=32768
    -eval_every=1024            # Evaluation frequency (in episodes)
    -n_paths_eval=16384         # Paths per evaluation (for stable pricing estimate)
    -munchausen=0               # Disable Munchausen RL (no entropy bonus in reward)
    -nstep=1
    --per_alpha=0.1             # PER extremely soft to mimic uniform early
    --per_beta_start=1.0        # Full IS correction (uniform effect early)
    --per_beta_frames=120000    # Very slow anneal (keeps beta ~1 through mid-run)
    --per_priority_floor=5e-6   # Minimal floor
    --per_priority_clip_pct=99.5   # Clip extreme priorities to curb late spikes
    --per_alpha_final=0.30         # Softer late PER (v35 was 0.32) to lower bias/variance
    --per_alpha_ramp_start=5000    # Start PER alpha ramp after ~5k episodes
    --per_alpha_ramp_end=20000     # Longer ramp (v35: 18k) to keep replay closer to uniform through mid-run
    --per_beta_final=1.0           # Stronger IS correction (v35: 0.95) to neutralize late PER bias
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
    -lr_a=1.9e-4                   # Slightly lower actor LR to smooth mid/late updates
    -lr_c=1.05e-4                  # Slightly lower critic LR; complements higher beta_final
    --final_lr_fraction=0.05       # Cosine decay to 5% of initial LR by the 65k-episode horizon
    --warmup_episodes=1024         # LR warmup hits full rate by episode 1,024
    --lr_schedule_episodes=65000   # LR schedule horizon (extends past 32k training episodes)
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
    -n_cores=2                     # Number of CPU cores to utilize for parallel processing
    --disable_csv_logging=1        # Turn off CSV outputs for this sweep
    --limit_logging_frequency=1    # Throttle per-step TensorBoard logging to shrink files

    # Swing Option Contract parameters (unchanged from default baseline contract)
    --strike=1.0
    --maturity=0.0833              # ~1 month in years
    --n_rights=22                  # 22 exercise opportunities (trading days in a month)
    --q_min=0.0                    # Min exercise per decision (no minimum)
    --q_max=2.0                    # Max exercise per decision
    --Q_min=0.0                    # Global minimum exercise (none)
    --Q_max=20.0                   # Global maximum exercise (e.g. 20 units total)
    --risk_free_rate=0.05          # 5% annual risk-free rate
    --min_refraction_periods=0     # Minimum refraction (cooldown) periods after exercise
    --c_cost=0
    --gamma_cost=1

    # LSM benchmark controls (defaults preserve legacy power basis / OLS behavior)
    --lsm_basis=chebyshev          # Polynomial family for LSM regression {power,laguerre,hermite,chebyshev}
    --lsm_degree=7                 # Highest polynomial degree to include in the LSM basis
    --lsm_reg=none                 # Regularization type for LSM regression {none,ridge,lasso}
    --lsm_reg_alpha=1e-6           # Regularization strength (alpha) for ridge/lasso

    # Stochastic process (HHK model) parameters (unchanged from baseline)
    --S0=1.0                       # Initial spot price
    --alpha=12.0                   # OU mean-reversion rate
    --sigma=1.2                    # OU volatility
    --beta=150.0                   # Jump decay rate
    --lam=6.0                      # Jump intensity (6 per year)
    --mu_J=0.3                     # Mean jump size (30% jumps)
)

python run.py "${args[@]}" -name "SwingOption_20_v37_11" -seed 11 &
python run.py "${args[@]}" -name "SwingOption_20_v37_12" -seed 12 &
python run.py "${args[@]}" -name "SwingOption_20_v37_13" -seed 13 &
python run.py "${args[@]}" -name "SwingOption_20_v37_14" -seed 14

## To activate the correct environment, run:
# cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options && conda activate EP11
# bash runv37.sh

## TensorBoard launch command:
# tensorboard --logdir=runs \                                                                                                               ok | base py
#   --load_fast=true \
#   --samples_per_plugin=scalars=500 \
#   --reload_interval=30 \
#   --max_reload_threads=4
