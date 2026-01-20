#!/bin/bash
# v61 (relative to v60):
# 
# ANALYSIS SUMMARY (v60 vs v59):
# - v60 achieved better late-stage stability (std 1.28% vs 2.86% at ep 32768)
# - But v60 was slower in early training (first 80% at ep 4096-6144 vs ep 3072)
# - v60 had higher early seed variance at ep 4096 (spread 36.3% vs 13.4%)
# 
# v61 CHANGES (addressing v60 limitations while preserving gains):
#
# - Fix #1: Gradual Warmup Noise Ramp (NEW CODE)
#   - Instead of flat 0.2× noise during warmup, noise now ramps linearly
#   - At ep 1: noise = 0.3× (start), at ep 1024: noise = 1.0× (full)
#   - Provides smooth transition from calibration preservation to full exploration
#   - Addresses: early seed variance at ep 4096 by allowing gradual policy adaptation
#
# - Fix #2: Reduced Critic Warmup (1024 episodes, was 2048)
#   - Critic likely stabilizes before 2048 episodes; shorter warmup speeds convergence
#   - Actor starts learning 1024 episodes earlier
#   - Addresses: slow early convergence (targeting first 80% by ep 3072-4096)
#
# - Fix #3: Higher β-Sigmoid Temperature (β=3.0, was β=2.0)
#   - β=2.0 was nearly identical to tanh01 in practice
#   - β=3.0 creates more differentiation: saturates at |u|≈1.7 vs tanh01's |u|≈2.5
#   - Provides softer gradients in boundary regions
#   - Addresses: potential gradient saturation issues
#
# - Fix #4: Later Target Noise Decay Start (ep 18000, was ep 15000)
#   - Preserve exploration benefits longer before decay
#   - Decay still completes by ~30k episodes (before end of training)
#   - Addresses: fine-tuning of v60's successful target noise decay
#
# - Fix #5: Higher Target Noise Floor (0.04, was 0.02)
#   - Less aggressive decay maintains some smoothing effect late
#   - Prevents over-fitting to noisy Q-estimates
#
# - Fix #6: Increased Adaptive Noise Scale (0.6, was 0.5)
#   - Compensates for potential under-exploration from warmup changes
#   - Ensures policy can escape saturation regions
#
# - Fix #7: Increased Warmup Noise Fraction (0.3, was 0.2)
#   - Combined with gradual ramp: starts at 0.3× and ramps to 1.0× over warmup
#   - Higher starting point reduces calibration-to-learning transition shock
#
# EXPECTED RESULTS:
#   a) Early convergence: first 80% by ep 3072-4096 (matching v59)
#   b) Early seed spread at ep 4096: <15% (vs 36% in v60, 13% in v59)
#   c) Late-stage std at ep 32768: <2% (preserving v60's improvement)
#   d) Final Delta%: >112% mean
#
# v60 Features retained: Target policy noise decay (with adjusted schedule)
# v59 Features retained: Critic warmup (shorter), Adaptive pre-squash noise
# v58 Features retained: Rprop Calibration
# v57 Features retained: Stratified Sampling, Profitability Gate

args=(
    # Same scale as v60: 32k training episodes
    -n_paths=32768
    -eval_every=1024            # Evaluation frequency (episodes): >0 = periodic (includes initial eval at path 1, plus final if misaligned), -1 = end-only; 0 invalid; no-eval not supported
    -n_paths_eval=32768         # Paths per evaluation (for stable pricing estimate)
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
    --target_policy_noise=0.15     # Initial target policy smoothing (will decay via v61 schedule)
    --target_policy_clip=0.25      # Target policy smoothing noise clip
    --compile=0                    # Disable torch.compile (for simplicity and compatibility)
    -n_cores=4                     # Number of CPU cores to utilize for parallel processing
    --disable_csv_logging=1        # Turn off CSV outputs for this sweep
    --limit_logging_frequency=1    # Throttle per-step TensorBoard logging to shrink files

    # v61 Parameters (Improved from v60 based on analysis)
    --critic_warmup_episodes=1024  # v61 Fix #2: Reduced from 2048 to speed early convergence
    --adaptive_noise_scale=0.6     # v61 Fix #6: Increased from 0.5 for better exploration
    --warmup_noise_fraction=0.3    # v61 Fix #7: Increased from 0.2 (now with gradual ramp in code)
    --target_noise_decay_start=18000  # v61 Fix #4: Later start (was 15000) to preserve exploration longer
    --target_noise_floor=0.04      # v61 Fix #5: Higher floor (was 0.02) for residual smoothing
    --actor_output_activation=beta_sigmoid_3.0  # v61 Fix #3: Higher β (was 2.0) for more differentiation

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
    --c_cost=0.00                # Convex exercise cost coefficient
    --gamma_cost=1.0             # Convex cost exponent

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

# Run multiple seeds for robustness (no-cost regime)
python run.py "${args[@]}" -name "SwingOption_20_v61_1_11" -seed 11 & pids=($!)
python run.py "${args[@]}" -name "SwingOption_20_v61_1_12" -seed 12 & pids+=($!)
python run.py "${args[@]}" -name "SwingOption_20_v61_1_13" -seed 13 & pids+=($!)
for p in "${pids[@]}"; do wait "$p" || exit 1; done

## To activate the correct environment, run:
# cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options && conda activate EP11
# bash runv61CC.sh
