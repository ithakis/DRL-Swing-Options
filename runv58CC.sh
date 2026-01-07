#!/bin/bash
# v58 (relative to v57CC):
# - Major Improvement: Rprop (Resilient Propagation) for Actor Calibration.
#   - Replaced Newton-based calibration with Rprop to maximize Swing Option Price directly.
#   - Uses sign-based adaptive step sizes to handle noisy gradient estimates from Monte Carlo.
#   - Eliminates instability and "limit cycle" oscillations seen with second-derivative methods.
#   - Integrated directly into Agent initialization (src/agent.py), removing boilerplate.
# - v57 Features retained: Stratified Sampling, Profitability Gate.

args=(
    # Same scale as v57: 32k training episodes
    -n_paths=32768
    -eval_every=1024            # Evaluation frequency
    -n_paths_eval=32768         # Paths per evaluation
    -munchausen=0               # Disable Munchausen RL
    -nstep=1
    --per_alpha=0.1             # PER extremely soft
    --per_beta_start=1.0        # Full IS correction
    --per_beta_frames=120000    # Very slow anneal
    --per_priority_floor=5e-6   # Minimal floor
    --per_priority_clip_pct=99.7
    --per_alpha_final=0.20
    --per_alpha_ramp_start=5000
    --per_alpha_ramp_end=25000
    --per_beta_final=0.98
    --gamma=1
    -learn_every=2
    -learn_number=1
    -iqn=0
    -noise_sigma0=1.30
    -noise_floor=0.26
    -noise_plateau=3200
    -per=1
    --min_replay_size=18000
    --max_replay_size=200000
    -t=0.0032
    -bs=128
    -layer_size=64
    --activation=silu
    --norm=layernorm
    --init_method=orthogonal
    -lr_a=1.6e-4
    -lr_c=9.0e-5
    --final_lr_fraction=0.20
    --warmup_episodes=1024
    --lr_schedule_episodes=40000
    --min_lr=1e-6
    --actor_grad_clip=1.0
    --critic_grad_clip=2.5
    --actor_grad_clip_type=norm
    --critic_grad_clip_type=norm
    --grad_clip_norm_type=2.0
    --weight_decay_actor=5e-5
    --weight_decay_critic=1.2e-4
    --critic_ema_decay=0.0
    --target_policy_noise=0.15
    --target_policy_clip=0.25
    --compile=0
    -n_cores=4
    --disable_csv_logging=1
    --limit_logging_frequency=1

    # Swing Option Contract parameters
    --strike=1.0
    --maturity=0.0833
    --n_rights=22
    --q_min=0.0
    --q_max=2.0
    --Q_min=0.0
    --Q_max=20.0
    --risk_free_rate=0.05
    --min_refraction_periods=0
    --c_cost=0.00
    --gamma_cost=1.0

    # LSM benchmark controls
    --lsm_basis=chebyshev
    --lsm_degree=7
    --lsm_reg=none
    --lsm_reg_alpha=1e-6

    # Stochastic process (HHK model)
    --S0=1.0
    --alpha=12.0
    --sigma=1.2
    --beta=150.0
    --lam=6.0
    --mu_J=0.3
)

# Run multiple seeds for robustness
python run.py "${args[@]}" -name "SwingOption_20_v58_11" -seed 11 &
python run.py "${args[@]}" -name "SwingOption_20_v58_12" -seed 12 &
python run.py "${args[@]}" -name "SwingOption_20_v58_13" -seed 13

# Run Convex Cost variants (CC)
python run.py "${args[@]}" -name "SwingOption_20_v58CC_11" -seed 11 --c_cost 0.15 --gamma_cost 2.0 &
python run.py "${args[@]}" -name "SwingOption_20_v58CC_12" -seed 12 --c_cost 0.15 --gamma_cost 2.0 &
python run.py "${args[@]}" -name "SwingOption_20_v58CC_13" -seed 13 --c_cost 0.15 --gamma_cost 2.0
