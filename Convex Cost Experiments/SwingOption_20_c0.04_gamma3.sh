#!/bin/bash
# Derived from run.sh for convex cost experiment: c_cost=0.04, gamma_cost=3.0
args=(
    -n_paths=32768
    -eval_every=1024
    -n_paths_eval=4096
    -munchausen=0
    -nstep=1
    --per_alpha=0.5
    --per_beta_start=0.7
    --per_beta_frames=150000
    --per_priority_floor=1e-6  # Minimum PER priority
    --per_priority_clip_pct=99.5 # Clip PER priorities to percentile (0 disables)
    --gamma=1
    -learn_every=2
    -learn_number=1
    -iqn=0
    -noise_sigma0=1.0          # Scale exploration noise (decays with epsilon)
    -per=1
    --min_replay_size=10000
    --max_replay_size=200000
    -t=0.002
    -bs=64
    -layer_size=64
    -lr_a=3e-4
    -lr_c=2e-4
    --final_lr_fraction=1.0
    --warmup_frac=0.0
    --min_lr=1e-6
    --actor_grad_clip=0
    --critic_grad_clip=0
    --actor_grad_clip_type=none
    --critic_grad_clip_type=none
    --grad_clip_norm_type=2.0
    --weight_decay_actor=5e-5
    --weight_decay_critic=1e-4
    --critic_ema_decay=0.0     # EMA decay for critic eval smoothing (0 disables)
    --compile=0
    -n_cores=2
    --strike=1.0
    --maturity=0.0833
    --n_rights=22
    --q_min=0.0
    --q_max=2.0
    --Q_min=0.0
    --Q_max=20.0
    --risk_free_rate=0.05
    --min_refraction_periods=0
    --c_cost=0.04
    --gamma_cost=3.0
    --S0=1.0
    --alpha=12.0
    --sigma=1.2
    --beta=150.0
    --lam=6.0
    --mu_J=0.3
)

python run.py "${args[@]}" -name "SwingOption_20_c0.04_gamma3_11" -seed 11 &
python run.py "${args[@]}" -name "SwingOption_20_c0.04_gamma3_12" -seed 12 &
python run.py "${args[@]}" -name "SwingOption_20_c0.04_gamma3_13" -seed 13 &
python run.py "${args[@]}" -name "SwingOption_20_c0.04_gamma3_14" -seed 14
