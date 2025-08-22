#!/bin/bash

# Trial: stronger target smoothing + calmer critic + more data per update.
args=(
    -n_paths=8192
    -eval_every=1024
    -n_paths_eval=4096

    -munchausen=0
    # -nstep=1
    --per_alpha=0.5
    --per_beta_start=0.7
    --per_beta_frames=150000

    --gamma=1 # No need for discounting since reward includes discounting
    -learn_every=2
    -learn_number=1
    -iqn=0
    -noise=gauss
    -epsilon=0.3
    -epsilon_decay=0.9999
    -per=1
    --min_replay_size=10000
    --max_replay_size=200000
    -t=0.003 # Soft update 
    -bs=64
    -layer_size=128
    -lr_a=3e-4
    -lr_c=2e-4
    --final_lr_fraction=1.0
    --warmup_frac=0.0
    --min_lr=1e-6
    --compile=0
    -n_cores=2

    # Contract (unchanged)
    --strike=1.0
    --maturity=0.0833
    --n_rights=22
    --q_min=0.0
    --q_max=2.0
    --Q_min=0.0
    --Q_max=20.0
    --risk_free_rate=0.05
    --min_refraction_days=0

    # HHK process (unchanged)
    --S0=1.0
    --alpha=12.0
    --sigma=1.2
    --beta=150.0
    --lam=6.0
    --mu_J=0.3
)

python run.py "${args[@]}" -name "SwingOption_BaseCase3_nstep2_11" -nstep 2 -seed 11 &
python run.py "${args[@]}" -name "SwingOption_BaseCase3_nstep2_12" -nstep 2 -seed 12 &
python run.py "${args[@]}" -name "SwingOption_BaseCase3_nstep2_13" -nstep 2 -seed 13 &
python run.py "${args[@]}" -name "SwingOption_BaseCase3_nstep2_14" -nstep 2 -seed 14