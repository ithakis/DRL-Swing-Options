Greetings G
#!/bin/bash
# Trial 3: reinforce exploration, keep wider critic, restore higher learning rate headroom.
args=(
    # 8192 * 4 = 32768 training episodes total (32k)
    # 8192 * 2 = 16384 training episodes (16k)
    -n_paths=32768              # Training episodes (paths)
    -eval_every=1024           # Evaluation frequency (in episodes)
    -n_paths_eval=4096         # Paths per evaluation (for stable pricing estimate)
    -munchausen=0              # Disable Munchausen RL (no entropy bonus in reward)
    -nstep=3                   # 3-step targets to propagate the global volume signal faster
    --per_alpha=0.45           # Slightly stronger priorities but still softer than baseline
    --per_beta_start=0.7       # Initial importance sampling correction
    --per_beta_frames=200000   # Faster beta anneal to avoid over-weighting early spikes
    --gamma=1                  # Rewards already include discounting
    -learn_every=2             # Learn after every other environment step
    -learn_number=2            # Two gradient passes per trigger for better sample efficiency
    -iqn=0                     # Keep scalar critic (IQN remains disabled)
    -noise=gauss               # Return to Gaussian exploration to avoid OU drift to the bounds
    -epsilon=0.25              # Higher initial epsilon to keep replay diverse
    -epsilon_decay=0.9998      # Slower decay so noise remains active longer
    -per=1                     # Enable Prioritized Experience Replay
    --min_replay_size=20000    # Longer random warm-up before learning begins
    --max_replay_size=200000   # Replay buffer capacity (stores up to 200k transitions)
    -t=0.003                   # Target network soft-update rate tau
    -bs=96                    # Batch size for each gradient update
    -layer_size=128            # Legacy hidden layer size for actor unless overridden
    --actor_hidden_size=128    # Actor width
    --critic_hidden_size=256   # Expanded critic for regime-switching complexity
    --actor_layers=2           # Actor depth
    --critic_layers=3          # Extra critic depth stays enabled
    -lr_a=3e-4                 # Actor learning rate (base value before decay)
    -lr_c=1.5e-4               # Critic LR nudged up for more adaptation
    --final_lr_fraction=0.5    # Decay optimizers to 50% (not all the way down)
    --warmup_frac=0.02         # Short warm-up before decay kicks in
    --min_lr=1e-5              # LR floor for both optimizers
    --actor_grad_clip=0.5      # Actor gradient clipping (norm)
    --critic_grad_clip=1.25    # Slightly looser critic clip than run2
    --actor_grad_clip_type=norm
    --critic_grad_clip_type=norm
    --grad_clip_norm_type=2.0
    --weight_decay_actor=1e-4  # Light L2 on actor
    --weight_decay_critic=1.5e-4 # Lightly stronger critic decay to counter extra depth
    --compile=0                # Disable torch.compile
    -n_cores=2                 # Number of CPU cores to utilize

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
    --c_cost=0               # Convex cost coefficient (kept disabled)
    --gamma_cost=1           # Convex cost exponent

    # Stochastic process (HHK model) parameters
    --S0=1.0                   # Initial spot price
    --alpha=12.0               # OU mean-reversion rate
    --sigma=1.2                # OU volatility
    --beta=150.0               # Jump decay rate
    --lam=6.0                  # Jump intensity (6 per year)
    --mu_J=0.3                 # Mean jump size (30% jumps)
    --regime_count=2           # Enable base + spike regimes

    # Base regime HHK overrides (applied when regime_count >= 1)
    --alpha_base=12.0          # Base regime mean reversion
    --sigma_base=1.2           # Base regime OU volatility
    --beta_base=150.0          # Base regime jump decay
    --lam_base=6.0             # Base regime jump intensity
    --mu_J_base=0.3            # Base regime jump size mean

    # Spike regime HHK parameters (only used when regime_count = 2)
    --alpha_spike=18.0         # Spike regime accelerates reversion after shocks
    --sigma_spike=2.0          # Spike regime volatility
    --beta_spike=220.0         # Spike regime jump decay (faster relaxation)
    --lam_spike=9.0            # Spike regime jump intensity
    --mu_J_spike=0.45          # Spike regime jump size mean

    # Regime transition probabilities (Markov chain)
    --p_base_to_spike=0.05     # P(base → spike) per step
    --p_spike_to_base=0.35     # P(spike → base) per step
)

python run.py "${args[@]}" -name "SwingOption_20_RegimeSwitching_wRegLab_32_11_r3" -seed 11 &
python run.py "${args[@]}" -name "SwingOption_20_RegimeSwitching_wRegLab_32_12_r3" -seed 12 &
python run.py "${args[@]}" -name "SwingOption_20_RegimeSwitching_wRegLab_32_13_r3" -seed 13 &
python run.py "${args[@]}" -name "SwingOption_20_RegimeSwitching_wRegLab_32_14_r3" -seed 14

# python run.py "${args[@]}" -name "SwingOption2_32k_15_r3" -seed 15 &
# python run.py "${args[@]}" -name "SwingOption2_32k_16_r3" -seed 16 &
# python run.py "${args[@]}" -name "SwingOption2_32k_17_r3" -seed 17 &
# python run.py "${args[@]}" -name "SwingOption2_32k_18_r3" -seed 18



## To activate the correct environment, run:
# > cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options
# > conda activate EP11
# > bash run3.sh
