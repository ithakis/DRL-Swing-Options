# Copilot Instructions for DRL-Swing-Options

- Purpose: price swing options via D4PG with HHK price paths; benchmarks include LSM and optional FDM.
- Core layout: training orchestration in run.py; evaluation helper in evaluate_agent.py; RL components in src/ (Agent, networks, replay buffer, env + contract, HHK simulator, benchmarks).
- Contracts & env: SwingContract enforces local/global/refraction constraints and discounting; SwingOptionEnv consumes pre-generated HHK paths, maps actions in [0,1] → quantities, masks exercises when net payoff ≤ 0, tracks refraction and global caps.
- HHK data: simulate_hhk_spot() builds Sobol + antithetic OU-with-jumps paths; env episodes index directly into the pre-simulated path array (episode id = path row).
- Agent setup: Agent defaults to 2×64 SiLU MLPs, AdamW with decoupled weight decay, PER on by default, Munchausen/IQN optional; supports per-network widths/depths, noise plateau→hyperbolic decay, LR warmup+cosine schedules, and optional gradient clipping (off unless thresholds >0).
- Networks: Actor uses LayerNorm blocks, orthogonal init, tanh01 head (0–1 actions); Critic merges state+action after first layer; IQN critic available for distributional returns.
- Replay: CircularReplayBuffer is the default; PrioritizedReplay enables Fenwick tree sampling with optional numba acceleration; n-step buffers flush transitions before storage.
- Training flow (run.py): parse CLI, seed, generate HHK paths, build contract/env/agent, optional torch.compile, optional warmup_calibrate_actor_outputs to match target action mean/std, then main loop with PER-aware replay, periodic evaluations, and CSV + TensorBoard logging.
- Logging/output: logs/<run>/ holds training/eval CSVs plus per-episode raw evals; runs/<name>.json stores hyperparameters and runs/<name>.pth stores actor weights; disable CSV via --disable_csv_logging=1; limit TB frequency via --limit_logging_frequency=1.
- Benchmarks: price_swing_option_lsm implements Longstaff-Schwartz with selectable basis (power/laguerre/hermite/chebyshev) and optional ridge/lasso; price_swing_option_fdm hook exists for PDE comparison (requires QuantLib if used).
- Evaluation: evaluate_agent.py loads runs/<name>.json + optional .pth, builds gym env, and reports pricing stats with confidence intervals; supports batched eval and cProfile via --profile_eval.
- Key flags to remember: --optimizer {adam,adamw}, --actor_hidden_size/--critic_hidden_size and --actor_layers/--critic_layers, --per_alpha/beta scheduling knobs, noise_sigma0/floor/plateau, --actor_grad_clip/--critic_grad_clip, --final_lr_fraction + warmup/schedule horizons, --c_cost/--gamma_cost for convex exercise costs.
- System knobs: -n_cores sets OMP + MKL threads and torch interop; --fp32 chooses float32 (default) vs float64; --compile toggles torch.compile; --device gpu only works if CUDA is available, otherwise falls back to CPU.
- Testing: python run_tests.py exercises replay buffer/PER integration; expand as you add features.
- Typical runs: quick experiment via python run.py -name Test ...; production sweep via bash run.sh; evaluate a saved run via python evaluate_agent.py --run_name <name> --runs 100 --eval_batch_size 64.
- Conventions: keep actions continuous in [0,1]; contract caps enforced inside env; refraction implemented as min_refraction_periods cooldown; reward function includes convex exercise cost and per-step discounting; actor/critic init prints parameter counts for traceability.
- Dependencies: prefer conda-forge stack with Python 3.11 + PyTorch CPU as in README; avoid mixing Brew libomp/pip wheels to prevent OpenMP clashes.

Please propose additions if any critical workflows, flags, or edge cases are unclear or missing.
