# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research implementation of **D4PG** (Distributional Deep Deterministic Policy Gradient) for pricing **swing options with convex exercise costs** in energy markets, benchmarked against **Least-Squares Monte Carlo (LSM)**. The spot price follows the **Hambly-Howison-Kluge (HHK)** two-factor OU-with-jumps model. This accompanies an academic paper under review.

## Commands

```bash
conda env create -f environment.yml   # Create the EP11 conda environment
conda activate EP11                    # Activate it

make sweep                             # Run the convex-cost experiment sweep (conv_cost_exps.sh)
make single-exp EXP_SCRIPT="Convex Cost Experiments/SwingOption_20_c0.05_gamma2.sh"
make train ARGS='-name "Test" -seed 42 -n_paths 8192 --c_cost 0.05 --gamma_cost 2'
make eval RUN_NAME=<name> RUNS=100     # Evaluate a saved agent
make compare-lsm-state-modes          # Compare reduced-state vs full-state LSM
make tensorboard                       # Launch TensorBoard on runs/
make paper                             # Build manuscript to Paper/build/
make clean-paper                       # Remove manuscript build artifacts
```

## Current Paper Configuration

The paper results use **v61 hyperparameters** with IQN disabled (`-iqn=0`) and a standard (non-distributional) critic. The experiment configs in `Convex Cost Experiments/` are the canonical scripts used to generate paper results. Key choices:
- 32768 training episodes, 65536 evaluation paths, seeds {11, 12, 13} per configuration
- Focal robustness study: c=0.04, gamma=2, seeds 11-25 (15 seeds)
- `--gamma=1` in D4PG because the environment reward already includes discounting
- Profitability gate enabled (STE gradients), actor output via beta-sigmoid(3.0)
- PER with soft alpha ramp 0.1->0.2, beta ~1.0
- 2x64 SiLU MLPs with LayerNorm, orthogonal init
- Convex cost sweep: c in {0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15} x gamma in {1, 1.5, 2, 3}

## Semi-Analytical Kernel (feat/semi-analytical-bootstrap)

Optional opt-in mechanism that replaces the critic's single-sample TD bootstrap with an analytical expectation over the HHK transition kernel.  **Bit-identical to v61 when off** (default).  Recommended for kernel-on training:

```bash
--use_expected_target=1
--kernel_M_x=3 --kernel_M_per_k=3 --kernel_N_max=2   # M=21, sweet spot
--critic_warmup_episodes=0                            # not needed with kernel
```

Headline result (focal c=0.04, gamma=2, 4096 ep, 3 seeds):
- Kernel on  (`H1_only`):   Delta% = +0.474 +/- 0.204 pp
- Kernel off (`B0_baseline`):Delta% = -2.051 +/- 0.440 pp
- Gap = +2.525 pp, t = +9.02, p = 0.004 (Welch's two-sample)

At 8192 ep the kernel-on seed std collapses to 0.022 pp (10x reduction).  No-cost regression: kernel +5.2 pp better than baseline.  Wall-clock cost: ~2x baseline per episode at M=36, ~1.5x at M=21.

See `Jupyter Notebooks/7: Phase 1 Findings - Semi-Analytical Kernel.ipynb` for a self-contained tour with statistics and parameter-selection recipe.  Other hypotheses tested (H4 warm-start, H5 Dyna, H6 IQN, H7 twin critics, H8 antithetic, H9 jump-IW) either did not help or actively hurt — see notebook for the full negative-results catalog.

## Architecture

### Data Flow

`run.py` is the single entry point for training. It parses CLI args, constructs a `SwingContract`, simulates HHK paths via `simulate_hhk_spot()`, builds a `SwingOptionEnv` (Gymnasium), creates an `Agent`, and runs the training loop. During training, an LSM baseline is also computed for comparison. Evaluation uses pre-generated out-of-sample paths.

### Source Files (`src/`)

| File | Role |
|------|------|
| `agent.py` | D4PG agent: actor-critic updates, PER scheduling, noise schedules (pre-squash Gaussian with plateau+hyperbolic decay), critic warmup, target network soft updates |
| `swing_env.py` | Gymnasium environment: maps agent actions to exercise quantities, enforces contract constraints (q_min/q_max, Q_min/Q_max, refraction), computes discounted rewards with convex costs. Also contains `approximate_Q_T()` for HHK-based expected quantity estimation |
| `networks.py` | Actor (profitability-gated with STE), Critic (standard TD), IQN (distributional, **not used in current paper**). Actor gates unprofitable exercises: `q_out = q_raw * 1[Pi(q) > 0]` |
| `lsm_swing_pricer.py` | LSM baseline: Numba-accelerated backward induction with configurable basis functions (power/laguerre/hermite/chebyshev), polynomial degree, and regularization. Two modes: `LSM_minimal` (spot-only features) and `LSM_full` (full HHK+contract state). Net profitability gate applied at both terminal and non-terminal steps |
| `simulate_hhk_spot.py` | HHK spot price simulation: S_t = exp(f(t) + X_t + Y_t) with mean-reverting OU diffusion and compound Poisson jumps. Uses Sobol quasi-random sequences, stratified sampling of terminal values |
| `swing_contract.py` | Dataclass defining the swing option contract: local/global exercise bounds, strike, maturity, convex cost params (c, gamma), refraction periods |
| `replay_buffer.py` | Circular replay buffer and Prioritized Experience Replay with Numba-accelerated Fenwick tree |
| `agent_evaluation.py` | Shared evaluation logic: batch evaluation, statistics (option price, confidence intervals, exercise stats, bang-bangness) |
| `MultiPro.py` | Multiprocessing wrapper for parallel environment stepping |

### Top-Level Scripts

| File | Role |
|------|------|
| `run.py` | Main training orchestrator (CLI parsing, training loop, evaluation, TensorBoard logging, parquet output) |
| `evaluate_saved_agent.py` | Standalone evaluation of a saved `.pth` model using its `.json` hyperparameters |
| `Price_Swing_Option(s).sh` | Reference v61 config with all hyperparameters documented (no-cost regime, seeds 11-13) |
| `conv_cost_exps.sh` | Sweep orchestrator: runs selected scripts from `Convex Cost Experiments/` sequentially |

### Tools (`tools/`)

| File | Role |
|------|------|
| `rebuild_results_v7.py` | Regenerates `Convex Costs Results 7.csv` and focal CSV from saved models using a common test set (seed=999, 65536 paths). This is the canonical results pipeline |
| `generate_seed_robustness_figure.py` | Produces Figure 4 (seed robustness strip+box plot) from the focal CSV |
| `compare_lsm_state_modes.py` | Compares LSM_minimal vs LSM_full prices across all saved convex-cost configs |
| `update_convex_costs_results.py` | Refreshes result CSVs to expose both LSM benchmark columns |
| `tune_lsm_params.py` | Grid search over LSM hyperparameters (degree, basis, regularization) |
| `validate_discretized_lsm.py` | Tests discretized-action LSM: backward compat, monotonicity in M, convex cost improvement |
| `analyze_lsm_estimator_sample.py` | LSM degree sweep over a 6-config sample |
| `compare_old_new_lsm.py` | Compares old bang-bang LSM vs new discretized-action LSM values |
| `build_latex.sh` / `clean_latex.sh` | Build/clean the LaTeX manuscript |
| `complete_focal_study.sh` | Runs the full 15-seed focal study for c=0.04, gamma=2 |

### Notebooks (`Jupyter Notebooks/`)

| Notebook | Role |
|----------|------|
| `1: Validation 1: Stochastic Process: 2FactorOUwJumps` | Validates HHK simulation correctness |
| `2: Validation 2: LSM Pricing` | Validates LSM benchmark |
| `3: Training Dashboard: TensorBoard Metrics Analysis` | Training diagnostics |
| `4: Evaluation 1: RL vs LSM Analysis` | Statistical comparison of RL vs LSM |
| `5: Convex Costs LSM vs RL` | Main results analysis across the c x gamma grid |
| `6: Convex costs 0.04 Analysis` | Detailed case study; **generates Figures 1-3** for the paper (HHK paths, main results, bang-bangness) |
| `Convex_Costs_Relationships` | Exploration of convex cost relationships |

### Paper (`Paper/`)

- `DRL_Swing_Options.tex` — main manuscript
- `sven.tex` — reviewer (Sven) comments preserved for traceability
- `Bibliography.bib` — references
- Built artifacts go to `Paper/build/`
- Figures live in `figs/` (referenced from manuscript via `../figs/...`)

### Output Locations

| Path | Contents |
|------|----------|
| `logs/<run>/` | Training/evaluation CSVs, parquet files |
| `logs/lsm_full_state/` | Full-state LSM benchmark artifacts |
| `runs/<name>.pth` | Saved actor weights |
| `runs/<name>.json` | Run hyperparameters |
| `Jupyter Notebooks/Convex Costs Results 7.csv` | Canonical results table (3-seed sweep) |
| `Jupyter Notebooks/Convex Costs Results 7 focal.csv` | Focal 15-seed results for c=0.04, gamma=2 |

## Key Metrics

- **Delta%**: `(RL_price / LSM_price - 1) * 100` — RL improvement over LSM
- **Bang-Bangness (B)**: fraction of exercises at q_max. Decreases as gamma grows (~66% at gamma=1, ~21% at gamma=2)

## Critical Implementation Details

1. The profitability gate in the Actor enforces `Pi(q) = q*(S-K)+ - c*q^gamma > 0` before exercise. A Straight-Through Estimator (STE) preserves gradients through the indicator.
2. The environment reward already includes per-step discounting, so D4PG uses `gamma=1`.
3. LSM with convex costs must gate on **net profitability** (`payoff_net > 0`), not gross ITM status, at both terminal and non-terminal steps.
4. `Convex Costs Results 7.csv` is the canonical results file. Regenerate with `tools/rebuild_results_v7.py`.
5. Exploration noise is pre-squash (added before beta-sigmoid), not post-squash.

## Guardrails

- Prefer `make` targets over reconstructing long shell commands.
- Do not run the full convex-cost sweep unless explicitly asked.
- The `HPT.md` file documents the full v1-v62 hyperparameter evolution history.

## Manuscript Color Legend

When reviewing the paper, colored text indicates provenance:
- **Purple**: addresses Sven's review comments
- **Orange**: new statistical methodology/results
- **Green**: new material added independently
- **Red**: Sven's original inline comments
