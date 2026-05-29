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
--critic_warmup_episodes=0        # not needed with kernel

# Fast option — M=4 (M_x=2, M_per_k=1, N_max=1): same Δ% as M=21, ~1.4x baseline wall-clock
--kernel_M_x=2 --kernel_M_per_k=1 --kernel_N_max=1

# Quality option — M=36 (M_x=4, M_per_k=4, N_max=2): tighter seed std, ~2x baseline wall-clock
--kernel_M_x=4 --kernel_M_per_k=4 --kernel_N_max=2
```

Key finding: **M_x is the sole controlling axis**.  M_x=1 collapses (variance ratio 290× vs M_x=2, F-test p=8.8e-8).  M_x≥2 forms a hard plateau: M_x=2,3,4,6 are statistically indistinguishable (Welch p>0.58, Levene p>0.70 for all pairwise pairs).  M_per_k and N_max are irrelevant conditional on M_x≥2 (Phase B, 6 seeds, 7 configs, all Welch p≥0.34).

Headline result (focal c=0.04, gamma=2, 4096 ep, 12 seeds):
- Kernel on  (`H1_only`):   Delta% = +0.47 +/- 0.21 pp
- Kernel off (`B0_baseline`):Delta% = -2.05 +/- 0.44 pp
- Gap = +2.52 pp (Welch's two-sample)

No-cost regression: kernel +5.2 pp better than baseline.  H8 (antithetic), H9 (jump-IW), and their combination confirmed dead at 12 seeds — no variance reduction, H8 marginally worse on mean.

See `Jupyter Notebooks/7: Phase 1 Findings - Semi-Analytical Kernel.ipynb` for the full statistical summary.  Other hypotheses tested (H4 warm-start, H5 Dyna, H6 IQN, H7 twin critics) either did not help or actively hurt.

## Function Approximators (`--approximator`)

With the kernel making the TD target deterministic, the actor/critic **function approximator** is the remaining lever for speed and C++-portability.  `--approximator` swaps the `2x64 SiLU+LayerNorm` net for a **curated feature map + linear head** (one matmul + a cheap transform, no LayerNorm, trivially expressible as a BLAS `gemv` in C++).  All approximators are torch `nn.Module`s honoring the existing actor `(B,9)->(B,1)` / critic `((B,9),(B,1))->(B,1)` signatures, so `agent.py` and `transition_kernel.py` are unchanged.  The actor keeps the beta-sigmoid output + profitability-gate STE.

```bash
--approximator nn        # default: current 2x64 SiLU+LN net (BIT-IDENTICAL to v61 when selected)
--approximator poly      # Chebyshev tensor-product basis      --poly_degree 3
--approximator rff       # Random Fourier Features             --rff_dim 256 --rff_lengthscale 1.0 [--rff_learnable 0]
--approximator rbf       # Radial Basis Function net           --rbf_centers 128 --rbf_bandwidth 1.0 [--rbf_learnable_bandwidth 0]
--approximator tiny_nn   # small 1-hidden-layer net, no LN     --tiny_width 32 --tiny_activation silu
--feature_use_cross 1    # shared: include domain cross-terms (moneyness*inv, ttm*inv, X*Y) in the curated features
```

Key facts:
- `--approximator nn` is the bit-identical default; the new classes have zero RNG side effects on the NN path (guarded by `test_nn_path_unchanged_defaults`).
- `--rbf_bandwidth` is a **relative multiplier of sqrt(dim)** (RBF features vanish at init if sigma is not scaled to the ~11-dim feature space).
- The critic couples the action: poly uses `[phi_s, a*phi_s, a, a^2]`; rff/rbf/tiny_nn concatenate the scaled action into the feature input.  This keeps `dQ/da` informative (DPG-compatible).
- **Implementation**: `src/networks.py` (`CuratedFeatures`, `Poly/RFF/RBF/MLPFeatureMap`, `BasisActor`/`BasisCritic`); injected via `actor_factory`/`critic_factory` in `agent.py`; wired in `run.py`.
- **Correctness**: `pytest tools/test_approximators.py` (37 tests — float64 gradcheck for dQ/da, dQ/dw, da/dtheta; shape; fitting; gate; kernel compatibility; bit-identical NN guard).
- **Sweeps**: `tools/sweep_approximators.py` (Stage A tuning -> Stage B screening 24 seeds -> Stage C finalists, accurate kernel).  CSVs land in `logs/_sweep_approx/` in the `tools/stats_analysis.py` schema.
- **Analysis**: `Jupyter Notebooks/8: Approximator Comparison.ipynb` (speed microbenchmark, correctness, screening/finalist stats, winner selection, end-to-end C++ port plan).  Isolated single-thread microbenchmark: poly ~2.0x, rbf/tiny_nn ~1.6x, rff ~1.45x updates/sec vs the NN.

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
| `test_approximators.py` | Pytest suite for the `--approximator` feature maps (gradcheck, shape, fitting, gate, kernel compatibility, bit-identical NN guard) |
| `sweep_approximators.py` | Orchestrates the approximator comparison: Stage A tuning -> Stage B screening (24 seeds) -> Stage C finalists (accurate kernel). Writes CSVs to `logs/_sweep_approx/` |
| `stats_analysis.py` | Shared statistics: Welch's t, Levene/Brown-Forsythe, F-ratio, paired-seed t, MDE, conservative Pareto. Consumes sweep CSVs |

### Notebooks (`Jupyter Notebooks/`)

| Notebook | Role |
|----------|------|
| `1: Validation 1: Stochastic Process: 2FactorOUwJumps` | Validates HHK simulation correctness |
| `2: Validation 2: LSM Pricing` | Validates LSM benchmark |
| `3: Training Dashboard: TensorBoard Metrics Analysis` | Training diagnostics |
| `4: Evaluation 1: RL vs LSM Analysis` | Statistical comparison of RL vs LSM |
| `5: Convex Costs LSM vs RL` | Main results analysis across the c x gamma grid |
| `6: Convex costs 0.04 Analysis` | Detailed case study; **generates Figures 1-3** for the paper (HHK paths, main results, bang-bangness) |
| `7: Phase 1 Findings - Semi-Analytical Kernel` | Statistical summary of the semi-analytical kernel study (M_x isolation, hypothesis tests) |
| `8: Approximator Comparison` | Compares the `--approximator` contenders: speed microbenchmark, correctness, screening/finalist stats, winner, end-to-end C++ port plan |
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
