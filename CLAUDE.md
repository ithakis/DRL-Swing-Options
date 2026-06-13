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
python export_notebooks_to_pdf.py      # Export notebooks to light-theme PDFs with codes hidden
```

## v64 Canonical Configuration (current default)

**`python run.py` with no flags reproduces a v64 focal run** — the winning configuration from the
"Mega campaign" (see HPT.md). The entire recipe lives in `run.py`'s argparse **defaults**; the
experiment scripts only override the study budget + each cell's `(c, gamma)`. v64 = the v63 kernel-on
canonical PLUS the three orthogonal Stage-C winners:
- **depth-3** actor/critic (`--actor_layers 3 --critic_layers 3`)
- **two gradient steps per interaction** (`-learn_number 2`)
- **softer squash** (`--actor_output_activation beta_sigmoid_1.5`)

On top of the inherited kernel-on canonical: semi-analytical kernel ON (fast `M_x=2`), `-lr_a 3e-4
-lr_c 6e-4`, linear noise + eval-EMA, closed-form warm-start, `--critic_warmup_episodes 512`, single
critic step, robust normalization, `-t 0.0032`. Contract = SwingOption_20 focal (strike 1.0, maturity
0.0833, n_rights 22, q_max 2.0, Q_max 20.0); HHK (α12, σ1.2, β150, λ6, μ_J0.3); `--gamma=1` because the
environment reward already includes discounting; profitability-gate STE; LSM-D benchmark = Chebyshev
degree-2. Study: 32768 episodes, 65536 eval paths, seeds {11,12,13} (focal: 11-25). Convex-cost sweep:
c ∈ {0.01,0.02,0.04,0.05,0.08,0.10,0.15} × gamma ∈ {1,1.5,2,3} (`Convex Cost Experiments/`).

**⚠️ v64 is NOT comparable to the published v61 paper agents** (β 1.5 vs 3.0; 3 vs 2 layers; kernel
ON). The paper requires re-baselining on v64. Reproduce old agents via their own saved-run JSONs.
Reference: `Price_Swing_Option_v64.sh`.

## Semi-Analytical Kernel (feat/semi-analytical-bootstrap)

Optional opt-in mechanism that replaces the critic's single-sample TD bootstrap with an analytical expectation over the HHK transition kernel.  **Bit-identical to v61 when off** (default).  Recommended for kernel-on training:

```bash
--use_expected_target=1
--critic_warmup_episodes=512      # REQUIRED for g1 (γ=1, c>0); removable for g2/nocost only

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

**⚠️ critic_warmup note:** Warmup is **structurally required** for g1 (γ=1, c=0.04) and cannot be removed or substituted.  A 12-seed cc_g1 investigation (under the single-step default) found `warmup=0` collapses **5/12 seeds to ~−14%** (even with a gentler actor LR — the collapse is not LR-fixable).  But 1024 was over-conservative: `warmup=512` and `256` have **0/12 blowups** and a slightly *better* mean than 1024.  **Canonical is now `--critic_warmup_episodes=512`** (best mean +0.337, safe worst-case +0.076; 256 also safe but thinner margin).  See HPT.md “Critic-warmup investigation”.

**⚠️ v63 LR / clipping / critic-step notes** (see HPT.md “v63 Runtime Feature Audit”):
- The LR is **constant — no warmup ramp, no decay**.  The cosine/linear LR-decay schedule was **deleted** (Task 2 below: neutral-to-harmful at 4k); `--final_lr_fraction`/`--lr_schedule_episodes`/`--min_lr` are still accepted but **inert**, and `--warmup_episodes` only sizes the `calibrate_bias` dataset.
- **Gradient clipping was removed in v63** (it was dead code — args were never applied; all v63 results are clip-free).  The `--*_grad_clip*` args no longer exist.
- **Single critic step is now canonical** (`--single_critic_step=1`, default).  A legacy duplicate `critic_optimizer.step()` (effective ~2× critic LR) was a bug; a 12-seed × 3-regime screen showed single-step strictly beats it on mean Δ%, seed std, and worst-case in every regime.  Set `--single_critic_step=0` only to reproduce pre-fix v63 numbers.

## Deterministic-Target Canonical (Tasks 1–3)

Once the kernel makes the TD target deterministic, three retunes were screened (3-seed → 12-seed,
regimes cc_g1/cc_g2/nocost, kernel-on fast M_x=2, 4096 ep) and **adopted into the kernel-on
canonical** (see HPT.md “Deterministic-target retune”). These three changes are now baked into
the v64 `run.py` defaults; no extra flags are needed.

```bash
# Kernel-on canonical after Tasks 1–3 (add to the kernel flags above):
--calibrate_bias_mode closed_form   # Task 1: O(1) myopic-FOC warm-start (default; 'rprop' = legacy loop)
--noise_schedule linear             # Task 2: linear σ0→floor over the FULL horizon (set --noise_plateau 0)
--noise_plateau 0
--weight_averaging ema --ema_decay 0.999   # Task 2: eval-only EMA of actor weights (Schedule-Free spirit)
-lr_a 3e-4 -lr_c 6e-4               # Task 3: faster critic (2× the inherited 3e-4) under the det. target
```

- **Task 1 — closed-form `calibrate_bias`**: replaces the 20-iter Rprop bias loop with a myopic FOC
  warm-start `q*(S)=clip(((S−K)₊/(c·γ))^{1/(γ−1)}, q_min, q_max)` averaged over the warmup spots,
  budget-capped by `Q_max/n_rights`, + one variance-scale + one squash-inverted bias shift (O(1) passes,
  ~10× faster calibration).  Accuracy a statistical wash vs Rprop (all Welch p≥0.44, zero blow-ups);
  adopted for simplicity/determinism/C++-portability.  `_calibrate_bias_closed_form`/`_output_slope` in `src/agent.py`.
- **Task 2 — noise + EMA**: reduced exploration via full-horizon linear noise decay lifts focal cc_g2
  **+0.37pp (Welch p=0.002)** and nocost to ~0; eval-only EMA tightens seed std.  **LR decay deleted**
  (cosine/linear branches + `--lr_decay_shape`/`--lr_warmup_episodes` removed — constant LR is canonical).
- **Task 3 — LR magnitude**: `lr_c=6e-4` (vs 3e-4) improves worst-seed AND seed-std in all 3 regimes and
  mean on cc_g1/nocost; cc_g2 mean dip is trivial/insignificant.  `lr_a` stays 3e-4.

## Function Approximators — REMOVED (debloat, chore/repo-debloat)

The curated-feature approximators (`poly/rff/rbf/tiny_nn`, `BasisActor`/`BasisCritic`,
`CuratedFeatures`, the `--approximator` family of flags, `actor_factory`/`critic_factory`, and
`tools/{test_approximators,sweep_approximators}.py`) **all lost to the NN** and were deleted. The NN
Actor/Critic is the only path. `FinanceInformedActor`/`--actor_type` and the unused `MultiPro.py` were
also removed. See HPT.md "Mega campaign" for the study that motivated this.

## Risk Management: Delta / Gamma (`src/greeks.py`)

Pathwise Greeks via **CRN bump-and-revalue**.  In `simulate_hhk_spot`, `S0` enters *only* through
`X[:,0]=log(S0)-f(0)`; every random draw is keyed off `seed` and is independent of `S0`.  Revaluing
a frozen policy on bundles started at `{S0-dS, S0, S0+dS}` with **one shared seed** therefore reuses
identical randomness — the bump is the near-deterministic multiplicative shift
`S_t(S0')=S_t(S0)*exp(dlogS0*e^{-alpha t})` and the finite-difference noise cancels path-by-path.

- `Delta ~= [V(S0+dS)-V(S0-dS)]/(2 dS)`, `Gamma ~= [V(S0+dS)-2V(S0)+V(S0-dS)]/dS^2`, relative bump
  `dS=h*S0` (default `h=0.01`).  A 5-point stencil + Richardson `(h, h/2)` extrapolation cancels the
  O(h^2) bias; `delta_se` (CRN std error) and `h_spread_*` are the reported error bars.
- `bump_greeks(price_fn, S0, h, seed)` is policy-agnostic; `make_rl_price_fn(agent, contract,
  hhk_params, n_paths)` adapts a trained Agent; `greeks_for_run(name)` loads a saved run end-to-end.
- **Daily-rebalanced hedge**: `rl_dynamic_delta_hedge(agent, contract, hhk_params, S0)` computes the
  per-date continuation Delta by spot-bump-and-re-roll (closed-form OU propagation of the bump, no
  re-sim, via `_roll_from`), **conditions** it on the date-t state (Longstaff–Schwartz, so the hedge
  ratio is F_t-measurable), and hedges with the HHK forward using the martingale increment
  `DF_{t+1}(F_{t+1}-F_t)`.  `regression_forward_hedge(cf, spot, X, Y, q_before, ...)` is the
  policy-agnostic regression-delta hedge used to compare RL vs LSM apples-to-apples.
- Tests: `pytest tools/test_greeks.py` (estimator exactness on closed-form V, Richardson on a quartic,
  CRN-coupling identity, `_roll_from` reproduces the canonical price, dynamic hedge is unbiased +
  variance-reducing, saved-run sanity).  `Jupyter Notebooks/Hedging.ipynb`: 9-pt revalued PV/Δ/Γ grid
  (RL vs LSM), daily hedge, RL-vs-LSM regression hedge, P&L with VaR/ES lines.

## CLI flag status (v64)

The `Convex Cost Experiments/*.sh` sweep scripts were regenerated to v64 in `chore/repo-debloat` —
they now only set the study budget, per-cell `(c, gamma)`, and seeds; everything else is a `run.py`
default. The following flags remain in `run.py` for backward compatibility with saved-run JSONs:

- **Inert (no-op at v64 defaults):** `--final_lr_fraction 1.0`, `--lr_schedule_episodes`, `--min_lr`,
  `--compile 0`, `--single_critic_step 1` (canonical; `=0` only reproduces pre-fix v63 numbers).
- **Still live and canonical:** `--adaptive_noise_scale 0.6`, `--warmup_noise_fraction 0.3/0.4`
  (per-regime), `--noise_schedule linear`, `--calibrate_bias_mode closed_form`.
- **Eval-critical (keep):** `--use_robust_normalization`, `--actor_output_activation`,
  `--weight_averaging` — read from each saved run's `.json` by `rebuild_results_v7`. The v64 defaults
  (`use_robust_normalization=1`, `beta_sigmoid_1.5`, `weight_averaging=ema`) differ from the v61 paper
  agents; always use the saved JSON to reconstruct the correct agent.

## Architecture

### Data Flow

`run.py` is the single entry point for training. It parses CLI args, constructs a `SwingContract`, simulates HHK paths via `simulate_hhk_spot()`, builds a `SwingOptionEnv` (Gymnasium), creates an `Agent`, and runs the training loop. During training, an LSM baseline is also computed for comparison. Evaluation uses pre-generated out-of-sample paths.

### Source Files (`src/`)

| File | Role |
|------|------|
| `agent.py` | D4PG agent: actor-critic updates, exploration-noise schedules (pre-squash Gaussian; `hyperbolic`/`const_floor`/`linear`, canonical = linear σ0→floor), closed-form/Rprop `calibrate_bias` warm-start, eval-only EMA weight averaging, critic warmup, constant-LR target soft updates |
| `swing_env.py` | Gymnasium environment: maps agent actions to exercise quantities, enforces contract constraints (q_min/q_max, Q_min/Q_max, refraction), computes discounted rewards with convex costs. Also contains `approximate_Q_T()` for HHK-based expected quantity estimation |
| `networks.py` | Actor (profitability-gated with STE) and Critic (standard TD). Actor gates unprofitable exercises: `q_out = q_raw * 1[Pi(q) > 0]`. (IQN and the curated-feature approximators were removed.) |
| `lsm_swing_pricer.py` | LSM baseline: Numba-accelerated backward induction with configurable basis functions (power/laguerre/hermite/chebyshev), polynomial degree, and regularization. Two modes: `LSM_minimal` (spot-only features) and `LSM_full` (full HHK+contract state). Net profitability gate applied at both terminal and non-terminal steps |
| `simulate_hhk_spot.py` | HHK spot price simulation: S_t = exp(f(t) + X_t + Y_t) with mean-reverting OU diffusion and compound Poisson jumps. Uses Sobol quasi-random sequences, stratified sampling of terminal values |
| `swing_contract.py` | Dataclass defining the swing option contract: local/global exercise bounds, strike, maturity, convex cost params (c, gamma), refraction periods |
| `replay_buffer.py` | Circular (uniform) replay buffer. (PER was removed in v63 — uniform replay is canonical.) |
| `agent_evaluation.py` | Shared evaluation logic: batch evaluation, statistics (option price, confidence intervals, exercise stats, bang-bangness) |
| `transition_kernel.py` | Analytical HHK transition kernel (quadrature mesh) backing `--use_expected_target=1` deterministic TD targets |
| `greeks.py` | Pathwise Delta/Gamma via CRN bump-and-revalue (`bump_greeks`, `make_rl_price_fn`, `greeks_for_run`); central differences + Richardson, relative bump `dS=h*S0` |
| `hedging_utils.py` | HHK forward price (`hhk_forward_price`), P&L risk metrics (`compute_pnl_risk_metrics`), trace normalization/summary helpers |

### Top-Level Scripts

| File | Role |
|------|------|
| `run.py` | Main training orchestrator (CLI parsing, training loop, evaluation, TensorBoard logging, parquet output) |
| `evaluate_saved_agent.py` | Standalone evaluation of a saved `.pth` model using its `.json` hyperparameters |
| `Price_Swing_Option_v64.sh` | v64 canonical reference (focal c=0.04/γ=2, seeds 11-13). The recipe lives in run.py defaults; this script documents it + sets the study budget |
| `conv_cost_exps.sh` | Sweep orchestrator: runs selected scripts from `Convex Cost Experiments/` sequentially |
| `export_notebooks_to_pdf.py` | Converts Jupyter notebooks to white-theme PDFs with code input cells hidden |

### Tools (`tools/`)

| File | Role |
|------|------|
| `rebuild_results_v7.py` | Regenerates `Convex Costs Results 7.csv` and focal CSV from saved models using a common test set (seed=999, 65536 paths). This is the canonical results pipeline |
| `generate_seed_robustness_figure.py` | Produces Figure 4 (seed robustness strip+box plot) from the focal CSV |
| `compare_lsm_state_modes.py` | Compares LSM_minimal vs LSM_full prices across all saved convex-cost configs |
| `update_convex_costs_results.py` | Refreshes result CSVs to expose both LSM benchmark columns |
| `tune_lsm_params.py` | Grid search over LSM hyperparameters (degree, basis, regularization) |
| `test_greeks.py` | Pytest suite for `src/greeks.py` (estimator exactness, Richardson, CRN-coupling, dynamic hedge, saved-run sanity) |
| `build_latex.sh` / `clean_latex.sh` | Build/clean the LaTeX manuscript |
| `complete_focal_study.sh` | Runs the full 15-seed focal study for c=0.04, gamma=2 |
| `stats_analysis.py` | Shared statistics utilities: Welch's t, Levene/Brown-Forsythe, F-ratio, paired-seed t, MDE, Holm–Bonferroni, conservative Pareto |

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
| `9: C++ Pricer - Speed & Validation` | Speed/validation of the `cpp_pricer/` C++ port: parity tables, Welch price-parity vs PyTorch, optimization ladder, time-vs-paths scaling |
| `Hedging` | **Risk management** — pathwise Delta/Gamma via CRN bump (`src/greeks.py`), value/Δ/Γ-vs-spot curves, and a static HHK-forward hedge backtest. Figures → `figs/hedging_*.png` |
| `Convex_Costs_Relationships` | Exploration of convex cost relationships (exploratory; candidate for removal) |

### C++ Pricer (`cpp_pricer/`)

Standalone, isolated C++ reimplementation of the **v64 kernel-on D4PG** pricer for max CPU speed on
Apple M1 (float32 + Accelerate BLAS + hand-derived backward). Minimizes the sum of (0→4k) train+price
and (4k→65k) OOS eval; **~8.8× faster than PyTorch eager** with the float32 price statistically
indistinguishable from Python (Welch p=0.37). Nothing in `src/` is modified.

- **`cpp_pricer/README.md`** — build/run/validate instructions.
- **`cpp_pricer/DEVELOPMENT_NOTES.md`** — ⚠️ **read this before extending the port.** Architecture
  rationale, the optimization journey (what worked: fast `exp`, BLAS fwd+bwd; what didn't: threading the
  512×64 GEMM, `__restrict__`), the bugs found (SiLU-backward buffer, resize-dangling-pointer, STE
  gradcheck, FP32 FD noise), the validation bar, and future ideas.
- Validate after any change to `mlp.cpp`/`kernel.cpp`/activations: FP64 `test_parity` + `test_grad`
  (<1e-4 / <1e-5), then the float32 price Welch test via `tools/python_baseline.py`.

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
