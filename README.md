# Deep Reinforcement Learning for Swing Option Pricing

A research-grade implementation of **D4PG** (Distributional Deep Deterministic Policy Gradient) for pricing **swing options** with **convex exercise costs** in energy markets.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Quick Start

### 1. Environment Setup

```bash
# Create the research environment from the tracked manifest
conda env create -f environment.yml
conda activate EP11
```

If you prefer a manual setup, `environment.yml` records the same conda-forge based package stack used by this repository.

### 2. Run the Convex Cost Experiments

```bash
make sweep
```

This sweeps over convex cost configurations (c × γ) and runs 3 seeds each.

### 3. Monitor Training

```bash
make tensorboard
```

### 4. Analyze Results

Open the Jupyter Notebooks:
- `5: Convex Costs LSM vs RL.ipynb` — Main comparison analysis
- `6: Convex costs 0.04 Analysis.ipynb` — Detailed case study
- `Hedging.ipynb` — Pathwise Delta/Gamma (CRN bump-and-revalue) and a forward-hedge backtest for risk management

### 5. Build The Paper

```bash
make paper
```

This compiles the manuscript in a temporary directory and copies the final PDF and auxiliary artifacts to `Paper/build/`.

To remove generated manuscript artifacts:

```bash
make clean-paper
```

### Stable Repo Commands

The repository now exposes a small set of stable entry points for humans and agents:

```bash
make help
make sweep
make single-exp EXP_SCRIPT="Convex Cost Experiments/SwingOption_20_c0.05_gamma2.sh"
make train ARGS='-name "MyExperiment" -seed 42 -n_paths 32768 --c_cost 0.05 --gamma_cost 2.0'
make eval RUN_NAME=MyExperiment RUNS=100
make compare-lsm-state-modes
make paper
```

---

## Research Context

### What are Swing Options?

Swing options are exotic derivatives in energy markets that grant holders **multiple exercise rights** over the contract lifetime, subject to:
- **Per-period bounds**: min/max exercise quantity per decision date
- **Global bounds**: total volume limits over the contract
- **Refraction periods**: cooldown between exercises

### Why Convex Costs?

With **convex exercise costs** (cost = c · q^γ, where γ > 1), the optimal policy becomes **continuous** rather than bang-bang. This makes the problem well-suited to deep RL with continuous action spaces.

### RL vs LSM

| Aspect | Deep RL (D4PG) | LSM |
|--------|----------------|-----|
| Policy | Neural network | Polynomial regression |
| Action space | Continuous [0, q_max] | Discretized by design |
| Convex costs | Native support | Requires modification |
| Computation | Train once, fast inference | Re-solve per path |

The repository now distinguishes two bang-bang LSM baselines:
- `LSM_minimal`: legacy reduced-state regression on spot-price basis functions only.
- `LSM_full`: upgraded full-state regression on the HHK and contract state, used for the current paper comparisons.

> **⚠️ LSM with convex costs**: The LSM exercise gate must check **net profitability** (`payoff_net > 0`), not gross in-the-money status, at both terminal and non-terminal decisions. The current implementation applies this consistently throughout backward induction.

---

## The Algorithm

### D4PG with Profitability-Gated Actor

The actor network enforces a **profitability constraint** directly:

```
q_raw = squash(u_θ(s))           # Raw action in [0,1]
q_actual = q_min + q_raw × (q_max - q_min)
Π(q) = q × (S - K)⁺ - c × q^γ   # Net profit
q_out = q_raw × 𝟙[Π(q) > 0]      # Gate unprofitable
```

A **Straight-Through Estimator (STE)** preserves gradients through the gate.

### HHK Stochastic Process

The spot price follows the Hambly-Howison-Kluge model:

```
S_t = exp(f(t) + X_t + Y_t)
dX_t = -α X_t dt + σ dW_t        # Mean-reverting diffusion
dY_t = -β Y_t dt + J dN_t        # Jump component
```

---

## Recent Capabilities

### Semi-Analytical Transition Kernel (`--use_expected_target=1`)

Replaces the critic's single-sample TD bootstrap with an **analytical expectation over the HHK
transition kernel**, making the TD target deterministic. Bit-identical to v61 when off. `M_x` is
the sole controlling accuracy axis (`M_x=2` is a hard fast plateau). See `src/transition_kernel.py`
and notebook 7.

### Risk Management: Delta / Gamma & Hedging

`src/greeks.py` computes **pathwise Delta and Gamma** via Common-Random-Number bump-and-revalue:
the policy is revalued on path bundles started at `{S0−δ, S0, S0+δ}` sharing one simulation seed
(in HHK the seed-keyed draws are independent of `S0`, so the bump is near-deterministic and the
finite-difference noise cancels). Central differences with a relative bump `δ=h·S0`, a 5-point
stencil and Richardson `(h, h/2)` extrapolation give low-variance, low-bias Greeks. `Hedging.ipynb`
adds a **9-point revalued PV/Δ/Γ grid (RL vs LSM)**, a **daily-rebalanced** delta hedge
(`rl_dynamic_delta_hedge`: per-date continuation Δ by bump-and-re-roll, conditioned to be
`F_t`-measurable, hedged with the HHK forward), an apples-to-apples RL-vs-LSM regression-delta hedge,
and final P&L distributions with **VaR/ES** lines. Validated by `tools/test_greeks.py`.

```python
from src.greeks import greeks_for_run
r = greeks_for_run("SwingOption_20_c0.04_gamma2_11", n_paths=16384, h=0.01, seed=999)
print(r.delta, r.gamma, r.delta_se)   # CRN standard error is tiny
```

---

## Experiment Configuration

### Swing Contract

| Parameter | Value | Description |
|-----------|-------|-------------|
| Strike (K) | 1.0 | Exercise strike price |
| Maturity | ~1 month | 0.0833 years |
| Decision dates | 22 | Exercise opportunities |
| q_min, q_max | 0, 2.0 | Per-period exercise bounds |
| Q_min, Q_max | 0, 20.0 | Global volume bounds |
| Risk-free rate | 5% | Discounting rate |

### Convex Cost Sweep

The main experiments (`conv_cost_exps.sh`) sweep over:

| c (cost coefficient) | γ (cost exponent) |
|---------------------|-------------------|
| 0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15 | 1, 1.5, 2, 3 |

Each configuration runs with seeds {11, 12, 13} for robustness.

### Focal Seed Robustness Study

For the representative configuration c = 0.04, γ = 2, an extended focal study trains 15 seeds (11–25) to assess training variability:

```bash
make single-exp EXP_SCRIPT="Convex Cost Experiments/SwingOption_20_c0.04_gamma2_focal.sh"
```

Results: RL mean **1.9613 ± 0.0088** (CV 0.45%) vs LSM baseline **1.8634** — all 15 seeds exceed LSM by at least +4.2%.

### HHK Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| S₀ | 1.0 | Initial spot |
| α | 12.0 | Mean-reversion speed |
| σ | 1.2 | Diffusion volatility |
| β | 150.0 | Jump decay rate |
| λ | 6.0 | Jump intensity (per year) |
| μ_J | 0.3 | Mean jump size |

---

## Running Experiments

### Full Sweep

```bash
make sweep
```

### Individual Configuration

```bash
make single-exp EXP_SCRIPT="Convex Cost Experiments/SwingOption_20_c0.05_gamma2.sh"
```

### Custom Run

```bash
make train ARGS=' \
    -name "MyExperiment" \
    -seed 42 \
    -n_paths 32768 \
    --c_cost 0.05 \
    --gamma_cost 2.0 \
    -eval_every 1024 \
    -n_paths_eval 65536'
```

### Key CLI Flags

Defaults below are the **v64 canonical** — `python run.py` with no flags reproduces a v64 focal run.

| Flag | v64 default | Description |
|------|---------|-------------|
| `-n_paths` | 10000 | Training episodes (the study scripts override to 32768) |
| `-eval_every` | 1000 | Evaluation frequency |
| `-n_paths_eval` | 1 | Paths per evaluation |
| `--c_cost` | 0.04 | Convex cost coefficient (focal cell) |
| `--gamma_cost` | 2.0 | Convex cost exponent (focal cell) |
| `-seed` | 0 | Random seed |
| `--use_expected_target` | 1 | Analytical HHK transition-kernel expectation (deterministic TD target) |
| `--kernel_M_x` | 2 | Gauss–Hermite nodes on the OU transition (sole accuracy axis; `2` is the fast plateau) |
| `--actor_layers` / `--critic_layers` | 3 | Net depth (v64 Stage-C winner; was 2) |
| `-learn_number` | 2 | Gradient steps per interaction (v64 Stage-C winner; was 1) |
| `--actor_output_activation` | beta_sigmoid_1.5 | Softer squash (v64 Stage-C winner; was 3.0). **Eval-critical** |
| `-lr_a` / `-lr_c` | 3e-4 / 6e-4 | Actor / critic learning rates |
| `--noise_schedule` | linear | σ0→floor over the horizon |
| `--weight_averaging` | ema | Eval-only EMA of actor weights |
| `--critic_warmup_episodes` | 512 | Freeze actor for the first 512 episodes |
| `--use_robust_normalization` | 1 | Log-moneyness + median/IQR input scaling |

> A fuller flag reference lives in `CLAUDE.md`. The config is being simplified on the
> `refactor/simplify-config` branch — several legacy/ablation knobs are slated for removal
> (see CLAUDE.md “CLI flag cleanup”).

---

## Output & Results

### Output Locations

| Path | Contents |
|------|----------|
| `logs/<run>/` | Training/evaluation CSVs, parquet files |
| `logs/lsm_full_state/` | Full-state LSM reruns and parquet diagnostics |
| `runs/<name>.pth` | Saved actor weights |
| `runs/<name>.json` | Hyperparameters |
| `Jupyter Notebooks/Convex Costs Results *.csv` | Aggregated results |
| `Paper/build/DRL_Swing_Options.pdf` | Built manuscript PDF |

### Benchmark Maintenance Tools

- `tools/compare_lsm_state_modes.py` compares `LSM_minimal` and `LSM_full` and writes benchmark summaries to `logs/`.
- `tools/update_convex_costs_results.py` refreshes the convex-cost result tables and CSVs used by the paper and notebooks after rerunning evaluations.
- `tools/rebuild_results_v7.py` regenerates `Convex Costs Results 7.csv` and `Convex Costs Results 7 focal.csv` from the latest evaluation logs, used by paper figures.
- `tools/generate_seed_robustness_figure.py` produces Figure 4 (seed robustness strip+box plot) from the focal results CSV.
- Full-state benchmark artifacts live under `logs/lsm_full_state/`.

### Key Metrics

- **Delta%**: `(RL_price / LSM_price - 1) × 100` — RL improvement over LSM
- **Average100**: Rolling average episodic return
- **Action_variance_mean**: Policy exploration health
- **Bang-Bangness (B)**: Fraction of exercises at maximum capacity (q = q_max). Measures how close the RL policy is to a bang-bang strategy. Decreases as convex cost exponent γ grows.

---

## Analysis Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `1: Validation 1: Stochastic Process` | HHK simulation validation | Stable |
| `2: Validation 2: LSM Pricing` | LSM benchmark validation | Stable |
| `3: Training Dashboard` | TensorBoard metrics analysis | Depends on deleted `logs/` — redo or drop |
| `4: Evaluation 1: RL vs LSM Analysis` | Statistical comparison | Rerun under the kernel-on canonical |
| `5: Convex Costs LSM vs RL` | **Main results analysis** | Rerun (paper figures) |
| `6: Convex costs 0.04 Analysis` | Case study; generates Figures 1–3 (HHK paths, main results, Bang-Bangness) | Rerun (paper figures) |
| `Hedging` | Pathwise Delta/Gamma (CRN bump) + forward-hedge backtest | New |

> The paper-figure notebooks (4–6) need a rerun once the simplified kernel-on canonical config
> is frozen, since the underlying numbers shift. Notebook 1 (the stochastic process) is solid.

---

## Repository Structure

```
DRL-Swing-Options/
├── run.py                      # Main training script
├── conv_cost_exps.sh           # Experiment sweep orchestrator
├── Paper/
│   ├── DRL_Swing_Options.tex   # Main manuscript source
│   ├── Bibliography.bib        # Manuscript bibliography
│   ├── cas-dc.cls              # Elsevier class file
│   ├── cas-common.sty          # Elsevier shared style file
│   ├── model1-num-names.bst    # Bibliography style
│   └── build/                  # Generated manuscript artifacts
├── Convex Cost Experiments/    # Individual experiment configs
│   └── SwingOption_20_c*.sh    # 25 configurations (c × γ)
├── src/
│   ├── agent.py                # D4PG agent implementation
│   ├── swing_env.py            # Gymnasium environment
│   ├── networks.py             # Actor (profitability-gated STE) + Critic
│   ├── transition_kernel.py    # Analytical HHK transition kernel (deterministic TD target)
│   ├── greeks.py               # Pathwise Delta/Gamma (CRN bump-and-revalue)
│   ├── hedging_utils.py        # HHK forward price, P&L risk metrics, trace helpers
│   ├── lsm_swing_pricer.py     # LSM_minimal and LSM_full benchmarks
│   ├── simulate_hhk_spot.py    # HHK simulation
│   └── replay_buffer.py        # circular (uniform) replay buffer
├── Jupyter Notebooks/          # Analysis and validation
├── tools/
│   ├── build_latex.sh                    # Local paper build script
│   ├── clean_latex.sh                    # Paper artifact cleanup
│   ├── compare_lsm_state_modes.py        # Compare reduced-state vs full-state LSM outputs
│   ├── update_convex_costs_results.py    # Refresh paper/notebook convex-cost result tables
│   ├── rebuild_results_v7.py             # Regenerate Results 7 CSVs from evaluation logs
│   └── generate_seed_robustness_figure.py # Generate Figure 4 (seed robustness strip+box plot)
├── logs/                       # Training outputs
├── runs/                       # Saved models
└── HPT.md                      # Hyperparameter tuning history
```

---

## Development Notes

### Hyperparameter Evolution

The algorithm has evolved through v1–v63 plus a post-v62 research phase (see `HPT.md`). Key milestones:
- **v42**: Profitability-gated actor with STE
- **v59**: Critic warmup + adaptive noise
- **v61**: Paper configuration for convex costs (IQN off, standard critic)
- **v63**: Semi-analytical kernel (deterministic TD target) + deterministic-target retune
  (closed-form warm-start, linear noise + eval-only EMA, faster critic LR); PER and TD3 target
  smoothing removed
- **v64** (current default): the v63 kernel-on canonical + the Mega-campaign Stage-C winners —
  depth-3 net, `learn_number=2`, `beta_sigmoid_1.5` (see `HPT.md` → "Mega campaign")

### Architecture Summary (v64 defaults)

- **Networks**: 3×64 MLPs, SiLU activation, LayerNorm
- **Replay**: uniform (PER removed in v63)
- **Noise**: Pre-squash Gaussian; full-horizon **linear** σ0→floor decay + eval-only EMA
- **Output**: β-sigmoid(1.5) for softer saturation (interior exercise under convex cost)

---

## Citation

```bibtex
@misc{drl_swing_options_2025,
    title={Deep Reinforcement Learning for Swing Option Pricing with Convex Costs},
    author={Ithakis, Alexander},
    year={2025},
    url={https://github.com/ithakis/DRL-Swing-Options}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
