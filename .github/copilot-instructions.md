# Copilot Instructions for DRL-Swing-Options

## Purpose

Research-grade implementation of **D4PG** (Deep Deterministic Policy Gradient) for pricing **swing options** in energy markets. The algorithm prices options with **convex exercise costs** using the **Hambly-Howison-Kluge (HHK)** stochastic process and benchmarks against **Least-Squares Monte Carlo (LSM)**.

---

## Running Experiments

### Main Experiment Sweep
```bash
cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options && conda activate EP11
bash conv_cost_exps.sh
```
This runs individual scripts from `Convex Cost Experiments/` (e.g., `SwingOption_20_c0.05_gamma2.sh`).

### Single Experiment
```bash
bash "Convex Cost Experiments/SwingOption_20_c0.05_gamma2.sh"
```

### Direct Python Training
```bash
python run.py -name "Test" -seed 42 -n_paths 8192 --c_cost 0.05 --gamma_cost 2 [other args]
```

### Evaluation
```bash
python evaluate_saved_agent.py --run_name <name> --runs 100
```

### TensorBoard
```bash
tensorboard --logdir=runs --load_fast=true
```

---

## Codebase Map

| File | Purpose |
|------|---------|
| `run.py` | Training orchestration: CLI parsing, env/agent setup, training loop, evaluation |
| `src/agent.py` | D4PG agent: networks, PER, noise schedules, optimizer configuration |
| `src/swing_env.py` | Gymnasium environment: state representation, reward calculation, contract constraints |
| `src/networks.py` | Neural architectures: Actor (profitability-gated), Critic, IQN |
| `src/lsm_swing_pricer.py` | LSM benchmark: Longstaff-Schwartz with polynomial basis |
| `src/simulate_hhk_spot.py` | HHK simulation: 2-factor OU with jumps, Sobol sequence |
| `src/swing_contract.py` | Contract specification: exercise bounds, refraction, discounting |
| `src/replay_buffer.py` | Replay buffers: Circular and PER with Fenwick tree |

### Paper Layout
- Manuscript sources live in `Paper/`
- Main paper entrypoint: `Paper/DRL_Swing_Options.tex`
- `Paper/sven.tex` preserves Sven's original red-comment review version; when working on manuscript claims, tables, or figures, treat Sven's feedback there as active review context we are trying to address
- Paper-local assets: `Paper/Bibliography.bib`, `Paper/cas-dc.cls`, `Paper/cas-common.sty`, `Paper/model1-num-names.bst`
- Generated paper artifacts live in `Paper/build/`
- Figures remain in `figs/` and are referenced from the manuscript via `../figs/...`

---

## Default Configuration (Convex Cost Experiments)

### Swing Contract
- Strike: `K = 1.0`
- Maturity: `~1 month (0.0833 years)`
- Decision dates: `n_rights = 22`
- Exercise bounds: `q ∈ [0, 2.0]` per period, `Q ∈ [0, 20.0]` total
- Risk-free rate: `r = 5%`

### HHK Process
- `S0 = 1.0`, `alpha = 12.0`, `sigma = 1.2`
- `beta = 150.0`, `lambda = 6.0`, `mu_J = 0.3`

### Convex Costs (experiment sweep)
- `c_cost ∈ {0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15}`
- `gamma_cost ∈ {1, 1.5, 2, 3}`

### Training
- Episodes: `32768`, Batch: `128`, Seeds: `{11, 12, 13}`
- PER: soft alpha ramp (0.1 → 0.2), beta ≈ 1.0
- Noise: pre-squash Gaussian, plateau + hyperbolic decay
- Networks: 2×64 SiLU MLPs, LayerNorm, orthogonal init

---

## Output Locations

| Path | Contents |
|------|----------|
| `logs/<run>/` | Training/evaluation CSVs, parquet files |
| `runs/<name>.pth` | Saved actor weights |
| `runs/<name>.json` | Run hyperparameters |
| `Jupyter Notebooks/*.csv` | Aggregated results tables |

---

## Key CLI Flags

**Contract**: `--strike`, `--maturity`, `--n_rights`, `--q_min`, `--q_max`, `--Q_max`, `--c_cost`, `--gamma_cost`

**Training**: `-n_paths`, `-eval_every`, `-n_paths_eval`, `-seed`, `-name`

**Agent**: `-per`, `--per_alpha`, `-noise_sigma0`, `-noise_floor`, `-lr_a`, `-lr_c`, `-layer_size`, `--activation`

**LSM**: `--lsm_basis`, `--lsm_degree`, `--lsm_reg`

**System**: `-n_cores`, `--compile`, `--disable_csv_logging`

---

## Development Resources

| Resource | Purpose |
|----------|---------|
| `HPT.md` | Hyperparameter tuning history (v1 → v62) |
| `Jupyter Notebooks/` | Validation and results analysis |
| `results.md` | Quick results summary |
| `Convex Cost Experiments/` | Pre-configured experiment scripts |

### Paper Build
```bash
cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options
./tools/build_latex.sh "$PWD/Paper" DRL_Swing_Options.tex
```
This compiles in a temporary directory and copies the final artifacts to `Paper/build/`.

### Paper Clean
```bash
cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options
./tools/clean_latex.sh
```

### Paper Figure Regeneration
The three paper figures are generated from `Jupyter Notebooks/6: Convex costs 0.04 Analysis.ipynb`.

Minimum reliable rerun path from a fresh kernel:
- Run cells 2, 3, 4, 5, and 6 to load paths, helper functions, evaluation data, and figure labels.
- For Figure 1 (`hist_exercise.pdf`): run cell 22.
- For Figure 2 (`spot_income_pv_hist.png`): run cell 14 first to rebuild `path_stats`, then run cell 25.
- For Figure 3 (`bang_bangness_rl.pdf`): run cell 27 first to refresh Bang-Bangness values in `Convex Costs Results 6.csv`, then run cell 28.

Minimum rerun path when the kernel already has the setup state loaded:
- Figure 1 only: rerun cell 22.
- Figure 2 only: rerun cell 25, unless `path_stats` was invalidated, in which case rerun cell 14 first.
- Figure 3 only: rerun cell 28, unless the underlying Bang-Bangness data changed, in which case rerun cell 27 first.

If multiple paper figures are edited in one pass, the smallest safe sequence from a fresh kernel is cells 2, 3, 4, 5, 6, 14, 22, 25, 27, and 28.

Minimum rerun path after editing the figure-producing cells themselves:
- If you edit cell 22 only, rerun cell 22 only.
- If you edit cell 25 only, rerun cell 25 only, as long as cells 2, 3, 4, 5, 6, and 14 are still valid in the current kernel.
- If you edit cell 27 only, rerun cells 27 and 28, because cell 27 updates the CSV consumed by cell 28.
- If you edit cell 28 only, rerun cell 28 only, as long as cell 27 has already produced up-to-date Bang-Bangness values.
- If you edit both cells 27 and 28, rerun cells 27 and 28.
- If you edit cells 14 and 25, rerun cells 14 and 25.

Smallest safe rebuild sets for agentic figure work:
- Figure 1 only: cells 22.
- Figure 2 only: cells 14 and 25 from a fresh setup, or just cell 25 if `path_stats` is already current.
- Figure 3 only: cells 27 and 28 when the metric data changes, or just cell 28 for styling-only edits.
- All three figures from a fresh kernel: cells 2, 3, 4, 5, 6, 14, 22, 25, 27, and 28.

---

## Critical Notes

1. **Profitability gate**: Actor enforces `Pi(q) > 0` before exercise (STE for gradients)
2. **Reward includes discounting**: Use `gamma = 1` in D4PG
3. **LSM uses out-of-sample evaluation**: Separate train/test datasets
4. **Pre-squash noise**: Exploration noise added before tanh01/beta-sigmoid squashing
5. **LSM convex cost caveat**: The terminal step must gate on **net profitability** (`payoff_net > 0`), not just ITM (`payoff_gross > 0`). With `c > 0`, exercising at the last step can be unprofitable even when in-the-money.
