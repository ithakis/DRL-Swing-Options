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

---

## Critical Notes

1. **Profitability gate**: Actor enforces `Pi(q) > 0` before exercise (STE for gradients)
2. **Reward includes discounting**: Use `gamma = 1` in D4PG
3. **LSM uses out-of-sample evaluation**: Separate train/test datasets
4. **Pre-squash noise**: Exploration noise added before tanh01/beta-sigmoid squashing
