# Deep Reinforcement Learning for Swing Option Pricing

A research-grade implementation of **D4PG** (Distributional Deep Deterministic Policy Gradient) for pricing **swing options** with **convex exercise costs** in energy markets.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment (Python 3.11, conda-forge only)
conda create -n EP11 -c conda-forge python=3.11
conda activate EP11

# Core dependencies
conda install -c conda-forge \
    numpy scipy pandas matplotlib seaborn scikit-learn \
    tqdm gymnasium tensorboard ipykernel numba

# PyTorch CPU (avoid pip to prevent OpenMP conflicts)
conda install -c conda-forge pytorch torchvision torchaudio
```

### 2. Run the Convex Cost Experiments

```bash
cd /path/to/DRL-Swing-Options
conda activate EP11
bash conv_cost_exps.sh
```

This sweeps over convex cost configurations (c × γ) and runs 3 seeds each.

### 3. Monitor Training

```bash
tensorboard --logdir=runs \
  --load_fast=true \
  --samples_per_plugin=scalars=500 \
  --reload_interval=30 \
  --max_reload_threads=4
```

### 4. Analyze Results

Open the Jupyter Notebooks:
- `5: Convex Costs LSM vs RL.ipynb` — Main comparison analysis
- `6: Convex costs 0.04 Analysis.ipynb` — Detailed case study

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
bash conv_cost_exps.sh
```

### Individual Configuration

```bash
bash "Convex Cost Experiments/SwingOption_20_c0.05_gamma2.sh"
```

### Custom Run

```bash
python run.py \
    -name "MyExperiment" \
    -seed 42 \
    -n_paths 32768 \
    --c_cost 0.05 \
    --gamma_cost 2.0 \
    -eval_every 1024 \
    -n_paths_eval 65536
```

### Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-n_paths` | 10000 | Training episodes |
| `-eval_every` | 1000 | Evaluation frequency |
| `-n_paths_eval` | 1 | Paths per evaluation |
| `--c_cost` | 0.0 | Convex cost coefficient |
| `--gamma_cost` | 1.0 | Convex cost exponent |
| `-seed` | 0 | Random seed |
| `-layer_size` | 64 | Network hidden size |
| `-per` | 1 | Enable PER |

---

## Output & Results

### Output Locations

| Path | Contents |
|------|----------|
| `logs/<run>/` | Training/evaluation CSVs, parquet files |
| `runs/<name>.pth` | Saved actor weights |
| `runs/<name>.json` | Hyperparameters |
| `Jupyter Notebooks/Convex Costs Results *.csv` | Aggregated results |

### Key Metrics

- **Delta%**: `(RL_price / LSM_price - 1) × 100` — RL improvement over LSM
- **Average100**: Rolling average episodic return
- **Action_variance_mean**: Policy exploration health

---

## Analysis Notebooks

| Notebook | Purpose |
|----------|---------|
| `1: Validation 1: Stochastic Process` | HHK simulation validation |
| `2: Validation 2: LSM Pricing` | LSM benchmark validation |
| `3: Training Dashboard` | TensorBoard metrics analysis |
| `4: Evaluation 1: RL vs LSM Analysis` | Statistical comparison |
| `5: Convex Costs LSM vs RL` | **Main results analysis** |
| `6: Convex costs 0.04 Analysis` | Detailed case study |

---

## Repository Structure

```
DRL-Swing-Options/
├── run.py                      # Main training script
├── conv_cost_exps.sh           # Experiment sweep orchestrator
├── Convex Cost Experiments/    # Individual experiment configs
│   └── SwingOption_20_c*.sh    # 25 configurations (c × γ)
├── src/
│   ├── agent.py                # D4PG agent implementation
│   ├── swing_env.py            # Gymnasium environment
│   ├── networks.py             # Actor/Critic/IQN networks
│   ├── lsm_swing_pricer.py     # LSM benchmark
│   ├── simulate_hhk_spot.py    # HHK simulation
│   └── replay_buffer.py        # PER and circular buffers
├── Jupyter Notebooks/          # Analysis and validation
├── logs/                       # Training outputs
├── runs/                       # Saved models
└── HPT.md                      # Hyperparameter tuning history
```

---

## Development Notes

### Hyperparameter Evolution

The algorithm has evolved through 62 versions (see `HPT.md`). Key milestones:
- **v42**: Profitability-gated actor with STE
- **v59**: Critic warmup + adaptive noise
- **v61**: Current recommended for convex costs

### Architecture Summary

- **Networks**: 2×64 MLPs, SiLU activation, LayerNorm
- **PER**: Soft alpha ramp (0.1 → 0.2), beta ≈ 1.0
- **Noise**: Pre-squash Gaussian, plateau + hyperbolic decay
- **Output**: β-sigmoid(3.0) for softer saturation

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
