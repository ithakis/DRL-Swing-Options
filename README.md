# D4PG-QR-FRM: Deep Reinforcement Learning for Swing Option Pricing

*A research-grade framework for pricing swing options using Distributional Deep Deterministic Policy Gradients under the Hambly-Howison-Kluge stochastic process*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Overview

This repository implements a comprehensive framework for pricing swing options in energy markets using advanced deep reinforcement learning techniques. The core innovation lies in applying **Distributional Deep Deterministic Policy Gradients (D4PG)** to solve the complex optimal exercise problem inherent in swing options, while modeling underlying energy prices using the industry-standard **Hambly-Howison-Kluge (HHK)** stochastic process.

### What are Swing Options?

Swing options are exotic derivatives prevalent in energy markets that grant holders multiple exercise rights over the contract's lifetime, subject to various constraints:

- **Local constraints**: Minimum/maximum exercise quantities per decision period
- **Global constraints**: Total volume limits over the contract lifetime  
- **Refraction periods**: Minimum time intervals between exercises
- **Complex payoff structures**: Non-trivial optimal exercise boundaries

Traditional dynamic programming approaches suffer from the curse of dimensionality, making deep reinforcement learning an attractive alternative for pricing and risk management of these instruments.

### Key Innovation

This implementation combines:

1. **D4PG Algorithm**: Distributional RL for uncertainty quantification in option pricing
2. **HHK Stochastic Process**: Realistic energy price modeling with mean reversion and jumps
3. **Advanced RL Extensions**: Prioritized Experience Replay, Munchausen RL, N-step bootstrapping
4. **Comprehensive Benchmarking**: Least-Squares Monte Carlo (LSM) and Finite Difference Methods (FDM)

## Mathematical Framework

### Swing Option Valuation

The swing option pricing problem is formulated as:

$$V_0 = \mathbb{E}\left[\sum_{t=1}^{T} e^{-rt} \cdot q_t \cdot \max(S_t - K, 0)\right]$$

Where:
- $q_t \in [q_{\min}, q_{\max}]$: Exercise quantity at time $t$
- $S_t$: Spot price following the HHK model
- $K$: Strike price
- $r$: Risk-free rate
- Constraints: $\sum_{t=1}^T q_t \leq Q_{\max}$, refraction periods, etc.

### HHK Stochastic Process

The underlying spot price follows:

$$S_t = \exp(f(t) + X_t + Y_t)$$

With:
- **Diffusive OU component**: $dX_t = -\alpha X_t dt + \sigma dW_t$
- **Jump component**: $dY_t = -\beta Y_t dt + J_t dN_t$
- **Seasonal function**: $f(t)$ (e.g., $\log(S_0) + A\cos(2\pi t)$)
- **Jump process**: $N_t \sim \text{Poisson}(\lambda t)$, $J_i \sim \text{Exp}(1/\mu_J)$

### D4PG for Swing Options

The RL formulation treats swing option exercise as a continuous control problem:

- **State Space**: $[S_t/K, Q_{\text{exercised}}/Q_{\max}, \text{TTM}, X_t, Y_t, \ldots]$
- **Action Space**: Normalized exercise quantity $\tilde{q}_t \in [0,1]$
- **Reward**: $r_t = e^{-rt} \cdot q_t \cdot \max(S_t - K, 0)$
- **Policy**: $\pi_\theta(q_t | s_t)$ learned via D4PG

## Repository Structure

```
D4PG-QR-FRM/
├── src/                              # Core implementation
│   ├── agent.py                      # D4PG agent with all extensions
│   ├── networks.py                   # Actor/Critic/IQN neural networks
│   ├── replay_buffer.py              # PER and standard replay buffers
│   ├── swing_env.py                  # Gymnasium environment for swing options
│   ├── swing_contract.py             # Contract specifications and validation
│   ├── simulate_hhk_spot.py          # HHK stochastic process simulation
│   ├── lsm_swing_pricer.py           # Longstaff-Schwartz benchmark
│   └── fdm_swing_pricer.py           # Finite difference benchmark
├── run.py                            # Main training script
├── evaluate_agent.py                 # Policy evaluation and testing
├── run.sh                            # Optimized hyperparameter configuration
├── Jupyter Notebooks/                # Analysis and validation notebooks
│   ├── 1: Validation 1: Stochastic Process: 2FactorOUwJumps.ipynb
│   ├── 2: Validation 2: LSM Pricing.ipynb
│   ├── 3: Training Dashboard: TensorBoard Metrics Analysis.ipynb
│   ├── 4: Evaluation 1: RL vs LSM Analysis.ipynb
│   └── Quantitative_Analysis.ipynb
├── logs/                             # Training logs and evaluation data
├── runs/                             # Saved models and configurations
└── HyperparameterTuning/             # Systematic parameter optimization
```

## Quick Start

### Installation

```bash
git clone https://github.com/ithakis/D4PG-QR-FRM.git
cd D4PG-QR-FRM

# Create conda environment
conda create -n swing_rl python=3.11
conda activate swing_rl

# Install dependencies
pip install torch torchvision gymnasium numpy pandas matplotlib seaborn
pip install scipy scikit-learn tqdm tensorboard quantlib-python
```

### Basic Training

```bash
# Train with optimal configuration for monthly contracts
python run.py \
    -name "SwingOption_Test" \
    -seed 42 \
    -n_paths 8192 \
    -eval_every 1024 \
    -n_paths_eval 2048 \
    --per 1 \
    --munchausen 0 \
    --iqn 0 \
    --nstep 1 \
    --gamma 1 \
    --learn_every 2 \
    --batch_size 64 \
    --tau 0.003 \
    --lr_a 3e-4 \
    --lr_c 2e-4
```

### Production Training

For publication-quality results, use the optimized configuration:

```bash
bash run.sh
```

This launches multiple seeds with 32,768 training episodes and comprehensive evaluation.

## Algorithm Features

### Core D4PG Extensions

- **✅ Distributional Critic (IQN)**: Learn full return distributions for uncertainty quantification
- **✅ Prioritized Experience Replay**: Focus learning on important transitions
- **✅ Munchausen RL**: Entropy-regularized policy improvement for better exploration
- **✅ N-Step Bootstrapping**: Multi-step returns for faster value propagation
- **✅ Soft Target Updates**: Stable target network synchronization

### Swing Option Adaptations

- **Constraint Handling**: Automatic feasibility checking for all contract constraints
- **Monte Carlo Pricing**: Statistical evaluation framework with confidence intervals
- **Multi-Benchmark Evaluation**: LSM and FDM comparison methods
- **Rich State Representation**: Spot price, exercise history, underlying factors, market regime

## Performance Results

Based on the monthly HHK swing contract configuration (see `Jupyter Notebooks/4: Evaluation 1: RL vs LSM Analysis.ipynb`):

### Contract Specifications
- **Maturity**: 1 month (22 decision dates)
- **Strike**: K = 1.0
- **Exercise bounds**: q ∈ [0, 2.0] per period
- **Global cap**: Q_max = 20.0
- **Risk-free rate**: r = 5%

### HHK Model Parameters
- **Mean reversion**: α = 12.0, σ = 1.2
- **Jump process**: β = 150.0, λ = 6.0, μ_J = 0.3
- **Initial spot**: S₀ = 1.0

### Empirical Results

| Method | Option Price | 95% CI | Avg Exercise | Sample Size | Training Time |
|--------|-------------|---------|--------------|-------------|---------------|
| **D4PG-RL** | 2.660 | ±0.085 | 11.66 | 4,096 paths | ~2 hours |
| **LSM Benchmark** | 2.682 | ±0.089 | 11.35 | 4,096 paths | ~10 minutes |
| **Relative Error** | -0.82% | - | +2.7% | - | - |

### Statistical Analysis

**Price Convergence:**
- Bootstrap mean delta (RL - LSM): -0.0204
- 95% Confidence interval: [-0.141, +0.097] 
- Welch's t-test: p = 0.724 (no significant difference)

**Exercise Efficiency:**
- RL efficiency slope: 0.2874 (95% CI: [0.280, 0.295])
- LSM efficiency slope: 0.2942 (95% CI: [0.287, 0.302])
- Both methods show similar value-per-unit allocation efficiency

### Key Insights

1. **Pricing Accuracy**: RL achieves statistically equivalent pricing (p = 0.724) with <1% deviation from LSM
2. **Exercise Behavior**: RL shows earlier exercise timing (p < 1e-17) but similar total allocation patterns
3. **Efficiency Metrics**: Both methods demonstrate comparable efficiency slopes (~0.29), indicating similar optimization quality
4. **Distributional Learning**: RL provides full return distributions while LSM gives point estimates with bootstrap CIs
5. **Temporal Patterns**: RL exhibits front-loading bias while LSM prefers more uniform allocation over time

## Validation & Benchmarking

### Stochastic Process Validation
The HHK implementation is validated against theoretical moments and Monte Carlo convergence (see `Notebook 1`).

### LSM Benchmark
Classical Longstaff-Schwartz pricing serves as the primary benchmark, with polynomial regression of degree 2-4 and bootstrap confidence intervals.

### Convergence Analysis
Training typically converges within 5K-10K episodes, with stable pricing estimates and decreasing policy variance.

## Research Applications

### Quantitative Finance
- **Exotic Option Pricing**: Extend to Asian swings, multi-underlying baskets
- **Risk Management**: VaR/CVaR estimation using distributional critics
- **Portfolio Optimization**: Multi-asset swing option portfolios

### Energy Markets
- **Storage Valuation**: Gas storage as swing options with injection/withdrawal
- **Power Plant Dispatch**: Unit commitment with ramping constraints
- **Renewable Integration**: Flexible generation scheduling

### Methodology Development
- **Algorithm Research**: Novel RL algorithms for derivatives pricing
- **Market Microstructure**: High-frequency trading applications
- **Behavioral Finance**: Learning from market data vs. model-based pricing

## Technical Details

### Neural Network Architecture
- **Actor**: 128-128 hidden layers with tanh activation
- **Critic**: 128-128 hidden layers with ReLU activation  
- **IQN**: 64 quantile samples with cosine embedding
- **Optimization**: Adam with learning rate scheduling

### Training Configuration
```python
# Optimal hyperparameters for swing options
{
    "per": True,              # Prioritized Experience Replay
    "per_alpha": 0.5,         # Prioritization exponent
    "per_beta_start": 0.7,    # Importance sampling bias correction
    "munchausen": False,      # Disabled for stability in current setup
    "iqn": False,            # Standard critic for simplicity
    "nstep": 1,              # Single-step TD learning
    "gamma": 1.0,            # No additional discounting (reward pre-discounted)
    "tau": 0.003,            # Target network soft update rate
    "batch_size": 64,        # Mini-batch size
    "learn_every": 2,        # Learning frequency
    "buffer_size": 200000,   # Replay buffer capacity
}
```

### Computational Requirements
- **Training**: ~2-4 hours on modern GPU (RTX 3080+)
- **Memory**: 4-8GB RAM for standard configurations
- **Storage**: ~1GB for logs and model checkpoints per experiment

## Notebooks & Analysis

### `1: Validation 1: Stochastic Process`
Validates HHK simulation against theoretical moments using bootstrap confidence intervals and convergence diagnostics.

### `2: Validation 2: LSM Pricing`
Implements and validates the Longstaff-Schwartz benchmark with polynomial regression and statistical testing.

### `3: Training Dashboard`
TensorBoard metrics analysis including convergence plots, loss curves, and policy evolution.

### `4: Evaluation 1: RL vs LSM Analysis`
Comprehensive comparison of RL vs LSM pricing with statistical significance testing, exercise pattern analysis, and efficiency metrics.

## Extensions & Future Work

### Immediate Extensions
- **Multi-Asset Swing Options**: Portfolio of correlated underlyings
- **Path-Dependent Payoffs**: Asian-style averaging, lookback features  
- **Regime-Switching Models**: Markov-modulated HHK parameters
- **Real Market Data**: Calibration to historical energy prices

### Research Directions
- **Model-Free Learning**: Direct market data training without HHK assumptions
- **Multi-Agent Systems**: Competitive swing option markets
- **Continuous-Time Limits**: SDE-based policy parameterization
- **Quantum Computing**: Variational quantum circuits for option pricing

## Citation

If you use this codebase in academic research:

```bibtex
@misc{d4pg_swing_options_2025,
    title={D4PG-QR-FRM: Deep Reinforcement Learning for Swing Option Pricing},
    author={[Your Name]},
    year={2025},
    url={https://github.com/ithakis/D4PG-QR-FRM},
    note={Research implementation of distributional deep RL for energy derivatives}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Priority Areas
- Additional stochastic process models (Ornstein-Uhlenbeck variations)
- Alternative RL algorithms (SAC, TD3, PPO adaptations)  
- Real market data integration and calibration tools
- Performance optimizations and distributed training

## Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for research questions and methodology
- **Email**: [maintainer@email.com] for private inquiries

---

*This framework was developed for quantitative finance research and is intended for academic and professional use. Past performance does not guarantee future results. Please ensure compliance with relevant financial regulations when adapting for commercial use.*
