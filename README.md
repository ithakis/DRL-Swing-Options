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

## Recent Updates

- Switchable optimizers via `--optimizer {adam, adamw}` (default: AdamW) with decoupled `--weight_decay_actor` and `--weight_decay_critic` controls (critic defaults to `1e-4`).
- Actor and critic networks now respect `-layer_size` and can be tuned independently with `--actor_hidden_size`, `--critic_hidden_size`, `--actor_layers`, and `--critic_layers`.
- Model build logs report the chosen widths/depths and parameter counts for both actor and critic to simplify experiment tracking.
- Configurable gradient clipping for actor/critic (`--actor_grad_clip`, `--critic_grad_clip`) with norm/value modes; disabled by default for speed, but available when stability demands it.
- Convex per-exercise cost term controllable via `--c_cost` and `--gamma_cost`, shared between the RL environment and LSM benchmark (defaults keep the cost disabled).
- Default hidden activation switched to **SiLU** (configurable via `--activation {relu, leaky_relu, silu}`) to smooth gradients and improve initial action variance.
- **v42 architecture**: profitability-constrained actor output using a hard gate + Straight-Through Estimator (STE), with exploration noise applied pre-gate so the executed policy never exercises unprofitably.

## Neural Network Architecture (v42)

### Actor: profitability-gated continuous policy

The actor is a lightweight MLP (default `2×64`, `LayerNorm` + `SiLU`) with a `tanh01` output head producing a normalized proposal `q_raw ∈ [0,1]`. In v42 we add an **actor-level feasibility layer** that enforces a *hard* profitability constraint for swing exercise with convex costs:

- Immediate profit used for gating:
  - `Pi(q) = q * relu(S - K) - c_cost * q**gamma_cost`
- The actor output is gated as:
  - if `Pi(q_raw) <= 0` → execute `q = 0`
  - else → execute `q = q_raw`

To avoid killing gradients in the masked region, the hard gate uses a **Straight-Through Estimator (STE)**: the forward pass uses the hard-gated action, while the backward pass propagates gradients as if the ungated `q_raw` was used.

Implementation lives in:
- `src/networks.py` (`Actor.apply_profitability_gate`, `Actor.forward_raw_and_gated`)
- `src/agent.py` (`Agent.act` applies noise pre-gate, then gates)

### Exploration: noise is applied pre-gate

Exploration noise is added in pre-activation space (before the `tanh01` squash) to form `q_raw`, and then the profitability gate is applied. Noise is **never** applied after gating. This keeps exploration effective near the profitability boundary while ensuring the environment never receives an unprofitable exercise due to noise.

### Why choose this over env-only masking?

The environment already applies a safety mask (sets `q=0` when realized net payoff is non-positive). However, env-only masking can create an off-policy learning mismatch: the replay buffer contains the proposed action while the environment executes a different masked action. That tends to flatten value gradients near the true profitability boundary (“mask-banging”) and increases seed-to-seed variability.

By moving the constraint into the actor:
- executed actions are feasible-by-construction,
- the critic is trained on actions consistent with the resulting rewards/next-states,
- the actor is optimized for the *executed* policy (not a policy that is later corrected downstream),
- seed robustness improves, especially late.

### Empirical impact (v42 vs v41)

Across the v41/v42 3-seed comparison runs, v42 shows:
- noticeably tighter late-stage seed-to-seed variance in `Pricing/Delta_Percent`,
- cleaner bands for `Policy/Action_variance_mean` and `Policy/Actions_at_upper_pct`,
- more stable/consistent exercise behavior metrics (less regime-dependent variance),
- small shifts in PER/TD-error statistics consistent with learning a sharper (more correct) boundary rather than exploiting a masked degeneracy.

For a detailed discussion of the metric-by-metric differences and the evaluation-noise interpretation, see `HPT.md` (v42 section).

### Related work (similar ideas)

This design combines two well-known patterns:

1) **Feasibility / safety layers** that enforce constraints by construction (often via projection or post-processing of actions), e.g.:
   - Dalal et al. (2018), *Safe Exploration in Continuous Action Spaces* (safety layer style constraints)
   - Achiam et al. (2017), *Constrained Policy Optimization* (constraint satisfaction as a first-class objective)
   - Amos & Kolter (2017), *OptNet* (differentiable optimization layers; a heavier-weight alternative to hard gating)

2) **Straight-through estimators** to train through discrete / non-differentiable decisions:
   - Bengio et al. (2013), *Estimating or Propagating Gradients Through Stochastic Neurons*
   - Courbariaux et al. (2015), *BinaryConnect* (popularizing STE in practice)
   - Jang et al. (2017) / Maddison et al. (2017), *Gumbel-Softmax / Concrete distributions* (stochastic relaxations; alternative to hard STE gating)

Our v42 choice is the “engineering-optimal” point for this repo: a minimal, transparent feasibility constraint that preserves gradient flow without introducing additional Lagrange multipliers, constraint critics, or high-variance estimators.

## Mathematical Framework

### Swing Option Valuation

The swing option pricing problem is formulated as the expected discounted value of net exercise payoffs:

$$
V_0 = \mathbb{E}\left[\sum_{t=0}^{T-1} e^{-r\,t\,\Delta t}\left(q_t \cdot \max(S_t - K, 0) - c_{\text{cost}} \, q_t^{\gamma_{\text{cost}}}\right)\right]
$$

Where:
- $q_t \in [q_{\min}, q_{\max}]$: Exercise quantity at time index $t$
- $S_t$: Spot price following the HHK model
- $K$: Strike price
- $r$: Risk-free rate
- $\Delta t = \text{maturity} / (n_{\text{rights}}-1)$: Time-step spacing between exercise opportunities
- $c_{\text{cost}}$, $\gamma_{\text{cost}}$: Convex exercise cost parameters (defaults $0$ and $1$ recover the no-cost baseline)
- Constraints: $\sum_{t=1}^T q_t \leq Q_{\max}$, refraction periods, etc.

### HHK Stochastic Process

The underlying spot price follows:

$$S_t = \exp(f(t) + X_t + Y_t)$$

With:
- **Diffusive OU component**: $dX_t = -\alpha X_t dt + \sigma dW_t$
- **Jump component**: $dY_t = -\beta Y_t dt + J_t dN_t$
- **Seasonal function**: $f(t)$ e.g., $\log(S_0) + A\cos(2\pi t)$
- **Jump process**: $N_t \sim \text{Poisson}(\lambda t)$, $J_i \sim \text{Exp}(1/\mu_J)$

### D4PG for Swing Options

The RL formulation treats swing option exercise as a continuous control problem:

- **State Space**: $[S_t/K, Q_{\text{exercised}}/Q_{\max}, \text{TTM}, X_t, Y_t, \ldots]$
- **Action Space**: Normalized exercise quantity $\tilde{q}_t \in [0,1]$
- **Reward**: $r_t = e^{-r\Delta t}\left[q_t \cdot \max(S_t - K, 0) - c_{\text{cost}} \, q_t^{\gamma_{\text{cost}}}\right]$
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

### Environment setup (macOS / Apple Silicon friendly)

The project is tested with **Python 3.11.13** and **PyTorch 2.7.1 (CPU)** from `conda-forge` inside a Miniforge/conda environment named `EP11`. The instructions below reproduce that stack and avoid the OpenMP conflicts that can appear when mixing Brew and pip packages.

```bash
# 0) Install Miniforge (if you have not already)
# https://github.com/conda-forge/miniforge

# 1) Create and activate the environment (Python 3.11, conda-forge only)
conda create -n EP11 -c conda-forge python=3.11
conda activate EP11

# 2) Core scientific + RL stack from conda-forge (keeps BLAS/OpenMP consistent)
conda install -c conda-forge \
    numpy=2.2.6 scipy=1.15.2 pandas=2.3.0 \
    matplotlib=3.10.3 seaborn=0.13.2 scikit-learn=1.7.1 \
    tqdm=4.67.1 gymnasium=1.0.0 tensorboard=2.20.0 \
    ipykernel=6.29.5 bootstrapped=0.0.2 scienceplots=2.1.1 plotly=6.1.2

# 3) Install the PyTorch CPU triplet from conda-forge (no pip wheels, no Brew libomp)
conda install -c conda-forge pytorch=2.7.1 torchvision=0.22.0 torchaudio=2.7.1

# 4) Optional extras
# QuantLib for FDM benchmarking
pip install QuantLib==1.39
# Snakeviz + numba for profiling/acceleration experiments
pip install snakeviz==2.2.2 numba==0.61.2

# 5) Register a Jupyter kernel (optional but convenient)
python -m ipykernel install --user --name EP11 --display-name "Python (EP11)"
```

Environment tips
- Always source packages from **conda-forge** inside this env. Avoid mixing with system Python or Brew `libomp`; the `conda-forge` build already ships `llvm-openmp 21.1.7`, which prevents duplicate-runtime crashes.
- Keep PyTorch, `libtorch`, and `llvm-openmp` in lockstep (`conda install --update-deps pytorch torchvision torchaudio llvm-openmp`).
- If you need GPU acceleration on another machine, use the matching CUDA build from `pytorch`’s channel but keep the same versions to maintain determinism.
- On Apple Silicon, set `OMP_NUM_THREADS` to the number of performance cores (usually 4) if you notice thread oversubscription, but it is optional with the above stack.

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
    --tau 0.002 \
    --lr_a 3e-4 \
    --lr_c 2e-4 \
    --optimizer adamw \
    --weight_decay_critic 1e-4 \
    --actor_layers 2 --critic_layers 2 \
    -layer_size 64
```

### Production Training

For publication-quality results, use the optimized configuration:

```bash
bash run.sh
```

This launches multiple seeds with 32,768 training episodes and comprehensive evaluation.

### Optimizer and Network Configuration

- **Optimizer selection**: Use `--optimizer adam` for legacy Adam or `--optimizer adamw` (default) for decoupled weight decay. Tweak decay with `--weight_decay_actor` (default `5e-5`) and `--weight_decay_critic` (default `1e-4`).
- **Network sizing**: `-layer_size` remains the global width knob (default 64 for a lightweight 2×64 baseline). Override per-network widths with `--actor_hidden_size` / `--critic_hidden_size`.
- **Depth control**: Set `--actor_layers` and `--critic_layers` (defaults: 2). The critic enforces at least two layers so that actions are merged after the state encoder.
- **Run diagnostics**: Model construction prints optimizer configuration plus actor/critic parameter counts so logs capture the exact architecture used in each run.
- **Activations**: Default hidden activation is **SiLU** (`--activation {relu, leaky_relu, silu}`), paired with a `tanh01` actor head (maps [-1,1] → [0,1]).

### Key CLI Flags (`run.py`)

Flags are grouped by what they primarily control; all defaults are chosen to reproduce the paper-style monthly swing setup and can be overridden from the command line.

- **Training loop & evaluation**
  - `-n_paths`, `-eval_every`, `-n_paths_eval`: number of training episodes, evaluation frequency, and evaluation sample size.  
  - `-seed`: RNG seed for env and networks.  
  - `--eval_batch_size`, `--eval_benchmark`, `--profile_eval`: batched evaluation throughput, evaluation-only runs, and cProfile of the eval path.

- **Swing contract & LSM benchmark**
  - `--strike`, `--maturity`, `--n_rights`, `--q_min`, `--q_max`, `--Q_min`, `--Q_max`, `--risk_free_rate`, `--min_refraction_periods`: financial contract and exercise/refraction constraints shared by RL and LSM/FDM.  
  - `--c_cost`, `--gamma_cost`: convex per-exercise cost term.  
  - `--lsm_basis`, `--lsm_degree`, `--lsm_reg`, `--lsm_reg_alpha`: LSM polynomial family, degree, and regularization.

- **HHK stochastic process**
  - `--S0`, `--alpha`, `--sigma`, `--beta`, `--lam`, `--mu_J`: parameters of the 2-factor OU-with-jumps HHK model used to generate spot paths.

- **Algorithm switches (D4PG/DDPG variants)**
  - `--device`: `"cpu"` or `"gpu"` (CUDA if available).  
  - `-per` plus `--per_alpha`, `--per_beta_start`, `--per_beta_frames`, `--per_alpha_final`, `--per_beta_final`, `--per_alpha_ramp_start`, `--per_alpha_ramp_end`, `--per_alpha_sigmoid`, `--per_priority_floor`, `--per_priority_clip_pct`: enable and schedule Prioritized Experience Replay.  
  - `-munchausen`: toggle Munchausen RL term in the critic target.  
  - `-iqn`: switch between standard critic and distributional IQN critic.  
  - `-nstep`, `--gamma`, `-t/--t`: N-step bootstrapping, discount factor, and Polyak target update rate.

- **Exploration noise**
  - Pre-squash Gaussian noise: `-noise_sigma0` (plateau std), `-noise_plateau` (episodes to hold σ0), `-noise_floor` (hyperbolic asymptote). Noise is added to the actor pre-activation, then squashed to [0,1].

- **Network architecture & optimization**
  - `-layer_size`: legacy width knob for both networks (default `64` → 2×64 MLPs).  
  - `--actor_hidden_size`, `--critic_hidden_size`: per-network widths overriding `-layer_size`.  
  - `--actor_layers`, `--critic_layers`: hidden depth (actor ≥1, critic ≥2).  
  - `--activation`: hidden-layer nonlinearity for actor/critic/IQN (`silu` default; `relu` or `leaky_relu` optional); actor head remains `tanh01`, critic/IQN heads remain linear.  
  - `--norm`: normalization layer in actor/critic/IQN MLPs (`layernorm` default; `rmsnorm` or `none`).  
  - `-lr_a`, `-lr_c`: actor and critic learning rates.  
  - `--optimizer`: `adam` or `adamw` (default).  
  - `--weight_decay_actor`, `--weight_decay_critic`: decoupled weight decay; norms and biases are excluded automatically.  
  - `--final_lr_fraction`, `--warmup_frac`, `--min_lr`: cosine-style LR warm-up and decay schedule controls.  
  - `--actor_grad_clip`, `--critic_grad_clip`, `--actor_grad_clip_type`, `--critic_grad_clip_type`, `--grad_clip_norm_type`: optional gradient clipping configuration.  
  - `--max_replay_size`, `--min_replay_size`, `-bs/--batch_size`: replay buffer capacity, warm-up length, and batch size.

- **System & performance**
  - `-n_cores`: cap on CPU cores used for PyTorch and environment stepping (defaults to all cores).  
  - `--compile`: opt-in `torch.compile` for actor/critic/IQN.  
  - `--fp32`: keep float32 default (1) or fall back to PyTorch’s default dtype (0).  
  - `--saved_model`: load a saved actor for evaluation-only or continued training runs.

### Exploration Noise (Pre-Squash)
- Actions: `a = 0.5*(tanh(u) + 1)`, with Gaussian exploration injected in `u` (pre-activation) then squashed to [0,1]; this keeps noise effective near bounds.
- Local sensitivity: `da/du = 0.5 * sech^2(u)`, so post-squash std ≈ `σ_a ≈ 0.5 * sech^2(u) * σ_u`. At `u=0`, `σ_a ≤ 0.5 σ_u`; as `|u|` grows, saturation dampens `σ_a`.
- Schedule (v34 baseline): plateau `σ_u = σ0 = 1.3` for 3,200 episodes, then hyperbolic decay toward `σ_floor = 0.32`  
  `σ_u(e) = σ_floor + (σ0 - σ_floor) / (1 + max(0, e - plateau) / plateau)`. v35 keeps the same shape with `σ_floor = 0.24`.

```mermaid
%%{init: {'theme':'base'}}%%
line
  title Pre-squash noise σ_u (plateau + hyperbolic decay)
  xAxis Episodes
  yAxis σ_u
  series
    label σ_u
    data 0:1.30, 3200:1.30, 10000:0.63, 20000:0.48, 32768:0.42
```

| Episode | σ_u (v34) |
| --- | --- |
| 1 | 1.30 |
| 3,200 (plateau end) | 1.30 |
| 10,000 | ≈0.63 |
| 20,000 | ≈0.48 |
| 32,768 | ≈0.42 |

## Algorithm Features

### Core D4PG Extensions

- **✅ Distributional Critic (IQN)**: Learn full return distributions for uncertainty quantification (optional, off by default)
- **✅ Prioritized Experience Replay**: Focus learning on important transitions
- **✅ Munchausen RL**: Entropy-regularized policy improvement for better exploration (optional, off by default)
- **✅ N-Step Bootstrapping**: Multi-step returns for faster value propagation
- **✅ Soft Target Updates**: Stable target network synchronization

### Swing Option Adaptations

- **Constraint Handling**: Automatic feasibility checking for all contract constraints
- **Refraction (Cooldown) Periods**: Use `--min_refraction_periods` to enforce a cooldown after exercise. For example, `--min_refraction_periods=1` blocks the immediate next period after any exercise. Applied consistently in both RL environment and LSM benchmark.
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

| Method           | Option Price | 95% CI   | Avg Exercise | Sample Size  |
|------------------|-------------|----------|--------------|--------------|
| **D4PG-RL**      | 2.660       | ±0.085   | 11.66        | 4,096 paths  |
| **LSM Benchmark**| 2.682       | ±0.089   | 11.35        | 4,096 paths  |
| **Relative Error** | -0.82%    | -        | +2.7%        | -            |

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
- **Actor**: 64-64 hidden layers with tanh activation (default baseline; increase `-layer_size` for larger models)
- **Critic**: 64-64 hidden layers with ReLU activation  
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
    "tau": 0.002,            # Target network soft update rate
    "layer_size": 64,        # Default actor/critic width (2×64 MLPs)
    "weight_decay_actor": 5e-5,
    "weight_decay_critic": 1e-4,
    "batch_size": 64,        # Mini-batch size
    "learn_every": 2,        # Learning frequency
    "buffer_size": 200000,   # Replay buffer capacity
}
```

### PER Scheduling (why and how)
- **Problem**: Static PER can over-focus early, spike TD errors, and drive action-variance collapse; turning PER off loses late-stage efficiency.
- **What works**: Keep PER effectively off early (alpha≈0, beta≈1) and ramp it in once the critic stabilizes. Moderate alpha (0.35–0.4) and beta (0.8–0.85) by ~14–15k episodes balanced stability and late refinement in our runs.
- **Schedules**: Linear ramps rise steadily; sigmoid ramps stay flatter early and rise smoothly mid-ramp. Sigmoid is preferred when you want to be “more uniform” in the first few thousand episodes.
- **Typical settings**: Start ramp ~3k–5k episodes, end ~14k–15k; alpha_final 0.35–0.4; beta_final 0.8–0.85; keep priority floor small (≈1e-6) and avoid clipping unless spikes appear. Adjust ramp start/end to control how long the run behaves like uniform replay before PER takes over.
### Exploration Noise Schedule

The Gaussian exploration noise now supports a plateau-then-decay schedule controlled by `--noise_plateau = N_p` episodes:

$$
\epsilon_t =
\begin{cases}
\epsilon_0, & t \le N_p \\
\epsilon_0 \cdot \epsilon_{\text{decay}}^{\,t - N_p}, & t > N_p
\end{cases}
,\qquad
\sigma_t = \max\left(\sigma_{\min},\; \sigma_0 \cdot \epsilon_t^{\alpha}\right)
$$

For the v17 runs we use a 2,000-episode plateau with $(\epsilon_0, \epsilon_{\text{decay}}, \sigma_0, \alpha, \sigma_{\min}) = (0.3,\ 0.99993,\ 1.1,\ 0.55,\ 0.15)$ so early exploration stays high before resuming the original decay curve.

![Plateaued Gaussian noise schedule](noise_plateau_schedule_v17.png)

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
@misc{drl_swing_options_2025,
    title={DRL-Swing-Options: Deep Reinforcement Learning for Swing Option Pricing},
    author={Alexander-Tsoskounoglou},
    year={2025},
    url={https://github.com/ithakis/DRL-Swing-Options},
    note={Research implementation of distributional deep RL for energy derivatives}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*This framework was developed for quantitative finance research and is intended for academic and professional use. Past performance does not guarantee future results. Please ensure compliance with relevant financial regulations when adapting for commercial use.*
