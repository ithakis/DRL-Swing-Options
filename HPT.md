# Hyperparameter and Algorithmic Tuning Journey (v1 → v63)

A comprehensive record of algorithmic evolution for **D4PG-based swing option pricing** using the Hambly-Howison-Kluge (HHK) stochastic process.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Key Concepts & Deep Dives](#key-concepts--deep-dives)
3. [Version History (v1–v62)](#version-history-v1v62)
4. [Phase 2: Post-v62 Research](#phase-2-post-v62-research)
   - [H1–H9: Semi-Analytical Bootstrap Hypotheses](#h1h9-semi-analytical-bootstrap-hypotheses)
   - [Function Approximator Comparison](#function-approximator-comparison-stage-ae)
   - [Group-0: Dead Code Removal](#group-0-dead-code-removal)
   - [v63: Config Simplification](#v63-config-simplification-stage-fg-screens)
5. [Tuning Guidelines](#tuning-guidelines)
6. [Architecture Summary](#architecture-summary)

---

## Quick Reference

### Recommended Configuration by Regime

| Regime | Gate | Warmup Noise | Key Version | Kernel | Final Delta% |
|--------|------|--------------|-------------|--------|--------------|
| **Convex Costs** (`c>0`, `γ>1`) | ✅ Enabled | 0.3 | **v63CC** (v61 base, no PER, no TPN) | `M_x=2` | +0.20±0.22 pp vs LSM |
| **No Costs** (`c=0`) | ❌ Disabled | 0.4 | **v63 no-cost** (v62 base, no PER, no TPN) | `M_x=2` | −0.30±0.43 pp vs LSM |

Note: Δ% is the agent price improvement over LSM at 4096 ep, 12 seeds. Paper results use 32768 ep.

### Core Hyperparameters (v63 Canonical)

| Category | Parameter | Value | Description |
|----------|-----------|-------|-------------|
| **Network** | `layer_size` | 64 | Hidden layer width (2×64 MLP) |
| | `activation` | SiLU | Hidden activation function |
| | `norm` | LayerNorm | Normalization layer |
| | `init_method` | orthogonal | Weight initialization |
| **Training** | `n_paths` | 32768 | Training episodes |
| | `batch_size` | 128 | Gradient update batch |
| | `lr_actor` | 1.6e-4 | Actor learning rate |
| | `lr_critic` | 9.0e-5 | Critic learning rate |
| **Exploration** | `noise_sigma0` | 1.30 | Initial pre-squash noise |
| | `noise_floor` | 0.26 | Minimum noise level |
| | `noise_plateau` | 3200 | Episodes at full noise |
| | `adaptive_noise_scale` | 0.6 | Pre-activation noise scaling |
| **Replay** | *(uniform)* | — | PER removed in v63 |
| **Target** | `tau` | 0.0032 | Soft update rate |
| | `critic_warmup` | 1024 | Critic-only training episodes |
| **Kernel** | `use_expected_target` | 1 | Semi-analytical bootstrap |
| | `kernel_M_x` | 2 | HHK quadrature X-axis points |
| | `kernel_M_per_k` | 1 | Nodes per jump count (Y-axis) |
| | `kernel_N_max` | 1 | Max jump count in mesh |

---

## Key Concepts & Deep Dives

### 1. Pre-Squash Exploration Noise (v34+)

**The Problem**: Post-squash noise (added after the activation) collapses near action boundaries. When `tanh(u) ≈ 1`, adding small noise doesn't change the output meaningfully.

**The Solution (v34)**: Apply Gaussian noise in **logit space** (before the squashing function):

$$
a = \text{squash}(u + \sigma \cdot \epsilon), \quad \epsilon \sim \mathcal{N}(0,1)
$$

**Key Parameters**:
- `noise_sigma0=1.3`: Initial noise scale
- `noise_floor=0.26`: Minimum noise (prevents zero exploration late)
- `noise_plateau=3200`: Episodes to hold initial noise before decay

**Outcome**: ✅ Major improvement. Eliminated boundary saturation ("bang-bang" collapse) while maintaining exploration throughout training.

---

### 2. Prioritized Experience Replay Scheduling (v25-v30)

**The Problem**: Static PER with high α causes over-focus on rare transitions early, destabilizing the critic. But uniform replay (no PER) slows late-stage refinement.

**The Solution (v26)**: Schedule PER from near-uniform to moderate prioritization:

| Phase | Episodes | α | β | Effect |
|-------|----------|---|---|--------|
| Early | 0–5k | 0.05 | 1.0 | Near-uniform (stable critic) |
| Mid | 5k–25k | 0.1→0.2 | 1.0→0.98 | Gradual prioritization |
| Late | 25k+ | 0.2 | 0.98 | Moderate PER (refinement) |

**Key Insight**: Priority statistics (`priority_mean`, `priority_std`, `priority_max`) are diagnostic—rising `priority_std` indicates over-focus on outliers.

**Outcome**: ✅ v26 achieved best stability and final delta. No collapse, fast convergence.

---

### 3. Action Output Activation (v33, v60-v61)

**Evolution of Actor Output Mapping**:

| Version | Activation | Formula | Saturation Point |
|---------|------------|---------|------------------|
| v1-v32 | sigmoid | $\sigma(u)$ | $\|u\| \approx 4$ |
| v33-v59 | tanh01 | $\frac{1}{2}(\tanh(u)+1)$ | $\|u\| \approx 2.5$ |
| v60-v62 | β-sigmoid | $\sigma(\beta \cdot u)$ | $\|u\| \approx 1.7$ (β=3) |

**Why β-sigmoid (v60+)**: Provides softer gradients near boundaries, preventing "sticky" policies that resist correction. With β=3.0, gradients remain ~4% at the saturation point vs <1% for tanh01.

**Outcome**: ✅ β-sigmoid restored healthy action variance and prevented gradient death.

---

### 4. Profitability Gate with STE (v42+)

**The Problem**: The environment masks unprofitable actions to `q=0`, but the actor still proposes them → wasted exploration, flat Q-regions, unstable learning.

**The Solution (v42)**: Add a **hard gate** in the actor that enforces profitability by construction:

$$
\Pi(q) = q \cdot \text{relu}(S - K) - c \cdot q^{\gamma}
$$

$$
q_{\text{out}} = \begin{cases} q & \text{if } \Pi(q) > 0 \\ 0 & \text{otherwise} \end{cases}
$$

**Straight-Through Estimator (STE)**: The gate is non-differentiable. Use detach trick:
```python
q_out = q_raw + (q_gated - q_raw).detach()
```
Forward uses `q_gated`, backward uses `q_raw` → gradients flow through the constraint.

**Key Benefits**:
1. Policy being trained = policy being executed (no mismatch)
2. Reduced wasted exploration on invalid actions
3. ~2× tighter seed-to-seed variance in final delta

**Outcome**: ✅ Core architectural improvement. Essential for convex cost regimes.

---

### 5. Actor Calibration & Warmup (v40-v43, v58)

**The Problem**: Untrained actors propose random actions, filling the replay buffer with low-quality data.

**Evolution of Calibration Methods**:

| Version | Method | Target | Outcome |
|---------|--------|--------|---------|
| v40 | Warmup | $\mathbb{E}[a] = Q_{\max}/n$ with std=0.05 | ⚠️ std too high |
| v41 | Warmup | Same with std=0.005 | ⚠️ Still variable |
| v43 | Q_T-approx | $\mathbb{E}[a] = \mathbb{E}[Q_T]/n$ | ✅ Better init |
| v58 | Rprop | Robust optimization | ✅ Stable calibration |

**Current Approach (v58+)**: Use Rprop (Resilient Propagation) for initial bias calibration—robust to Monte Carlo noise in the objective.

---

### 6. Critic Warmup & Gradient Protection (v59-v61)

**The Problem (v58 Failure)**: Seed 12 showed "action variance collapse"—policy saturated immediately and never recovered.

**Root Cause**: After calibration, training starts with an untrained critic providing "garbage gradients" that destroy the calibrated policy.

**The Solution (v59-v61)**:

1. **Critic Warmup** (`--critic_warmup_episodes=1024`): Freeze actor updates, train critic only for first 1024 episodes.

2. **Gradual Warmup Noise Ramp** (v61):
   $$
   \text{noise} = (\text{floor} + (1-\text{floor}) \cdot \frac{t}{T_{\text{warmup}}}) \cdot \sigma_0
   $$
   Ramps from 0.3× to 1.0× over warmup period.

3. **Adaptive Pre-Squash Noise**:
   $$
   \epsilon = \sigma \cdot (1 + k \cdot |u|) \cdot z, \quad z \sim \mathcal{N}(0,1)
   $$
   Scales noise with pre-activation magnitude to escape saturation.

**Outcome**: ✅ Eliminated variance collapse across all seeds.

---

### 7. Normalization: LayerNorm vs RMSNorm (v45, v47, v52)

| Norm | Speed | Stability | Outcome |
|------|-------|-----------|---------|
| LayerNorm | Baseline | ✅ Stable | Recommended |
| RMSNorm | +<1% faster | ❌ Heavier TD tails | Not recommended |

**Why LayerNorm Wins**: Mean-centering stabilizes hidden pre-activations when feature scales shift during training. RMSNorm only rescales magnitude, allowing activation mean drift → larger TD outliers → PER over-focus → policy instability.

**Lesson**: RMSNorm is attractive for throughput but increases tail-risk in this environment. Keep LayerNorm.

---

### 8. Initialization: Orthogonal vs He (v46, v53)

**The v52 Failure**: 2/9 seeds got stuck at -17.5% delta (hard collapse).

**The Fix (v53)**: Explicit orthogonal initialization with activation gain restored v43-like stability.

| Init Method | Seed Collapse Rate | Recommendation |
|-------------|-------------------|----------------|
| Orthogonal + gain | 0/9 | ✅ Use this |
| He (Kaiming) | Untested in isolation | — |
| Default/mixed | 2/9 collapsed | ❌ Avoid |

**Implementation**:
```bash
--init_method=orthogonal
```

---

## Version History

### Early Development (v1-v19)

#### v1-v4: Initial Baselines
- Larger networks, default noise/epsilon
- Occasional stuck policies, high variance
- **Outcome**: ❌ Unstable baseline

#### v5-v7: First Stable Baseline
- **v7**: 2×64 networks, n_step=1, PER on, batch 64
- Good convergence, delta% ~-2% to -4%
- **Outcome**: ✅ "Best baseline" for early experiments

#### v8-v13: Batch Size & PER Experiments
- Attempted PER clipping, batch 128
- PER over-focus flattened learning
- Safe-training interventions caused regressions
- **Outcome**: ❌ Batch 128 + aggressive PER unstable

#### v14-v19: Action Regularization & Target Smoothing
- Added action L2 regularization, target policy smoothing
- **v19**: noise_floor=0.15, sigma=1.2, plateau=2k
- Stable convergence (Average100 ~2.5-2.7), delta% ~-1% to -1.6%
- **Outcome**: ✅ Improved stability

---

### Exploration Refinement (v20-v26)

#### v20: Retuned Noise & LR
- noise_sigma=1.25, anneal_power=0.65, plateau=1500
- final_lr_fraction=0.8 (slower LR decay)
- **Outcome**: ✅ Balanced exploration/exploitation

#### v21-v22: Collapse Investigation
- v21 had 2/4 seeds collapse (Actions_at_upper_pct >40%)
- v22 softened PER (alpha=0.45), higher noise floor=0.22
- **Outcome**: ⚠️ Only 1/4 collapsed, but still unstable

#### v23-v25: Action Regularization Gating
- Applied action L2 only for first 4000 episodes
- **v25**: PER soft (alpha=0.2, beta_start=1.0), action_reg=3e-3 cutoff 3.5k
- **Outcome**: ⚠️ No collapse but early phase slower

#### v26: PER Scheduling Breakthrough ✅
- **Key Change**: Schedule PER from near-uniform (α=0.05) to moderate (α_final=0.4)
- Ramp: 5k→15k episodes
- **Delta%**: -0.5% to -1.1% (best so far)
- **Outcome**: ✅ Best stability and performance

---

### PER Refinement (v27-v32)

#### v27: Aggressive PER Ramp (Failed)
- Pushed PER softer early (α=0.05), higher noise (σ=1.35, plateau=3500)
- Sigmoid ramp 2.5k→14k, action_reg=4.5e-3 cutoff 4.5k
- **Result**: One seed collapsed, priority spikes, delta worse (-17.9)
- **Outcome**: ❌ Regression

#### v28-v29: Sigmoid PER Ramp Refinement
- v28: Reverted to v26 noise, α_final=0.35, stable but weaker delta
- v29: α_final=0.32, β_final=0.9, late delta improved but shy of v26
- **Outcome**: ⚠️ Better stability, not beating v26

#### v30-v32: Softer PER Experiments
- v30: α_final=0.30, regression vs v26 (delta -3.3 to -1.0)
- v31-v32: Various noise/PER combinations, led to v33 architecture changes
- **Outcome**: ⚠️ Minor variations, insight: need architectural changes not just tuning

---

### Architecture Evolution (v33-v44)

#### v33: SiLU + Calibrated Output Scaling
- Hidden activations: SiLU (was ReLU/tanh)
- Actor output: tanh01 head with calibrated bias/scale
- Target: $\mathbb{E}[a] \approx \mathbb{E}[Q_T]/n$
- Removed action regularization entirely
- **Outcome**: ✅ Faster convergence, tighter action variance

#### v34: Pre-Squash Exploration Noise ✅
- Noise applied in logit space before squashing
- Schedule: plateau (σ=1.3) for 3.2k episodes → hyperbolic decay
- Removed legacy OU/epsilon plumbing
- **Outcome**: ✅ Major improvement—preserved action variance without boundary saturation

#### v35-v36: Late PER & LR Tuning
- v35: per_alpha_final 0.32, noise_floor 0.24, lower LRs
- v36: Added priority clipping at 99.5th percentile, per_alpha_final=0.30
- **Outcome**: ✅ Improved late stability

#### v37-v38: Cosine LR Schedule
- **v37**: 5% warmup into cosine decay over 65k horizon
- **v38**: Explicit warmup_episodes=1024, slightly stronger late PER
- **Outcome**: ✅ Best late-stage convergence

#### v39-v41: Initialization Refinement
- **v40**: Warmup calibration targets std=0.05
- **v41**: Reduced std to 0.005, lower peak LRs
- **Outcome**: ⚠️ Helped early but still variable seeds

#### v42: Profitability Gate ✅
- **Key Innovation**: Hard gate in actor with STE
- Policy enforces $\Pi(q) > 0$ by construction
- Noise moved to pre-gate space
- **Results**: ~2× tighter seed variance (std 0.16% vs 0.33%)
- **Outcome**: ✅ Core architectural improvement

#### v43: Q_T-Approximate Warmup
- Calibration targets $\mathbb{E}[Q_T]/n$ using HHK-based approximation
- Better early Delta%, more stable policy stats
- **Outcome**: ✅ Improved initialization

#### v44: No-Clipping Test
- Removed all clipping (PER, gradient, target noise)
- **Result**: Destabilized—heavier TD tails, more oscillatory
- **Outcome**: ❌ Clipping is essential for stability

---

### Normalization & Initialization (v45-v53)

#### v45: RMSNorm Test
- **Result**: Faster (~<1%) but less stable
- Higher TD tails, more saturation, worse pricing convergence
- **Outcome**: ❌ RMSNorm increases instability

#### v46-v47: Initialization Experiments
- Compared He vs orthogonal with activation gain
- **Outcome**: Orthogonal preferred for consistency

#### v48: QMC Variance Reduction
- Seeded Sobol for OU driver, stratified jump draws
- Reduced across-seed error variance
- **Outcome**: ✅ Improved seed robustness

#### v49-v50: LAP (Loss-Adjusted Priorities)
- Used Huberized loss for PER priorities
- **v50**: κ=0.5, floor=0.05, clip=99.7
- **Outcome**: ⚠️ No clear pricing win, stability warning

#### v51-v53: LayerNorm Restoration
- **v52**: Switched back to LayerNorm from RMSNorm
- **v52 Issue**: 2/9 seeds collapsed at -17.5% delta
- **v53 Fix**: Explicit `--init_method=orthogonal`
- **Outcome**: ✅ v53 restored full stability

---

### Convex Cost Experiments (v54-v62)

#### v55: Initial Profitability Gate Restoration
- Re-applied rejection gate after target policy noise
- Cost-aware warmup calibration
- **First 80%**: ~18% delta (vs v56's ~28%)
- **Outcome**: ✅ Better than v56

#### v56: Projected Gate + Greedy Calibration
- Projection gate: `min(q, q_break_even)` + STE
- Greedy optimal calibration target
- **Issue**: Worse initial guess, slower recovery
- **Outcome**: ⚠️ Gate works but calibration inferior

#### v57: Stratified Sampling for HHK Paths
- Post-generation stratification by terminal spot $S_T$
- Every batch of 128 contains representative distribution
- **Outcome**: ✅ Reduced batch-to-batch variance

#### v58: Rprop Calibration
- Replaced Newton-based optimization with Rprop
- **Issue**: Seed 12 collapsed (action variance → 0)
- **Root Cause**: Calibration pushed into saturation, untrained critic destroyed policy
- **Outcome**: ⚠️ Calibration robust, but critic warmup needed

#### v59: Critic Warmup + Adaptive Noise
- `--critic_warmup_episodes=2048`: Freeze actor, train critic only
- `--adaptive_noise_scale=0.5`: Noise ∝ (1 + k×|u|)
- **Outcome**: ✅ Seed 12 recovery, general stability improved

#### v60: Target Noise Decay + β-Sigmoid
- `--warmup_noise_fraction=0.2`: Reduce noise during warmup
- `--target_noise_decay_start=15000`: Linear decay 0.15→0.02
- `--actor_output_activation=beta_sigmoid_2.0`
- **Results**: 
  - Late-stage std: **1.28%** (vs v59's 2.86%) ✅
  - Early convergence: Slower ❌
- **Outcome**: ⚠️ Better late, worse early

#### v61: Gradual Warmup Ramp ✅
- **Changes vs v60**:
  - Gradual noise ramp: 0.3× → 1.0× over warmup
  - critic_warmup: 1024 (was 2048)
  - β-sigmoid: β=3.0 (was 2.0)
  - target_noise_decay_start: 18000 (was 15000)
  - target_noise_floor: 0.04 (was 0.02)
  - adaptive_noise_scale: 0.6 (was 0.5)
- **Results**:
  - First 80%: ep 2048 (fastest) ✅
  - Late std: 2.59% (between v59/v60)
  - Action variance: healthy (0.068)
- **Outcome**: ✅ Best overall for convex costs

#### v62: Robust HHK Normalization
- **New**: `--use_robust_normalization=1`
  - Log-Moneyness: $\log(S/K)$
  - Median/IQR scaling for X_t, Y_t
- **Gate Experiment**: Disabling gate better for non-CC, gate needed for CC
- **Results**:
  - Non-CC: **-0.37%** delta (best ever) ✅
  - CC: 99.17% (regression) ❌
- **Outcome**: ⚠️ Regime-dependent; v61 for CC, v62 for non-CC

---

---

## Phase 2: Post-v62 Research

*Branches: `feat/semi-analytical-bootstrap` (from `main` at v62) → `refactor/simplify-config`*

This phase replaces the noisy single-sample TD bootstrap with an analytical expectation over the HHK transition kernel, tests nine structural hypotheses, screens function approximators, and systematically removes features whose sole rationale was noise-suppression.

**Priority order:** accuracy (Δ% over LSM) > seed-to-seed consistency > speed.  
**Removal criterion:** only challenge features that are (1) computationally expensive or (2) add significant C++ port complexity. Cheap features stay even if marginally useful.

---

### H1–H9: Semi-Analytical Bootstrap Hypotheses

**Setup:** All hypotheses use 4096 training episodes, kernel ON (fast M_x=2, M_per_k=1, N_max=1), 12 seeds, focal regime c=0.04, γ=2 unless noted. Δ% = (RL price / LSM price − 1) × 100.

---

#### H1: Expected-Critic Bootstrap ✅

Replace the single-sample TD bootstrap with an analytical expectation over the HHK transition kernel:

$$Q^{\text{target}}(s,a) = r + \gamma \cdot \mathbb{E}_{(X',Y') \sim p_{\text{HHK}}} \!\left[Q^{\pi_{\text{tgt}}}(s', \pi_{\text{tgt}}(s'))\right]$$

The expectation is approximated by M quadrature points sampled from the conditional distribution of (X', Y') given (X_t, Y_t), with weights from a Sobol-stratified mesh.

**Headline result** (focal c=0.04, γ=2, 4096 ep, 12 seeds):

| Condition | Δ% mean | Δ% std |
|---|---|---|
| Kernel ON (H1 alone) | +0.47 | 0.21 |
| Kernel OFF (baseline) | −2.05 | 0.44 |
| **Gap** | **+2.52 pp** | — |

No-cost regression: kernel +5.2 pp vs baseline. Welch p < 1e-10.

**Implementation:** `src/transition_kernel.py` + `--use_expected_target=1`.  
**Key constraint:** requires n_step=1 (bootstraps across one HHK step only).

---

#### H2–H3: Kernel Accuracy — M_x Isolation ✅

**Question:** Which quadrature axis governs accuracy?

M_x is the **sole controlling axis**. M_per_k and N_max are irrelevant conditional on M_x≥2 (Phase B: 6 seeds, 7 configs, all Welch p≥0.34).

| M_x | Relative variance | Welch p vs M_x=2 | Verdict |
|---|---|---|---|
| 1 | 290× higher (F-test p=8.8e-8) | — | ❌ collapses |
| 2 | 1× (anchor) | — | ✅ |
| 3 | ≈1× | p>0.90 | ✅ equivalent |
| 4 | ≈1× | p>0.84 | ✅ equivalent |
| 6 | ≈1× | p>0.58 | ✅ equivalent |

**Canonical fast option** (M=4 total, ~1.4× baseline wall-clock):
```
--kernel_M_x=2 --kernel_M_per_k=1 --kernel_N_max=1
```
**Quality option** (M=36, ~2× baseline): `--kernel_M_x=4 --kernel_M_per_k=4 --kernel_N_max=2`

---

#### H4: Critic Warm-Start ❌

Semi-analytical backward-induction initialisation for the critic network before RL training.

- **Result:** Neutral — did not help or hurt at 12 seeds.
- **Outcome:** ❌ Dead. `src/critic_warmstart.py` deleted.

---

#### H5: Dyna-Style Synthetic Experience Augmentation ❌

Use the kernel to generate synthetic next-state transitions for additional critic training steps.

- **Tested:** 4 orders of magnitude in augmentation rate λ.
- **Result:** No improvement at any λ.
- **Outcome:** ❌ Dead. `src/dyna_augment.py` deleted.

---

#### H6: Kernel-Expected IQN Distributional Target ❌

Apply the semi-analytical expectation over the IQN distributional critic (quantile regression).

- **Result:** Adds code complexity, no pricing gain. IQN was already disabled in v61.
- **Outcome:** ❌ Dead. IQN class and all distributional branches deleted.

---

#### H7: Twin Critics ❌

TD3-style twin-critic min-target combined with the kernel expectation.

- **Result:** Neutral at 12 seeds.
- **Outcome:** ❌ Dead. `critic_local_2`/`critic_target_2` sites deleted.

---

#### H8: Antithetic TD Target ❌

Average two antithetic-pair TD targets to cancel the signed HHK diffusion noise in the bootstrap.

- **Result:** Marginally worse on mean; confirmed dead at 12 seeds (including stratify-preserved antithetic variant).
- **Outcome:** ❌ Dead. Antithetic branches deleted.

---

#### H9: Jump Importance Weighting ❌

Re-weight TD targets by jump-event likelihood to reduce variance from rare compound-Poisson jumps.

- **Result:** No variance reduction; confirmed dead at 12 seeds.
- **Outcome:** ❌ Dead. Jump-IW branches deleted.

---

### Function Approximator Comparison (Stage A–E)

**Motivation:** With a deterministic TD target, the function approximator is the remaining lever for speed and C++ portability. Four approximators — each reducible to a single BLAS `gemv` — were introduced as drop-in replacements for the 2×64 SiLU+LN network.

| Approximator | Description | Updates/sec vs NN | Δ% vs NN | Verdict |
|---|---|---|---|---|
| `poly` | Chebyshev tensor-product basis | ~2.0× faster | worse | ❌ |
| `rff` | Random Fourier Features | ~1.45× faster | worse | ❌ |
| `rbf` | Radial Basis Function centers | ~1.6× faster | worse | ❌ |
| `tiny_nn` | 1-hidden-layer, no LN | ~1.6× faster | worse | ❌ |

**Finding:** All four lost to the 2×64 NN on accuracy across the 4-regime × 24-seed screen. The speed gain (~1.5–2×) does not compensate for the accuracy loss. The NN remains canonical (`--approximator nn`, default, bit-identical to v61).

**Correctness suite:** `pytest tools/test_approximators.py` (37 tests — gradcheck, shape, fitting, gate, kernel compatibility, bit-identical NN guard).

---

### Group-0: Dead Code Removal (refactor/simplify-config)

Features that were off in every paper config and confirmed inactive. Removal is provably zero-runtime-impact (all guards were `if False:`-equivalent):

| Feature | Action |
|---|---|
| IQN / distributional critic | Deleted — `IQN` class, `learn_distribution`, all distributional branches |
| Munchausen entropy bonus | Deleted — branches and arg |
| Twin critic (H7) | Deleted — `critic_local_2`/`critic_target_2` sites |
| Dyna augmentation (H5) | Deleted — `src/dyna_augment.py` already removed |
| Jump-IW (H9) | Deleted — branches and arg |
| Antithetic target (H8) | Deleted — branches and arg |
| Critic warm-start (H4) | Deleted — `src/critic_warmstart.py` already removed, 8 args removed |
| n-step > 1 path | Collapsed — kernel requires n_step=1; n_step arg retained |
| `critic_ema_decay` | Removed — already 0.0 |

**Verification:** `pytest tools/test_approximators.py` → 37/37 pass after each removal.

---

### v63: Config Simplification (Stage F/G Screens)

**Anchor configs (4 regimes, 12 seeds, 4096 ep, kernel ON fast M_x=2):**

| Tag | γ | c | Base config |
|---|---|---|---|
| `g1` | 1.0 | 0.04 | v61CC (gate ON, robust-norm ON) |
| `g15` | 1.5 | 0.04 | v61CC |
| `g2` | 2.0 | 0.04 | v61CC (focal) |
| `nocost` | 1.0 | 0.00 | v62 (gate OFF, warmup_noise_fraction=0.4) |

---

#### Stage G: Speed-Lever Screen (All Rejected)

Only `g2` and `nocost` ran at 3 seeds (triage); rejected before full expansion.

| Arm | Change | g2 Δ% | nocost Δ% | Verdict |
|---|---|---|---|---|
| `G_anchor` | — | −0.19±0.42 | −0.63±0.43 | baseline |
| `G_learn4` | learn_every=4 (skip every other step) | −0.11±0.35 | **−1.06±0.50** | ❌ nocost drops |
| `G_no_layernorm` | norm=none | **−1.46±0.31** | **−4.35±1.57** | ❌ catastrophic |
| `G_width32` | layer_size=32 | −0.14±0.83 | −1.29±0.54 | ❌ variance increases |

**Result:** No speed lever passes accuracy-first. LayerNorm is load-bearing (especially nocost). Even iso-wall-clock (6144 ep at learn_every=4 vs 4096 ep at learn_every=2), learn_every=4 loses accuracy.

---

#### Stage F: Complexity-Removal Screen (12 seeds × 4 regimes)

Decision rule: remove a feature iff Welch p≥0.05 **and** mean within ~0.3 pp of anchor **and** Levene p≥0.05 in **all 4 regimes**.

| Arm | Feature removed | g1 Δ%±std | g15 Δ%±std | g2 Δ%±std | nocost Δ%±std | Verdict |
|---|---|---|---|---|---|---|
| `F_anchor` | — | +0.22±0.22 | +0.21±0.23 | +0.18±0.18 | −0.30±0.43 | baseline |
| `F_no_per` | PER → uniform replay | +0.19±0.23 | +0.21±0.21 | +0.19±0.25 | −0.35±0.37 | ✅ **Remove** |
| `F_no_target_noise` | target_policy_noise=0 | +0.27±0.22 | +0.31±0.18 | +0.25±0.22 | −0.27±0.42 | ✅ **Remove** |
| `F_no_critic_warmup` | critic_warmup=0 | **−2.15±5.75** | +0.24±0.16 | +0.31±0.29 | −0.35±0.26 | ❌ g1 collapses |
| `F_minimal` | PER+noise+warmup | **−0.97±4.42** | +0.27±0.22 | +0.30±0.23 | −0.26±0.23 | ❌ g1 collapses |
| `F_minimal2` | PER+noise (warmup kept) | +0.20±0.26 | +0.28±0.19 | +0.20±0.22 | −0.30±0.43 | ✅ **All pass** |

**Decision summary:**

- **PER (Fenwick tree) → Removed.** Largest C++ port complexity. Neutral on all 4 regimes (Welch p>0.7 in all). The priorities-must-fight-noise rationale is gone with a deterministic target.
- **target_policy_noise → Removed.** TD3-style target smoothing. Zero rationale with a deterministic kernel target. Slightly *helps* mean when removed (the noise was adding bias). All 4 regimes pass.
- **critic_warmup → Kept.** g1 (low-cost, γ=1) collapses without it: Δ% drops from +0.22±0.22 to −2.15±5.75 (5 of 12 seeds blow up). Load-bearing in the hardest regime.
- **adaptive_noise → Kept.** 3-seed triage at nocost showed −5.2 pp collapse. Cheap to run; not worth risking.

**v63 canonical = `F_minimal2`:** v61CC/v62-nocost settings minus PER minus target-policy-noise.

---

#### Code Deleted for v63

**`src/replay_buffer.py`:** All Fenwick tree helpers (`_py_fenwick_update`, `_py_fenwick_rebuild`, `_py_fenwick_find_prefix_indices`, `_py_fenwick_batch_update`, numba wrappers), the entire `PrioritizedReplay` class, and PER-compat stubs.

**`src/agent.py`:** `per`/`per_*` params and assignments, `_compute_base_priorities`, `_maybe_update_per_schedule`, `_get_target_policy_noise`, `target_policy_noise`/`target_policy_clip`/`target_noise_decay_start`/`target_noise_floor` params, priority logic in `learn_()`, PER step-diagnostics in `step()`, dead `calculate_huber_loss` and `calc_fraction_loss`.

**`src/transition_kernel.py`:** `target_policy_noise` param removed from `expected_critic_target`; both in-loop noise/gate blocks removed.

**`run.py`:** 13 `per_*` argparse args, `--target_policy_noise`, `--target_policy_clip`, `--target_noise_decay_start`, `--target_noise_floor`, corresponding Agent() kwargs.

**`evaluate_saved_agent.py` / `tools/rebuild_results_v7.py`:** Per-compat kwargs removed; old per=1 JSONs eval cleanly via `**kwargs` absorption.

**After deletion:** `pytest tools/test_approximators.py` → 37/37 pass. Kernel-on smoke (1024 ep, cc_g2) → runs clean.

---

#### CLAUDE.md Note — critic_warmup with kernel

CLAUDE.md previously stated `--critic_warmup_episodes=0 # not needed with kernel`. This is **false for g1 (γ=1, c=0.04)**. The correct guidance:

- g2/nocost: critic_warmup removable (kernel stabilises the target)
- **g1: critic_warmup required** — without it g1 collapses

**Update (v63 audit):** the value was retuned from 1024 → **512** (see "Critic-warmup investigation"
below). Retain `--critic_warmup_episodes=512` in all canonical v63 configs; never set it to 0 for g1.

---

#### v63 Runtime Feature Audit (post-Stage-F)

A read-the-code audit of the *surviving* runtime features surfaced three latent issues and confirmed the
rest should stay. Priority: accuracy > seed-consistency > speed; remove only if expensive or a C++-port
burden.

**F1 — LR schedule is inert under `final_lr_fraction=1.0`.** `lr_lambda` returns `1.0` and both schedulers
are `None` (`src/agent.py`), so the v63 canonical config runs a **constant LR (3e-4), with no warmup ramp
and no cosine decay**. Consequences:
- `--lr_schedule_episodes` does nothing unless `final_lr_fraction<1` (it is *conditionally*-active, not dead
  code — keep the arg).
- `--warmup_episodes` does **not** ramp the LR, but it is **reused as the `calibrate_bias` Rprop batch
  size** (`run.py`, `n_episodes=args.warmup_episodes`) — so it is still load-bearing, just not for LR.
- This is the deliberate v63 simplification vs v62 (`lr_a=1.6e-4, lr_c=9e-5, final_lr_fraction=0.20,
  lr_schedule_episodes=40000`). Constant LR is the most C++-port-friendly choice → **keep**.

**F2 — Gradient clipping removed (was dead code).** The clip args were overwritten to `None` in
`Agent.__init__` and `learn_()` never called `clip_grad_norm_`, so *every* v63 result is clip-free. The
five `--*_grad_clip*` args + the Agent params were deleted from `run.py`/`src/agent.py`. (The v44
"removing clipping destabilized" finding predates the deterministic kernel target and no longer applies.)

**F3 — Critic optimizer stepped twice per `learn_()` (effective ~2× critic LR) — RESOLVED.** A legacy
duplicate `self.critic_optimizer.step()` ran on the same un-zeroed grad. A 12-seed × 3-regime screen
(below) showed dropping it (single step, `lr_c` unchanged at 3e-4) **strictly beats** the double-step on
mean Δ%, seed std, AND worst-case in every regime. **Single step is now the default** (`single_critic_step=1`).

**Audit verdicts:** DELETE = grad clips (F2). FIX = double critic step (F3 → single-step default).
LEAVE (cheap/correct/zero-port-cost) = noise schedule (plateau correctly sized for the 32768-ep paper
horizon, not the 4096 screen), adaptive_noise=0.6, warmup_noise_fraction, tau, batch_size, weight_decay,
LayerNorm, constant LR. EDIT-tested then **rejected** (neutral / not worth it): **E3** tau↑0.005 (neutral),
**E5** learn_every=1 (unconfirmed gain, ~2× wall-clock). Validation harness: `tools/sweep_v63_audit.py`.

#### Validation campaign (`tools/sweep_v63_audit.py`, kernel-on fast M_x=2, 4096 ep)

**Wave 1 — 3-seed screen {11,12,13}, 36 runs.** No regression, nothing significant (all Welch p≥0.17);
greenlit a 12-seed confirmation of the two high-value changes (E1 warmup-reduction, E4 single-step) and
dropped E3/E5 as neutral.

**Wave 2 — 12-seed confirmation {11–22}, 132 runs.** Every E4 and E1 arm beat its anchor on mean **and**
std **and** worst-case seed, with zero blow-ups (Δ% mean ± std, worst-seed):

| Regime | anchor (double-step) | single-step `lr_c=3e-4` | single-step `lr_c=6e-4` | E1 warmup512 |
|---|---|---|---|---|
| cc_g1 | +0.247 ± 0.282 (−0.143) | +0.296 ± 0.142 (+0.118) | +0.375 ± 0.212 (−0.054) | +0.281 ± 0.190 (+0.036) |
| cc_g2 | −0.038 ± 0.408 (−0.804) | +0.152 ± 0.271 (−0.254) | +0.042 ± 0.322 (−0.435) | — |
| nocost | −0.063 ± 0.382 (−0.832) | +0.013 ± 0.254 (−0.267) | +0.170 ± 0.135 (−0.030) | — |

Direction is consistent across all E4 cells (6/6 positive mean Δ, 6/6 tighter std, 6/6 better worst-case;
sign-test p≈0.016 each) — the double-step was mildly *harmful* (critic over-stepping). **Adopted: single
step + `lr_c=3e-4`** (minimal change; best cc_g2, tightest cc_g1, best cc_g2 tail). `lr_c=6e-4` is a viable
alternative (higher mean cc_g1+nocost, tightest nocost) but adds a second change and a worse cc_g2 tail.
E1 critic_warmup 1024→512/256 validated safe on cc_g1 (no blow-up, slightly tighter) but low-stakes →
folded into the separate **critic-warmup investigation** (W-family) run under the new single-step default.
CSVs archived: `logs/_sweep_v63_audit/sweep_v63_audit_{3seed,12seed}.csv`.

#### Critic-warmup investigation (W-family, under the single-step default)

Three questions: (Q1) is warmup still needed? (Q2) is 1024 correctly tuned? (Q3) can retuning another
parameter make it obsolete? cc_g1 is the binding regime; cc_g2/nocost are insensitive to warmup at all
levels (0/256/512 all fine). **12-seed cc_g1 confirmation:**

| critic_warmup | Δ% mean ± std | worst seed | blowups (<−2%) |
|---|---|---|---|
| 1024 (old) | +0.296 ± 0.142 | +0.118 | 0 / 12 |
| **512 (new canonical)** | **+0.337 ± 0.180** | +0.076 | 0 / 12 |
| 256 | +0.311 ± 0.197 | +0.013 | 0 / 12 |
| **0** | **−5.764 ± 7.521** | −14.526 | **5 / 12** (Welch p=0.018) |

- **Q1 — needed: YES.** `warmup=0` collapses 5/12 cc_g1 seeds to ~−14%. The single-step fix did **not**
  remove the dependency.
- **Q2 — over-tuned: YES.** 256 and 512 are both collapse-free at 12 seeds and *beat* 1024 on mean (fewer
  frozen episodes → more actor improvement). Retuned to **512** (best mean, safest worst-case; 256's worst
  seed +0.013 sits too close to the collapse cliff). No wall-clock change (warmup episodes still run critic
  updates; only the actor update is skipped).
- **Q3 — substitutable: NO.** At `warmup=0`, a gentler actor LR did not reliably prevent collapse
  (`lr_a=1.5e-4` collapsed; `lr_a=2e-4`'s 3-seed survival was a fluke — its gentler sibling blew up, so the
  collapse is stochastic, not LR-ordered). Warmup is structural: it is the only mechanism that lets the
  critic converge on the calibrated policy before *any* actor movement, and with constant LR there is no
  ramp to lean on (F1). **Keep the warmup branch.**

CSVs archived: `sweep_v63_audit_warmup{3seed,12seed}.csv`.

**v63 audit net result:** removed grad clips (F2, dead), fixed the double critic-step (F3 → single-step
default, strictly better), retuned critic_warmup 1024→512 (accuracy + consistency, collapse-safe). Rejected:
tau↑, learn_every=1, warmup removal, lr_a-substitute-for-warmup. Everything else (noise schedule,
adaptive_noise, LayerNorm, tau, batch_size, weight_decay, constant LR) left as cheap/correct/well-motivated.

---

#### Price-vs-#paths convergence study (pure actor-NN evaluation, no M_x)

**Motivation.** (1) Quantify how the evaluated option price and its Monte-Carlo std behave as a function of
the number of evaluation paths; (2) confirm that *evaluation* of a trained agent is a pure actor-NN rollout
and never depends on the training-time kernel quadrature axis `M_x`.

**M_x is training-only — confirmed by code audit.** `src/agent_evaluation.py` (`evaluate_agent` →
`_evaluate_swing_agent`) rolls the actor forward on out-of-sample episodes via `agent.act(state, add_noise=
False)` and averages per-path discounted returns. There is **zero** reference to the kernel, `M_x`,
`use_expected_target`, or any quadrature object in the evaluation path. `M_x` enters only the critic's TD
*target* during training (`src/transition_kernel.py`). So the reported price is a clean policy-rollout
estimate; raising/lowering `M_x` cannot change what evaluation measures, only how the critic was trained.

**Design.** 12 canonical agents (focal cc_g2: c=0.04, γ=2; single-step, critic_warmup=512, kernel fast
M_x=2, β-sigmoid(3.0), robust-norm), seeds 11–22, each evaluated on a **shared 131,072-path** out-of-sample
pool (eval seed 999). 30 log-spaced path counts from 1 → 32,768 (denser at low N). Per-N uncertainty band =
12 seeds × bootstrap (BOOT=400) over the shared returns matrix. LSM benchmark trained on 16,384 paths,
priced on the same pool; its 95% band is the bootstrap CI of the mean (from `lsm_swing_pricer`'s
`np.percentile(boot_means,[2.5,97.5])`). Harness: `tools/price_vs_paths.py` (`--skip-train`, `--force-eval`;
returns cached at `logs/_price_vs_paths/returns_matrix.npy`).

**Results.**

| Quantity | Value |
|---|---|
| RL converged price | **1.9665** (seed range 1.9619–1.9702, ≈0.4% spread) |
| LSM (~131k paths) | **1.9685** ± 0.013 (95% CI) |
| **Δ% RL vs LSM** | **−0.10%** (inside the LSM CI — statistically indistinguishable) |

- The price-estimate std is a textbook **1/√N** line (bottom panel matches the gray reference slope) → the
  estimator is unbiased-variance-dominated, no pathological heavy tail.
- At very low N the band fills the frame and the *median* sits below the converged price: the per-path
  payoff is right-skewed (mostly zeros + occasional large exercises), so small-sample means skew low. This is
  a sampling artifact, not policy bias — it vanishes by N≈1k–10k.
- The **seed-to-seed dispersion (~0.4%) is the irreducible floor**, not MC noise: it does not shrink with N.
  Beyond ~10k eval paths, adding paths buys nothing; reducing seed variance is the only remaining lever on
  reported-price tightness.

**Figure:** `logs/_price_vs_paths/price_vs_paths.{png,pdf}` — 2-panel, publication-quality (price vs N with
RL 95% band + converged line + LSM line/band; std vs N log-log with 1/√N reference). Pending: move to
`Paper/figs/` if adopted for the manuscript.

#### Deterministic-target retune (Tasks 1–3, post-v63 audit)

With the kernel TD target deterministic, three sequenced pieces of work were run under the standard
accuracy>consistency>speed gate (3-seed screen {11,12,13} → 12-seed confirm {11–22}; regimes cc_g1
[γ=1,c=.04, binding], cc_g2 [focal], nocost; kernel-on fast M_x=2, 4096 ep; metric Δ% mean ± seed-std
+ worst-seed + blow-ups<−2%). Harness: `tools/sweep_v63_audit.py` (`CB`/`N`/`L`/`C`/`LR` arm families;
new `--resume` flag makes interrupted sweeps recoverable). CSVs archived under
`logs/_sweep_v63_audit/sweep_v63_audit_{CB,T2,LR}_{3,12}seed.csv`.

**Task 1 — closed-form `calibrate_bias` (replaces the 20-iter Rprop loop).** The old warm-start ran
~60 full-batch price rollouts to tune the scalar output bias. Replaced with an O(1)-pass myopic
warm-start from the convex-cost FOC `q*(S)=clip(((S−K)₊/(c·γ))^{1/(γ−1)}, q_min, q_max)` (γ=2 →
(S−K)₊/2c; γ=1,c>0 / c=0 → bang-bang), averaging the normalized optimum over the warmup dataset spots
(samples of the HHK/kernel density), budget-capped by `Q_max/n_rights`, then one variance-scale + one
bias shift via local squash inversion (`src/agent.py` `_calibrate_bias_closed_form` / `_output_slope`).
Guarded by `--calibrate_bias_mode {closed_form,rprop}`, default **closed_form**.

| 12-seed | cc_g1 | cc_g2 | nocost | calib time |
|---|---|---|---|---|
| closed_form (new) | +0.326 ± 0.208 (−0.036) | +0.118 ± 0.278 (−0.435) | −0.117 ± 0.169 (−0.334) | **0.064 s** |
| rprop (legacy) | +0.337 ± 0.180 (+0.076) | +0.173 ± 0.177 (−0.132) | −0.195 ± 0.303 (−0.675) | 0.611 s |

Verdict: **statistical wash** (all Welch p≥0.44, zero blow-ups); closed_form wins nocost, ties the
binding cc_g1, marginally soft on cc_g2 — all within noise. Calibration **9.5× faster** (but only
~0.3% of a ~190 s run, so the real win is simplicity + determinism + C++-portability). **Adopted** on
those grounds.

**Task 2 — scheduler simplification (two screens, then combine).** *Screen A (noise):* the deterministic
target made σ0=1.30 over-exploration. *Screen B (LR):* `--weight_averaging {off,ema}` (eval-only EMA of
actor weights; default off), and re-enabled LR decay (cosine/linear) as a test. 3-seed screens → 12-seed
confirmation of the combined winner **`N_full_ema`** = linear σ0→floor over the full horizon
(`noise_schedule=linear`, `noise_plateau=0`) + eval-only EMA, constant LR:

| 12-seed | cc_g1 | cc_g2 | nocost |
|---|---|---|---|
| anchor (hyperbolic noise, no EMA) | +0.326 ± 0.208 (−0.036) | +0.118 ± 0.278 (−0.435) | −0.117 ± 0.169 (−0.334) |
| **N_full_ema (adopted)** | +0.333 ± 0.195 (+0.036) | **+0.483 ± 0.215 (+0.121)** p=.002 | −0.000 ± 0.177 (−0.350) |

Verdict: **significant focal-cc_g2 win (+0.37pp, Welch p=0.002)**, nocost lifted to ~0, cc_g1 tied,
tighter std, worst-seeds blow-up-free — EMA specifically tightens the looser nocost that fulldecay
introduces. **LR decay was neutral-to-harmful at 4k** (3-seed: cosine drifted negative; linear
decay-to-0 FAILED nocost) → **constant LR confirmed; the cosine/linear `lr_lambda` branches +
`--lr_decay_shape`/`--lr_warmup_episodes` were DELETED.** `final_lr_fraction`/`lr_schedule_episodes`/
`min_lr` remain accepted but inert (constant LR only).

**Task 3 — LR magnitude re-tune (on the locked schedule).** 2-D grid `lr_c∈{2,3,4.5,6}e-4 ×
lr_a∈{1.5,3}e-4`, 3-seed → 12-seed escalation of the `lr_c=6e-4` cells. The critic wants to be ~2×
faster under the deterministic single-step target:

| 12-seed | cc_g1 | cc_g2 | nocost |
|---|---|---|---|
| anchor (lr_c=3e-4) | +0.333 ± 0.195 (+0.036) | +0.483 ± 0.215 (+0.121) | −0.000 ± 0.177 (−0.350) |
| **lr_c=6e-4, lr_a=3e-4 (adopted)** | **+0.375 ± 0.181 (+0.099)** | +0.449 ± 0.207 (**+0.203**) | **+0.091 ± 0.133 (−0.090)** |

Verdict: **better worst-seed AND tighter std in all three regimes** (the consistency lever), higher
mean on cc_g1+nocost, average Δ% 0.272→0.305; cc_g2 mean dips a trivial insignificant −0.034 (p=0.70,
offset by its better worst-case). `lr_a=3e-4` beat `lr_a=1.5e-4` on mean+worst+std everywhere.
**Adopted lr_c=6e-4, lr_a=3e-4.**

**Net canonical (kernel-on) after Tasks 1–3:** closed-form warm-start · linear noise decay
(`noise_schedule=linear`, `noise_plateau=0`) · eval-only EMA (`weight_averaging=ema`, `ema_decay=0.999`)
· constant LR `lr_a=3e-4`/`lr_c=6e-4`. Deleted: Rprop warm-start (now opt-in), the cosine/linear
LR-decay branches and their flags. All changes flag-guarded to the prior default elsewhere;
`pytest tools/test_approximators.py` stays 37/37.

---

## Tuning Guidelines

### Exploration (Noise/Epsilon)

| Problem | Symptom | Fix |
|---------|---------|-----|
| Boundary saturation | Actions_at_upper_pct > 40% | Increase noise_sigma0, noise_floor |
| Too much noise | High Average100 variance | Reduce noise_sigma0, extend plateau |
| Early collapse | Action variance → 0 | Add critic warmup, increase noise_floor |

**Recommended Values**:
- `noise_sigma0`: 1.2-1.4
- `noise_floor`: 0.20-0.30
- `noise_plateau`: 2500-3500 episodes

### Learning Rates

| Batch Size | lr_actor | lr_critic | final_lr_fraction |
|------------|----------|-----------|-------------------|
| 64 | 3e-4 | 1.6-1.8e-4 | 0.8 |
| 128 | 1.6-2.0e-4 | 0.9-1.1e-4 | 0.2-0.3 |

**Cosine Schedule**: only active when `final_lr_fraction<1` (e.g. v62: warmup_episodes=1024,
lr_schedule_episodes=40000). **v63 canonical uses `final_lr_fraction=1.0` ⇒ constant LR, no
warmup/cosine** (see F1 in the v63 Runtime Feature Audit). `warmup_episodes` then only sizes the
`calibrate_bias` batch.

### Replay Buffer

PER was **removed in v63**. The replay buffer is now uniform (`CircularReplayBuffer`). No PER tuning is required or applicable. The Fenwick-tree priority infrastructure is gone from the codebase.

### Target Updates

| Parameter | Value | Notes |
|-----------|-------|-------|
| tau | 0.0032 | Moderate smoothing |
| target_policy_noise | *(removed v63)* | Was 0.15→0.04; redundant with deterministic kernel target |
| target_policy_clip | *(removed v63)* | Was 0.25 |

### Critic Warmup

| Parameter | Value | Notes |
|-----------|-------|-------|
| critic_warmup_episodes | 512 | Freeze actor for first 512 eps (v63 audit: retuned from 1024; required for g1, 0 collapses 5/12 seeds) |
| warmup_noise_fraction | 0.3→1.0 | Gradual ramp |

---

## Architecture Summary

### What Works Across All Versions

1. **Pre-squash noise** (v34+): Exploration in logit space
2. **Uniform replay** (v63): PER removed — deterministic kernel target makes priority-weighting redundant
3. **Cosine LR schedule** (v37+): Aggressive early, gentle late
4. **LayerNorm + SiLU** (v33+): Stable hidden representations; load-bearing in nocost regime
5. **Orthogonal initialization** (v43+): Consistent starting conditions
6. **Profitability gate** (v42+): Essential for convex costs; must be OFF for no-cost
7. **Semi-analytical TD bootstrap** (v63+): `--use_expected_target=1`, M_x=2 — +2.52 pp over single-sample baseline
8. **Critic warmup** (v59+): Required for g1 regime (γ=1, c>0); 1024 episodes minimum

### Regime-Specific Settings

```python
if c_cost == 0:  # No costs (bang-bang optimal)
    disable_actor_gate = True
    warmup_noise_fraction = 0.4
    # Use v62 settings
else:  # Convex costs (continuous optimal)
    disable_actor_gate = False
    warmup_noise_fraction = 0.3
    # Use v61 settings
```

### Performance Summary (v59-v62)

| Version | CC Final% | CC Std | Non-CC Final% | Non-CC Std | Key Change |
|---------|-----------|--------|---------------|------------|------------|
| v59 | 111.63% | 2.86% | -1.30% | 0.08% | Critic warmup + adaptive noise |
| v60 | 111.93% | **1.28%** | -0.70% | 0.50% | β-sigmoid + target noise decay |
| v61 | 111.47% | 2.59% | -1.00% | 0.57% | Gradual warmup + β=3.0 |
| **v62** | 99.17% ❌ | 5.23% | **-0.37%** ✅ | **0.17%** | Gate disabled |

### Future Directions

1. **Conditional gate**: Enable gate only for CC automatically
2. **Cost-aware actor**: Add c_cost to observation
3. **Dual-policy training**: Separate actors for CC vs non-CC
4. **Hindsight Experience Replay**: Learn from masked actions as negative examples

---

## Appendix: Lessons Learned

| Attempt | Version | Result | Lesson |
|---------|---------|--------|--------|
| Remove action regularization | v33 | ✅ | L2 on actions hurts exploration |
| Pre-squash noise | v34 | ✅ | Key innovation for continuous actions |
| Calibrated warmup | v40 | ⚠️ | Helped early, but std=0.05 too high |
| β-sigmoid activation | v60 | ✅ | β=2.0 insufficient, β=3.0 optimal |
| Disable actor gate | v62 | ⚠️ | Works for non-CC, fails for CC |
| RMSNorm | v45 | ❌ | Increases TD tail risk |
| Remove clipping | v44 | ❌ | Clipping essential for stability |
| LSM terminal step | — | ✅ | Gate on net profitability (`payoff_net > 0`), not ITM, when `c > 0` |
| Semi-analytical bootstrap | H1 | ✅ | +2.52 pp gap vs noisy TD; M_x sole axis |
| M_x=1 kernel | H2 | ❌ | 290× variance increase; collapses |
| M_x≥2 kernel | H2 | ✅ | All equivalent; plateau at M_x=2 |
| Critic warm-start | H4 | ❌ | Neutral at 12 seeds; code deleted |
| Dyna augmentation | H5 | ❌ | Dead across 4 orders of magnitude |
| IQN distributional | H6 | ❌ | Already off; code deleted |
| Twin critics | H7 | ❌ | Neutral; code deleted |
| Antithetic TD target | H8 | ❌ | Marginally worse; code deleted |
| Jump importance weighting | H9 | ❌ | No variance reduction; code deleted |
| Approximators (poly/rff/rbf/tiny_nn) | Stage A–E | ❌ | 1.5–2× faster updates; accuracy loss → NN stays |
| Remove PER | Stage F | ✅ | Neutral on all 4 regimes; Fenwick deleted |
| Remove target_policy_noise | Stage F | ✅ | Neutral; slightly improves mean |
| Remove critic_warmup | Stage F | ❌ | g1 collapses (−2.15±5.75 pp) |
| Remove LayerNorm | Stage G | ❌ | Catastrophic in nocost (−4.35 pp) |
| learn_every=4 | Stage G | ❌ | nocost drops −1 pp even iso-wall-clock |

---

*Document last updated: v63 (May 2026)*