# Hyperparameter and Algorithmic Tuning Journey (v1 → v62)

A comprehensive record of algorithmic evolution for **D4PG-based swing option pricing** using the Hambly-Howison-Kluge (HHK) stochastic process.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Key Concepts & Deep Dives](#key-concepts--deep-dives)
3. [Version History](#version-history)
4. [Tuning Guidelines](#tuning-guidelines)
5. [Architecture Summary](#architecture-summary)

---

## Quick Reference

### Recommended Configuration by Regime

| Regime | Gate | Warmup Noise | Key Version | Final Delta% |
|--------|------|--------------|-------------|--------------|
| **Convex Costs** (`c>0`, `γ>1`) | ✅ Enabled | 0.3 | v61CC | 111.47% |
| **No Costs** (`c=0`) | ❌ Disabled | 0.4 | v62 | -0.37% |

### Core Hyperparameters (v61 Baseline)

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
| **PER** | `per_alpha` | 0.1→0.2 | Priority exponent (ramped) |
| | `per_beta` | 1.0→0.98 | IS correction |
| **Target** | `tau` | 0.0032 | Soft update rate |
| | `critic_warmup` | 1024 | Critic-only training episodes |

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

**Cosine Schedule**: warmup_episodes=1024, lr_schedule_episodes=40000

### PER

| Issue | Symptom | Fix |
|-------|---------|-----|
| Over-focus | priority_std rising, TD spikes | Lower per_alpha, add clipping |
| Under-focus | Slow late refinement | Increase per_alpha_final |
| Early instability | Critic loss spikes early | Start with low alpha, ramp later |

**Recommended Schedule**:
- Start: α=0.1, β=1.0 (near uniform)
- End: α=0.20, β=0.98
- Ramp: 5k→25k episodes

### Target Updates

| Parameter | Value | Notes |
|-----------|-------|-------|
| tau | 0.0032 | Moderate smoothing |
| target_policy_noise | 0.15→0.04 | Decay after ep 18k |
| target_policy_clip | 0.25 | Standard TD3 value |

### Critic Warmup

| Parameter | Value | Notes |
|-----------|-------|-------|
| critic_warmup_episodes | 1024 | Freeze actor for first 1024 eps |
| warmup_noise_fraction | 0.3→1.0 | Gradual ramp |

---

## Architecture Summary

### What Works Across All Versions

1. **Pre-squash noise** (v34+): Exploration in logit space
2. **PER with soft annealing** (v26+): α=0.1→0.2 over 25k episodes
3. **Cosine LR schedule** (v37+): Aggressive early, gentle late
4. **LayerNorm + SiLU** (v33+): Stable hidden representations
5. **Orthogonal initialization** (v43+): Consistent starting conditions
6. **Profitability gate** (v42+): Essential for convex costs

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

---

*Document last updated: v62 (March 2026)*