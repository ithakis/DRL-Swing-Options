# Hyperparameter Tuning Journey (v1 → v20)

This document summarizes the evolution of the hyperparameters and algorithmic tweaks across versions 1–20, what worked, what failed, and how to tune the key knobs for swing option pricing. It also notes parameters we tried to keep constant and why.

---

## Update (v33+)

- Action regularization (`action_reg_weight`, `action_reg_cutoff`) has been removed from the codebase; references below are kept only as historical notes for pre-v33 runs.
- Hidden activations now default to **SiLU**; actor outputs use the **tanh01** head (maps [-1, 1] → [0, 1]). A 1,024-episode warmup pass estimates the untrained policy mean/std and shifts/scales the output layer so `E[action] ≈ E[Q_T] / n_rights`, where `E[Q_T]` is approximated analytically from the HHK model (Beta–Binomial fit for the correlated ITM count; see `approximate_Q_T` in `src/swing_env.py`). This prevents initial saturation/masking without any action L2.
- **v33 vs. v26 (4 seeds each, green vs. gray in the attached TensorBoard plots)**:
  - Pricing/Delta_Percent: v33 climbs out of the initial drawdown much faster and is less volatile through the first ~5k episodes; finals cluster in the -1.2 to -2.6% band (v26: -0.5 to -2.2% with one slower seed). Average100 ramps earlier for v33, converging to ~2.5–2.6.
  - Action variance: Action_variance_mean stays tighter (~0.16–0.17 vs. v26 drifting above 0.19), Actions_at_upper_pct is visibly lower, and Actions_at_lower_pct remains at 0 (no lower-bound collapse). Target_drift decays faster for v33.
- Exercise behavior: Avg_Exercise_Count stabilizes around 9–10 vs. v26’s 6–8; Avg_Total_Exercised holds ~11.5–12.2 (v26 sits lower early and recovers slowly).
- Loss/TD metrics: Actor_loss declines more smoothly; TD_Error percentiles and PER priorities rise steadily without the spikes seen in weaker v26 seeds, indicating healthier replay focus.
- Housekeeping: keep runvXX scripts identical except for parameter/value changes and version labels; preserve the comment blocks when copying to new versions.
- Housekeeping (inline comments): carry over inline comments that annotate params in runvXX scripts and update them when values change; avoid dropping them between versions.
- Mid-Late Training Dynamics (open): Having solved the early stage dynamics I then wanted to push the algorithm to the limits from around 15k episodes which it seems to plateau or even worsen. I have understood that I need to tune the interaction between exploration (normal noise added to the actor NN output), the priority in the prioritized experience replay network, and the learning rate of the networks. This remains open and will spend a couple of days to better understand the interaction.

## v34: Pre-squash exploration noise
- Change: Exploration noise is applied in pre-squash logit space; actor outputs tanh01 → [0,1]; schedule = plateau (`noise_sigma0=1.3`) for 3.2k episodes then hyperbolic decay toward `noise_floor=0.32`. Legacy OU/epsilon/noise_type plumbing was removed; TD3 target smoothing is unchanged.
- Motivation: Preserve healthy action variance without boundary saturation from post-squash noise collapse.
- CSV evidence (v34CSVResults; delta CSV for seed 12 unavailable):  
  - Delta_Percent (seeds 11/13/14): at ~20k → [-1.7, -0.9, -2.7], end → [-1.0, -0.4, -1.3]; std shrank 0.74 → 0.37.  
  - Average100 (seeds 11/12/13/14): at ~20k → [2.79, 2.88, 2.43, 2.37], end → [2.63, 2.74, 2.59, 2.51]; std shrank 0.22 → 0.08; seeds 11/12 drifted down after 20k while 13/14 rose.
- Observed dynamics (TensorBoard/logs): action_variance_mean climbs past 0.2 late with Actions_at_upper_pct rising; PER priority_std re-accelerates after ~400k; target_drift lowest so far; TD percentiles rise smoothly but diverge per seed mid/late.
- Likely causes of post-20k divergence (ranked):  
  1) PER outlier focus late (alpha_final=0.4, beta_final=0.8) → rising priority_std and seed-dependent TD spikes.  
  2) High pre-squash noise floor (0.32) sustains sizable post-squash noise, amplifying action variance near bounds.  
  3) Actor/critic LR (2.2e-4 / 1.2e-4) with shallow decay (final_lr_fraction=0.95) keeps late steps relatively large, interacting with PER skew.  
  4) PER ramp (5k–15k) may still let prioritization bite while critic is adapting to the new noise regime.

## v35 plan: reduce late PER bias and noise-driven spread
- Parameter changes vs. v34:
  - per_alpha_final 0.32 (from 0.4) and per_alpha_ramp_end 18k (from 15k) to soften/delay prioritization; per_beta_final 0.95 (from 0.8) to increase IS correction late.
  - noise_floor 0.24 (from 0.32) to temper late action variance while keeping the early plateau intact.
  - lr_a 2.0e-4 (from 2.2e-4), lr_c 1.1e-4 (from 1.2e-4), final_lr_fraction 0.90 (from 0.95) to shrink late-step sizes.
- Expected signals if v35 works:
  - Delta_Percent seeds converge tighter by 32k (aim std < ~0.25) with flat or improving slopes after 20k.
  - Average100 keeps flat/slightly up slopes after 20k (no reversals for strong early seeds); dispersion below v34.
  - Action_variance_mean stabilizes ~0.18–0.19; Actions_at_upper_pct flattens instead of rising.
  - PER priority_std growth slows after ~400k; priority_mean settles below v34’s ~0.42–0.45 band.

## v35 results
- Delta_Percent: converges and tightens late; one seed lags early/mid but catches up toward the end. Dispersion late smaller than v34; improvement aligns with lower LR/stronger IS.
- Average100: broadly similar to v34; the lagging seed shows up here as well but others track close to v34.
- Actor loss: higher mid/late vs. v34 (consistent with lower LR/stronger PER bias correction).
- Critic loss: similar overall; slightly higher early and slightly lower late than v34.
- PER: priority_mean/priority_std trend higher mid/late despite softer alpha; clipping not present in v35 (added in v36).
- Target drift: lower in late stages; directionally positive for stability.
- Epsilon: not used; perceived “lower epsilon” is the decaying pre-squash noise schedule.

## v36: PER clipping + softer late bias, higher noise floor
- Parameter changes vs. v35: added priority clipping at 99.5 pct; per_alpha_final lowered to 0.30 with a longer ramp to 20k; beta_final raised to 1.0; noise_floor bumped to 0.26; lr_a/lr_c trimmed to 1.9e-4/1.05e-4 (same final_lr_fraction=0.90); alpha ramp start unchanged (5k).
- Delta_Percent: climbs faster early and reaches a slightly higher plateau than v35; seed spread is narrower than v35 late but still visible within the same episode (stochastic rewards likely widening the bands). Residual mid-run wiggles suggest a bit more late focusing could help.
- Average100: most seeds higher in mid–late stages than v35; one seed sits ~0.05–0.1 below the cluster, pointing to under-updating for that run.
- Losses/TD: critic_loss is lower in early–mid (then similar late); actor_loss trends slightly higher/less negative with one higher seed. TD percentiles sit below v35 across p50/p90/p99, consistent with smaller effective PER weights + lower LRs.
- PER stats: priority_max is effectively flat (clip + beta_final=1 pushed PER toward uniform); priority_mean/std markedly lower vs. v35; entropy similar. PER is now almost neutral—good for stability, but may be under-weighting hard samples.
- Policy/behavior: action_variance_mean similar to v35; actions_at_upper_pct higher (noise_floor 0.26) but no collapse. Avg_Exercise_Count/Avg_Total_Exercised drift lower than v35, implying a slightly more conservative exercise policy while still improving delta.
- Stability: target_drift unchanged vs. v35; noise schedule unchanged aside from the higher floor.
- Open items for next iteration: need a mechanism to pull the weakest seed up and tighten within-episode variance—likely by reintroducing mild PER focus without large spikes (e.g., slightly higher alpha_final or a gentler clip like 99.0) or a touch more late LR while keeping beta_final high. Could also test rolling the noise_floor back toward 0.24 if upper-bound pressure starts to increase.

## v37: Cosine LR schedule (long horizon)
- Change: Replace linear LR decay with a 5% warmup into cosine decay over a 65k-episode horizon; floor at 5% of initial LR (final_lr_fraction=0.05), warmup_frac=0.05, lr_schedule_episodes=65000. Training still stops at 32k.
- Rationale: Keep aggressive learning through 32k (LR ≈0.56·η₀ at 32k) while allowing a seamless extension to ~65k episodes without retuning; should improve late stability without hurting mid-run updates.
- What stayed the same: PER/noise schedules, optimizers, and all other hyperparameters from v36. LR logging remains in TensorBoard to visualize the new cosine curve.

## v38: Cosine LR with fixed warmup episodes + slightly stronger late PER
- Config deltas vs. v37: warmup_episodes=1024 (explicit), lr_a/lr_c back to 2.0e-4/1.1e-4, final_lr_fraction=0.08 (cosine horizon 65k), per_alpha_final=0.33, per_beta_final=0.98, priority_clip_pct=99.7. Noise unchanged (sigma0 1.3, floor 0.26).
- Delta_Percent (blue): mid/late levels improve vs. v37 and v36; most seeds finish ~-1 to -1.8, one lagging around -2.2/-2.5. Seed spread narrower than v37, similar/slightly better than v36; early ramp marginally slower than v37 but catches up by ~12–15k.
- Average100 (not shown): climbs steadily with a late plateau a touch above v37; the lagging Delta seed mirrors a small gap here.
- Policy variance: action_variance_mean settles ~0.19–0.21 with actions_at_upper_pct in mid-20s/low-30s—higher than v36, lower than v37—keeping exploration without boundary push.
- TD/losses: TD p50/p90/p99 track close to v37 with slightly lower p99 spread late; critic_loss is among the lowest late, actor_loss smooth with no late bumps.
- PER stats: priority_mean/std above v36 but below v37; priority_max capped by the 99.7 clip; entropy stable. PER focus increased without reintroducing spikes.
- Exercise behavior: Avg_Exercise_Count declines more slowly than v37 (stays ~9–9.5 vs. magenta drifting toward ~8–8.5); Avg_Total_Exercised remains ~11.5–12.5 with similar noise. No collapse observed.

## v39 plan: soften late PER and decay LRs faster to tighten seed spread
- Script changes (from v38): final_lr_fraction=0.20, lr_schedule_episodes=40000 (faster decay by 32k); per_alpha_final=0.20 with ramp_end=25000 (longer, softer PER); keep warmup_episodes=1024 and existing noise/per settings otherwise the same.
- Optimizer (code-level): set AdamW betas to actor (0.9, 0.99) and critic (0.85, 0.99) to let the critic adapt faster to PER-driven variance while keeping the actor smoother.
- Expectation: lower PER bias late plus quicker LR decay (~20–30% of initial LR by episode 32k) should reduce priority_std/TD p99 spread and pull weaker seeds toward the pack, trading a bit of peak LR for tighter convergence and lower seed-to-seed variance.

## v40: Warmup calibration targets std=0.05 directly
- Change: the 1,024-episode warmup now measures the untrained policy’s mean/std and rescales the actor head so `E[action]` hits `Q_max / n_rights` and `Std[action]` is fixed at 0.05. Removed the previous “99% mass in [0,1]” solver; still noise-free during warmup and mirrored to the target net.
- Motivation: avoid relying on Gaussian mass assumptions and lock in a consistent, small exploration scale at startup; should reduce seed-to-seed variance in early behavior and keep actions away from bounds before noise is added.
- Expected signals: action_variance_mean near 0.0025 at episode 0 (before added exploration noise), smoother initial TD errors, and fewer early action-at-upper spikes; downstream effects on delta spread to be validated against v39.
- Results vs. v39 (green=v39, orange=v40): Delta_Percent/Avg100 similar late, but v40 shows wider early/mid spread and slightly higher TD p90/p99; action_variance_mean and Actions_at_upper_pct are higher and more dispersed, and Avg_Exercise_Count drifts lower. PER priority_mean/std trend higher for v40. Net: calibration helped early neutralization but the larger 0.05 std plus the same LR let seeds separate mid-run.

## v41 plan: tighter init std + lower peak LRs
- Code change: warmup target std reduced 10× (0.005) to start policies closer together and lessen early action variance before noise is added.
- Script change: lower peak LRs while keeping the same schedule (warmup 1,024; cosine to 20% by 40k): actor LR 1.6e-4 (from 2.0e-4), critic LR 9.0e-5 (from 1.1e-4). All PER/noise/tau/clip settings remain as v40.
- Rationale: images show mid-run divergence in action variance/Actions_at_upper_pct and TD p90/p99; lowering both the initial std and peak LRs should curb early overshoot and tighten seed spread without altering late decay behavior.

## v41 results (red vs. v39/v40 in green/yellow)
- Delta_Percent: early ramp matches v40, mid/late spread tighter; one seed that led briefly drifted slightly late (likely eval noise). Overall convergence is smoother with less oscillation.
- LR traces: reduced peaks as intended; smoother decay tracks match the v41 cosine with lower max.
- TD errors: p50/p90/p99 sit slightly below v40; the p90 tail for v40 (yellow) runs higher—v41 cuts late TD variance.
- Average100: overlays are virtually identical late; early/mid trajectories are smoother and slightly less noisy.
- Exercise metrics: Avg_Exercise_Count drifts lower but stays clustered; Avg_Total_Exercised oscillates in a tighter band—no collapse.
- Policy stats: action_variance_mean and Actions_at_upper_pct bands are narrower and centered lower than v40; Actions_at_lower_pct stays at 0.
- Target drift: continues to decay smoothly; v41 aligns with or below v40 late.
- PER stats: priority_mean/std sit below v40 late, indicating softer sampling skew; entropy/clip unchanged.
- Losses: actor/critic losses descend more smoothly with smaller spread, consistent with lower step sizes and smaller init std.

## v42: Profitability-constrained actor output (hard gate + STE)

### Summary

In **v42** we introduced an **architectural constraint** in the **Actor (policy)** so the executed action is *feasible-by-construction*: the policy will **never execute an unprofitable exercise** given the swing option’s *immediate* net payoff with convex costs. Concretely, we added a **profitability gate** that outputs **exactly `q_t = 0`** whenever the proposed exercise would yield non-positive immediate profit. To preserve a useful learning signal (and avoid dead gradients), the gate uses a **Straight-Through Estimator (STE)** so backprop behaves as if the ungated action were used.

We also enforced a clean exploration rule: **exploration noise is applied only in “pre-gate” space**, and the profitability gate is applied afterwards. This preserves exploratory behavior near the profitability boundary while ensuring the environment never receives an unprofitable action due to noise.

This change is primarily about **learning dynamics and stability**: it removes reliance on the environment’s post-hoc action masking for correctness, prevents “mask-banging”, and makes the actor’s optimized policy match the executed policy.

### Motivation (why env-level masking was not enough)

The environment already masks actions to zero when the realized immediate net payoff is non-positive. In the vanilla pipeline, the actor can still propose a continuum of invalid actions; the environment collapses them all to the same executed action (`q=0`) and the same reward (`r=0`). That creates a broad **flat region** in the data distribution seen by the critic, making it harder for the actor-critic pair to learn a sharp and stable profitability boundary. Practically this shows up as:

- **Wasted exploration**: many samples propose invalid exercises that always get zeroed out.
- **Weak gradients near the true boundary**: many distinct proposed actions map to the same transition and reward after masking.
- **Higher seed variance**: early random dynamics can push policies into repeatedly proposing masked actions (“banging against the mask”).

v42 moves the constraint into the actor so the “policy being trained” matches the “policy being executed”.

### The constraint (what is “profitable”)

We gate on the swing option’s **immediate net profit** with convex exercise cost:

`Pi(q) = q * relu(S - K) - c_cost * q**gamma_cost`

where:
- `relu(S-K)` is the intrinsic value per unit exercise at the current decision time,
- `c_cost, gamma_cost` are the convex cost parameters from the contract,
- `q` is the **denormalized** exercise quantity (contract units).

Important detail for this repo: the actor outputs a **normalized action** `q_raw ∈ [0, 1]` (via `tanh01`). For gating, we denormalize to contract units using the same mapping as the environment:

`q_actual = q_min + q_raw * (q_max - q_min)`

Then we compute `Pi(q_actual)` and gate using a strict `> 0` test (so zero-profit exercises become `q=0`).

### What changed in the actor (hard gate in forward pass)

Conceptually:

1) Actor produces the usual continuous proposal:
- `q_raw = tanh01(u_theta(s))`  (normalized to `[0,1]`)

2) Compute profit:
- `payoff_per_unit = relu(S - K)`
- `q_actual = q_min + q_raw * (q_max - q_min)`
- `Pi = q_actual * payoff_per_unit - c_cost * q_actual**gamma_cost`

3) Apply a hard gate:
- `mask = 1[Pi > 0]`
- `q_forward = q_raw * mask`  (forward-time executed action)

This guarantees the policy never executes an unprofitable exercise.

### Straight-Through Estimator (STE): hard forward, smooth backward

The hard gate is non-differentiable. If treated literally, gradients vanish in the “masked” region and learning stalls. We use an STE via the common detach trick:

- forward uses the hard-gated value `q_forward`,
- backward behaves as if output was `q_raw`.

Implementation pattern (PyTorch):
- `q = q_raw + (q_forward - q_raw).detach()`

This keeps the executed policy hard-constrained while preserving a useful gradient signal for the actor.

### Exploration noise moved to pre-gate space

v42 enforces a strict rule: **never add noise after gating**.

In this codebase, exploration noise is applied in **pre-activation** space (pre-squash), then squashed into `[0,1]` to form `q_raw`, and only then profitability gating is applied. This ensures:

- exploration can still probe near the profitability threshold,
- but the final action passed to the environment is always profitable-or-zero,
- and we never accidentally “un-gate” a zero action by adding post-gate noise.

### Implementation notes (where it lives in the code)

- **State input for `(S-K)`**: `SwingOptionEnv` places `spot_minus_strike (S-K)` at `state[0]`. v42 uses `state[..., 0]` as the profitability signal.
- **Contract parameters** (`q_min`, `q_max`, `c_cost`, `gamma_cost`): pulled from the environment’s `contract` and cached onto both the local and target actors via a one-time init. This keeps the gating logic identical for local/target networks.
- **Training alignment**: the actor’s `forward()` now returns the gated action, so the actor loss is computed against the executed policy (no mismatch between “optimized action” and “executed action”).

### Why this design was selected (trade-offs and expectations)

Primary goals:
1) **Hard constraint satisfaction**: never execute unprofitable exercises.
2) **Maintain gradients**: avoid dead zones from a non-differentiable gate.
3) **Minimal disruption**: keep the DDPG/D4PG plumbing intact (critic/replay/target updates unchanged).
4) **Improve stability and seed robustness**: reduce wasted exploration and remove “mask-banging”.

Expected practical outcomes:
- **Cleaner learning signal near the profitability boundary**: the critic sees consistent state-action pairs (no downstream collapse of many actions into `q=0`), improving the actor’s ability to learn the decision boundary.
- **More efficient exploration**: exploration mass is not wasted on actions that are guaranteed to be masked.
- **Lower seed variance**: fewer early “bad proposals” that push training into unstable regimes.
- **Better policy correctness**: even if the environment mask remains as a safety net, correctness no longer depends on it.

Potential side-effects to watch:
- More conservative behavior right at the threshold (because `Pi > 0` is strict). If we ever want to allow exactly-zero profit exercises, we’d switch to `Pi >= 0`, but v42 intentionally forbids them.
- If `q_max` is large and costs are steep, the gate can create a large masked region early; STE keeps gradients flowing, but monitoring action-rate / exercised-volume metrics remains important.

### v42 results vs. v41 (observations and interpretation)

Below is an interpretation of the **v42 (profitability-gated actor)** vs. **v41 (no actor-level gate, env masks downstream)** results, using the end-of-run scalar snapshots from the 3-seed runs:

- v41: `SwingOption_20_v41_{11,12,13}`
- v42: `SwingOption_20_v42_{11,12,13}`

#### 1) Pricing/Delta_Percent: why the late “non-zero” is still a win

End-of-run `Pricing/Delta_Percent` (RL vs. LSM) is:
- v41: `[-1.1, -0.7, -0.3]` → mean ≈ `-0.7%`, seed std ≈ `0.33%`
- v42: `[-1.1, -0.7, -0.9]` → mean ≈ `-0.9%`, seed std ≈ `0.16%`

So v42 achieves **substantially lower seed-to-seed dispersion** in the price delta (about ~2× tighter).

On the “why not 0% at the end?” question: a large fraction of the remaining offset is plausibly **evaluation noise** (and/or benchmark noise), not policy instability.

The evaluation price is a Monte Carlo mean of discounted returns across `n_paths_eval=32768`. For a Monte Carlo estimator, the **Central Limit Theorem** implies the estimator error is approximately **Normal** for large `n`:

- `RL_Price ≈ Normal(true_price + bias, (sigma_RL^2 / n))`

In the logged runs, `Price_Std` at the end is ~`2.65–2.83`. That implies:
- Standard error of RL price ≈ `Price_Std / sqrt(32768)` ≈ `2.7 / 181` ≈ `0.015`
- Relative standard error (percent) ≈ `(0.015 / 2.66) * 100` ≈ `0.55%`
- 95% CI width (percent) ≈ `~1.1%`

If the LSM benchmark itself has ~`~1%` variability vs. the “true” value (as your bootstrap plot indicates), then the observed end deltas in the `~-0.7%` to `~-1.1%` range are **completely consistent** with:
- a small remaining bias (possibly slightly underpricing vs LSM),
- plus a measurement distribution that is roughly **Gaussian** (Normal) at this sample size.

So your interpretation (“these values are logical given ~1% evaluation error”) is reasonable. With only 3 seeds, the safest takeaway is:
- v42 appears to have **tightened the distribution** (lower variance),
- and the remaining non-zero delta is within the **expected measurement noise envelope**.

**What distribution is it following?**
- The *evaluation estimator noise* is approximately **Normal** due to CLT.
- The *Delta_Percent* is a smooth transform of that estimator (roughly linear when errors are small), so it is also approximately **Normal**, up to the rounding in `signed_zero_aware_pct_change()` (it rounds to 0.1%).
- If you consider “RL vs LSM” as the difference of two noisy estimates, the difference remains approximately Normal (with variance depending on correlation between the two estimators; here they share the same underlying eval path set, so the errors can be correlated, often shrinking the difference variance).

#### 2) Why v42 reduces seed variance (core mechanism)

The architectural change removes a major source of instability in v41:

- In v41, the environment can mask an unprofitable exercise to `q_actual=0`, but the replay buffer stores the **proposed normalized action** that was passed into `env.step()`, not the masked `q_actual`. That creates an **off-policy mismatch**: the critic is trained on `(s, a_proposed, r_masked)` tuples where the reward corresponds to a *different executed action*.
- In v42, the actor applies the same profitability constraint before the environment sees the action, so the stored action and executed action align. This makes the critic’s learning problem **better-posed** and reduces the “flat Q around invalid actions” pathology.

That alignment tends to:
- reduce useless exploration mass (invalid proposals that all collapse to the same outcome),
- make gradients more consistent near the profitability boundary,
- and thus tighten seed-to-seed behavior late.

#### 3) Policy action stats: why the variance bands look “awesome”

End-of-run action stats:
- `Policy/Action_variance_mean`:
  - v41 mean ≈ `0.211 ± 0.023`
  - v42 mean ≈ `0.193 ± 0.018`
- `Policy/Actions_at_upper_pct`:
  - v41 mean ≈ `0.320 ± 0.067`
  - v42 mean ≈ `0.250 ± 0.055`

This is the most directly “explained-by-architecture” improvement:

- The profitability gate forces `q=0` whenever `relu(S-K)` is too small to overcome costs (or is negative). That collapses the action distribution in those states in a *structured, state-dependent* way. You stop seeing random seeds spend long stretches “trying” large exercises in OTM/near-threshold states (which then get masked downstream).
- Because **noise is pre-gate**, exploration can still perturb the raw proposal, but the gate prevents noise from turning an unprofitable situation into an executed large action. This prevents spurious saturation at the upper bound driven by noise in states where exercise makes no economic sense.

Net effect: the policy occupies a more meaningful region of action space and avoids boundary saturation artifacts, so the **band narrows** and becomes more consistent across seeds.

#### 4) PER differences: higher priority mean/std (and why that can happen)

End-of-run PER stats:
- `PER/priority_mean`: v41 ≈ `0.332 ± 0.0056`, v42 ≈ `0.333 ± 0.0051`
- `PER/priority_std`: v41 ≈ `0.314 ± 0.0034`, v42 ≈ `0.316 ± 0.0038`
- `PER/priority_entropy`: essentially unchanged (`~12.167–12.168`)

So the “PER looks different” observation aligns with the data: v42 has **slightly higher priority dispersion**.

Why this is plausible:
- With actor-level gating, the dataset is less dominated by “invalid proposals that always get r=0”, so more transitions carry a *real signal* (profitable exercise or correctly chosen no-exercise). That can increase TD-error heterogeneity, which PER reflects.
- The critic is no longer learning on inconsistent `(a_proposed, r_executed)` pairs in the masked region, so TD errors can shift upward in the transitions that actually matter (and downward where things became easier). That can raise priority_std even if training is healthier overall.

This kind of PER change is not automatically “bad”; it often means the replay is focusing on more informative parts of the state-action space.

#### 5) Critic loss and TD error: why they can be higher under a better policy

End-of-run critic/TD snapshots show:
- `Critic_loss`: v41 ≈ `0.096 ± 0.013`, v42 ≈ `0.142 ± 0.025`
- `TD_Error/p90`: v41 ≈ `0.978 ± 0.043`, v42 ≈ `1.014 ± 0.068`
- `TD_Error/p99`: v41 ≈ `1.78 ± 0.21`, v42 ≈ `1.98 ± 0.41`

Why can TD error (and critic loss) go up while policy quality and stability improve?

1) **Hard boundary introduces a sharper function to learn.**  
   The profitability gate creates a discontinuity at `Pi(q_raw)=0`. Even with STE (which helps the actor), the critic still has to model value around a sharper decision boundary. That can raise residual errors, especially at higher quantiles (p90/p99).

2) **You removed an “easy mode” for the critic.**  
   With env masking + stored `a_proposed`, many different actions produce the same observed reward/next-state behavior, which can artificially flatten the Q surface and reduce TD error (but also reduce useful gradients). After gating, the action-reward relationship is more consistent and informative, which can increase TD errors on meaningful transitions (the critic is solving a harder, more correct problem).

3) **Higher TD error is only bad if it’s unstable.**  
   If TD p99 rises but outlier spikes and collapse events reduce, that’s typically a trade: the critic is tracking a richer/less-degenerate target rather than “cheating” via masked collapse. The important signals to watch are runaway growth, exploding priorities (not happening due to clipping), and destabilized evaluation metrics (which improved in seed variance).

#### 6) Actor loss “lower” (more negative): what it likely means here

Actor loss in this DDPG-style setup is `-E[Q(s, pi(s))]`. More negative usually indicates the critic is assigning higher Q to the actor’s actions. In v42 it is more negative on average at the end:
- v41 ≈ `-1.66 ± 0.04`
- v42 ≈ `-1.77 ± 0.09`

That can be consistent with the actor focusing on “economically meaningful” actions (profitable exercises) and avoiding wasted action mass. But it can also reflect a scale shift in the critic; so treat it as a supporting indicator, not a primary objective.

#### 7) Exercise count / total exercised: why they can look healthier

Your qualitative observations (“better spread”) are consistent with the mechanism:
- v42 removes invalid attempts, so exercise decisions become more tied to true ITM opportunities and cost structure.
- That tends to produce more stable and interpretable exercise patterns, especially late, because the policy is not fighting the environment mask.

Endpoint snapshot alone shows similar averages, but the key benefit is typically in the *trajectory stability* and reduced “mask-banging” dynamics over time.

#### 8) Target drift: interpreting this metric under gating

Target drift in this repo is logged as a drift proxy between the critic estimate on replay actions and the target critic on target-policy actions. Under gating, small changes in `q_raw` near the profitability threshold can flip the hard mask; that can make “drift” look larger in absolute terms even if the overall system is more stable.

So for v42, focus on whether target drift:
- converges smoothly,
- avoids sudden regime shifts,
- and correlates with improvements in pricing stability (it appears to, from the reduced Delta_Percent dispersion you observed).

### Action variance progression (v23 → v26 → v33)
- **v23**: Used early action L2 (cutoff ~4k) to fight boundary lock-in; helped two seeds but one still collapsed and delta hovered around -1.2 to -18 across seeds. Saturation risk persisted despite strong early regularization.
- **v26**: Switched to tanh01 output mapped to [0, 1], boosted exploration floor (noise_floor 0.18, plateau 3.2k), and scheduled PER (alpha ramp 5k→15k). This eliminated collapse without action L2; actions_at_upper_pct dropped into the low 20s%, and delta_percent improved to roughly -0.5 to -1.1 on best seeds.
- **v33**: Added SiLU throughout the MLPs plus the calibrated output scaling/bias (targeting Q_max/n_rights and 95% mass in [0,1]). Action_variance_mean tightened (~0.16–0.17), actions_at_upper_pct slid further down, and Avg_Exercise_Count/Avg_Total_Exercised rose by ~1–2 units vs. v26 while maintaining stable TD/priority profiles. This combo keeps early behavior diverse without any action regularization.

## Version-by-Version Notes (concise chronology)

- **v1–v4**: Early baselines (not fully documented in this chat). Larger networks, default noise/epsilon, no action regularization; occasional stuck policies and high variance.
- **v5–v6**: Began adjusting PER, noise, and tau; introduced 2×64 networks and more stable defaults. Still saw occasional stuck runs and flat delta_percent.
- **v7**: Strong, stable baseline (2×64 nets, n_step=1, PER on, batch 64). Good convergence; moderate delta_percent (~-2 to -4). No action regularization; noise and LR defaults. Considered “best baseline” for a while.
- **v8–v9**: Attempted PER clipping and higher batch sizes; PER over-focus flattened learning. v9 showed plateauing similar to earlier failures; stuck policies persisted.
- **v10**: Added action clipping/clamp on output (briefly) and safe-training interventions; didn’t improve delta_percent; some policies still saturated. Later removed the clamp.
- **v11**: Removed stuck behavior; stayed close to v7 performance. Action variance healthy; delta_percent slightly weaker than v7.
- **v12–v13**: Batch 128 experiments; saturation and stuck runs reappeared; safe-training interventions caused regressions. Outcome: batch 128 + aggressive PER/noise schedules were unstable. Eventually removed safe-training entirely.
- **v14–v15**: Code cleanup, no safe features, added action L2 regularization in the actor loss; added target policy smoothing and min_action_noise to reduce saturation. Stability improved, but occasional single-seed failures remained.
- **v16–v18**: Noise plateau feature introduced; min_action_noise lowered (v18). With too-low noise floor, some runs regressed to boundary solutions and flat delta_percent.
- **v19**: Raised noise floor to 0.15, sigma=1.2, anneal_power=0.6, plateau=2k; stable convergence (Average100 ~2.5–2.7) and delta_percent ~ -1 to -1.6; however, exploration tapered quickly mid-run, limiting late improvement.
- **v20**: Retuned noise and LR to balance exploration/exploitation:
  - noise_sigma=1.25, noise_anneal_power=0.65, plateau=1500, min_action_noise=0.12, epsilon_decay=0.99994.
  - final_lr_fraction=0.8 (slower LR decay) to preserve late adjustment.
  - Batch 128 retained; PER unchanged (alpha 0.5, beta_start 0.8, clip_pct 99.5).
  - Action_reg and target smoothing kept to avoid saturation.
- **v21**: Pushed exploration floor/plateau higher (sigma=1.3, anneal_power=0.6, plateau=2200, min_action_noise=0.18, epsilon_decay=0.99995; PER alpha 0.5; final_lr_fraction=0.8) to avoid late collapse.
  - Outcome: Changes were unsuccessful. Two seeds collapsed in action variance (Policy/Action_variance_mean → ~0 with >40–50% actions at upper bound after ~200k steps); delta_percent stayed poor (-17.9 for one, -7.6 for another). The “healthier” seeds still ended with delta_percent around -1.0 to -1.8 (target is <1%) and Average100 hovered ~2.3–2.6. Overall: more boundary lock-in and weaker delta vs. v19.
  - Next (v22) adjustments: soften PER (alpha=0.45), extend plateau to 2500, slow decay (epsilon_decay=0.99996, anneal_power=0.55), raise noise floor to 0.22, and keep more LR late (final_lr_fraction=0.85) to counter collapse and improve delta.
- **v22**: Softer PER (alpha=0.45), longer plateau (2500), gentler anneal_power=0.55, higher noise floor=0.22, slower epsilon decay=0.99996, final_lr_fraction=0.85; action_reg_weight stayed at 1e-3 (always on).
  - Outcome: Improved stability vs. v21 (only 1/4 seeds stuck), faster wall-clock (~1.4–1.5 hr vs ~4.3 hr), but one seed still collapsed (delta_percent ~-18). Best seeds reached delta_percent ~-1.2 to -1.5; one seed at -3.2. Average100 ~2.1–2.9. Action L2 likely helped curb full saturation but constant reg may bias against boundary-optimal policies late.
  - Observations: Action_variance_mean healthier for non-collapsed seeds; collapsed seed still drifts to upper bound. PER means remain moderate; TD percentiles stable for good seeds.
- **v23 plan**: Keep softer PER, extend exploration slightly, and gate stronger action L2 to early training only.
  - noise_sigma=1.3, noise_anneal_power=0.55, noise_plateau=3000, min_action_noise=0.20, epsilon_decay=0.999965, final_lr_fraction=0.9 (preserve late learning).
  - action_reg_weight=0.002 applied only for first 4000 episodes (action_reg_cutoff=4000), then 0 to avoid biasing boundary-optimal actions.
  - Goal: eliminate remaining collapse and push all seeds to delta_percent < 1 by 32k episodes while retaining generality across swing parameter regimes.
  - Outcome: Mixed. Two seeds converged near target (delta_percent ≈ -1.2, -1.5), one mid performer (-2.8), one failure (-18.4). Average100 ~2.3–2.7; wall-clock ~1.35–1.55 hr. Gated action L2 helped keep most seeds off the boundary, but a single seed still collapsed; late delta remained slightly above the <1 target for successful seeds.
- **v24**: Kept softer PER; extended plateau slightly and slowed decay; stronger early action L2 and longer LR warmup.
  - noise_sigma=1.3, noise_anneal_power=0.55, noise_plateau=3200, min_action_noise=0.20, epsilon_decay=0.99997, final_lr_fraction=0.95, warmup_frac=0.03; per_alpha=0.45; batch 128; action_reg_weight=5e-3 with cutoff at 4000 episodes.
  - Outcome: Slightly better when not stuck: best seed reached delta_percent ~-0.9; two seeds at ~-1.7/-2.1; one seed still collapsed (~-17.7). Average100 clustered ~2.1–2.6; runtime ~1.2–1.34 hr. Action variance collapse persists on one seed; stronger early reg didn’t fully eliminate single-seed failure.
- **NP1 (no PER)**: Disabled PER entirely; kept v24 noise/LR schedule, lowered min_replay_size to 18k, increased target_policy_noise to 0.15, action_reg_weight=5e-3 cutoff at 4k.
  - Outcome: No action variance collapse; faster early Average100 ramp. Delta_percent did not improve enough (best seeds around -1.4 to -2.8; one seed dipped to ~-8.4). Critic/actor losses were higher, likely due to uniform replay’s higher gradient variance plus stronger target smoothing noise. Runtime ~1.3–1.5 hr.
- **v25**: PER re-enabled but softened (alpha=0.2, beta_start=1.0, slow beta anneal 60k, no clipping, floor 1e-6); kept late-exploitation tweaks (plateau 3200, min_action_noise=0.18, epsilon_decay=0.999965, final_lr_fraction=0.95), action_reg_weight=3e-3 cutoff 3.5k, target_policy_noise=0.15, min_replay_size=18k.
  - Outcome: No collapse; one seed frequently dipped below -1 delta_percent in evals; finals around -1.5, -2.3, -1.8, -1.5. Early phase slower than NP1 (one seed lagged until ~5k episodes). Higher variance in delta_percent mid-run, stabilizing near the end. Average100 comparable to NP1/v24; runtime ~1.23–1.34 hr.
- **v26 plan**: PER alpha/beta scheduling (near-uniform first 5–7k, ramp to moderate PER by ~15k); min_action_noise 0.18; action_reg_weight 4e-3 cutoff 4k; same late-exploitation LR/epsilon settings.
  - Outcome: Best so far. No collapse, faster start than v25, and finals around -0.5 to -1.1; early lag largely gone. PER schedule kept early behavior close to uniform and enabled refinement later.
- **v27 plan**: Even more uniform early and slightly stronger early guard, then ramp PER by ~14–15k. Tweaks vs v26: per_alpha=0.05, ramp start/end 2.5k→14k (alpha_final=0.4, beta_final=0.8), noise_sigma=1.35, noise_plateau=3500, epsilon_decay=0.999975, action_reg_weight=4.5e-3 cutoff=4500, final_lr_fraction=0.96.

Key outcomes:
- Best stability: v7 (batch 64) and v19 (batch 128 with higher noise floor).
- Stuck/saturation issues: v8–v9, v12–v13 (batch 128 + low noise floor/overly strong PER), v18 (low noise floor despite plateau).
- Safe-training/logging: Tried, then removed; kept the code lean and focused on core algorithmic tweaks (noise schedules, action regularization, target smoothing).

### PER behavior notes (v24 vs NP1 vs v25)
- PER ON (v24): occasional variance collapse; lower actor/critic losses but higher risk of boundary lock-in; delta spread tighter when not collapsed.
- PER OFF (NP1): no collapse and faster early ramp; higher actor/critic losses due to uniform sampling + target smoothing; delta improved mid-run but still above target.
- PER soft (v25): no collapse, but early learning slower than NP1; mid-run delta variance higher; finals modestly better than v24. PER priority stats show growing skew mid-run, raising gradient variance. Conclusion: PER needs scheduling—uniform early to avoid destabilizing the critic, then gradual PER later for efficient refinement.
- PER scheduled (v26): uniform-like until ~5–7k (alpha near 0, beta ~1), ramping to moderate PER by ~15k delivered the best mix: no collapse, faster convergence, and strong final delta. Scheduling PER is key; static PER (even soft) either slows early learning or risks collapse; no PER limits late refinement.

---

## How to Tune the Hyperparameters (runv{}.sh)

### Exploration (noise/epsilon)
- **noise_sigma**: Controls initial exploration scale. Higher (1.1–1.3) improves early exploration and reduces boundary lock-in. Start high if saturation is observed; reduce if training is too noisy.
- **noise_anneal_power**: How strongly noise tracks epsilon. Smaller (<1) decays slower; larger (>1) decays faster. Use ~0.6–0.7 for a gentle taper.
- **noise_plateau**: Number of episodes to hold initial noise/epsilon. Useful to keep policies from collapsing early. Typical: 1–2k.
- **min_action_noise**: Noise floor to prevent zero-variance policies late. Higher floors (0.12–0.15) prevent saturation; too low (0.05) led to stuck runs.
- **epsilon/epsilon_decay**: Start around 0.3. Slower decay (0.99994–0.99995) retains exploration; faster decay hurts late improvement.

### Learning Rates / Schedules
- **lr_a / lr_c**: For batch 64, actor 3e-4, critic ~1.6–1.8e-4 worked well. For batch 128, scale down: actor ~2.0–2.2e-4, critic ~1.1–1.2e-4.
- **final_lr_fraction**: How much to decay by the end (1.0 = no decay). Slower decay (0.8) keeps capacity for late-stage refinement; 0.7 is more conservative.
- **warmup_frac**: 0.02–0.05 typical. Keep brief to reach steady learning quickly.

### PER
- **per_alpha**: Controls prioritization strength. 0.45–0.55 safer; higher can over-focus and flatten learning.
- **per_beta_start / per_beta_frames**: 0.8 start, 150k frames good defaults; adjust if bias correction is too slow/fast.
- **per_priority_floor / per_priority_clip_pct**: Floor avoids zero priorities; clipping caps runaway spikes. Moderate clip (99–99.5) or disable (0) depending on stability; too aggressive clipping flattened PER effect.

### Target Updates / Smoothing
- **tau**: 0.003–0.0035 moderate smoothing. Too fast drifts targets; too slow slows adaptation.
- **target_policy_noise / target_policy_clip**: Default 0.1 / 0.25 (TD3-style) to smooth targets; set to 0 to disable.

### Batch / Replay
- **batch_size**: Batch 64 was most robust; batch 128 needs lower LRs and higher noise floor.
- **min_replay_size**: Larger with bigger batch (e.g., 24k) to ensure diversity at start.

---

## Parameters to Keep Constant vs. Worth Tuning

Kept mostly constant (unless experimenting):
- Network size: 2×64 (actor/critic) — good trade-off for CPU and stability.
- n_step: 1 (best stability).
- gamma: 1 (domain-specific reward already discounted).
- actor/critic depth: 2; target smoothing noise/clipping (0.1/0.25) usually left as default.

Worth tuning:
- Exploration: noise_sigma, noise_anneal_power, noise_plateau, min_action_noise, epsilon_decay.
- Learning rates and final_lr_fraction (especially when changing batch size).
- PER hyperparameters: alpha, beta_start, priority floor/clip.
- tau (target update speed) within a narrow band (0.003–0.0035).

---

## Most Important Parameters for Swing Option Pricing

1) **Exploration schedule (noise/epsilon)**: Prevents early collapse to boundary actions. High initial sigma + sufficient plateau + meaningful noise floor are critical.
2) **Learning rate and decay**: If decayed too fast, late-stage refinement stalls; if too high, instability. Coupled with batch size.
3) **PER strength**: Too strong → over-focus, flattening learning; too weak → slower convergence. Moderate alpha (~0.5) worked best.
4) **Min action noise + activations**: Min noise floors plus the tanh01 actor head and SiLU hidden layers keep action variance healthy without extra regularization.
5) **Batch size vs. LR**: Larger batches need lower LRs and larger warmup buffers.

Tuning approach:
- Start from a stable preset (e.g., v7 for batch 64 or v19/v20 for batch 128).
- Adjust exploration first (sigma, floor, decay, plateau) if you see boundary saturation or mid-run dips.
- Adjust LR decay (final_lr_fraction) if late improvement stalls.
- Lightly adjust PER (alpha) if priorities are too flat or over-spiky.
- Only then tweak tau within a narrow range if target drift looks off.

---

## v43 results: Q_T-approx warm-start (green) vs v42 (pink)

- Change: warmup calibration targets `E[action] ≈ E[Q_T]/n_rights` using the HHK-based `approximate_Q_T` (instead of `Q_max/n_rights`).
- Outcome: better early profitability (higher `Average100`, less-negative `Pricing/Delta_Percent` from episode 0), more stable policy statistics, and slightly improved late-stage convergence vs. v42.

Plot-by-plot read:
- **Pricing/Delta_Percent**: v43 starts materially closer to 0 than v42 and tracks slightly higher through training; worst-case seed behavior is improved (no deep early drawdown like the v42 outlier).
- **Average100**: v43 ramps faster and sits a touch higher late; dispersion across seeds is tighter earlier, consistent with reduced “bad init” variance.
- **Policy/Action_variance_mean**: v43 is flatter and stays in a narrower band; v42 drifts upward more, indicating stronger policy dispersion and more sensitivity to noise/replay.
- **Policy/Actions_at_upper_pct**: v43 stays lower, suggesting less saturation at the upper bound and fewer hard-clipped actions.
- **Avg_Exercise_Count / Avg_Total_Exercised**: v43 is more stable mid/late (less drift and less spread), consistent with starting closer to realistic utilization.
- **TD_Error (p50/p90/p99)**: v43 runs slightly higher across percentiles (stronger learning signal) without blow-ups; v42’s tails look more muted late.
- **PER/priority_mean/std/min/max**: v43 priorities are significantly larger and more dispersed, meaning PER is actually differentiating transitions; v42 looks close-to-uniform (flat priorities), which tends to slow policy improvement once the critic plateaus.
- **Actor_loss / Critic_loss / Target_drift**: losses are broadly similar, but v43’s actor loss trends more negative late while target drift is slightly lower—consistent with steadier policy improvement.

Why v43 is better:
- With the profitability gate (v42+), the early replay distribution is very sensitive to the untrained policy’s mean action. Targeting `Q_max/n_rights` biases the startup policy away from realistic utilization, increasing the chance of “mask-heavy” rollouts and flatter gradients. Using a model-based `E[Q_T]` warm-start improves early reward density and produces a healthier initial replay mix, which PER can exploit (higher, more structured priorities) while keeping policy saturation in check.

## v44 plan: v43 baseline with no clipping

- Change vs. v43 run config: remove *all clipping knobs* from the run script: PER priority clipping (`--per_priority_clip_pct`), actor/critic gradient clipping (`--actor_grad_clip*`, `--critic_grad_clip*`), and target smoothing noise clip (`--target_policy_clip`).
- Hypothesis: if v43’s mid/late plateaus or seed-to-seed differences are being “papered over” by clipping, removing clips will make the true instability source visible (PER skew vs. exploration noise vs. LR schedule), so we can tune those drivers directly.
- Primary readouts: `Pricing/Delta_Percent` slope after ~10k–20k episodes, `TD_Error` percentiles + PER priority dispersion (`PER/priority_std`, `PER/priority_max`), action variance/boundary rates, and `Target_drift` during any TD bursts.
- If v44 destabilizes (preferred responses that still keep v44 unclipped): (1) soften late PER (`--per_alpha_final` down and/or `--per_beta_final` up), (2) reduce variance injection (`--target_policy_noise` and/or `-noise_floor`), (3) shrink late step sizes (lower `-lr_a/-lr_c` and/or faster decay via `--final_lr_fraction` / `--lr_schedule_episodes`).

## v44 results: removing clipping destabilized training (v43 better)

Runs compared:
- v43: `SwingOption_20_v43_{11,12,13}` (clipping enabled in the run config)
- v44: `SwingOption_20_v44_{11,12,13}` (no clipping)

Empirical takeaway from the TensorBoard overlays: **clipping is useful** in this project. With v44 (no clipping), the learning dynamics become more **heavy-tailed** and **oscillatory**, and the policy drifts toward less stable / more saturated behavior. v43 stays noticeably more controlled.

Plot-by-plot read (why clipping helps):

- **PER/priority_max**: v44 shows step-like jumps to much larger maxima (rare extreme-TD transitions dominate); with clipping, the max stays lower and grows more smoothly. This matters because PER sampling probability is a steep function of priority—without a cap, a handful of transitions can monopolize replay, driving non-stationary updates.
- **PER/priority_std**: v44 has materially higher dispersion and abrupt regime changes; clipping compresses the tail and keeps priority dispersion in a narrower band, improving replay diversity and reducing “replay collapse” onto a few outliers.
- **PER/priority_mean**: v44’s mean priorities sit higher and fluctuate more, consistent with a critic that is being repeatedly yanked by tail events; clipping reduces the impact of those tails so the average TD magnitude doesn’t drift upward as aggressively.
- **TD_Error (p50/p90/p99)**: the upper percentiles in v44 climb more and stay elevated, indicating heavier tails in Bellman error. Clipping is acting as a tail-risk control: it prevents extreme TD errors from turning into extreme priorities and extreme gradients.
- **Critic_loss**: v44 exhibits larger spikes/outliers (classic signature of bootstrapped target instability amplified by PER outliers). Clipping reduces the amplitude/frequency of those spikes, which stabilizes the value landscape the actor is optimizing against.
- **Stability/Target_drift**: v44 shows noisier and (at times) higher target drift, consistent with larger effective update steps; clipping reduces abrupt critic shifts, so target networks track the online networks more smoothly.
- **Policy/Action_variance_mean**: v44 drifts upward and is more volatile, suggesting that the policy is being pushed around by unstable critic gradients and/or chasing transient high-Q regions; clipping dampens those swings and yields a tighter variance band.
- **Policy/Actions_at_upper_pct**: v44 climbs higher (more saturation at the upper bound), which is consistent with “chasing” behavior under an unstable critic; clipping helps keep the actor away from bang-bang extremes.
- **Avg_Exercise_Count**: v44 drifts downward with more dispersion (exercise behavior becomes less consistent), aligning with the story that the actor is oscillating between regimes; clipping makes the exercised-rights profile more stable over training.
- **Pricing/Delta_Percent**: v44 is visibly more jagged (higher variance) and less reliably improving; clipping reduces oscillations and improves run-to-run stability in the price estimate trajectory.

Why clipping is especially valuable here (mechanism):
- The profitability-gated actor + sparse reward structure naturally creates **heavy-tailed TD errors** (rare “very informative” profitable states, many near-zero states).
- PER amplifies those tails by oversampling high-priority items; without a cap, this creates a feedback loop: outlier TD → outlier priority → oversampling → large critic shift → new outliers.
- Clipping (priority/gradient/target-noise) is a practical way to **bound the influence of tail events** so learning stays in a regime where bootstrapping remains stable.

Recommendation going forward:
- Treat clipping as a stability primitive (not just a band-aid): keep it on while tuning late-stage dynamics (PER ramp, noise floor, LR decay).

## v45 plan: v43 baseline + RMSNorm

- Change vs. v43: set `--norm=rmsnorm` (RMSNorm) while keeping the v43 run config identical otherwise (PER priority clipping, grad clipping, and target policy noise clipping all remain enabled).
- Goal: test whether RMSNorm improves optimization stability (critic TD tails, loss spikes, target drift) without sacrificing policy learning speed or pushing the actor toward boundary saturation.
- Expected outcome (if RMSNorm helps): smoother `TD_Error`/`Critic_loss` trajectories, less oscillatory `Target_drift`, and equal-or-better `Pricing/Delta_Percent` / `Average100` with reduced seed variance vs. v43.
- Failure mode to watch: worse boundary behavior (higher `Actions_at_upper_pct`) or noisier PER priorities if the changed normalization shifts activation scales in a way that amplifies TD tails.

## v45 results: RMSNorm was faster but less stable (worse convergence)

Overall: RMSNorm increased throughput (higher `Paths_Per_Second`) but produced a more heavy-tailed / regime-switchy training dynamic: PER priorities became more volatile, TD-error tails inflated, the policy drifted toward bang-bang saturation (more `Actions_at_upper_pct`), and pricing convergence was less reliable than the LayerNorm baseline (v43).

Graph-by-graph read (v45 RMSNorm vs v43 LayerNorm baseline):

- **Paths_Per_Second**: higher for v45 → RMSNorm is cheaper than LayerNorm (no mean-centering), so wall-clock speed improves.
- **PER/priority_std**: v45 shows a sharp step-up / regime change mid-run and ends materially higher → replay becomes dominated by a wider priority spread (more “tail events”).
- **PER/priority_mean**: v45 runs higher and shows a noticeable mid-run jump before partially relaxing → average TD magnitude increased and became less stationary.
- **TD_Error/p50**: only modestly worse → the *typical* Bellman error is not the core problem.
- **TD_Error/p90 & TD_Error/p99**: v45 is consistently higher late → the main change is *fatter tails* (rare, very large TD errors).
- **Critic_loss**: smoothed trend looks similar, but the background variance/spike density is higher under v45 → consistent with the TD tail story (loss spikes driven by outliers).
- **Policy/Action_variance_mean**: v45 trends higher and climbs more persistently → policy distribution spreads more, increasing sensitivity to critic noise and bootstrapping error.
- **Policy/Actions_at_upper_pct**: v45 is higher and increases faster → policy spends more mass at `q≈q_max` (bang-bang behavior).
- **Avg_Exercise_Count**: v45 drifts downward and is more dispersed → despite more saturation-at-max when exercising, the policy exercises on fewer dates (more “all-or-nothing”: either skip or max).
- **Actor_loss**: v45 is less negative / flattens higher → weaker and noisier policy-improvement signal (actor not consistently finding better actions under the critic).
- **Average100**: broadly similar → average episodic return is not the discriminating metric here; pricing error and stability are.
- **Pricing/Delta_Percent**: v45 is more jagged and shows worse mid/late convergence (including a clear dip in one seed) → instability translates into worse/less reliable pricing.

Interpretation (why RMSNorm likely hurt here):

- **LayerNorm’s mean-centering matters in this critic**: the critic ingests concatenated state+action with non-stationary feature scales (HHK state statistics shift over training; the actor distribution shifts too). LayerNorm cancels mean shifts and stabilizes hidden pre-activations; RMSNorm only rescales magnitude. In this setting, *activation mean drift* can translate into Q-scale drift and larger TD outliers.
- **PER amplifies tail instability**: the environment + profitability gate already induces heavy-tailed TD errors (rare profitable states, many near-zero states). RMSNorm further fattened TD tails, which PER oversampled; this creates a feedback loop (outlier TD → high priority → oversampling → critic jolts → more outliers) visible as the mid-run “priority_std/mean regime change”.
- **Policy responds by saturating**: with a noisier critic landscape, the actor is nudged toward extreme actions rather than learning a smooth exercise surface, hurting pricing convergence and seed robustness.

Takeaway:
- RMSNorm is attractive for throughput but, under the current PER + target-noise + LR settings, it increases TD tail risk and destabilizes learning. If we revisit RMSNorm, it likely needs a “stability retune” (lower LR and/or stronger priority tail control, plus possibly reduced target noise) rather than being a drop-in swap.

## TensorBoard note

You can read runv18 logs by changing the tensorboard path in the bash script (e.g., pass `-name` accordingly). The scripts default to logging under `runs/<name>`. Adjust `-name` to keep experiments separated for comparison in TensorBoard. 
- **v27**: Tried to push PER softer early and higher noise (sigma 1.35, plateau 3500, epsilon_decay 0.999975), action_reg=4.5e-3 cutoff 4.5k, PER alpha=0.05→0.4 ramp 2.5k–14k (sigmoid), beta_final=0.8, final_lr_fraction=0.96.
  - Outcome: Regression. One seed failed early (action variance collapse, Actions_at_upper_pct spikes), delta_percent worse (one at -17.9), higher TD/p99 and priority spikes (priority_max/std jumped). Likely causes: higher initial noise with slower decay plus stronger reg/plateau increased TD noise, and the PER ramp (even with sigmoid) still let a few high-error samples dominate mid-run, driving collapse for seed 12. Early phase also slower vs v26.
- **v28 plan**: Sigmoid PER ramp, revert to v26 noise/decay (sigma 1.3, plateau 3200, epsilon_decay 0.99997), alpha_final 0.35, ramp 3k–14k, beta_final 0.85, min_action_noise 0.18, action_reg 4e-3 cutoff 4k, final_lr_fraction 0.95. Goal: keep v26 stability, reduce PER aggressiveness that hurt v27, retain early NP1-like behavior but with sigmoid smoothing.
  - Outcome: Stable (no collapse) but weaker late delta vs v26; higher run-to-run variance. Finals around -1.1 to -3.0; Average100 slightly below v26. Actor loss higher mid/late; critic similar with occasional spikes. PER priority_mean/std higher than v26 late, suggesting PER still over-focusing. Action_variance_mean lower with larger spread; Actions_at_upper_pct lower (less boundary pressure). Net: PER still too strong late for best delta; early robustness preserved but learning efficiency dipped.
- **v29**: Sigmoid PER ramp with gentler late PER: per_alpha_final=0.32, per_beta_final=0.9, ramp 3.2k→14k; alpha=0.05 start; noise back to v26; min_action_noise=0.18; epsilon_decay=0.99997; action_reg=3.5e-3 cutoff 4k; final_lr_fraction=0.96.
  - Outcome: No collapse; late delta improved over v28 but still slightly weaker than v26 (finals ~-0.9 to -3.0). Early robustness maintained. Early–mid learning slower than v26 (and some seeds lagged vs v28). Average100 a bit below v26. Actor loss slightly higher late; critic loss lowest late. PER priority_mean/std modestly higher late, with some seed variance; action_variance_mean generally OK but one seed lower, hinting at under-exploration in that seed. Interpretation: PER still a bit strong late; learning rate decay and lighter reg helped stability; lower critic loss late likely reflects steadier targets, but delta didn’t fully reach v26 levels.
- **v30**: Softer late PER and more LR retained: per_alpha_final=0.30 with sigmoid ramp 3k→13k; beta_final=0.9; noise same as v26; min_action_noise=0.18; epsilon_decay=0.99997; action_reg=3.5e-3 cutoff 4k; final_lr_fraction=0.97.
  - Outcome: Regression vs. v26. Final delta_percent around -3.3 to -1.0 (v26: -1.1 to -0.5). Pricing/Delta lines sit ~0.5–2% below v26; Average100 similar early but trails late. Action_variance_mean sagged earlier and Actions_at_upper_pct dropped faster (more conservative policy). TD_Error percentiles lower than v26; PER priority_max lower with higher std spread, pointing to weaker/high-variance sampling signal. Actor_loss flattened higher (less negative), suggesting muted policy updates; critic_loss similar to prior versions. Runtime ~1.30–1.32 hr (slightly slower than v29).
- **v31 plan**: Raise action variance and TD focus without losing v30 stability: per_alpha_final=0.34 with sigmoid ramp 3.2k→15k; beta_final=0.9; min_action_noise=0.20; keep v30 LR/decay; shorten action_reg window (cutoff 3.5k) to avoid over-conservatism.

### PER behavior notes (v24 vs NP1 vs v25/v26/v27/v28/v29/v30)
- PER ON (v24): occasional variance collapse; lower actor/critic losses but higher risk of boundary lock-in.
- PER OFF (NP1): no collapse and faster early ramp; higher losses; delta improved mid-run but above target.
- PER soft (v25): no collapse, but early slower; mid-run delta variance higher; finals modest.
- PER scheduled linear (v26): best mix—uniform early, moderate PER by ~15k; strong delta and no collapse.
- PER scheduled, more aggressive (v27): regressions; priority spikes, collapse in one seed.
- PER scheduled, softer (v28): stable but weaker late delta; higher priority_mean/std; lower action variance.
- PER scheduled, gentler late (v29): stable, late delta better than v28 but shy of v26; priority stats still somewhat high; one seed with lower action variance.
- PER scheduled, even softer late (v30): TD errors and priorities lower; action variance dipped; delta percent regressed ~0.5–2% vs v26. Softer PER + unchanged noise floor likely under-drove useful samples and policy updates.

### Critic loss vs performance
- Lower critic loss late can indicate stable targets and smoother value estimates; good when paired with healthy action variance and improving delta.
- If critic loss is low but PER priorities/variance are high and action variance drops, it may signal over-smoothing or over-confident critic with insufficient exploration, leading to suboptimal delta.
- Better late delta in this project correlated with: (1) moderate PER (alpha ~0.35–0.4) after a uniform early phase; (2) maintained action variance (min_action_noise ≥0.18); (3) sufficient LR late (final_lr_fraction ≥0.95); (4) PER priority stats not spiking excessively (priority_std moderate).

---

## Postscript: Reverted v26_v2–v6 experiments (failed, not kept)
- Changes tried: forcing episodes to run full horizon (no early stop at Q_max), adding gross/net payoff features to state, multiple PER/noise/regularization schedules (v26_v2–v6).
- Outcomes: higher Actions_at_upper_pct, action variance collapses in some seeds, delta_percent worse than v1 baseline, occasional late divergence; critic loss often stayed low so it was not a good health signal.
- Likely causes: longer episodes increased update density; early PER schedules/clipping and strong early regularization either flattened priorities or starved critics; exploration floors/schedules were insufficient to avoid boundary lock-in.
- Status: codebase reverted to pre-experiment (v1) behavior; keep these notes to avoid repeating the same avenues without deeper changes.
