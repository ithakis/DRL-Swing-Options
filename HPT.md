# Hyperparameter Tuning Journey (v1 → v20)

This document summarizes the evolution of the hyperparameters and algorithmic tweaks across versions 1–20, what worked, what failed, and how to tune the key knobs for swing option pricing. It also notes parameters we tried to keep constant and why.

---

## Update (v33+)

- Action regularization (`action_reg_weight`, `action_reg_cutoff`) has been removed from the codebase; references below are kept only as historical notes for pre-v33 runs.
- Hidden activations now default to **SiLU**; actor outputs use the **tanh01** head (maps [-1, 1] → [0, 1]). A 1,024-episode warmup pass estimates the untrained policy mean/std and shifts/scales the output layer so `E[action] ≈ Q_max / n_rights` and ~95% of mass stays in [0, 1] (sigma solved via brentq/bisection). This prevents initial saturation without any action L2.
- **v33 vs. v26 (4 seeds each, green vs. gray in the attached TensorBoard plots)**:
  - Pricing/Delta_Percent: v33 climbs out of the initial drawdown much faster and is less volatile through the first ~5k episodes; finals cluster in the -1.2 to -2.6% band (v26: -0.5 to -2.2% with one slower seed). Average100 ramps earlier for v33, converging to ~2.5–2.6.
  - Action variance: Action_variance_mean stays tighter (~0.16–0.17 vs. v26 drifting above 0.19), Actions_at_upper_pct is visibly lower, and Actions_at_lower_pct remains at 0 (no lower-bound collapse). Target_drift decays faster for v33.
- Exercise behavior: Avg_Exercise_Count stabilizes around 9–10 vs. v26’s 6–8; Avg_Total_Exercised holds ~11.5–12.2 (v26 sits lower early and recovers slowly).
- Loss/TD metrics: Actor_loss declines more smoothly; TD_Error percentiles and PER priorities rise steadily without the spikes seen in weaker v26 seeds, indicating healthier replay focus.
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
