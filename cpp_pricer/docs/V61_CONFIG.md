# Literal v61 (no-kernel) configuration — source of truth for the v67 no-kernel build

This pins the **exact published-paper v61** focal config (c=0.04, γ=2) so the C++ `build_v67_nokernel`
port can be validated against it. Recovered from git commit **`3c76417`** (the commit *before*
`435eab8` "Debloat + adopt v64 canonical", which deleted PER/TPN/LR-decay), specifically
`Convex Cost Experiments/SwingOption_20_c0.04_gamma2.sh` (focal, seeds 11–13) and the v61-era
`src/agent.py` / `run.py`. Cross-checked against HPT.md §3/§6, lines 638, 922, 948.

Published v61 focal result (HPT.md:948): **Δ% = 111.47%, seed-std 2.59%** (vs Chebyshev **degree-7**
LSM); nocost −1.00% / 0.57%. "Key change: gradual warmup + β=3.0."

## Hyperparameters (focal c=0.04, γ=2, 32k episodes)

| Group | Knob | v61 value | C++ home / status |
|---|---|---|---|
| **Net** | actor_layers / critic_layers | **2 / 2** | `--actor_layers 2 --critic_layers 2` (CLI ✓) |
| | layer_size (width) | **64** | `--hidden 64` (default ✓) |
| | activation (hidden) | **SiLU** | **default build** (no `-DGELU_*`) ✓ |
| | norm | layernorm | hardcoded ✓ |
| | init_method | orthogonal | `--init_method 1` (✓; v65 uses He default) |
| | actor_output_activation | **beta_sigmoid_3.0** | **`-DACTOR_BETA_VAL=3`** (compile-time; mlp.hpp:17) ✗ ADD |
| | profitability gate | **ON** (c>0) | `c_cost>0` ✓ |
| **Opt** | lr_a / lr_c | **1.6e-4 / 9.0e-5** | `--lr_a 1.6e-4 --lr_c 9e-5` (CLI ✓) |
| | **double critic step** | **YES** (legacy dup `critic_optimizer.step()`) | ✗ ADD `--double_critic_step 1` (≈2× critic LR) |
| | learn_every / learn_number | 2 / 1 | `--learn_every 2 --learn_number 1` (CLI ✓) |
| | batch (bs) | 128 | `--batch 128` (default ✓) |
| | tau (-t) | 0.0032 | `--tau 0.0032` (default ✓) |
| | gamma | 1 | default ✓ |
| | weight_decay_actor / critic | 5e-5 / **1.2e-4** | `--wd_c 1.2e-4` (CLI; wd_a default 5e-5 ✓) |
| | b1/b2 (Adam) | actor .9/.99, critic .85/.99 | default ✓ (verify vs v61) |
| **LR decay** | schedule | **cosine + linear warmup** | ✗ ADD `--lr_schedule cosine` |
| | warmup_episodes | 1024 | ✗ ADD `--lr_warmup_episodes 1024` |
| | lr_schedule_episodes (horizon) | **40000** | ✗ ADD `--lr_schedule_episodes 40000` |
| | final_lr_fraction | **0.20** | ✗ ADD `--final_lr_fraction 0.20` |
| | min_lr | 1e-6 | ✗ ADD `--min_lr 1e-6` |
| **Noise** | noise_sigma0 / floor | 1.30 / 0.26 | `--noise_sigma0 --noise_floor` (CLI ✓) |
| | noise_schedule | **hyperbolic** | ✗ ADD `--noise_schedule hyperbolic` (C++ linear-only) |
| | noise_plateau | **3200** | ✗ ADD `--noise_plateau 3200` (no CLI today) |
| | adaptive_noise_scale | 0.6 | `--adaptive_noise_scale 0.6` (default ✓) |
| | warmup_noise_fraction | **0.4** (focal; HPT regime table says 0.3) | `--warmup_noise_fraction 0.4` (CLI ✓) |
| **Critic warmup** | critic_warmup_episodes | 1024 | `--critic_warmup 1024` (CLI ✓) |
| **PER** | per | **1 (ON)** | ✗ ADD `--per 1` |
| | per_alpha (base) | 0.1 | (overridden by ramp below) |
| | per_alpha_final, ramp_start, ramp_end | **0.20, 5000, 25000** | ✗ ADD α-ramp |
| | per_beta_start, per_beta_final, per_beta_frames | 1.0, 0.98, 120000 | ✗ ADD β-anneal |
| | per_priority_floor / clip_pct | 5e-6 / 99.7 | ✗ ADD |
| | priority scheme | standard (|TD|) | ✗ ADD |
| **TPN** | target_policy_noise | **0.15** | ✗ ADD `--target_policy_noise 0.15` |
| | target_policy_clip | 0.25 (**unused** in code) | skip (dead) |
| | target_noise_decay_start / floor | 20000 / 0.04 | ✗ ADD decay |
| **Calibrate** | method | **rprop** (v58+) | C++ has closed_form; rprop is a "statistical wash" → use closed_form, NOTE the substitution |
| **Eval** | weight_averaging | **off** (no EMA) | ✗ ADD `--weight_avg 2` (off) |
| | use_robust_normalization | 1 | hardcoded ✓ |
| **Replay** | min_replay / max_replay | 18000 / 200000 | ✗ ADD CLI (C++ defaults 1000/100000) |
| **Kernel** | use_expected_target | **0** (single-sample TD) | `--kernel_off` ✓ |
| **Grad clip** | actor/critic_grad_clip | set (1.0/2.5) but **forced to None** in agent.py:415-418 | **DEAD — do not port** |
| **Munchausen / IQN / nstep** | — | 0 / 0 / 1 | off ✓ |

LSM benchmark in the v61 script: Chebyshev **degree-7** (note: v64/v65 pipeline uses degree-2). The
111.47% Δ% is vs degree-7; the v67 notebook three-way will use the current canonical LSM for
apples-to-apples with the kernel mode, so report **price** (and seed-std), not the historical Δ%.

## Recovered algorithm bodies (from `3c76417:src/agent.py`)

### Hyperbolic noise (`_pre_noise_sigma`) — identical to current `src/agent.py:443`
```
e = max(1, episode); plateau = max(0, noise_plateau)
if plateau>0 and e<plateau:  sigma = sigma0                       # hold during plateau
else: t = e-plateau (if plateau>0 else e); denom = 1 + t/max(1, plateau or 1)
      sigma = floor + (sigma0 - floor)/denom                      # hyperbolic decay
if e <= critic_warmup and warmup_noise_fraction < 1:             # v61 gradual warmup ramp
      frac = warmup_noise_fraction + (1-warmup_noise_fraction)*(e/critic_warmup); sigma *= frac
```
(C++ `agent.cpp:139` already does the linear branch + the warmup ramp; add the hyperbolic + plateau-hold branches, default stays linear.)

### Cosine LR (`lr_lambda`, stepped once/episode)
```
if final_lr_fraction >= 1.0: scale = 1.0                          # no schedule (v65 default)
elif episode <= warmup_episodes: scale = episode/warmup_episodes  # linear warmup
elif episode <  total_episodes:  t=(episode-warmup)/(total-warmup)
     scale = final_lr_fraction + (1-final_lr_fraction)*0.5*(1+cos(pi*t))
else: scale = final_lr_fraction
scale = max(min_lr/init_lr, scale);  lr = init_lr * scale
```
`total_episodes = lr_schedule_episodes` (40000). Apply per-optimizer (lr_a, lr_c separately).

### Target-policy noise (`_get_target_policy_noise` + `learn_`)
```
# schedule:
if decay_start<=0: tpn = target_policy_noise
elif e<decay_start: tpn = initial(0.15)
else: prog=min(1,(e-decay_start)/decay_start); tpn = floor + (initial-floor)*(1-prog)   # → 0.04 by 2*decay_start
# application inside the single-sample TD target (no_grad):
next_a = actor_target(next_s)
if tpn>0: next_a = clamp(next_a + N(0,tpn), 0, 1); next_a = gate(next_a, next_s)
q_next = critic_target(next_s, next_a)              # target_policy_clip is UNUSED
```

### PER (`learn_` + `_compute_base_priorities` + `_maybe_update_per_schedule`)
```
# sampling: proportional to priority^alpha; IS weight w_i = (N*P_i)^(-beta)/max_w
# loss: critic_loss = mean(weights * (q_target - q_expected)^2)     # IS-weighted MSE
# priorities = clamp_min(|td|, floor); optional percentile clip (99.7) ; update_priorities(idx)
# alpha/beta schedule (focal): alpha=0 for e<=5000 (PURE UNIFORM), linear→0.20 by 25000;
#                              beta = 1.0 for e<=5000, linear→0.98 by 25000.
```
**At ≤4096-episode budgets (the notebook R1 / optimization screen), alpha≡0 ⇒ PER is exactly uniform
replay.** PER only bites in the full 32k run, and even there α≤0.20 ("extremely soft, mimic uniform").

### Double critic step (the ~2× effective critic LR)
`learn_` calls `self.critic_optimizer.step()` **twice** in a row (no zero_grad/backward between) — two
Adam updates on the same gradient. Reproduce with a `--double_critic_step 1` flag (default 0 = v65
single-step). lr_c=9e-5 with the double step ≈ 1.8e-4-effective.

## Porting priorities (→ Phase 1)
**Must add for fidelity:** `-DACTOR_BETA_VAL` build macro; hyperbolic noise + `--noise_plateau`;
`--weight_avg 2` (off); cosine LR (`--lr_schedule cosine` + warmup/horizon/final/min); TPN + decay;
double critic step; min/max replay CLI. **PER** is required only for the 32k full run (inert ≤5k ep) —
implement via the existing Fenwick sampler in `replay.hpp`, but it can be the last piece since the 4096-ep
notebook/optimization work doesn't exercise it.
**Bit-identical invariant:** every new flag defaults to v65/off so `build_v67_kernel` == v65.
