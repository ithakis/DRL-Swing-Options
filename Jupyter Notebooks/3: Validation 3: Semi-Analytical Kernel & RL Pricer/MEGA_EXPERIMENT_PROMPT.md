# Mega-Experiment Prompt — "Close the LSM gap under the deterministic kernel target"

> Paste this entire prompt into a fresh Claude Code session (Opus 4.8, high effort, plan mode) in the `DRL-Swing-Options` repo. It is self-contained and ready to run.

---

You are a quantitative researcher working on a reinforcement-learning swing-option pricer. Your job is to design and run a rigorous, publishable experiment campaign to close the remaining pricing gap between the kernel-on D4PG agent and the full-state LSM-D benchmark — without increasing the training budget. Read CLAUDE.md in full before doing anything else; it is authoritative on every implementation detail.

## Current state

The pricer is D4PG (standard critic, IQN removed) for a swing option with convex exercise cost Π(q) = q(S−K)₊ − c·q^γ under the HHK two-factor OU-with-jumps spot model. The key architectural feature is a semi-analytical transition kernel (`--use_expected_target=1`) that replaces the single-sample TD bootstrap with an analytical expectation over the HHK kernel — making the Bellman target **deterministic**. This is the central fact that drives everything below.

Current performance at the focal regime (c=0.04, γ=2), 4096-episode budget, fast kernel (M_x=2):
- Anchor config (v61 hyperparameters): Δ% = −0.66% ± 0.18 vs LSM-D M=5
- After Task-2 retune (linear noise decay + eval EMA): Δ% = −0.42% ± 0.10
- At 16384 episodes (same config): Δ% = −0.10%

The benchmark is **full-state LSM-D, Chebyshev basis, degree 2** (NOT degree 7 which overfits OOS), evaluated on the **common fresh test set: seed=999, 65536 paths**, LSM trained on seed=998. This is the `tools/rebuild_results_v7.py` protocol — reuse it, do not reinvent it. Both LSM-D and the RL rollout are lower bounds on the true price; higher is strictly better.

The unifying hypothesis: **the v61 recipe was tuned to fight single-sample TD variance that no longer exists**. Every lever below is a bet that some piece of the old recipe is now neutral-to-harmful. Let the data decide.

## Objective

Find a new, frozen, flag-guarded kernel-on canonical that — at the same 4096-episode budget — achieves:
1. Focal Δ% ≥ 0% vs LSM-D M=5, and ≥ −0.2% vs LSM-D M=17
2. No regression on guard regimes: nocost (c=0, γ=1), g1 (c=0.04, γ=1), g15 (c=0.04, γ=1.5)
3. Seed-std and worst-seed held or improved
4. Every claim backed by paired-seed Welch tests with Holm–Bonferroni correction across the tested family

## The ten hypotheses to test

Each is flag-guarded and defaults to the current canonical. Test one lever at a time in Stage A.

**H1 — Critic LR is mis-scaled.** The v61 critic LR (9e-5) was tiny to survive TD noise. With a deterministic target it can be larger. Variants: (a) lr_c=3e-4; (b) lr_c=6e-4 with lr_a=3e-4; (c) short critic-only LR boost during warmup.

**H2 — PER is mis-specified.** PER up-weights high-TD-error transitions to combat target variance. With an exact target, residual error is approximation error and PER may bias the fit toward payoff kinks. Variants: (a) uniform replay (PER off); (b) softer α ramp to 0; (c) anneal PER off after the warmup window.

**H3 — Too few gradient steps per episode.** With a deterministic target, extra updates are pure signal. Variants: (a) learn_every=1 (currently 2); (b) learn_number=2; (c) batch size 256. Report Δ% per unit wall-clock, not just per episode.

**H4 — β-sigmoid(3.0) squash is too stiff for interior exercise.** Convex cost means the optimal q is interior; a sharp squash biases toward bang-bang exactly where γ>1 rewards intermediate exercise. Variants: (a) beta_sigmoid_2.0; (b) beta_sigmoid_1.5; (c) temperature-annealed β. Note: `actor_output_activation` is eval-critical — any change requires re-baselining saved agents.

**H5 — Profitability-gate STE distorts gradients near the exercise boundary.** Variants: (a) temperature-relaxed soft gate annealed to hard; (b) train ungated, gate only at eval; (c) tune STE temperature. Verify no unprofitable exercise leaks at eval.

**H6 — Critic warm-start seeds a biased basin.** Variants: (a) longer `critic_warmup_episodes` (256/512/1024 — g1 requires ≥256 to avoid collapse); (b) tune the closed-form FOC budget cap; (c) revisit backward-induction warm-start as initialisation only.

**H7 — Residual kernel bias caps the achievable price.** Even on the M_x≥2 plateau a tiny systematic target bias could limit the ceiling. Variants: (a) accurate kernel M=36 (M_x=4, M_per_k=4, N_max=2); (b) M_x=3; (c) Richardson extrapolation across M_x∈{2,4}. Run this early — it separates tuning gap from kernel gap.

**H8 — Target-policy smoothing noise is redundant.** v61 carried TD3-style `target_policy_noise=0.15`; the kernel already integrates over transitions. First verify whether this flag still exists in `run.py` (it may have been removed in v63). If present: (a) remove it; (b) halve it; (c) decay to zero.

**H9 — Slow target network tracking lags an exact target.** τ=0.0032 was conservative to filter noise. Variants: (a) τ=0.01; (b) τ=0.005; (c) periodic hard update.

**H10 — Network capacity under-resolves convex-cost curvature.** The 2×64 SiLU+LN net may under-fit the extra curvature in q. Variants: (a) 2×128; (b) 3×64; (c) `--feature_use_cross=1` on the nn path. Any capacity win must still beat LSM on wall-clock time.

## Statistical protocol — pre-register then execute, no peeking

Before training anything, write `logs/_mega/PREREG.md`: for each variant, the exact flag delta, directional hypothesis, primary endpoint, decision rule, and seed budget.

**Stage A — pilot (seeds 11–13, fast kernel M_x=2, 4096 ep):** Promote only if focal mean Δ% improves ≥ +0.15 pp over the kernel-on anchor AND no guard regime drops > 0.15 pp. Kill the rest. Negatives are results.

**Stage B — confirm (seeds 11–22, fast kernel, 4096 ep):** Paired-seed Welch t-test, Brown–Forsythe scale test, MDE report, Holm–Bonferroni across the promoted family. A variant wins only if it clears the multiplicity-corrected threshold and is non-inferior on all guards and seed-std.

**Stage C — finalist (≥24 seeds, accurate kernel M=36, 4096 ep + spot-check at 16384 ep):** Combine Stage-B winners, confirm on the accurate kernel, pick the new canonical by conservative Pareto (mean Δ%, seed-std, worst-seed) — not mean alone.

Use `tools/stats_analysis.py` throughout: `welch_t_test`, `scale_test`, `f_var_ratio`, `minimum_detectable_effect`, paired-seed t.

## Infrastructure to reuse

- Training: `tools/sweep_expected_target.py` (`base_args`, `run_one`), `tools/sweep_v63_audit.py` (has `--resume`). Extend these rather than writing from scratch. Cap at ~4 concurrent subprocesses.
- Evaluation: `tools/rebuild_results_v7.py` (`build_agent`, `build_contract`, `build_hhk_params`, `evaluate_rl_on_test_set`, `evaluate_lsm_on_test_set`). Always evaluate on the common test set seed=999.
- Stats: `tools/stats_analysis.py`. Write all CSVs to `logs/_mega/` in the existing sweep-CSV schema.
- Build a `tools/sweep_mega.py` harness with `--resume` so the campaign is restartable.

## Hard constraints

- Touch only the four regimes above. No full convex-cost sweep.
- Every new behaviour is opt-in behind a flag; current canonical must reproduce bit-exactly when all new flags are at their defaults.
- `pytest tools/test_approximators.py` must stay 37/37 and the nn-path bit-identical guard must hold after any code change.
- Eval-critical flags (`use_robust_normalization`, `actor_output_activation`, `weight_averaging`) change the saved-agent forward pass — any change requires re-baselining, not just re-evaluation.
- No peeking-and-stopping. Stage promotions follow pre-registered rules and corrected p-values only.

## Deliverables

1. `logs/_mega/PREREG.md` written before any training
2. `tools/sweep_mega.py` with `--resume`
3. `logs/_mega/findings.csv` and a results summary (hypothesis table, forest plot of per-H Δ% effects with CIs, before/after focal strip+box)
4. An `HPT.md` entry documenting each hypothesis verdict and the new frozen canonical flags
5. An honest paragraph on what did not work

## Where to start

Run H7 (kernel-fidelity control) first — it tells you whether you are chasing a tuning gap or a kernel gap before interpreting any other result. The highest-prior cheap trio is **H1 (LR) + H2 (PER) + H3 (more updates)**; they are orthogonal and most likely to move the focal mean within budget. Keep H4 and H5 on a short leash — they are eval-critical.
