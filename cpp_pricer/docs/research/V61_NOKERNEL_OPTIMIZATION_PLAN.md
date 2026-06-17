# v67 no-kernel optimization — TOST-gated re-screen of v65 levers on single-sample TD

**Goal.** Starting from the literal published-paper v61 no-kernel recipe (ported to C++, see
[`../V61_CONFIG.md`](../V61_CONFIG.md)), find changes that improve **≥1 of {wall-clock, price (Δ% vs
LSM), seed-to-seed variance}** with **no regression on the others**, while keeping the defining no-kernel
trait: the **single-sample TD target** (`--kernel_off`). Mirrors the methodology of
`SPEEDUP_RESEARCH_PLAN.md` / `NN_ARCHITECTURE_RESEARCH_PLAN.md`.

**Protocol.** `build_v67_nokernel` (SiLU + β-sigmoid 3.0), 4096 episodes, OOS eval on 65 536 paths.
Three regimes — **g2** (focal c=0.04 γ=2), **g1** (c=0.04 γ=1), **nocost** (c=0) — at **12 seeds**
(11–22). Each lever applied *individually* on top of the literal-v61 baseline (B0). Price is the OOS
price; LSM is constant per regime, so price-mean ranking == Δ% ranking. A lever is **adopted only if it
is clean across all three regimes** (per-regime price within seed noise or better, variance not worse,
no time regression unless explicitly trading time for price).

## Per-lever result (12 seeds; Δ vs B0 in price-pp% and seed-std)

| Lever (on top of literal v61) | g2 Δpp / Δstd | g1 Δpp / Δstd | nocost Δpp / Δstd | Verdict |
|---|---|---|---|---|
| **eval-EMA on** (`--weight_avg 0`) | +0.28 / **−0.0037** | +0.07 / −0.0013 | −0.06 / **−0.0091** | ✅ **ADOPT** — variance↓ in all 3 regimes, price flat-to-up |
| **deeper critic** (`--critic_layers 3`) | −0.08 / −0.0089 | **+0.74** / +0.004 | **+1.71** / +0.002 | ✅ adopt (price↑ on guards; costs ~+33% train time) |
| single critic step (`--double_critic_step 0`) | −0.14 / −0.002 | **−1.87 / +0.056** | **−1.67** / −0.016 | ❌ **g1/nocost collapse** — v61's double step is load-bearing |
| drop TPN (`--target_policy_noise 0`) | +0.04 / +0.002 | −0.45 / +0.006 | −0.60 / −0.001 | ❌ regresses guard price |
| linear noise (`--noise_schedule linear`) | 0.00 / +0.004 | +0.65 / +0.002 | −0.08 / +0.005 | ❌ adds variance; nocost price dips |
| swish-β3 hidden activation | −1.77 (g2, 5-seed) | — | — | ❌ hurts no-kernel (helps only kernel mode) |
| EMA + single step | −0.11 / −0.001 | −1.84 / +0.055 | −1.92 / −0.012 | ❌ inherits the single-step collapse |

## Headline findings

1. **The literal-v61 recipe is already near-optimal for single-sample TD.** The v63→v65 simplifications
   that the *kernel* made safe — single critic step, dropping target-policy noise, linear noise — each
   **regress the guard regimes** when the kernel is absent. The most striking: removing the legacy
   **double critic step collapses g1 (−1.87 pp, seed-std 0.015→0.071)**. Without the deterministic kernel
   target, the critic needs the ~2× effective LR (double step) and the TD3 target smoothing (TPN) that
   v61 used to fight single-sample TD variance. This is the converse of the kernel-mode result and
   vindicates the user's premise: **the no-kernel mode must be the v61 lineage, not "v65 minus kernel."**
   (Direct check: v65 architecture with `--kernel_off` prices **−3.9 pp** vs literal v61 at focal g2.)

2. **Eval-EMA is the one clean, free win.** v61 did not use weight averaging; turning on the eval-only
   EMA (decay 0.999) lowers seed-to-seed variance in **all three regimes** with price flat-to-slightly-up.
   This directly serves the "lower seed variance" goal at zero training-time cost.

3. **Deeper critic (3 layers) is a price lever on the guard regimes** (+0.74 pp g1, +1.71 pp nocost,
   flat g2) at ~+33% train time (~7→~9.5 s, still ~30× faster than PyTorch). Adopted as part of the
   optimized recipe because absolute time stays tiny and the user prioritized higher price; documented as
   the one deliberate time-for-price trade.

## Combined-recipe confirmation (12 seeds × 3 regimes)

| recipe | g2 Δpp / Δstd | g1 Δpp / Δstd | nocost Δpp / Δstd | train s |
|---|---|---|---|---|
| B0 literal v61 | 0.00 / 0.0000 | 0.00 / 0.0000 | 0.00 / 0.0000 | 7.0 |
| **R_ema** = v61 + eval-EMA | +0.28 / **−0.0037** | +0.07 / **−0.0013** | −0.06 / **−0.0091** | 7.0 |
| R_full = R_ema + critic-3 | +0.02 / −0.0058 | **+0.67** / +0.0035 | **+1.67** / +0.0055 | 9.0 |

## Frozen v67 no-kernel recipe

**`literal v61` + eval-EMA on** (`--weight_avg 0 --ema_decay 0.999`, dropping v61's `--weight_avg 2`;
critic depth stays 2). This is the **only change with zero regression on any axis in any regime** —
seed-to-seed variance drops in all three regimes (the user's "lower seed variance" goal) with price
flat-to-up. The biggest gains vs the published Python v61 come from the **C++ port itself** (~7 s vs
minutes) and are orthogonal to this lever.

**Optional "price mode": + `--critic_layers 3`.** Adds large guard-regime price gains (+0.67 pp g1,
+1.67 pp nocost) and the lowest focal-g2 variance, but slightly raises guard-regime variance (+0.004/
+0.006) and costs ~+30% train time (~7→~9 s) — so it trades the speed/variance axes for price and is
NOT in the clean default. Enable it when price on the guard regimes is the priority.
