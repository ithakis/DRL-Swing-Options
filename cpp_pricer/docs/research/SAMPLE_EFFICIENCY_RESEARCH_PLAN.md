# Sample-Efficiency Research Plan — better-than-MC scenario generation for the D4PG swing pricer

*Status: DESIGN (gated on user review before any large sweep). Author hand-off doc, v0.1 — June 2026.*

## 0. One-paragraph thesis

The pricer currently feeds the actor/critic from **forward Monte-Carlo paths** (antithetic + terminal
stratification) and a **uniform replay buffer**. The conjecture (user's, and well-supported by the
swing-option numerics literature) is that MC is *not* the most information-efficient way to populate the
critic's training distribution: because the HHK state has **near-zero dispersion at t=0 and a dispersion
that grows monotonically toward maturity**, an MC ensemble wastes samples re-covering the tight early
marginals and *under*-covers the wide late marginals and the exercise boundary. We have an **analytic
transition kernel** already (`transition_kernel.py` / `cpp_pricer/.../kernel.cpp`), so we can build a
**deterministic, time-graded grid (optimal-quantization tree) that is sparse early and dense late**, and/or
use **array-RQMC** to get low-discrepancy state coverage per step. The deliverable is a 2-D efficiency
study — **estimator quality vs sample budget** at budgets {512, 1024, 2048, 4096, 8192} across the 3 cost
regimes and many seeds — answering: *can we get a higher price and/or lower seed-to-seed variance at equal
budget, or equal quality at a smaller budget (⇒ faster)?*

---

## 1. Problem definition & success metric

### 1.1 The two axes
This is explicitly a **2-D convergence study**: for each method we trace **quality as a function of the
sample budget** `N ∈ {512, 1024, 2048, 4096, 8192}`. We are not looking for one number; we are looking for
a **curve that dominates the MC curve** — either shifted up (more quality at equal N) or left (equal
quality at smaller N ⇒ speed).

### 1.2 Quality metrics (primary)
1. **OOS price, higher-is-better.** Justified in-repo: the OOS price is the mean discounted *net* payoff on
   a frozen test set; the convex cost strictly penalises over-exercise, so a higher OOS price is a strictly
   better policy, not an optimistic bias (this is the same argument HPT uses for Δ% and the v65 activation
   result). Report the price and its 95% CI from the common 65 536-path test set (seed 999), identical
   across methods so the *test* noise is shared and differences are policy-only.
2. **Seed-to-seed dispersion, lower-is-better.** Train K seeds per (method, N, regime); report the
   **across-seed std of the OOS price**. This is the user's "error estimate" and the thing that actually
   bites at small N.

### 1.3 Quality metrics (secondary / robustness — the "other ways to test the error")
3. **Self-convergence (Cauchy) curve.** ‖price(N) − price(N/2)‖ vs N — a method that converges *smoothly*
   and *monotonically* is preferable to one that is high-variance even if its mean is similar.
4. **Spot-perturbation stability (the user's "slight change to S").** Re-price at S₀·(1±h) with **common
   random numbers / common grid** and report the finite-difference **Δ and Γ** and their CRN std error
   (we already have this machinery in `src/greeks.py`; the C++ port in Phase B1 will expose it). A good
   estimator is not just high-mean/low-seed-var but **smooth in S₀** — a jagged value-vs-S₀ curve signals
   an under-covered state space even when the at-the-money price looks fine.
5. **Policy-quality cross-check vs LSM-D** (the established Δ% metric) at N = 2k and 4k, to make sure a
   "higher price" is a better *exercise policy* and not a degenerate artefact.

### 1.4 The headline question
> Does any method reach **N=4096 MC quality at N≤2048** (a ≥2× sample/wall-clock saving) while being
> **no worse on seed-std**, across **all three regimes**? Secondary win: strictly higher price at equal N.

### 1.5 Cost accounting (so "fewer samples ⇒ faster" is honest)
Report, alongside quality: (a) `cpu_train` (load-independent user-sec, already emitted), (b) scenario-build
time (quantization-tree construction is a one-off per config, *amortised over seeds*), and (c) total
sample-steps actually fed to the optimiser (batch × learn_number × #updates). A grid method that needs an
expensive build but trains in 1/4 the steps may or may not win on wall-clock — we measure, not assume.

---

## 2. The current baseline (what we must beat)

From `cpp_pricer/src/hhk_sim.cpp` and `include/replay.hpp` (v65):

- **Forward MC** of the HHK chain, `n_paths` independent trajectories over the 22 decision dates.
- **Variance reduction already present:** antithetic pairs on the OU Gaussian driver (+z/−z) *and* on the
  jump marks (V / 1−V) with a shared Poisson count per pair; **terminal stratification** (systematic
  reorder by S_T into `num_batches` strata) so each minibatch spans the terminal distribution.
- **Replay:** circular **uniform** buffer; *every* transition (state, a, r, next-state) from *every* step
  of *every* path is inserted with equal weight and sampled uniformly. Capacity ≥ all transitions, so it is
  effectively "train on the full set of (path × step) transitions, uniformly".
- **Already-tried-and-dropped (do not re-litigate):** Prioritized Experience Replay (PER) — removed in v63,
  uniform is canonical. Dyna / model-based replay (H5) — did not help. So *TD-error* prioritisation and
  *synthetic-rollout* replay are dead ends here; the live ideas below are about the **scenario-generation
  measure** and **coverage**, which is a different lever.

**Key structural fact we exploit:** `Var[X_t] = σ²(1−e^{−2αt})/(2α)` grows from 0 at t=0 to the stationary
σ²/(2α); the jump leg Y_t likewise accumulates. So the per-step marginal is *narrow early, wide late* —
exactly the user's intuition. MC spends an equal number of points on every step regardless.

---

## 3. Methods that exist (literature survey) and how they map here

### 3.1 Optimal quantization scenario trees — **the direct match to the user's "grid"**
Optimal quantization replaces a random variable X by the best N-point discrete approximation q(X)
minimising distortion `E‖X−q(X)‖²`. By Zador/Bucklew–Wise, the optimal point **density ∝ f_X^{d/(d+2)}** —
i.e. *points cluster where the law has mass*, automatically sparse where the marginal is tight (early) and
dense where it is spread (late). Chaining per-time grids with transition weights gives a **quantization
tree**.
- Bally, Pagès, Printems (2005), *A quantization tree method for pricing and hedging multidimensional
  American options*, Math. Finance — the backward-DP-on-a-grid template.
- **Bardou, Bouthemy, Pagès (2009), *Optimal quantization for the pricing of swing options*, Applied Math.
  Finance** — literally our problem; per-date optimal grids + the Bellman recursion on the grid.
- Bardou, Bouthemy, Pagès (2010), *When are swing options bang-bang?* — connects to our B-metric.
- Pagès, Printems — Gaussian / functional quantization (the OU leg is conditionally Gaussian, so its
  per-step grid is a 1-D optimal-Gaussian grid we can tabulate once; the Y/jump leg gets its own grid).
**Fit:** we already have the analytic transition kernel, so the tree's edge-weights are exact (no nested MC
needed). The per-step grid size can be **graded in time** `N_t ∝ √Var[state_t]` (sparse → dense), which is
the user's hypothesis made precise. The grid + weights become the replay set the critic/actor train on.

### 3.2 (Randomized) Quasi-Monte Carlo + Brownian-bridge / array-RQMC — **"better MC"**
- Glasserman (2004), *Monte Carlo Methods in Financial Engineering* — Brownian-bridge & stratified
  constructions that put the low-discrepancy dimensions where variance concentrates.
- Owen — scrambled (t,m,s)-nets / randomized QMC: low-discrepancy **and** an unbiased seed-to-seed variance
  estimator (so we keep the user's "include some randomness to judge seed variance").
- **L'Ecuyer, Lécot, Tuffin (2008), *A randomized quasi-Monte Carlo simulation method for Markov chains*,
  Oper. Res. — array-RQMC.** Simulate the whole ensemble of HHK chains in lock-step, **sort the chains by
  state at each step**, and advance all of them with one RQMC point set. This is a *generalisation of the
  terminal stratification already in the code to every step*, and is the lowest-risk "better-MC" arm.
**Fit:** array-RQMC needs only a per-step sort (we already sort by S_T) + an RQMC driver; it keeps the exact
same downstream training code, so it isolates the *sampling* effect cleanly.

### 3.3 Time/step budget reallocation & branching (bushy trees) — **cheap intermediate**
The user's "first-step samples don't vary much, later steps vary more" ⇒ **don't spend equal samples per
step**. Two concrete forms:
- **Stratified-in-time replay weighting:** keep MC generation, but weight/oversample later-step transitions
  in the buffer (coverage weighting, *not* TD-error PER). Tunable density schedule over the 22 steps.
- **Branching / non-recombining bushy tree:** each early node spawns several children deeper in time, so
  total transitions are reallocated toward late steps at fixed budget. (Broadie–Glasserman stochastic-tree
  flavour, but here only to *populate the buffer*, not to price directly.)

### 3.4 Importance sampling toward the exercise boundary
The informative states for a swing critic are near the **net-profitability frontier** Π(q)=q(S−K)₊−c·q^γ.
A change of measure that oversamples near-ITM / boundary states (drift the OU, reweight rewards) à la
Glasserman–Heidelberger–Shahabuddin focuses samples where the policy gradient is large. Higher risk
(reweighting interacts with the kernel target) ⇒ later phase.

### 3.5 Coverage/density-based replay sampling (not TD-error PER)
Sample the buffer to **flatten state-space coverage** (grid-bin the state, inverse-count weights) rather
than uniformly — directly fights the "early steps over-represented" problem. Distinct from the
already-dropped PER (which prioritised by TD error). Low implementation cost; pairs with §3.3.

### 3.6 Anchors from this repo (don't rediscover)
- The semi-analytical kernel (`docs/...`) already removed *target* variance; this plan attacks the
  **state-distribution** variance, which is orthogonal — expect the two to compose.
- M_x studies show the kernel mesh is on a hard plateau at M_x≥2; the quantization tree here is a *training
  scenario set*, a different object from the kernel quadrature mesh — keep them conceptually separate.

---

## 4. Candidate arms (ranked by expected value × tractability)

| # | Arm | Maps to user idea | Impl. cost | Risk | Why it could win |
|---|-----|-------------------|-----------|------|------------------|
| A0 | **MC baseline** (current) | — | 0 | — | control |
| A1 | **array-RQMC** (per-step sort + RQMC driver) | "better MC" | low | low | low-discrepancy coverage at every step, not just terminal |
| A2 | **Time-graded replay weighting** (density schedule over steps) | "denser toward the end" | low | low | reallocates fixed budget to high-variance late steps |
| A3 | **Optimal-quantization tree** (per-step Gaussian/jump grids graded in t, kernel edge-weights) | "create a grid better than MC" | high | med | deterministic, distortion-optimal coverage; sparse→dense by construction |
| A4 | **Coverage-flattening replay** (grid-bin inverse-count) | "more informative samples" | low | low | direct anti-redundancy |
| A5 | **Boundary importance sampling** | "more informative samples" | med | high | concentrates on policy-gradient-relevant states |

**Recommended order:** A1 + A2 + A4 first (cheap, low-risk, fast to falsify), then A3 (the headline grid),
then A5 only if A1–A4 leave room. A3 is the scientifically most interesting and the best match to the
user's intuition and the swing literature, but it is the most code; the cheap arms also de-risk the harness
and give an early read on whether *any* sampling change moves the needle at this network size.

---

## 5. Experimental design (statistically significant by construction)

### 5.1 Factor grid
- **Budget N**: {512, 1024, 2048, 4096, 8192} (5 levels) — the convergence axis.
- **Regime**: nocost (c=0), g1 (c=0.04, γ=1), g2 (c=0.04, γ=2, focal) (3 levels) — *always all three*.
- **Method/arm**: A0…A_k.
- **Seeds**: start at **12 seeds** (11–22) for the screen; escalate winners to **24 seeds** (matches the
  v65/kernel protocol). Paired CRN where the method allows a shared data seed (LSM-matched semantics via
  `--agent_seed`) so we can run paired tests, not just two-sample.

### 5.2 Protocol (reuse the v65 harness exactly)
- Build v65 once (`-DPRICER_GELU_FAST -DGELU_SLOPE=3`), run with the canonical v65 flags
  (`--hidden 48 --actor_layers 2 --critic_layers 4 --hidden_actor 32 --batch 64 --learn_number 3
  --lr_c 5e-4`). Only the **scenario generator / replay** changes between arms; everything downstream is
  bit-identical, so the contrast is the sampling method alone.
- Parallel runs, **4 threads each**, **clean idle machine** for any wall-clock number; use load-independent
  `cpu_train` otherwise (the v65 notes are explicit that concurrent sweeps pollute wall-clock, not price).
- Common 65 536-path test set (seed 999) for every price.

### 5.3 Statistics (reuse `tools/research_equiv.py` + `tools/stats_analysis.py`)
- **Mean (price) superiority:** Welch one-sided + paired-seed t where CRN-paired; report effect size and
  Holm–Bonferroni across the 3 regimes.
- **Variance (seed-std) change:** Pitman–Morgan (paired) / Levene–Brown-Forsythe (unpaired) + bootstrap
  std-ratio CI — the user's "lower seed-to-seed variance" claim must clear a variance test, not eyeballing.
- **Equivalence / "no worse":** TOST with a pre-registered margin (±0.5% price, matching the kernel TOST
  band) so we can assert "A reaches 4k-quality at 2k" rigorously.
- **The 2-D claim:** fit price-vs-N (and std-vs-N) per method; compare curves via the **N at which each
  method first enters the 4k-MC TOST band** (the "left-shift" = speedup) and the area between curves.

### 5.4 Decision rule
A method is **adopted** iff, across **all 3 regimes**: (i) price ≥ MC (one-sided, Holm-corrected) **or** it
hits the 4k-MC band at ≤½ the budget, **and** (ii) seed-std not significantly worse (TOST), **and** (iii)
the spot-perturbation Δ/Γ curve is no less smooth than MC. Otherwise it is logged as a negative result.

---

## 6. Phased plan

- **P0 — harness (no science yet):** add a `--sampler {mc,arqmc,...}` switch + a `--step_density` schedule
  to `price_swing`/`hhk_sim`; add a results collector that emits a tidy CSV
  `sampler, regime, N, seed, price, ci95, std, cpu_train, build_s`. Verify `mc` reproduces v65 bit-for-bit.
- **P1 — cheap screen (A1,A2,A4):** 1 hard config (g2) × N∈{1024,2048,4096} × 12 seeds. Kill or promote.
- **P2 — quantization tree (A3):** implement per-step graded grids + kernel weights; screen as P1.
- **P3 — confirm:** survivors at **full grid** (5 N × 3 regimes × 24 seeds) + secondary metrics (Cauchy,
  Δ/Γ-smoothness). Write up like the v65 HPT section; update `HPT.md`.
- **P4 (optional) — A5** only if headroom remains.

Compute envelope (per the measured ~27 s/run at 4k, ~ linear in N): P1 ≈ 12 seeds × 3 N × 3 arms ≈ 108 runs
≈ under an hour wall on 8 cores; full P3 confirm ≈ 5×3×24×(#survivors) — a few hours. **No sweep runs until
this plan is approved.**

---

## 7. Threats to validity / honest caveats

- **Network-size floor.** v65 found the net is already minimal and training time "has converged at this
  size"; if the critic is the bottleneck, better *coverage* may help accuracy but not wall-clock. We report
  both so we don't oversell a speed win that isn't there.
- **Kernel interaction.** The deterministic target already kills target variance; a grid that changes the
  state distribution could shift the warm-start (`calibrate_bias`) — re-validate the closed-form warm-start
  under each sampler.
- **Quantization bias.** A grid is biased at finite N (unlike MC, which is unbiased); the OOS price is still
  evaluated on *MC* test paths, so the policy is what's judged — but we must check the grid doesn't induce a
  systematically *different* (not just better) policy. The Δ%-vs-LSM and bang-bangness cross-checks guard
  this.
- **"Higher price" misread.** Guard with the LSM-D Δ% and net-payoff decomposition so a higher number is
  always a better *policy*, never a leak.
- **Bit-identity gate.** `--sampler mc` must reproduce current v65 exactly, or the comparison baseline is
  contaminated.

## 8. Deliverables
1. `--sampler`/`--step_density` in the C++ pricer (+ `mc` bit-identity test).
2. `tools/collect_sampler_results.py` → tidy CSV; a notebook section (candidate: extend Notebook 3's
   "Sample efficiency" R2 panel, or a new "Validation 4: scenario efficiency") with the 2-D curves and the
   stat tests.
3. An HPT.md "v66 sampling" section with the adopt/reject verdict per arm.
