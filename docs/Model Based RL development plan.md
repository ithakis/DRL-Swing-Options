# Model-Based RL Development Plan — A Next-Generation Swing-Option Pricer

> **Audience.** This document is written to be executed by an autonomous coding agent (Claude Opus 4.8) *and*
> read by a quant researcher. It is self-contained: it states the math, the hypotheses, the experiment
> protocol, the statistical decision rules, and the exact repo touch-points. The companion file
> `Literature Review - Model-Based RL for Swing Options.md` holds the grounded equation dump and references
> (cited here as **[LR §x]**). The final section is a ready-to-paste **autonomous research prompt**.
>
> **One-sentence thesis.** The swing problem in this repo is low-dimensional with a *fully known* transition
> kernel; therefore the value function can be obtained by **backward dynamic programming on a single
> recombinant grid using the analytic kernel for expectations** — which is (i) orders of magnitude faster than
> the current actor-critic RL, (ii) at least as accurate, and (iii) end-to-end differentiable (greeks,
> hedging, portfolios). The research program is to build this, prove it competitive across the 4 canonical
> regimes and many seeds, and fold a single value/policy network into it for generalisation and the
> portfolio/hedging extension.

---

## Table of Contents
1. Current solution — architecture, strengths, limitations
2. What the user's ideas get right (and where to sharpen them)
3. Mathematical foundation (the object we are computing)
4. The method ladder M0–M5 (definitions, math, why faster/more accurate)
5. Profiling & performance plan (find the cost, set the targets)
6. Experimental protocol (4 regimes, seed ladder, statistics, oracles)
7. Hypothesis register with if-this-then-that branches
8. Validation: dual upper bounds & grid-convergence
9. Staged roadmap & milestones
10. Extensions: greeks, hedging under TC, portfolio super-pricer
11. Risk register & kill criteria
12. Deliverables & repo integration
- Appendix A: exact equations & pseudocode
- Appendix B: file/representation conventions to reuse
- Appendix C: **Autonomous research prompt for Claude Opus 4.8**

---

## 1. Current solution — architecture, strengths, limitations

### 1.1 What it is (the "latest version")
A CPU-only **DDPG-lineage actor-critic** (`src/agent.py`, `src/networks.py`) pricing a convex-cost swing on
the HHK spot model, benchmarked against **LSM** (`src/lsm_swing_pricer.py`). The decisive recent change is the
**semi-analytical kernel** (`src/transition_kernel.py`, `--use_expected_target=1`): the critic's TD target is
replaced by a quadrature **expectation over the one-step HHK transition kernel**:
$$
y \;=\; r + \gamma\sum_{m=1}^{M} w_m\, Q_{\bar\theta}\big(z'_m,\ \mu_{\bar\theta}(z'_m)\big),\qquad
M = M_x\cdot M_y,\ \ M_y = 1+N_{\max}M_{\text{per\_k}} ,
$$
with $M_x$ Gauss–Hermite nodes on $X'$ and a Sobol-stratified jump-decay mesh on $Y'$. Empirically (HPT.md):
$M_x\!\ge\!2$ is a hard accuracy plateau; the deterministic target made PER, target-policy-noise, n-step, twin
critics, IQN, Dyna, antithetic, jump-IW all **redundant or dead**; the canonical adds closed-form FOC actor
warm-start, linear noise decay, eval-only EMA, single critic step, constant LR ($lr_a{=}3e{-}4,lr_c{=}6e{-}4$),
2×64 SiLU+LayerNorm nets, β-sigmoid(3) output, profitability-gate STE.

**Canonical regimes** (focal contract: $K{=}1$, $T{\approx}1$ month, $N{=}22$ dates, $\overline q{=}2$,
$Q_{\max}{=}20$, no refraction; HHK $\alpha{=}12,\sigma{=}1.2,\beta{=}150,\lambda{=}6,\mu_J{=}0.3$):
`cc_g1` ($\gamma{=}1$), `cc_g15`, `cc_g2` (focal), `nocost`. Metric **Δ% = (RL/LSM − 1)·100**.

### 1.2 Strengths (keep these)
- **S1 — Analytic kernel target.** Variance-reduced, near-deterministic TD target; the single most valuable
  idea already in the repo. Bit-identical-off design, disk-cached, numba hot path, float64, C++-portable. **[LR §4,§8]**
- **S2 — Domain-informed policy.** The convex-cost FOC $a^\star(s)=\mathrm{clip}(((S{-}K)^+/(c\gamma))^{1/(\gamma-1)},\underline q,\overline q)$
  is used both as the actor warm-start and as the profitability-gate STE — a strong analytic prior. **[LR §1]**
- **S3 — Rigorous experimental culture.** Pre-registration, 4 regimes, 12–24 seeds, Welch/Levene/Holm,
  conservative-Pareto selection, harnesses (`tools/sweep_*.py`, `tools/stats_analysis.py`). This is reusable
  infrastructure — the new methods plug straight in.
- **S4 — Strong, honest benchmark.** LSM (and LSM-D discretised-action, Chebyshev full-state) with net-profit
  gating; price-vs-paths convergence study confirms the evaluator is unbiased $1/\sqrt N$. **[LR §2]**
- **S5 — Greeks already exist.** CRN bump-and-revalue Δ/Γ (`src/greeks.py`) exploiting the HHK CRN identity
  $\partial_{\log S_0}\log S_t=e^{-\alpha t}$; daily-rebalanced regression hedge. **[LR §10]**

### 1.3 Limitations (what the new program attacks)
- **L1 — One-step bootstrap, not backward induction.** The kernel integrates a *single* transition; the value
  still propagates by a *self-referential fixed point* (critic learns from a moving target). The information is
  available to integrate **all the way to maturity by backward induction** (only $N{=}22$ steps). This is the
  central inefficiency: **$\epsilon_{\text{optimisation}}$ dominates the residual gap** (HPT.md: accurate $M{=}36$
  ≈ fast $M{=}4$; depth/updates/softer-squash helped — i.e. the *optimiser*, not the integrator, is the limiter). **[LR §13]**
- **L2 — Online, per-step, single-threaded rollout.** Training is a Python `for current_path … while True`
  env loop (`run.py:1112`), one `agent.act`/`agent.step` per decision date, with a replay buffer. ~190 s/run
  even at 4096 ep. The env rollout and replay are pure overhead for a problem whose dynamics are known in closed form. **[LR §0]**
- **L3 — Two coupled networks + replay + target nets + noise schedule + warmup.** Much surviving machinery
  (critic warmup, EMA, noise decay, soft target updates, calibration) exists only to stabilise a *bootstrapped*
  critic. Backward induction removes the moving target and most of this scaffolding.
- **L4 — Under-uses problem structure.** State is effectively $(X,Y,Q^{\text{rem}},t)\in\mathbb R^2\times[0,Q_{\max}]\times\{0..N\}$;
  the transition is **time-homogeneous** given fixed $\Delta t$ (so the transition-weight matrix is built once).
  None of this is exploited — the agent re-discovers the value by simulation. **[LR §3,§0]**
- **L5 — No certified bound.** Δ% is measured against LSM (itself lower-biased). No **dual upper bound**, so we
  cannot say how close to *optimal* we are — only how we compare to another approximate method. **[LR §9]**
- **L6 — Seed dispersion is the floor.** ~0.4% irreducible seed-to-seed price spread that does not shrink with
  eval paths (HPT.md price-vs-paths). A deterministic backward-DP method has **no seed dispersion at all** in the
  grid limit — a structural win on the exact axis the project says is the remaining lever.

> **Diagnosis.** The repo has already proven (HPT.md) that the kernel/integration is *not* the bottleneck; the
> actor-critic *optimisation* is. The fix is to stop bootstrapping and start **inducting backward** on the grid
> the kernel already defines.

---

## 2. What the user's ideas get right (and where to sharpen them)

The user proposed: *a deterministic recombinant grid (FDM/lattice-like, few points), reused each iteration;
price the grid with the NN; use the transition kernel for expectations to get the slope the actor improves
toward; one NN, one grid, iterate to convergence; a model-based RL solution, novel and competitive.*

| User's idea | Verdict | Sharpened form (what to actually build) | Anchor |
|---|---|---|---|
| Deterministic, recombinant grid reused every iteration | **Correct & central** | A *fixed* grid $\Gamma$ in $(X,Y)$ + a **time-homogeneous transition-weight matrix** $P[i,j]$ built once from the analytic kernel. This is **optimal quantization / stochastic mesh**. | [LR §3,§4] |
| "Like FDM, not too many points" | **Correct** | $\mathcal O(N^{-1/d})$ quantization error ($d{=}2$): a few hundred nodes reach near-exact. Cost is a few matmuls. | [LR §3] |
| Replace critic with a math/kernel answer | **Correct, but go further** | Don't just integrate *one* step — **induct backward to maturity**. The exact "critic" is the backward-DP continuation $C_t=e^{-r\Delta t}P\,V_{t+1}$. | [LR §5,§7] |
| "Multiple steps ahead, possibly to completion" | **This is the key upgrade** | Full backward induction $V_T\to V_0$ — no fixed-point bootstrap, no moving target. | [LR §7] |
| Use NN to price the grid; kernel gives the slope to improve the actor | **Correct & novel** | Policy improvement gradient is **exact**: $\nabla_a[\pi(s,a)+e^{-r\Delta t}\,\partial_Q C_t]$ — analytic, no learned critic needed. | [LR §1 FOC, §7] |
| One NN, one grid, iterate to converge | **Correct** | Single date-indexed value/policy network fit on analytic backward targets (**deep backward DP**, Huré–Pham–Warin / Bachouch et al). | [LR §7] |
| Faster & more accurate; novel | **Achievable** | Pure Grid-DP ≈ **milliseconds** (vs ~190 s) and near-exact; the single-NN version generalises + differentiates for greeks/portfolio. Novelty = the *fusion* for convex-cost continuous-control swing with autograd greeks + dual certificate. | [LR §14] |

**Two cautions to encode in the plan:**
- **C-1 (the NN is not needed for the base price).** For *this* low-dim problem, the honest finding may be that
  **pure Grid-DP beats the RL on both speed and accuracy with no network at all**. The plan must test this head-on
  (M1) and report it even if it "demotes" the NN. The NN earns its place via **generalisation across contracts/
  parameters, differentiability for greeks, and the portfolio extension** — not by beating Grid-DP on a single price.
- **C-2 (recombination needs a *fixed* grid).** A standardised grid that *shifts with the conditional mean* per
  node is **not** recombinant (it re-meshes every step). True recombination = one fixed grid + transition matrix.
  Because HHK is time-homogeneous with constant $\Delta t$, that matrix is **identical at every step** — build once.

---

## 3. Mathematical foundation

**State & dynamics.** Markov state $z=(X,Y,Q^{\text{rem}},t,\,\kappa)$ ($\kappa$=cooldown, $\equiv0$ in focal
regime). Spot $S=\exp(f(t)+X+Y)$. Exogenous transition $(X,Y)\to(X',Y')$ has the closed-form law in [LR §1].

**Per-step net payoff (control $a\in[\underline q,\overline q]$, feasibility set $\mathcal A_t(z)$):**
$\pi(S,a)=a\,(S-K)^+-c\,a^{\gamma}$, with $\pi$ floored at 0 (no exercise if net-negative).

**Bellman / backward DP (the exact object):**
$$
V_N(z)=\max_{a\in\mathcal A_N}\pi(S,a),\qquad
V_t(z)=\max_{a\in\mathcal A_t(z)}\Big\{\pi(S,a)+\underbrace{e^{-r\Delta t}\,\mathbb E\big[V_{t+1}(X',Y',Q^{\text{rem}}-a,\,t{+}1)\mid X,Y\big]}_{\text{continuation }C_t(z,a)}\Big\}.
$$
Price $V_0(z_0)$. The expectation is over $(X',Y')$ **only** (inventory transition is deterministic given $a$):
this is why one fixed $(X,Y)$ transition operator suffices. Continuous-control optimum via FOC [LR §1].

**Discretisation choices (the "grid"):**
- **Exogenous grid** $\Gamma=\{(x_i,y_i)\}_{i=1}^{N_\Gamma}$: options are (a) tensor Gauss–Hermite($X$) × jump-mesh($Y$)
  reusing `transition_kernel.py`; (b) **optimal quadratic quantization** of the stationary/marginal $(X,Y)$ law
  [LR §3]; (c) tensor of empirical quantiles. Transition weights $P[i,j]=\mathbb E[\mathbf 1\{(X',Y')\in \text{cell}_j\}\mid (x_i,y_i)]$
  or the soft Gauss–Hermite/jump-mesh weights, computed once (time-homogeneous).
- **Inventory grid** $Q^{\text{rem}}\in\{0,\delta,2\delta,\dots,Q_{\max}\}$, $\delta=\overline q/n_a$ (so $a$ lands on the grid),
  $n_Q=Q_{\max}/\delta+1$ levels. Bang-bang ($\gamma{=}1$, $c{=}0$): $n_a{=}1$ (full lift). Convex: $n_a\in\{4..16\}$.
- **Action set** $\mathcal A_t(z)$: $\{0,\delta,\dots,\min(\overline q,Q^{\text{rem}})\}$, plus the interior FOC point as a candidate.

**Exact backward sweep (vectorised):** for $t=N{-}1\dots0$:
$$
C_t[\,i,Q\,]=e^{-r\Delta t}\sum_j P[i,j]\,V_{t+1}[\,j,Q\,]\quad(\text{one }N_\Gamma\times N_\Gamma\times n_Q\text{ matmul}),\qquad
V_t[i,Q]=\max_{a}\big\{\pi(S_i,a)+C_t[i,\,Q{-}a]\big\}.
$$
This is the entire pricer. Greeks: differentiate $V_0$ w.r.t. $S_0$ (which enters only through $X$'s initial
condition and the grid embedding) — analytic or autograd. **[LR §3,§5,§10]**

---

## 4. The method ladder M0–M5

> Build in order; each is a strict, independently-publishable increment. Gate each with the §6 protocol.

### M0 — Baseline (frozen control)
The current kernel-on actor-critic canonical. **Do not modify.** It is the control arm for every comparison and
the reproducibility anchor. Reuse `tools/sweep_v63_audit.py`/`sweep_mega.py` to reproduce focal `cc_g2`
$\Delta\%\approx-0.39\pm0.07$ (M5) before touching anything.

### M1 — **Grid-DP** (pure backward induction on the recombinant grid; *no network*) — *the speed/accuracy oracle*
Implement Appendix-A pseudocode. New module `src/grid_dp.py`:
- `build_exo_grid(hhk, dt, spec)` → nodes $\Gamma$, weights $P$ (reuse `transition_kernel.precompute_kernel`; the
  $P$ matrix is the kernel mesh evaluated on a **fixed node set**, time-homogeneous → built once).
- `backward_dp(contract, grid)` → arrays $V_t[i,Q]$, policy $a^\star_t[i,Q]$.
- `price(z0)` by interpolating $V_0$ at $(X_0,Y_0,Q_{\max})$; **forward simulate the greedy policy** on the *same
  OOS path pool* as LSM/RL for an apples-to-apples Δ% and a lower-bound price.

**Why faster:** no env rollout, no replay, no SGD, no seeds — a handful of matmuls. Expect **$10^2$–$10^4×$**
speedup (target: focal price in **< 1 s** vs ~190 s). **Why accurate:** quantization/quadrature error
$\mathcal O(N_\Gamma^{-1/2})$ with controllable constant; converges in grid resolution (§8). **[LR §3,§4]**

**This M1 likely becomes the new ground-truth benchmark** (replacing/augmenting LSM) and a production-grade
pricer in its own right. Honest reporting (C-1) required.

### M2 — **Single-network backward DP** (deep backward fitted value/policy iteration) — *the novel "model-based RL"*
One date-indexed network (or one time-conditioned network) $\mathcal V^\theta_t(z)$ and/or policy
$\mathfrak a^\eta_t(z)$, trained **backward** $t=N{-}1\dots0$ on **analytic kernel targets** (Huré–Pham–Warin
"hybrid", Bachouch et al, Lapeyre–Lelong) **[LR §7]**:
$$
\theta^\star_t=\arg\min_\theta\ \mathbb E_{z\sim\nu_t}\Big|\mathcal V^\theta_t(z)-\big[\max_a(\pi(S,a)+e^{-r\Delta t}\,\widehat{\mathbb E}\,\mathcal V_{t+1}(z'))\big]\Big|^2,
$$
warm-start $\theta_t\leftarrow\theta_{t+1}$; sampling distribution $\nu_t$ = grid nodes ∪ simulated states.
Policy by the **exact improvement gradient** (no learned critic):
$\mathfrak a^\star_t=\arg\max_a[\pi+e^{-r\Delta t}\partial_Q\widehat{\mathbb E}\mathcal V_{t+1}]$ (FOC closed form
for convex cost). **One NN, the kernel, backward sweep, no replay, no target net, no noise schedule.** This is
the literal realisation of the user's "one NN + one grid + kernel slope, iterate to converge," with convergence
theory. Differentiable → greeks/portfolio.

### M3 — **Exact-critic policy iteration** (keep the actor, replace the critic with Grid-DP)
Howard policy iteration: (i) **policy evaluation** = backward DP of the *current* actor's value on the grid
(exact, kernel-based, no bootstrap); (ii) **policy improvement** = one actor SGD epoch against that exact value;
repeat. This is the minimal-surgery upgrade of M0 — same actor, but the critic is now the analytic kernel-DP
value rather than a self-bootstrapped NN. Tests whether the *actor representation* alone (with a perfect critic)
matches Grid-DP. **[LR §7,§8]**

### M4 — **Differentiable Grid-DP** (pathwise policy/greeks through the recursion)
Make the M1 backward recursion differentiable in (a) a smooth policy's parameters and (b) market inputs $S_0$,
using a **soft-max / smooth-max** over actions for gradient flow. Yields (i) policy gradient ascent through the
DP (an exact, low-variance alternative to DPG) and (ii) **greeks in one autograd backward pass** (AAD-style,
[LR §10]). Optional/advanced; gate behind M1–M2.

### M5 — **Flagship synthesis** ("Grid-anchored single-network backward DP, certified")
Combine: M1 grid → exact targets; M2 single network distilled for generalisation + differentiability; §8 dual
upper bound for a **certified price interval**; §10 greeks/hedging/portfolio on the shared grid. This is the
deliverable that is faster *and* more accurate *and* novel *and* extensible.

**Decision logic across the ladder:**
- *If M1 ≥ LSM accuracy at ≪ cost (expected):* adopt M1 as benchmark/oracle and as a production pricer; proceed
  to M2 for the NN-generalisation story.
- *If M2 matches M1 within seed-noise at competitive cost:* M2 is the headline "model-based RL" result.
- *If M2 < M1 but M3 (exact critic) ≈ M1:* the gap is the *actor representation*, not the critic → report that
  the value side is solved and the policy parametrisation is the remaining lever.
- *If even M3 < M1:* the NN function class is the limiter → enlarge/curate features (Chebyshev/B-spline head,
  [LR §5]) or accept Grid-DP as the production method and frame the NN purely as the portfolio/greeks vehicle.

---

## 5. Profiling & performance plan

**Goal.** Quantify where M0 spends time, set targets, and prove M1/M2 hit them.

**P1 — Profile M0.** Use `python -m cProfile -o prof.out run.py …` (and the existing `--profile_eval`) on a
1024-ep `cc_g2` run; also `torch.profiler` around `agent.learn_`. Expected cost centres: (a) Python env
rollout `SwingOptionEnv.step` + `agent.act` per decision date; (b) replay `sample`; (c) `expected_critic_target`
B×M forward passes; (d) soft updates. Record wall-clock breakdown to `logs/_profile/m0_breakdown.json`.

**P2 — Microbench the kernel target.** Time `expected_critic_target` vs grid matmul $P V$ at matched accuracy.
Hypothesis: the backward matmul is cheaper *and* removes the per-step rollout entirely.

**P3 — Targets (pre-registered):**
- M1 focal price (grid converged) in **< 1 s** single-thread; full 4-regime sweep in **< 1 min**.
- M2 trained focal value/policy in **< 20 s** (vs ~190 s M0 at 4096 ep) at equal-or-better Δ%.
- Seed dispersion: M1 **= 0** (deterministic); M2 **≤ ½** of M0's ~0.4%.

**P4 — C++-portability ledger.** Every adopted component must be expressible as fixed-shape float64 array ops
(matmul, gather/scatter for the inventory shift, a 1-D `max` over actions). Maintain `docs/cpp_port_notes.md`.

---

## 6. Experimental protocol (reuse the existing rigor)

**Regimes (4):** `cc_g1` ($\gamma{=}1,c{=}0.04$, binding/bang-bang), `cc_g15`, `cc_g2` (focal, $\gamma{=}2$),
`nocost` ($c{=}0$). Focal contract & HHK params per §1.1.

**Seed ladder (escalation gate):**
1. **Smoke:** focal `cc_g2`, seeds {11,12,13}, smallest grid/network. *Pass* → continue.
2. **4-regime × 12 seeds {11–22}.** Primary inference. Stats: Welch $t$ (mean), Levene (variance), paired-seed
   $t$ vs M0 on the **shared OOS pool**, Holm–Bonferroni across regimes. (All in `tools/stats_analysis.py`.)
3. **Confirm leader × 24 seeds {11–34}** + a 16384-ep / fine-grid spot check. Select by **conservative Pareto**
   (mean Δ%, seed-std, worst-seed) exactly as `tools/stats_analysis.py` already does.

**Oracles / references for Δ%:**
- LSM and **LSM-D** (discretised-action, Chebyshev full-state) — existing.
- **Grid-DP (M1)** at high resolution — new near-exact primal oracle.
- **FD-HJB** on focal (one-off, §8) and **dual upper bound** (§8) → a *certified interval*; report Δ% vs the
  interval midpoint and whether each method lies inside it.

**Metric & success (pre-registered, focal `cc_g2`, M=5/M=17 discretisation):** a method "wins" if its Δ% vs the
certified interval is $\ge$ M0's **and** seed-std $\le$ M0's **and** wall-clock $\le$ M0's, with **no guard
regression** ($\le0.15$ pp worse) in `cc_g1/cc_g15/nocost`.

**Harness.** New `tools/sweep_modelbased.py` mirroring `sweep_v63_audit.py` (arm families M1/M2/M3/M4, `--resume`,
CSVs to `logs/_sweep_modelbased/` in the `stats_analysis` schema). Notebook
`Jupyter Notebooks/9: Model-Based DP vs RL.ipynb` for the writeup.

**Iterate-on-results discipline.** For each method: (i) smoke → (ii) inspect failure modes (grid coarseness,
inventory aliasing, FOC clipping, NN underfit) → (iii) fix and re-smoke → (iv) escalate. Log every round in
`HPT.md` under a new "Model-Based DP campaign" section with the same table style.

---

## 7. Hypothesis register (if-this-then-that)

Each hypothesis has a **prediction**, a **test**, and a **branch**. Pre-register success thresholds before running.

**H-A — Backward DP removes the optimisation gap.**
*Predict:* M1 Grid-DP focal Δ% ≥ 0 vs LSM and inside the dual interval; seed-std = 0.
*Test:* M1 at $N_\Gamma\in\{64,128,256,512\}$, $n_a\in\{4,8,16\}$; grid-convergence (§8).
*If true →* adopt M1 as oracle; proceed to M2. *If false (Δ% < LSM) →* the discretisation is too coarse or the
inventory/action grid aliases the FOC; refine $n_a$, add the interior FOC candidate action, check the no-exercise
floor; if still failing, suspect a kernel-weight normalisation bug (cross-check $\sum_j P[i,j]=1$ and against MC).

**H-B — One network suffices (model-based RL headline).**
*Predict:* M2 matches M1 within seed-noise at ≤ M0 wall-clock, 4 regimes / 12 seeds.
*If true →* headline result; escalate to 24 seeds + fine grid. *If M2 < M1 →* go to H-C.

**H-C — The gap (if any) is the actor representation, not the critic.**
*Test:* M3 exact-critic policy iteration. *If M3 ≈ M1 →* report "value problem solved; policy parametrisation is
the lever"; try richer policy heads (B-spline/Chebyshev in $z$, [LR §5]) or directly output the FOC residual.
*If M3 < M1 too →* NN function class limits; either enlarge curated features or accept Grid-DP as production.

**H-D — Time-homogeneous transition matrix is reusable across all dates.**
*Predict:* one $P$ built once gives the same accuracy as per-date kernels.
*Test:* compare per-date kernel vs single $P$ on focal. *If true →* large precompute saving; *if false →*
seasonality/$\Delta t$ varies → build a small per-date family (still cheap).

**H-E — Optimal quantization beats tensor Gauss–Hermite at equal $N_\Gamma$.**
*Test:* M1 with grid = (a) GH×jump-mesh vs (b) quadratic-optimal quantizer of the $(X,Y)$ marginal [LR §3].
*If (b) better at equal cost →* adopt; *else* keep GH (already in repo, simpler).

**H-F — Greeks come for free and beat CRN-bump.**
*Test:* autograd/AAD Δ,Γ through M1/M4 vs `src/greeks.py` CRN bump on the closed-form-checkable cases
(`tools/test_greeks.py`). *If lower variance / same value →* adopt; enables the hedging extension. **[LR §10]**

**H-G — A certified interval is achievable.**
*Test:* implement the additive dual (Rogers/Haugh–Kogan; Bender multiple-exercise) using M1's value to define
the martingale [LR §9]; report primal–dual gap on focal. *If gap small →* the paper claims become certified.

**H-H — Coarser grid + NN smoothing beats fine grid alone (accuracy/speed Pareto).**
*Test:* M2 trained on a coarse grid vs M1 fine grid at matched wall-clock. *If NN interpolation recovers fine-grid
accuracy from coarse targets →* the NN earns its keep on speed too (not only generalisation).

**Anti-hypotheses to actively try to kill (avoid self-deception):**
- *"NN needed for the base price"* — H-A/H-B may refute it; report honestly (C-1).
- *"More $M_x$/grid always helps"* — HPT.md says $M_x$ plateaus; expect a grid plateau too; find it, don't overspend.
- *"Backward DP is exact"* — it is exact only in the grid/quadrature limit; quantify $\epsilon_{\text{integration}}$ (§8).

---

## 8. Validation: dual bounds & grid-convergence

**V1 — Grid-convergence (Richardson).** Price M1 at $N_\Gamma$ and $2N_\Gamma$; the change bounds
$\epsilon_{\text{integration}}$; Richardson-extrapolate. Report a convergence table (price vs $N_\Gamma$, $n_a$).
Quantization theory predicts $\mathcal O(N_\Gamma^{-1/2})$ in 2-D [LR §3].

**V2 — Cross-oracle agreement.** Focal price must agree across {M1 fine grid, LSM-D high-degree, FD-HJB} within
their CIs. Build a minimal **FD-HJB** solver for the focal config (semi-Lagrangian jump term, policy iteration
for $\sup_a$) as a one-off third oracle [LR §6].

**V3 — Dual upper bound (certificate).** Additive dual $V_0\le\mathbb E[\max_t(e^{-rt}g_t-M_t)]$ with $M$ from
M1's value; for the multiple-exercise structure use Bender's multiple-exercise dual [LR §9]. Report the
primal–dual interval; this is the single biggest credibility upgrade for the paper (turns "−0.39 pp vs LSM"
into "inside a certified ±x pp band of the true price").

**V4 — Policy sanity.** The recovered exercise policy must (i) honour the FOC interior optimum where unconstrained,
(ii) reproduce the bang-bang fraction "B" trend (~66% at $\gamma{=}1$ → ~21% at $\gamma{=}2$), (iii) respect
$Q_{\max}$/refraction. Visualise $a^\star_t(X,Y,Q)$ heatmaps.

---

## 9. Staged roadmap & milestones

| Stage | Work | Exit criterion |
|---|---|---|
| **0. Repro** | Reproduce M0 focal; profile (P1–P2) | M0 numbers match HPT.md; cost breakdown logged |
| **1. Grid-DP (M1)** | `src/grid_dp.py`; grid-convergence (V1); Δ% vs LSM on shared pool | H-A passes; price < 1 s; converges in $N_\Gamma$ |
| **2. Oracles** | FD-HJB (V2) + dual bound (V3) on focal | Certified interval on focal; cross-oracle agreement |
| **3. Single-NN DP (M2)** | `deep backward DP`; 4-regime/12-seed | H-B decided; wall-clock ≤ M0 |
| **4. Diagnosis (M3/M4)** | exact-critic PI; differentiable DP | H-C/H-F decided; greeks validated |
| **5. Flagship (M5)** | synthesis + 24-seed confirm + notebook + HPT writeup | Conservative-Pareto winner chosen; paper-ready |
| **6. Extensions** | greeks/hedging/portfolio (§10) | Δ/Γ + TC hedge + 2-contract book on one grid |

---

## 10. Extensions (the long-horizon vision, designed in from day 1)

- **E1 — Greeks by AAD.** Because M1/M2/M4 are differentiable, Δ,Γ,vega,$\partial_c$,$\partial_\gamma$ come from
  one autograd backward pass (cheap-gradient principle, [LR §10]). Validate against `src/greeks.py` CRN bump
  (`tools/test_greeks.py`). Replaces 5-point stencils with exact sensitivities.
- **E2 — Hedging under transaction costs.** Plug the differentiable pricer into a **deep-hedging** objective
  $\min_\theta\rho(-Z+\sum_k\delta^\theta_k\Delta F_k-\sum_k c_{\text{tx}}(\delta^\theta_k))$ [LR §11]; the swing's
  forward-martingale hedge already in `hedging_utils.py` is the frictionless special case.
- **E3 — Portfolio super-pricer.** Value a *book* on the **same shared grid** in one backward sweep: stack
  contract-specific payoffs as extra value channels $V^{(k)}_t[i,Q^{(k)}]$; portfolio price + all cross-greeks
  fall out together. The recombinant grid is the enabling structure (one $P$, many payoffs). **[LR §3,§14]**

---

## 11. Risk register & kill criteria

| Risk | Symptom | Mitigation / kill |
|---|---|---|
| Inventory aliasing | M1 price non-monotone in $n_a$ | add interior-FOC candidate action; finer $\delta$; smooth-max (M4) |
| Kernel-weight bug | $\sum_jP[i,j]\ne1$; price off vs MC | unit test weights; cross-check single-step vs `transition_kernel` |
| Grid blow-up in 2-D corners | extreme-$Y$ jump nodes dominate | clip/curate node set (mirror `CuratedFeatures` scales) |
| NN underfit (M2) | M2 ≪ M1 across seeds | richer head (Chebyshev/B-spline), backward warm-start, more nodes |
| Over-claiming novelty | reviewer cites Bardou/Kirkby | position as *fusion*: convex-cost continuous control + single NN + kernel + autograd greeks + dual cert |
| Scope creep | extensions before base proven | **gate E1–E3 behind M1–M2 passing §6** |
| **Honest-negative** | Grid-DP beats NN outright | report it (C-1); reframe NN as generalisation/greeks/portfolio engine, not price-beater |

**Hard kill criteria:** abandon a method arm if, after one fix-and-retry round, it fails its pre-registered
smoke threshold on focal `cc_g2` *and* shows no path to the guard regimes.

---

## 12. Deliverables & repo integration

- `src/grid_dp.py` (M1), `src/deep_backward_dp.py` (M2/M3), optional `src/diff_dp.py` (M4); all CPU/float64,
  reusing `transition_kernel.py`, `swing_contract.py`, `agent_evaluation.py` evaluation conventions.
- `tools/sweep_modelbased.py`, `tools/test_grid_dp.py` (grid-convergence, weight-normalisation, vs-LSM, vs-MC,
  greeks-vs-closed-form), `tools/dual_bound.py`.
- `Jupyter Notebooks/9: Model-Based DP vs RL.ipynb` (speed/accuracy Pareto, certified interval, policy heatmaps).
- HPT.md "Model-Based DP campaign" section (same table discipline). Update CLAUDE.md architecture table.
- **Bit-identical guard:** M0 path untouched; new methods behind explicit `--method {ac_kernel,grid_dp,deep_bdp}`.

---

## Appendix A — Exact equations & pseudocode

**A.1 Transition weights (time-homogeneous, build once).**
```
# Reuse src/transition_kernel.precompute_kernel for nodes/weights.
# Fixed exo grid Γ = {(x_i,y_i)}; for each source node i, the kernel gives
# next-state nodes z'_m and weights w_m (Gauss-Hermite_X × jump-mesh_Y).
# Recombine onto Γ: P[i, :] = projection/interpolation of (z'_m, w_m) onto Γ.
# Check: row-sums == 1 (renormalise tiny drift). HHK is time-homogeneous → one P for all t.
```

**A.2 Backward DP (M1).**
```
for Q in inventory_levels: V[N, :, Q] = max_a∈A(Q)  π(S_i, a)          # terminal
for t = N-1 .. 0:
    C[:, :] = exp(-rΔt) * (P @ V[t+1])                                 # (N_Γ × N_Γ)·(N_Γ × n_Q)
    for i, Q:
        V[t,i,Q]   = max over a∈A_t(i,Q) of [ π(S_i,a) + C[i, Q - a] ] # inventory shift = gather
        astar[t,i,Q] = argmax
price = interp(V[0], (X0,Y0,Q_max))
# OOS lower bound: forward-simulate astar on the shared eval pool, average discounted net cashflows.
```

**A.3 Single-network backward DP (M2).**
```
init θ_N from terminal payoff; for t = N-1 .. 0:
    sample z ~ ν_t  (grid nodes ∪ simulated states)
    target(z) = max_a [ π(S,a) + exp(-rΔt) * Σ_m w_m V_{θ_{t+1}}(z'_m(z,a)) ]   # kernel expectation
    θ_t = argmin_θ E | V_θ(z) - target(z) |^2     (warm-start θ_t ← θ_{t+1})
    # policy: a*_t(z) = argmax_a [...] ; for convex cost use FOC closed form + boundary check
```

**A.4 Continuous-control FOC (interior optimum).**
$a^\star=\mathrm{clip}\big(((S-K)^+/(c\gamma))^{1/(\gamma-1)},\underline q,\overline q\big)$, compared against the
continuation-adjusted optimum $\arg\max_a[\pi(S,a)+\partial_Q C\cdot a]$ (the continuation slope $\partial_Q C$
shifts the marginal value of volume).

**A.5 Dual bound (V3).** $V_0\le \frac1L\sum_{\ell}\max_t\big(e^{-rt}g_t^{(\ell)}-M_t^{(\ell)}\big)$, $M$ the
discrete martingale from M1's value increments; multiple-exercise version via Bender [LR §9].

## Appendix B — Conventions to reuse
- State layout = `transition_kernel.OBS_IDX_*` (S−K, …, S, X, Y, …). Strike-scaled units ($K{=}1$).
- Evaluation on the **shared OOS pool** (seed 999, ≥65,536 paths) exactly as `agent_evaluation`/`rebuild_results_v7`.
- Stats via `tools/stats_analysis.py` (Welch, Levene/Brown–Forsythe, paired-seed $t$, Holm–Bonferroni, conservative Pareto).
- Regimes/contract/HHK params per §1.1; β-sigmoid/robust-norm only matter for the M0 actor.

---

## Appendix C — Autonomous research prompt for Claude Opus 4.8

> Paste the block below as the opening instruction to a fresh Claude Code (Opus 4.8) session rooted at the repo.
> It is self-contained and references this plan + the literature review as its specification.

```
ROLE
You are a senior quantitative researcher + numerical-optimisation engineer working in the
DRL-Swing-Options repository. You price convex-cost swing options on the Hambly–Howison–Kluge
(HHK) spot model. Your job: execute the research program in
"Model Based RL development plan.md" (the PLAN) end-to-end, autonomously, with statistical rigor,
grounding every method in "Literature Review - Model-Based RL for Swing Options.md" (the LITREV).

PRIME DIRECTIVE
Build and prove a next-generation pricer that is FASTER and AT LEAST AS ACCURATE as the current
kernel-on actor-critic (M0), by replacing one-step TD bootstrapping with BACKWARD DYNAMIC
PROGRAMMING on a single recombinant (X,Y)×inventory grid that uses the analytic HHK transition
kernel for expectations — first with no network (M1 Grid-DP), then with a single value/policy
network (M2 deep backward DP), then diagnose/fuse (M3–M5). Honest negatives are required: if pure
Grid-DP beats the NN on price, say so and reposition the NN for generalisation, greeks, and the
portfolio extension.

GROUND RULES
- Read first, in order: CLAUDE.md, HPT.md (esp. the kernel + v63 + mega sections), the PLAN, the
  LITREV, then src/{transition_kernel,agent,networks,swing_env,lsm_swing_pricer,greeks}.py.
- Do NOT modify the M0 path. Add methods behind --method {ac_kernel,grid_dp,deep_bdp}. Keep CPU/float64.
- Reuse: transition_kernel.precompute_kernel, the shared OOS eval pool (seed 999), agent_evaluation
  conventions, tools/stats_analysis.py (Welch, Levene, paired-seed t, Holm–Bonferroni, conservative Pareto).
- Statistics gate (escalation ladder): smoke {seeds 11–13, focal cc_g2} → 4 regimes × 12 seeds {11–22}
  → confirm leader × 24 seeds {11–34} + a fine-grid/16384-ep spot check. Regimes: cc_g1, cc_g15,
  cc_g2 (focal), nocost. Metric Δ% = (method/LSM − 1)·100, plus Δ% vs the certified interval (V3).
- Pre-register success thresholds in HPT.md (new "Model-Based DP campaign" section) BEFORE each wave;
  log results in the same table style. Use tools/sweep_modelbased.py (mirror sweep_v63_audit.py, with --resume).

EXECUTION ORDER (gate each stage on its exit criterion in PLAN §9)
0. Reproduce M0 focal; profile it (PLAN §5 P1–P2); record cost breakdown.
1. Implement src/grid_dp.py (PLAN §3, Appendix A): fixed (X,Y) grid + time-homogeneous weight matrix P
   from the kernel; backward DP over inventory; greedy-policy OOS lower bound on the shared pool.
   Validate: row-sums of P == 1; grid-convergence (Richardson) in N_Γ and n_a; Δ% vs LSM. (H-A, V1)
2. Build oracles: a minimal FD-HJB focal solver (V2) and an additive dual upper bound (V3, Rogers/
   Haugh–Kogan; Bender multiple-exercise) → a CERTIFIED price interval on focal.
3. Implement src/deep_backward_dp.py (M2, PLAN §4/Appendix A.3): one date-indexed value/policy network
   trained backward on analytic kernel targets (Huré–Pham–Warin/Bachouch/Lapeyre–Lelong). 4 regimes ×
   12 seeds vs M0 (paired-seed). (H-B)
4. Diagnose: M3 exact-critic policy iteration (H-C) and M4 differentiable Grid-DP greeks (H-F), only if needed.
5. Flagship M5 synthesis: 24-seed confirm, conservative-Pareto selection, notebook
   "9: Model-Based DP vs RL.ipynb", HPT.md writeup, CLAUDE.md update.
6. Extensions (only after M1–M2 pass): AAD greeks vs src/greeks.py (tools/test_greeks.py); deep-hedging
   under transaction costs; 2-contract portfolio on one shared grid.

CODING/VALIDATION REQUIREMENTS
- Add tools/test_grid_dp.py: weight normalisation; grid-convergence monotonicity; Grid-DP vs a brute-force
  MC backward induction on a tiny config; greeks vs closed-form where available. Keep
  pytest tools/test_approximators.py and tools/test_greeks.py green.
- Targets (PLAN §5 P3): M1 focal price < 1 s and 4-regime sweep < 1 min; M2 focal < 20 s at ≥ M0 Δ%;
  M1 seed-std = 0; M2 seed-std ≤ ½ of M0's ~0.4%.
- Maintain docs/cpp_port_notes.md (every adopted op = matmul / gather-scatter / 1-D max).

REPORTING
After each wave: a short markdown report (means ± seed-std, worst-seed, wall-clock, Welch/Holm p-values,
Pareto verdict, and the if-this-then-that branch taken from PLAN §7), appended to HPT.md. End with a
recommendation: which method becomes the new canonical/benchmark and why, with the certified interval.

Begin with Stage 0. Think hard, move in small verified steps, and prefer exact/analytic computation over
simulation wherever the HHK kernel makes it possible.
```

---

*Companion: `Literature Review - Model-Based RL for Swing Options.md`. Plan authored for the DRL-Swing-Options
project as the specification for the model-based backward-DP pricer initiative.*
