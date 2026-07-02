# grid_dp_pricer — Development Notes & Lessons Learned

Read this before touching the backward loop, the kernel, or the inner-control solver.  Companion to
`README.md` (the *how to build/run*); this is the *why it's built this way + what to avoid*.

## What this is

A deterministic **grid backward-DP** reference price for the convex-cost swing option, with an
**analytical HHK transition kernel**.  It exists to give LSM-D and RL/D4PG a ground-truth third leg:
on the focal cell `DP ≥ RL-kernel ≥ LSM-D ≥ RL-sample`, with the DP converged to ±5e-5.

## Why it can be near-exact

The contract is **discrete in time** (N=22 decision dates) and the HHK one-step transition is
**exact**, not Euler-discretized:
- `X' = decay_X·X + sigma_X·Z`, `Z~N(0,1)` — exact OU.
- `Y' = decay_Y·Y + Δ`, `Δ = Σ J_i e^{-beta(dt-U_i)}` over `Poisson(lam·dt)` jumps, **independent of
  `Y`** — exact jump-OU.
So there is **zero time-discretization error**.  Remaining error = Gauss-Hermite truncation (spectral),
jump-mesh quadrature, Poisson truncation at `N_max`, `(X,Y,Q)` interpolation, and grid truncation —
each independently convergence-studied (`RESULTS.md`).

## Architecture decisions (and the reasoning)

- **Separable expectation → two GEMMs.** Because `X'`/`Y'` factor and interpolation is linear in the
  value array, `E[U_{j+1}]` collapses to constant transfer matrices `A_X (nX×nX)`, `A_Y (nY×nY)` built
  once: `W[:,:,Q'] = A_X·U[:,:,Q']·A_Yᵀ`.  Value layout is `U[ix][q][iy]` precisely so this is two
  *large* Accelerate GEMMs per step (contract X over `nQ·nY` columns, then Y over `nX·nQ` rows) rather
  than `nQ` tiny ones.  The GEMMs are a small fraction of runtime — the **inner control dominates**.
- **Native kernel, loaded fixture only for parity.** The runtime kernel is built in C++ (Golub–Welsch
  Gauss-Hermite + analytic Gauss-Legendre×Gauss-Laguerre jump mesh) so there is no Python at runtime.
  The scrambled-Sobol jump mesh of `src/transition_kernel.py` is **not** reproduced in C++ (same
  decision as `cpp_pricer`); the 1e-10 parity test instead *loads* the Python mesh and checks the C++
  weighted-sum integrator reproduces Python's integral — validating the integrator, not the RNG.
- **Interpolation split.** Not-a-knot cubic spline in `(X,Y)` (linear-in-values, so its cardinal
  weights can be baked into `A_X/A_Y`).  Shape-preserving **PCHIP** in `Q` for the inner control — it
  does not enter the GEMM, so its non-linearity is free, and it is overshoot-free near the
  budget-constraint kinks.
- **FP64 reference, FP32 optional.** Default build is FP64 (the parity bar is 1e-10).  FP32 agrees to
  ~1e-6 but is only marginally faster *here* because the bottleneck is the scalar inner-control sweep,
  not the BLAS (the opposite of `cpp_pricer`, where tiny-tensor autograd was the cost).

## The bug that mattered: nQ divergence (signed-spline wiggle × non-robust optimizer)

**Symptom.** Refining every axis at once made the price *drift down* (1.987→1.950→1.918); isolating
axes showed **nQ alone** diverging — and *accelerating* with refinement (increments grew), the
signature of a compounding bias, not slow convergence.  Coarse nQ hid it because `q∈[0,q_max]` fit
inside a single smooth spline cell.

**Root cause.** Cubic-spline transfer matrices `A_X/A_Y` have **negative side-lobe weights**, so
`W = A_X·U·A_Yᵀ` is a *signed* combination of Q-concave columns → `W(Q')` is only "almost" concave and
its PCHIP **wiggles** between nodes.  The original inner solver was a `g'`-monotone bisection; the
wiggle gave `g'` spurious sign changes, so it latched onto a wrong local optimum.  A tiny per-step
under-bias then **compounded over 22 backward steps**, growing as finer nQ spanned more wiggly cells.

**Fix.** A globally-robust inner maximizer: scan the **exact grid-node** continuation values (concave
up to negligible noise, immune to between-node wiggle) to bracket the global optimum, then a parabolic
seed + short golden polish inside the winning cell.  Result: nQ now converges monotonically at order
~1 to the same limit as every other axis.  `test_inner_control` carries a deliberately non-concave
"wiggle" continuation as a regression guard.

## What worked (speed)

| change | effect |
|---|---|
| multithread the inner-control sweep over X-rows (per-thread PCHIP/buffers) | ~4× on M1 (bit-identical) |
| parabolic seed + golden(16) instead of golden(32) on the polish | ~2.5× backward, price Δ<1e-7 |
| fast integer powers `pow_q` (gamma∈{1,2,3}) instead of `std::pow` | folded into the above |
| value layout `U[ix][q][iy]` → 2 big GEMMs/step (not nQ small ones) | GEMM never the bottleneck |

## What did NOT work / wasn't worth it

- **FP32 for speed.** Only ~12% faster — the inner-control scalar loop dominates, and the GEMM (where
  FP32 helps) is minor.  Kept as an option, not the default; price agrees to ~1e-6.
- **`g'`-bisection / plain golden as the *only* optimizer.** Not robust to the signed-spline wiggle
  (see the bug above).  The exact-node scan is load-bearing.
- **Widening Y / more jump quadrature to chase the MC.** The early 65k-path MC sat ~1 SE *above*
  backward; widening `Y_hi` 3→6 moved the price only +9e-5 — the apparent gap was MC noise, confirmed
  by a 4M-path MC landing *below* backward (gap +0.002).  Don't over-truncate-tune on a noisy MC.

## Gotchas (don't re-introduce)

1. **The inner optimum is interior, not bang-bang.** Convex cost (`gamma>1`) makes the per-step
   objective strictly concave; using a 2-action (0/q_max) maximizer under-prices badly (this is exactly
   why the canonical LSM-D uses `n_actions=5`, not 2 — a 2-action LSM gives 1.863 vs the canonical
   1.982 at the focal cell).  The gate is enforced by comparing to `q=0`, never by forcing a corner.
2. **Discounting is 0-based.** Date `j` contributes `df^j·pi_j` (`df^0=1` at `t=0`), matching
   `calculate_standardized_reward` in `src/swing_env.py`.  `V_0 = U_0` in date-0 money.
3. **Price point is interpolated, not assumed on-grid.** `V_0 = U_0(log S0, 0, 0)`; the default grid
   places 0 on the X and Q grids exactly, but `eval_xyq` interpolates so off-grid `S0` still works.
4. **Block-buffered stdout.** The Python sweep tools print per-axis but stdout is block-buffered to a
   file — output appears only at exit.  Don't mistake a quiet log for a hang.
5. **Accelerate deprecation.** Built with `-DACCELERATE_NEW_LAPACK=1` to use the modern CBLAS headers
   (LP64).  Without it clang warns on every `cblas_*gemm`.
6. **Kernel moment test uses N_max=6.** Production uses `N_max=3` (tail ~1e-8); the moment test pushes
   `N_max` out only to isolate *quadrature* accuracy from the (deliberate) Poisson-tail fold.

## Validation bar (keep it)

- FP64 kernel parity ≤ 1e-10 vs `transition_kernel.py`; Gauss rules exact to 2n-1; spline reproduces
  cubics; inner solve vs brute argmax ≤ 1e-7 (incl. wiggle case); zero-vol DP == exact Lagrangian.
- **Self-consistency**: forward-MC of the DP greedy policy on independent OOS paths converges to
  backward `U_0` **from below** (4M paths: 1.98833 ≤ 1.99033, gap +0.002).
- **Ordering**: `DP ≥ LSM-D` and `DP ≥ RL` on the focal cell and across the `c×γ` grid.
- Re-run `ctest` after any change to `kernel.hpp`, `spline.hpp`, `payoff.hpp`, or `dp.hpp`.

## Ideas for future work

- **Refraction / Q_min** (cooldown axis): mirror the `(cooldown+1)` dimension of
  `src/lsm_swing_pricer.py`; the state and loops are already factored to add it.
- **Greeks**: the DP value surface `U_j(X,Y,Q)` gives Δ/Γ by differencing the price point in `S0` —
  a deterministic alternative to the CRN bump in `src/greeks.py`.
- Push single-threaded `<1e-4` under 1s: the inner control is the only remaining cost; a vectorized
  (SIMD) batch solve over `iy` or a coarser-but-Richardson-corrected nQ are the candidates.
