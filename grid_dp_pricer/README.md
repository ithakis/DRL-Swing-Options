# grid_dp_pricer — deterministic grid backward-DP reference pricer

A self-contained C++ pricer for the **convex-cost swing option** under the Hambly–Howison–Kluge
(HHK) spot model studied in this repo.  It is the **missing third leg** next to the two incumbents:

| pricer | kind | bias |
|---|---|---|
| LSM-D (`src/lsm_swing_pricer.py`) | Least-Squares Monte-Carlo | low-biased (regression + MC) |
| RL / D4PG (`src/`, `cpp_pricer/`) | policy-gradient | policy-suboptimal |
| **this** | **grid backward dynamic programming + analytical HHK kernel** | **near-exact reference** |

The DP is near-exact because the contract is **discrete in time** and the HHK one-step transition is
**exact, not discretized**: `X' = decay_X·X + sigma_X·Z` (Gaussian OU) and `Y' = decay_Y·Y + Δ`
(jump increment, `Δ ⟂ Y`).  The only errors are spatial grid + interpolation + quadrature truncation,
all controllable and convergence-studied.  Both incumbents can finally be measured against it.

Everything lives under this folder; nothing in `src/`, `cpp_pricer/`, or the notebooks is modified.
Python (EP11 env) is used only for cross-checks/fixtures — the pricer itself is pure C++.

## Build

```bash
cmake -S grid_dp_pricer -B grid_dp_pricer/build -DCMAKE_BUILD_TYPE=Release   # Accelerate BLAS, FP64
cmake --build grid_dp_pricer/build -j
```

Options: `-DGRID_DP_FP64=OFF` (FP32 fast path), `-DGRID_DP_ACCELERATE=OFF` (portable GEMM fallback).
The default build is the **FP64 reference / parity** build.

## Run

```bash
# v64 focal contract (c=0.04, gamma=2) at the converged reference grid:
./build/price_dp                       # -> {"price": 1.99033..., timings, ...}
./build/price_dp --c 0.10 --gamma 3    # any convex-cost cell
./build/price_dp --mc 4000000          # + forward-MC self-consistency (greedy policy on OOS paths)
./build/price_dp --threads 1 --nQ 101  # single-threaded, coarser Q grid
```

Key flags: contract `--c --gamma --K --T --n_rights --q_max --Q_max --r`; HHK `--S0 --alpha --sigma
--beta --lam --mu_J`; grid `--nX --nY --nQ --Xlo --Xhi --Ylo --Yhi`; quadrature `--Mx --Nmax --glU
--glagJ`; `--interp {0 linear,1 cubic}`; `--threads`; `--mc N --mc_seed`; `--csv`.

## Method

Backward recursion on a tensor grid `(X, Y, Q)` (Q = cumulative volume used), `df = e^{-r·dt}`:

```
U_{N-1}(x,y,Q) = max_{0<=q<=min(q_max,Q_max-Q)} pi(q; S)                       # terminal
U_j   (x,y,Q)  = max_{0<=q<=min(q_max,Q_max-Q)} [ pi(q; S) + df·W_j(x,y,Q+q) ]
W_j(x,y,Q')    = E[ U_{j+1}(X',Y',Q') | x,y ]                                  # analytical kernel
pi(q) = q·(S-K)^+ - c·q^gamma,   S = exp(x+y),   V_0 = U_0(log S0, 0, 0)
```

- **Separable expectation = two GEMMs.** `X'` depends only on `x`, `Y'` only on `y`, so with constant
  transfer matrices `A_X (nX×nX)`, `A_Y (nY×nY)` built once from the kernel + interpolation weights,
  `W_j[:,:,Q'] = A_X · U_{j+1}[:,:,Q'] · A_Yᵀ` — batched over Q' into two Accelerate GEMMs per step.
- **Analytical kernel** (`include/grid_dp/kernel.hpp`): Gauss–Hermite for the Gaussian `X'`, and a
  Gauss-Legendre × Gauss-Laguerre jump mesh per Poisson count for `Δ` (Poisson tail folded at N_max).
- **Interpolation**: not-a-knot cubic spline in `(X,Y)` (linear-in-values, feeds the GEMM); shape-
  preserving PCHIP in `Q` for the inner control.
- **Inner control** (`include/grid_dp/payoff.hpp`): the per-step objective is concave (convex cost),
  so the optimum is interior.  We scan the exact grid-node continuation (robust to the signed-spline
  wiggle), parabola-refine the winning cell, and polish — matching a brute-force argmax to ~1e-7.  The
  env's profitability gate is enforced by comparison against `q=0`.

## Validation (FP64 parity build)

```bash
( cd grid_dp_pricer/build && ctest --output-on-failure )
```

- `test_gauss` — Gauss-Hermite/Legendre/Laguerre exact on polynomials to 2n-1.
- `test_kernel_parity` — C++ integrator vs `src/transition_kernel.py` mesh, **< 1e-10**
  (fixture from `tools/export_kernel_fixture.py`).
- `test_kernel_moments` — kernel reproduces closed-form `E/Var[X'], E/Var[Δ]`; `A_X/A_Y` row-sums + means.
- `test_spline` — cubic reproduces cubics; PCHIP monotone/overshoot-free + analytic derivative.
- `test_inner_control` — inner solve vs brute-force argmax incl. a non-concave "wiggle" stress case.
- `test_dp_limits` — zero-vol path == exact Lagrangian budget allocation; monotonicities; bang-bang.

Cross-checks (EP11): `tools/{export_kernel_fixture, set_grid_ranges, run_lsm_baseline,
compare_incumbents, dp_grid_sweep, convergence_analysis, pareto}.py`.  See `RESULTS.md`.

## Layout

```
include/grid_dp/  config, linalg (BLAS), gauss, kernel, spline, payoff, dp, hhk_sim, io
apps/             price_dp.cpp (CLI)
tests/            test_{gauss,spline,kernel_moments,kernel_parity,inner_control,dp_limits}
tools/            Python cross-checks (EP11) — not used at runtime
docs/             DEVELOPMENT_NOTES.md, figs/
```

See `docs/DEVELOPMENT_NOTES.md` before extending the hot loop or validation harness.
