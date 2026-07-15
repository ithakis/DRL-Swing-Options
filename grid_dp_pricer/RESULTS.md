# RESULTS — grid backward-DP reference pricer

All numbers are the **v64 focal** contract/HHK unless noted: `K=1, T=0.0833, N=22, q_max=2, Q_max=20,
r=0.05`; HHK `S0=1, α=12, σ=1.2, β=150, λ=6, μ_J=0.3`; cost `c=0.04, γ=2`.  FP64 build, Apple M1.
Reference grid = `nX=nY=121, nQ=151, Mx=24, X∈[-1.6,1.6], Y∈[0,4]` (the binary default).

## Headline

> **Converged DP reference price (focal c=0.04, γ=2): `V₀ = 1.99033 ± 5.4e-5`.**
> Ordering on the focal cell: **DP ≥ RL-kernel ≥ LSM-D ≥ RL-sample**, all incumbents below the DP.
> RL-kernel is within **0.27%**, LSM-D within **0.41%**, RL-sample within **1.56%** of the DP reference.
> The DP reaches ≤1e-3 in **0.28 s** and ≤1e-4 in **1.3 s** (8 threads, M1) — **3–150× faster than the
> incumbents (LSM-D 14.5 s, RL 43 s / 381 s) while being the more accurate leg.**

## 1. Validation — why this is a reference (GOAL 2)

### 1.1 Kernel parity (FP64 ≤ 1e-10)
`test_kernel_parity` loads the exact `src/transition_kernel.py` mesh (M_x=6, N_max=3, M_per_k=8) and
recomputes three analytic integrals; max |Δ| **< 1e-10** (the C++ weighted-sum integrator reproduces
the Python kernel bit-faithfully on identical nodes/weights).  `test_kernel_moments` additionally
confirms the *native* C++ kernel reproduces the closed-form one-step moments `E[X']=decay_X·x`,
`Var[X']=σ_X²`, `E[Δ]=λμ_J(1−decay_Y)/β`, `Var[Δ]=λμ_J²(1−decay_Y²)/β` to ≤1e-9, and that the
transfer matrices `A_X,A_Y` are partitions of unity reproducing the conditional means.

### 1.2 Self-consistency — forward-MC from below (decisive)
Greedy DP policy rolled forward on independent OOS HHK paths (true kernel, env gate), `Σ df^j π_j`:

| MC paths | forward-MC | 95% CI | backward U₀ | backward − MC |
|---:|---:|---|---:|---:|
| 65 536 | 1.99912 | [1.98009, 2.01816] | 1.99027 | −0.0089 (0.9 SE — noise) |
| 4 000 000 | **1.98833** | [1.98589, 1.99078] | **1.99033** | **+0.0020** |

With a tight (4M-path) MC the forward value sits **below** the backward reference (gap +0.002, within
the residual policy-interpolation + quadrature error) — i.e. the greedy-policy MC, a true lower bound
on the optimum, **converges to U₀ from below**, exactly as a valid reference requires.  The earlier
"MC above" was pure 65k-path noise.

### 1.3 Ordering vs incumbents (focal)
LSM-D recomputed fresh (canonical full-state Chebyshev deg-2, **n_actions=5**, fit seed 998 / OOS seed
999, 65 536 paths); RL from the canonical 8-seed `Convex Costs Results 9.csv`.

| method | price | within DP |
|---|---:|---:|
| **DP reference** | **1.99033** | — |
| RL-kernel (D4PG, kernel-on) | 1.98498 | 0.27% |
| LSM-D (n_actions=5, fresh 1.98223 / Results 9 1.98235) | 1.98223 | 0.41% |
| RL-sample (D4PG, no kernel) | 1.95925 | 1.56% |

`compare_incumbents.py` asserts `DP ≥ LSM-D` and `DP ≥ RL` (all pass).  A 2-action (bang-bang) LSM
gives 1.863 — it under-prices badly because convex cost makes the per-step optimum **interior**; this
is why the canonical benchmark uses n_actions=5.

### 1.4 Limiting cases (`test_dp_limits`, all pass)
- **Zero-vol deterministic path** (σ→0, λ=0): DP == exact Lagrangian budget allocation to ≤3e-3.
- **Monotonicities**: price ↓ in `c`, ↑ in `Q_max`, ↑ in `q_max`, ↑ in `σ`.  Price ↓ in `γ` **only
  where optimal lifts are ≥1** (e.g. the focal c=0.04 column): when high cost pushes the optimum
  interior with q<1, `c·q^γ` *decreases* in γ, so the price turns **non-monotone** — on the full grid
  V(c=0.15, γ=3)=1.0997 > V(c=0.15, γ=2)=1.0659.  Economics, not a solver artifact.
- **γ=1 / c=0 corner**: ITM lifts are ≥95% bang-bang (∈{0,q_max}) — the linear-payoff limit.

## 2. Convergence (GOAL 3)

Each axis swept with the others held fine; error vs a Richardson-extrapolated limit; `error ∝ hᵖ`.

| axis | observed order p | residual at reference | note |
|---|---:|---:|---|
| nX | (noise floor) | 4.8e-6 | X support tiny (std 0.18); over-resolved at ±1.6 |
| nY | **≈ 4.75** | 1.6e-6 | not-a-knot cubic spline (4th order) |
| **nQ** | **≈ 1.02** | **3.3e-5** | **controlling axis** — PCHIP/policy in Q is 1st order |
| Mx | (spectral) | 1.5e-5 | Gauss-Hermite; converged by Mx≈12 |

**Controlling axis = nQ** (the cumulative-volume / budget dimension), converging at order ~1.
Richardson over the finest three points gives **V₀ = 1.99033 ± 5.4e-5** (summed residual band).
Data: `data/convergence.csv`; figure: `docs/figs/convergence.png`.

## 3. Time complexity (GOAL 4)

Per backward step the cost is **two Accelerate GEMMs** `O(nQ·(nX²·nY + nX·nY²))` plus the
**inner-control sweep** `O(nX·nY·nQ·κ)` (κ ≈ scan + parabola + golden(16) evals).  Over `N` dates:

```
T_backward ≈ N · [ c_gemm · nQ·(nX²·nY + nX·nY²) + c_inner · nX·nY·nQ·κ ]
Memory     ≈ O(nX·nY·nQ)  (value/continuation buffers);  policy storage adds ×N (forward-MC only).
```

Empirically the **inner control dominates** (the GEMM is never the bottleneck — confirmed by FP32
giving only ~12% speedup), so wall-clock scales ≈ **linearly in the node count nX·nY·nQ**:

| grid (nX/nY/nQ/Mx) | nodes (nX·nY·nQ) | t_total (8 thr) | t_backward (1 thr) |
|---|---:|---:|---:|
| 41/41/51/8   | 8.6e4 | 0.28 s | — |
| 61/61/76/12  | 2.8e5 | 0.55 s | 2.0 s |
| 81/81/101/16 | 6.6e5 | 1.31 s | 5.0 s |
| 121/121/151/24 | 2.2e6 | 4.24 s | 12.8 s |
| 161/161/201/32 | 5.2e6 | 10.7 s | — |

Threading the inner sweep over X-rows gives ~4× on M1 (bit-identical price). Data: `data/pareto.csv`.

**Timing methodology.** The price is fully deterministic (bit-identical across thread counts and
repeat runs — e.g. focal = `1.99032944023` every time); only wall-clock varies with background
system load (macOS daemons can spike a run 5–20×, as `cpp_pricer/DEVELOPMENT_NOTES.md` also warns).
The times reported here are representative low-contention runs; treat them as ±1 significant figure and
prefer the min over repeats when benchmarking.

## 4. Accuracy / time Pareto + head-to-head (GOAL 5)

Cheapest config per accuracy bar (8 threads, M1; error vs the 201/201/251/36 reference):

| bar | config | time |
|---|---|---:|
| ≤ 1e-3 | 41/41/51/8 | **0.28 s** |
| ≤ 1e-4 | 81/81/101/16 | **1.31 s** |
| ~1e-6 | 121/121/151/24 (default) | 4.24 s |

**Head-to-head** (same M1, focal cell):

| pricer | wall-clock | accuracy / bias |
|---|---:|---|
| **DP (≤1e-3)** | **0.28 s** | reference, ±2.6e-4 |
| **DP (≤1e-4)** | **1.31 s** | reference, ±1.6e-5 |
| DP (converged) | 4.24 s | ±1e-6 (±5e-5 incl. all axes) |
| LSM-D (65 536 paths, n_actions=5) | 14.5 s | low-biased −0.41% |
| RL D4PG C++ (4k train + 65k eval) | ~43 s | policy-suboptimal −0.27% (kernel) |
| RL D4PG PyTorch | ~381 s | same |

The DP is **orders of magnitude faster than the incumbents at higher accuracy** — it is both the
cheapest and the most accurate of the three legs.  Recommended **production config = 81/81/101/16**
(≤1e-4 in ~1.3 s) or the default **121/121/151/24** when ±1e-6 is wanted.  Figure: `docs/figs/pareto.png`.

**FP32 fast path**: price agrees with FP64 to ~1e-6 but is only ~12% faster — the inner-control scalar
loop, not the BLAS, is the cost (the opposite of `cpp_pricer`).  FP64 stays the default.

## 5. Full c × γ grid (GOAL 2.5)

DP price for every cell vs the canonical `Convex Costs Results 9.csv` (LSM-D / RL-sample / RL-kernel,
8-seed).  `Δ%` = (DP − method)/DP · 100.  **The DP dominates every incumbent's mean in all 28 cells.**
Full data: `data/dp_grid_sweep.csv` (default grid 121/121/151/24).

| c | γ | DP | LSM-D Δ% | RL-kernel Δ% | RL-sample Δ% |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 1.0 | 2.55489 | 0.09 | 0.24 | 1.03 |
| 0.01 | 1.5 | 2.51301 | 0.09 | 0.22 | 0.90 |
| 0.01 | 2.0 | 2.46138 | 0.10 | 0.26 | 0.88 |
| 0.01 | 3.0 | 2.33428 | 0.27 | 0.24 | 1.53 |
| 0.02 | 1.0 | 2.44558 | 0.07 | 0.21 | 1.06 |
| 0.02 | 1.5 | 2.36964 | 0.10 | 0.24 | 0.89 |
| 0.02 | 2.0 | 2.28514 | 0.17 | 0.26 | 1.27 |
| 0.02 | 3.0 | 2.11579 | 0.59 | 0.19 | 1.43 |
| 0.04 | 1.0 | 2.23859 | 0.09 | 0.27 | 1.24 |
| 0.04 | 1.5 | 2.10963 | 0.18 | 0.25 | 1.04 |
| **0.04** | **2.0** | **1.99033** | **0.40** | **0.27** | **1.56** |
| 0.04 | 3.0 | 1.81649 | 1.32 | 0.16 | 1.38 |
| 0.05 | 1.0 | 2.13903 | 0.06 | 0.21 | 0.94 |
| 0.05 | 1.5 | 1.99123 | 0.24 | 0.28 | 1.26 |
| 0.05 | 2.0 | 1.86477 | 0.55 | 0.24 | 1.75 |
| 0.05 | 3.0 | 1.70396 | 1.67 | 0.15 | 1.24 |
| 0.08 | 1.0 | 1.86257 | 0.03 | 0.19 | 1.35 |
| 0.08 | 1.5 | 1.67543 | 0.50 | 0.26 | 1.70 |
| 0.08 | 2.0 | 1.55140 | 1.12 | 0.30 | 1.47 |
| 0.08 | 3.0 | 1.44800 | 2.70 | 0.16 | 1.20 |
| 0.10 | 1.0 | 1.69621 | 0.10 | 0.24 | 1.16 |
| 0.10 | 1.5 | 1.49416 | 0.73 | 0.21 | 2.31 |
| 0.10 | 2.0 | 1.38348 | 1.59 | 0.26 | 1.31 |
| 0.10 | 3.0 | 1.32201 | 3.38 | 0.17 | 1.14 |
| 0.15 | 1.0 | 1.33060 | 0.01 | 0.20 | 2.28 |
| 0.15 | 1.5 | 1.12660 | 1.57 | 0.28 | 3.14 |
| 0.15 | 2.0 | 1.06588 | 3.16 | 0.13 | 1.23 |
| 0.15 | 3.0 | 1.09969 | 5.05 | 0.17 | 1.29 |

Reading the grid:
- **LSM-D Δ% grows with cost convexity** — ~0.01–0.10% at γ=1 (bang-bang, where LSM-D is near-exact)
  up to **5.05%** at (c=0.15, γ=3).  This is exactly the convex regime where the interior optimum
  departs most from the bang-bang structure LSM-D leans on — and precisely where a true reference was
  missing.
- **RL-kernel is uniformly 0.13–0.30% below the DP across all cells** (near-optimal everywhere), and
  lies within its 8-seed BCa CI of the DP in most cells — i.e. statistically at the reference.
- **RL-sample (no kernel) is 0.88–3.14% below** — the kernel's contribution, now certified against an
  exact reference rather than against the low-biased LSM-D.

## Reproduce

```bash
cmake -S grid_dp_pricer -B grid_dp_pricer/build -DCMAKE_BUILD_TYPE=Release && cmake --build grid_dp_pricer/build -j
( cd grid_dp_pricer/build && ctest --output-on-failure )                 # FP64 validation suite
./grid_dp_pricer/build/price_dp --mc 4000000                              # focal price + self-consistency
conda run -n EP11 python grid_dp_pricer/tools/compare_incumbents.py       # ordering + certified gaps
conda run -n EP11 python grid_dp_pricer/tools/convergence_analysis.py     # convergence + order
python grid_dp_pricer/tools/pareto.py --threads 8                         # accuracy/time Pareto
python grid_dp_pricer/tools/dp_grid_sweep.py                              # full c x gamma grid
```
