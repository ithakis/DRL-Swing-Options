# Prompt — Grid Backward-DP + Analytical-Kernel Swing Pricer (C++)

> Paste everything below the line into a fresh Claude Code (Opus 4.8) session opened at the repo root
> `/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options`.

---

You are a senior quantitative-finance C++ engineer. Build a **deterministic grid backward dynamic-programming pricer, with an analytical HHK transition kernel**, for the swing option with convex exercise costs that this repository studies. The existing pricers are **LSM-D** (Monte-Carlo, low-biased) and **RL/D4PG** (policy-based). Neither is a ground truth. Your DP is meant to be the **near-exact reference price** that both can be measured against — the missing third leg.

Read `CLAUDE.md`, `cpp_pricer/README.md`, and `cpp_pricer/DEVELOPMENT_NOTES.md` first to absorb the model, the canonical contract, and this repo's C++ conventions (CMake + Apple-Accelerate BLAS, FP64 parity philosophy, isolated-folder discipline).

## Hard constraints on how you work
- **Isolated folder.** Put *everything* in a new top-level folder `grid_dp_pricer/` (its own `CMakeLists.txt`, `include/`, `src/`, `tests/`, `tools/`, `docs/`, `README.md`, `RESULTS.md`). **Do not modify anything under `src/`, `cpp_pricer/`, or the notebooks.** You may *read* them and you may call the Python pipeline for cross-checks.
- **Workflow (use these Claude Code features):**
  1. **Start in plan mode.** Produce a concrete implementation + validation plan and call `ExitPlanMode` to get my approval *before* writing code.
  2. Maintain a **todo list** (TaskCreate/TaskUpdate) for the phases below; keep it current.
  3. **TDD:** write the FP64 parity/unit tests *before or alongside* each numerical component.
  4. After implementation, run the **`/verify`** skill (drive the binary, observe real output) and the **`/code-review`** skill on your diff before declaring done.
  5. You may spawn **subagents** to parallelize independent validation runs (e.g. convergence sweeps on different grid axes) — only if it genuinely saves wall-clock; otherwise stay inline.
- Treat each numbered **GOAL** below as a deliverable. State the goal, do it, show evidence.

---

## The pricing problem (define it exactly — match the existing env semantics)

**Spot (Hambly–Howison–Kluge):** `S_t = exp(f(t) + X_t + Y_t)` with `f ≡ 0` (no seasonality, `f_id="no_seasonal"`).
- `dX = -α X dt + σ dW` (OU). One-step transition is **exact and Gaussian**: `X' | X ~ N(decay_X·X, σ_X²)`, `decay_X = e^{-α·dt}`, `σ_X² = σ²(1−decay_X²)/(2α)`.
- `dY = -β Y dt + J dN`, `N ~ Poisson(λ)`, jump marks `J ~ Exp(1/μ_J)` (positive). One-step transition: `Y' = decay_Y·Y + Δ`, `decay_Y = e^{-β·dt}`, where the fresh increment `Δ = Σ_i J_i e^{-β(dt−U_i)}`, `U_i ~ U(0,dt)` arrival times, is **independent of `Y`**. So the one-step kernel is the product of an exact Gaussian in `X'` and a compound-Poisson-decay law in `Δ` shifted by `decay_Y·Y`.

**Contract (`SwingContract`):** discrete decision dates `j = 0,…,N−1`, `N = n_rights`, `dt = maturity/(N−1)`, `t_j = j·dt`, per-step discount `df = e^{-r·dt}`.
State = `(X_j, Y_j, Q_j)` where `Q_j ∈ [0, Q_max]` is cumulative volume exercised *before* date `j`. Control `q_j ∈ [q_min, q_max]`, feasible `q_j ≤ Q_max − Q_j` (and, if `Q_min>0`, must keep the global minimum reachable in the remaining dates; canonical `Q_min=0`). Optional refraction `min_refraction_periods` adds a discrete cooldown axis exactly as `src/lsm_swing_pricer.py` does (canonical = 0 → implement that first, leave a clean hook).

**Per-step net cashflow with profitability gate (must match `calculate_standardized_reward` in `src/swing_env.py`):**
`π_j(q) = q·(S_j − K)⁺ − c·q^γ`. The env **forces `q=0` whenever `π_j(q) ≤ 0`** (net-profitability gate). Because exercising at a loss also consumes budget, `q=0` is always available in the inner max, so the gate is consistent with optimality — but enforce it explicitly so you reproduce the env to the digit.

**Total value (price) = present value at `t_0`:** `V_0 = E[ Σ_{j} df^{j} · π_j(q_j) ]`, evaluated at the single initial state `X_0 = log(S_0) − f(0) = 0`, `Y_0 = 0`, `Q_0 = 0`.

> ⚠️ **The convex cost (`γ>1`) destroys bang-bang.** With a linear payoff the optimum is `q*∈{q_min,q_max}` (classic multiple-stopping). Here the per-step objective is strictly concave in `q`, so the optimum is **interior**: `q*(S) = clip( ((S−K)⁺/(cγ))^{1/(γ−1)}, q_min, q_max )`, further capped by the residual budget `Q_max−Q`. This is a genuine **continuous-control** DP — your inner Bellman step is a 1-D concave maximization, *not* a discrete max. (This same FOC is what LSM-D and the RL warm-start use; reuse it for consistency.)

### Backward recursion to implement
Let `U_j(x,y,Q)` = optimal value at date `j` discounted to `t_j`. Then `V_0 = U_0(0,0,0)` and
```
U_{N-1}(x,y,Q) = max_{0 ≤ q ≤ min(q_max, Q_max−Q)} π_{N-1}(q; S_{N-1})            # terminal: no continuation
U_j   (x,y,Q) = max_{0 ≤ q ≤ min(q_max, Q_max−Q)} [ π_j(q; S_j) + df · W_j(x,y,Q+q) ]
W_j(x,y,Q')   = E[ U_{j+1}(X',Y',Q') | x,y ]                                       # the analytical kernel
```
with `π` gated to 0 when non-positive, and `q ≥ q_min` only when exercising (q=0 always allowed).

## Canonical parameters (v64 focal — your primary test case)
| Group | Values |
|---|---|
| HHK | `S0=1.0, α=12.0, σ=1.2, β=150.0, λ=6.0, μ_J=0.3, f≡0` |
| Contract | `K=1.0, maturity T=0.0833 (~1mo), n_rights N=22, q_min=0, q_max=2.0, Q_min=0, Q_max=20.0, r=0.05, min_refraction=0` |
| Convex cost | `c_cost=0.04, γ_cost=2.0` (focal). Also support the full sweep `c∈{0.01,0.02,0.04,0.05,0.08,0.10,0.15} × γ∈{1,1.5,2,3}`. |
| Derived | `dt≈0.0039667, decay_X≈0.9535, σ_X≈0.0738` (stationary std of X ≈0.245); `decay_Y≈0.552, λ·dt≈0.0238` (jump ≈2.4%/step). `R=Q_max/q_max=10` rights. |

> Note `γ=1` is the **linear/bang-bang** corner (still inside the family). Treat it as a special-cased validation case, not the main one. `c=0` ⇒ pure linear swing.

## Assets to reuse (study, then re-implement in C++ — do not import Python at runtime)
- **`src/transition_kernel.py`** is the analytical one-step HHK kernel already used by the RL critic. It gives you the *exact* quadrature you need: Gauss-Hermite nodes/weights `(z_X, w_X)` for the Gaussian `X'` step, and a stratified-QMC jump mesh `(delta_Y, w_Y)` for `Δ` (no-jump node + `M_per_k` samples per jump count `k=1..N_max`). Your DP's expectation is the **same kernel pushed to high quadrature order**, which also makes a clean parity test: your C++ `W_j` evaluated at a single node must match a Python reference that calls this module's quadrature on the same value array (FP64, tol ≤ 1e-10).
- `cpp_pricer/` shows the CMake + Accelerate setup, the `-DCPP_PRICER_FP64=ON` parity build, and `cpp_pricer/data/kernel_v64*.bin` (an exported kernel — you may mirror the format, but building the mesh from parameters in C++ is cleaner and self-contained).
- `src/simulate_hhk_spot.py` is the path simulator — use it (via a small Python helper script under `grid_dp_pricer/tools/`) to (a) set grid truncation ranges from realized `(X,Y)` support and (b) generate OOS paths for the forward-MC validation in GOAL 2.

---

## GOAL 1 — Implementation (accuracy-first, then speed)
Recommended architecture (justify or improve on it in your plan):
- **Tensor-product grid** in `(X, Y, Q)`. Set `X`-range and `Y`-range from simulated marginal support over the horizon (e.g. ~1e-5/1−1e-5 quantiles padded; `X` roughly `[−1.5,1.5]`, `Y` roughly `[0, ~3]`, `Y≥0` since jumps are positive). `Q`-grid on `[0,Q_max]`.
- **Smooth interpolation** of `U_{j+1}` in `(X,Y)` (bicubic / tensor cubic spline preferred over linear for spectral-ish convergence) and in `Q` (cubic; exploit that `U` is **concave in `Q`** — verify this and use it).
- **Separable expectation as two BLAS GEMMs (the speed key).** Because the kernel image of a grid node is `X'_ix = decay_X·x + σ_X·z_X[ix]` (depends on `x` only) and `Y'_iy = decay_Y·y + delta_Y[iy]` (depends on `y` only), and interpolation is linear in the value array, the operator `W_j = E[U_{j+1}]` factorizes into a constant `A_X` (`n_X×n_X`) acting on the `X`-axis and a constant `A_Y` (`n_Y×n_Y`) on the `Y`-axis: `W = A_X · U(:,:,Q') · A_Yᵀ` per `Q'`-slice. **Precompute `A_X, A_Y` once.** Each backward step is then a handful of GEMMs (Accelerate) — `O(N · n_Q · (n_X² n_Y + n_X n_Y²))`.
- **Inner control:** for each node, maximize `g(q) = π_j(q) + df·W_j(x,y,Q+q)` over `q∈[0, min(q_max,Q_max−Q)]`. Use the closed-form FOC root as a seed; since `g` is concave (concave payoff + concave-in-Q continuation), refine with golden-section/Newton on the `Q`-interpolant. Always compare against `q=0` and apply the gate.
- Initial price is just `U_0(0,0,0)` — a single Bellman evaluation at the known initial state (no interpolation needed there).
- FP64 throughout for the reference; optionally an FP32 fast path *measured against* FP64.
- Clean CLI: contract + HHK + grid-resolution flags; emit the price and timing as JSON/CSV.

## GOAL 2 — Validation (this is what makes it a *reference*)
1. **Kernel parity (FP64, tol ≤ 1e-10):** C++ one-step expectation vs `src/transition_kernel.py` quadrature on identical inputs.
2. **Self-consistency (the decisive test):** extract the DP greedy policy, simulate it forward on a large **independent** OOS path set (reuse `simulate_hhk_spot`), and accumulate `Σ df^j π_j`. This forward-MC price is a *lower bound* of the induced policy and must converge to the backward `U_0` **from below** as the grid refines. Agreement within MC error ⇒ no bug; persistent gap ⇒ interpolation/indexing bug. Report both with a 95% CI on the MC leg.
3. **Ordering vs incumbents:** on the *same* canonical contract, `DP ≥ LSM-D` and `DP ≥ RL` (both are lower bounds). Recompute LSM-D fresh with `src/lsm_swing_pricer.py` on a common OOS set (seed 999, 65 536 paths to match the repo's `tools/rebuild_results_v7.py` protocol) and pull an RL price from a saved focal run. Quantify the gaps in price and Δ% — **this is the headline scientific result** (e.g. "RL is within X% of the DP reference").
4. **Limiting cases:** `c=0`/`γ=1` ⇒ bang-bang; compare to LSM bang-bang and check the optimal lift is `∈{0,q_max}`. `R=1` (single right) ⇒ a tractable single-decision check. Deep-ITM/zero-vol degenerate sanity. Monotonicities: price ↑ in `q_max,Q_max`; ↓ in `c`.
5. **Full grid:** produce DP prices for the entire `c×γ` sweep and tabulate against the existing `Convex Costs Results 9.csv` cells.

## GOAL 3 — Convergence analysis
Empirically establish convergence and **observed order** in each axis independently and jointly: `n_X, n_Y, n_Q`, quadrature `(M_x, N_max, M_per_k)`, and interpolation order. Show the price vs resolution curves, estimate the order `p` (`error ∝ h^p`), and apply **Richardson extrapolation** to report a converged price with an error bar. Identify the controlling axis (analogous to how the RL kernel found `M_x` dominant). Deliver plots + a table in `RESULTS.md`.

## GOAL 4 — Time-complexity analysis
Derive the asymptotic cost of the backward pass (GEMM-dominated) and the inner control, and the memory footprint. Then **measure** wall-clock vs each resolution axis on the M1 and confirm the empirical scaling matches the derivation. Report the **accuracy/time Pareto frontier** and the single recommended "production" resolution.

## GOAL 5 — Best accuracy in the lowest time (optimize against the incumbents)
Tune the Pareto-optimal config and **benchmark wall-clock head-to-head** against: LSM-D (65 536 paths) and the RL train+eval. Target (adjust if you can beat it): a reference price converged to **≤1e-4 absolute** in **well under 1 second** single-threaded on M1, i.e. orders of magnitude faster than LSM-D/RL for *higher* accuracy. Use Accelerate GEMMs, precomputed `A_X/A_Y`, contiguous memory, and (optionally, measured) multithreading across `Q`-slices. Document what worked and what didn't, mirroring `cpp_pricer/DEVELOPMENT_NOTES.md`.

---

## Deliverables
- `grid_dp_pricer/` building cleanly via CMake (Accelerate on; `-DGRID_DP_FP64=ON` parity build), with the CLI binary, the FP64 test suite (parity + self-consistency + limiting cases, all green), and the Python cross-check helpers under `tools/`.
- `grid_dp_pricer/README.md` — build/run/validate.
- `grid_dp_pricer/RESULTS.md` — the converged focal price + error bar, the full `c×γ` table, the **DP-vs-LSM-D-vs-RL** comparison (the key result), convergence curves + estimated order, the time-complexity derivation + measured scaling, and the accuracy/time Pareto frontier vs the incumbents.
- `grid_dp_pricer/docs/DEVELOPMENT_NOTES.md` — architecture rationale, the optimization journey, bugs found, and the validation bar (in the spirit of the existing C++ notes).

Begin by reading the three docs above and `src/{transition_kernel,swing_env,swing_contract,lsm_swing_pricer,simulate_hhk_spot}.py`, then enter plan mode and present your plan.
