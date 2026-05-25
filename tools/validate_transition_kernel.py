"""
Phase-0 validation for src/transition_kernel.py.

Compares quadrature expectations against direct Monte Carlo ground truth on
the focal HHK / contract setting (alpha=12, sigma=1.2, beta=150, lam=6,
mu_J=0.3, n_rights=22, maturity=0.0833 years). Verifies:

  1. Conditional moments E[g(X_{t+1}, Y_{t+1}) | X_t, Y_t] for several
     test functions g, across a panel of (X_t, Y_t) test states.
  2. Smallest grid (M_x, M_per_k, N_max) achieving 4-digit relative accuracy.
  3. Per-call njit kernel-build time (<= 50 us on M1 target).

Run:
    python tools/validate_transition_kernel.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transition_kernel import (  # noqa: E402
    KernelParams,
    precompute_kernel,
    build_next_state_grid_batched,
    build_quadrature_weights,
)


# Focal HHK / contract regime from Convex Cost Experiments/SwingOption_20_c0.04_gamma2.sh
ALPHA = 12.0
SIGMA = 1.2
BETA = 150.0
LAM = 6.0
MU_J = 0.3
MATURITY = 0.0833
N_RIGHTS = 22
DT = MATURITY / (N_RIGHTS - 1)
STRIKE = 1.0


def f_no_seasonal(t: float) -> float:
    return 0.0


def mc_one_step_samples(
    X_t: float, Y_t: float, n_paths: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct Monte Carlo of (X_{t+1}, Y_{t+1}) | (X_t, Y_t) for one step."""
    decay_X = math.exp(-ALPHA * DT)
    sigma_X = math.sqrt(SIGMA ** 2 * (1.0 - decay_X ** 2) / (2.0 * ALPHA))
    decay_Y = math.exp(-BETA * DT)

    z = rng.standard_normal(n_paths)
    X = decay_X * X_t + sigma_X * z

    # Compound-Poisson jump increment with decay weighting on uniform jump times.
    counts = rng.poisson(LAM * DT, size=n_paths)
    n_tot = int(counts.sum())
    if n_tot == 0:
        Delta = np.zeros(n_paths)
    else:
        U = rng.uniform(0.0, DT, size=n_tot)
        V = np.clip(rng.random(n_tot), 1e-12, 1.0 - 1e-12)
        J = -MU_J * np.log(V)
        decay = np.exp(-BETA * (DT - U))
        contrib = J * decay
        # bincount assigns the i-th contribution to its parent path.
        path_ids = np.repeat(np.arange(n_paths), counts)
        Delta = np.bincount(path_ids, weights=contrib, minlength=n_paths)
    Y = decay_Y * Y_t + Delta
    S = np.exp(f_no_seasonal(DT) + X + Y)
    return S, X, Y


def kernel_one_step_expectations(
    X_t: float, Y_t: float, params: KernelParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (S_grid, X_grid, Y_grid, weights) for the (X_t, Y_t) cell."""
    kernel = precompute_kernel(params, N_RIGHTS, f_no_seasonal, maturity=MATURITY)
    X_arr = np.array([X_t], dtype=np.float64)
    Y_arr = np.array([Y_t], dtype=np.float64)
    t_idx_next = np.array([1], dtype=np.int64)  # at t_idx_next=1 -> exp_f = exp(f(dt))
    S_grid, X_grid, Y_grid = build_next_state_grid_batched(X_arr, Y_arr, t_idx_next, kernel)
    weights = build_quadrature_weights(kernel)
    return S_grid[0], X_grid[0], Y_grid[0], weights


def relative_error(approx: float, truth: float) -> float:
    denom = max(abs(truth), 1e-12)
    return abs(approx - truth) / denom


def analytical_E_S(X_t: float, Y_t: float) -> float:
    """Closed form E[S_{t+1} | X_t, Y_t] via the MGF of the joint X' + Y'.

    With X' ~ N(decay_X X_t, sigma_X^2) and Y' = decay_Y Y_t + Delta where Delta
    has compound-Poisson MGF M_Delta(theta) = ((1 - a) / (1 - b))^(lam/beta),
    a = theta mu_J exp(-beta dt), b = theta mu_J.  Setting theta = 1:
        E[S] = exp(decay_X X_t + 0.5 sigma_X^2 + decay_Y Y_t) * M_Delta(1).
    """
    decay_X = math.exp(-ALPHA * DT)
    sigma_X2 = SIGMA ** 2 * (1.0 - decay_X ** 2) / (2.0 * ALPHA)
    decay_Y = math.exp(-BETA * DT)
    a = MU_J * decay_Y
    b = MU_J
    if a >= 1.0 or b >= 1.0:
        return float("nan")
    M_delta = ((1.0 - a) / (1.0 - b)) ** (LAM / BETA)
    return math.exp(decay_X * X_t + 0.5 * sigma_X2 + decay_Y * Y_t) * M_delta


def analytical_E_S2(X_t: float, Y_t: float) -> float:
    """Closed form E[S_{t+1}^2 | X_t, Y_t] (theta = 2 in the joint MGF)."""
    decay_X = math.exp(-ALPHA * DT)
    sigma_X2 = SIGMA ** 2 * (1.0 - decay_X ** 2) / (2.0 * ALPHA)
    decay_Y = math.exp(-BETA * DT)
    theta = 2.0
    a = theta * MU_J * decay_Y
    b = theta * MU_J
    if a >= 1.0 or b >= 1.0:
        return float("nan")
    M_delta = ((1.0 - a) / (1.0 - b)) ** (LAM / BETA)
    return math.exp(theta * decay_X * X_t + 0.5 * theta ** 2 * sigma_X2 + theta * decay_Y * Y_t) * M_delta


def run_accuracy_test():
    print(f"DT = {DT:.6f},  beta*dt = {BETA * DT:.4f},  alpha*dt = {ALPHA * DT:.4f}, "
          f"lam*dt = {LAM * DT:.4f}")
    print("=" * 78)

    # Test states span a realistic spread of (X_t, Y_t) seen during training.
    # X_t corresponds to log(S_t/exp(f(t)) - Y_t); we pick X spanning roughly +-3 sigma_X
    # accumulated over many steps, and Y from 0 to several jump magnitudes.
    test_states = [
        (0.0, 0.0),
        (0.05, 0.0),
        (-0.05, 0.0),
        (0.0, 0.3),
        (0.0, 1.0),
        (0.1, 0.5),
        (-0.1, 0.2),
        (0.0, 2.0),
    ]

    grid_configs = [
        (4, 6, 2),
        (6, 8, 2),
        (6, 8, 3),
        (8, 16, 3),
        (12, 16, 3),
    ]

    # 5M MC paths give ~0.05% standard error on E[(S-K)+]; this lets us
    # actually resolve quadrature errors below 1e-3.
    n_paths_mc = 5_000_000
    rng = np.random.default_rng(42)

    # Precompute MC ground truth once per test state (re-used across grids)
    mc_truth = {}
    print("Pre-rolling 5M Monte Carlo paths per test state...")
    for (X_t, Y_t) in test_states:
        S_mc, X_mc, Y_mc = mc_one_step_samples(X_t, Y_t, n_paths_mc, rng)
        sig_vals = 1.0 / (1.0 + np.exp(-10.0 * (S_mc - STRIKE)))
        smth_po_vals = (S_mc - STRIKE) * sig_vals
        truth = {
            "E[S]": (float(np.mean(S_mc)), analytical_E_S(X_t, Y_t)),
            "E[S^2]": (float(np.mean(S_mc * S_mc)), analytical_E_S2(X_t, Y_t)),
            "E[(S-K)+]": (float(np.mean(np.maximum(S_mc - STRIKE, 0.0))), None),
            "E[sig(S-K)]": (float(np.mean(sig_vals)), None),
            "E[smth_PO]": (float(np.mean(smth_po_vals)), None),
        }
        se = {
            "E[S]": float(np.std(S_mc) / math.sqrt(n_paths_mc)),
            "E[S^2]": float(np.std(S_mc * S_mc) / math.sqrt(n_paths_mc)),
            "E[(S-K)+]": float(np.std(np.maximum(S_mc - STRIKE, 0.0)) / math.sqrt(n_paths_mc)),
            "E[sig(S-K)]": float(np.std(sig_vals) / math.sqrt(n_paths_mc)),
            "E[smth_PO]": float(np.std(smth_po_vals) / math.sqrt(n_paths_mc)),
        }
        mc_truth[(X_t, Y_t)] = (truth, se)
    print("Done.\n")

    # The critic Q-function is smooth, so the relevant integrand class is smooth
    # bounded functions. We test:
    #   E[S], E[S^2]: smooth - bottlenecked by Y-mesh quadrature, target ~1e-3.
    #   E[(S-K)+]: kink at S=K - hardest case, mostly bottlenecked by X-mesh
    #   resolution near the strike.
    #   E[sigmoid(S - K)]: smooth proxy for a critic Q-function near the strike,
    #   target ~1e-4. THIS IS THE ACCURACY THAT MATTERS for the agent.
    test_funcs = {
        "E[S]": lambda S, X, Y: S,
        "E[S^2]": lambda S, X, Y: S * S,
        "E[(S-K)+]": lambda S, X, Y: np.maximum(S - STRIKE, 0.0),
        "E[sig(S-K)]": lambda S, X, Y: 1.0 / (1.0 + np.exp(-10.0 * (S - STRIKE))),
        "E[smth_PO]": lambda S, X, Y: (S - STRIKE) * (1.0 / (1.0 + np.exp(-10.0 * (S - STRIKE)))),
    }

    # Sanity check: MC vs analytical for E[S] (zero-quadrature-error baseline)
    print("MC-vs-analytical sanity check (E[S]):")
    for (X_t, Y_t), (truth, se) in mc_truth.items():
        mc_val, anal_val = truth["E[S]"]
        diff = abs(mc_val - anal_val)
        print(f"  (X={X_t:+.2f}, Y={Y_t:+.2f})  MC={mc_val:.6f}  ANAL={anal_val:.6f}  "
              f"diff={diff:.2e}  MC_SE={se['E[S]']:.2e}")
    print()

    for (M_x, M_per_k, N_max) in grid_configs:
        params = KernelParams(
            alpha=ALPHA, sigma=SIGMA, beta=BETA, lam=LAM, mu_J=MU_J, dt=DT,
            M_x=M_x, N_max=N_max, M_per_k=M_per_k,
        )
        kernel = precompute_kernel(params, N_RIGHTS, f_no_seasonal, maturity=MATURITY,
                                   force_rebuild=True)
        print(f"--- Grid: M_x={M_x}, M_per_k={M_per_k}, N_max={N_max}, "
              f"M_total={kernel.M} ---")

        worst = {name: 0.0 for name in test_funcs}
        for (X_t, Y_t) in test_states:
            S_g, X_g, Y_g, w = kernel_one_step_expectations(X_t, Y_t, params)
            truth, se = mc_truth[(X_t, Y_t)]
            row = []
            for name, g in test_funcs.items():
                approx = float(np.sum(w * g(S_g, X_g, Y_g)))
                if name == "E[S]":
                    # Use analytical ground truth (zero MC noise)
                    truth_val = truth[name][1]
                else:
                    truth_val = truth[name][0]
                err = relative_error(approx, truth_val)
                # Subtract MC SE-equivalent relative noise floor for non-analytical
                if name != "E[S]":
                    floor = se[name] / max(abs(truth_val), 1e-12)
                    err = max(err - floor, 0.0)
                row.append((name, err))
                worst[name] = max(worst[name], err)
            errs_str = "  ".join(f"{n}={e:.2e}" for n, e in row)
            print(f"  (X={X_t:+.2f}, Y={Y_t:+.2f})  {errs_str}")
        worst_str = ", ".join(f"{n}={v:.2e}" for n, v in worst.items())
        print(f"  >>> worst per-fn: {worst_str}\n")


def run_speed_test():
    print("\n=== Per-call kernel-build timing (njit warm cache) ===")
    params = KernelParams(
        alpha=ALPHA, sigma=SIGMA, beta=BETA, lam=LAM, mu_J=MU_J, dt=DT,
        M_x=6, N_max=3, M_per_k=8,
    )
    kernel = precompute_kernel(params, N_RIGHTS, f_no_seasonal, maturity=MATURITY)
    print(f"Kernel size M = {kernel.M}  (M_x={kernel.M_x}, M_y={kernel.M_y})")

    # Warm njit cache
    X = np.zeros(1, dtype=np.float64)
    Y = np.zeros(1, dtype=np.float64)
    t_idx = np.ones(1, dtype=np.int64)
    _ = build_next_state_grid_batched(X, Y, t_idx, kernel)

    # Single-state timing
    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        _ = build_next_state_grid_batched(X, Y, t_idx, kernel)
    elapsed = time.perf_counter() - t0
    print(f"Single state: {elapsed / n * 1e6:.2f} us per call (n={n})")

    # Batched timing (B=128, mimicking critic batch)
    B = 128
    Xb = np.random.randn(B).astype(np.float64) * 0.1
    Yb = np.abs(np.random.randn(B).astype(np.float64)) * 0.5
    tb = np.ones(B, dtype=np.int64)
    t0 = time.perf_counter()
    for _ in range(n):
        _ = build_next_state_grid_batched(Xb, Yb, tb, kernel)
    elapsed = time.perf_counter() - t0
    print(f"Batch B={B}:  {elapsed / n * 1e6:.2f} us per call (n={n})")


if __name__ == "__main__":
    run_accuracy_test()
    run_speed_test()
