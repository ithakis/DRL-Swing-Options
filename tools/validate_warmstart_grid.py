"""
Phase-H4 sanity check.

Builds the (X, Y, Q, t) value grid via backward induction and compares the
implied option price V(X=0, Y=0, Q_max, t=0) against the LSM benchmark for
the focal regime (c=0.04, gamma=2).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.transition_kernel import KernelParams, precompute_kernel
from src.critic_warmstart import (
    build_value_grid, grid_price_at_origin,
    compute_Q_targets, sample_synthetic_states,
)
from src.simulate_hhk_spot import no_seasonal_function


# Focal regime (mirrors SwingOption_20_c0.04_gamma2.sh)
ALPHA, SIGMA, BETA, LAM, MU_J = 12.0, 1.2, 150.0, 6.0, 0.3
N_RIGHTS = 22
MATURITY = 0.0833
DT = MATURITY / (N_RIGHTS - 1)
STRIKE = 1.0
C_COST = 0.04
GAMMA_COST = 2.0
Q_MAX = 20.0
q_max = 2.0
q_min = 0.0
RFR = 0.05
DISCOUNT = np.exp(-RFR * DT)


def main() -> None:
    print(f"dt={DT:.5f}, discount/step={DISCOUNT:.5f}")
    params = KernelParams(
        alpha=ALPHA, sigma=SIGMA, beta=BETA, lam=LAM, mu_J=MU_J, dt=DT,
        M_x=4, N_max=2, M_per_k=4,
    )
    kernel = precompute_kernel(params, N_RIGHTS, no_seasonal_function, maturity=MATURITY)
    print(f"Kernel M = {kernel.M}")

    # Try a few grid sizes to see V0 convergence
    for n_X, n_Y, n_a in [(15, 12, 11), (25, 20, 11), (35, 28, 15)]:
        t0 = time.perf_counter()
        grid = build_value_grid(
            kernel=kernel,
            strike=STRIKE, c_cost=C_COST, gamma_cost=GAMMA_COST,
            q_min=q_min, q_max=q_max, Q_max=Q_MAX,
            n_rights=N_RIGHTS, discount_factor=DISCOUNT,
            n_X=n_X, n_Y=n_Y, n_actions=n_a,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        price0 = grid_price_at_origin(grid)
        print(f"  (n_X={n_X:3d}, n_Y={n_Y:3d}, n_a={n_a:3d}): V0={price0:.4f}  "
              f"build_time={elapsed:.1f}s")

    # Use the medium grid for the rest of the diagnostics
    grid = build_value_grid(
        kernel=kernel,
        strike=STRIKE, c_cost=C_COST, gamma_cost=GAMMA_COST,
        q_min=q_min, q_max=q_max, Q_max=Q_MAX,
        n_rights=N_RIGHTS, discount_factor=DISCOUNT,
        n_X=25, n_Y=20, n_actions=11,
        verbose=False,
    )

    print(f"\nV grid shape: {grid.V.shape}, nonzero fraction: "
          f"{(grid.V > 0).mean():.3f}")
    print(f"V[X=0, Y=0, Q_max, t=0] = {grid_price_at_origin(grid):.4f}")
    # Reference: the focal sweeps reported LSM ~ 1.94-1.99 at this regime.

    # Sample Q* targets, sanity-check statistics
    rng = np.random.default_rng(0)
    states, actions = sample_synthetic_states(grid, kernel, 4096, rng=rng)
    targets = compute_Q_targets(states, actions, grid, kernel)
    print(f"\nQ* targets on 4096 random (s, a):")
    print(f"  mean={targets.mean():.3f}, std={targets.std():.3f}, "
          f"min={targets.min():.3f}, max={targets.max():.3f}")
    print(f"  Fraction Q* > 0: {(targets > 0).mean():.3f}")


if __name__ == "__main__":
    main()
