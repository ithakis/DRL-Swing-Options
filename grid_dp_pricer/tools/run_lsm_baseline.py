#!/usr/bin/env python
"""Recompute the LSM-D benchmark fresh on the canonical common test set, matching
tools/rebuild_results_v7.py: full-state Chebyshev degree-2, fit on seed=998 (65536 paths),
priced out-of-sample on seed=999 (65536 paths).  Importable (lsm_baseline) and runnable.

    conda run -n EP11 python grid_dp_pricer/tools/run_lsm_baseline.py --c 0.04 --gamma 2
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from src.lsm_swing_pricer import fit_lsm_estimators  # noqa: E402
from src.lsm_swing_pricer import price_swing_option_lsm_oos  # noqa: E402
from src.simulate_hhk_spot import no_seasonal_function, simulate_hhk_spot  # noqa: E402
from src.swing_contract import SwingContract  # noqa: E402

TEST_SEED = 999
LSM_TRAIN_SEED = 998
N_PATHS = 65536


def _focal_contract(c_cost, gamma_cost):
    return SwingContract(
        q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0, maturity=0.0833,
        n_rights=22, min_refraction_periods=0, r=0.05, c_cost=c_cost, gamma_cost=gamma_cost,
    )


def _hhk():
    return dict(S0=1.0, alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3,
                f=no_seasonal_function)


def _sim(contract, seed):
    return simulate_hhk_spot(
        S0=1.0, T=contract.maturity, n_steps=contract.n_rights - 1, n_paths=N_PATHS,
        alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3, f=no_seasonal_function,
        seed=seed, dtype=np.float64, stratify=True, batch_size=128,
    )


# Canonical LSM-D (matches tools/rebuild_results_v7.py LSM_N_ACTIONS): full-state,
# Chebyshev deg-2, multi-level exercise grid {0, q_max/4, ..., q_max}. n_actions=2 is the
# original bang-bang variant — correct for linear payoffs (gamma=1) but it badly under-prices
# convex (gamma>1) cells where the optimum is interior.
LSM_N_ACTIONS = 5


def lsm_baseline(c_cost, gamma_cost, degree=2, basis="chebyshev", n_actions=LSM_N_ACTIONS):
    """Return dict with the canonical full-state LSM-D price + 95% CI, plus the bang-bang
    (n_actions=2) variant for reference."""
    contract = _focal_contract(c_cost, gamma_cost)
    train_ds = _sim(contract, LSM_TRAIN_SEED)
    test_ds = _sim(contract, TEST_SEED)

    # Canonical: full-state, multi-level, fit in-sample (seed 998), price OOS (seed 999).
    est = fit_lsm_estimators(contract, train_ds, poly_degree=degree, basis_type=basis,
                             state_mode="full", n_actions=n_actions)
    full_price, (full_lo, full_hi) = price_swing_option_lsm_oos(
        contract, test_ds, est, seed=TEST_SEED + 1, csv_path=None)

    # Bang-bang (n_actions=2) full-state, for reference.
    est_bb = fit_lsm_estimators(contract, train_ds, poly_degree=degree, basis_type=basis,
                                state_mode="full", n_actions=2)
    bb_price, (bb_lo, bb_hi) = price_swing_option_lsm_oos(
        contract, test_ds, est_bb, seed=TEST_SEED + 1, csv_path=None)

    return {
        "c": c_cost, "gamma": gamma_cost, "n_actions": n_actions,
        "lsm_full": full_price, "lsm_full_lo": full_lo, "lsm_full_hi": full_hi,
        "lsm_bangbang": bb_price, "lsm_bangbang_lo": bb_lo, "lsm_bangbang_hi": bb_hi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=0.04)
    ap.add_argument("--gamma", type=float, default=2.0)
    args = ap.parse_args()
    r = lsm_baseline(args.c, args.gamma)
    print(f"LSM-D  c={r['c']} gamma={r['gamma']}")
    print(f"  full-state canonical (n_actions=5): {r['lsm_full']:.6f}  CI[{r['lsm_full_lo']:.6f}, "
          f"{r['lsm_full_hi']:.6f}]")
    print(f"  full-state bang-bang (n_actions=2): {r['lsm_bangbang']:.6f}  CI[{r['lsm_bangbang_lo']:.6f}, "
          f"{r['lsm_bangbang_hi']:.6f}]")


if __name__ == "__main__":
    main()
