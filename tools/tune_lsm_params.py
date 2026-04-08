#!/usr/bin/env python
"""Tune LSM parameters for the discretized-action baseline (M=5).

Iteration 1: 10 diverse setups across parameter space.
Iteration 2: 10 more setups concentrated near best performers.

Parameters to tune:
- poly_degree: [2, 5]
- basis_type: {chebyshev, laguerre, power, hermite}
- reg_type: {none, ridge}
- reg_alpha: float (only for ridge)
- n_training_paths: training set size

Fixed: n_actions=5, state_mode=full
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos
from src.simulate_hhk_spot import no_seasonal_function, simulate_hhk_spot
from src.swing_contract import SwingContract


def make_contract(c_cost, gamma_cost):
    return SwingContract(
        q_min=0.0,
        q_max=2.0,
        Q_min=0.0,
        Q_max=20.0,
        strike=1.0,
        maturity=0.0833,
        n_rights=22,
        r=0.05,
        min_refraction_periods=0,
        c_cost=c_cost,
        gamma_cost=gamma_cost,
    )


HHK_BASE = {
    "S0": 1.0,
    "alpha": 12.0,
    "sigma": 1.2,
    "beta": 150.0,
    "lam": 6.0,
    "mu_J": 0.3,
    "f": no_seasonal_function,
    "dtype": np.float32,
}


def make_dataset(n_paths, seed):
    contract = make_contract(0.0, 1.0)
    hhk = {**HHK_BASE, "T": contract.maturity, "n_steps": contract.n_rights - 1}
    return simulate_hhk_spot(**hhk, n_paths=n_paths, seed=seed, stratify=True, batch_size=128)


CONFIGS = [
    ("A", 0.00, 1.0),
    ("B", 0.04, 1.0),
    ("C", 0.04, 1.5),
    ("D", 0.04, 2.0),
]


def evaluate_setup(setup, train_ds, test_ds):
    """Evaluate a parameter setup across all 4 configs. Returns dict of prices."""
    prices = {}
    for label, c, gamma in CONFIGS:
        contract = make_contract(c, gamma)
        train_f64 = tuple(np.asarray(a, dtype=np.float64) for a in train_ds)
        test_f64 = tuple(np.asarray(a, dtype=np.float64) for a in test_ds)
        est = fit_lsm_estimators(
            contract=contract,
            dataset=train_f64,
            poly_degree=setup["degree"],
            basis_type=setup["basis"],
            state_mode="full",
            reg_type=setup["reg_type"],
            reg_alpha=setup.get("reg_alpha", 1e-6),
            n_actions=5,
        )
        price, _ = price_swing_option_lsm_oos(
            contract=contract,
            dataset=test_f64,
            estimators=est,
            seed=1000,
            csv_path=None,
        )
        prices[label] = price
    return prices


def main():
    print("Generating datasets...")
    # Use larger training set candidates
    train_32k = make_dataset(32768, seed=998)
    train_65k = make_dataset(65536, seed=998)
    test_ds = make_dataset(65536, seed=999)
    print(f"  Done. Test: {test_ds[1].shape}")

    # ==================== ITERATION 1 ====================
    setups_iter1 = [
        {"name": "cheb_d2_ols_32k", "degree": 2, "basis": "chebyshev", "reg_type": "none", "train": "32k"},
        {"name": "cheb_d3_ols_32k", "degree": 3, "basis": "chebyshev", "reg_type": "none", "train": "32k"},
        {
            "name": "cheb_d2_ridge1e-4_32k",
            "degree": 2,
            "basis": "chebyshev",
            "reg_type": "ridge",
            "reg_alpha": 1e-4,
            "train": "32k",
        },
        {
            "name": "cheb_d3_ridge1e-4_32k",
            "degree": 3,
            "basis": "chebyshev",
            "reg_type": "ridge",
            "reg_alpha": 1e-4,
            "train": "32k",
        },
        {"name": "power_d2_ols_32k", "degree": 2, "basis": "power", "reg_type": "none", "train": "32k"},
        {"name": "power_d3_ols_32k", "degree": 3, "basis": "power", "reg_type": "none", "train": "32k"},
        {"name": "laguerre_d2_ols_32k", "degree": 2, "basis": "laguerre", "reg_type": "none", "train": "32k"},
        {"name": "hermite_d2_ols_32k", "degree": 2, "basis": "hermite", "reg_type": "none", "train": "32k"},
        {"name": "cheb_d2_ols_65k", "degree": 2, "basis": "chebyshev", "reg_type": "none", "train": "65k"},
        {
            "name": "cheb_d4_ridge1e-3_32k",
            "degree": 4,
            "basis": "chebyshev",
            "reg_type": "ridge",
            "reg_alpha": 1e-3,
            "train": "32k",
        },
    ]

    print("\n" + "=" * 90)
    print("ITERATION 1: 10 diverse setups")
    print("=" * 90)

    results_iter1 = []
    for setup in setups_iter1:
        train = train_32k if setup["train"] == "32k" else train_65k
        t0 = time.time()
        prices = evaluate_setup(setup, train, test_ds)
        elapsed = time.time() - t0
        avg = np.mean(list(prices.values()))
        results_iter1.append((setup["name"], prices, avg, elapsed))
        print(
            f"  {setup['name']:35s}  A={prices['A']:.4f}  B={prices['B']:.4f}  C={prices['C']:.4f}  D={prices['D']:.4f}  avg={avg:.4f}  ({elapsed:.1f}s)"
        )

    # Sort by average price (higher is better)
    results_iter1.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Best iter1: {results_iter1[0][0]} (avg={results_iter1[0][2]:.4f})")
    print(f"  2nd:        {results_iter1[1][0]} (avg={results_iter1[1][2]:.4f})")
    print(f"  3rd:        {results_iter1[2][0]} (avg={results_iter1[2][2]:.4f})")

    # ==================== ITERATION 2 ====================
    # Focus on best-performing region from iteration 1
    setups_iter2 = [
        {"name": "cheb_d2_ols_65k_v2", "degree": 2, "basis": "chebyshev", "reg_type": "none", "train": "65k"},
        {"name": "cheb_d3_ols_65k", "degree": 3, "basis": "chebyshev", "reg_type": "none", "train": "65k"},
        {
            "name": "cheb_d2_ridge1e-5_32k",
            "degree": 2,
            "basis": "chebyshev",
            "reg_type": "ridge",
            "reg_alpha": 1e-5,
            "train": "32k",
        },
        {
            "name": "cheb_d2_ridge1e-6_32k",
            "degree": 2,
            "basis": "chebyshev",
            "reg_type": "ridge",
            "reg_alpha": 1e-6,
            "train": "32k",
        },
        {
            "name": "cheb_d3_ridge1e-5_32k",
            "degree": 3,
            "basis": "chebyshev",
            "reg_type": "ridge",
            "reg_alpha": 1e-5,
            "train": "32k",
        },
        {
            "name": "cheb_d3_ridge1e-3_32k",
            "degree": 3,
            "basis": "chebyshev",
            "reg_type": "ridge",
            "reg_alpha": 1e-3,
            "train": "32k",
        },
        {"name": "cheb_d4_ols_32k", "degree": 4, "basis": "chebyshev", "reg_type": "none", "train": "32k"},
        {
            "name": "cheb_d3_ridge1e-5_65k",
            "degree": 3,
            "basis": "chebyshev",
            "reg_type": "ridge",
            "reg_alpha": 1e-5,
            "train": "65k",
        },
        {
            "name": "cheb_d4_ridge1e-4_65k",
            "degree": 4,
            "basis": "chebyshev",
            "reg_type": "ridge",
            "reg_alpha": 1e-4,
            "train": "65k",
        },
        {
            "name": "cheb_d5_ridge1e-3_32k",
            "degree": 5,
            "basis": "chebyshev",
            "reg_type": "ridge",
            "reg_alpha": 1e-3,
            "train": "32k",
        },
    ]

    print("\n" + "=" * 90)
    print("ITERATION 2: 10 focused setups")
    print("=" * 90)

    results_iter2 = []
    for setup in setups_iter2:
        train = train_32k if setup["train"] == "32k" else train_65k
        t0 = time.time()
        prices = evaluate_setup(setup, train, test_ds)
        elapsed = time.time() - t0
        avg = np.mean(list(prices.values()))
        results_iter2.append((setup["name"], prices, avg, elapsed))
        print(
            f"  {setup['name']:35s}  A={prices['A']:.4f}  B={prices['B']:.4f}  C={prices['C']:.4f}  D={prices['D']:.4f}  avg={avg:.4f}  ({elapsed:.1f}s)"
        )

    # Combined ranking
    all_results = results_iter1 + results_iter2
    all_results.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 90)
    print("COMBINED RANKING (all 20 setups, sorted by average price)")
    print("=" * 90)
    for rank, (name, prices, avg, elapsed) in enumerate(all_results, 1):
        print(
            f"  {rank:2d}. {name:35s}  A={prices['A']:.4f}  B={prices['B']:.4f}  C={prices['C']:.4f}  D={prices['D']:.4f}  avg={avg:.4f}"
        )

    winner = all_results[0]
    print(f"\n  WINNER: {winner[0]} with avg={winner[2]:.4f}")
    print(f"  Prices: {winner[1]}")


if __name__ == "__main__":
    main()
