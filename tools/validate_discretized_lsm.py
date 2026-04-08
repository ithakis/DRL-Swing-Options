#!/usr/bin/env python
"""Validate the discretized-action LSM implementation.

Tests:
1. Backward compatibility: n_actions=2 should match old bang-bang results.
2. Monotonicity in M: M=2 <= M=3 <= M=5 for convex costs.
3. Linear cost sanity: M=5 ≈ M=2 when gamma=1 (bang-bang is optimal).
4. Convex cost improvement: M=5 > M=2 when gamma ≥ 2.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos
from src.simulate_hhk_spot import no_seasonal_function, simulate_hhk_spot
from src.swing_contract import SwingContract


def make_contract(c_cost=0.04, gamma_cost=2.0):
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


def make_dataset(n_paths, seed):
    contract = make_contract()
    hhk = {
        "S0": 1.0,
        "T": contract.maturity,
        "n_steps": contract.n_rights - 1,
        "alpha": 12.0,
        "sigma": 1.2,
        "beta": 150.0,
        "lam": 6.0,
        "mu_J": 0.3,
        "f": no_seasonal_function,
        "dtype": np.float32,
    }
    return simulate_hhk_spot(**hhk, n_paths=n_paths, seed=seed, stratify=True, batch_size=128)


def run_lsm(contract, train_ds, test_ds, n_actions=2, degree=2, basis="chebyshev"):
    train_f64 = tuple(np.asarray(a, dtype=np.float64) for a in train_ds)
    test_f64 = tuple(np.asarray(a, dtype=np.float64) for a in test_ds)
    t0 = time.time()
    est = fit_lsm_estimators(
        contract=contract,
        dataset=train_f64,
        poly_degree=degree,
        basis_type=basis,
        state_mode="full",
        reg_type="none",
        reg_alpha=1e-6,
        n_actions=n_actions,
    )
    fit_time = time.time() - t0
    t0 = time.time()
    price, (ci_lo, ci_hi) = price_swing_option_lsm_oos(
        contract=contract,
        dataset=test_f64,
        estimators=est,
        seed=1000,
        csv_path=None,
        _print_results=False,
    )
    eval_time = time.time() - t0
    return price, ci_lo, ci_hi, fit_time, eval_time


def main():
    print("Generating datasets...")
    train_ds = make_dataset(32768, seed=998)
    test_ds = make_dataset(65536, seed=999)
    print(f"  Train: {train_ds[1].shape}, Test: {test_ds[1].shape}")

    # Reference: old bang-bang values from Convex Costs Results 7.csv
    reference_lsm = {
        (0.04, 1.0): 2.2375,
        (0.04, 1.5): 2.0751,
        (0.04, 2.0): 1.8634,
        (0.04, 3.0): 1.2652,
    }

    print("\n" + "=" * 80)
    print("TEST 1: Backward compatibility (n_actions=2 vs old bang-bang)")
    print("=" * 80)
    for (c, gamma), ref in reference_lsm.items():
        contract = make_contract(c, gamma)
        price, ci_lo, ci_hi, ft, et = run_lsm(contract, train_ds, test_ds, n_actions=2)
        diff_pct = 100 * (price - ref) / ref
        status = "PASS" if abs(diff_pct) < 2.0 else "FAIL"
        print(
            f"  c={c}, γ={gamma}: new={price:.4f}, ref={ref:.4f}, diff={diff_pct:+.2f}%  [{status}]  (fit={ft:.1f}s, eval={et:.1f}s)"
        )

    print("\n" + "=" * 80)
    print("TEST 2: Monotonicity in M (c=0.04, γ=2)")
    print("=" * 80)
    contract = make_contract(0.04, 2.0)
    prices_by_M = {}
    for M in [2, 3, 5]:
        price, ci_lo, ci_hi, ft, et = run_lsm(contract, train_ds, test_ds, n_actions=M)
        prices_by_M[M] = price
        print(f"  M={M}: price={price:.4f}  (fit={ft:.1f}s, eval={et:.1f}s)")
    mono_ok = prices_by_M[2] <= prices_by_M[3] + 0.005 and prices_by_M[3] <= prices_by_M[5] + 0.005
    print(f"  Monotonicity: {'PASS' if mono_ok else 'WARN (not strictly monotone but may be within noise)'}")

    print("\n" + "=" * 80)
    print("TEST 3: Linear cost sanity (c=0.04, γ=1.0, M=5 ≈ M=2)")
    print("=" * 80)
    contract_linear = make_contract(0.04, 1.0)
    price_2, *_ = run_lsm(contract_linear, train_ds, test_ds, n_actions=2)
    price_5, *_ = run_lsm(contract_linear, train_ds, test_ds, n_actions=5)
    diff = 100 * (price_5 - price_2) / price_2
    print(f"  M=2: {price_2:.4f}, M=5: {price_5:.4f}, diff={diff:+.2f}%")
    print(f"  {'PASS' if abs(diff) < 2.0 else 'WARN'} (expected small difference under linear costs)")

    print("\n" + "=" * 80)
    print("TEST 4: Convex cost improvement (c=0.04, γ=3, M=5 > M=2)")
    print("=" * 80)
    contract_convex = make_contract(0.04, 3.0)
    price_2, *_ = run_lsm(contract_convex, train_ds, test_ds, n_actions=2)
    price_5, *_ = run_lsm(contract_convex, train_ds, test_ds, n_actions=5)
    diff = 100 * (price_5 - price_2) / price_2
    print(f"  M=2: {price_2:.4f}, M=5: {price_5:.4f}, diff={diff:+.2f}%")
    print(f"  {'PASS' if price_5 > price_2 else 'FAIL'} (expected M=5 > M=2 under strong convexity)")

    print("\n" + "=" * 80)
    print("TEST 5: Zero cost sanity (c=0, γ=1, M=5 ≈ M=2)")
    print("=" * 80)
    contract_zero = make_contract(0.0, 1.0)
    price_2, *_ = run_lsm(contract_zero, train_ds, test_ds, n_actions=2)
    price_5, *_ = run_lsm(contract_zero, train_ds, test_ds, n_actions=5)
    diff = 100 * (price_5 - price_2) / price_2
    print(f"  M=2: {price_2:.4f}, M=5: {price_5:.4f}, diff={diff:+.2f}%")
    print(f"  {'PASS' if abs(diff) < 2.0 else 'WARN'} (expected similar under zero costs)")


if __name__ == "__main__":
    main()
