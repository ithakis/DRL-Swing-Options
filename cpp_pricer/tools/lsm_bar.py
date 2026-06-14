#!/usr/bin/env python3
"""O1 — the LSM-D benchmark bar at the 2k-4k-path regime (the target the RL must beat).

For each seed, trains LSM on n_train paths (matched to the RL), prices OUT-OF-SAMPLE on a
65k test set (seed+777, matching the C++ RL eval-seed convention), across seeds 11-25.
Reports, per (n_train, state_mode, degree): mean price (accuracy) + seed-std (variance) + fit/eval time.

LSM-D = Chebyshev basis. "5-7 estimators" = reduced(spot-only) degree 4/6 (=5/7 regressors);
also reports the canonical full-state degree-2 paper benchmark and reduced degree-2.

Run:  ~/miniforge3/bin/python3 cpp_pricer/tools/lsm_bar.py [--seeds 11-25] [--budgets 2048,4096]
Out:  cpp_pricer/data/research/lsm_bar.json  (+ console table)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.swing_contract import SwingContract
from src.simulate_hhk_spot import simulate_hhk_spot, no_seasonal_function
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos

# Focal contract (matches cpp_pricer/include/config.hpp exactly)
CONTRACT = SwingContract(q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0,
                         maturity=0.0833, n_rights=22, min_refraction_periods=0,
                         r=0.05, c_cost=0.04, gamma_cost=2.0)
HHK = dict(S0=1.0, T=CONTRACT.maturity, n_steps=CONTRACT.n_rights - 1,
           alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3,
           f=no_seasonal_function, dtype=np.float32)
N_TEST = 65536
N_ACTIONS = 5   # canonical LSM action grid {0, q/4, q/2, 3q/4, q}

# (state_mode, degree) configs.  reduced degree d => d+1 regressors.
CONFIGS = [("reduced", 2), ("reduced", 4), ("reduced", 6), ("full", 2)]


def lsm_price(n_train, seed, state_mode, degree):
    tr = simulate_hhk_spot(**HHK, n_paths=n_train, seed=seed, stratify=True, batch_size=128)
    tr = tuple(np.asarray(a, dtype=np.float64) for a in tr)
    t0 = time.perf_counter()
    est = fit_lsm_estimators(contract=CONTRACT, dataset=tr, poly_degree=degree,
                             basis_type="chebyshev", state_mode=state_mode,
                             reg_type="none", reg_alpha=1e-6, n_actions=N_ACTIONS)
    t_fit = time.perf_counter() - t0
    te = simulate_hhk_spot(**HHK, n_paths=N_TEST, seed=seed + 777, stratify=False, batch_size=128)
    te = tuple(np.asarray(a, dtype=np.float64) for a in te)
    t0 = time.perf_counter()
    price, _ = price_swing_option_lsm_oos(contract=CONTRACT, dataset=te, estimators=est,
                                          seed=seed + 777, csv_path=None, _print_results=False)
    t_eval = time.perf_counter() - t0
    return price, t_fit, t_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11-25")
    ap.add_argument("--budgets", default="2048,3072,4096")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-")) if "-" in a.seeds else (None, None)
    seeds = list(range(lo, hi + 1)) if lo else [int(x) for x in a.seeds.split(",")]
    budgets = [int(x) for x in a.budgets.split(",")]

    out = {}
    print(f"LSM-D bar | focal c=0.04 g=2 | {len(seeds)} seeds | test {N_TEST} OOS | n_actions={N_ACTIONS}")
    print(f"{'budget':>7} {'state':>8} {'deg':>3} {'regr':>4} | {'price':>9} {'seed-std':>9} | {'t_fit':>6} {'t_eval':>6}")
    for nt in budgets:
        for sm, deg in CONFIGS:
            prices, tf, tev = [], [], []
            for s in seeds:
                p, a_, b_ = lsm_price(nt, s, sm, deg)
                prices.append(p); tf.append(a_); tev.append(b_)
            mean = float(np.mean(prices)); sd = float(np.std(prices, ddof=0))
            regr = (deg + 1) if sm == "reduced" else None
            key = f"{nt}_{sm}_d{deg}"
            out[key] = dict(budget=nt, state=sm, degree=deg, regressors=regr,
                            price=mean, seed_std=sd, prices=prices,
                            t_fit=float(np.mean(tf)), t_eval=float(np.mean(tev)))
            print(f"{nt:>7} {sm:>8} {deg:>3} {str(regr):>4} | {mean:>9.4f} {sd:>9.4f} | "
                  f"{np.mean(tf):>6.2f} {np.mean(tev):>6.2f}")
    outp = ROOT / "cpp_pricer/data/research/lsm_bar.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
