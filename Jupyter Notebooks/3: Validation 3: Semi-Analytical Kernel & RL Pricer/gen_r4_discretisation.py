"""Validation 3 - R4 support: LSM-D price / 95% CI / compute-time as a function of the
action-discretisation level M, for each of the 4 regimes.

Extends the action grid below the canonical {5,9,17} down to the coarsest meaningful
levels M in {2,3,4} (M=2 is bang-bang; M=1 is degenerate -- no exercise-size choice --
and is rejected by the LSM pricer, so it is omitted).  For each (regime, M) we record:
  * price, ci95      -- OOS price on the common fresh test set (seed 999, 65 536 paths)
  * wall_fit         -- LSM regression-fit wall time (s)
  * wall_total       -- fit + OOS pricing wall time (s)  == "time to calculate the price"

Same protocol as gen_rl_validation.py (chebyshev, degree 2, full state, train seed 998).

Run:  python gen_r4_discretisation.py
"""
from __future__ import annotations
import contextlib, io, json, time
from pathlib import Path

import numpy as np
import pandas as pd

import gen_rl_validation as G  # reuse discovery + builders + constants
import rebuild_results_v7 as rb
from src.simulate_hhk_spot import simulate_hhk_spot
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos

HERE = Path(__file__).resolve().parent
M_GRID = [2, 3, 4, 5, 9, 17]   # M=1 is degenerate (no action choice) -> omitted


N_TIME = 3  # timing repetitions; we report the MIN (least-contended) of CPU time


def lsm_at_M(contract, train, test, M, key):
    """Price LSM at action level M, plus a CONTENTION-ROBUST compute-time measurement.

    Timing uses ``time.process_time()`` (CPU time charged to THIS process only — the LSM
    pricer is single-threaded numba, ``@jit(nopython=True)`` with no ``prange``), so other
    processes running in the background do not inflate it the way wall-clock would.  We take
    the MIN over ``N_TIME`` repeats (the cleanest run) for extra robustness.  ``wall_total``
    is this robust CPU "time to compute the price"; ``wall_clock_total`` keeps the raw
    wall-clock for reference.
    """
    pq = HERE / "_lsm_parquet" / f"{key}_M{M}.parquet"
    pq.parent.mkdir(exist_ok=True)
    # --- official fit + price (for the reported price / CI) ---
    with contextlib.redirect_stdout(io.StringIO()):
        est = fit_lsm_estimators(contract=contract, dataset=train, poly_degree=G.LSM_DEGREE,
                                 basis_type=G.LSM_BASIS, state_mode="full", reg_type="none",
                                 reg_alpha=1e-6, n_actions=M)
        price, (lo, hi) = price_swing_option_lsm_oos(contract=contract, dataset=test,
                                                     estimators=est, seed=G.TEST_SEED + 1,
                                                     csv_path=str(pq), _print_results=False)
    # --- contention-robust timing: min CPU-time over N_TIME repeats ---
    fit_cpu, tot_cpu, tot_wall = [], [], []
    for _ in range(N_TIME):
        with contextlib.redirect_stdout(io.StringIO()):
            c0, w0 = time.process_time(), time.perf_counter()
            est_t = fit_lsm_estimators(contract=contract, dataset=train, poly_degree=G.LSM_DEGREE,
                                       basis_type=G.LSM_BASIS, state_mode="full", reg_type="none",
                                       reg_alpha=1e-6, n_actions=M)
            c_fit = time.process_time() - c0
            price_swing_option_lsm_oos(contract=contract, dataset=test, estimators=est_t,
                                       seed=G.TEST_SEED + 1, csv_path=None, _print_results=False)
            tot_cpu.append(time.process_time() - c0)
            tot_wall.append(time.perf_counter() - w0)
            fit_cpu.append(c_fit)
    return dict(price=price, ci95=(hi - lo) / 2.0,
                wall_fit=min(fit_cpu), wall_total=min(tot_cpu),
                wall_clock_total=min(tot_wall), n_time=N_TIME)


def main():
    rows = []
    for key, c, gamma in G.REGIMES:
        params = rb.dotdict(json.load(open(next(iter(G.discover_v64_canonical(c, gamma).values())))))
        contract = rb.build_contract(params)
        hhk = rb.build_hhk_params(params)
        with contextlib.redirect_stdout(io.StringIO()):
            test = tuple(np.asarray(a, np.float64) for a in
                         simulate_hhk_spot(**hhk, n_paths=G.N, seed=G.TEST_SEED, stratify=True, batch_size=128))
            train = tuple(np.asarray(a, np.float64) for a in
                          simulate_hhk_spot(**hhk, n_paths=G.N, seed=G.TRAIN_SEED, stratify=True, batch_size=128))
        for M in M_GRID:
            r = lsm_at_M(contract, train, test, M, key)
            rows.append(dict(regime=key, c=c, gamma=gamma, M=M, **r))
            print(f"[{key:>6}] M={M:>2}: price={r['price']:.4f} ci95={r['ci95']:.4f} "
                  f"fit={r['wall_fit']:.1f}s total={r['wall_total']:.1f}s")
    pd.DataFrame(rows).to_csv(HERE / "lsm_discretisation.csv", index=False)
    print(f"\nWrote lsm_discretisation.csv ({len(rows)} rows)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    main()
