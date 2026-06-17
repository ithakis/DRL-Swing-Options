"""Convex Costs Results 8 — the full c x gamma table under **v67**, methodology-matched to
Validation-3 R1.  For every one of the 29 (c, gamma) cells we price THREE methods on the SAME
out-of-sample paths and bootstrap each:

  * lsm_M5     = full-state Chebyshev deg-2 LSM-D, 5 action levels (Python; src/lsm_swing_pricer)
  * kernel_on  = v67 kernel mode  = build_v67_kernel (GELU-beta3), balanced v65 recipe, 4096 ep
  * kernel_off = v67 no-kernel    = build_v67_nokernel (optimised literal-v61 + eval-EMA), 4096 ep

Methodology: 4 estimators per method, all evaluated on the SAME 4 OOS 65 536-path sets (OOS k is
dumped once by the kernel C++ run at eval seed = train_seed_k + 777; the no-kernel build reproduces
the identical OOS deterministically, and Python LSM prices the same dumped paths).  Per cell we
bootstrap (1000 resamples) the mean and +/-95% CI (= 1.96 x bootstrap SE) of each method, and the
PAIRED Delta% (each RL price vs the LSM price on the *same* OOS set).

Reuses all the C++/LSM plumbing from gen_rl_validation_v67.py.

Produces (in this folder's parent = "Jupyter Notebooks/"):
  Convex Costs Results 8.csv     wide summary (one row per cell; mean/CI95/Delta% for each method)
  results8_pricing_long.csv      long format (Configuration,c,gamma,method,seed,price)

Run:  EP11python gen_results8_v67.py        (EP11 = ~/miniforge3/envs/EP11/bin/python)
      VAL3_NTEST=16384 EP11python gen_results8_v67.py   # quick pass
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import gen_rl_validation_v67 as g   # cpp_run, read_sxy, lsm_worker, KON_BALANCED, koff_flags, bins

HERE = g.HERE
NB_DIR = HERE.parent                       # "Jupyter Notebooks/"
SEEDS = [11, 12, 13, 14]                   # 4 seeds per method (user spec)
N_TRAIN = 4096                             # matches R1 budget for both RL modes
OOS_DIR = g.CPP_DIR / "data" / "_v67oos8"; OOS_DIR.mkdir(exist_ok=True)

# Full convex-cost grid (mirrors "Convex Costs Results 7.csv"): c=0 has gamma=1 only.
GAMMAS = [(1.0, "1"), (1.5, "1.5"), (2.0, "2"), (3.0, "3")]
GRID = [(0.0, 1.0, "1")] + [(c, gv, gl) for c in (0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15)
                            for gv, gl in GAMMAS]


def cfg_name(c, gl):
    return f"SwingOption_20_c{c:.2f}_gamma{gl}"


def boot(x, n_boot=1000, seed=0):
    """Bootstrap mean and SE (std of the resampled means); CI95 = 1.96*SE downstream."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(x, (n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(means.std())


def _rl_job(job):
    kind, c, gamma, gl, seed = job
    if kind == "K":
        dump = OOS_DIR / f"{cfg_name(c, gl)}_{seed}.bin"
        return job, g.cpp_run(c, gamma, seed, N_TRAIN, g.KON_BALANCED, dump=dump, tag="kbal")
    return job, g.cpp_run(c, gamma, seed, N_TRAIN, None, tag="nokernel")


def main():
    t0 = time.time()
    print(f"Results 8 generator | N={g.N} | {len(GRID)} cells x {len(SEEDS)} seeds | "
          f"{g.WORKERS} workers x 1 thread\n", flush=True)
    assert g.KERNEL_BIN.exists() and g.NOKERNEL_BIN.exists(), "build both v67 binaries first"

    # ---- Phase A: all RL runs in parallel (kernel dumps the shared OOS; no-kernel reuses it) ----
    jobs = [(k, c, gamma, gl, s) for (c, gamma, gl) in GRID for s in SEEDS for k in ("K", "N")]
    rl, done = {}, 0
    with ThreadPoolExecutor(max_workers=g.WORKERS) as ex:
        for job, price in ex.map(_rl_job, jobs):
            rl[job] = price; done += 1
            if done % g.WORKERS == 0:
                print(f"  RL {done}/{len(jobs)} runs done [{time.time()-t0:.0f}s]", flush=True)
    print(f"  Phase A (RL) done in {time.time()-t0:.0f}s.", flush=True)

    # ---- Phase B: LSM-D (M=5) priced on the identical dumped OOS sets (process pool) ----
    lsm_jobs = [(cfg_name(c, gl), c, gamma, s, k, str(OOS_DIR / f"{cfg_name(c, gl)}_{s}.bin"), [5])
                for (c, gamma, gl) in GRID for k, s in enumerate(SEEDS)]
    lsm = {}
    with ProcessPoolExecutor(max_workers=g.WORKERS) as ex:
        for name, c, gamma, s, out in ex.map(g.lsm_worker, lsm_jobs):
            lsm[(name, s)] = out[5]
    print(f"  Phase B (LSM) done in {time.time()-t0:.0f}s.", flush=True)

    # ---- assemble long + wide ----
    long_rows, wide_rows = [], []
    for (c, gamma, gl) in GRID:
        name = cfg_name(c, gl)
        prices = {"lsm_M5": [], "kernel_on": [], "kernel_off": []}
        for s in SEEDS:
            pl = lsm[(name, s)]
            pk = rl[("K", c, gamma, gl, s)]
            pn = rl[("N", c, gamma, gl, s)]
            for m, p in [("lsm_M5", pl), ("kernel_on", pk), ("kernel_off", pn)]:
                prices[m].append(p)
                long_rows.append(dict(Configuration=name, c=c, gamma=gamma, method=m, seed=s, price=p))
        L = np.array(prices["lsm_M5"]); K = np.array(prices["kernel_on"]); Nn = np.array(prices["kernel_off"])
        lm, ls = boot(L); km, ks = boot(K); nm, ns = boot(Nn)
        kdm, kds = boot(100 * (K / L - 1)); ndm, nds = boot(100 * (Nn / L - 1))
        wide_rows.append(dict(
            Configuration=name, c=c, gamma=gamma,
            LSM_mean=lm, LSM_CI95=1.96 * ls,
            KernelOn_mean=km, KernelOn_CI95=1.96 * ks,
            KernelOn_Delta_pct=kdm, KernelOn_Delta_CI95=1.96 * kds,
            KernelOff_mean=nm, KernelOff_CI95=1.96 * ns,
            KernelOff_Delta_pct=ndm, KernelOff_Delta_CI95=1.96 * nds))

    pd.DataFrame(long_rows).to_csv(HERE / "results8_pricing_long.csv", index=False)
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(NB_DIR / "Convex Costs Results 8.csv", index=False)
    print(f"\nDone in {time.time()-t0:.0f}s.")
    print(f"  wide -> {NB_DIR / 'Convex Costs Results 8.csv'}")
    print(f"  long -> {HERE / 'results8_pricing_long.csv'}")
    print(wide.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
