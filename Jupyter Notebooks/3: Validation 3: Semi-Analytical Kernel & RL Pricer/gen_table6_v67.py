"""Table 6 (tab:lsm_grid) — action-grid sensitivity, regenerated under the 8-seed Results-9 protocol.

Replaces the stale single-seed-999 numbers (the focal AC-sample read 1.9233 in Table 6 vs 1.9592
in Table 5 — a cross-table contradiction). For each of the 3 representative cells
[(0.04, 1), (0.04, 2), (0.15, 2)]:

  * LSM-D M in {5, 9, 17}: fit on 32 768 paths (train seed 700+k), priced on each seed's OWN OOS
    dump (eval seed = train seed + 777) — the IDENTICAL protocol as Results 9 / Table 5. So the
    M=5 column equals Table 5's LSM-D to the last digit (asserted below), and M=9/17 sit on the
    same per-seed OOS as the AC values.
  * AC-kernel, AC-sample: pulled verbatim from 'Convex Costs Results 9.csv' (= Table 5).

Per-seed LSM prices are BCa-bootstrapped (2000 resamples, asymmetric 95%) for mean + CI, matching
Table 5's style. Reuses gen_rl_validation_v67 (cpp_run for OOS dumps, lsm_worker for the M ladder).

This supersedes gen_r4_discretisation.py as the source for tab:lsm_grid (gen_r4 remains for the
M=2..17 timing/price exploration). Output: table6_lsm_grid_v67.csv.

Run:  conda activate EP11
      cd "Jupyter Notebooks/3: Validation 3: Semi-Analytical Kernel & RL Pricer"
      python gen_table6_v67.py
Smoke: T6_FOCAL=1 VAL3_SEEDS=11,12,13 python gen_table6_v67.py
"""
import os
os.environ["VAL3_NTEST"] = "32768"      # MATCH Results 9 LSM fit-path count (force, not setdefault)
os.environ.setdefault("VAL3_SEEDS", "11,12,13,14,15,16,17,18")
os.environ.update(VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TQDM_DISABLE="1")
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_rl_validation_v67 as g   # cpp_run, lsm_worker, read_sxy, SEEDS, WORKERS, N, N_EVAL, OOS_DIR, KON_BALANCED

CELLS = [(0.04, 1.0), (0.04, 2.0), (0.15, 2.0)]   # the 3 paper cells (tab:lsm_grid)
MS = [5, 9, 17]
SEEDS = g.SEEDS
if os.environ.get("T6_FOCAL"):
    CELLS = [(0.04, 2.0)]
OOS = g.OOS_DIR
R9_PATH = g.HERE.parent / "Convex Costs Results 9.csv"
OUT = g.HERE / "table6_lsm_grid_v67.csv"

# Current (stale) Table 6 values from the manuscript, for side-by-side reporting only.
OLD = {(0.04, 1.0): dict(M5=2.2378, M9=2.2378, M17=2.2378, ACk=2.2367, ACs=2.2186),
       (0.04, 2.0): dict(M5=1.9822, M9=1.9880, M17=1.9892, ACk=1.9887, ACs=1.9233),
       (0.15, 2.0): dict(M5=1.0321, M9=1.0573, M17=1.0636, ACk=1.0653, ACs=1.0337)}


def bca(x, B=2000, alpha=0.05, seed=0):
    """Bias-corrected & accelerated bootstrap -> (mean, lo95, hi95). Same as gen_results9_v67.bca."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = x.size
    if n < 2:
        m = float(x.mean()) if n else float("nan"); return m, m, m
    th = x.mean(); bs = np.random.default_rng(seed).choice(x, (B, n), True).mean(1)
    z0 = norm.ppf((np.sum(bs < th) + 0.5) / (B + 1))
    d = x.mean() - np.array([np.delete(x, i).mean() for i in range(n)])
    acc = (d ** 3).sum() / (6 * (d ** 2).sum() ** 1.5 + 1e-300)
    z = norm.ppf([alpha / 2, 1 - alpha / 2])
    q = norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    lo, hi = np.quantile(bs, np.clip(q, 0, 1))
    return float(th), float(lo), float(hi)


def tag(c, gamma): return f"t6_c{c}_g{gamma}"


def main():
    t0 = time.time()
    assert g.N == 32768, f"VAL3_NTEST must be 32768 to match Results 9, got {g.N}"
    assert g.KERNEL_BIN.exists(), "build the v67 kernel binary first"
    r9 = pd.read_csv(R9_PATH)
    print(f"Table 6 regen | cells={CELLS} | seeds={SEEDS} | LSM fit N={g.N} | OOS {g.N_EVAL}", flush=True)

    # ---- Phase 1: dump per-seed OOS for each cell (kernel build, 4096 ep; price discarded). The OOS
    # (eval seed = seed+777) is identical to what Results 9 priced its AC values on. ----
    dump_jobs = [(c, gamma, s) for (c, gamma) in CELLS for s in SEEDS]

    def _dump(job):
        c, gamma, s = job
        g.cpp_run(c, gamma, s, 4096, g.KON_BALANCED, dump=OOS / f"{tag(c, gamma)}_{s}.bin", tag="kbal")
        return job
    with ThreadPoolExecutor(max_workers=g.WORKERS) as ex:
        list(ex.map(_dump, dump_jobs))
    print(f"  Phase 1 (OOS dumps) done in {time.time()-t0:.0f}s.", flush=True)

    # ---- Phase 2: LSM M-ladder per (cell, seed) on its OOS (N=32768 fit) ----
    lsm_jobs = [(tag(c, gamma), c, gamma, s, k, str(OOS / f"{tag(c, gamma)}_{s}.bin"), MS)
                for (c, gamma) in CELLS for k, s in enumerate(SEEDS)]
    ladder = {}
    with ProcessPoolExecutor(max_workers=g.WORKERS) as ex:
        for _, c, gamma, seed, out in ex.map(g.lsm_worker, lsm_jobs):
            ladder[(c, gamma, seed)] = out
    print(f"  Phase 2 (LSM ladder) done in {time.time()-t0:.0f}s.", flush=True)

    # ---- Phase 3: assemble. LSM ladder BCa over seeds; AC cols verbatim from Results 9. ----
    rows = []
    for (c, gamma) in CELLS:
        rec = dict(c=c, gamma=gamma, n_seeds=len(SEEDS))
        for M in MS:
            arr = np.array([ladder[(c, gamma, s)][M] for s in SEEDS])
            m, lo, hi = bca(arr)
            rec[f"LSM_M{M}"], rec[f"LSM_M{M}_lo"], rec[f"LSM_M{M}_hi"] = m, lo, hi
            rec[f"LSM_M{M}_sd"] = float(arr.std(ddof=1))
        sub = r9[np.isclose(r9.c, c) & np.isclose(r9.gamma, gamma)]
        assert len(sub) == 1, f"cell ({c},{gamma}) not unique in Results 9"
        rr = sub.iloc[0]
        rec["ACkernel"], rec["ACsample"], rec["LSM_M5_R9"] = rr.ACkernel_mean, rr.ACsample_mean, rr.LSM_mean
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    # ---- consistency checks + readable report ----
    print(f"\n  Phase 3 done. Wrote {OUT}\n", flush=True)
    ok = True
    for rec in rows:
        d = abs(rec["LSM_M5"] - rec["LSM_M5_R9"])
        flag = "OK" if d < 1e-4 else "**MISMATCH**"
        if d >= 1e-4: ok = False
        print(f"  consistency (c={rec['c']}, g={rec['gamma']}): "
              f"LSM_M5={rec['LSM_M5']:.4f} vs Results-9 LSM={rec['LSM_M5_R9']:.4f}  Δ={d:.2e}  {flag}")
    print(f"\n  {'='*92}\n  NEW Table 6 (8-seed) vs OLD (stale single-seed):")
    print(f"  {'cell':>12} | {'M=5':>16} | {'M=9':>16} | {'M=17':>16} | {'AC-kernel':>16} | {'AC-sample':>16}")
    for rec in rows:
        c, gamma = rec["c"], rec["gamma"]; o = OLD.get((c, gamma), {})
        def cmp(new, old): return f"{new:.4f}({old:+.4f})" if old else f"{new:.4f}"
        print(f"  {f'({c},{gamma})':>12} | "
              f"{cmp(rec['LSM_M5'], rec['LSM_M5']-o.get('M5',rec['LSM_M5'])):>16} | "
              f"{cmp(rec['LSM_M9'], rec['LSM_M9']-o.get('M9',rec['LSM_M9'])):>16} | "
              f"{cmp(rec['LSM_M17'], rec['LSM_M17']-o.get('M17',rec['LSM_M17'])):>16} | "
              f"{rec['ACkernel']:.4f}(old {o.get('ACk',float('nan')):.4f}) | "
              f"{rec['ACsample']:.4f}(old {o.get('ACs',float('nan')):.4f})")
    print(f"\n  {'='*92}")
    print(f"  M=17 vs AC-kernel agreement (the paper's '<0.2%' claim):")
    for rec in rows:
        gap = 100 * (rec["ACkernel"] / rec["LSM_M17"] - 1)
        print(f"    (c={rec['c']}, g={rec['gamma']}): AC-kernel={rec['ACkernel']:.4f}  LSM M=17={rec['LSM_M17']:.4f}"
              f"  -> {gap:+.2f}%")
    print(f"\n  {'CONSISTENCY OK' if ok else 'CONSISTENCY FAILED — investigate'} | total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
