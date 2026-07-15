"""Convex Costs Results 9 — the paper's headline table, one self-contained run.

Three methods priced on the SAME 65 536-path OOS set per seed (f32 C++ sim; LSM reads the dump):
  * LSM        = full-state Chebyshev deg-2 LSM-D, M=5, fit on 32 768 paths (Python)
  * AC-sample  = v67 no-kernel C++ build, 32 768 episodes
  * AC-kernel  = v67 kernel    C++ build,  4 096 episodes
8 seeds/method (11..18); seed k -> its own OOS at eval-seed k+777. Per cell we BCa-bootstrap
(2000 resamples, asymmetric 95% interval) each method's mean price AND the paired Δ% (RL/LSM-1 on
the same OOS), and report the per-seed SD. The CSV is the paper table verbatim (just round + LaTeX it).

RAM-safe (≤8 GB): one cell at a time, OOS dumps deleted after the row is written. Restartable: skips
cells already in the CSV and reuses the C++/LSM on-disk caches. 8 single-thread C++ runs in parallel.

Run:  conda activate EP11
      cd "Jupyter Notebooks/3: Validation 3: Semi-Analytical Kernel & RL Pricer"
      python gen_results9_v67.py
Smoke test first:  VAL3_NTEST=16384 VAL3_NEVAL=16384 python gen_results9_v67.py
"""
import os, sys, time, csv
os.environ.update(VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",  # pin BLAS for the in-proc LSM
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TQDM_DISABLE="1")  # quiet LSM sim bars
os.environ.setdefault("VAL3_NTEST", "32768")   # LSM-D *fit* paths (overridable for smoke tests)
os.environ.setdefault("VAL3_NEVAL", "65536")   # OOS evaluation paths (all 3 methods)
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_rl_validation_v67 as g  # cpp_run, lsm_worker, KON_BALANCED, WORKERS, N(=32768 LSM fit), N_EVAL(=65536 OOS)

SEEDS = g.SEEDS                                          # honors VAL3_SEEDS (default 11..18)
EP_K = int(os.environ.get("R9_EP_K", 4096))              # AC-kernel episode budget
EP_N = int(os.environ.get("R9_EP_N", 32768))             # AC-sample episode budget
GAMMAS = [(1.0, "1"), (1.5, "1.5"), (2.0, "2"), (3.0, "3")]
GRID = [(0.0, 1.0, "1")] + [(c, gv, gl) for c in (0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15) for gv, gl in GAMMAS]
if os.environ.get("R9_FOCAL"):                           # smoke test: single focal cell
    GRID = [(0.04, 2.0, "2")]
OUT = g.HERE.parent / "Convex Costs Results 9.csv"
OOS = g.CPP_DIR / "data" / "_v9oos"; OOS.mkdir(exist_ok=True)
HDR = ["Configuration", "c", "gamma", "LSM_fit_paths", "ACsample_ep", "ACkernel_ep", "n_oos", "n_seeds",
       "LSM_mean", "LSM_lo", "LSM_hi", "LSM_sd", "ACsample_mean", "ACsample_lo", "ACsample_hi", "ACsample_sd",
       "ACkernel_mean", "ACkernel_lo", "ACkernel_hi", "ACkernel_sd",
       "ACsample_delta_pct", "ACsample_delta_lo", "ACsample_delta_hi",
       "ACkernel_delta_pct", "ACkernel_delta_lo", "ACkernel_delta_hi"]


def bca(x, B=2000, alpha=0.05, seed=0):
    """Bias-corrected & accelerated bootstrap -> (mean, lo95, hi95); asymmetric, calibrated for n=8."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = x.size
    if n < 2:
        m = float(x.mean()) if n else float("nan"); return m, m, m
    th = x.mean(); bs = np.random.default_rng(seed).choice(x, (B, n), True).mean(1)
    z0 = norm.ppf((np.sum(bs < th) + 0.5) / (B + 1))
    d = x.mean() - np.array([np.delete(x, i).mean() for i in range(n)])  # jackknife acceleration
    acc = (d ** 3).sum() / (6 * (d ** 2).sum() ** 1.5 + 1e-300)
    z = norm.ppf([alpha / 2, 1 - alpha / 2])
    q = norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    lo, hi = np.quantile(bs, np.clip(q, 0, 1))
    return float(th), float(lo), float(hi)


def name_of(c, gl): return f"SwingOption_20_c{c:.2f}_gamma{gl}"


def main():
    done = set(np.loadtxt(OUT, delimiter=",", usecols=0, dtype=str, skiprows=1, ndmin=1)) if OUT.exists() else set()
    need_hdr = (not OUT.exists()) or OUT.stat().st_size == 0   # header once, never on restart
    f = open(OUT, "a", newline=""); w = csv.writer(f)
    if need_hdr: w.writerow(HDR); f.flush()
    print(f"Results 9 | {len(GRID)} cells x {len(SEEDS)} seeds | LSM fit {g.N} / AC-sample {EP_N}ep / "
          f"AC-kernel {EP_K}ep | OOS {g.N_EVAL} | {g.WORKERS} cores | {len(done)} cells already done", flush=True)
    t0, ran, total = time.time(), 0, len(GRID)
    for c, gamma, gl in GRID:
        nm = name_of(c, gl)
        if nm in done:
            continue
        ts = time.time()
        jobs = [("K", s) for s in SEEDS] + [("N", s) for s in SEEDS]
        run = lambda j: (j, g.cpp_run(c, gamma, j[1], EP_K, g.KON_BALANCED, dump=OOS / f"{nm}_{j[1]}.bin",
                                      tag="kbal") if j[0] == "K"
                         else g.cpp_run(c, gamma, j[1], EP_N, None, tag="nokernel"))
        with ThreadPoolExecutor(max_workers=g.WORKERS) as ex:
            rl = dict(ex.map(run, jobs))
        ljobs = [(nm, c, gamma, s, k, str(OOS / f"{nm}_{s}.bin"), [5]) for k, s in enumerate(SEEDS)]
        with ProcessPoolExecutor(max_workers=g.WORKERS) as ex:
            lsm = {s: out[5] for _, _, _, s, out in ex.map(g.lsm_worker, ljobs)}
        L = np.array([lsm[s] for s in SEEDS]); Nn = np.array([rl[("N", s)] for s in SEEDS]); K = np.array([rl[("K", s)] for s in SEEDS])
        row = [nm, c, gamma, g.N, EP_N, EP_K, g.N_EVAL, len(SEEDS)]
        for a in (L, Nn, K):
            row += [*bca(a), float(a.std(ddof=1))]
        for a in (Nn, K):
            row += list(bca(100 * (a / L - 1)))
        w.writerow(row); f.flush()
        for s in SEEDS:
            (OOS / f"{nm}_{s}.bin").unlink(missing_ok=True)
        ran += 1; el = time.time() - t0; eta = el / ran * (total - len(done) - ran)
        print(f"[{len(done)+ran}/{total}] {nm}  cell {time.time()-ts:.0f}s | elapsed {el/60:.1f}m | "
              f"ETA {eta/60:.1f}m  (Lavg {L.mean():.4f})", flush=True)
    f.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f}m -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
