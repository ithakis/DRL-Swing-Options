"""A2 — Sample-efficiency curve (new §5 exhibit), 8-seed Results-9 protocol, focal cell (0.04, 2).

For episode budgets B in {2048, 4096, 8192, 32768}, price both headline RL methods at their
Table-5 recipes (so the curve passes THROUGH the Table-5 points):
  * AC-kernel  = v67 kernel build, KON_BALANCED (ln3)  -> B=4096 reproduces Table 5's 1.9850
  * AC-sample  = v67 no-kernel build, koff recipe       -> B=32768 reproduces Table 5's 1.9592
8 seeds {11..18}, each on its own OOS (eval seed k+777). BCa(2000) per (method, B). The LSM-D
benchmark (fit on 32 768 paths, M=5) is drawn as a horizontal reference = Results 9's 1.9823.

Reuses gen_rl_validation_v67.cpp_run (cache already holds kbal@4096 and nokernel@32768).
Output: episode_efficiency_v67.csv + ../../figs/convex_costs_0p04/sample_efficiency.pdf
Run:  python gen_efficiency_v67.py            (add --plot_only to re-draw from the CSV)
"""
import os
os.environ.update(VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TQDM_DISABLE="1")
os.environ.setdefault("VAL3_SEEDS", "11,12,13,14,15,16,17,18")
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_rl_validation_v67 as g

C, GAMMA = 0.04, 2.0
BUDGETS = [2048, 4096, 8192, 32768]
SEEDS = g.SEEDS
OUT_CSV = g.HERE / "episode_efficiency_v67.csv"
FIG = g.HERE.parents[1] / "figs" / "convex_costs_0p04" / "sample_efficiency.pdf"
R9 = pd.read_csv(g.HERE.parent / "Convex Costs Results 9.csv")
LSM_BENCH = float(R9[np.isclose(R9.c, C) & np.isclose(R9.gamma, GAMMA)].iloc[0].LSM_mean)


def bca(x, B=2000, alpha=0.05, seed=0):
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


def compute():
    t0 = time.time()
    # Phase 0: dump the g2 OOS per seed so the LSM-fit-at-budget curve is priced on the IDENTICAL
    # paths the RL agents were evaluated on (apples-to-apples training-cost comparison).
    def _dump(s):
        g.cpp_run(C, GAMMA, s, 4096, g.KON_BALANCED, dump=g.OOS_DIR / f"g2eff_{s}.bin", tag="kbal")
        return s
    with ThreadPoolExecutor(max_workers=g.WORKERS) as ex:
        list(ex.map(_dump, SEEDS))

    # Phase 1: RL = AC-kernel (KON_BALANCED) + AC-sample (no-kernel) at each budget. job=(method,B,seed).
    jobs = [(m, B, s) for m in ("ACkernel", "ACsample") for B in BUDGETS for s in SEEDS]

    def run(job):
        m, B, s = job
        p = g.cpp_run(C, GAMMA, s, B, g.KON_BALANCED, tag="kbal") if m == "ACkernel" \
            else g.cpp_run(C, GAMMA, s, B, None, tag="nokernel")
        return job, p
    prices = {}
    with ThreadPoolExecutor(max_workers=g.WORKERS) as ex:
        for job, p in ex.map(run, jobs):
            prices[job] = p
    print(f"  RL runs done in {time.time()-t0:.0f}s.", flush=True)

    # Phase 2: LSM-fit-at-budget curve (fairness upgrade) -- LSM-D M=5 fit on B paths, priced on the
    # same g2 OOS. Reuses gen_rl_validation_v67.lsm_r2_worker (focal c=0.04, gamma=2 hardwired).
    lsm_jobs = [(s, k, str(g.OOS_DIR / f"g2eff_{s}.bin"), BUDGETS) for k, s in enumerate(SEEDS)]
    lsm_curve = {}
    with ProcessPoolExecutor(max_workers=g.WORKERS) as ex:
        for s, out in ex.map(g.lsm_r2_worker, lsm_jobs):
            lsm_curve[s] = out
    print(f"  LSM-fit-at-budget curve done in {time.time()-t0:.0f}s.", flush=True)

    rows = []
    for m in ("ACkernel", "ACsample"):
        for B in BUDGETS:
            arr = np.array([prices[(m, B, s)] for s in SEEDS])
            mean, lo, hi = bca(arr)
            dpm, dlo, dhi = bca(100 * (arr / LSM_BENCH - 1))
            rows.append(dict(method=m, episodes=B, n_seeds=len(SEEDS), price=mean,
                             price_lo=lo, price_hi=hi, price_sd=float(arr.std(ddof=1)),
                             delta_pct=dpm, delta_lo=dlo, delta_hi=dhi))
    for B in BUDGETS:                                   # LSM-D fit on B paths (its own sample-efficiency)
        arr = np.array([lsm_curve[s][B] for s in SEEDS])
        mean, lo, hi = bca(arr)
        rows.append(dict(method="LSM_fit", episodes=B, n_seeds=len(SEEDS), price=mean,
                         price_lo=lo, price_hi=hi, price_sd=float(arr.std(ddof=1)),
                         delta_pct=np.nan, delta_lo=np.nan, delta_hi=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"  wrote {OUT_CSV}", flush=True)
    return df


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Sized for a single cas-dc column (columnwidth = 3.30 in) at scale 1.0, so the on-page
    # font sizes match the surrounding ~9pt body / 8pt caption text without LaTeX rescaling.
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 130, "axes.labelsize": 8.5,
                         "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
    sty = {"ACkernel": dict(c="#1b7837", marker="o", label="AC-kernel (HHK target)"),
           "ACsample": dict(c="#d95f02", marker="s", label="AC-sample (model-agnostic)"),
           "LSM_fit": dict(c="#3b6fb0", marker="^", label="LSM-D ($M{=}5$, fit on $N$ paths)")}
    fig, ax = plt.subplots(figsize=(3.30, 2.55))
    ax.axhline(LSM_BENCH, color="#888", ls=":", lw=1.0, zorder=0,
               label=f"LSM-D full ($32{{,}}768$ paths) = {LSM_BENCH:.4f}")
    for m, s in sty.items():
        d = df[df.method == m].sort_values("episodes")
        ax.fill_between(d.episodes, d.price_lo, d.price_hi, color=s["c"], alpha=0.15)
        ax.plot(d.episodes, d.price, color=s["c"], marker=s["marker"], ms=4, lw=1.4, label=s["label"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS); ax.set_xticklabels([f"{b//1024}k" if b >= 1024 else b for b in BUDGETS])
    ax.set_xlabel("training budget (RL episodes / LSM fit paths)"); ax.set_ylabel("out-of-sample value")
    ax.set_title(f"Sample efficiency at the focal cell ($c{{=}}0.04,\\ \\gamma_c{{=}}2$)")
    ax.legend(fontsize=6, loc="lower right", handlelength=1.6, borderpad=0.4, labelspacing=0.3)
    fig.tight_layout(pad=0.4); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight"); fig.savefig(FIG.with_suffix(".png"), bbox_inches="tight")
    print(f"  wrote {FIG}", flush=True)


def main():
    if "--plot_only" in sys.argv:
        df = pd.read_csv(OUT_CSV)
    else:
        df = compute()
    # readable summary + the Table-5 anchor checks
    print(f"\n  LSM-D benchmark (Results 9) = {LSM_BENCH:.4f}")
    for m in ("ACkernel", "ACsample"):
        print(f"  {m}:")
        for _, r in df[df.method == m].sort_values("episodes").iterrows():
            print(f"    B={int(r.episodes):>5}: {r.price:.4f} [{r.price_lo:.4f},{r.price_hi:.4f}]"
                  f"  Δ%={r.delta_pct:+.2f} [{r.delta_lo:+.2f},{r.delta_hi:+.2f}]")
    ack4 = df[(df.method == "ACkernel") & (df.episodes == 4096)].iloc[0].price
    acs32 = df[(df.method == "ACsample") & (df.episodes == 32768)].iloc[0].price
    print(f"\n  Table-5 anchor check: AC-kernel@4k={ack4:.4f} (T5 1.9850)  "
          f"AC-sample@32k={acs32:.4f} (T5 1.9592)")
    plot(df)


if __name__ == "__main__":
    main()
