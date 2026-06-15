"""Task 3 — analyze the sample-efficiency screen CSV (from collect_sampler_results.py).

For each (regime, N) and each arm vs the A0 (mc/uniform) control:
  * price superiority  : paired-seed t (CRN-shared seeds) one-sided (arm > A0), + Welch fallback
  * seed-std change    : F variance ratio (arm/A0); ratio < 1 ⇒ tighter (better)
  * "no worse" (TOST)  : paired two-one-sided test on price within ±MARGIN (default 0.5% of A0 mean)

Decision rule (per the research plan): an arm is PROMOTED if, in the screened regime(s),
its price is ≥ A0 (one-sided p<0.05) OR it reaches A0's larger-N price band at smaller N,
AND its seed-std is not significantly worse. Otherwise REJECTED (logged negative result).

Run:  EP11python tools/analyze_sampler_screen.py [csv]   # default logs/sampler_study/p1_screen.csv
      MARGIN=0.5 ...   # TOST equivalence band, % of A0 mean
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from stats_analysis import f_var_ratio, welch_t_test  # noqa: E402


def paired(arm_by_seed: dict, ref_by_seed: dict):
    seeds = sorted(set(arm_by_seed) & set(ref_by_seed))
    if len(seeds) < 2:
        return None
    d = np.array([arm_by_seed[s] - ref_by_seed[s] for s in seeds])
    n = d.size
    mean = float(d.mean()); sd = float(d.std(ddof=1)); se = sd / np.sqrt(n)
    if se == 0:
        return dict(n=n, mean=mean, se=0.0, p_gt=0.0 if mean > 0 else 1.0, t=np.inf)
    t = mean / se
    p_gt = float(1 - sps.t.cdf(t, n - 1))     # H1: arm > ref
    return dict(n=n, mean=mean, se=se, t=t, p_gt=p_gt, d=d, seeds=seeds)


def tost(d: np.ndarray, margin: float):
    """Paired TOST: H0 |mean| >= margin vs H1 within (-margin, margin). Returns max of the two p."""
    n = d.size; mean = float(d.mean()); se = float(d.std(ddof=1)) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(mean) < margin else 1.0
    t_lo = (mean + margin) / se      # H1a: mean > -margin
    t_hi = (mean - margin) / se      # H1b: mean <  margin
    p_lo = float(1 - sps.t.cdf(t_lo, n - 1))
    p_hi = float(sps.t.cdf(t_hi, n - 1))
    return max(p_lo, p_hi)           # equivalence established if this < 0.05


def main():
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "sampler_study" / "p1_screen.csv"
    margin_pct = float(os.environ.get("MARGIN", 0.5))
    df = pd.read_csv(csv)
    print(f"loaded {len(df)} rows from {csv}\n")

    verdicts = {}   # arm -> list of (regime,N,status)
    for regime in sorted(df.regime.unique()):
        for N in sorted(df.N.unique()):
            sub = df[(df.regime == regime) & (df.N == N)]
            if sub.empty:
                continue
            ref = sub[sub.arm == "A0"]
            ref_by_seed = dict(zip(ref.seed, ref.price))
            ref_mean = float(ref.price.mean()); ref_std = float(ref.price.std(ddof=1))
            margin = margin_pct / 100.0 * ref_mean
            print(f"### regime={regime}  N={N}  | A0: mean={ref_mean:.4f} std={ref_std:.4f} "
                  f"(n={len(ref)})  TOST±{margin:.4f}")
            print(f"{'arm':>4} {'mean':>9} {'Δprice':>9} {'paired_p>':>10} {'std':>8} "
                  f"{'std_ratio':>9} {'F_p':>7} {'TOST_p':>7}  verdict")
            for arm in sorted(a for a in sub.arm.unique() if a != "A0"):
                ar = sub[sub.arm == arm]
                arm_by_seed = dict(zip(ar.seed, ar.price))
                pr = paired(arm_by_seed, ref_by_seed)
                amean = float(ar.price.mean()); astd = float(ar.price.std(ddof=1))
                fr = f_var_ratio(ar.price.values, ref.price.values)
                std_ratio = fr.get("std_ratio", float("nan")) if fr.get("valid") else float("nan")
                f_p = fr.get("p", float("nan")) if fr.get("valid") else float("nan")
                tp = tost(pr["d"], margin) if pr and "d" in pr else float("nan")
                dp = pr["mean"] if pr else float("nan")
                pgt = pr["p_gt"] if pr else float("nan")
                # verdict: better price (one-sided) OR equivalent (TOST), AND not worse std
                better = pgt < 0.05
                equiv = tp < 0.05
                worse_std = (std_ratio > 1.0) and (f_p < 0.05)
                if (better or equiv) and not worse_std:
                    status = "PROMOTE" if better else "tie-ok"
                else:
                    status = "reject"
                verdicts.setdefault(arm, []).append((regime, N, status))
                print(f"{arm:>4} {amean:>9.4f} {dp:>+9.4f} {pgt:>10.3f} {astd:>8.4f} "
                      f"{std_ratio:>9.3f} {f_p:>7.3f} {tp:>7.3f}  {status}")
            print()

    print("=== arm summary (status across regime×N cells) ===")
    for arm, cells in sorted(verdicts.items()):
        promo = sum(1 for _, _, s in cells if s == "PROMOTE")
        tie = sum(1 for _, _, s in cells if s == "tie-ok")
        rej = sum(1 for _, _, s in cells if s == "reject")
        print(f"  {arm}: PROMOTE={promo}  tie-ok={tie}  reject={rej}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
