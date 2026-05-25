"""
Variance-aware analysis of the H4-v2 sweep.

For each config (CTRL, H1_only, H4v2_*):
  * Per-seed Delta% = (eval_price / lsm_price - 1) * 100
  * Mean Delta%, sample std, standard error of mean (= std / sqrt(n))
  * 95% CI on mean (mean +/- 1.96 * SE)
  * Mean wall-clock seconds

Then for each non-reference config, report the *gap* to H1_only:
  gap = mean_config - mean_H1
  SE(gap) = sqrt(SE_config^2 + SE_H1^2)
  z = gap / SE(gap)

A config is decisively better than H1_only only if z > +1.96 AND SE is
acceptable.  A config is no different if |z| < 1.0.  A worse config has
z < -1.0.

Usage:
    python tools/analyze_h4_v2.py
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "logs" / "_sweep_h1" / "sweep_h4_v2_n4096.csv"


def load() -> Dict[str, List[Dict]]:
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return {}
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            rows.append(r)
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    return by_label


def per_config_stats(rows: List[Dict]) -> Dict:
    deltas: List[float] = []
    walls: List[float] = []
    evals: List[float] = []
    lsms: List[float] = []
    for r in rows:
        try:
            ep = float(r["eval_price"]); lsm = float(r["lsm_price"])
            if lsm <= 0: continue
            d = (ep / lsm - 1.0) * 100.0
            deltas.append(d); evals.append(ep); lsms.append(lsm)
            walls.append(float(r.get("wall_seconds", "0")))
        except Exception:
            continue
    n = len(deltas)
    if n == 0:
        return {"n": 0}
    mean = sum(deltas) / n
    std = statistics.stdev(deltas) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean_delta": mean,
        "std_delta": std,
        "se_delta": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "mean_eval": sum(evals) / n,
        "mean_lsm": sum(lsms) / n,
        "mean_wall": sum(walls) / n,
        "seeds": [r.get("seed", "") for r in rows],
    }


def main() -> None:
    by_label = load()
    if not by_label:
        return

    print(f"H4-v2 sweep analysis ({CSV_PATH.name})\n")
    print(f"{'config':<22} {'n':>3} {'mean_Δ%':>9} {'std':>7} {'SE':>7} "
          f"{'95% CI':>20} {'wall_s':>8}")
    print("-" * 90)

    stats = {label: per_config_stats(rows) for label, rows in by_label.items()}
    # Sort by mean delta descending
    order = sorted(stats.items(), key=lambda kv: -(kv[1].get("mean_delta", -1e9)))
    for label, s in order:
        if s["n"] == 0:
            print(f"{label:<22} no runs")
            continue
        ci_str = f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
        print(f"{label:<22} {s['n']:>3} {s['mean_delta']:>+9.3f} {s['std_delta']:>7.3f} "
              f"{s['se_delta']:>7.3f} {ci_str:>20} {s['mean_wall']:>8.1f}")
    print("-" * 90)

    # Gap analysis vs H1_only
    ref = stats.get("H1_only", None)
    if ref and ref["n"] > 1:
        print(f"\nReference: H1_only mean Δ% = {ref['mean_delta']:+.3f}% (SE={ref['se_delta']:.3f})\n")
        print(f"{'config':<22} {'gap_Δ%':>9} {'SE(gap)':>9} {'z':>7} {'verdict':<28}")
        print("-" * 80)
        for label, s in order:
            if label == "H1_only" or s["n"] == 0:
                continue
            gap = s["mean_delta"] - ref["mean_delta"]
            se_gap = math.sqrt(s["se_delta"] ** 2 + ref["se_delta"] ** 2)
            z = gap / se_gap if se_gap > 0 else 0.0
            if z > 1.96:
                v = "DECISIVELY BETTER"
            elif z > 1.0:
                v = "weakly better"
            elif z > -1.0:
                v = "no different"
            elif z > -1.96:
                v = "weakly worse"
            else:
                v = "DECISIVELY WORSE"
            print(f"{label:<22} {gap:>+9.3f} {se_gap:>9.3f} {z:>+7.2f} {v:<28}")

    # Quality summary: aim for high mean Δ, low SE, low wall
    print("\nPareto-ish quality summary (higher mean Δ, lower SE, lower wall is better):")
    for label, s in order:
        if s["n"] == 0:
            continue
        score = s["mean_delta"] - 2 * s["se_delta"]  # conservative lower CI
        print(f"  {label:<22} score (mean - 2*SE) = {score:+.3f}% , wall={s['mean_wall']:.0f}s")


if __name__ == "__main__":
    main()
