"""
Parse the sweep CSV + per-config log files into a comparison report.

Produces:
  * A printed leaderboard table sorted by eval price (descending).
  * The full Average100 trajectory per config for convergence visualisation.
  * Wall-clock + paths/sec comparison.
  * A JSON file with the structured data for downstream plotting.

Usage:
    python tools/analyze_sweep_h1.py [--csv logs/_sweep_h1/sweep_results_n4096.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "logs" / "_sweep_h1" / "sweep_results_n4096.csv"
LOG_DIR = ROOT / "logs" / "_sweep_h1"


PARSE_AVG100 = re.compile(r"Path\s+(\d+)/\d+.*?Average100\s*=\s*([\d.\-eE]+)")
PARSE_PATHS_SEC = re.compile(r"Paths/sec\s*=\s*([\d.\-eE]+)")


def load_log_trajectory(log_path: Path) -> Dict[str, List[float]]:
    if not log_path.exists():
        return {"path_idx": [], "avg100": [], "paths_sec": []}
    text = log_path.read_text(errors="replace")
    paths = []
    avgs = []
    pps = []
    for m in PARSE_AVG100.finditer(text):
        paths.append(int(m.group(1)))
        avgs.append(float(m.group(2)))
    for m in PARSE_PATHS_SEC.finditer(text):
        pps.append(float(m.group(1)))
    return {"path_idx": paths, "avg100": avgs, "paths_sec": pps}


def fmt(x: Optional[str], width: int = 8, prec: int = 4) -> str:
    if x is None or x == "":
        return "-" * width
    try:
        f = float(x)
        return f"{f:>{width}.{prec}f}"
    except Exception:
        return f"{str(x):>{width}}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    p.add_argument("--horizon", type=int, default=4096,
                   help="Training horizon used in the sweep (for picking convergence checkpoint).")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    rows: List[Dict[str, str]] = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print("No completed runs yet in the CSV.")
        return

    # Group by label (one entry per seed)
    by_label: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    # Load trajectories and compute summary stats
    summary = []
    for label, grp in by_label.items():
        log_paths = [LOG_DIR / f"{label}_s{r['seed']}.log" for r in grp]
        trajs = [load_log_trajectory(p) for p in log_paths]
        # average final 100 episodes' Average100 across seeds
        final_avgs = []
        avg100_at_q = {q: [] for q in (0.25, 0.50, 0.75, 1.00)}
        for tr in trajs:
            if tr["avg100"]:
                final_avgs.append(tr["avg100"][-1])
                for q in avg100_at_q:
                    idx = max(0, int(q * len(tr["avg100"])) - 1)
                    avg100_at_q[q].append(tr["avg100"][idx])
        eval_prices = []
        for r in grp:
            try:
                eval_prices.append(float(r["eval_price"]))
            except Exception:
                pass
        walls = [float(r["wall_seconds"]) for r in grp if r["wall_seconds"]]
        lsm_prices = [float(r["lsm_price"]) for r in grp if r["lsm_price"]]
        summary.append({
            "label": label,
            "n_seeds": len(grp),
            "status": ",".join(set(r["status"] for r in grp)),
            "eval_price_mean": (sum(eval_prices) / len(eval_prices)) if eval_prices else None,
            "eval_price_std": (statistics.stdev(eval_prices) if len(eval_prices) > 1 else 0.0)
                              if eval_prices else None,
            "lsm_price": (sum(lsm_prices) / len(lsm_prices)) if lsm_prices else None,
            "wall_seconds_mean": (sum(walls) / len(walls)) if walls else None,
            "final_avg100_mean": (sum(final_avgs) / len(final_avgs)) if final_avgs else None,
            **{f"avg100@{int(q*100)}pct": (sum(vs) / len(vs)) if vs else None
               for q, vs in avg100_at_q.items()},
            "note": grp[0]["note"],
            "overrides": grp[0]["overrides"],
        })

    # Sort by eval price descending (higher = better, since RL aims to beat LSM)
    summary.sort(key=lambda s: -(s["eval_price_mean"] or -1e9))

    # Print leaderboard
    print(f"\n=== Sweep leaderboard ({len(summary)} configs, horizon = {args.horizon} ep) ===")
    print(f"{'label':<22} {'eval':>9} {'avg100':>9} "
          f"{'@25%':>8} {'@50%':>8} {'@75%':>8} {'@100%':>8} "
          f"{'wall_s':>8} {'note':<60}")
    print("-" * 160)
    for s in summary:
        note = s["note"][:58] if s["note"] else ""
        print(f"{s['label']:<22} {fmt(s['eval_price_mean'])} {fmt(s['final_avg100_mean'])} "
              f"{fmt(s['avg100@25pct'])} {fmt(s['avg100@50pct'])} "
              f"{fmt(s['avg100@75pct'])} {fmt(s['avg100@100pct'])} "
              f"{fmt(s['wall_seconds_mean'], 8, 1)} {note}")
    print("-" * 160)

    # LSM baseline reference
    lsm_vals = [s["lsm_price"] for s in summary if s["lsm_price"] is not None]
    if lsm_vals:
        lsm_mean = sum(lsm_vals) / len(lsm_vals)
        print(f"\nLSM benchmark price (mean across runs): {lsm_mean:.4f}")
        print(f"Best RL Delta%: {((max(s['eval_price_mean'] for s in summary if s['eval_price_mean']) / lsm_mean) - 1) * 100:+.2f}%")

    # Save structured json for downstream use
    out_json = csv_path.with_suffix(".analysis.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nStructured summary written to {out_json}")


if __name__ == "__main__":
    main()
