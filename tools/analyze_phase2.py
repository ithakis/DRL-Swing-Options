"""
Combined analysis across all H1 sweep stages: wide1, wide2, phase2.

Computes per-seed Delta% vs the seed's own LSM benchmark (LSM varies across
seeds because path stratification is seeded), then groups by (label, n_paths,
contract) and reports mean +- std of Delta% and eval price.

Usage:
    python tools/analyze_phase2.py
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "_sweep_h1"

# Source CSVs
SOURCES = [
    ("wide1", LOG_DIR / "sweep_results_n4096.csv", "focal", 4096),
    ("wide2", LOG_DIR / "sweep_results_n3072_wide2.csv", "focal", 3072),
    ("phase2", LOG_DIR / "sweep_h1_phase2.csv", None, None),  # contract/n_paths in row
]


def load_all_rows() -> List[Dict]:
    rows: List[Dict] = []
    for source, path, default_contract, default_n in SOURCES:
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                r["source"] = source
                if "contract" not in r or not r["contract"]:
                    r["contract"] = default_contract or ""
                if "n_paths" not in r or not r["n_paths"]:
                    r["n_paths"] = str(default_n) if default_n else ""
                rows.append(r)
    return rows


def compute_delta(r: Dict) -> Tuple[float, float, float] | None:
    try:
        eval_p = float(r["eval_price"])
        lsm = float(r["lsm_price"])
        if lsm <= 0:
            return None
        return eval_p, lsm, (eval_p / lsm - 1.0) * 100.0
    except Exception:
        return None


def main() -> None:
    rows = load_all_rows()
    print(f"Loaded {len(rows)} ok runs across {len(SOURCES)} sources")

    # Group by (contract, n_paths, label)
    groups: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("contract", "focal"), r.get("n_paths", ""), r["label"])
        groups[key].append(r)

    # Per-group summary
    print("\n=== Per-config Delta% (mean ± std across seeds) ===")
    print(f"{'config':<22} {'cont':<6} {'ep':<6} {'n_seeds':>7} "
          f"{'eval_mean':>10} {'lsm_mean':>10} {'Delta_mean%':>12} {'Delta_std%':>11} {'wall_s_mean':>12}")
    print("-" * 110)

    summary_rows = []
    for (contract, n_paths, label), grp in sorted(groups.items()):
        deltas: List[float] = []
        evals: List[float] = []
        lsms: List[float] = []
        walls: List[float] = []
        seeds: List[str] = []
        for r in grp:
            cd = compute_delta(r)
            if cd is None:
                continue
            ep, lsm, d = cd
            deltas.append(d)
            evals.append(ep)
            lsms.append(lsm)
            seeds.append(r.get("seed", ""))
            try:
                walls.append(float(r.get("wall_seconds", "0")))
            except Exception:
                pass
        if not deltas:
            continue
        summary_rows.append({
            "label": label,
            "contract": contract,
            "n_paths": n_paths,
            "n_seeds": len(deltas),
            "seeds": ",".join(seeds),
            "eval_mean": sum(evals) / len(evals),
            "lsm_mean": sum(lsms) / len(lsms),
            "delta_mean": sum(deltas) / len(deltas),
            "delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "wall_mean": (sum(walls) / len(walls)) if walls else float("nan"),
        })

    # Sort within each (contract, n_paths) by Delta_mean descending
    summary_rows.sort(key=lambda s: (s["contract"], s["n_paths"], -s["delta_mean"]))

    for s in summary_rows:
        print(f"{s['label']:<22} {s['contract']:<6} {s['n_paths']:<6} {s['n_seeds']:>7d} "
              f"{s['eval_mean']:>10.4f} {s['lsm_mean']:>10.4f} {s['delta_mean']:>+12.3f}  "
              f"{s['delta_std']:>10.3f}  {s['wall_mean']:>12.1f}")

    # Headline comparison: best kernel vs baseline at 4096 focal
    print("\n=== Headline: 4096-ep focal, top kernel vs baseline (per-seed) ===")
    target = [s for s in summary_rows if s["contract"] == "focal" and s["n_paths"] == "4096"]
    target.sort(key=lambda s: -s["delta_mean"])
    if target:
        best = target[0]
        baseline = next((s for s in target if s["label"] == "B0_baseline"), None)
        if baseline:
            diff_mean = best["delta_mean"] - baseline["delta_mean"]
            # SE of difference assuming independent seeds
            se = ((best["delta_std"] ** 2) + (baseline["delta_std"] ** 2)) ** 0.5
            se /= max(min(best["n_seeds"], baseline["n_seeds"]) ** 0.5, 1.0)
            print(f"Best kernel: {best['label']}: Delta% = {best['delta_mean']:+.3f}% (n={best['n_seeds']})")
            print(f"Baseline:    {baseline['label']}: Delta% = {baseline['delta_mean']:+.3f}% (n={baseline['n_seeds']})")
            print(f"Gap mean: {diff_mean:+.3f} pp,  SE ~ {se:.3f} pp,  z ~ {diff_mean / max(se, 1e-9):.1f}")

    # No-cost regression check
    print("\n=== No-cost regression check ===")
    nocost = [s for s in summary_rows if s["contract"] == "nocost"]
    nocost.sort(key=lambda s: -s["delta_mean"])
    for s in nocost:
        print(f"  {s['label']:<22} Delta = {s['delta_mean']:+.3f}%  (eval={s['eval_mean']:.4f}, "
              f"lsm={s['lsm_mean']:.4f}, seeds={s['seeds']})")

    # Save structured json
    out = LOG_DIR / "phase2_analysis.json"
    with open(out, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nStructured summary -> {out}")


if __name__ == "__main__":
    main()
