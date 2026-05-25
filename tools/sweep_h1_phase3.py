"""
Phase-3 sweep: medium-horizon validation at 8192 episodes.

At 4096 episodes the H1 kernel beats baseline by ~2.5 pp on focal and ~5 pp
on no-cost.  Phase 3 doubles the training horizon to verify the advantage
persists (or grows) at the medium horizon, which is roughly halfway to the
published 32768-ep focal study.

Runs (6 total at max_workers=3):
  K36_no_warmup x seeds {11, 12, 13}  -- the winning config
  B0_baseline x seeds {11, 12, 13}    -- control

Usage:
    python tools/sweep_h1_phase3.py --max_workers 3
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.sweep_expected_target import (  # noqa: E402
    SweepConfig, base_args, run_one, make_configs, _apply_contract, LOG_DIR,
)


def make_phase3_runs(n_paths: int) -> List[Dict]:
    base_cfgs = {c.label: c for c in make_configs()}
    runs: List[Dict] = []
    for label in ("K36_no_warmup", "B0_baseline"):
        for seed in (11, 12, 13):
            runs.append({
                "cfg": base_cfgs[label], "seed": seed,
                "n_paths": n_paths, "contract": "focal", "tag": "_p3",
            })
    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_workers", type=int, default=3)
    p.add_argument("--n_paths", type=int, default=8192,
                   help="Training horizon (default 8192, ~halfway to focal 32k).")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    runs = make_phase3_runs(args.n_paths)
    print(f"Phase-3 sweep: {len(runs)} runs at max_workers={args.max_workers}, n={args.n_paths}")

    csv_path = LOG_DIR / f"sweep_h1_phase3_n{args.n_paths}.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []

    def execute(run: Dict) -> Dict:
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract=run["contract"])
        base = _apply_contract(base, run["contract"])
        # n_paths-aware schedule adjustments: lr_schedule, warmup, etc.
        # The phase-2 base_args already scales these for shorter horizons; for
        # 8192 we want LR to still be decaying late, not flat.
        base["--lr_schedule_episodes"] = str(max(args.n_paths + 2000, 5000))
        base["--per_beta_frames"] = str(max(args.n_paths * 30, 100000))
        base["--target_noise_decay_start"] = str(int(args.n_paths * 0.85))
        tagged = SweepConfig(label=cfg.label, overrides=dict(cfg.overrides),
                              seed=run["seed"], note=cfg.note)
        row = run_one(tagged, base, args.dry_run, tag_suffix=run["tag"])
        row.update({
            "tag": run["tag"],
            "n_paths": str(run["n_paths"]),
            "contract": run["contract"],
        })
        return row

    if args.dry_run:
        for r in runs:
            print(f"[DRY] {r['cfg'].label} seed={r['seed']} n={r['n_paths']}")
        return

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(execute, r) for r in runs]
        for fut in cf.as_completed(futs):
            try:
                row = fut.result()
                rows.append(row)
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for r in rows:
                        w.writerow({k: r.get(k, "") for k in fieldnames})
            except Exception as e:
                print(f"FUTURE FAILED: {e}", file=sys.stderr)

    print(f"\nResults written to {csv_path} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
