"""
H6: kernel-expected IQN target sweep.

Tests whether the distributional critic + kernel-expected quantile target
beats H1 alone.  IQN's per-quantile gradient may extract richer signal
from each transition than the scalar TD target H1 uses.

Configs (focal, 2048 ep -- shorter than other sweeps because IQN+kernel
is the most expensive setup):
  CTRL_proper       baseline (no kernel, no IQN, warmup=256)
  H1_only           kernel only (Phase 1 winner)
  IQN_only          IQN on, no kernel, warmup=256
  H6                IQN on + kernel-expected quantile target, warmup=0

Seeds: 11, 12, 13.

IQN_N=8 to make wall-clock tractable on M1.

Usage:
    python tools/sweep_h6.py --max_workers 3
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
    SweepConfig, base_args, run_one, LOG_DIR,
)


KERNEL_ON_NO_WARMUP = {
    "--use_expected_target": "1",
    "--kernel_M_x": "4",
    "--kernel_M_per_k": "4",
    "--kernel_N_max": "2",
    "--critic_warmup_episodes": "0",
}
KERNEL_OFF = {"--use_expected_target": "0"}
IQN_ON = {"-iqn": "1", "--iqn_N": "8"}
IQN_OFF = {"-iqn": "0"}


PHASE_CONFIGS = {
    "CTRL_proper":   {**KERNEL_OFF, **IQN_OFF},
    "H1_only":       {**KERNEL_ON_NO_WARMUP, **IQN_OFF},
    "IQN_only":      {**KERNEL_OFF, **IQN_ON},
    "H6":            {**KERNEL_ON_NO_WARMUP, **IQN_ON},
}


def make_runs(n_paths: int, seeds: List[int]) -> List[Dict]:
    runs = []
    for label, overrides in PHASE_CONFIGS.items():
        for seed in seeds:
            cfg = SweepConfig(label=label, overrides=dict(overrides),
                               seed=seed, note=label)
            runs.append({"cfg": cfg, "n_paths": n_paths})
    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_workers", type=int, default=3)
    p.add_argument("--n_paths", type=int, default=2048,
                   help="Training horizon (default 2048 -- shorter than other "
                   "sweeps because IQN+kernel is expensive).")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    runs = make_runs(args.n_paths, args.seeds)
    print(f"H6 sweep: {len(runs)} runs at max_workers={args.max_workers}")

    csv_path = LOG_DIR / f"sweep_h6_n{args.n_paths}.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []

    def execute(run: Dict) -> Dict:
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix="_h6")
        row.update({"tag": "_h6", "n_paths": str(run["n_paths"]), "contract": "focal"})
        return row

    if args.dry_run:
        for r in runs:
            print(f"[DRY] {r['cfg'].label:<18} seed={r['cfg'].seed}")
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
