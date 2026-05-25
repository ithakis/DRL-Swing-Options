"""
H8 (antithetic-pair targets) + H9 (jump-event IW) ablation.

All configs stack on H1 (the established winner). Sweep tests whether
either of these additional variance-reduction mechanisms helps further.

Configs (focal, 4096 ep, 3 seeds):
  H1_only             reference (kernel only, no H8, no H9)
  H8_only             kernel + antithetic-pair averaged target
  H9_only             kernel + jump-event IW (weight=3.0)
  H8_plus_H9          kernel + both (stacked)

Usage:
    python tools/sweep_h8_h9.py --max_workers 3 --seeds 11 12 13
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


KERNEL_ON = {
    "--use_expected_target": "1",
    "--kernel_M_x": "4",
    "--kernel_M_per_k": "4",
    "--kernel_N_max": "2",
    "--critic_warmup_episodes": "0",
}


PHASE_CONFIGS = {
    "H1_only":     {**KERNEL_ON, "--use_antithetic_target": "0", "--use_jump_iw": "0"},
    "H8_only":     {**KERNEL_ON, "--use_antithetic_target": "1", "--use_jump_iw": "0"},
    "H9_only":     {**KERNEL_ON, "--use_antithetic_target": "0", "--use_jump_iw": "1",
                    "--jump_iw_weight": "3.0"},
    "H8_plus_H9":  {**KERNEL_ON, "--use_antithetic_target": "1", "--use_jump_iw": "1",
                    "--jump_iw_weight": "3.0"},
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
    p.add_argument("--n_paths", type=int, default=4096)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    runs = make_runs(args.n_paths, args.seeds)
    print(f"H8+H9 sweep: {len(runs)} runs at max_workers={args.max_workers}")

    csv_path = LOG_DIR / f"sweep_h8h9_n{args.n_paths}.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []

    def execute(run: Dict) -> Dict:
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix="_h89")
        row.update({"tag": "_h89", "n_paths": str(run["n_paths"]), "contract": "focal"})
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

    print(f"\nResults -> {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
