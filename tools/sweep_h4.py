"""
H4 ablation sweep: warmstart on/off × kernel on/off, 4 configs × 3 seeds.

Configs (4096 episodes, focal c=0.04 gamma=2):
  CTRL:        baseline (no kernel, no warmstart)             [reference]
  H1_only:     kernel on, no warmstart                          [phase 1 winner]
  H4_only:     no kernel, warmstart on                          [does H4 alone help?]
  H1_plus_H4:  kernel on, warmstart on                          [stacking test]

Goal: determine whether H4 stacks with H1 (Phase 1 winner), replaces it,
or hurts.

Usage:
    python tools/sweep_h4.py --max_workers 3
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


# Best Phase-1 settings: kernel M=36, critic_warmup=0 (kernel itself stabilises).
KERNEL_ON = {
    "--use_expected_target": "1",
    "--kernel_M_x": "4",
    "--kernel_M_per_k": "4",
    "--kernel_N_max": "2",
    "--critic_warmup_episodes": "0",
}
KERNEL_OFF = {
    "--use_expected_target": "0",
}
WARMSTART_ON = {
    "--use_critic_warmstart": "1",
    "--warmstart_n_X": "25",
    "--warmstart_n_Y": "20",
    "--warmstart_n_actions": "11",
    "--warmstart_n_samples": "16384",
    "--warmstart_n_epochs": "50",
}
WARMSTART_OFF = {
    "--use_critic_warmstart": "0",
}


PHASE_CONFIGS = {
    "H4_CTRL":    {**KERNEL_OFF, **WARMSTART_OFF},
    "H4_H1only":  {**KERNEL_ON,  **WARMSTART_OFF},
    "H4_only":    {**KERNEL_OFF, **WARMSTART_ON},
    "H4_H1plus":  {**KERNEL_ON,  **WARMSTART_ON},
}


def make_runs(n_paths: int, seeds: List[int]) -> List[Dict]:
    runs = []
    for label, overrides in PHASE_CONFIGS.items():
        cfg = SweepConfig(label=label, overrides=overrides, note=label)
        for seed in seeds:
            r = dict(seed=seed, n_paths=n_paths)
            r["cfg"] = SweepConfig(label=label, overrides=dict(overrides),
                                    seed=seed, note=label)
            runs.append(r)
    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_workers", type=int, default=3)
    p.add_argument("--n_paths", type=int, default=4096)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    runs = make_runs(args.n_paths, args.seeds)
    print(f"H4 sweep: {len(runs)} runs at max_workers={args.max_workers}")

    csv_path = LOG_DIR / f"sweep_h4_n{args.n_paths}.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []

    def execute(run: Dict) -> Dict:
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix="_h4")
        row.update({
            "tag": "_h4",
            "n_paths": str(run["n_paths"]),
            "contract": "focal",
        })
        return row

    if args.dry_run:
        for r in runs:
            print(f"[DRY] {r['cfg'].label:<14} seed={r['cfg'].seed} n={r['n_paths']}")
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
