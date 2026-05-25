"""
H4 refinement sweep (v2): 6 configs x 5 seeds, statistically rigorous.

The Phase-1 H4 sweep showed that warm-start hurts (worse mean AND 2-10x
seed-std), most plausibly because the V grid is biased ~3% high and the
supervised pre-training drags the critic toward inflated Q targets.

Two refinements to investigate:
  1. Fewer supervised epochs (5 or 10 instead of 50) -- less overfit.
  2. Do NOT copy critic_local to critic_target after supervised step;
     TD updates pull the target back to truth.

Configs:
  CTRL                 baseline (no kernel, no warmstart)
  H1_only              Phase 1 winner (kernel only)
  H4v2_ep10            kernel + warmstart 10 ep + target copy
  H4v2_ep5             kernel + warmstart 5 ep + target copy
  H4v2_no_tgt_ep10     kernel + warmstart 10 ep, no target copy
  H4v2_no_tgt_ep50     kernel + warmstart 50 ep, no target copy

Seeds: 11..15 (5 seeds).
Horizon: 4096 episodes.

Statistical reporting: mean Δ%, std Δ%, SE of mean, and a per-config
SE-of-the-gap-to-H1_only.

Usage:
    python tools/sweep_h4_v2.py --max_workers 3 --seeds 11 12 13 14 15
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


KERNEL_ON_BASE = {
    "--use_expected_target": "1",
    "--kernel_M_x": "4",
    "--kernel_M_per_k": "4",
    "--kernel_N_max": "2",
    "--critic_warmup_episodes": "0",
}
KERNEL_OFF = {
    "--use_expected_target": "0",
}
WARMSTART_BASE = {
    "--use_critic_warmstart": "1",
    "--warmstart_n_X": "25",
    "--warmstart_n_Y": "20",
    "--warmstart_n_actions": "11",
    "--warmstart_n_samples": "16384",
    "--warmstart_batch_size": "256",
}


PHASE_CONFIGS = {
    "CTRL":             {**KERNEL_OFF, "--use_critic_warmstart": "0"},
    "H1_only":          {**KERNEL_ON_BASE, "--use_critic_warmstart": "0"},
    "H4v2_ep10":        {**KERNEL_ON_BASE, **WARMSTART_BASE,
                         "--warmstart_n_epochs": "10",
                         "--warmstart_copy_target": "1"},
    "H4v2_ep5":         {**KERNEL_ON_BASE, **WARMSTART_BASE,
                         "--warmstart_n_epochs": "5",
                         "--warmstart_copy_target": "1"},
    "H4v2_no_tgt_ep10": {**KERNEL_ON_BASE, **WARMSTART_BASE,
                         "--warmstart_n_epochs": "10",
                         "--warmstart_copy_target": "0"},
    "H4v2_no_tgt_ep50": {**KERNEL_ON_BASE, **WARMSTART_BASE,
                         "--warmstart_n_epochs": "50",
                         "--warmstart_copy_target": "0"},
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
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    runs = make_runs(args.n_paths, args.seeds)
    print(f"H4-v2 sweep: {len(runs)} runs at max_workers={args.max_workers}")

    csv_path = LOG_DIR / f"sweep_h4_v2_n{args.n_paths}.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []

    def execute(run: Dict) -> Dict:
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix="_h4v2")
        row.update({
            "tag": "_h4v2",
            "n_paths": str(run["n_paths"]),
            "contract": "focal",
        })
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
