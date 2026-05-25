"""
H5 (Dyna-style augmentation) ablation sweep.

Configs (all on top of H1 kernel except CTRL):
  CTRL                  baseline (no kernel, no Dyna)        -- reference
  H1_only               kernel only                          -- Phase 1 winner
  H5_K64_l1             H1 + Dyna K=64 lambda=1.0
  H5_K128_l1            H1 + Dyna K=128 lambda=1.0
  H5_K256_l1            H1 + Dyna K=256 lambda=1.0
  H5_K128_l0p5          H1 + Dyna K=128 lambda=0.5
  H5_K128_no_actor      H1 + Dyna K=128 lambda=1.0, no actor augmentation
  H5_alone_K128         Dyna K=128 lambda=1.0 *without* H1 (pure Dyna check)

Seeds: 11, 12, 13 (3 seeds for initial pass).
Horizon: 4096 episodes.

Statistical reporting via analyze_h5.py (gap-to-H1 + conservative score).

Usage:
    python tools/sweep_h5.py --max_workers 3 --seeds 11 12 13
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
KERNEL_OFF = {"--use_expected_target": "0", "--critic_warmup_episodes": "0"}
DYNA_OFF = {"--use_dyna_augment": "0"}


def D(K, lam, actor=1):
    return {
        "--use_dyna_augment": "1",
        "--dyna_n_synthetic": str(K),
        "--dyna_lambda": str(lam),
        "--dyna_actor_augment": str(actor),
    }


PHASE_CONFIGS = {
    "CTRL":             {**KERNEL_OFF, **DYNA_OFF},
    "H1_only":          {**KERNEL_ON,  **DYNA_OFF},
    "H5_K64_l1":        {**KERNEL_ON,  **D(64,  1.0)},
    "H5_K128_l1":       {**KERNEL_ON,  **D(128, 1.0)},
    "H5_K256_l1":       {**KERNEL_ON,  **D(256, 1.0)},
    "H5_K128_l0p5":     {**KERNEL_ON,  **D(128, 0.5)},
    "H5_K128_no_actor": {**KERNEL_ON,  **D(128, 1.0, actor=0)},
    "H5_alone_K128":    {**KERNEL_OFF, **D(128, 1.0)},
}


def make_runs(n_paths: int, seeds: List[int], only=None) -> List[Dict]:
    runs = []
    items = PHASE_CONFIGS.items() if only is None else [
        (k, PHASE_CONFIGS[k]) for k in only if k in PHASE_CONFIGS
    ]
    for label, overrides in items:
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
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated whitelist of config labels.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    only = args.only.split(",") if args.only else None
    runs = make_runs(args.n_paths, args.seeds, only=only)
    print(f"H5 sweep: {len(runs)} runs at max_workers={args.max_workers}")

    csv_path = LOG_DIR / f"sweep_h5_n{args.n_paths}.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []

    def execute(run: Dict) -> Dict:
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix="_h5")
        row.update({
            "tag": "_h5",
            "n_paths": str(run["n_paths"]),
            "contract": "focal",
        })
        return row

    if args.dry_run:
        for r in runs:
            print(f"[DRY] {r['cfg'].label:<22} seed={r['cfg'].seed}")
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
