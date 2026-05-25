"""
H5-v4: final attempt at extremely small lambda.

H5 confirmed broken at lambda in {0.1, 0.5, 1.0}, K in {16, 64, 128}, with
or without actor augmentation. All 15 runs collapsed to identical broken
states (eval ~1.63, Delta% ~ -17%).

Mechanism: kernel-implied Q* on uniform-state distribution ranges 0-35,
while real TD errors are ~1-5. Even lambda=0.1 lets dyna_loss dominate
real_critic_loss by 10-100x.

Final test: very small lambda where dyna contribution becomes a small
regularisation rather than a dominating signal.

Configs:
  H5_K64_l0p001    K=64, lambda=0.001 (1000x smaller than v3 minimum)
  H5_K64_l0p0001   K=64, lambda=0.0001

Seeds 11, 12, 13. 4096 ep. ~25 min at max_workers=3.
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


def D(K, lam, actor=1):
    return {
        "--use_dyna_augment": "1",
        "--dyna_n_synthetic": str(K),
        "--dyna_lambda": str(lam),
        "--dyna_actor_augment": str(actor),
    }


PHASE_CONFIGS = {
    "H5_K64_l0p001":    {**KERNEL_ON, **D(64, 0.001,  actor=1)},
    "H5_K64_l0p0001":   {**KERNEL_ON, **D(64, 0.0001, actor=1)},
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
    print(f"H5-v4 sweep: {len(runs)} runs at max_workers={args.max_workers}")

    csv_path = LOG_DIR / f"sweep_h5_v4_n{args.n_paths}.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []

    def execute(run: Dict) -> Dict:
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix="_h5v4")
        row.update({"tag": "_h5v4", "n_paths": str(run["n_paths"]), "contract": "focal"})
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
