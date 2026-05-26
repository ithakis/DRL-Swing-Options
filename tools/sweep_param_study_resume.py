"""Resume the param study: the 16 runs that didn't complete + 1 that failed.

Missing:
  - K_M21 seed 13 (failed: cache race)
  - K_M150 seeds 11, 12, 13 (killed mid-flight)
  - P_n1024 seeds 11, 12, 13
  - P_n2048 seeds 11, 12, 13
  - H1_seeds45 seeds 14, 15
  - H8_nostrat_seeds45 seeds 14, 15
  - H8_strat_seeds45 seeds 14, 15
"""

from __future__ import annotations
import argparse, concurrent.futures as cf, csv, sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sweep_expected_target import SweepConfig, base_args, run_one, LOG_DIR


def kov(Mx, Mpk, Nmax):
    return {
        "--use_expected_target": "1", "--critic_warmup_episodes": "0",
        "--kernel_M_x": str(Mx), "--kernel_M_per_k": str(Mpk), "--kernel_N_max": str(Nmax),
    }


# Each entry: (label, overrides, seed, n_paths)
RUNS_SPEC = []
# K_M21 s13 retry
RUNS_SPEC.append(("K_M21", kov(3, 3, 2), 13, 4096))
# K_M150 all three
for s in (11, 12, 13):
    RUNS_SPEC.append(("K_M150", kov(6, 8, 3), s, 4096))
# Path counts
for n_p in (1024, 2048):
    for s in (11, 12, 13):
        RUNS_SPEC.append((f"P_n{n_p}", kov(4, 4, 2), s, n_p))
# Antithetic 5-seed extension
for s in (14, 15):
    RUNS_SPEC.append(("H1_seeds45",
                       {**kov(4, 4, 2)}, s, 4096))
    RUNS_SPEC.append(("H8_nostrat_seeds45",
                       {**kov(4, 4, 2),
                        "--use_antithetic_target": "1",
                        "--antithetic_preserve_stratify": "0"}, s, 4096))
    RUNS_SPEC.append(("H8_strat_seeds45",
                       {**kov(4, 4, 2),
                        "--use_antithetic_target": "1",
                        "--antithetic_preserve_stratify": "1"}, s, 4096))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_workers", type=int, default=3)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    runs = []
    for label, ovr, seed, n_paths in RUNS_SPEC:
        cfg = SweepConfig(label=label, overrides=dict(ovr), seed=seed, note=label)
        runs.append({"cfg": cfg, "n_paths": n_paths})
    print(f"Param-study resume: {len(runs)} runs at max_workers={args.max_workers}")
    csv_path = LOG_DIR / "sweep_param_study_resume.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []
    def execute(run):
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix="_pa")
        row.update({"tag": "_pa", "n_paths": str(run["n_paths"]), "contract": "focal"})
        return row
    if args.dry_run:
        for r in runs:
            print(f"[DRY] {r['cfg'].label:<22} seed={r['cfg'].seed:>3} n={r['n_paths']}")
        return
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(execute, r) for r in runs]
        for fut in cf.as_completed(futs):
            try:
                row = fut.result(); rows.append(row)
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
                    for r in rows: w.writerow({k: r.get(k, "") for k in fieldnames})
            except Exception as e:
                print(f"FAIL: {e}", file=sys.stderr)
    print(f"\nResults -> {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
