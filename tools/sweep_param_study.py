"""
Comprehensive parameter study covering three axes (focal regime):

  A) KERNEL ACCURACY: how big does the quadrature grid need to be?
     Sweeps M = M_x x (1 + N_max * M_per_k) over 5 settings:
       (M_x=2, M_per_k=2, N_max=1) -> M = 6     -- minimum
       (M_x=3, M_per_k=3, N_max=2) -> M = 21    -- low
       (M_x=4, M_per_k=4, N_max=2) -> M = 36    -- current default
       (M_x=6, M_per_k=4, N_max=3) -> M = 78    -- moderate
       (M_x=6, M_per_k=8, N_max=3) -> M = 150   -- high

  B) TRAINING PATHS: how many training paths drive the empirical PV?
     Sweeps n_paths in {1024, 2048, 4096, 8192} at the default kernel.

  C) ANTITHETIC HEAD-TO-HEAD (2 extra seeds each):
     H1_only, H8_nostrat, H8_strat, all at seeds {14, 15} to extend
     the prior 3-seed data to 5 seeds.

3 seeds (11, 12, 13) for A and B; 2 new seeds (14, 15) for C.

Usage:
    python tools/sweep_param_study.py --max_workers 3
"""

from __future__ import annotations

import argparse, concurrent.futures as cf, csv, sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sweep_expected_target import SweepConfig, base_args, run_one, LOG_DIR


KERNEL_ON_BASE = {
    "--use_expected_target": "1",
    "--critic_warmup_episodes": "0",
}


def kernel_overrides(M_x: int, M_per_k: int, N_max: int) -> Dict[str, str]:
    return {
        **KERNEL_ON_BASE,
        "--kernel_M_x": str(M_x),
        "--kernel_M_per_k": str(M_per_k),
        "--kernel_N_max": str(N_max),
    }


# === Axis A: kernel accuracy at n_paths=4096 ===
KERNEL_CONFIGS = [
    ("K_M6",   2,  2, 1),
    ("K_M21",  3,  3, 2),
    ("K_M36",  4,  4, 2),     # current default - same as H1_only
    ("K_M78",  6,  4, 3),
    ("K_M150", 6,  8, 3),
]

# === Axis B: training paths at default kernel ===
PATH_CONFIGS = [1024, 2048]   # 4096 already in prior data; 8192 too

# === Axis C: antithetic head-to-head, 2 new seeds ===
ANTI_CONFIGS = {
    "H1_seeds45":           {"--kernel_M_x": "4", "--kernel_M_per_k": "4", "--kernel_N_max": "2",
                             **KERNEL_ON_BASE},
    "H8_nostrat_seeds45":   {"--kernel_M_x": "4", "--kernel_M_per_k": "4", "--kernel_N_max": "2",
                             **KERNEL_ON_BASE,
                             "--use_antithetic_target": "1",
                             "--antithetic_preserve_stratify": "0"},
    "H8_strat_seeds45":     {"--kernel_M_x": "4", "--kernel_M_per_k": "4", "--kernel_N_max": "2",
                             **KERNEL_ON_BASE,
                             "--use_antithetic_target": "1",
                             "--antithetic_preserve_stratify": "1"},
}


def make_runs(seeds_abc: List[int], seeds_c: List[int]) -> List[Dict]:
    runs = []
    # Axis A: kernel size
    for label, Mx, Mpk, Nmax in KERNEL_CONFIGS:
        if label == "K_M36":
            continue  # we already have 3-seed H1 data at M=36, n_paths=4096
        for seed in seeds_abc:
            ovr = kernel_overrides(Mx, Mpk, Nmax)
            cfg = SweepConfig(label=label, overrides=dict(ovr), seed=seed, note=label)
            runs.append({"cfg": cfg, "n_paths": 4096, "tag": "_pa"})
    # Axis B: path count
    for nP in PATH_CONFIGS:
        for seed in seeds_abc:
            ovr = kernel_overrides(4, 4, 2)
            cfg = SweepConfig(label=f"P_n{nP}", overrides=dict(ovr), seed=seed, note=f"n_paths={nP}")
            runs.append({"cfg": cfg, "n_paths": nP, "tag": "_pa"})
    # Axis C: antithetic seeds 14, 15
    for label, ovr in ANTI_CONFIGS.items():
        for seed in seeds_c:
            cfg = SweepConfig(label=label, overrides=dict(ovr), seed=seed, note=label)
            runs.append({"cfg": cfg, "n_paths": 4096, "tag": "_pa"})
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_workers", type=int, default=3)
    p.add_argument("--seeds_abc", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--seeds_c", type=int, nargs="+", default=[14, 15])
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    runs = make_runs(args.seeds_abc, args.seeds_c)
    print(f"Parametric study: {len(runs)} runs at max_workers={args.max_workers}")
    csv_path = LOG_DIR / "sweep_param_study.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []

    def execute(run):
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix=run["tag"])
        row.update({"tag": run["tag"], "n_paths": str(run["n_paths"]), "contract": "focal"})
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
