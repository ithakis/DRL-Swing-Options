"""Denser kernel-size sweep for the semi-analytical bootstrap study.

The original parameter study only sampled a few anchor points in kernel size.
This sweep adds a denser set of unique M values between 6 and 36 while keeping
the sweep on the actual quadrature composition parameters (M_x, M_per_k, N_max)
rather than treating total M as the primitive control knob.

Usage:
    python tools/sweep_kernel_density.py --max_workers 3 --seeds 11 12 13
    python tools/sweep_kernel_density.py --max_workers 3 --seeds 11 12 13 14 15 --include_high_anchors
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.sweep_expected_target import LOG_DIR, SweepConfig, base_args, run_one

KERNEL_ON_BASE = {
    "--use_expected_target": "1",
    "--critic_warmup_episodes": "0",
}


def kernel_overrides(m_x: int, m_per_k: int, n_max: int) -> Dict[str, str]:
    return {
        **KERNEL_ON_BASE,
        "--kernel_M_x": str(m_x),
        "--kernel_M_per_k": str(m_per_k),
        "--kernel_N_max": str(n_max),
    }


# Dense low-to-mid grid. Each label is unique in total M so the notebook plot
# can map directly from label -> M while still retaining the structural params.
DENSE_CONFIGS: List[Tuple[str, int, int, int]] = [
    ("K_M6", 2, 2, 1),
    ("K_M8", 2, 3, 1),
    ("K_M9", 3, 2, 1),
    ("K_M10", 2, 4, 1),
    ("K_M12", 3, 3, 1),
    ("K_M15", 3, 2, 2),
    ("K_M18", 3, 5, 1),
    ("K_M20", 4, 2, 2),
    ("K_M21", 3, 3, 2),
    ("K_M24", 6, 3, 1),
    ("K_M25", 5, 2, 2),
    ("K_M28", 4, 3, 2),
    ("K_M30", 6, 2, 2),
    ("K_M35", 5, 3, 2),
    ("K_M36", 4, 4, 2),
]

HIGH_ANCHORS: List[Tuple[str, int, int, int]] = [
    ("K_M78", 6, 4, 3),
    ("K_M150", 6, 8, 3),
]


def make_runs(n_paths: int, seeds: List[int], include_high_anchors: bool) -> List[Dict[str, object]]:
    configs = list(DENSE_CONFIGS)
    if include_high_anchors:
        configs.extend(HIGH_ANCHORS)
    runs: List[Dict[str, object]] = []
    for label, m_x, m_per_k, n_max in configs:
        overrides = kernel_overrides(m_x, m_per_k, n_max)
        note = f"M={m_x * (1 + n_max * m_per_k)} via (Mx={m_x}, Mpk={m_per_k}, Nmax={n_max})"
        for seed in seeds:
            cfg = SweepConfig(label=label, overrides=dict(overrides), seed=seed, note=note)
            runs.append({"cfg": cfg, "n_paths": n_paths})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--n_paths", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--include_high_anchors", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    runs = make_runs(args.n_paths, args.seeds, include_high_anchors=args.include_high_anchors)
    print(f"Kernel density sweep: {len(runs)} runs at max_workers={args.max_workers}")

    csv_path = LOG_DIR / f"sweep_kernel_density_n{args.n_paths}.csv"
    fieldnames = [
        "tag",
        "label",
        "seed",
        "n_paths",
        "contract",
        "status",
        "eval_price",
        "eval_price_se",
        "lsm_price",
        "final_avg100",
        "final_paths_per_sec",
        "training_time",
        "wall_seconds",
        "note",
        "overrides",
    ]
    rows: List[Dict[str, str]] = []

    def execute(run: Dict[str, object]) -> Dict[str, str]:
        cfg = run["cfg"]
        assert isinstance(cfg, SweepConfig)
        base = base_args(n_paths=int(run["n_paths"]), contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix="_kd")
        row.update({"tag": "_kd", "n_paths": str(run["n_paths"]), "contract": "focal"})
        return row

    if args.dry_run:
        for run in runs:
            cfg = run["cfg"]
            assert isinstance(cfg, SweepConfig)
            print(f"[DRY] {cfg.label:<8} seed={cfg.seed:>3} n={run['n_paths']} note={cfg.note}")
        return

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(execute, run) for run in runs]
        for future in cf.as_completed(futures):
            try:
                row = future.result()
                rows.append(row)
                with open(csv_path, "w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for written_row in rows:
                        writer.writerow({key: written_row.get(key, "") for key in fieldnames})
            except Exception as exc:
                print(f"FAIL: {exc}", file=sys.stderr)

    print(f"\nResults -> {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
