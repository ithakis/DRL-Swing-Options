"""Resume the H7+H8+H9 sweep after computer shutdown: run only H9 and H8_plus_H9."""

from __future__ import annotations
import argparse, concurrent.futures as cf, csv, sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sweep_expected_target import SweepConfig, base_args, run_one, LOG_DIR

KERNEL_ON = {
    "--use_expected_target": "1",
    "--kernel_M_x": "4", "--kernel_M_per_k": "4", "--kernel_N_max": "2",
    "--critic_warmup_episodes": "0",
}

PHASE_CONFIGS = {
    "H9":         {**KERNEL_ON, "--use_twin_critic": "0",
                   "--use_antithetic_target": "0", "--use_jump_iw": "1",
                   "--jump_iw_weight": "3.0"},
    "H8_plus_H9": {**KERNEL_ON, "--use_twin_critic": "0",
                   "--use_antithetic_target": "1", "--use_jump_iw": "1",
                   "--jump_iw_weight": "3.0"},
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_workers", type=int, default=3)
    p.add_argument("--n_paths", type=int, default=4096)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    args = p.parse_args()
    runs = []
    for label, overrides in PHASE_CONFIGS.items():
        for seed in args.seeds:
            cfg = SweepConfig(label=label, overrides=dict(overrides), seed=seed, note=label)
            runs.append({"cfg": cfg, "n_paths": args.n_paths})
    print(f"H789 resume: {len(runs)} runs at max_workers={args.max_workers}")
    csv_path = LOG_DIR / f"sweep_h789_resume_n{args.n_paths}.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []
    def execute(run):
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract="focal")
        row = run_one(cfg, base, False, tag_suffix="_h789r")
        row.update({"tag": "_h789r", "n_paths": str(run["n_paths"]), "contract": "focal"})
        return row
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
