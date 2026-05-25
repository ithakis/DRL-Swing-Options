"""
Phase-2 sweep launcher: gap-fill + multi-seed validation + no-cost regression.

Batches all remaining runs needed to draw conclusions about H1 (expected
critic target) into a single orchestrator queue. Re-uses run_one() and
base_args() from sweep_expected_target.py.

Run sequence (max_workers=3 by default):
  Stage A (gap-fill at 3072 ep): K36_bs64 (missed); B0_baseline at 3072 ep
    so we have an apples-to-apples Delta% reference for wide2 results.
  Stage B (multi-seed validation at 4096 ep, seeds 11/12/13):
    K36_default, K36_no_warmup, K36_no_noise, B0_baseline.
  Stage C (no-cost regression at 4096 ep, seed 11):
    K36_default, B0_baseline.

Usage:
    python tools/sweep_h1_phase2.py --max_workers 3
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.sweep_expected_target import (  # noqa: E402
    SweepConfig, base_args, run_one, make_configs, _apply_contract, LOG_DIR,
)


def make_phase2_runs() -> List[Dict]:
    """Return a list of dicts {cfg, seed, n_paths, contract, tag} for each run."""
    base_cfgs = {c.label: c for c in make_configs()}
    runs: List[Dict] = []

    # --- Stage A: gap-fill at 3072 episodes -----------------------------------
    # We re-run B0_baseline at 3072 so the wide2 Delta% comparison is consistent.
    for label in ("B0_baseline", "K36_bs64"):
        runs.append({
            "cfg": base_cfgs[label], "seed": 11,
            "n_paths": 3072, "contract": "focal", "tag": "_p2A",
        })

    # --- Stage B: multi-seed validation at 4096 episodes ----------------------
    for label in ("K36_default", "K36_no_warmup", "K36_no_noise", "B0_baseline"):
        for seed in (11, 12, 13):
            runs.append({
                "cfg": base_cfgs[label], "seed": seed,
                "n_paths": 4096, "contract": "focal", "tag": "_p2B",
            })

    # --- Stage C: no-cost regression at 4096 episodes -------------------------
    for label in ("K36_default", "B0_baseline"):
        runs.append({
            "cfg": base_cfgs[label], "seed": 11,
            "n_paths": 4096, "contract": "nocost", "tag": "_p2C",
        })

    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_workers", type=int, default=3,
                   help="Parallel cap (default 3 for free M1; raise to 4 if memory permits).")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_stage", type=str, default="",
                   help="Comma list of stage tags to skip (e.g. _p2A).")
    args = p.parse_args()

    runs = make_phase2_runs()
    skips = set(s.strip() for s in args.skip_stage.split(",") if s.strip())
    runs = [r for r in runs if r["tag"] not in skips]
    print(f"Phase-2 sweep: {len(runs)} runs at max_workers={args.max_workers}")

    csv_path = LOG_DIR / "sweep_h1_phase2.csv"
    fieldnames = ["tag", "label", "seed", "n_paths", "contract", "status",
                  "eval_price", "eval_price_se", "lsm_price",
                  "final_avg100", "final_paths_per_sec", "training_time",
                  "wall_seconds", "note", "overrides"]
    rows: List[Dict] = []

    def execute(run: Dict) -> Dict:
        cfg = run["cfg"]
        base = base_args(n_paths=run["n_paths"], contract=run["contract"])
        base = _apply_contract(base, run["contract"])
        # Re-tag the run so log/CSV rows are clearly grouped
        tagged = SweepConfig(label=cfg.label, overrides=dict(cfg.overrides),
                              seed=run["seed"], note=cfg.note)
        row = run_one(tagged, base, args.dry_run, tag_suffix=run["tag"])
        row.update({
            "tag": run["tag"],
            "n_paths": str(run["n_paths"]),
            "contract": run["contract"],
        })
        return row

    if args.dry_run:
        for r in runs:
            print(f"[DRY] {r['tag']:6s} {r['cfg'].label:<22} seed={r['seed']} "
                  f"n={r['n_paths']} contract={r['contract']}")
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
