#!/usr/bin/env python
"""Restartable v64 convex-cost sweep orchestrator.

Retrains the paper convex-cost grid under the **v64 canonical** recipe (which lives
entirely in ``run.py``'s argparse defaults) at a chosen episode budget. The cell set is
single-sourced from the shell scripts in ``Convex Cost Experiments/`` — this harness only
overrides the study budget, each cell's ``(c, gamma)``, the run name, and the seed, exactly
as those scripts do.

Run names: ``SwingOption_20_c{c}_gamma{g}_v64_{budget}_{seed}``  (budget in {4k, 32k}).
The ``_v64_{budget}`` infix keeps the 4k and 32k runs in distinct ``config_base_name``
groups for ``tools/rebuild_results_v7.py`` (it groups by everything before the trailing
``_{seed}``).

Restartability: a (cell, seed) is **skipped** when both ``runs/<name>.pth`` and
``runs/<name>.json`` already exist, so the sweep can be killed and re-run freely.

Concurrency: a bounded pool of ``python run.py`` subprocesses (default 4 for 4k, 3 for 32k
to respect 8 GB RAM — each process holds a 65536-path eval set in memory).

Usage:
    python tools/run_v64_sweep.py --budget 4k                 # 4 concurrent (default)
    python tools/run_v64_sweep.py --budget 32k                # 3 concurrent (default)
    python tools/run_v64_sweep.py --budget 4k --concurrency 2 --seeds 11 12 13
    python tools/run_v64_sweep.py --budget 4k --dry_run       # print the plan, launch nothing
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP_DIR = os.path.join(ROOT, "Convex Cost Experiments")
RUNS_DIR = os.path.join(ROOT, "runs")
LOG_DIR = os.path.join(ROOT, "logs", "_v64_sweep")

BUDGETS = {"4k": 4096, "32k": 32768}
DEFAULT_CONCURRENCY = {"4k": 4, "32k": 3}
DEFAULT_SEEDS = [11, 12, 13]

# Fixed per-cell overrides, identical to the Convex Cost Experiments/*.sh scripts
# (everything else is a run.py v64 default).
EVAL_EVERY = 1024
N_PATHS_EVAL = 65536


def discover_cells() -> List[Tuple[str, str]]:
    """Return the (c_str, gamma_str) cells from the experiment scripts.

    Parses the filename ``SwingOption_20_c{c}_gamma{g}.sh`` and keeps the *textual* form
    of c and gamma (e.g. "0.04", "1.5") so the generated run names match the historical
    naming convention exactly. ``*focal*`` scripts are excluded.
    """
    cells: List[Tuple[str, str]] = []
    pattern = os.path.join(EXP_DIR, "SwingOption_20_c*_gamma*.sh")
    for path in sorted(glob.glob(pattern)):
        stem = os.path.basename(path)[: -len(".sh")]
        if "focal" in stem:
            continue
        # stem = SwingOption_20_c0.04_gamma2
        try:
            c_part = stem.split("_c", 1)[1]  # 0.04_gamma2
            c_str, gamma_str = c_part.split("_gamma")
        except (IndexError, ValueError):
            print(f"  WARNING: could not parse cell from {stem!r}, skipping", file=sys.stderr)
            continue
        cells.append((c_str, gamma_str))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    return unique


def run_name(c_str: str, gamma_str: str, budget: str, seed: int) -> str:
    return f"SwingOption_20_c{c_str}_gamma{gamma_str}_v64_{budget}_{seed}"


def is_done(name: str) -> bool:
    pth = os.path.join(RUNS_DIR, name + ".pth")
    js = os.path.join(RUNS_DIR, name + ".json")
    return os.path.exists(pth) and os.path.exists(js)


def build_cmd(c_str: str, gamma_str: str, budget: str, seed: int, name: str) -> List[str]:
    n_paths = BUDGETS[budget]
    return [
        sys.executable,
        "run.py",
        f"-n_paths={n_paths}",
        f"-eval_every={EVAL_EVERY}",
        f"-n_paths_eval={N_PATHS_EVAL}",
        f"--c_cost={c_str}",
        f"--gamma_cost={gamma_str}",
        # Logging OFF: the re-baseline only needs the saved actor (.pth/.json); per-run eval
        # CSV/parquet logs are ~1.5 GB each at 32k and fill the disk. Evaluation is done
        # post-hoc from runs/ by rebuild_results_v7 / the notebook-3 generators.
        "--disable_csv_logging=1",
        "--limit_logging_frequency=1",
        "-name",
        name,
        "-seed",
        str(seed),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", choices=list(BUDGETS), required=True, help="Episode budget: 4k or 32k")
    ap.add_argument("--concurrency", type=int, default=None, help="Max concurrent run.py procs (default: 4 for 4k, 3 for 32k)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds (default: 11 12 13)")
    ap.add_argument("--dry_run", action="store_true", help="Print the plan and exit without launching")
    args = ap.parse_args()

    budget = args.budget
    concurrency = args.concurrency if args.concurrency is not None else DEFAULT_CONCURRENCY[budget]
    seeds = args.seeds

    cells = discover_cells()
    if not cells:
        print("ERROR: no cells discovered from Convex Cost Experiments/*.sh", file=sys.stderr)
        return 1

    os.makedirs(LOG_DIR, exist_ok=True)

    # Build the full job list (skipping already-complete runs).
    jobs: List[Tuple[str, str, int, str]] = []  # (c, gamma, seed, name)
    skipped = 0
    for c_str, gamma_str in cells:
        for seed in seeds:
            name = run_name(c_str, gamma_str, budget, seed)
            if is_done(name):
                skipped += 1
                continue
            jobs.append((c_str, gamma_str, seed, name))

    total = len(cells) * len(seeds)
    print(f"=== v64 sweep | budget={budget} (n_paths={BUDGETS[budget]}) | concurrency={concurrency} ===")
    print(f"Cells: {len(cells)} | seeds: {seeds} | total runs: {total}")
    print(f"Already complete (skipped): {skipped} | to run: {len(jobs)}")

    if args.dry_run or not jobs:
        for c_str, gamma_str, seed, name in jobs:
            print(f"  WOULD RUN: {name}")
        if not jobs:
            print("Nothing to do — all runs complete.")
        return 0

    # Bounded process pool.
    active: List[Tuple[subprocess.Popen, str, "object", float]] = []  # (proc, name, logfile, t0)
    queue = list(jobs)
    done = 0
    failed: List[str] = []
    sweep_t0 = time.time()

    def launch(job):
        c_str, gamma_str, seed, name = job
        cmd = build_cmd(c_str, gamma_str, budget, seed, name)
        log_path = os.path.join(LOG_DIR, name + ".log")
        logf = open(log_path, "w")
        logf.write("CMD: " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        print(f"[launch] {name}  (pid {proc.pid})  log -> {os.path.relpath(log_path, ROOT)}")
        return (proc, name, logf, time.time())

    try:
        while queue or active:
            # Fill the pool.
            while queue and len(active) < concurrency:
                active.append(launch(queue.pop(0)))
            # Poll for completions.
            time.sleep(2.0)
            still_active = []
            for proc, name, logf, t0 in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, name, logf, t0))
                    continue
                logf.close()
                wall = time.time() - t0
                done += 1
                ok = ret == 0 and is_done(name)
                status = "OK" if ok else f"FAIL(ret={ret})"
                if not ok:
                    failed.append(name)
                print(
                    f"[done {done}/{len(jobs)}] {name}  {status}  ({wall/60:.1f} min)  "
                    f"| running: {len(still_active)} | queued: {len(queue)}"
                )
            active = still_active
    except KeyboardInterrupt:
        print("\nInterrupted — terminating active runs (completed runs are preserved; re-run to resume).")
        for proc, name, logf, _ in active:
            proc.terminate()
            logf.close()
        return 130

    elapsed = (time.time() - sweep_t0) / 60.0
    print(f"\n=== sweep complete | budget={budget} | {done} run, {len(failed)} failed | {elapsed:.1f} min ===")
    if failed:
        print("FAILED runs (re-run the sweep to retry — completed ones are skipped):")
        for name in failed:
            print(f"  {name}  (see logs/_v64_sweep/{name}.log)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
