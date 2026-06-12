#!/usr/bin/env python
"""Train extra v64 seeds for notebook 3's robustness figures (R1 nocost CI, R3 full set).

Adds, at the v64 canonical recipe (run.py defaults: kernel-on M_x=2, depth-3, 4096 ep):
  * focal (c=0.04, gamma=2) seeds 14..25  -> total 15 focal seeds, matching the v61 paper
    family used in R3 / R1-g2 / Summary.  Named like the 4k sweep agents so the generators
    (discover_v64_canonical) and the comparison eval pick them up automatically.
  * no-cost (c=0, gamma=1) seeds 14..22    -> a real cross-seed CI for the R1 no-cost bar
    (the v61 no-cost comparator has only 1 saved seed and cannot grow).

Light eval (the generators re-evaluate on the 65 536-path test set), restartable
(skip if runs/<name>.{pth,json} exist).  Default concurrency 2 (coexists with the 32k sweep).

Usage:  python tools/train_nb3_extra_seeds.py [--concurrency 2] [--dry_run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(ROOT, "runs")
LOG_DIR = os.path.join(ROOT, "logs", "_nb3_extra")
COMMON = ["-n_paths=4096", "-eval_every=4096", "-n_paths_eval=4096",
          "--disable_csv_logging=1", "--limit_logging_frequency=1"]


def jobs() -> List[Tuple[str, List[str]]]:
    out = []
    for s in range(14, 26):  # focal g2 seeds 14..25
        out.append((f"SwingOption_20_c0.04_gamma2_v64_4k_{s}",
                    ["--c_cost=0.04", "--gamma_cost=2", "-seed", str(s)]))
    for s in range(14, 23):  # no-cost seeds 14..22
        out.append((f"nb3_v64_nocost_g1_ep4096_s{s}",
                    ["--c_cost=0", "--gamma_cost=1", "-seed", str(s)]))
    return out


def is_done(name: str) -> bool:
    return (os.path.exists(os.path.join(RUNS_DIR, name + ".pth"))
            and os.path.exists(os.path.join(RUNS_DIR, name + ".json")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)

    plan = [(n, fl) for n, fl in jobs() if not is_done(n)]
    print(f"=== nb3 extra-seed training | concurrency={args.concurrency} ===")
    print(f"Total {len(jobs())} | done {len(jobs())-len(plan)} | to run {len(plan)}")
    for n, _ in plan:
        print("  TODO", n)
    if args.dry_run or not plan:
        return 0

    active: List = []
    queue = list(plan)
    done = 0
    failed: List[str] = []

    def launch(job):
        name, fl = job
        cmd = [sys.executable, "run.py"] + COMMON + fl + ["-name", name]
        logf = open(os.path.join(LOG_DIR, name + ".log"), "w")
        logf.write("CMD: " + " ".join(cmd) + "\n"); logf.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        print(f"[launch] {name} (pid {p.pid})")
        return (p, name, logf, time.time())

    while queue or active:
        while queue and len(active) < args.concurrency:
            active.append(launch(queue.pop(0)))
        time.sleep(2.0)
        still = []
        for p, name, logf, t0 in active:
            rc = p.poll()
            if rc is None:
                still.append((p, name, logf, t0)); continue
            logf.close(); done += 1
            ok = rc == 0 and is_done(name)
            if not ok:
                failed.append(name)
            print(f"[done {done}/{len(plan)}] {name} {'OK' if ok else f'FAIL(rc={rc})'} "
                  f"({(time.time()-t0)/60:.1f} min) | running {len(still)} | queued {len(queue)}")
        active = still

    print(f"\n=== extra-seed training done | {done} run, {len(failed)} failed ===")
    for n in failed:
        print("  FAILED", n)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
