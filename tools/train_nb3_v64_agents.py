#!/usr/bin/env python
"""Train the v64 agents needed to regenerate notebook 3 (Validation 3) Part II under v64.

The notebook's Part II figures isolate three axes around the focal regime (c=0.04, gamma=2),
plus a no-cost regime for R1.  Everything uses the **v64 canonical recipe** (run.py defaults:
kernel-on M_x=2, depth-3, beta_sigmoid_1.5, lr_a 3e-4 / lr_c 6e-4, learn_number 2,
critic_warmup 512, linear noise + EMA).  Each job only overrides the axis it isolates.

Agents trained here (seeds 11,12,13), all LIGHT (<=8192 episodes):
  * R1 no-cost   : c=0,    gamma=1, 4096 ep                       (the 4th R1 regime)
  * R5 M_x sweep : c=0.04, gamma=2, 4096 ep, kernel_M_x in {1,3,4,6}
  * R2 episodes  : c=0.04, gamma=2, n_paths in {2048, 8192}

Sourced from elsewhere (NOT trained here, to avoid duplicate/competing heavy runs):
  * R1 g1/g15/g2 @4096 and R2 4096-point  -> the v64 4k sweep (SwingOption_20_c0.04_gamma{..}_v64_4k_*)
  * R2 32768-point                         -> the v64 32k sweep (..._v64_32k_*); fills in when it lands.
  * R5 M_x=2 (canonical)                    -> the v64 4k focal sweep agent.

Restartable: skips a job when runs/<name>.{pth,json} both exist.  Eval cadence is minimised
(one final eval, small eval set) since the actor weights are what the generators re-evaluate
on the common 65 536-path test set; the in-training eval does not affect the saved policy.

Usage:
    python tools/train_nb3_v64_agents.py                 # concurrency 2 (M1-friendly)
    python tools/train_nb3_v64_agents.py --concurrency 3
    python tools/train_nb3_v64_agents.py --dry_run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(ROOT, "runs")
LOG_DIR = os.path.join(ROOT, "logs", "_nb3_v64")
SEEDS = [11, 12, 13]

# Minimal-overhead eval during training (the generators do the real 65536-path eval).
COMMON = ["-n_paths_eval=4096", "--disable_csv_logging=1", "--limit_logging_frequency=1"]


def jobs() -> List[Tuple[str, List[str], int]]:
    """Return [(name, extra_flags, n_paths), ...].  extra_flags excludes -name/-seed/-n_paths."""
    out: List[Tuple[str, List[str], int]] = []
    # R1 no-cost regime (c=0, gamma=1) @ 4096
    out.append(("nb3_v64_nocost_g1_ep4096", ["--c_cost=0", "--gamma_cost=1"], 4096))
    # R5 M_x isolation @ focal g2, 4096 (M_x=2 comes from the canonical sweep agent)
    for mx in (1, 3, 4, 6):
        out.append((f"nb3_v64_R5_g2_Mx{mx}_ep4096",
                    ["--c_cost=0.04", "--gamma_cost=2", f"--kernel_M_x={mx}"], 4096))
    # R2 episode-efficiency @ focal g2 (4096 + 32768 sourced from the sweeps)
    for ep in (2048, 8192):
        out.append((f"nb3_v64_R2_g2_ep{ep}",
                    ["--c_cost=0.04", "--gamma_cost=2"], ep))
    return out


def is_done(name: str) -> bool:
    return (os.path.exists(os.path.join(RUNS_DIR, name + ".pth"))
            and os.path.exists(os.path.join(RUNS_DIR, name + ".json")))


def build_cmd(name: str, extra: List[str], n_paths: int, seed: int) -> List[str]:
    return ([sys.executable, "run.py", f"-n_paths={n_paths}", f"-eval_every={n_paths}"]
            + COMMON + extra + ["-name", name, "-seed", str(seed)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", type=int, default=2, help="Max concurrent run.py procs (default 2)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    # Order light -> heavy so quick jobs finish first; 8192 episodes last.
    plan: List[Tuple[str, List[str], int, int]] = []
    skipped = 0
    for name, extra, n_paths in sorted(jobs(), key=lambda j: j[2]):
        for seed in args.seeds:
            full = f"{name}_s{seed}"
            if is_done(full):
                skipped += 1
                continue
            plan.append((full, extra, n_paths, seed))

    total = len(jobs()) * len(args.seeds)
    print(f"=== nb3 v64 training | concurrency={args.concurrency} | seeds={args.seeds} ===")
    print(f"Total jobs: {total} | already done: {skipped} | to run: {len(plan)}")
    for full, _, n_paths, _ in plan:
        print(f"  TODO {full}  (n_paths={n_paths})")
    if args.dry_run or not plan:
        if not plan:
            print("Nothing to do.")
        return 0

    active: List[Tuple[subprocess.Popen, str, object, float]] = []
    queue = list(plan)
    done = 0
    failed: List[str] = []
    t_start = time.time()

    def launch(job):
        full, extra, n_paths, seed = job
        cmd = build_cmd(full[:-len(f"_s{seed}")], extra, n_paths, seed)  # name w/o seed suffix
        # build_cmd uses 'name' = base; re-inject the seeded -name:
        cmd[cmd.index("-name") + 1] = full
        logf = open(os.path.join(LOG_DIR, full + ".log"), "w")
        logf.write("CMD: " + " ".join(cmd) + "\n"); logf.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        print(f"[launch] {full} (pid {p.pid}, n_paths={n_paths})")
        return (p, full, logf, time.time())

    try:
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
    except KeyboardInterrupt:
        print("\nInterrupted; terminating active (completed runs preserved, re-run to resume).")
        for p, _, logf, _ in active:
            p.terminate(); logf.close()
        return 130

    print(f"\n=== nb3 training done | {done} run, {len(failed)} failed | {(time.time()-t_start)/60:.1f} min ===")
    if failed:
        for n in failed:
            print(f"  FAILED {n} (see logs/_nb3_v64/{n}.log)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
