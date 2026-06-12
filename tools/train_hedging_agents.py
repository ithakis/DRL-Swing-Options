#!/usr/bin/env python
"""Train the focal (c=0.04, gamma=2) agents behind the Hedging notebook's three-way
comparison: LSM vs RL (no kernel) vs RL-kernel.

The two RL variants use DIFFERENT canonical configurations -- the kernel makes the TD
target deterministic, and several knobs were re-tuned for that (HPT.md "Deterministic-
target retune" + "Mega campaign").  This file is the single place both configs are
defined; the notebook only selects saved runs by name.

  * kernel    -> v64 canonical = the run.py defaults (kernel ON, fast M_x=2, depth-3,
                 beta-sigmoid 1.5, lr_c 6e-4, learn_number 2, linear noise + eval-EMA,
                 critic_warmup 512).  No overrides.
  * nokernel  -> the pre-deterministic-target (v63-era) canonical with the kernel OFF.
                 Every override below restores the documented pre-kernel value of a knob
                 that was later re-tuned for the deterministic target:
                   --use_expected_target=0            the switch itself
                   --actor_layers=2 --critic_layers=2 pre-Stage-C depth (v61/v63)
                   --actor_output_activation=beta_sigmoid_3.0  pre-Stage-C squash
                   -learn_number=1                    pre-Stage-C single gradient step
                   -lr_c=3e-4                         pre-Task-3 critic LR
                   --noise_schedule=hyperbolic        pre-Task-2 exploration ...
                   --noise_plateau=<10% of budget>    ... with the v61 plateau (3.2k @ 32k)
                   --weight_averaging=off             pre-Task-2 (no eval EMA)
                   --critic_warmup_episodes=1024      v59-v61 warmup (512 was tuned under
                                                      the kernel)
                 Everything else (tau, lr_a 3e-4, robust normalisation, profitability
                 gate, uniform replay, single critic step, closed-form warm-start) is
                 shared between the variants.  NB: the published v61 paper agents also
                 used PER, target-policy noise and an LR-decay schedule -- all three were
                 deleted from the codebase after being shown neutral-to-harmful, so this
                 is the closest reproducible no-kernel canonical.

Run names: SwingOption_20_c0.04_gamma2_{kernel,nokernel}_{<ep>}_{seed}
(the kernel 4k/32k agents already exist as SwingOption_20_c0.04_gamma2_v64_{4k,32k}_*;
this harness only needs to train the nokernel ones, but can train both).

Usage:
    python tools/train_hedging_agents.py --variant nokernel --episodes 32768 --seeds 11
    python tools/train_hedging_agents.py --variant both --episodes 256 --smoke   # switch validation
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(ROOT, "runs")
LOG_DIR = os.path.join(ROOT, "logs", "_hedging_agents")

# Minimal-overhead eval during training; the notebook re-evaluates the saved actor.
COMMON = ["-n_paths_eval=4096", "--disable_csv_logging=1", "--limit_logging_frequency=1"]

FOCAL = ["--c_cost=0.04", "--gamma_cost=2"]


def variant_flags(variant: str, episodes: int) -> list[str]:
    if variant == "kernel":
        return []  # v64 canonical = run.py defaults
    if variant == "nokernel":
        return [
            "--use_expected_target=0",
            "--actor_layers=2", "--critic_layers=2",
            "--actor_output_activation=beta_sigmoid_3.0",
            "-learn_number=1",
            "-lr_c=3e-4",
            "--noise_schedule=hyperbolic",
            f"-noise_plateau={max(episodes // 10, 0)}",
            "--weight_averaging=off",
            "--critic_warmup_episodes=1024",
        ]
    raise ValueError(f"unknown variant {variant!r}")


def run_name(variant: str, episodes: int, seed: int, smoke: bool) -> str:
    tag = {4096: "4k", 32768: "32k"}.get(episodes, str(episodes))
    base = f"SwingOption_20_c0.04_gamma2_{variant}_{tag}_{seed}"
    return f"smoke_{base}" if smoke else base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["kernel", "nokernel", "both"], default="nokernel")
    ap.add_argument("--episodes", type=int, default=32768)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11])
    ap.add_argument("--smoke", action="store_true",
                    help="Prefix run names with smoke_ (switch validation; delete afterwards)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    variants = ["kernel", "nokernel"] if args.variant == "both" else [args.variant]
    rc = 0
    for variant in variants:
        for seed in args.seeds:
            name = run_name(variant, args.episodes, seed, args.smoke)
            if (os.path.exists(os.path.join(RUNS_DIR, name + ".pth"))
                    and os.path.exists(os.path.join(RUNS_DIR, name + ".json"))):
                print(f"[skip] {name} already saved")
                continue
            cmd = ([sys.executable, "run.py", f"-n_paths={args.episodes}",
                    f"-eval_every={args.episodes}"]
                   + COMMON + FOCAL + variant_flags(variant, args.episodes)
                   + ["-name", name, "-seed", str(seed)])
            if args.episodes < 1024:
                # tiny smoke budgets: the calibrate_bias dataset must fit in n_paths
                cmd += [f"--warmup_episodes={args.episodes // 2}"]
            print(f"[run ] {' '.join(cmd)}")
            if args.dry_run:
                continue
            log = os.path.join(LOG_DIR, name + ".log")
            t0 = time.time()
            with open(log, "w") as fh:
                p = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
            print(f"[done] {name}: {status} in {time.time() - t0:.0f}s  (log: {log})")
            rc = rc or p.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
