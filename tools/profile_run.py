#!/usr/bin/env python3
"""Whole-run cProfile for a single training run, bucketed by source module.

Grounds the "is this feature expensive relative to end-to-end run time?" question
from the kernel-simplification plan. Runs one focal training (kernel ON by default)
under cProfile, then aggregates self-time (tottime) into interpretable buckets:
HHK simulation, env stepping, replay/PER, kernel quadrature, NN fwd/bwd, LSM,
evaluation, and torch/other.

Usage (defaults to focal cc_g2, kernel ON, M_x=2, 4096 ep):
    python tools/profile_run.py
    python tools/profile_run.py --n_paths 2048 --extra -per 0
    python tools/profile_run.py --no-kernel        # baseline single-sample target
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys

# Map a source-file basename -> human bucket. Order doesn't matter; lookup is exact.
_MODULE_BUCKETS = {
    "simulate_hhk_spot.py": "HHK simulation",
    "swing_env.py": "Env stepping",
    "MultiPro.py": "Env stepping",
    "replay_buffer.py": "Replay / PER",
    "transition_kernel.py": "Kernel quadrature",
    "networks.py": "NN fwd/bwd",
    "agent.py": "Agent update",
    "lsm_swing_pricer.py": "LSM baseline",
    "agent_evaluation.py": "Evaluation",
    "swing_contract.py": "Other (project)",
    "hedging_utils.py": "Other (project)",
    "run.py": "Orchestration",
}


def _bucket_for(filename: str) -> str:
    base = os.path.basename(filename)
    if base in _MODULE_BUCKETS:
        return _MODULE_BUCKETS[base]
    if "torch" in filename:
        return "torch internals"
    if "numpy" in filename or "numba" in filename:
        return "numpy / numba"
    if filename in ("~", "") or filename.startswith("<"):
        return "builtins"
    return "stdlib / third-party"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_paths", type=int, default=4096)
    ap.add_argument("--n_paths_eval", type=int, default=4096)
    ap.add_argument("--eval_every", type=int, default=2048,
                    help="Trigger ~1-2 evals so eval cost is captured. -1 for end-only.")
    ap.add_argument("--c_cost", type=float, default=0.04)
    ap.add_argument("--gamma_cost", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-kernel", dest="kernel", action="store_false")
    ap.add_argument("--n_cores", type=int, default=1,
                    help="Single core by default for a clean, comparable profile.")
    ap.add_argument("--out", type=str, default="profilelogs/profile_run.prof")
    ap.add_argument("--top", type=int, default=25, help="Top-N functions by tottime to print.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra args forwarded verbatim to run.py (e.g. --extra -per 0).")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    run_argv = [
        "run.py",
        "-name", "_profile_run",
        "-seed", str(args.seed),
        "-n_paths", str(args.n_paths),
        "-n_paths_eval", str(args.n_paths_eval),
        "-eval_every", str(args.eval_every),
        "--c_cost", str(args.c_cost),
        "--gamma_cost", str(args.gamma_cost),
        "-g", "1",
        "-n_cores", str(args.n_cores),
    ]
    if args.kernel:
        run_argv += ["--use_expected_target=1", "--kernel_M_x=2",
                     "--kernel_M_per_k=1", "--kernel_N_max=1"]
    run_argv += list(args.extra)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import run as run_module  # noqa: E402  (after sys.path setup)

    sys.argv = run_argv
    profiler = cProfile.Profile()
    print(f"Profiling: {' '.join(run_argv)}\n")
    profiler.enable()
    try:
        run_module.main()
    finally:
        profiler.disable()

    profiler.dump_stats(args.out)
    stats = pstats.Stats(profiler)

    total = sum(tt for (_cc, _nc, tt, _ct, _callers) in stats.stats.values())
    buckets: dict[str, float] = {}
    for (filename, _line, _func), (_cc, _nc, tt, _ct, _callers) in stats.stats.items():
        buckets[_bucket_for(filename)] = buckets.get(_bucket_for(filename), 0.0) + tt

    print("\n" + "=" * 64)
    print(f"WHOLE-RUN PROFILE  (total self-time = {total:.1f}s, kernel={'ON' if args.kernel else 'OFF'})")
    print("=" * 64)
    for name, tt in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<22} {tt:8.1f}s  {100 * tt / total:5.1f}%")

    print("\n" + "-" * 64)
    print(f"TOP {args.top} FUNCTIONS BY SELF-TIME")
    print("-" * 64)
    stats.sort_stats("tottime").print_stats(args.top)
    print(f"\nSaved raw profile to {args.out}")


if __name__ == "__main__":
    main()
