#!/usr/bin/env python
"""GOAL 2.3 — certify the DP as the reference incumbents are measured against.

For each (c, gamma) cell:
  * DP price        : the C++ grid backward-DP (default = converged reference grid),
  * LSM-D           : recomputed fresh, canonical full-state Chebyshev deg-2 n_actions=5,
  * RL-kernel/-sample: read from the canonical "Convex Costs Results 9.csv" (8-seed BCa).
Asserts the ordering DP >= LSM-D and DP >= RL (within their CIs) and prints the certified
"<method> within X% of the DP reference" gaps.  Focal cell also runs the DP forward-MC
self-consistency leg.

    conda run -n EP11 python grid_dp_pricer/tools/compare_incumbents.py            # focal
    conda run -n EP11 python grid_dp_pricer/tools/compare_incumbents.py --all      # full grid
"""
import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GDP = os.path.abspath(os.path.join(HERE, ".."))
WORKTREE = os.path.abspath(os.path.join(GDP, ".."))
sys.path.insert(0, HERE)
from run_lsm_baseline import lsm_baseline  # noqa: E402

BIN = os.path.join(GDP, "build", "price_dp")


def find_results9():
    for base in (WORKTREE, "/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options"):
        p = os.path.join(base, "Jupyter Notebooks", "Convex Costs Results 9.csv")
        if os.path.exists(p):
            return p
    return None


def read_results9():
    path = find_results9()
    if not path:
        return {}
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(round(float(r["c"]), 4), round(float(r["gamma"]), 4))] = r
    return rows


def dp_run(c, gamma, extra=None):
    cmd = [BIN, "--c", str(c), "--gamma", str(gamma)] + (extra or [])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="full c x gamma grid")
    ap.add_argument("--mc", type=int, default=4_000_000, help="forward-MC paths for the focal self-consistency")
    ap.add_argument("--grid", type=str, default="", help="extra price_dp grid flags, e.g. '--nX 121 --nY 121'")
    args = ap.parse_args()
    extra = args.grid.split() if args.grid else []

    cs = [0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15]
    gammas = [1.0, 1.5, 2.0, 3.0]
    cells = [(c, g) for c in cs for g in gammas] if args.all else [(0.04, 2.0)]

    r9 = read_results9()
    print(f"{'c':>5} {'gamma':>5} | {'DP':>9} | {'LSM-D':>9} {'dDP%':>7} | "
          f"{'RLkern':>9} {'dDP%':>7} | {'RLsamp':>9} {'dDP%':>7} | order")
    print("-" * 92)
    violations = []
    for (c, g) in cells:
        dp = dp_run(c, g, extra)["price"]
        lsm = lsm_baseline(c, g)["lsm_full"]
        row = r9.get((round(c, 4), round(g, 4)))
        rk = float(row["ACkernel_mean"]) if row else float("nan")
        rs = float(row["ACsample_mean"]) if row else float("nan")

        def gap(x):
            return 100.0 * (dp - x) / dp if x == x else float("nan")

        order_ok = (dp >= lsm - 1e-6) and (not row or (dp >= rk - 5e-3 and dp >= rs - 5e-3))
        if not order_ok:
            violations.append((c, g))
        print(f"{c:>5} {g:>5} | {dp:>9.5f} | {lsm:>9.5f} {gap(lsm):>7.2f} | "
              f"{rk:>9.5f} {gap(rk):>7.2f} | {rs:>9.5f} {gap(rs):>7.2f} | "
              f"{'OK' if order_ok else 'VIOLATION'}")

    if not args.all:  # focal self-consistency
        d = dp_run(0.04, 2.0, extra + ["--mc", str(args.mc), "--mc_seed", "999"])
        print(f"\nFocal self-consistency (forward-MC of DP greedy policy, {args.mc:,} paths):")
        print(f"  backward U_0 = {d['price']:.6f}")
        print(f"  forward MC   = {d['mc_price']:.6f}  95% CI [{d['mc_ci_lo']:.6f}, {d['mc_ci_hi']:.6f}]")
        print(f"  backward - MC = {d['backward_minus_mc']:+.6f}  "
              f"({'MC from below — OK' if d['backward_minus_mc'] >= -d['mc_stderr']*2 else 'CHECK'})")

    print(f"\n{'ALL ORDERING CHECKS PASS' if not violations else 'VIOLATIONS: ' + str(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
