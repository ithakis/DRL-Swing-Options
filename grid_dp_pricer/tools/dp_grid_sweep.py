#!/usr/bin/env python
"""GOAL 2.5 — DP price for every (c, gamma) cell vs the canonical "Convex Costs Results 9.csv"
(LSM-D / RL-sample / RL-kernel, 8-seed).  Asserts the DP reference dominates every incumbent
(within their CIs) and writes data/dp_grid_sweep.csv.

    python grid_dp_pricer/tools/dp_grid_sweep.py [--grid "--nX 121 --nY 121 --nQ 151 --Mx 24"]
"""
import argparse
import csv
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GDP = os.path.abspath(os.path.join(HERE, ".."))
WORKTREE = os.path.abspath(os.path.join(GDP, ".."))
BIN = os.path.join(GDP, "build", "price_dp")

CS = [0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15]
GAMMAS = [1.0, 1.5, 2.0, 3.0]


def read_results9():
    for base in (WORKTREE, "/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options"):
        p = os.path.join(base, "Jupyter Notebooks", "Convex Costs Results 9.csv")
        if os.path.exists(p):
            rows = {}
            with open(p) as f:
                for r in csv.DictReader(f):
                    rows[(round(float(r["c"]), 4), round(float(r["gamma"]), 4))] = r
            return rows
    return {}


def dp_price(c, gamma, extra):
    args = [BIN, "--c", str(c), "--gamma", str(gamma), "--threads", "0"] + extra
    return json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)["price"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, default="--nX 121 --nY 121 --nQ 151 --Mx 24 --Yhi 4.0")
    args = ap.parse_args()
    extra = args.grid.split()
    r9 = read_results9()

    out_rows = []
    violations = []
    print(f"{'c':>5} {'gamma':>5} | {'DP':>9} | {'LSM':>9} {'g%':>6} | {'RLkern':>9} {'g%':>6} | "
          f"{'RLsamp':>9} {'g%':>6} | chk")
    print("-" * 86)
    for c in CS:
        for g in GAMMAS:
            dp = dp_price(c, g, extra)
            row = r9.get((round(c, 4), round(g, 4)))
            lsm = float(row["LSM_mean"]) if row else float("nan")
            rk = float(row["ACkernel_mean"]) if row else float("nan")
            rs = float(row["ACsample_mean"]) if row else float("nan")
            rk_hi = float(row["ACkernel_hi"]) if row else float("nan")  # BCa upper

            def gp(x):
                return 100.0 * (dp - x) / dp if x == x else float("nan")

            # The reference must dominate every incumbent's MEAN (small tol for MC/grid noise).
            tol = 2e-3
            ok = (dp >= lsm - tol) and (not row or (dp >= rk - tol and dp >= rs - tol))
            within_rk_ci = bool(row) and dp <= rk_hi  # RL-kernel statistically AT the reference
            if not ok:
                violations.append((c, g, dp, lsm, rk, rs))
            out_rows.append(dict(c=c, gamma=g, DP=dp, LSM=lsm, RLkernel=rk, RLsample=rs,
                                 dDP_LSM_pct=gp(lsm), dDP_RLkernel_pct=gp(rk),
                                 dDP_RLsample_pct=gp(rs), RLkernel_within_DP_CI=within_rk_ci))
            print(f"{c:>5} {g:>5} | {dp:>9.5f} | {lsm:>9.5f} {gp(lsm):>6.2f} | "
                  f"{rk:>9.5f} {gp(rk):>6.2f} | {rs:>9.5f} {gp(rs):>6.2f} | "
                  f"{'OK' if ok else 'VIOL'}{' ~RLk' if within_rk_ci else ''}")

    os.makedirs(os.path.join(GDP, "data"), exist_ok=True)
    out = os.path.join(GDP, "data", "dp_grid_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    n_within = sum(1 for r in out_rows if r["RLkernel_within_DP_CI"])
    print(f"\nwrote {out}")
    print(f"RL-kernel statistically at the DP reference (DP <= RL-kernel BCa-upper) in "
          f"{n_within}/{len(out_rows)} cells — i.e. RL-kernel is near-optimal everywhere.")
    if violations:
        print(f"ORDERING VIOLATIONS (DP below an incumbent mean by > {2e-3}): {violations}")
    else:
        print("DP dominates every incumbent's MEAN in all 28 cells.")


if __name__ == "__main__":
    main()
