#!/usr/bin/env python3
"""dp_publication_sweep.py — the ONE canonical DP number set for the paper.

For every cell of the convex-cost grid (28 positive-cost cells + the zero-cost row) this
produces a quoted DP price with a per-cell numerical-uncertainty band U_num, following the
Grid Convergence Index methodology (Roache 1994; ASME V&V 20-2009) with the least-squares
observed-order fit of Eca & Hoekstra (JCP 2014) — required here because the refinement
ladder is deliberately non-geometric.

Two regimes:
  * LINEAR payoff cells (gamma==1 or c==0): exact Q-lattice mode (--latticeQ, nQ=11).
    The Q/action dimension is solved exactly (see payoff.hpp:solve_inner_lattice); the
    remaining error is spatial/quadrature, estimated from a spatial ladder.
  * CONVEX cells (gamma>1): balanced four-grid ladder; the controlling axis is nQ
    (observed order ~1, see RESULTS.md), so h = 1/(nQ-1).

Per cell: V_fine (quoted price = finest grid, NOT extrapolated — ASME convention),
V_ext (LS Richardson limit), p_obs (observed order), U_num = Fs * max(|V_ext - V_fine|,
RMS fit residual) with Fs = 1.25, plus full grid/range provenance and timings.

Usage:  python grid_dp_pricer/tools/dp_publication_sweep.py [--binary PATH] [--threads 8]
Output: grid_dp_pricer/results/dp_publication_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

C_GRID = [0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15]
G_GRID = [1.0, 1.5, 2.0, 3.0]

# Convex cells: pure-nQ geometric ladder (r=2) at FIXED converged spatial/quadrature
# (121/121, Mx=24 — per-axis residuals <=2.4e-5 at the focal cell, T1).  A balanced
# all-axes ladder is NOT usable for the uncertainty band: errors of different sign
# cancel along the diagonal refinement path and the band comes out overconfident
# (observed: +/-4e-6 on the focal cell vs the known ~6e-5 nQ-axis residual).  The nQ
# axis is the controlling first-order axis, and a geometric ratio makes the 3-point
# Richardson/GCI textbook-clean.  The spatial+quadrature residual is added on top.
CONVEX_LADDER = [(121, 101, 24), (121, 201, 24), (121, 401, 24)]
U_SPATIAL = 3.0e-5  # per-axis nX/nY/Mx residuals at the reference spatial grid (T1_summary)
# Spatial ladder for linear (lattice) cells: nQ pinned to the exact 11-node lattice, so the
# spatial axis is the ONLY error source and is cheap enough to brute-force.  At gamma=1 the
# value surface has a KINK along the exercise boundary (x+y = log(1+c/K)); the signed
# cubic-spline transfer then oscillates with node-vs-kink alignment (~1e-3 at nXY<=161,
# decaying to ~1e-4 by nXY>=281), so the band is the oscillation ENVELOPE of the finest
# rungs, not a Richardson fit (the LS order is meaningless on oscillatory data).
LATTICE_LADDER = [(281, 11, 48), (321, 11, 48), (361, 11, 48),
                  (401, 11, 48), (481, 11, 48), (641, 11, 48)]

FS = 1.25  # GCI factor of safety (three+ grid study with observed order)


def run_cell(binary: str, c: float, gamma: float, nxy: int, nq: int, mx: int,
             threads: int, lattice: bool) -> dict:
    cmd = [binary, "--c", str(c), "--gamma", str(gamma), "--nX", str(nxy), "--nY", str(nxy),
           "--nQ", str(nq), "--Mx", str(mx), "--threads", str(threads)]
    if lattice:
        cmd.append("--latticeQ")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def ls_order_fit(h: np.ndarray, V: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit V(h) = V0 + C*h^p (Eca & Hoekstra).  Returns (V0, p, rms)."""
    best = (np.nan, np.nan, np.inf)
    for p in np.arange(0.2, 6.001, 0.005):
        X = np.column_stack([np.ones_like(h), h ** p])
        coef, *_ = np.linalg.lstsq(X, V, rcond=None)
        r = V - X @ coef
        rms = float(np.sqrt(np.mean(r * r)))
        if rms < best[2]:
            best = (float(coef[0]), float(p), rms)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build" / "price_dp"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "dp_publication_sweep.csv"))
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()

    cells = [(0.0, 1.0)] + [(c, g) for c in C_GRID for g in G_GRID]
    rows = []
    for c, gamma in cells:
        lattice = (gamma == 1.0) or (c == 0.0)
        ladder = LATTICE_LADDER if lattice else CONVEX_LADDER
        vals, hs, t_tot, fine = [], [], 0.0, None
        for nxy, nq, mx in ladder:
            d = run_cell(args.binary, c, gamma, nxy, nq, mx, args.threads, lattice)
            vals.append(d["price"])
            hs.append(1.0 / (nxy - 1) if lattice else 1.0 / (nq - 1))
            t_tot += d["t_total_ms"] / 1e3
            fine = d
        V = np.array(vals)
        h = np.array(hs)
        if lattice:
            # Oscillation-envelope band over the last 4 rungs; quote the finest rung.
            tail = V[-4:]
            u_num = FS * float(tail.max() - tail.min())
            V0, p_obs, rms = float("nan"), float("nan"), float("nan")
        else:
            # Classic 3-point Richardson on the geometric (r=2) nQ ladder.
            d1, d2 = V[1] - V[0], V[2] - V[1]
            rms = float("nan")
            if d2 != 0.0 and d1 / d2 > 1.0:
                p_obs = float(np.log2(d1 / d2))
                V0 = float(V[2] + d2 / (2.0 ** p_obs - 1.0))
                u_num = FS * (abs(V0 - V[2]) + U_SPATIAL)
            else:
                # Non-monotone tail: converged to the spatial noise floor; band on the
                # last-rung spread + spatial residual.
                V0, p_obs = float("nan"), float("nan")
                u_num = FS * (abs(d2) + U_SPATIAL)
        rows.append({
            "c": c, "gamma": gamma, "mode": "latticeQ" if lattice else "ladder",
            "V_dp": f"{V[-1]:.9f}", "U_num": f"{u_num:.2e}",
            "V_ext": f"{V0:.9f}" if np.isfinite(V0) else "",
            "p_obs": f"{p_obs:.2f}" if np.isfinite(p_obs) else "",
            "fit_rms": f"{rms:.2e}" if np.isfinite(rms) else "",
            "ladder_V": ";".join(f"{v:.9f}" for v in vals),
            "nX": fine["nX"], "nY": fine["nY"], "nQ": fine["nQ"], "Mx": fine["Mx"],
            "Nmax": fine["Nmax"], "interp": fine["interp"], "fp64": fine["fp64"],
            "X_lo": -1.6, "X_hi": 1.6, "Y_lo": 0.0, "Y_hi": 4.0,
            "threads": args.threads, "t_ladder_s": f"{t_tot:.1f}", "git": commit,
        })
        print(f"c={c:<5g} g={gamma:<4g} {rows[-1]['mode']:>8}  V={V[-1]:.7f} "
              f"±{u_num:.1e}  p={rows[-1]['p_obs'] or 'n/a':>5}  ({t_tot:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    sys.exit(main())
