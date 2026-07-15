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

# Balanced ladder for convex cells: (nXY, nQ, Mx).  Non-geometric on purpose (the
# T1/T11 finding that nQ controls depends on it); the LS fit handles arbitrary h.
CONVEX_LADDER = [(61, 76, 12), (81, 101, 16), (121, 151, 24), (161, 201, 32)]
# Spatial ladder for linear (lattice) cells: nQ pinned to the exact 11-node lattice.
LATTICE_LADDER = [(61, 11, 12), (81, 11, 16), (121, 11, 24), (161, 11, 32)]

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
        V0, p_obs, rms = ls_order_fit(h, V)
        delta_re = abs(V0 - V[-1])
        # Fallback when the ladder is at the noise floor (lattice cells: errors ~1e-6 and
        # the order fit is unconstrained): band on the last-rung spread instead.
        spread = abs(V[-1] - V[-2])
        if not np.isfinite(V0) or rms > max(delta_re, 1e-12):
            u_num = FS * max(spread, rms if np.isfinite(rms) else 0.0)
            V0, p_obs = float("nan"), float("nan")
        else:
            u_num = FS * max(delta_re, rms)
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
