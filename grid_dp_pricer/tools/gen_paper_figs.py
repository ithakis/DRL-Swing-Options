#!/usr/bin/env python3
"""gen_paper_figs.py — Figure D1 for the manuscript's DP-reference subsection.

Two panels, full text width (6.85in), 8pt serif to match the manuscript:
  (a) grid convergence at the focal cell: |V - V_ext| vs axis resolution h (log-log)
      per axis (nX, nY, nQ), with fitted observed orders — data: DP Validation T1_spatial.csv
  (b) accuracy-vs-time Pareto: DP balanced-ladder frontier (T12_pareto.csv) with the three
      incumbents (LSM-D, AC-kernel, AC-sample) placed at their wall-clock and their
      DP-certified accuracy gap (dp_publication_sweep.csv x Convex Costs Results 9.csv).

Output: figs/dp_convergence_pareto.pdf (+ .png) — referenced by the paper as
../figs/dp_convergence_pareto.pdf.

Run from repo root: python grid_dp_pricer/tools/gen_paper_figs.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T1 = ROOT / "Jupyter Notebooks" / "DP Validation" / "results" / "T1_spatial.csv"
T12 = ROOT / "Jupyter Notebooks" / "DP Validation" / "results" / "T12_pareto.csv"
DPCSV = ROOT / "grid_dp_pricer" / "results" / "dp_publication_sweep.csv"
R9 = ROOT / "Jupyter Notebooks" / "Convex Costs Results 9.csv"
OUT = ROOT / "figs" / "dp_convergence_pareto.pdf"

# Incumbent wall-clocks on the same M1 (single fit / single train+price incl. 65k OOS eval):
# LSM-D 14.5 s; AC-kernel C++ 43 s at its 4,096-episode budget; AC-sample scaled linearly
# to its 32,768-episode budget (8 x 43 s) — see grid_dp_pricer/RESULTS.md "Head-to-head".
# Accuracy gaps are computed live from the sweep CSVs below.
T_LSM, T_ACK, T_ACS = 14.5, 43.0, 8 * 43.0

AXIS_STYLE = {"nX": ("$n_X$", "tab:blue", "o"),
              "nY": ("$n_Y$", "tab:green", "s"),
              "nQ": ("$n_Q$", "tab:red", "^")}


def panel_convergence(ax):
    df = pd.read_csv(T1)
    for axis, (lab, col, mk) in AXIS_STYLE.items():
        sub = df[df.axis == axis].sort_values("h")
        # Per-axis Richardson limit from the three finest rungs (matches T1_summary).
        V = sub.price.to_numpy()
        h = sub.h.to_numpy()
        v3, v2, v1 = V[:3]  # finest three (ascending h)
        h3, h2, h1 = h[:3]
        # generalized two-ratio Richardson: solve for p, then extrapolate
        from scipy.optimize import brentq
        f = lambda p: (v2 - v3) * (h1**p - h2**p) - (v1 - v2) * (h2**p - h3**p)
        try:
            p = brentq(f, 0.05, 8.0)
            vext = v3 + (v3 - v2) * h3**p / (h2**p - h3**p)
        except ValueError:
            p, vext = float("nan"), v3
        err = np.abs(V - vext)
        err = np.where(err < 1e-9, 1e-9, err)  # floor for log axis
        order_lab = f"{lab}" + (f" ($p\\approx{p:.1f}$)" if np.isfinite(p) and axis != "nX" else " (noise floor)")
        ax.loglog(h, err, marker=mk, ms=3.5, lw=1.0, color=col, label=order_lab)
    ax.set_xlabel("axis step $h$")
    ax.set_ylabel(r"$|V(h) - V_{\mathrm{ext}}|$")
    ax.set_title("(a) Grid convergence, focal cell", fontsize=8)
    ax.legend(fontsize=6, loc="lower right", handlelength=1.6)
    ax.grid(True, which="both", alpha=0.3)


def panel_pareto(ax):
    df = pd.read_csv(T12)
    # Trustworthy regime only: the nXY<31 rungs are the deliberate breakdown study
    # (grossly under-resolved Q axis) and are not part of the operating frontier.
    bal = df[(df.family == "balanced") & (df.nX >= 31)].sort_values("t_total_ms")
    err = np.maximum(bal.err_pct.to_numpy(), 1e-6)
    t = bal.t_total_ms.to_numpy() / 1e3
    ax.loglog(t, err, ls="none", marker="o", ms=2.5, color="tab:purple", alpha=0.45)
    # Lower (Pareto) envelope of the ladder.
    env_t, env_e, best = [], [], np.inf
    for ti, ei in zip(t, err):
        if ei < best:
            best = ei
            env_t.append(ti); env_e.append(ei)
    ax.loglog(env_t, env_e, marker="o", ms=3.5, lw=1.2, color="tab:purple",
              label="DP ladder (Pareto envelope)")

    # Incumbent certified gaps at the focal cell: |method mean / V_dp - 1| * 100.
    dp = pd.read_csv(DPCSV)
    vdp = float(dp[(dp.c == 0.04) & (dp.gamma == 2.0)].V_dp.iloc[0])
    r9 = pd.read_csv(R9)
    row = r9[(r9.c == 0.04) & (r9.gamma == 2.0)].iloc[0]
    marks = [("LSM-D ($M{=}5$)", row.LSM_mean, T_LSM, "tab:blue", "s"),
             ("AC-kernel (4k ep)", row.ACkernel_mean, T_ACK, "tab:orange", "D"),
             ("AC-sample (32k ep)", row.ACsample_mean, T_ACS, "tab:green", "v")]
    for lab, v, tt, col, mk in marks:
        gap = abs(v / vdp - 1.0) * 100.0
        ax.scatter([tt], [gap], marker=mk, s=26, color=col, zorder=5, label=lab)
    ax.set_xlabel("wall-clock (s, 8 threads M1)")
    ax.set_ylabel("price error vs converged DP (%)")
    ax.set_title("(b) Accuracy vs time, focal cell", fontsize=8)
    ax.legend(fontsize=6, loc="lower left", handlelength=1.6)
    ax.grid(True, which="both", alpha=0.3)


def main():
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                         "grid.alpha": 0.3, "axes.titlesize": 8, "axes.labelsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.55))
    panel_convergence(axes[0])
    panel_pareto(axes[1])
    fig.tight_layout(w_pad=2.0)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT} and .png")


if __name__ == "__main__":
    main()
