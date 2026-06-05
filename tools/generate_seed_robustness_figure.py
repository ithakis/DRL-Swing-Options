#!/usr/bin/env python
"""
Generate the seed-robustness figure (Figure 4) for the paper.

Reads the focal CSV output from rebuild_results_v7.py (focal mode) and produces
a strip/swarm + box plot showing the distribution of RL test-set prices across
15 independent seeds, with the LSM baseline as a dashed reference line.

Usage:
  python tools/generate_seed_robustness_figure.py                        # defaults
  python tools/generate_seed_robustness_figure.py --csv path/to/focal.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style (matching paper figures in notebook 6)
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")
FONTSIZE = 15
TITLE_SIZE = 22
DPI = 300


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate seed robustness figure")
    p.add_argument(
        "--csv",
        type=str,
        default="Jupyter Notebooks/Convex Costs Results 7 focal.csv",
        help="Path to the focal-study CSV produced by rebuild_results_v7.py",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="figs/convex_costs_0p04",
        help="Output directory for the figure",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load focal CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(args.csv)
    if len(df) != 1:
        print(f"WARNING: Expected 1 row (focal config), got {len(df)}. Using first row.")
    row = df.iloc[0]

    lsm_price = float(row["LSM_full"])

    # Collect all RL_seedNN columns
    seed_cols = sorted([c for c in df.columns if c.startswith("RL_seed")])
    if not seed_cols:
        print("ERROR: No RL_seed* columns found in CSV.")
        sys.exit(1)

    seed_prices = []
    seed_labels = []
    for col in seed_cols:
        val = row[col]
        if pd.notna(val):
            seed_prices.append(float(val))
            seed_labels.append(col.replace("RL_seed", ""))
    seed_prices = np.array(seed_prices)
    n_seeds = len(seed_prices)

    print(f"Loaded {n_seeds} seed prices from {args.csv}")
    print(f"  LSM baseline: {lsm_price:.4f}")
    print(f"  RL mean: {np.mean(seed_prices):.4f} ± {np.std(seed_prices, ddof=1):.4f}")
    print(f"  RL range: [{np.min(seed_prices):.4f}, {np.max(seed_prices):.4f}]")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)

    # Swarm / strip plot of individual seed prices
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=n_seeds)
    ax.scatter(
        seed_prices,
        jitter,
        s=78,
        color="tab:orange",
        edgecolors="white",
        linewidths=0.6,
        zorder=5,
        label=f"RL seeds ($n={n_seeds}$)",
    )

    # Box plot (horizontal) underneath
    ax.boxplot(
        seed_prices,
        vert=False,
        positions=[0],
        widths=0.22,
        patch_artist=True,
        boxprops=dict(facecolor="tab:orange", alpha=0.25, linewidth=1.2),
        medianprops=dict(color="tab:orange", linewidth=2),
        whiskerprops=dict(color="tab:orange", linewidth=1.2),
        capprops=dict(color="tab:orange", linewidth=1.2),
        flierprops=dict(marker="", markersize=0),  # hide outlier markers (shown as scatter)
    )

    # LSM baseline
    ax.axvline(
        lsm_price,
        color="tab:blue",
        linestyle="--",
        linewidth=2,
        zorder=4,
        label=f"LSM-D baseline ({lsm_price:.4f})",
    )

    # RL mean line
    rl_mean = np.mean(seed_prices)
    ax.axvline(
        rl_mean,
        color="tab:orange",
        linestyle="-",
        linewidth=1.5,
        alpha=0.7,
        zorder=4,
        label=f"RL mean ({rl_mean:.4f})",
    )

    # Formatting
    ax.set_xlabel("Option value", fontsize=FONTSIZE, fontweight="bold")
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=FONTSIZE - 1)
    ax.legend(fontsize=FONTSIZE - 2, loc="upper left", framealpha=0.9)
    ax.set_title(
        r"Seed robustness: $c = 0.04$, $\gamma_c = 2$",
        fontsize=FONTSIZE + 2,
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        outpath = os.path.join(args.outdir, f"seed_robustness.{ext}")
        fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {outpath}")

    plt.close(fig)


if __name__ == "__main__":
    main()
