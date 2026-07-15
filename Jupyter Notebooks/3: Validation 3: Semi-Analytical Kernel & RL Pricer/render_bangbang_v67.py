"""Render Fig 4 (bang_bangness_rl) from the 8-seed v67 sweep, with mean +/- SE bands.

Faithful to NB6 cell 27's style (viridis by cost, RL solid + LSM dashed, x=gamma), upgraded to show
the 8-seed mean with a +/-1 SE band on the RL line. Reads bangbang_v67.csv. Appendix A (cell 29,
threshold sensitivity, single-seed c=0.05) is rendered separately by render_appendixA_v67.py.

Run:  python render_bangbang_v67.py
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CSV = HERE / "bangbang_v67.csv"
FIG = ROOT / "figs" / "convex_costs_0p04" / "bang_bangness_rl"
PLOT_COSTS = [0.01, 0.02, 0.04, 0.08, 0.15]

df = pd.read_csv(CSV)
df = df[df.c.isin(PLOT_COSTS) & (df.gamma <= 2.2)]

# Sized for a single cas-dc column (columnwidth = 3.30 in) at scale 1.0, so the on-page
# font sizes match the surrounding ~9pt body / 8pt caption text without any LaTeX rescaling.
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7})
fig, ax = plt.subplots(1, 1, figsize=(3.30, 2.95), dpi=300)
cmap = plt.cm.viridis
n_c = len(PLOT_COSTS)
c_colors = {c: cmap(i / max(n_c - 1, 1)) for i, c in enumerate(PLOT_COSTS)}

for c in PLOT_COSTS:
    sub = df[df.c == c].sort_values("gamma")
    color = c_colors[c]
    gam = sub.gamma.values
    rl, rl_se = sub.RL_BangBangness_mean.values, sub.RL_BangBangness_se.values
    ax.fill_between(gam, rl - rl_se, rl + rl_se, color=color, alpha=0.18, lw=0)
    ax.plot(gam, rl, "-o", color=color, ms=3.5, lw=1.4)
    ax.plot(gam, sub.LSM_BangBangness.values, "--*", color=color, ms=5, lw=1.3)

# Title intentionally omitted -- the LaTeX caption carries the description.
ax.set_xlabel(r"$\gamma$ (cost exponent)")
ax.set_ylabel("Bang-Bangness $B$")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xticks([1.0, 1.5, 2.0])
ax.grid(alpha=0.3)

cost_handles = [Line2D([0], [0], color=c_colors[c], lw=2, label=f"$c={c}$") for c in PLOT_COSTS]
style_handles = [Line2D([0], [0], color="0.3", lw=1.4, marker="o", ms=3.5, label="RL (AC-kernel)"),
                 Line2D([0], [0], color="0.3", lw=1.3, ls="--", marker="*", ms=5, label="LSM-D")]
leg1 = ax.legend(handles=cost_handles, loc="upper right", fontsize=6.5, title="cost coeff.",
                 title_fontsize=6.5, handlelength=1.6, borderpad=0.4, labelspacing=0.3)
ax.add_artist(leg1)
ax.legend(handles=style_handles, loc="lower left", fontsize=6.5, handlelength=1.8,
          borderpad=0.4, labelspacing=0.3)

fig.tight_layout(pad=0.4)
FIG.parent.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG}.{ext}", bbox_inches="tight", dpi=300)
print(f"wrote {FIG}.png / .pdf")
print(df[["c", "gamma", "RL_BangBangness_mean", "RL_BangBangness_se", "LSM_BangBangness"]]
      .to_string(index=False))
