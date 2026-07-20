"""Render Appendix A (bang_bangness_sensitivity) from v67 c=0.05 RL parquets.

Reproduces NB6 cell 29: the Bang-Bangness hit-ratio B(k) as the threshold k sweeps 0.01..1.0, for
c=0.05 and gamma in {1,1.5,2} (seed 11). Shows the qualitative monotone B-vs-gamma ordering is
robust to the 0.95 threshold choice. Reads the v67 RL parquets saved by gen_bangbang_v67.py.

Run:  python render_appendixA_v67.py
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"
FIG = ROOT / "figs" / "convex_costs_0p04" / "bang_bangness_sensitivity"
Q_MAX, Q_MAX_STEP = 20.0, 2.0
threshold_grid = np.linspace(0.01, 1.0, 100)
GAMMAS = [(1.0, "1"), (1.5, "1p5"), (2.0, "2")]
colors = {1.0: "tab:blue", 1.5: "tab:orange", 2.0: "tab:green"}

fig, ax = plt.subplots(figsize=(10, 6))
for gv, gl in GAMMAS:
    pq = LOGS / f"SwingOption_20_c0.05_gamma{gl}_v67_11" / "evaluations" / "rl_episode_4096.parquet"
    if not pq.exists():
        print(f"MISSING {pq}"); continue
    df = pd.read_parquet(pq)
    q_rem = df["q_remaining_norm"].values * Q_MAX
    capable = df[q_rem >= (Q_MAX_STEP - 1e-5)]
    ratios = capable[capable["q_t"] > 1e-5]["q_t"].values / Q_MAX_STEP
    curve = [float((ratios >= k).mean()) for k in threshold_grid]
    ax.plot(threshold_grid, curve, label=f"$\\gamma={gv}$", color=colors[gv], lw=3)

ax.axvline(0.95, color="red", ls="--", alpha=0.5, label="Default 0.95 threshold")
ax.set_xlabel("Bang-Bang threshold $k$  ($q_t/q_{\\max} \\geq k$)", fontsize=12)
ax.set_ylabel("Bang-Bangness $B(k)$", fontsize=12)
ax.set_title("Threshold sensitivity of Bang-Bangness (RL, $c=0.05$, seed 11)", fontsize=13)
ax.legend(fontsize=11); ax.grid(alpha=0.3)
fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG}.{ext}", bbox_inches="tight", dpi=300)
print(f"wrote {FIG}.png / .pdf")
