"""Regenerate episode_efficiency.csv (R2) and r6_closing_gap.csv (Discussion) under v64.

The Discussion figure now tells the v61->v64 story directly: focal (c=0.04, gamma=2) Delta%
vs the full-state LSM-D (M=5) for the published v61 paper agents (@32 768 ep) versus the v64
re-baseline (@4 096 ep, and @32 768 ep if the 32k sweep has reached focal).  Both are scored
on the SAME common fresh test set / LSM protocol, so the bars are directly comparable.

Run:  python gen_closing_gap.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("genrl", os.path.join(HERE, "gen_rl_validation.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# 1) regenerate episode_efficiency.csv (R2) from v64 agents
m.gen_R2().to_csv(os.path.join(HERE, "episode_efficiency.csv"), index=False)
print("episode_efficiency.csv regenerated")

# 2) focal context (test set + LSM M=5), cached
ctx = m.context("g2", 0.04, 2.0)
lsm5 = ctx["lsm"][5]["price"]


def _delta_stats(config, prices, episodes):
    prices = np.asarray(prices, float)
    d = 100.0 * (prices / lsm5 - 1.0)
    return dict(config=config, episodes=episodes, n=len(d), mean_price=prices.mean(),
                mean_delta=d.mean(), seed_std=(d.std(ddof=1) if len(d) > 1 else 0.0),
                best_delta=d.max(), worst_delta=d.min())


rows = []

# v61 paper @ 32 768 (deleted agents) — per-seed saved prices from the paper CSV
v61 = m.v61_paper_seeds(0.04, 2.0)
if v61:
    rows.append(_delta_stats("v61 paper (32k)", list(v61.values()), 32768))

# v64 @ 4 096 (canonical) — evaluated on the common test set
for label, ep in [("v64 (4k)", 4096), ("v64 (32k)", 32768)]:
    runs = m.discover_v64(0.04, 2.0, ep, 2, 1, 1)
    if not runs:
        print(f"{label}: no agents yet (skip)")
        continue
    prices = [m.eval_agent(jp, ctx["contract"], ctx["test"])["test_price"] for _, jp in sorted(runs.items())]
    rows.append(_delta_stats(label, prices, ep))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "r6_closing_gap.csv"), index=False)
print(f"\nLSM(M=5) focal price = {lsm5:.4f}")
for _, r in df.iterrows():
    print(f"  {r['config']:18s} n={int(r['n']):2d}  Delta%={r['mean_delta']:+.3f} +/- {r['seed_std']:.3f}")
