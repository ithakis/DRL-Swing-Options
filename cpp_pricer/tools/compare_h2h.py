#!/usr/bin/env python3
"""Head-to-head: RL finalists vs LSM-D bar on all three metrics at the 2k-4k regime.

Reads RL per-seed sweeps (data/research/<tag>_seed{s}.json) and the LSM bar
(data/research/lsm_bar.json), prints a price / seed-std / time table per budget, plus the
ensemble (σ/√K) variance projection for the recommended RL config.

Usage: compare_h2h.py
"""
import glob, json, math, statistics as st
from pathlib import Path

R = Path("data/research")


def rl_stats(tag):
    p = []
    for f in sorted(glob.glob(str(R / f"{tag}_seed*.json"))):
        d = json.load(open(f))
        p.append((d["price"], d.get("t_train", float("nan"))))
    if not p:
        return None
    pr = [x[0] for x in p]; tt = [x[1] for x in p]
    return dict(n=len(pr), price=st.mean(pr), std=st.pstdev(pr), t=st.mean(tt))


# RL finalists: (label, tag, budget)
RL = [
    ("RL a2c4    ", "hn1_a2c4",        3072),
    ("RL a2c4+reu", "fin_a2c4_reuse",  3072),
    ("RL a2c3+reu", "hn1_a2c3_reuse",  3072),
    ("RL a3c4    ", "hn1_a3c4",        3072),
    ("RL a2c4    ", "fin_a2c4_2048",   2048),
    ("RL a2c4    ", "fin_a2c4_4096",   4096),
    ("RL a3c4    ", "fin_a3c4_4096",   4096),
]

lsm = {}
lp = R / "lsm_bar.json"
if lp.exists():
    lsm = json.load(open(lp))

print(f"{'method':<13} {'bud':>5} {'price':>9} {'seed-std':>9} {'time(s)':>8}  notes")
print("-" * 64)
# LSM rows
for k, v in sorted(lsm.items(), key=lambda kv: (kv[1]['budget'], kv[1]['state'], kv[1]['degree'])):
    lab = f"LSM {v['state'][:4]} d{v['degree']}"
    reg = f"{v['regressors']}reg" if v.get('regressors') else "full"
    print(f"{lab:<13} {v['budget']:>5} {v['price']:>9.4f} {v['seed_std']:>9.4f} "
          f"{v['t_fit']+v['t_eval']:>8.2f}  {reg}")
print("-" * 64)
# RL rows
for lab, tag, bud in RL:
    s = rl_stats(tag)
    if s:
        print(f"{lab:<13} {bud:>5} {s['price']:>9.4f} {s['std']:>9.4f} {s['t']:>8.2f}  n={s['n']}")

# Ensemble projection for the recommended config (a2c4 @ 3072 if present)
rec = rl_stats("hn1_a2c4") or rl_stats("fin_a2c4_4096")
if rec:
    print("-" * 64)
    print(f"ensemble of K a2c4 agents (price={rec['price']:.4f}, single-σ={rec['std']:.5f}):")
    for K in (4, 9, 16, 25, 43):
        print(f"   K={K:>2}: seed-std≈{rec['std']/math.sqrt(K):.5f}  (wall≈{K}×single if serial)")
