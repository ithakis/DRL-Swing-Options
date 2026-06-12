"""Targeted R2 update: evaluate the 3 v64-32k focal agents and add them to episode_efficiency.csv.
Reuses the LSM price already in the CSV rather than re-fitting LSM from scratch (~5 min save).
"""
import sys, json, contextlib, io, re, glob
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import rebuild_results_v7 as rb
from src.simulate_hhk_spot import simulate_hhk_spot

HERE = ROOT / "Jupyter Notebooks" / "3: Validation 3: Semi-Analytical Kernel & RL Pricer"
CSV  = HERE / "episode_efficiency.csv"

# --- reuse the canonical LSM price that's already in the CSV ---
ee = pd.read_csv(CSV)
lsm5      = float(ee["lsm_M5_price"].iloc[0])
lsm5_ci95 = float(ee["lsm_M5_ci95"].iloc[0])
print(f"Reusing cached LSM M=5 price: {lsm5:.6f}  (ci95={lsm5_ci95:.6f})")

# --- locate the three 32k focal agents ---
V64_LR_A = 3e-4
runs = {}
for jp in glob.glob(str(ROOT / "runs" / "*.json")):
    try:
        d = json.load(open(jp))
    except Exception:
        continue
    if not (abs((d.get("c_cost") or -1) - 0.04) < 1e-9 and
            abs((d.get("gamma_cost") or -1) - 2.0) < 1e-9 and
            d.get("n_paths") == 32768 and
            int(d.get("use_expected_target", 0) or 0) == 1 and
            d.get("actor_layers") == 3 and
            abs((d.get("lr_a") or 0) - V64_LR_A) < 1e-9 and
            d.get("kernel_M_x") == 2):
        continue
    m = re.search(r"_s?(\d+)$", Path(jp).stem)
    seed = int(m.group(1)) if m else d.get("seed")
    if seed is not None and seed not in runs:
        runs[seed] = jp

print(f"Found 32k focal agents: seeds={sorted(runs)}")
if not runs:
    print("ERROR: no 32k focal agents matched — check runs/")
    sys.exit(1)

# --- simulate the common test set ---
params = rb.dotdict(json.load(open(next(iter(runs.values())))))
contract = rb.build_contract(params)
hhk = rb.build_hhk_params(params)
print("Simulating 65536-path test set (seed=999)...")
with contextlib.redirect_stdout(io.StringIO()):
    test = tuple(np.asarray(a, np.float64) for a in
                 simulate_hhk_spot(**hhk, n_paths=65536, seed=999, stratify=True, batch_size=128))

# --- evaluate each agent ---
new_rows = []
for s, jp in sorted(runs.items()):
    params_s = rb.dotdict(json.load(open(jp)))
    with contextlib.redirect_stdout(io.StringIO()):
        agent = rb.build_agent(params_s)
        agent.actor_local.load_state_dict(torch.load(jp[:-5] + ".pth", map_location="cpu"))
        agent.actor_local.eval()
        r = rb.evaluate_rl_on_test_set(agent, contract, test)
    dpct = 100 * (r["test_price"] / lsm5 - 1)
    print(f"  seed={s:>2}: price={r['test_price']:.6f}  Δ%={dpct:+.3f}")
    new_rows.append(dict(method="kernel_on", episodes=32768, M_per_k=1, seed=s,
                         price=r["test_price"], ci95=r["test_CI95"],
                         delta_pct=dpct, bangbang=r["bangbangness"],
                         lsm_M5_price=lsm5, lsm_M5_ci95=lsm5_ci95))
    del agent

# --- drop any stale kernel_on/32768 rows, append fresh ones, write back ---
ee = ee[~((ee["method"] == "kernel_on") & (ee["episodes"] == 32768))]
ee = pd.concat([ee, pd.DataFrame(new_rows)], ignore_index=True)
ee.to_csv(CSV, index=False)

mean_p = np.mean([r["price"] for r in new_rows])
mean_d = np.mean([r["delta_pct"] for r in new_rows])
print(f"\nWrote {len(new_rows)} new 32k rows → {CSV.name}")
print(f"v64 @ 32k: mean price={mean_p:.6f}  mean Δ%={mean_d:+.3f} pp  (vs LSM={lsm5:.6f})")
