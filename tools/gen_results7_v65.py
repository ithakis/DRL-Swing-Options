"""Regenerate 'Convex Costs Results 7.csv' under **v65 (C++ pricer)** — the canonical convex-cost
sweep table that feeds notebooks 4, 5 and NB6's bang-bangness section.

Same schema/protocol as tools/rebuild_results_v7.py, but RL prices come from the C++ v65 pricer
(kernel-on, 4096 episodes) instead of saved PyTorch agents. LSM_full is the unchanged Python
Chebyshev deg-2 full-state benchmark (n_actions=5), via rb.evaluate_lsm_on_test_set so the LSM
columns match the old pipeline exactly. RL OOS = each run's own 65 536-path eval (seed+777).

Grid: c in {0,0.01,0.02,0.04,0.05,0.08,0.10,0.15} x gamma in {1,1.5,2,3} (the 26 sweep cells),
3 seeds (11,12,13); focal c0.04/g2 uses 15 seeds (11-25). For c=0.05 configs we also dump RL eval
parquets (+run JSON) so NB6's threshold-sensitivity cell is v65 too.

Run:  EP11python tools/gen_results7_v65.py            # ~1 hr (LSM-bound)
      RESULTS7_SEEDS=11,12,13 RESULTS7_WORKERS=8 ...
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
import rebuild_results_v7 as rb  # noqa: E402
from src.simulate_hhk_spot import simulate_hhk_spot  # noqa: E402

CPP = ROOT / "cpp_pricer" / "build_v65" / "price_swing"
KERNEL = ROOT / "cpp_pricer" / "data" / "kernel_v64.bin"
N_TRAIN = int(os.environ.get("RESULTS7_NTRAIN", 4096))
N_EVAL = int(os.environ.get("RESULTS7_NEVAL", 65536))
WORKERS = int(os.environ.get("RESULTS7_WORKERS", 8))
BASE_SEEDS = [int(s) for s in os.environ.get("RESULTS7_SEEDS", "11,12,13").split(",")]
FOCAL_SEEDS = list(range(11, 26))   # c0.04/g2 gets 15 seeds
OUT_CSV = ROOT / "Jupyter Notebooks" / "Convex Costs Results 7.csv"
V65 = ["--hidden", "48", "--actor_layers", "2", "--critic_layers", "4", "--hidden_actor", "32",
       "--batch", "64", "--learn_number", "3", "--lr_c", "5e-4", "--threads", "2", "--quiet"]

CS = [0.0, 0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15]
GAMMAS = [1.0, 1.5, 2.0, 3.0]
# the published 26-cell grid: nocost only at gamma=1; c>0 has all 4 gammas (drop c0 g!=1)
GRID = [(c, g) for c in CS for g in GAMMAS if not (c == 0.0 and g != 1.0)]


def cfg_name(c, gamma):
    gs = ("%g" % gamma)
    return f"SwingOption_20_c{c:.2f}_gamma{gs}"


def mk_params(c, gamma):
    return rb.dotdict(dict(q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0, maturity=0.0833,
        n_rights=22, risk_free_rate=0.05, min_refraction_periods=0, c_cost=c, gamma_cost=gamma,
        S0=1.0, alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3))


def cpp_run(c, gamma, seed, trace_blob=None, sxy_blob=None):
    cmd = [str(CPP), "--seed", str(seed), "--n_train", str(N_TRAIN), "--n_eval", str(N_EVAL),
           "--c_cost", str(c), "--gamma_cost", str(gamma), "--kernel", str(KERNEL), *V65]
    if trace_blob:
        cmd += ["--trace", trace_blob, "--dump_eval_paths", sxy_blob]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT / "cpp_pricer"))
    if out.returncode != 0:
        raise RuntimeError(f"cpp failed (c={c},g={gamma},s={seed}): {out.stderr[-400:]}")
    j = json.loads(out.stdout)
    return j["price"], j["bangbang"]


def dump_rl_parquet(trace_blob, sxy_blob, run_name, seed, c, gamma):
    """For NB6 cell-29 sensitivity: write logs/<run>/evaluations/rl_episode_<N>.parquet + run JSON."""
    with open(sxy_blob, "rb") as f:
        magic, n, T = struct.unpack("<iii", f.read(12)); buf = np.fromfile(f, dtype="<f4")
    S = buf[:n * T].reshape(n, T).astype(np.float64)
    with open(trace_blob, "rb") as f:
        struct.unpack("<iii", f.read(12)); tb = np.fromfile(f, dtype="<f4")
    sz = n * T
    q, reward, cost, gross = (tb[i * sz:(i + 1) * sz].reshape(n, T).astype(np.float64) for i in range(4))
    q_excl = np.cumsum(q, axis=1) - q
    Qmax = 20.0
    df = pd.DataFrame({
        "path": np.repeat(np.arange(n), T), "time_step": np.tile(np.arange(T), n),
        "spot": S.reshape(-1), "q_t": q.reshape(-1), "reward": reward.reshape(-1),
        "exercise_cost": cost.reshape(-1), "payoff": gross.reshape(-1), "payoff_gross": gross.reshape(-1),
        "q_exercised_so_far": q_excl.reshape(-1), "q_remaining_norm": ((Qmax - q_excl) / Qmax).reshape(-1),
        "q_exercised_norm": (q_excl / Qmax).reshape(-1),
    })
    pq = ROOT / "logs" / run_name / "evaluations" / f"rl_episode_{N_TRAIN}.parquet"
    pq.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(pq, index=False)
    with open(ROOT / "runs" / f"{run_name}.json", "w") as fh:
        json.dump({**dict(mk_params(c, gamma)), "n_paths": N_TRAIN}, fh, indent=2)


def main():
    assert CPP.exists(), f"build the v65 binary first: {CPP}"
    (ROOT / "logs" / "lsm_full_state").mkdir(parents=True, exist_ok=True)

    # ---- RL phase: all runs in parallel ----
    jobs = []
    for c, gamma in GRID:
        seeds = FOCAL_SEEDS if (abs(c - 0.04) < 1e-9 and gamma == 2.0) else BASE_SEEDS
        for s in seeds:
            jobs.append((c, gamma, s))
    print(f"v65 Results-7: {len(GRID)} configs, {len(jobs)} RL runs @ {N_TRAIN}ep, {WORKERS} workers")

    rl = {}   # (c,gamma,seed) -> (price, bb)
    tmp = ROOT / "cpp_pricer" / "data" / "_r7tmp"; tmp.mkdir(exist_ok=True)

    def run_one(job):
        c, gamma, s = job
        is05 = abs(c - 0.05) < 1e-9
        tb = str(tmp / f"{cfg_name(c,gamma)}_{s}_t.bin") if is05 else None
        sx = str(tmp / f"{cfg_name(c,gamma)}_{s}_s.bin") if is05 else None
        price, bb = cpp_run(c, gamma, s, tb, sx)
        if is05:
            dump_rl_parquet(tb, sx, f"{cfg_name(c,gamma)}_{s}", s, c, gamma)
        return job, price, bb

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for job, price, bb in ex.map(run_one, jobs):
            rl[job] = (price, bb); done += 1
            if done % 10 == 0:
                print(f"  RL {done}/{len(jobs)}")
    print("  RL phase done.")

    # ---- LSM + assembly (sequential) ----
    rows = []
    for c, gamma in GRID:
        cfg = cfg_name(c, gamma)
        params = mk_params(c, gamma); contract = rb.build_contract(params); hhk = rb.build_hhk_params(params)
        with contextlib.redirect_stdout(io.StringIO()):
            test = simulate_hhk_spot(**hhk, n_paths=N_EVAL, seed=999, stratify=True, batch_size=128)
            lsm = rb.evaluate_lsm_on_test_set(contract, hhk, test_dataset=test,
                                              parquet_dir=str(ROOT / "logs" / "lsm_full_state"),
                                              config_name=cfg)
        seeds = FOCAL_SEEDS if (abs(c - 0.04) < 1e-9 and gamma == 2.0) else BASE_SEEDS
        prices = {s: rl[(c, gamma, s)][0] for s in seeds}
        bbs = [rl[(c, gamma, s)][1] for s in seeds if not np.isnan(rl[(c, gamma, s)][1])]
        parr = np.array(list(prices.values()))
        rl_mean = float(parr.mean()); rl_std = float(parr.std(ddof=1)) if len(parr) > 1 else 0.0
        rl_best = float(parr.max()); rl_best_seed = max(prices, key=prices.get)
        lp = lsm["lsm_price"]
        row = {"Configuration": cfg, "c": c, "gamma": gamma, "LSM_full": lp,
               "LSM_full_CI95": lsm["lsm_CI95"]}
        for s in sorted(prices): row[f"RL_seed{s}"] = prices[s]
        row.update({"RL_mean": rl_mean, "RL_std": rl_std, "RL_best": rl_best,
                    "RL_best_seed": int(rl_best_seed),
                    "PctDiff_mean": 100.0 * (rl_mean - lp) / lp,
                    "PctDiff_best": 100.0 * (rl_best - lp) / lp,
                    "PctDiff_CI95": (100.0 * (1.96 * rl_std / np.sqrt(len(parr))) / lp) if len(parr) > 1 else float("nan"),
                    "RL_BangBangness_mean": float(np.mean(bbs)) if bbs else float("nan"),
                    "LSM_BangBangness": lsm["lsm_bangbangness"]})
        rows.append(row)
        print(f"{cfg:<32} LSM={lp:.4f} RL={rl_mean:.4f} ({100*(rl_mean/lp-1):+.2f}%) BB={row['RL_BangBangness_mean']:.3f}")

    df = pd.DataFrame(rows)
    # match the original column order (15 seed columns, NaN where absent)
    if OUT_CSV.exists():
        cols = list(pd.read_csv(OUT_CSV, nrows=0).columns)
        df = df.reindex(columns=cols)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
