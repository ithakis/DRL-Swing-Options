"""Regenerate Notebook 6 (Convex costs 0.04 Analysis) artifacts under **v65 (C++ pricer)**.

For each representative (config, seed) NB6 cell expects:
  * runs/<run>.json                                  — contract/HHK config
  * logs/<run>/evaluations/rl_episode_<step>.parquet — RL per-(path,step) trace
  * logs/lsm_full_state/<config_base>_lsm.parquet    — LSM-D per-(path,step) trace

RL comes from the C++ v65 pricer (kernel-on), which also dumps its OOS eval paths so the Python
LSM-D prices the IDENTICAL test set => the per-path RL-vs-LSM merge in NB6 is valid. LSM-D is the
canonical Chebyshev deg-2 full-state benchmark (n_actions=5).

Run:
    EP11python tools/gen_nb6_v65.py            # default N_EVAL, n_train=4096
    NB6_NEVAL=16384 ... to change OOS size
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)

import rebuild_results_v7 as rb  # noqa: E402
from src.simulate_hhk_spot import simulate_hhk_spot  # noqa: E402
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos  # noqa: E402

CPP = ROOT / "cpp_pricer" / "build_v65" / "price_swing"
KERNEL = ROOT / "cpp_pricer" / "data" / "kernel_v64.bin"
N_EVAL = int(os.environ.get("NB6_NEVAL", 16384))
N_TRAIN = int(os.environ.get("NB6_NTRAIN", 4096))
LSM_TRAIN_N = int(os.environ.get("NB6_LSM_TRAIN", 16384))
THREADS = int(os.environ.get("NB6_THREADS", "8"))
LSM_KW = dict(poly_degree=2, basis_type="chebyshev", state_mode="full",
              reg_type="none", reg_alpha=1e-6, n_actions=5)
V65 = ["--hidden", "48", "--actor_layers", "2", "--critic_layers", "4",
       "--hidden_actor", "32", "--batch", "64", "--learn_number", "3", "--lr_c", "5e-4",
       "--threads", str(THREADS), "--quiet"]

# (label, c, gamma, seed) — the four NB6 representative configs, now v65.
CONFIGS = [
    ("nocost", 0.0, 1.0, 13),
    ("g1",     0.04, 1.0, 11),
    ("g15",    0.04, 1.5, 12),
    ("g2",     0.04, 2.0, 13),
]


def gname(c, gamma):
    gs = ("%g" % gamma).replace(".", "p") if gamma != int(gamma) else str(int(gamma))
    return f"SwingOption_20_c{c:.2f}_gamma{gs}_v65"


def read_sxy(path):
    with open(path, "rb") as f:
        magic, n, T = struct.unpack("<iii", f.read(12))
        assert magic == 0x53585950, f"bad SXYP magic {magic:#x}"
        buf = np.fromfile(f, dtype="<f4")
    sz = n * T
    S, X, Y = (buf[i * sz:(i + 1) * sz].reshape(n, T).astype(np.float64) for i in range(3))
    return n, T, S, X, Y


def trace_to_parquet(blob, out_parquet, spot_S, Q_max):
    """Build the RL eval parquet. `spot` comes from the dumped eval paths; `q_remaining_norm`
    and `q_exercised_so_far` are derived from the cumulative exercised quantity (pre-decision),
    so the parquet matches the v64 run.py RL eval schema NB6 expects."""
    with open(blob, "rb") as f:
        magic, n, T = struct.unpack("<iii", f.read(12))
        assert magic == 0x53575452, f"bad SWTR magic {magic:#x}"
        buf = np.fromfile(f, dtype="<f4")
    sz = n * T
    q, reward, cost, gross = (buf[i * sz:(i + 1) * sz].reshape(n, T).astype(np.float64) for i in range(4))
    q_excl = np.cumsum(q, axis=1) - q                  # pre-decision q_exercised_so_far
    q_rem_norm = (Q_max - q_excl) / Q_max
    paths = np.repeat(np.arange(n, dtype=np.int32), T)
    steps = np.tile(np.arange(T, dtype=np.int32), n)
    df = pd.DataFrame({
        "path": paths, "time_step": steps, "spot": spot_S.reshape(-1),
        "q_t": q.reshape(-1), "reward": reward.reshape(-1),
        "exercise_cost": cost.reshape(-1), "payoff": gross.reshape(-1),
        "payoff_gross": gross.reshape(-1),
        "q_exercised_so_far": q_excl.reshape(-1), "q_remaining_norm": q_rem_norm.reshape(-1),
    })
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return float(df.groupby("path")["reward"].sum().mean())


def main():
    assert CPP.exists(), f"build the v65 binary first: {CPP}"
    (ROOT / "logs" / "lsm_full_state").mkdir(parents=True, exist_ok=True)
    specs = []
    for label, c, gamma, seed in CONFIGS:
        base = gname(c, gamma)
        run = f"{base}_{seed}"
        print(f"\n=== {label}: {run} (n_train={N_TRAIN}, n_eval={N_EVAL}) ===")
        params = rb.dotdict(dict(
            q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0, maturity=0.0833,
            n_rights=22, risk_free_rate=0.05, min_refraction_periods=0, c_cost=c, gamma_cost=gamma,
            S0=1.0, alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3, seed=seed))
        # --- config JSON for NB6 (needs maturity, n_rights, risk_free_rate, etc.) ---
        cfg = dict(params); cfg["n_paths"] = N_TRAIN
        (ROOT / "runs").mkdir(exist_ok=True)
        with open(ROOT / "runs" / f"{run}.json", "w") as fh:
            json.dump(cfg, fh, indent=2)

        # --- C++ v65 RL: trace + eval-paths dump ---
        blob = f"/tmp/nb6_{run}_trace.bin"
        sxy = f"/tmp/nb6_{run}_sxy.bin"
        cmd = [str(CPP), "--seed", str(seed), "--n_train", str(N_TRAIN), "--n_eval", str(N_EVAL),
               "--c_cost", str(c), "--gamma_cost", str(gamma), "--kernel", str(KERNEL),
               "--trace", blob, "--dump_eval_paths", sxy, *V65]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT / "cpp_pricer"))
        if out.returncode != 0:
            raise RuntimeError(f"cpp failed: {out.stderr[-500:]}")
        jr = json.loads(out.stdout)
        # --- eval paths (shared with LSM) ---
        contract = rb.build_contract(params)
        hhk = rb.build_hhk_params(params)
        n, T, S, X, Y = read_sxy(sxy)
        rl_pq = ROOT / "logs" / run / "evaluations" / f"rl_episode_{N_TRAIN}.parquet"
        rl_price = trace_to_parquet(blob, rl_pq, S, contract.Q_max)
        print(f"  RL  price (json) {jr['price']:.4f} | trace mean {rl_price:.4f} | bb {jr['bangbang']:.3f}")

        dt = contract.maturity / (contract.n_rights - 1)
        t_grid = np.arange(T, dtype=np.float64) * dt
        with contextlib.redirect_stdout(io.StringIO()):
            train = tuple(np.asarray(a, np.float64) for a in
                          simulate_hhk_spot(**hhk, n_paths=LSM_TRAIN_N, seed=998, stratify=True, batch_size=128))
            est = fit_lsm_estimators(contract=contract, dataset=train, **LSM_KW)
            lsm_pq = ROOT / "logs" / "lsm_full_state" / f"{base}_lsm.parquet"
            lsm_price, _ = price_swing_option_lsm_oos(
                contract=contract, dataset=(t_grid, S, X, Y), estimators=est, seed=999,
                csv_path=str(lsm_pq), _print_results=False)
        print(f"  LSM price {lsm_price:.4f} | Δ% RL vs LSM {100*(jr['price']/lsm_price-1):+.2f}")
        specs.append(dict(label=f"{label}_seed{seed}", c=c, gamma=gamma, seed=seed,
                          step=N_TRAIN, run_name=run))

    print("\n# NB6 RUN_SPECS (paste into cell 2):")
    print("RUN_SPECS = [")
    for s in specs:
        print(f'    {{"label": "{s["label"]}", "c": {s["c"]}, "gamma": {s["gamma"]}, '
              f'"seed": {s["seed"]}, "step": {s["step"]}, "run_name": "{s["run_name"]}"}},')
    print("]")


if __name__ == "__main__":
    main()
