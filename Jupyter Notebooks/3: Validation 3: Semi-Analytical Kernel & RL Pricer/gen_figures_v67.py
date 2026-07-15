"""A3 — regenerate the §5 exercise-behavior figure inputs under v67 (AC-kernel policy).

For the 4 figure cells (nocost, g1=0.04/1, g15=0.04/1.5, g2=0.04/2) at seed 11, produces RL + LSM
per-(path,step) parquets in the EXACT v65 schema NB6 consumes, but from the v67 kernel build:
  * RL  = build_v67_kernel/price_swing (KON_BALANCED) --trace + --dump_eval_paths, joined so the
          trace (q/reward/cost/gross) carries `spot` from the OOS SXYP dump (same eval_paths).
  * LSM = src/lsm_swing_pricer priced on the IDENTICAL OOS (csv_path -> parquet, native schema).

Writes to logs/<v67_run>/evaluations/rl_episode_4096.parquet and
logs/lsm_full_state/<v67_config>_lsm.parquet so NB6 (cell 2 RUN_SPECS repointed) regenerates
figs/convex_costs_0p04/{hist_exercise,spot_income_pv_hist}.* unchanged.

Run:  python gen_figures_v67.py
"""
import os
os.environ.update(VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TQDM_DISABLE="1")
import contextlib
import io
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_rl_validation_v67 as g
import rebuild_results_v7 as rb
from src.simulate_hhk_spot import simulate_hhk_spot
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos

SEED = int(os.environ.get("FIG_SEED", 11))
N_EVAL = int(os.environ.get("FIG_NEVAL", 16384))   # match v65 figure path count
N_FIT = 32768                                       # LSM fit paths (match Table 5)
CELLS = [("nocost", 0.0, 1.0, "1"), ("g1", 0.04, 1.0, "1"),
         ("g15", 0.04, 1.5, "1p5"), ("g2", 0.04, 2.0, "2")]
LOGS = g.ROOT / "logs"
LSM_DIR = LOGS / "lsm_full_state"


def read_swtr(path):
    """C++ --trace blob: int32 magic('SWTR'),n,T, then float32[n*T] q,reward,cost,gross."""
    with open(path, "rb") as f:
        magic, n, T = struct.unpack("<iii", f.read(12))
        assert magic == 0x53575452, f"bad SWTR magic {magic:#x}"
        buf = np.fromfile(f, dtype="<f4")
    sz = n * T
    q, reward, cost, gross = (buf[i * sz:(i + 1) * sz].reshape(n, T) for i in range(4))
    return n, T, q, reward, cost, gross


def build_contract_hhk(c, gamma):
    p = rb.dotdict(dict(q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0, maturity=0.0833,
        n_rights=22, risk_free_rate=0.05, min_refraction_periods=0, c_cost=c, gamma_cost=gamma,
        S0=1.0, alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3))
    return rb.build_contract(p), rb.build_hhk_params(p), p


def rl_parquet(swtr_path, sxy_path, Q_max=20.0):
    """Join trace (q/reward/cost/gross) + OOS spot -> the 11-col v65 RL schema."""
    n, T, q, reward, cost, gross = read_swtr(swtr_path)
    S, _, _ = g.read_sxy(sxy_path)
    S = S[:n, :T]
    qcum_before = np.concatenate([np.zeros((n, 1)), np.cumsum(q, axis=1)[:, :-1]], axis=1)  # inventory entering step
    paths = np.repeat(np.arange(n, dtype=np.int32), T)
    steps = np.tile(np.arange(T, dtype=np.int32), n)
    return pd.DataFrame({
        "path": paths, "time_step": steps, "spot": S.reshape(-1),
        "q_t": q.reshape(-1), "reward": reward.reshape(-1), "exercise_cost": cost.reshape(-1),
        "payoff": gross.reshape(-1), "payoff_gross": gross.reshape(-1),
        "q_exercised_so_far": qcum_before.reshape(-1),
        "q_remaining_norm": ((Q_max - qcum_before) / Q_max).reshape(-1),
        "q_exercised_norm": (qcum_before / Q_max).reshape(-1),
    })


def main():
    assert g.KERNEL_BIN.exists()
    LSM_DIR.mkdir(parents=True, exist_ok=True)
    specs = []
    for tag, c, gamma, glabel in CELLS:
        run_name = f"SwingOption_20_c{c:.2f}_gamma{glabel}_v67_{SEED}"
        config_base = run_name.rsplit("_", 1)[0]
        evdir = LOGS / run_name / "evaluations"; evdir.mkdir(parents=True, exist_ok=True)
        swtr = g.CPP_DIR / "data" / f"_fig_{tag}.swtr"
        sxy = g.CPP_DIR / "data" / f"_fig_{tag}.sxyp"
        # 1) RL: v67 kernel build with trace + OOS dump (same eval_paths)
        cmd = [str(g.KERNEL_BIN), "--seed", str(SEED), "--n_train", "4096", "--n_eval", str(N_EVAL),
               "--c_cost", str(c), "--gamma_cost", str(gamma), "--threads", "8", "--quiet",
               *g.KON_BALANCED, "--kernel", str(g.KERNEL),
               "--trace", str(swtr), "--dump_eval_paths", str(sxy)]
        env = {**os.environ, "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(g.CPP_DIR), env=env)
        if r.returncode != 0:
            raise RuntimeError(f"cpp failed for {tag}: {r.stderr[-400:]}")
        rl_df = rl_parquet(swtr, sxy)
        rl_path = evdir / "rl_episode_4096.parquet"
        rl_df.to_parquet(rl_path, index=False)
        # 2) LSM on the IDENTICAL OOS (csv_path -> parquet, native schema)
        contract, hhk, _ = build_contract_hhk(c, gamma)
        with contextlib.redirect_stdout(io.StringIO()):
            train = simulate_hhk_spot(**hhk, n_paths=N_FIT, seed=700, stratify=True, batch_size=128)
            est = fit_lsm_estimators(contract=contract, dataset=train, poly_degree=2,
                                     basis_type="chebyshev", state_mode="full",
                                     reg_type="none", reg_alpha=1e-6, n_actions=5)
            S, X, Y = g.read_sxy(sxy)
            lsm_path = LSM_DIR / f"{config_base}_lsm.parquet"
            price_swing_option_lsm_oos(contract=contract, dataset=(0, S, X, Y), estimators=est,
                                       seed=12345, csv_path=str(lsm_path), _print_results=False)
        rl_price = rl_df.groupby("path")["reward"].sum().mean()
        print(f"[{tag}] RL PV(seed{SEED})={rl_price:.4f}  rl={rl_path.relative_to(g.ROOT)}  "
              f"lsm={lsm_path.relative_to(g.ROOT)}", flush=True)
        specs.append(dict(label=f"{tag}_seed{SEED}", c=c, gamma=gamma, seed=SEED, step=4096,
                          run_name=run_name))
        swtr.unlink(missing_ok=True); sxy.unlink(missing_ok=True)
    print("\nRUN_SPECS for NB6 cell 2:")
    for s in specs:
        print(f"    {s},")


if __name__ == "__main__":
    main()
