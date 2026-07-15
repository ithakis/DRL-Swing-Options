"""A3-remainder — v67 bang-bangness sweep for Fig 4 (cell 27) + Appendix A (cell 29).

Fig 4 cells (c in {0.01,0.02,0.04,0.08,0.15} x gamma in {1,1.5,2}) are swept over 8 seeds {11..18}
and reported as mean +/- SE (rigor upgrade to match Table 5). Appendix A (c=0.05 x gamma) stays
single-seed 11 (threshold-sensitivity is a single-run diagnostic) and its RL parquets are saved.

Per (cell, seed): v67 kernel --trace + OOS -> RL parquet; LSM-D priced on the same OOS -> LSM parquet.
B = P(q_t/q_max >= 0.95 | feasible (remaining >= q_max), q_t>0) -- paper's metric, RL & LSM identical.

Output: bangbang_v67.csv (Configuration, c, gamma, RL_BangBangness_mean/se, LSM_BangBangness/se,
RL_best_seed, n_seeds). Run: python gen_bangbang_v67.py
"""
import os
os.environ.update(VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TQDM_DISABLE="1")
import contextlib
import io
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_rl_validation_v67 as g
from gen_figures_v67 import rl_parquet, build_contract_hhk
from src.simulate_hhk_spot import simulate_hhk_spot
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos

N_EVAL, N_FIT, Q_MAX, Q_MAX_STEP = 16384, 32768, 20.0, 2.0
SEEDS = [11, 12, 13, 14, 15, 16, 17, 18]
GAMMAS = [(1.0, "1"), (1.5, "1p5"), (2.0, "2")]
COSTS_FIG4 = [0.01, 0.02, 0.04, 0.08, 0.15]    # Fig 4: 8-seed mean +/- SE
COSTS_APPA = [0.05]                            # Appendix A: single seed 11, parquets saved
LOGS = g.ROOT / "logs"
OUT = g.HERE / "bangbang_v67.csv"


def bangbang(df, method):
    q_before = (df["q_remaining_norm"].values * Q_MAX if method == "RL"
                else Q_MAX - df["q_exercised_so_far"].values)
    cond = (df["q_t"].values > 1e-5) & (q_before >= Q_MAX_STEP - 1e-5)
    if cond.sum() == 0:
        return np.nan
    return float(((df["q_t"].values[cond] / Q_MAX_STEP) >= 0.95).mean())


def run_cell(args):
    c, gv, gl, seed, save = args
    tag = f"_bb_{c}_{gl}_{seed}"
    swtr = g.CPP_DIR / "data" / f"{tag}.swtr"
    sxy = g.CPP_DIR / "data" / f"{tag}.sxyp"
    cmd = [str(g.KERNEL_BIN), "--seed", str(seed), "--n_train", "4096", "--n_eval", str(N_EVAL),
           "--c_cost", str(c), "--gamma_cost", str(gv), "--threads", "2", "--quiet",
           *g.KON_BALANCED, "--kernel", str(g.KERNEL), "--trace", str(swtr), "--dump_eval_paths", str(sxy)]
    env = {**os.environ, "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(g.CPP_DIR), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"cpp failed ({c},{gv},s{seed}): {r.stderr[-300:]}")
    rl_df = rl_parquet(swtr, sxy)
    contract, hhk, _ = build_contract_hhk(c, gv)
    run_name = f"SwingOption_20_c{c:.2f}_gamma{gl}_v67_{seed}"
    # SEED-UNIQUE temp LSM parquet -> no race across the 8 concurrent seeds of a cell, and no
    # collision with the figure LSM parquets in logs/lsm_full_state.
    lsm_path = g.CPP_DIR / "data" / f"_bblsm_{c}_{gl}_{seed}.parquet"
    k = SEEDS.index(seed) if seed in SEEDS else 0
    with contextlib.redirect_stdout(io.StringIO()):
        train = simulate_hhk_spot(**hhk, n_paths=N_FIT, seed=700 + k, stratify=True, batch_size=128)
        est = fit_lsm_estimators(contract=contract, dataset=train, poly_degree=2, basis_type="chebyshev",
                                 state_mode="full", reg_type="none", reg_alpha=1e-6, n_actions=5)
        S, X, Y = g.read_sxy(sxy)
        price_swing_option_lsm_oos(contract=contract, dataset=(0, S, X, Y), estimators=est,
                                   seed=12345, csv_path=str(lsm_path), _print_results=False)
    lsm_df = pd.read_parquet(lsm_path)
    lsm_path.unlink(missing_ok=True)
    rl_b, lsm_b = bangbang(rl_df, "RL"), bangbang(lsm_df, "LSM")
    if save:   # Appendix A needs the c=0.05 RL parquet + config on disk
        evdir = LOGS / run_name / "evaluations"; evdir.mkdir(parents=True, exist_ok=True)
        rl_df.to_parquet(evdir / "rl_episode_4096.parquet", index=False)
        cfg = dict(c_cost=c, gamma_cost=gv, strike=1.0, maturity=0.0833, n_rights=22, risk_free_rate=0.05,
                   q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, S0=1.0, alpha=12.0, sigma=1.2,
                   beta=150.0, lam=6.0, mu_J=0.3)
        (g.ROOT / "runs" / f"{run_name}.json").write_text(json.dumps(cfg, indent=2))
    swtr.unlink(missing_ok=True); sxy.unlink(missing_ok=True)
    return (c, gv, gl, seed, rl_b, lsm_b)


def main():
    t0 = time.time()
    jobs = [(c, gv, gl, s, False) for c in COSTS_FIG4 for gv, gl in GAMMAS for s in SEEDS]
    jobs += [(c, gv, gl, 11, True) for c in COSTS_APPA for gv, gl in GAMMAS]
    print(f"v67 bang-bangness | {len(jobs)} runs (Fig4 {len(COSTS_FIG4)}x{len(GAMMAS)}x{len(SEEDS)} "
          f"+ AppA {len(COSTS_APPA)}x{len(GAMMAS)})", flush=True)
    res = {}   # (c,gl) -> list of (rl_b, lsm_b)
    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for c, gv, gl, seed, rl_b, lsm_b in ex.map(run_cell, jobs):
            res.setdefault((c, gv, gl), []).append((rl_b, lsm_b))
            done += 1
            if done % 8 == 0:
                print(f"  {done}/{len(jobs)} [{time.time()-t0:.0f}s]", flush=True)
    rows = []
    for (c, gv, gl), vals in res.items():
        rl = np.array([v[0] for v in vals]); lsm = np.array([v[1] for v in vals])
        n = len(vals)
        se = lambda a: float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append(dict(Configuration=f"SwingOption_20_c{c:.2f}_gamma{gl}", c=c, gamma=gv,
                         RL_BangBangness_mean=float(rl.mean()), RL_BangBangness_se=se(rl),
                         LSM_BangBangness=float(lsm.mean()), LSM_BangBangness_se=se(lsm),
                         RL_best_seed=11, n_seeds=n))
    pd.DataFrame(rows).sort_values(["c", "gamma"]).to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(rows)} cells) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
