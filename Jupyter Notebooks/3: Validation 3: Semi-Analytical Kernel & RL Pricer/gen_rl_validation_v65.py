"""Validation 3 - Part II under **v65 (C++ pricer)**.

Same protocol/CSV schemas as gen_rl_validation.py, but the RL prices come from the C++ v65
pricer (cpp_pricer/build_v65/price_swing) instead of saved PyTorch agents — so the RL vs LSM
comparison is on the v65 architecture (swish-β3 + a2c4/w48/actor32 + b64/ln3). LSM-D (M=5,9,17)
is unchanged (Python, src/lsm_swing_pricer), giving an apples-to-apples three-way comparison.

Each (regime, seed) RL run trains 0->n_train and evaluates on its own fresh 65 536-path OOS set
(eval seed = train seed + 777, identical for kernel_on vs kernel_off so on/off is paired per seed).
LSM-D is fit on seed 998 and evaluated on seed 999, as in the canonical pipeline.

Produces:
  rl_lsm_pricing.csv     R1 - kernel_on / kernel_off / lsm_M{5,9,17}, per regime & seed
  episode_efficiency.csv R2 - focal g2 kernel-on at episodes {2048,4096,8192} + delta% vs LSM(M=5)

Run (from anywhere):
    EP11python gen_rl_validation_v65.py                 # full N=65536
    VAL3_NTEST=16384 EP11python gen_rl_validation_v65.py  # quick pass
    VAL3_SEEDS=11,12,13,14,15 ... to widen seeds
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)

import rebuild_results_v7 as rb  # noqa: E402
from src.simulate_hhk_spot import simulate_hhk_spot  # noqa: E402
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos  # noqa: E402

# ---- config -----------------------------------------------------------------
N = int(os.environ.get("VAL3_NTEST", 65536))     # LSM test & train size
N_EVAL = int(os.environ.get("VAL3_NEVAL", 65536)) # C++ RL OOS eval size
TEST_SEED, TRAIN_SEED = 999, 998
LSM_DEGREE, LSM_BASIS = 2, "chebyshev"
LSM_MS = [5, 9, 17]
SEEDS = [int(s) for s in os.environ.get("VAL3_SEEDS", "11,12,13").split(",")]
THREADS = int(os.environ.get("VAL3_THREADS", "8"))

CPP = ROOT / "cpp_pricer" / "build_v65" / "price_swing"
KERNEL = ROOT / "cpp_pricer" / "data" / "kernel_v64.bin"
# canonical v65 run flags (HPT v65 "balanced"): swish-β3 is compiled into build_v65.
V65 = ["--hidden", "48", "--actor_layers", "2", "--critic_layers", "4",
       "--hidden_actor", "32", "--batch", "64", "--learn_number", "3", "--lr_c", "5e-4",
       "--threads", str(THREADS), "--quiet"]

REGIMES = [("nocost", 0.0, 1.0), ("g1", 0.04, 1.0), ("g15", 0.04, 1.5), ("g2", 0.04, 2.0)]


# ---- RL via the C++ v65 binary ---------------------------------------------
def cpp_run(c, gamma, seed, n_train, kernel_on):
    """Train+eval one v65 agent; return dict(price, ci95, bangbang, avg_exercised)."""
    cmd = [str(CPP), "--seed", str(seed), "--n_train", str(n_train), "--n_eval", str(N_EVAL),
           "--c_cost", str(c), "--gamma_cost", str(gamma), *V65]
    cmd += (["--kernel", str(KERNEL)] if kernel_on else ["--kernel_off"])
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT / "cpp_pricer"))
    if out.returncode != 0:
        raise RuntimeError(f"cpp failed: {' '.join(cmd)}\n{out.stderr[-500:]}")
    j = json.loads(out.stdout)
    return dict(price=j["price"], ci95=j["ci95"], bangbang=j["bangbang"],
                avg_exercised=j["avg_exercised"])


# ---- contract / HHK params (focal SwingOption_20) ---------------------------
def mk_params(c, gamma):
    return rb.dotdict(dict(
        q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0, maturity=0.0833,
        n_rights=22, risk_free_rate=0.05, min_refraction_periods=0, c_cost=c, gamma_cost=gamma,
        S0=1.0, alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3))


# ---- LSM-D context per regime (Python, cached) ------------------------------
_CTX: dict = {}


def context(key, c, gamma):
    if key in _CTX:
        return _CTX[key]
    params = mk_params(c, gamma)
    contract = rb.build_contract(params)
    hhk = rb.build_hhk_params(params)
    with contextlib.redirect_stdout(io.StringIO()):
        test = tuple(np.asarray(a, np.float64) for a in
                     simulate_hhk_spot(**hhk, n_paths=N, seed=TEST_SEED, stratify=True, batch_size=128))
        train = tuple(np.asarray(a, np.float64) for a in
                      simulate_hhk_spot(**hhk, n_paths=N, seed=TRAIN_SEED, stratify=True, batch_size=128))
    lsm = {}
    for M in LSM_MS:
        t0 = time.time()
        pq = HERE / "_lsm_parquet" / f"{key}_M{M}.parquet"
        pq.parent.mkdir(exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            est = fit_lsm_estimators(contract=contract, dataset=train, poly_degree=LSM_DEGREE,
                                     basis_type=LSM_BASIS, state_mode="full", reg_type="none",
                                     reg_alpha=1e-6, n_actions=M)
            price, (lo, hi) = price_swing_option_lsm_oos(contract=contract, dataset=test,
                                                         estimators=est, seed=TEST_SEED + 1,
                                                         csv_path=str(pq), _print_results=False)
        bb = float("nan")
        if pq.exists():
            df = pd.read_parquet(pq)
            ex = df[df["q_t"] > 1e-6]
            if len(ex):
                bb = float((ex["q_t"] >= 0.95 * contract.q_max).sum() / len(ex))
        lsm[M] = dict(price=price, ci95=(hi - lo) / 2.0, bangbang=bb)
        print(f"      LSM M={M:>2}: {price:.4f} (bb={bb:.3f})  [{time.time()-t0:.0f}s]")
    _CTX[key] = dict(contract=contract, hhk=hhk, lsm=lsm)
    return _CTX[key]


# ---- R1: pricing table ------------------------------------------------------
def gen_R1():
    rows = []
    for key, c, gamma in REGIMES:
        print(f"[R1] {key} (c={c}, gamma={gamma})")
        ctx = context(key, c, gamma)
        for M, r in ctx["lsm"].items():
            rows.append(dict(regime=key, c=c, gamma=gamma, method=f"lsm_M{M}", seed=-1,
                             episodes=np.nan, price=r["price"], ci95=r["ci95"],
                             bangbang=r["bangbang"], M_x=np.nan, eval_wall=np.nan))
        for method, kon in [("kernel_on", True), ("kernel_off", False)]:
            for s in SEEDS:
                t0 = time.time()
                r = cpp_run(c, gamma, s, 4096, kon)
                rows.append(dict(regime=key, c=c, gamma=gamma, method=method, seed=s, episodes=4096,
                                 price=r["price"], ci95=r["ci95"], bangbang=r["bangbang"],
                                 M_x=(2 if kon else np.nan), eval_wall=time.time() - t0))
            mp = np.mean([x["price"] for x in rows if x["method"] == method and x["regime"] == key])
            print(f"      {method}: mean price={mp:.4f}")
    return pd.DataFrame(rows)


# ---- R2: episode efficiency (focal g2) -------------------------------------
def gen_R2():
    ctx = context("g2", 0.04, 2.0)
    lsm5 = ctx["lsm"][5]["price"]
    rows = []
    for ep in [2048, 4096, 8192]:
        for s in SEEDS:
            r = cpp_run(0.04, 2.0, s, ep, True)
            rows.append(dict(method="kernel_on", episodes=ep, M_per_k=1, seed=s, price=r["price"],
                             ci95=r["ci95"], delta_pct=100 * (r["price"] / lsm5 - 1),
                             bangbang=r["bangbang"]))
        print(f"[R2] ep={ep}: seeds={SEEDS}")
    df = pd.DataFrame(rows)
    df["lsm_M5_price"] = lsm5
    df["lsm_M5_ci95"] = ctx["lsm"][5]["ci95"]
    return df


def main():
    t0 = time.time()
    print(f"v65 C++ generator | N(test=train)={N} N_EVAL={N_EVAL} seeds={SEEDS}\n")
    assert CPP.exists(), f"build the v65 binary first: {CPP}"
    gen_R1().to_csv(HERE / "rl_lsm_pricing.csv", index=False)
    gen_R2().to_csv(HERE / "episode_efficiency.csv", index=False)
    print(f"\nDone in {time.time() - t0:.0f}s. CSVs in {HERE}")


if __name__ == "__main__":
    main()
