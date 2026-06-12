"""Validation 3 - Part II: RL pricer validation by forward rollout on a common
fresh Monte-Carlo test set, vs a strengthened full-state LSM-D baseline (M in {5,9,17}).

Restricted (per agreement) to the 4 regimes with rich saved kernel-on agents:
  nocost (c=0, gamma=1) and c=0.04 at gamma in {1, 1.5, 2}.

Produces three tidy CSVs consumed by the notebook:
  rl_lsm_pricing.csv     R1  - kernel_on / kernel_off / lsm_M{5,9,17}, per regime & seed
  episode_efficiency.csv R2  - focal g2, kernel-on fast kernel at episodes {512..8192} + paper@32768
  mx_isolation.csv       R5  - focal g2, trained-agent price vs M_x in {1,2,3,4,6}

Protocol (matches tools/rebuild_results_v7.py, the canonical paper pipeline):
  * common fresh test set: seed=999, 65536 paths (never seen in training);
  * LSM fit on a separate train set (seed=998) with full state, chebyshev basis,
    DEGREE 2 (the paper pipeline's canonical degree; higher degree overfits OOS),
    and discretised action grids M in {5,9,17} (Sven: stronger than the 5-action default).
  * Every LSM fit is computed ONCE per (regime, M) and cached; R2/R5 reuse focal-g2.
No training: reuses tools/rebuild_results_v7.build_agent / evaluate_rl_on_test_set.

Run:
    python "gen_rl_validation.py"                  # full: TEST/TRAIN = 65536
    VAL3_NTEST=16384 python "gen_rl_validation.py" # quick validation pass
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)

import rebuild_results_v7 as rb  # noqa: E402
from src.simulate_hhk_spot import simulate_hhk_spot  # noqa: E402
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos  # noqa: E402

N = int(os.environ.get("VAL3_NTEST", 65536))   # test & LSM-train size
TEST_SEED, TRAIN_SEED = 999, 998
LSM_DEGREE, LSM_BASIS = 2, "chebyshev"
LSM_MS = [5, 9, 17]
V64_LR_A = 3e-4                                  # v64 canonical actor LR (distinguishes from v61's 1.6e-4)
V61_CSV = ROOT / "Jupyter Notebooks" / "Convex Costs Results 7.csv"  # v61 paper agents (gone from runs/)

REGIMES = [  # (key, c, gamma) — the 4 regimes with v64 agents on disk
    ("nocost", 0.0, 1.0),
    ("g1", 0.04, 1.0),
    ("g15", 0.04, 1.5),
    ("g2", 0.04, 2.0),
]
STD_S0 = 1.0  # standard contract spot


# --------------------------------------------------------------------------- #
# v64 agent discovery (by hyperparameter fingerprint) + v61 paper CSV baseline
# --------------------------------------------------------------------------- #
# Part II is now re-baselined on the v64 canonical (kernel-on M_x=2, depth-3,
# beta_sigmoid_1.5, lr_a 3e-4 / lr_c 6e-4, learn_number 2, critic_warmup 512).
# The deleted v61 "kernel-off" paper agents are replaced by their saved prices,
# read from 'Convex Costs Results 7.csv' (method="v61_paper").
def _load(jp):
    try:
        return json.load(open(jp))
    except Exception:
        return None


def _seed(d, name):
    s = d.get("seed")
    if isinstance(s, int):
        return s
    m = re.search(r"_s?(\d+)$", name)
    return int(m.group(1)) if m else None


def _regime_match(d, c, gamma):
    dc, dg = d.get("c_cost"), d.get("gamma_cost")  # None-safe (0.0 is a valid value!)
    return dc is not None and dg is not None and abs(dc - c) < 1e-9 and abs(dg - gamma) < 1e-9


def _is_v64(d):
    """v64 canonical marker: kernel-on, depth-3 actor, lr_a=3e-4 (vs v61's 1.6e-4)."""
    return (int(d.get("use_expected_target", 0) or 0) == 1
            and d.get("actor_layers") == 3
            and abs((d.get("lr_a") or 0) - V64_LR_A) < 1e-9)


def discover_v64(c, gamma, n_paths, M_x, M_per_k=1, N_max=1):
    """v64 agents matching (c, gamma, n_paths, kernel mesh).  {seed: json_path}.

    Robust to run name: the v64 4k/32k sweep agents (SwingOption_20_c..._v64_{4k,32k}_*)
    and the notebook-3 helper agents (nb3_v64_*) are all matched by fingerprint.
    """
    out = {}
    for jp in glob.glob(str(ROOT / "runs" / "*.json")):
        d = _load(jp)
        if d is None or not _regime_match(d, c, gamma):
            continue
        if (_is_v64(d) and d.get("n_paths") == n_paths and d.get("kernel_M_x") == M_x
                and d.get("kernel_M_per_k") == M_per_k and d.get("kernel_N_max") == N_max
                and abs((d.get("S0") or 0) - STD_S0) < 1e-6):
            s = _seed(d, Path(jp).stem)
            if s is not None and s not in out:
                out[s] = jp
    return out


def discover_v64_canonical(c, gamma):
    """The v64 canonical 4096-episode, M_x=2 agent for (c, gamma)."""
    return discover_v64(c, gamma, 4096, 2, 1, 1)


def v61_paper_seeds(c, gamma):
    """Per-seed v61 paper prices for (c, gamma) from 'Convex Costs Results 7.csv'.

    Returns {seed: price}.  These are the published paper agents (deleted from runs/),
    evaluated on the SAME common test-set protocol, so they are directly comparable.
    """
    if not V61_CSV.exists():
        return {}
    df = pd.read_csv(V61_CSV)
    row = df[(df["c"].round(4) == round(c, 4)) & (df["gamma"].round(4) == round(gamma, 4))]
    if row.empty:
        return {}
    r = row.iloc[0]
    out = {}
    for col in df.columns:
        m = re.match(r"RL_seed(\d+)$", col)
        if m and pd.notna(r[col]):
            out[int(m.group(1))] = float(r[col])
    return out


# --------------------------------------------------------------------------- #
# Per-regime context (test set + LSM fits), computed ONCE and cached
# --------------------------------------------------------------------------- #
_CTX: dict = {}


def context(key, c, gamma, suffix=None):
    if key in _CTX:
        return _CTX[key]
    canon = discover_v64_canonical(c, gamma)
    if not canon:
        raise RuntimeError(f"No v64 canonical (4096-ep, M_x=2) agent found for {key} "
                           f"(c={c}, gamma={gamma}); train it before generating Part II.")
    params = rb.dotdict(json.load(open(next(iter(canon.values())))))
    contract = rb.build_contract(params)
    hhk = rb.build_hhk_params(params)
    with contextlib.redirect_stdout(io.StringIO()):
        test = simulate_hhk_spot(**hhk, n_paths=N, seed=TEST_SEED, stratify=True, batch_size=128)
        test = tuple(np.asarray(a, np.float64) for a in test)
        train = simulate_hhk_spot(**hhk, n_paths=N, seed=TRAIN_SEED, stratify=True, batch_size=128)
        train = tuple(np.asarray(a, np.float64) for a in train)
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
    _CTX[key] = dict(contract=contract, hhk=hhk, test=test, lsm=lsm)
    return _CTX[key]


def eval_agent(json_path, contract, test):
    params = rb.dotdict(json.load(open(json_path)))
    with contextlib.redirect_stdout(io.StringIO()):
        agent = rb.build_agent(params)
        agent.actor_local.load_state_dict(torch.load(json_path[:-5] + ".pth", map_location="cpu"))
        agent.actor_local.eval()
        t0 = time.time()
        r = rb.evaluate_rl_on_test_set(agent, contract, test)
    r["eval_wall"] = time.time() - t0
    del agent
    return r


# --------------------------------------------------------------------------- #
# R1 - pricing: kernel_on / kernel_off / lsm_M{5,9,17}  x 4 regimes
# --------------------------------------------------------------------------- #
def gen_R1():
    rows = []
    for key, c, gamma in REGIMES:
        canon = discover_v64_canonical(c, gamma)
        v61 = v61_paper_seeds(c, gamma)
        print(f"[R1] {key}: v64 kernel-on seeds={sorted(canon)} | v61 paper seeds={sorted(v61)}")
        if not canon:
            print(f"     [skip {key}] no v64 canonical agent on disk yet — train it, then re-run.")
            continue
        ctx = context(key, c, gamma)
        for M, r in ctx["lsm"].items():
            rows.append(dict(regime=key, c=c, gamma=gamma, method=f"lsm_M{M}", seed=-1,
                             episodes=np.nan, price=r["price"], ci95=r["ci95"],
                             bangbang=r["bangbang"], M_x=np.nan, eval_wall=np.nan))
        # v64 canonical kernel-on (4096 ep), evaluated per seed on the common test set
        for s, jp in sorted(canon.items()):
            r = eval_agent(jp, ctx["contract"], ctx["test"])
            rows.append(dict(regime=key, c=c, gamma=gamma, method="kernel_on", seed=s, episodes=4096,
                             price=r["test_price"], ci95=r["test_CI95"], bangbang=r["bangbangness"],
                             M_x=2, eval_wall=r["eval_wall"]))
        if canon:
            mp = np.mean([x["price"] for x in rows if x["method"] == "kernel_on" and x["regime"] == key])
            print(f"      kernel_on (v64): mean price={mp:.4f}")
        # v61 paper baseline (deleted agents) — saved per-seed prices from the paper CSV
        for s, price in sorted(v61.items()):
            rows.append(dict(regime=key, c=c, gamma=gamma, method="v61_paper", seed=s, episodes=32768,
                             price=price, ci95=np.nan, bangbang=np.nan, M_x=np.nan, eval_wall=np.nan))
        if v61:
            print(f"      v61_paper: mean price={np.mean(list(v61.values())):.4f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# R2 - episode efficiency (focal g2): kernel-on fast kernel at {512,2048,4096,8192}
#       + paper @ 32768.  delta% vs canonical LSM(M=5).
# --------------------------------------------------------------------------- #
def gen_R2():
    ctx = context("g2", 0.04, 2.0)
    lsm5 = ctx["lsm"][5]["price"]
    rows = []
    # v64 kernel-on episode-efficiency curve (M_x=2 production mesh). 2048/8192 are the
    # notebook-3 helper agents; 4096 is the canonical sweep agent; 32768 comes from the
    # 32k sweep focal cell (appears once that sweep reaches c0.04/gamma2 — skipped if absent).
    for ep in [2048, 4096, 8192, 32768]:
        runs = discover_v64(0.04, 2.0, ep, 2, 1, 1)
        for s, jp in sorted(runs.items()):
            r = eval_agent(jp, ctx["contract"], ctx["test"])
            rows.append(dict(method="kernel_on", episodes=ep, M_per_k=1, seed=s, price=r["test_price"],
                             ci95=r["test_CI95"], delta_pct=100 * (r["test_price"] / lsm5 - 1),
                             bangbang=r["bangbangness"]))
        print(f"[R2] ep={ep}: v64 kernel-on seeds={sorted(runs)}")
    # v61 paper baseline @ 32768 (deleted agents) — per-seed saved prices from the paper CSV
    for s, price in sorted(v61_paper_seeds(0.04, 2.0).items()):
        rows.append(dict(method="v61_paper", episodes=32768, seed=s, price=price,
                         ci95=np.nan, delta_pct=100 * (price / lsm5 - 1), bangbang=np.nan))
    df = pd.DataFrame(rows)
    df["lsm_M5_price"] = lsm5
    df["lsm_M5_ci95"] = ctx["lsm"][5]["ci95"]
    return df


# --------------------------------------------------------------------------- #
# R5 - end-to-end M_x isolation (focal g2): ep=4096, warm=0, M_per_k=1, N_max=1
# --------------------------------------------------------------------------- #
def gen_R5():
    ctx = context("g2", 0.04, 2.0)
    lsm5 = ctx["lsm"][5]["price"]
    rows = []
    for M_x in [1, 2, 3, 4, 6]:
        runs = discover_v64(0.04, 2.0, 4096, M_x, 1, 1)
        for s, jp in sorted(runs.items()):
            r = eval_agent(jp, ctx["contract"], ctx["test"])
            rows.append(dict(M_x=M_x, M=2 * M_x, seed=s, price=r["test_price"], ci95=r["test_CI95"],
                             delta_pct=100 * (r["test_price"] / lsm5 - 1), bangbang=r["bangbangness"]))
        print(f"[R5] M_x={M_x}: seeds={sorted(runs)}")
    df = pd.DataFrame(rows)
    df["lsm_M5_price"] = lsm5
    return df


# --------------------------------------------------------------------------- #
# R3 reference - Monte-Carlo measurement noise at 64k paths
# --------------------------------------------------------------------------- #
def gen_mc_noise(n_seeds=15):
    """Training-seed variance of the LSM estimator — independently trained LSMs.

    Each row is one LSM trained from scratch on a fresh train set (train_seed) and evaluated
    on its own fresh OOS test set (eval_seed).  This gives the *training-seed variance* of
    the LSM — the same concept as RL seed dispersion, making the R3 comparison apples-to-apples.

    ci95 = half-width of the 95% CI on the single OOS evaluation, computed by bootstrap
    (≈ 1.96 × SE(path_payoffs) / √N for large N).  This is the irreducible MC noise floor
    for that individual estimate.

    delta_pct = (lsm_price / canonical_price − 1) × 100, where canonical_price is the
    fixed LSM-D M=5 evaluated at train=998 / test=999 (same denominator as every other Δ%).
    """
    canon = discover_v64_canonical(0.04, 2.0)
    if not canon:
        print("[MC] no v64 focal agent — skip")
        return pd.DataFrame()
    params = rb.dotdict(json.load(open(next(iter(canon.values())))))
    contract = rb.build_contract(params)
    hhk = rb.build_hhk_params(params)

    # Canonical reference price (LSM M=5, train 998 / test 999) — same denominator as R1/R3
    ctx = context("g2", 0.04, 2.0)
    canonical_price = ctx["lsm"][5]["price"]

    # Seeds chosen to not overlap with TRAIN_SEED=998, TEST_SEED=999, or old test-seeds 1000-1006
    TRAIN_SEEDS = list(range(100, 100 + n_seeds))   # 100 … 114
    EVAL_SEEDS  = list(range(200, 200 + n_seeds))   # 200 … 214

    rows = []
    for train_seed, eval_seed in zip(TRAIN_SEEDS, EVAL_SEEDS):
        with contextlib.redirect_stdout(io.StringIO()):
            train = tuple(np.asarray(a, np.float64) for a in
                          simulate_hhk_spot(**hhk, n_paths=N, seed=train_seed,
                                            stratify=True, batch_size=128))
            est = fit_lsm_estimators(contract=contract, dataset=train,
                                     poly_degree=LSM_DEGREE, basis_type=LSM_BASIS,
                                     state_mode="full", reg_type="none",
                                     reg_alpha=1e-6, n_actions=5)
            test = tuple(np.asarray(a, np.float64) for a in
                         simulate_hhk_spot(**hhk, n_paths=N, seed=eval_seed,
                                           stratify=True, batch_size=128))
            price, (lo, hi) = price_swing_option_lsm_oos(
                contract=contract, dataset=test, estimators=est,
                seed=eval_seed + 1, csv_path=None, _print_results=False)
        ci95 = (hi - lo) / 2.0   # ≈ 1.96 × SE(path_payoffs) via bootstrap
        delta_pct = 100.0 * (price / canonical_price - 1.0)
        rows.append(dict(train_seed=train_seed, eval_seed=eval_seed,
                         lsm_price=price, ci95=ci95, delta_pct=delta_pct))
        print(f"[MC] train={train_seed} eval={eval_seed}: "
              f"price={price:.4f}  ci95={ci95:.5f}  Δ%={delta_pct:+.3f}")

    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print(f"N(test=train)={N}, LSM={LSM_BASIS} deg={LSM_DEGREE} M in {LSM_MS}\n")
    gen_R1().to_csv(HERE / "rl_lsm_pricing.csv", index=False)
    gen_R2().to_csv(HERE / "episode_efficiency.csv", index=False)
    gen_R5().to_csv(HERE / "mx_isolation.csv", index=False)
    gen_mc_noise().to_csv(HERE / "mc_noise.csv", index=False)
    print(f"\nDone in {time.time() - t0:.0f}s. CSVs in {HERE}")


if __name__ == "__main__":
    main()
