"""Validation 3 - Part I: mathematical validation of the HHK semi-analytical kernel.

Generates the five `kernel_*.csv` consumed by the notebook. Self-contained: the
analytical OU/jump MGF moments and the brute-force nested-MC one-step sampler are
re-implemented here (parameterised by the focal HHK regime loaded from a saved run),
and the kernel mesh is built with the production `src.transition_kernel` API.

Run (a few minutes, no training):
    python "gen_kernel_validation.py"
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.transition_kernel import (  # noqa: E402
    KernelParams,
    precompute_kernel,
    build_next_state_grid_batched,
    build_quadrature_weights,
    OBS_IDX_S_MINUS_K,
    OBS_IDX_S,
    OBS_IDX_X,
    OBS_IDX_Y,
)

# --------------------------------------------------------------------------- #
# Focal HHK regime (loaded from the canonical v64 kernel-on focal run)
# --------------------------------------------------------------------------- #
def _find_focal_anchor():
    """Locate a v64 focal (c=0.04, gamma=2) agent for HHK params + the F4 critic.

    Prefers the canonical 4k sweep agent; falls back to any v64 focal agent.  (The
    old v61 kernel-study anchor `_sw_h1_approxF_F_anchor_g2_s11` was deleted.)
    """
    import glob as _glob
    cands = sorted(_glob.glob(str(ROOT / "runs" / "SwingOption_20_c0.04_gamma2_v64_4k_*.json")))
    if cands:
        return Path(cands[0])
    for jp in sorted(_glob.glob(str(ROOT / "runs" / "*.json"))):
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        if (d.get("c_cost") == 0.04 and d.get("gamma_cost") == 2.0 and d.get("actor_layers") == 3
                and int(d.get("use_expected_target", 0) or 0) == 1):
            return Path(jp)
    raise FileNotFoundError("No v64 focal (c=0.04, gamma=2) agent found for Part I HHK params/critic.")


ANCHOR = _find_focal_anchor()
P = json.load(open(ANCHOR))
ALPHA, SIGMA, BETA = float(P["alpha"]), float(P["sigma"]), float(P["beta"])
LAM, MU_J = float(P["lam"]), float(P["mu_J"])
MATURITY, N_RIGHTS = float(P["maturity"]), int(P["n_rights"])
STRIKE = float(P["strike"])
DT = MATURITY / (N_RIGHTS - 1)

FAST = dict(M_x=2, M_per_k=1, N_max=1)     # M = 4
ACCURATE = dict(M_x=4, M_per_k=4, N_max=2)  # M = 36


def f_no_seasonal(t: float) -> float:
    return 0.0


# --------------------------------------------------------------------------- #
# Closed-form conditional moments (joint OU + compound-Poisson MGF, theta=1,2)
# --------------------------------------------------------------------------- #
def _ou(lam=LAM):
    dX = math.exp(-ALPHA * DT)
    sX2 = SIGMA ** 2 * (1.0 - dX ** 2) / (2.0 * ALPHA)
    dY = math.exp(-BETA * DT)
    return dX, sX2, dY


def anal_E_S(X, Y, lam=LAM):
    dX, sX2, dY = _ou()
    a, b = MU_J * dY, MU_J
    M = ((1 - a) / (1 - b)) ** (lam / BETA) if (a < 1 and b < 1) else float("nan")
    return math.exp(dX * X + 0.5 * sX2 + dY * Y) * M


def anal_E_S2(X, Y, lam=LAM):
    dX, sX2, dY = _ou()
    th = 2.0
    a, b = th * MU_J * dY, th * MU_J
    M = ((1 - a) / (1 - b)) ** (lam / BETA) if (a < 1 and b < 1) else float("nan")
    return math.exp(th * dX * X + 0.5 * th ** 2 * sX2 + th * dY * Y) * M


# --------------------------------------------------------------------------- #
# Brute-force nested Monte-Carlo one-step sampler  (X_{t+1}, Y_{t+1} | X_t, Y_t)
# --------------------------------------------------------------------------- #
def mc_one_step(X_t, Y_t, n, rng, lam=LAM):
    dX, sX2, dY = _ou()
    sX = math.sqrt(sX2)
    X = dX * X_t + sX * rng.standard_normal(n)
    counts = rng.poisson(lam * DT, size=n)
    n_tot = int(counts.sum())
    if n_tot == 0:
        D = np.zeros(n)
    else:
        U = rng.uniform(0.0, DT, size=n_tot)
        V = np.clip(rng.random(n_tot), 1e-12, 1 - 1e-12)
        J = -MU_J * np.log(V)
        contrib = J * np.exp(-BETA * (DT - U))
        D = np.bincount(np.repeat(np.arange(n), counts), weights=contrib, minlength=n)
    Y = dY * Y_t + D
    S = np.exp(f_no_seasonal(DT) + X + Y)
    return S, X, Y


# --------------------------------------------------------------------------- #
# Kernel one-step grid for a single (X,Y) cell
# --------------------------------------------------------------------------- #
def kernel_grid(X_t, Y_t, M_x, M_per_k, N_max, lam=LAM):
    kp = KernelParams(alpha=ALPHA, sigma=SIGMA, beta=BETA, lam=lam, mu_J=MU_J, dt=DT,
                      M_x=M_x, N_max=N_max, M_per_k=M_per_k)
    k = precompute_kernel(kp, N_RIGHTS, f_no_seasonal, maturity=MATURITY, force_rebuild=True)
    S, X, Y = build_next_state_grid_batched(
        np.array([X_t], float), np.array([Y_t], float), np.array([1], np.int64), k)
    w = build_quadrature_weights(k)
    return S[0], X[0], Y[0], w, k.M


# Smooth test functions (proxies for a critic continuation value near the strike)
def g_S(S):
    return S
def g_S2(S):
    return S * S
def g_kink(S):
    return np.maximum(S - STRIKE, 0.0)
def g_smooth(S):  # smooth payoff proxy
    sig = 1.0 / (1.0 + np.exp(-10.0 * (S - STRIKE)))
    return (S - STRIKE) * sig


STATE_PANEL = [(0.0, 0.0), (0.10, 0.0), (-0.10, 0.0), (0.0, 0.5), (0.10, 0.5), (-0.05, 0.2)]


# --------------------------------------------------------------------------- #
# F1 - Gaussian (OU) leg exactness  +  full-moment panel
# --------------------------------------------------------------------------- #
def gen_F1():
    rows = []
    # (a) Gaussian leg only (lam=0): Gauss-Hermite spectral convergence in M_x
    X0 = 0.10
    eS_exact = anal_E_S(X0, 0.0, lam=0.0)
    eS2_exact = anal_E_S2(X0, 0.0, lam=0.0)
    for M_x in [1, 2, 3, 4, 6, 8, 12]:
        S, X, Y, w, M = kernel_grid(X0, 0.0, M_x, 1, 1, lam=0.0)
        eS = float(np.sum(w * S)); eS2 = float(np.sum(w * S * S))
        rows.append(dict(leg="gaussian", X=X0, Y=0.0, M_x=M_x, M=M,
                         relerr_ES=abs(eS - eS_exact) / eS_exact,
                         relerr_ES2=abs(eS2 - eS2_exact) / eS2_exact))
    # (b) full kernel (with jumps) moments vs MGF over the state panel, fast & accurate
    for tag, cfg in [("fast", FAST), ("accurate", ACCURATE)]:
        for (X0, Y0) in STATE_PANEL:
            S, X, Y, w, M = kernel_grid(X0, Y0, **cfg)
            eS = float(np.sum(w * S)); eS2 = float(np.sum(w * S * S))
            rows.append(dict(leg=f"full_{tag}", X=X0, Y=Y0, M_x=cfg["M_x"], M=M,
                             relerr_ES=abs(eS - anal_E_S(X0, Y0)) / anal_E_S(X0, Y0),
                             relerr_ES2=abs(eS2 - anal_E_S2(X0, Y0)) / anal_E_S2(X0, Y0)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# F2 - kernel-vs-nested-MC convergence in M_x  (jump mesh fixed at accurate)
# --------------------------------------------------------------------------- #
def gen_F2(n_mc=2_000_000, seed=42):
    rng = np.random.default_rng(seed)
    funcs = {"E[S]": g_S, "E[S^2]": g_S2, "E[(S-K)+]": g_kink, "E[smooth]": g_smooth}
    rows = []
    for (X0, Y0) in STATE_PANEL:
        S_mc, _, _ = mc_one_step(X0, Y0, n_mc, rng)
        truth = {n: float(np.mean(f(S_mc))) for n, f in funcs.items()}
        se = {n: float(np.std(funcs[n](S_mc)) / math.sqrt(n_mc)) for n in funcs}
        for M_x in [1, 2, 3, 4, 6, 8, 12]:
            S, X, Y, w, M = kernel_grid(X0, Y0, M_x, ACCURATE["M_per_k"], ACCURATE["N_max"])
            for n, f in funcs.items():
                approx = float(np.sum(w * f(S)))
                rows.append(dict(X=X0, Y=Y0, M_x=M_x, M=M, func=n,
                                 kernel=approx, mc=truth[n], mc_se=se[n],
                                 relerr=abs(approx - truth[n]) / max(abs(truth[n]), 1e-12)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# F3 - jump-mesh convergence (vs N_max, vs M_per_k)  + Poisson tail mass
# --------------------------------------------------------------------------- #
def gen_F3(n_mc=2_000_000, seed=7):
    rng = np.random.default_rng(seed)
    funcs = {"E[S]": g_S, "E[S^2]": g_S2, "E[(S-K)+]": g_kink, "E[smooth]": g_smooth}
    rows = []
    X0, Y0 = 0.10, 0.5
    S_mc, _, _ = mc_one_step(X0, Y0, n_mc, rng)
    truth = {n: float(np.mean(f(S_mc))) for n, f in funcs.items()}
    se = {n: float(np.std(funcs[n](S_mc)) / math.sqrt(n_mc)) for n in funcs}
    lam_dt = LAM * DT
    for axis, values, fixed in [("N_max", [1, 2, 3, 4, 5], dict(M_x=4, M_per_k=8)),
                                ("M_per_k", [1, 2, 4, 8, 16, 32], dict(M_x=4, N_max=3))]:
        for v in values:
            cfg = dict(fixed); cfg[axis] = v
            S, X, Y, w, M = kernel_grid(X0, Y0, **cfg)
            # Poisson tail mass beyond the truncation N_max
            Nmax = cfg["N_max"]
            tail = float(1.0 - sum(math.exp(-lam_dt) * lam_dt ** k / math.factorial(k)
                                   for k in range(Nmax + 1)))
            for n, f in funcs.items():
                approx = float(np.sum(w * f(S)))
                rows.append(dict(axis=axis, value=v, M_x=cfg["M_x"],
                                 M_per_k=cfg.get("M_per_k"), N_max=cfg["N_max"], M=M,
                                 func=n, kernel=approx, mc=truth[n], mc_se=se[n],
                                 relerr=abs(approx - truth[n]) / max(abs(truth[n]), 1e-12),
                                 poisson_tail=tail))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# F4 - expected-Bellman-target unbiasedness: kernel vs nested-MC on fixed (s,a)
#       for smooth payoff proxies + a frozen architecture-matched critic network
# --------------------------------------------------------------------------- #
def _build_frozen_nets():
    """Load the actor (trained) + an architecture-matched critic (frozen) from the anchor run."""
    import torch
    sys.path.insert(0, str(ROOT / "tools"))
    import rebuild_results_v7 as rb
    params = rb.dotdict(json.load(open(ANCHOR)))
    with contextlib.redirect_stdout(io.StringIO()):
        agent = rb.build_agent(params)
        agent.actor_local.load_state_dict(
            torch.load(str(ANCHOR)[:-5] + ".pth", map_location="cpu"))
    agent.actor_local.eval(); agent.critic_local.eval()
    return agent, torch


def _critic_samples(agent, torch, S, X, Y, base_state, chunk=200_000):
    """Per-sample Q(s', pi(s')) for next states built from (S,X,Y) + the fixed base vector."""
    K = S.shape[0]
    out = np.empty(K)
    for i in range(0, K, chunk):
        sl = slice(i, min(i + chunk, K))
        st = np.tile(base_state, (S[sl].shape[0], 1)).astype(np.float32)
        st[:, OBS_IDX_S_MINUS_K] = S[sl] - STRIKE
        st[:, OBS_IDX_S] = S[sl]
        st[:, OBS_IDX_X] = X[sl]
        st[:, OBS_IDX_Y] = Y[sl]
        with torch.no_grad():
            a = agent.act(st, add_noise=False)
            q = agent.critic_local(torch.as_tensor(st),
                                   torch.as_tensor(np.asarray(a, np.float32))).cpu().numpy().ravel()
        out[sl] = q
    return out


def _critic_E(agent, torch, S, X, Y, base_state, w):
    """Quadrature-weighted E[ Q(s', pi(s')) ] over kernel nodes (S,X,Y)."""
    q = _critic_samples(agent, torch, S, X, Y, base_state)
    return float(np.sum(w * q))


def gen_F4(n_mc=4_000_000, seed=123):
    """Kernel expected-target vs a *gold* nested-MC reference (n=4M).

    The kernel is a deterministic quadrature: it carries a small, bounded bias
    (the jump-mesh discretisation, ~0.1-1%), not MC noise. So we report the
    relative error vs the gold MC mean and show it shrinks as the mesh refines
    (fast -> accurate -> rich). `boot_std` is the standard deviation of the
    integrand under the transition -- i.e. the noise a single-sample (n=1) TD
    bootstrap target would carry; it is the variance the expected backup removes.
    """
    agent, torch = _build_frozen_nets()
    rng = np.random.default_rng(seed)
    base = np.zeros(9, np.float32)
    base[1] = 0.30; base[2] = 0.70; base[3] = 0.50; base[4] = 0.50; base[8] = 0.10
    tiers = [("fast", FAST), ("accurate", ACCURATE),
             ("rich", dict(M_x=6, M_per_k=16, N_max=3))]
    rows = []
    for (X0, Y0) in STATE_PANEL:
        S_mc, X_mc, Y_mc = mc_one_step(X0, Y0, n_mc, rng)
        gk, gs = g_kink(S_mc), g_smooth(S_mc)
        cq_samples = _critic_samples(agent, torch, S_mc, X_mc, Y_mc, base)
        mc = {  # func -> (mean, se, boot_std)
            "E[(S-K)+]": (float(np.mean(gk)), float(np.std(gk) / math.sqrt(n_mc)), float(np.std(gk))),
            "E[smooth]": (float(np.mean(gs)), float(np.std(gs) / math.sqrt(n_mc)), float(np.std(gs))),
            "E[S]": (float(np.mean(S_mc)), float(np.std(S_mc) / math.sqrt(n_mc)), float(np.std(S_mc))),
            "E[critic_Q]": (float(np.mean(cq_samples)), float(np.std(cq_samples) / math.sqrt(n_mc)),
                            float(np.std(cq_samples))),
        }
        for tag, cfg in tiers:
            S, X, Y, w, M = kernel_grid(X0, Y0, **cfg)
            ker = {
                "E[(S-K)+]": float(np.sum(w * g_kink(S))),
                "E[smooth]": float(np.sum(w * g_smooth(S))),
                "E[S]": float(np.sum(w * S)),
                "E[critic_Q]": _critic_E(agent, torch, S, X, Y, base, w=w),
            }
            for fn in ker:
                mc_mean, mc_se, boot = mc[fn]
                rows.append(dict(X=X0, Y=Y0, kernel_tag=tag, M=M, func=fn,
                                 kernel=ker[fn], mc_ref=mc_mean, mc_se=mc_se,
                                 rel_err=abs(ker[fn] - mc_mean) / max(abs(mc_mean), 1e-12),
                                 boot_std=boot))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# F5 - cost / accuracy Pareto (build time, total nodes M, worst-case rel error)
# --------------------------------------------------------------------------- #
def gen_F5(n_mc=2_000_000, seed=999, n_timing=300):
    rng = np.random.default_rng(seed)
    funcs = {"E[S]": g_S, "E[S^2]": g_S2, "E[(S-K)+]": g_kink, "E[smooth]": g_smooth}
    # MC truth over the panel
    truth = {}
    for (X0, Y0) in STATE_PANEL:
        S_mc, _, _ = mc_one_step(X0, Y0, n_mc, rng)
        truth[(X0, Y0)] = {n: float(np.mean(f(S_mc))) for n, f in funcs.items()}
    configs = [
        ("M_x1 (degenerate)", dict(M_x=1, M_per_k=1, N_max=1)),
        ("FAST M=4", FAST),
        ("M_x=2,Nmax2", dict(M_x=2, M_per_k=2, N_max=2)),
        ("M_x=3", dict(M_x=3, M_per_k=3, N_max=2)),
        ("ACCURATE M=36", ACCURATE),
        ("M_x=6 rich", dict(M_x=6, M_per_k=4, N_max=3)),
    ]
    rows = []
    for label, cfg in configs:
        # accuracy: mean rel error on the smooth integrands (robust; the kinky OTM
        # payoff inflates worst-case via a near-zero denominator), plus worst-case.
        worst = 0.0
        es_errs, smooth_errs = [], []
        Mtot = None
        for (X0, Y0) in STATE_PANEL:
            S, X, Y, w, M = kernel_grid(X0, Y0, **cfg)
            Mtot = M
            for n, f in funcs.items():
                approx = float(np.sum(w * f(S)))
                t = truth[(X0, Y0)][n]
                e = abs(approx - t) / max(abs(t), 1e-12)
                worst = max(worst, e)
                if n == "E[S]":
                    es_errs.append(e)
                if n == "E[smooth]":
                    smooth_errs.append(e)
        # timing: per-call build_next_state_grid_batched on a B=128 batch (warm cache)
        kp = KernelParams(alpha=ALPHA, sigma=SIGMA, beta=BETA, lam=LAM, mu_J=MU_J, dt=DT, **cfg)
        k = precompute_kernel(kp, N_RIGHTS, f_no_seasonal, maturity=MATURITY)
        Xb = (rng.standard_normal(128) * 0.1)
        Yb = np.abs(rng.standard_normal(128) * 0.5)
        tb = np.ones(128, np.int64)
        _ = build_next_state_grid_batched(Xb, Yb, tb, k)  # warm
        t0 = time.perf_counter()
        for _ in range(n_timing):
            _ = build_next_state_grid_batched(Xb, Yb, tb, k)
        us = (time.perf_counter() - t0) / n_timing * 1e6
        rows.append(dict(label=label, M_x=cfg["M_x"], M_per_k=cfg["M_per_k"],
                         N_max=cfg["N_max"], M=Mtot,
                         relerr_ES=float(np.mean(es_errs)),
                         relerr_smooth=float(np.mean(smooth_errs)),
                         worst_relerr=worst, build_us_b128=us))
    return pd.DataFrame(rows)


def main():
    out = HERE
    t0 = time.time()
    print(f"[F1] Gaussian-leg exactness + full-moment panel ...")
    gen_F1().to_csv(out / "kernel_moments.csv", index=False)
    print(f"[F2] M_x convergence vs nested MC ...")
    gen_F2().to_csv(out / "kernel_quadrature_convergence.csv", index=False)
    print(f"[F3] jump-mesh convergence ...")
    gen_F3().to_csv(out / "kernel_jump_mesh_convergence.csv", index=False)
    print(f"[F4] nested-MC agreement (payoff + frozen critic) ...")
    gen_F4().to_csv(out / "kernel_nested_mc_agreement.csv", index=False)
    print(f"[F5] cost / accuracy Pareto ...")
    gen_F5().to_csv(out / "kernel_cost_accuracy.csv", index=False)
    print(f"Done in {time.time() - t0:.1f}s. CSVs written to:\n  {out}")


if __name__ == "__main__":
    main()
