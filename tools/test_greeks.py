"""Tests for src/greeks.py — CRN bump-and-revalue Delta/Gamma.

Run with: pytest tools/test_greeks.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.greeks import (  # noqa: E402
    _load_actor_weights,
    bump_greeks,
    greeks_for_run,
    make_rl_price_fn,
)
from src.simulate_hhk_spot import no_seasonal_function, simulate_hhk_spot  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 1. Estimator math on a closed-form price_fn (no simulation): the central
#    differences must recover the analytic derivatives of a known V(S0).
# ─────────────────────────────────────────────────────────────────────────────
def _poly_price_fn(a, b, c):
    """V(S0) = a*S0^2 + b*S0 + c  (one 'path', deterministic)."""

    def price_fn(s0, seed):  # noqa: ARG001 - seed unused for a deterministic fn
        return np.array([a * s0 * s0 + b * s0 + c], dtype=np.float64)

    return price_fn


def test_quadratic_delta_gamma_exact():
    a, b, c = 0.7, -1.3, 2.0
    S0 = 100.0
    r = bump_greeks(_poly_price_fn(a, b, c), S0=S0, h=0.02)
    # Quadratic: central diffs are exact (no O(h^2) bias).
    assert r.delta == pytest.approx(2 * a * S0 + b, rel=1e-6)
    assert r.gamma == pytest.approx(2 * a, rel=1e-6)
    assert r.delta_richardson == pytest.approx(2 * a * S0 + b, rel=1e-6)
    assert r.gamma_richardson == pytest.approx(2 * a, rel=1e-6)
    # On a quadratic the h and h/2 estimates coincide -> zero spread.
    assert r.h_spread_delta == pytest.approx(0.0, abs=1e-6)
    assert r.h_spread_gamma == pytest.approx(0.0, abs=1e-6)


def test_richardson_beats_plain_on_quartic():
    """A quartic has nonzero d4V/dS4, so plain central Gamma carries an O(h^2)
    bias (= dS^2/12 * V'''') that Richardson cancels. (Degree<=3 is exact, which
    is exactly why a cubic would NOT exercise this.)"""
    S0 = 1.0
    h = 0.1

    def price_fn(s0, seed):  # noqa: ARG001
        return np.array([s0**4], dtype=np.float64)

    true_gamma = 12 * S0**2  # d2/ds2 (s^4) = 12 s^2
    r = bump_greeks(price_fn, S0=S0, h=h)
    plain_err = abs(r.gamma - true_gamma)
    rich_err = abs(r.gamma_richardson - true_gamma)
    # Plain bias is ~ dS^2/12 * 24 = 2 dS^2 = 0.02; Richardson removes it.
    assert plain_err > 1e-3
    assert rich_err < plain_err
    assert rich_err == pytest.approx(0.0, abs=1e-6)


def test_linear_payoff_zero_gamma():
    """A forward-like linear payoff has Delta=slope, Gamma=0."""
    r = bump_greeks(_poly_price_fn(0.0, 2.5, 1.0), S0=50.0, h=0.01)
    assert r.delta == pytest.approx(2.5, rel=1e-9)
    assert r.gamma == pytest.approx(0.0, abs=1e-6)


def test_invalid_inputs():
    pf = _poly_price_fn(1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        bump_greeks(pf, S0=-1.0, h=0.01)
    with pytest.raises(ValueError):
        bump_greeks(pf, S0=100.0, h=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CRN coupling: with a shared seed, bumping S0 shifts the OU log-factor X by
#    an exact deterministic decaying amount; all randomness is identical.
# ─────────────────────────────────────────────────────────────────────────────
def test_crn_coupling_is_exact():
    common = dict(
        T=1.0, n_steps=20, n_paths=256, alpha=7.0, sigma=1.4, beta=200.0,
        lam=4.0, mu_J=0.4, f=no_seasonal_function, seed=12345, stratify=True,
    )
    S0 = 100.0
    h = 0.01
    t1, S1, X1, Y1 = simulate_hhk_spot(S0=S0, **common)
    t2, S2, X2, Y2 = simulate_hhk_spot(S0=S0 * (1 + h), **common)

    # f(0)=0, so X[:,0] = log(S0). The OU recursion X_{k+1}=X_k*e^{-a dt}+noise
    # shares the same noise across the two runs, so the difference is the initial
    # log-bump propagated by pure mean reversion: dX_k = dlogS0 * e^{-alpha t_k}.
    dlog = np.log(S0 * (1 + h)) - np.log(S0)
    expected = dlog * np.exp(-7.0 * t2)  # (n_steps+1,)
    diff = X2 - X1
    # Same for every path (randomness cancels) and equals the analytic decay.
    assert np.allclose(diff, expected[None, :], atol=1e-4)
    # Jump component is identical (jumps independent of S0).
    assert np.allclose(Y1, Y2, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. End-to-end on a saved canonical RL run (skipped if weights are absent).
# ─────────────────────────────────────────────────────────────────────────────
_RUN = "SwingOption_20_c0.04_gamma2_11"
_HAVE_RUN = os.path.exists(os.path.join("runs", f"{_RUN}.pth")) and os.path.exists(
    os.path.join("runs", f"{_RUN}.json")
)


@pytest.mark.skipif(not _HAVE_RUN, reason=f"saved run {_RUN} not present")
def test_roll_from_step0_matches_price_fn():
    """The continuation roller from step 0 must reproduce the canonical per-path PV
    that make_rl_price_fn (built on _evaluate_swing_batch) produces."""
    import json

    from src.greeks import _roll_from, make_rl_decide, make_rl_price_fn
    from tools.rebuild_results_v7 import build_agent, build_contract, build_hhk_params, dotdict

    with open(os.path.join("runs", f"{_RUN}.json")) as fh:
        params = dotdict(json.load(fh))
    agent = build_agent(params)
    _load_actor_weights(agent, os.path.join("runs", f"{_RUN}.pth"))
    contract = build_contract(params)
    hhk = build_hhk_params(params)
    S0 = float(params.S0)
    n = 2048

    price_fn = make_rl_price_fn(agent, contract, hhk, n)
    pv_ref = price_fn(S0, 777)

    base = {k: v for k, v in hhk.items() if k != "S0"}
    _, S, X, Y = simulate_hhk_spot(S0=S0, n_paths=n, seed=777, **base)
    pv_roll = _roll_from(
        make_rl_decide(agent), contract, S, X, Y, 0, np.zeros(n), np.full(n, -1)
    )
    assert np.allclose(pv_roll, pv_ref, atol=1e-6)


@pytest.mark.skipif(not _HAVE_RUN, reason=f"saved run {_RUN} not present")
def test_dynamic_hedge_reduces_pnl_variance():
    """The daily-rebalanced forward hedge must reduce seller P&L variance and have
    ~zero mean (the forward instrument is a fair martingale)."""
    import json

    from src.greeks import rl_dynamic_delta_hedge
    from tools.rebuild_results_v7 import build_agent, build_contract, build_hhk_params, dotdict

    with open(os.path.join("runs", f"{_RUN}.json")) as fh:
        params = dotdict(json.load(fh))
    agent = build_agent(params)
    _load_actor_weights(agent, os.path.join("runs", f"{_RUN}.pth"))
    contract = build_contract(params)
    hhk = build_hhk_params(params)

    r = rl_dynamic_delta_hedge(agent, contract, hhk, float(params.S0), n_paths=2048, seed=999, h=0.01)
    assert r.pnl_hedged.std() < r.pnl_unhedged.std()           # variance reduction
    assert abs(r.pnl_hedged.mean()) < 0.1 * r.pnl_unhedged.std()  # ~zero mean
    assert (r.delta_t[:, : contract.n_rights - 1] >= -1e-6).mean() > 0.9  # mostly positive delta


@pytest.mark.skipif(not _HAVE_RUN, reason=f"saved run {_RUN} not present")
def test_saved_run_greeks_sane_and_crn_tight():
    r = greeks_for_run(_RUN, n_paths=4096, h=0.01, seed=999, return_paths=True)
    # Long convex swing: positive price, positive delta, non-negative gamma.
    assert r.price > 0
    assert r.delta > 0
    assert r.gamma > -1e-6
    # CRN must make the finite-difference standard error tiny vs the level.
    assert r.delta_se < 0.05 * abs(r.delta)
    # Discretization bias small: h and h/2 estimates nearly agree.
    assert r.h_spread_delta < 0.02 * abs(r.delta)
    # Most paths contribute positive pathwise delta.
    assert (r.delta_path > 0).mean() > 0.8
