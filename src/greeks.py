"""Pathwise Delta / Gamma for swing options via CRN bump-and-revalue.

The swing price is ``V(S0) = E[ sum of discounted net cashflows under a frozen
exercise policy pi ]`` where ``pi`` is the trained RL actor (or any policy that
maps the 9-d state to an exercise quantity).  For risk management we want the
sensitivities of ``V`` to the initial spot ``S0``:

    Delta = dV/dS0 ,   Gamma = d2V/dS0^2 .

Why bump-and-revalue is essentially exact here (Common Random Numbers).  In
``simulate_hhk_spot`` the initial spot enters *only* through the deterministic
``X[:,0] = log(S0) - f(0)``; every random draw (the Sobol OU increments and the
Poisson jumps) is keyed off ``seed`` and is **independent of S0**.  So evaluating
the policy on path bundles generated with the *same* ``seed`` at
``S0 in {S0-dS, S0, S0+dS}`` reuses identical randomness — the bump is the
near-deterministic multiplicative shift ``S_t(S0') = S_t(S0)*exp(dlogS0 * e^{-a t})``
(mean reversion decays it).  The finite-difference noise therefore (nearly)
cancels path-by-path, which both collapses the estimator variance and yields a
genuine *pathwise* Delta usable for delta-hedging each path.

Estimators (central differences, O(dS^2) accurate):

    Delta ~= [V(S0+dS) - V(S0-dS)] / (2 dS)
    Gamma ~= [V(S0+dS) - 2 V(S0) + V(S0-dS)] / dS^2

The bump is **relative** to the spot, ``dS = h * S0`` (so it auto-scales with the
price level).  Because the policy carries a profitability-gate kink the bias is
not perfectly smooth, so we evaluate on a 5-point stencil
``S0 * {1-h, 1-h/2, 1, 1+h/2, 1+h}`` and additionally report a Richardson
extrapolation of the ``(h, h/2)`` central differences, which cancels the leading
O(h^2) bias.  The spread between the ``h`` and ``h/2`` estimates is reported as a
discretization error bar.

This module is policy-agnostic: :func:`bump_greeks` takes a ``price_fn`` that maps
``(S0_value, seed) -> per-path PV array``.  :func:`make_rl_price_fn` builds such a
callable from a trained :class:`~src.agent.Agent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

try:  # package-relative first, mirror of src/agent.py import guard
    from .agent_evaluation import _build_state_batch, _evaluate_swing_batch, _feasible_actions
    from .hedging_utils import hhk_forward_price
    from .simulate_hhk_spot import simulate_hhk_spot
except ImportError:  # pragma: no cover - direct-script execution
    from agent_evaluation import _build_state_batch, _evaluate_swing_batch, _feasible_actions
    from hedging_utils import hhk_forward_price
    from simulate_hhk_spot import simulate_hhk_spot


PriceFn = Callable[[float, Optional[int]], np.ndarray]


@dataclass
class GreeksResult:
    """Delta/Gamma of an option value w.r.t. the initial spot ``S0``.

    All ``*_se`` fields are CRN standard errors of the per-path mean (they are
    tiny when CRN is working).  ``h_spread_*`` are the |G(h) - G(h/2)|
    discretization error bars.  ``*_richardson`` are the bias-cancelled
    ``(4 G(h/2) - G(h)) / 3`` estimates and are the recommended point values.
    """

    price: float
    delta: float
    gamma: float
    delta_se: float
    gamma_se: float
    delta_richardson: float
    gamma_richardson: float
    h: float
    dS: float
    h_spread_delta: float
    h_spread_gamma: float
    n_paths: int
    # Per-path pathwise sensitivities at bump h (empty unless return_paths=True).
    delta_path: np.ndarray = field(default=None, repr=False)
    gamma_path: np.ndarray = field(default=None, repr=False)
    pv_path: np.ndarray = field(default=None, repr=False)

    def as_dict(self) -> Dict[str, float]:
        return {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "delta_se": self.delta_se,
            "gamma_se": self.gamma_se,
            "delta_richardson": self.delta_richardson,
            "gamma_richardson": self.gamma_richardson,
            "h": self.h,
            "dS": self.dS,
            "h_spread_delta": self.h_spread_delta,
            "h_spread_gamma": self.h_spread_gamma,
            "n_paths": self.n_paths,
        }


def make_rl_price_fn(
    agent,
    contract,
    hhk_params: Dict,
    n_paths: int,
    *,
    eval_batch_size: int = 4096,
    stratify: bool = True,
) -> PriceFn:
    """Build a ``price_fn(S0_value, seed) -> (n_paths,) per-path PV`` for an Agent.

    ``hhk_params`` is the dict returned by ``tools.rebuild_results_v7.build_hhk_params``
    (its ``"S0"`` entry is overridden per call).  The same ``seed`` must be passed
    for every bump so the bundles share randomness (CRN).
    """
    base = dict(hhk_params)
    base.pop("S0", None)
    base.pop("n_paths", None)
    base.pop("seed", None)

    def price_fn(s0_value: float, seed: Optional[int]) -> np.ndarray:
        dataset = simulate_hhk_spot(
            S0=float(s0_value), n_paths=n_paths, seed=seed, stratify=stratify, **base
        )
        per_path: List[float] = []
        for start in range(0, n_paths, eval_batch_size):
            end = min(start + eval_batch_size, n_paths)
            returns, _, _ = _evaluate_swing_batch(
                agent,
                contract,
                dataset,
                list(range(start, end)),
                collect_path_data=False,
            )
            per_path.extend(returns)
        return np.asarray(per_path, dtype=np.float64)

    return price_fn


def bump_greeks(
    price_fn: PriceFn,
    S0: float,
    *,
    h: float = 0.01,
    seed: Optional[int] = 999,
    return_paths: bool = False,
) -> GreeksResult:
    """Central-difference Delta/Gamma of ``V(S0)`` with a relative bump ``dS=h*S0``.

    Evaluates ``price_fn`` on the 5-point stencil ``S0*{1-h, 1-h/2, 1, 1+h/2, 1+h}``
    sharing one ``seed`` (CRN), reports the ``h`` central differences plus a
    Richardson ``(h, h/2)`` extrapolation and a discretization error bar.
    """
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if not (0.0 < h < 1.0):
        raise ValueError(f"relative bump h must be in (0,1), got {h}")

    dS = h * S0
    dS_half = 0.5 * dS

    # Shared-randomness evaluations on the 5-point stencil.
    pv_mm = price_fn(S0 - dS, seed)       # S0 - h
    pv_m = price_fn(S0 - dS_half, seed)   # S0 - h/2
    pv_0 = price_fn(S0, seed)             # S0
    pv_p = price_fn(S0 + dS_half, seed)   # S0 + h/2
    pv_pp = price_fn(S0 + dS, seed)       # S0 + h

    n = pv_0.shape[0]
    price = float(pv_0.mean())

    # Pathwise central differences at bump h (the per-path hedge sensitivities).
    delta_path_h = (pv_pp - pv_mm) / (2.0 * dS)
    gamma_path_h = (pv_pp - 2.0 * pv_0 + pv_mm) / (dS * dS)
    delta_h = float(delta_path_h.mean())
    gamma_h = float(gamma_path_h.mean())
    if n > 1:
        delta_se = float(delta_path_h.std(ddof=1) / np.sqrt(n))
        gamma_se = float(gamma_path_h.std(ddof=1) / np.sqrt(n))
    else:
        delta_se = gamma_se = 0.0

    # Central differences at bump h/2 (uses the inner stencil points).
    delta_h2 = float(((pv_p - pv_m) / (2.0 * dS_half)).mean())
    gamma_h2 = float(((pv_p - 2.0 * pv_0 + pv_m) / (dS_half * dS_half)).mean())

    # Richardson extrapolation: central diffs have leading error O(step^2), so
    # (4*G(h/2) - G(h)) / 3 cancels it.
    delta_rich = (4.0 * delta_h2 - delta_h) / 3.0
    gamma_rich = (4.0 * gamma_h2 - gamma_h) / 3.0

    return GreeksResult(
        price=price,
        delta=delta_h,
        gamma=gamma_h,
        delta_se=delta_se,
        gamma_se=gamma_se,
        delta_richardson=delta_rich,
        gamma_richardson=gamma_rich,
        h=h,
        dS=dS,
        h_spread_delta=abs(delta_h - delta_h2),
        h_spread_gamma=abs(gamma_h - gamma_h2),
        n_paths=n,
        delta_path=delta_path_h if return_paths else None,
        gamma_path=gamma_path_h if return_paths else None,
        pv_path=pv_0 if return_paths else None,
    )


def greeks_for_run(
    run_name: str,
    *,
    runs_dir: str = "runs",
    n_paths: int = 16384,
    h: float = 0.01,
    seed: int = 999,
    eval_batch_size: int = 4096,
    return_paths: bool = False,
) -> GreeksResult:
    """Convenience: load a saved RL run and compute its t=0 Delta/Gamma.

    Reuses the canonical reconstruction helpers in ``tools/rebuild_results_v7.py``
    so the agent/contract/HHK params exactly match the results pipeline.
    """
    import json
    import os

    # Local import to avoid a hard tools<-src dependency at module import time.
    try:
        from tools.rebuild_results_v7 import (
            build_agent,
            build_contract,
            build_hhk_params,
            dotdict,
        )
    except ImportError:  # pragma: no cover
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from tools.rebuild_results_v7 import (  # type: ignore
            build_agent,
            build_contract,
            build_hhk_params,
            dotdict,
        )

    json_path = os.path.join(runs_dir, f"{run_name}.json")
    pth_path = os.path.join(runs_dir, f"{run_name}.pth")
    with open(json_path, "r") as fh:
        params = dotdict(json.load(fh))

    agent = build_agent(params)
    _load_actor_weights(agent, pth_path)
    contract = build_contract(params)
    hhk_params = build_hhk_params(params)

    price_fn = make_rl_price_fn(
        agent, contract, hhk_params, n_paths, eval_batch_size=eval_batch_size
    )
    return bump_greeks(
        price_fn, S0=float(params.S0), h=h, seed=seed, return_paths=return_paths
    )


def _load_actor_weights(agent, pth_path: str) -> None:
    """Load saved actor weights into ``agent.actor_local`` (eval-only).

    Mirrors the canonical loader in ``tools/rebuild_results_v7.py`` and
    ``evaluate_saved_agent.py``: the ``.pth`` is a plain actor ``state_dict``.
    """
    import torch

    state = torch.load(pth_path, map_location="cpu")
    if isinstance(state, dict) and "actor_state_dict" in state:
        state = state["actor_state_dict"]
    agent.actor_local.load_state_dict(state)
    agent.actor_local.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Daily-rebalanced dynamic delta hedge
#
# To hedge along a path we need the option's per-date Delta wrt the prompt spot,
# Δ_t = ∂V_t/∂S_t, where V_t is the *continuation value* at the realised contract
# state (spot, remaining inventory, refraction). We obtain it by bumping the spot
# at date t and re-rolling the frozen policy to maturity. The bump propagates
# through the *already-simulated* OU factor in closed form (Common Random
# Numbers, no re-simulation): a log-bump δx at date t shifts
#   X_k -> X_k + δx·e^{-α(t_k - t_t)},  S_k -> S_k·exp(δx·e^{-α(t_k - t_t)})  (k ≥ t),
# with the jump component Y untouched. The policy re-decides on the bumped spot,
# so Δ_t captures the policy's response.
#
# Hedge instrument: the HHK forward to the option maturity, discounted to t=0,
# H̃_t = DF_t·F_t(T) — a Q-martingale, costless to enter. The per-date hedge ratio
# is θ_t = ΔṼ_t / ΔH̃_t (both bumped by the same δx on X_t). The self-financing,
# daily-rebalanced seller P&L (discounted to 0) is
#   Π_i = E[Ṽ_0] − Ṽ_0,i + Σ_t θ_t,i (H̃_{t+1,i} − H̃_{t,i}).
# ─────────────────────────────────────────────────────────────────────────────
def make_rl_decide(agent):
    """Return ``decide(states)->actions in [0,1]`` for a trained Agent (eval, no noise)."""

    def decide(states: np.ndarray) -> np.ndarray:
        return np.asarray(agent.act(states, add_noise=False)).reshape(-1)

    return decide


def _roll_from(
    decide,
    contract,
    S: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    start_step: int,
    q_exercised0: np.ndarray,
    last_ex0: np.ndarray,
    *,
    collect: bool = False,
):
    """Roll a policy from ``start_step`` to maturity over ALL paths at once.

    Mirrors ``_evaluate_swing_batch`` exactly (feasibility, profitability gate,
    discounting to t=0) but starts from an arbitrary step and contract state.
    Returns the discounted-to-0 continuation PV per path; with ``collect=True``
    also returns per-step discounted cashflows and the *pre-decision* contract
    state at every step (needed to re-roll continuations).
    """
    B = S.shape[0]
    N = contract.n_rights
    q_min, q_max, Q_max = contract.q_min, contract.q_max, contract.Q_max
    strike, c, gamma = contract.strike, contract.c_cost, contract.gamma_cost
    df = contract.discount_factor

    q_exercised = q_exercised0.astype(np.float64).copy()
    last_ex = last_ex0.astype(np.int64).copy()
    done = (start_step >= N) | (q_exercised >= Q_max - 1e-6)
    pv = np.zeros(B, dtype=np.float64)

    cf = np.zeros((B, N), dtype=np.float64) if collect else None
    q_before = np.zeros((B, N), dtype=np.float64) if collect else None
    lastex_before = np.full((B, N), -1, dtype=np.int64) if collect else None

    for k in range(start_step, N):
        if collect:
            q_before[:, k] = q_exercised
            lastex_before[:, k] = last_ex
        active = np.nonzero(~done)[0]
        if active.size == 0:
            continue
        steps_k = np.full(active.size, k, dtype=np.int64)
        state = _build_state_batch(
            contract, S, X, Y, active, steps_k, q_exercised[active], last_ex[active]
        )
        actions = np.clip(decide(state), 0.0, 1.0)
        q_proposed = q_min + actions * (q_max - q_min)
        q_actual = _feasible_actions(contract, steps_k, q_exercised[active], q_proposed, last_ex[active])

        spot = state[:, 5].astype(np.float64)
        net = q_actual * np.maximum(spot - strike, 0.0) - c * np.power(q_actual, gamma)
        disc = np.power(df, k) * net
        nonpos = net <= 0.0
        disc[nonpos] = 0.0
        q_actual[nonpos] = 0.0

        exercised = q_actual > 1e-6
        last_ex[active[exercised]] = k
        pv[active] += disc
        if collect:
            cf[active, k] = disc
        q_exercised[active] += q_actual
        done[active] = (k + 1 >= N) | (q_exercised[active] >= Q_max - 1e-6)

    if collect:
        return pv, cf, q_before, lastex_before
    return pv


@dataclass
class DynamicHedgeResult:
    """Per-path daily-rebalanced delta-hedge diagnostics (discounted to t=0)."""

    pv: np.ndarray                # (n_paths,) realised option PV
    pnl_unhedged: np.ndarray      # (n_paths,) seller P&L E[pv]-pv
    pnl_hedged: np.ndarray        # (n_paths,) seller P&L after the daily hedge
    delta_t: np.ndarray           # (n_paths, n_rights) per-date dV/dS
    theta_t: np.ndarray           # (n_paths, n_rights) per-date forward hedge ratio
    Htilde: np.ndarray            # (n_paths, n_rights) discounted forward instrument
    spot: np.ndarray              # (n_paths, n_rights) spot path
    t_grid: np.ndarray            # (n_rights,) time grid
    cf: np.ndarray = field(default=None, repr=False)         # (n_paths, n_rights) discounted cashflows
    X: np.ndarray = field(default=None, repr=False)          # (n_paths, n_rights) OU factor
    Y: np.ndarray = field(default=None, repr=False)          # (n_paths, n_rights) jump factor
    q_before: np.ndarray = field(default=None, repr=False)   # (n_paths, n_rights) pre-decision q_exercised


def _condition_on_state(values: np.ndarray, spot: np.ndarray, q_rem: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Project a per-path quantity onto a polynomial of the date-t state, making it
    F_t-measurable (Longstaff–Schwartz-style conditional expectation). Returns the
    fitted values on the active paths; inactive paths get 0."""
    out = np.zeros_like(values)
    if active.sum() < 8:
        out[active] = values[active].mean() if active.any() else 0.0
        return out
    m = spot[active]
    q = q_rem[active]
    B = np.column_stack([np.ones_like(m), m, m * m, q, q * q, m * q])
    coef, *_ = np.linalg.lstsq(B, values[active], rcond=None)
    out[active] = B @ coef
    return out


def rl_dynamic_delta_hedge(
    agent,
    contract,
    hhk_params: Dict,
    S0: float,
    *,
    n_paths: int = 4096,
    seed: int = 999,
    h: float = 0.01,
    condition: bool = True,
) -> DynamicHedgeResult:
    """Daily-rebalanced delta hedge of the RL policy with the HHK forward.

    For each date the per-path continuation Delta is obtained by spot-bump-and-re-roll
    (closed-form OU propagation of the bump, Common Random Numbers). The pathwise
    delta anticipates the realised future, so (``condition=True``) it is projected
    onto the date-t state to make the hedge ratio F_t-measurable. The hedge holds
    forward contracts to maturity; the discounted P&L increment per contract is the
    martingale difference ``DF_{t+1}(F_{t+1}-F_t)``. Returns seller P&L with/without
    the daily hedge.
    """
    base = {k: v for k, v in hhk_params.items() if k not in ("S0", "n_paths", "seed")}
    alpha = float(hhk_params["alpha"])
    t_grid, S, X, Y = simulate_hhk_spot(S0=float(S0), n_paths=n_paths, seed=seed, **base)
    N = contract.n_rights
    T = float(contract.maturity)
    Q_max = contract.Q_max
    decide = make_rl_decide(agent)

    # Base roll: realised PV, per-step cashflows, pre-decision contract states.
    pv, cf, q_before, lastex_before = _roll_from(
        decide, contract, S, X, Y, 0, np.zeros(n_paths), np.full(n_paths, -1), collect=True
    )

    fwd_kw = dict(
        delivery_time=T, alpha=alpha, sigma=float(hhk_params["sigma"]),
        beta=float(hhk_params["beta"]), lam=float(hhk_params["lam"]), mu_J=float(hhk_params["mu_J"]),
    )
    DF = np.power(contract.discount_factor, np.arange(N, dtype=np.float64))  # (N,)

    # Undiscounted forward to maturity at every date (the tradeable's price).
    F_raw = np.zeros((n_paths, N), dtype=np.float64)
    for t in range(N):
        F_raw[:, t] = hhk_forward_price(
            current_time=np.full(n_paths, t_grid[t]), X_t=X[:, t], Y_t=Y[:, t], **fwd_kw
        )
    Htilde = DF[None, :] * F_raw  # discounted forward, for reference/plots

    delta_t = np.zeros((n_paths, N), dtype=np.float64)   # dV/dS (continuation)
    theta_t = np.zeros((n_paths, N), dtype=np.float64)   # forward hedge ratio dV/dF
    for t in range(N - 1):
        decay = np.exp(-alpha * (t_grid[t:] - t_grid[t]))           # (N-t,)
        Sp, Sm = S.copy(), S.copy()
        Xp, Xm = X.copy(), X.copy()
        Sp[:, t:] = S[:, t:] * np.exp(h * decay)
        Sm[:, t:] = S[:, t:] * np.exp(-h * decay)
        Xp[:, t:] = X[:, t:] + h * decay
        Xm[:, t:] = X[:, t:] - h * decay
        Vp = _roll_from(decide, contract, Sp, Xp, Y, t, q_before[:, t], lastex_before[:, t])
        Vm = _roll_from(decide, contract, Sm, Xm, Y, t, q_before[:, t], lastex_before[:, t])
        # Same δx bump on X_t (Y unchanged) for the undiscounted forward.
        Fp = hhk_forward_price(current_time=np.full(n_paths, t_grid[t]), X_t=X[:, t] + h, Y_t=Y[:, t], **fwd_kw)
        Fm = hhk_forward_price(current_time=np.full(n_paths, t_grid[t]), X_t=X[:, t] - h, Y_t=Y[:, t], **fwd_kw)
        dV = Vp - Vm
        dF = Fp - Fm
        delta_raw = dV / (S[:, t] * (np.exp(h) - np.exp(-h)))
        theta_raw = np.where(np.abs(dF) > 1e-12, dV / dF, 0.0)

        active = q_before[:, t] < Q_max - 1e-6
        if condition:
            q_rem = (Q_max - q_before[:, t]) / Q_max
            delta_t[:, t] = _condition_on_state(delta_raw, S[:, t] - contract.strike, q_rem, active)
            theta_t[:, t] = _condition_on_state(theta_raw, S[:, t] - contract.strike, q_rem, active)
        else:
            delta_t[:, t] = np.where(active, delta_raw, 0.0)
            theta_t[:, t] = np.where(active, theta_raw, 0.0)

    # Discounted forward P&L increment per contract: DF_{t+1}(F_{t+1}-F_t) (martingale diff).
    incr = DF[1:][None, :] * (F_raw[:, 1:] - F_raw[:, :-1])     # (n, N-1)
    gains = (theta_t[:, :-1] * incr).sum(axis=1)
    EpV = float(pv.mean())
    return DynamicHedgeResult(
        pv=pv,
        pnl_unhedged=EpV - pv,
        pnl_hedged=EpV - pv + gains,
        delta_t=delta_t,
        theta_t=theta_t,
        Htilde=Htilde,
        spot=S,
        t_grid=np.asarray(t_grid, dtype=np.float64),
        cf=cf,
        X=X,
        Y=Y,
        q_before=q_before,
    )


def regression_forward_hedge(
    cf: np.ndarray,
    spot: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    q_before: np.ndarray,
    t_grid: np.ndarray,
    contract,
    hhk_params: Dict,
) -> Dict[str, np.ndarray]:
    """Daily forward hedge with a **regression (Longstaff–Schwartz) delta**.

    Policy-agnostic: needs only per-path realised discounted cashflows ``cf`` (n,N),
    the spot/OU factor paths and the pre-decision inventory ``q_before``. At each date
    the realised continuation value ``V_t = Σ_{k≥t} cf_k`` is regressed cross-sectionally
    on a polynomial of the date-t state ``(S_t-K, q_remaining)``; its spot-derivative is
    the conditional (F_t-measurable) Delta. Hedged with the forward to maturity using the
    martingale increment ``DF_{t+1}(F_{t+1}-F_t)``. Putting RL and LSM through this one
    estimator makes the hedge comparison apples-to-apples.
    """
    n, N = spot.shape
    alpha = float(hhk_params["alpha"])
    T = float(contract.maturity)
    K = contract.strike
    Q_max = contract.Q_max
    DF = np.power(contract.discount_factor, np.arange(N, dtype=np.float64))
    fwd_kw = dict(delivery_time=T, alpha=alpha, sigma=float(hhk_params["sigma"]),
                  beta=float(hhk_params["beta"]), lam=float(hhk_params["lam"]), mu_J=float(hhk_params["mu_J"]))

    F_raw = np.zeros((n, N))
    for t in range(N):
        F_raw[:, t] = hhk_forward_price(current_time=np.full(n, t_grid[t]), X_t=X[:, t], Y_t=Y[:, t], **fwd_kw)

    Vmat = np.cumsum(cf[:, ::-1], axis=1)[:, ::-1]  # V_t = sum_{k>=t} cf_k (discounted to 0)
    theta = np.zeros((n, N))
    for t in range(N - 1):
        active = q_before[:, t] < Q_max - 1e-6
        if active.sum() < 8:
            continue
        m = spot[active, t] - K
        q = (Q_max - q_before[active, t]) / Q_max
        B = np.column_stack([np.ones_like(m), m, m * m, q, q * q, m * q])
        coef, *_ = np.linalg.lstsq(B, Vmat[active, t], rcond=None)
        delta_S = coef[1] + 2.0 * coef[2] * m + coef[5] * q          # dV_t/dS_t (conditional)
        dF_dS = np.exp(-alpha * (T - t_grid[t])) * F_raw[active, t] / spot[active, t]
        theta[active, t] = np.where(np.abs(dF_dS) > 1e-12, delta_S / dF_dS, 0.0)

    incr = DF[1:][None, :] * (F_raw[:, 1:] - F_raw[:, :-1])
    gains = (theta[:, :-1] * incr).sum(axis=1)
    pv = cf.sum(axis=1)
    EpV = float(pv.mean())
    return {"pv": pv, "pnl_unhedged": EpV - pv, "pnl_hedged": EpV - pv + gains, "theta_t": theta}


def forward_hedge_from_curve_delta(
    pv: np.ndarray,
    spot: np.ndarray,
    Htilde: np.ndarray,
    t_grid: np.ndarray,
    delta_of_spot: Callable[[np.ndarray], np.ndarray],
    *,
    alpha: float,
    T: float,
) -> Dict[str, np.ndarray]:
    """Daily forward hedge using a value-*curve* delta ``delta_of_spot(S)`` (= dV/dS).

    Policy-agnostic: works for any policy whose realised PV (``pv``), spot path
    (``spot``) and discounted forward (``Htilde``) are supplied. Converts the
    spot-delta to a forward hedge ratio via ``∂F_t/∂S_t = e^{-α(T-t)}·F_t/S_t``.
    Used to put RL and LSM on the *same* hedge methodology for comparison.
    """
    n_paths, N = spot.shape
    theta = np.zeros((n_paths, N), dtype=np.float64)
    for t in range(N - 1):
        # H̃_t = DF_t·F_t and dF/dS = e^{-α(T-t)}·F_t/S_t, so dH̃_t/dS = e^{-α(T-t)}·H̃_t/S_t.
        # θ_t (units of H̃) = δ_S / (dH̃_t/dS) makes θ_t·dH̃ ≈ δ_S·dS = dV.
        dHtilde_dS = np.exp(-alpha * (T - t_grid[t])) * Htilde[:, t] / spot[:, t]
        dS = delta_of_spot(spot[:, t])
        theta[:, t] = np.where(np.abs(dHtilde_dS) > 1e-12, dS / dHtilde_dS, 0.0)
    gains = (theta[:, :-1] * (Htilde[:, 1:] - Htilde[:, :-1])).sum(axis=1)
    EpV = float(pv.mean())
    return {"pnl_unhedged": EpV - pv, "pnl_hedged": EpV - pv + gains, "theta_t": theta}
