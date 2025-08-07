"""Finite-difference swing option pricer using QuantLib.

This module exposes :func:`price_swing_option_fdm` which prices a swing
option under the Hambly–Howison–Kluge (HHK) model using QuantLib's
finite-difference engine ``FdSimpleExtOUJumpSwingEngine``.

The spot is modelled as ``S_t = exp(f(t) + X_t + Y_t)`` with a
diffusive Ornstein–Uhlenbeck component ``X`` and a mean-reverting jump
component ``Y``.  The engine solves a three‑dimensional PDE accounting
for the number of exercise rights as an additional state variable.
"""
from __future__ import annotations

from typing import Callable, Dict

import math

import QuantLib as ql

from .swing_contract import SwingContract


def _build_exercise_dates(contract: SwingContract, today: ql.Date) -> list[ql.Date]:
    """Construct exercise dates for the swing option.

    The contract gives the maturity in years and the number of exercise
    rights.  For simplicity we assume the rights are uniformly spaced in
    time between the evaluation date and maturity.
    """
    n = contract.n_rights
    if n <= 0:
        raise ValueError("contract.n_rights must be positive")

    total_days = max(1, int(round(contract.maturity * 365)))
    step = max(1, int(round(total_days / n)))
    return [today + i * step for i in range(1, n + 1)]


def price_swing_option_fdm(
    contract: SwingContract,
    stochastic_process_params: Dict[str, float | Callable[[float], float]],
    tGrid: int = 25,
    xGrid: int = 25,
    yGrid: int = 50,
) -> float:
    """Price a swing option using QuantLib's finite difference engine.

    Parameters
    ----------
    contract:
        Swing option contract description.
    stochastic_process_params:
        Parameters of the HHK model.  Required keys are ``S0``, ``alpha``,
        ``sigma``, ``beta``, ``lam`` and ``mu_J``.  An optional key ``f``
        provides the deterministic log-drift function ``f(t)``.
    tGrid, xGrid, yGrid:
        Grid sizes for the time, ``X`` and ``Y`` dimensions of the PDE
        solver.

    Returns
    -------
    float
        The present value of the swing option.
    """
    # Unpack HHK parameters
    S0 = float(stochastic_process_params["S0"])
    alpha = float(stochastic_process_params["alpha"])
    sigma = float(stochastic_process_params["sigma"])
    beta = float(stochastic_process_params["beta"])
    lam = float(stochastic_process_params["lam"])
    mu_J = float(stochastic_process_params["mu_J"])
    f: Callable[[float], float] = stochastic_process_params.get("f", lambda t: 0.0)  # type: ignore

    # Market data and evaluation date
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    r_curve = ql.FlatForward(today, contract.r, day_count)

    # Exercise schedule
    exercise_dates = _build_exercise_dates(contract, today)
    exercise = ql.SwingExercise(exercise_dates)

    # Payoff and option instrument
    payoff = ql.VanillaForwardPayoff(ql.Option.Call, contract.strike)
    # QuantLib's swing option counts integer exercise rights.  The
    # contract specifies volume limits, so we translate them to an
    # equivalent number of rights by assuming each right corresponds to
    # ``q_max`` units.
    min_rights = int(round(contract.Q_min / contract.q_max))
    max_rights = int(round(contract.Q_max / contract.q_max))
    option = ql.VanillaSwingOption(payoff, exercise, min_rights, max_rights)

    # HHK stochastic process
    x0 = math.log(S0) - f(0.0)
    ou_process = ql.ExtendedOrnsteinUhlenbeckProcess(alpha, sigma, x0, lambda t: 0.0)
    eta = 1.0 / mu_J  # QuantLib uses rate of exponential jump distribution
    process = ql.ExtOUWithJumpsProcess(ou_process, 0.0, beta, lam, eta)

    # Deterministic drift/seasonality curve f(t)
    times = [day_count.yearFraction(today, d) for d in exercise_dates]
    curve_shape = [(t, f(t)) for t in times]

    engine = ql.FdSimpleExtOUJumpSwingEngine(
        process,
        r_curve,
        int(tGrid),
        int(xGrid),
        int(yGrid),
        curve_shape,
    )
    option.setPricingEngine(engine)

    return option.NPV()
