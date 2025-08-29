"""Least-squares Monte Carlo pricer for swing options."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .swing_contract import SwingContract


def _regress(X: np.ndarray, y: np.ndarray, degree: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return fitted values of polynomial regression."""
    if mask is not None and mask.sum() >= degree + 1:
        Xm = X[mask]
        ym = y[mask]
    else:
        Xm = X
        ym = y
    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    return X @ beta


def price_swing_option_lsm(
    contract: SwingContract,
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    poly_degree: int = 2,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
    csv_path: str = "swing_option_lsm_paths.csv",
    _print_results: bool = False,
) -> Tuple[float, Tuple[float, float]]:
    """Price swing option using the Longstaff–Schwartz method.

    Parameters
    ----------
    contract : SwingContract
        Swing option contract specifications.
    dataset : tuple
        Tuple ``(t, S, X, Y)`` with simulated spot price paths ``S`` of
        shape ``(n_paths, n_rights)``.
    poly_degree : int, optional
        Degree of polynomial basis used in regressions.
    n_bootstrap : int, optional
        Number of bootstrap samples for confidence interval.
    seed : int, optional
        Random seed for bootstrap.
    csv_path : str, optional
        Destination for CSV log of optimal exercises.
    """
    t, S, _, _ = dataset
    prices = S  # all decision prices (including initial spot at t=0)
    n_paths, n_steps = prices.shape
    assert n_steps == contract.n_rights, "Mismatch between paths and contract rights"

    df = contract.discount_factor
    strike = contract.strike
    qmax = contract.q_max
    cooldown = max(0, int(getattr(contract, "min_refraction_periods", 0)))

    # number of discrete rights (assumes Q_max multiple of q_max)
    R = int(round(contract.Q_max / qmax))

    # Add cooldown state c in [0..cooldown]. c>0 means must wait c more periods to be allowed to exercise
    values = np.zeros((cooldown + 1, R + 1, n_paths))
    exercise = np.zeros((cooldown + 1, R + 1, n_paths, n_steps), dtype=bool)

    payoff_T = qmax * np.maximum(prices[:, -1] - strike, 0.0)
    itm_T = prices[:, -1] > strike
    # Terminal step: can exercise only if cooldown state c==0 and r>=1
    for c in range(cooldown + 1):
        for r in range(1, R + 1):
            if c == 0:
                values[c, r] = payoff_T
                exercise[c, r, itm_T, n_steps - 1] = True
            else:
                # Cannot exercise at terminal if in cooldown; value is zero (no future)
                values[c, r] = 0.0

    X_poly = np.empty((n_paths, poly_degree + 1))

    for j in range(n_steps - 2, -1, -1):
        price = prices[:, j]
        payoff = qmax * np.maximum(price - strike, 0.0)
        X_poly[:, 0] = 1.0
        for k in range(1, poly_degree + 1):
            X_poly[:, k] = price ** k
        mask = payoff > 0
        old_vals = values.copy()
        new_vals = values.copy()
        # Iterate cooldown states and rights remaining
        for c in range(cooldown + 1):
            for r in range(1, R + 1):
                # If we keep (no exercise now): cooldown counts down (cannot go below 0)
                c_keep = max(c - 1, 0)
                y_keep = df * old_vals[c_keep, r]
                cont_keep = _regress(X_poly, y_keep, poly_degree, mask)

                if c == 0:
                    # If we exercise now: cooldown resets to full, rights reduce by 1
                    c_ex = cooldown
                    y_ex = df * old_vals[c_ex, r - 1]
                    cont_ex = _regress(X_poly, y_ex, poly_degree, mask)
                    exc = (payoff + cont_ex > cont_keep) & (payoff > 0)
                    exercise[c, r, exc, j] = True
                    new_vals[c, r] = np.where(exc, payoff + y_ex, y_keep)
                else:
                    # In cooldown: cannot exercise
                    new_vals[c, r] = y_keep
        values = new_vals

    rights = np.full(n_paths, R, dtype=int)
    cool_state = np.zeros(n_paths, dtype=int)  # forward cooldown tracker
    q_used = np.zeros(n_paths)
    path_payoffs = np.zeros(n_paths)
    records = []
    for j in range(n_steps):
        price = prices[:, j]
        # Use 0-based discount exponent so that the first decision at t=0 is undiscounted
        disc = df ** j
        for i in range(n_paths):
            r = rights[i]
            c = cool_state[i]
            q_before = q_used[i]
            if r > 0 and exercise[c, r, i, j]:
                q = min(qmax, contract.Q_max - q_before)
                pay = q * max(price[i] - strike, 0.0)
                rights[i] -= 1
                q_used[i] += q
                path_payoffs[i] += disc * pay
                cool_state[i] = cooldown  # reset cooldown after exercising
            else:
                q = 0.0
                pay = 0.0
                # countdown cooldown if in effect
                if cool_state[i] > 0:
                    cool_state[i] -= 1
            records.append({
                "path": i,
                "time_step": j,
                "spot": price[i],
                "q_exercised_so_far": q_before,
                "q_t": q,
                "payoff": pay,
            })
    # if _print_results: print(f'csv_path: {csv_path}')
    pd.DataFrame(records).to_csv(csv_path, index=False)

    price_estimate = path_payoffs.mean()
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_paths, n_paths)
        boot_means[b] = path_payoffs[idx].mean()
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    if _print_results: 
        print(
            f"Swing option price: {price_estimate:.4f}\n"
            f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]"
        )
    return price_estimate, (ci_low, ci_high)