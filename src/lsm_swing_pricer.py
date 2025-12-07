"""Least-squares Monte Carlo pricer for swing options."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numba import jit
from scipy import special
from sklearn.linear_model import Lasso, Ridge

from .swing_contract import SwingContract


@jit(nopython=True)
def _fill_basis(X_poly, x, basis_code, degree):
    # basis_code: 0=power, 1=laguerre, 2=hermite, 3=chebyshev
    n_samples = x.shape[0]

    if basis_code == 0:  # Power
        X_poly[:, 0] = 1.0
        for k in range(1, degree + 1):
            X_poly[:, k] = x**k

    elif basis_code == 1:  # Laguerre
        # L_0 = 1, L_1 = 1-x
        # (k+1) L_{k+1} = (2k+1-x)L_k - k L_{k-1}
        X_poly[:, 0] = 1.0
        if degree >= 1:
            X_poly[:, 1] = 1.0 - x
        for k in range(1, degree):
            # Compute L_{k+1} at column k+1
            # Indices: k-1 is col k, k is col k+1? No.
            # Col 0 is L_0. Col k is L_k.
            # We want L_{k+1}.
            # L_{k+1} = ((2k+1-x)*L_k - k*L_{k-1}) / (k+1)
            # X_poly[:, k] is L_k. X_poly[:, k-1] is L_{k-1}.
            X_poly[:, k + 1] = ((2 * k + 1 - x) * X_poly[:, k] - k * X_poly[:, k - 1]) / (k + 1)

    elif basis_code == 2:  # Hermite
        # H_0 = 1, H_1 = 2x
        # H_{k+1} = 2x H_k - 2k H_{k-1}
        X_poly[:, 0] = 1.0
        if degree >= 1:
            X_poly[:, 1] = 2.0 * x
        for k in range(1, degree):
            X_poly[:, k + 1] = 2.0 * x * X_poly[:, k] - 2.0 * k * X_poly[:, k - 1]

    elif basis_code == 3:  # Chebyshev
        # T_0 = 1, T_1 = x
        # T_{k+1} = 2x T_k - T_{k-1}
        X_poly[:, 0] = 1.0
        if degree >= 1:
            X_poly[:, 1] = x
        for k in range(1, degree):
            X_poly[:, k + 1] = 2.0 * x * X_poly[:, k] - X_poly[:, k - 1]


@jit(nopython=True)
def _lsm_core_ols(prices, strike, qmax, R, cooldown, cost_coeff, cost_exp, df, poly_degree, basis_code):
    n_paths, n_steps = prices.shape

    exercise_cost_qmax = cost_coeff * (qmax**cost_exp)

    # values: (cooldown+1, R+1, n_paths)
    values = np.zeros((cooldown + 1, R + 1, n_paths))
    # exercise: (cooldown+1, R+1, n_paths, n_steps)
    exercise = np.zeros((cooldown + 1, R + 1, n_paths, n_steps), dtype=np.bool_)

    # Terminal step
    payoff_T_gross = qmax * np.maximum(prices[:, -1] - strike, 0.0)
    payoff_T_net = payoff_T_gross - exercise_cost_qmax
    itm_T = payoff_T_gross > 0.0

    for c in range(cooldown + 1):
        for r in range(1, R + 1):
            if c == 0:
                values[c, r] = payoff_T_net
                for i in range(n_paths):
                    if itm_T[i]:
                        exercise[c, r, i, n_steps - 1] = True
            else:
                values[c, r] = 0.0

    X_poly = np.empty((n_paths, poly_degree + 1))

    for j in range(n_steps - 2, -1, -1):
        price = prices[:, j]
        payoff_gross = qmax * np.maximum(price - strike, 0.0)
        payoff_net = payoff_gross - exercise_cost_qmax

        # Basis construction
        if basis_code == 3:  # Chebyshev normalization
            p_min = price.min()
            p_max = price.max()
            if p_max > p_min:
                mid = 0.5 * (p_min + p_max)
                scale = p_max - p_min
                x_norm = 2.0 * (price - mid) / scale
            else:
                x_norm = np.zeros_like(price)
            _fill_basis(X_poly, x_norm, basis_code, poly_degree)
        else:
            _fill_basis(X_poly, price, basis_code, poly_degree)

        mask = payoff_gross > 0
        mask_indices = np.nonzero(mask)[0]
        use_mask = len(mask_indices) >= poly_degree + 1

        if use_mask:
            Xm = X_poly[mask_indices]
            # SVD decomposition for stability (matches lstsq)
            U, S, Vt = np.linalg.svd(Xm, full_matrices=False)
        else:
            Xm = X_poly
            U, S, Vt = np.linalg.svd(Xm, full_matrices=False)

        # Invert singular values with threshold (rcond)
        if len(S) > 0:
            threshold = np.max(S) * 1e-12
            S_inv = np.zeros_like(S)
            mask_S = S > threshold
            S_inv[mask_S] = 1.0 / S[mask_S]
        else:
            S_inv = S  # empty

        old_vals = values.copy()
        new_vals = values.copy()

        # Iterate cooldown states and rights remaining
        for c in range(cooldown + 1):
            for r in range(1, R + 1):
                # Keep
                c_keep = max(c - 1, 0)
                y_keep = df * old_vals[c_keep, r]

                if use_mask:
                    ym = y_keep[mask_indices]
                else:
                    ym = y_keep

                # OLS via SVD
                # beta = Vt.T @ (S_inv * (U.T @ ym))
                # U.T @ ym
                tmp = U.T @ ym
                # S_inv * tmp
                tmp = S_inv * tmp
                # Vt.T @ tmp
                beta = Vt.T @ tmp

                cont_keep = X_poly @ beta

                if c == 0:
                    # Exercise
                    c_ex = cooldown
                    y_ex = df * old_vals[c_ex, r - 1]

                    if use_mask:
                        ym_ex = y_ex[mask_indices]
                    else:
                        ym_ex = y_ex

                    tmp_ex = U.T @ ym_ex
                    tmp_ex = S_inv * tmp_ex
                    beta_ex = Vt.T @ tmp_ex

                    cont_ex = X_poly @ beta_ex

                    # Update exercise and values
                    for i in range(n_paths):
                        if payoff_gross[i] > 0:
                            if payoff_net[i] + cont_ex[i] > cont_keep[i]:
                                exercise[c, r, i, j] = True
                                new_vals[c, r, i] = payoff_net[i] + y_ex[i]
                            else:
                                new_vals[c, r, i] = y_keep[i]
                        else:
                            new_vals[c, r, i] = y_keep[i]
                else:
                    # In cooldown
                    new_vals[c, r] = y_keep

        values = new_vals

    return exercise


def _regress(
    X: np.ndarray,
    y: np.ndarray,
    degree: int,
    mask: Optional[np.ndarray] = None,
    reg_type: str = "none",
    reg_alpha: float = 0.0,
) -> np.ndarray:
    """Return fitted values of polynomial regression with optional regularization."""
    if mask is not None and mask.sum() >= degree + 1:
        Xm = X[mask]
        ym = y[mask]
    else:
        Xm = X
        ym = y

    if reg_type == "none":
        # Standard OLS regression (default Longstaff-Schwartz behavior)
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        return X @ beta

    if reg_type == "ridge":
        # Ridge regression adds L2 penalty to stabilize coefficients
        model = Ridge(alpha=reg_alpha, fit_intercept=False)
        model.fit(Xm, ym)
        return model.predict(X)

    if reg_type == "lasso":
        # Lasso regression adds L1 penalty to promote sparsity / reduce overfit
        model = Lasso(alpha=reg_alpha, fit_intercept=False, max_iter=10000)
        model.fit(Xm, ym)
        return model.predict(X)

    raise ValueError(f"Unsupported reg_type '{reg_type}'. Expected 'none', 'ridge', or 'lasso'.")


def price_swing_option_lsm(
    contract: SwingContract,
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    poly_degree: int = 2,
    basis_type: str = "power",
    reg_type: str = "none",
    reg_alpha: float = 1e-6,
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
    basis_type : str, optional
        Basis family for regression. Supported: ``power`` (default), ``laguerre``,
        ``hermite``, ``chebyshev``.
    reg_type : str, optional
        Regression regularization: ``none`` (default OLS), ``ridge`` or ``lasso``.
    reg_alpha : float, optional
        Regularization strength (ignored when ``reg_type='none'``).
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

    basis_type = basis_type.lower()
    valid_basis = {"power", "laguerre", "hermite", "chebyshev"}
    if basis_type not in valid_basis:
        raise ValueError(f"Unsupported basis_type '{basis_type}'. Expected one of {valid_basis}.")

    reg_type = reg_type.lower()
    valid_reg = {"none", "ridge", "lasso"}
    if reg_type not in valid_reg:
        raise ValueError(f"Unsupported reg_type '{reg_type}'. Expected one of {valid_reg}.")

    df = contract.discount_factor
    strike = contract.strike
    qmax = contract.q_max
    cooldown = max(0, int(getattr(contract, "min_refraction_periods", 0)))
    cost_coeff = contract.c_cost
    cost_exp = contract.gamma_cost
    exercise_cost_qmax = cost_coeff * (qmax**cost_exp)

    # number of discrete rights (assumes Q_max multiple of q_max)
    R = int(round(contract.Q_max / qmax))

    if reg_type == "none":
        basis_map = {"power": 0, "laguerre": 1, "hermite": 2, "chebyshev": 3}
        basis_code = basis_map[basis_type]

        exercise = _lsm_core_ols(
            prices, strike, qmax, R, cooldown, cost_coeff, cost_exp, df, poly_degree, basis_code
        )
    else:
        # Add cooldown state c in [0..cooldown]. c>0 means must wait c more periods to be allowed to exercise
        values = np.zeros((cooldown + 1, R + 1, n_paths))
        exercise = np.zeros((cooldown + 1, R + 1, n_paths, n_steps), dtype=bool)

        payoff_T_gross = qmax * np.maximum(prices[:, -1] - strike, 0.0)
        payoff_T_net = payoff_T_gross - exercise_cost_qmax
        itm_T = payoff_T_gross > 0.0
        # Terminal step: can exercise only if cooldown state c==0 and r>=1
        for c in range(cooldown + 1):
            for r in range(1, R + 1):
                if c == 0:
                    values[c, r] = payoff_T_net
                    exercise[c, r, itm_T, n_steps - 1] = True
                else:
                    # Cannot exercise at terminal if in cooldown; value is zero (no future)
                    values[c, r] = 0.0

        X_poly = np.empty((n_paths, poly_degree + 1))

        for j in range(n_steps - 2, -1, -1):
            price = prices[:, j]
            payoff_gross = qmax * np.maximum(price - strike, 0.0)
            payoff_net = payoff_gross - exercise_cost_qmax
            if basis_type == "power":
                # Classic monomial basis used in vanilla Longstaff-Schwartz
                X_poly[:, 0] = 1.0
                for k in range(1, poly_degree + 1):
                    X_poly[:, k] = price**k
            elif basis_type == "laguerre":
                # Using Laguerre polynomials L_k (orthogonal on [0, inf))
                for k in range(poly_degree + 1):
                    X_poly[:, k] = special.eval_laguerre(k, price)
            elif basis_type == "hermite":
                # Using physicist's Hermite polynomials H_k
                for k in range(poly_degree + 1):
                    X_poly[:, k] = special.eval_hermite(k, price)
            elif basis_type == "chebyshev":
                # Normalize prices to [-1, 1] for Chebyshev polynomials T_k
                p_min = price.min()
                p_max = price.max()
                if p_max > p_min:
                    mid = 0.5 * (p_min + p_max)
                    scale = p_max - p_min
                    x_norm = 2.0 * (price - mid) / scale
                else:
                    x_norm = np.zeros_like(price)
                for k in range(poly_degree + 1):
                    X_poly[:, k] = special.eval_chebyt(k, x_norm)
            else:  # pragma: no cover
                raise RuntimeError("basis_type validation failed unexpectedly")
            mask = payoff_gross > 0
            old_vals = values.copy()
            new_vals = values.copy()
            # Iterate cooldown states and rights remaining
            for c in range(cooldown + 1):
                for r in range(1, R + 1):
                    # If we keep (no exercise now): cooldown counts down (cannot go below 0)
                    c_keep = max(c - 1, 0)
                    y_keep = df * old_vals[c_keep, r]
                    cont_keep = _regress(
                        X_poly,
                        y_keep,
                        poly_degree,
                        mask,
                        reg_type=reg_type,
                        reg_alpha=reg_alpha,
                    )

                    if c == 0:
                        # If we exercise now: cooldown resets to full, rights reduce by 1
                        c_ex = cooldown
                        y_ex = df * old_vals[c_ex, r - 1]
                        cont_ex = _regress(
                            X_poly,
                            y_ex,
                            poly_degree,
                            mask,
                            reg_type=reg_type,
                            reg_alpha=reg_alpha,
                        )
                        exc = (payoff_net + cont_ex > cont_keep) & (payoff_gross > 0)
                        exercise[c, r, exc, j] = True
                        new_vals[c, r] = np.where(exc, payoff_net + y_ex, y_keep)
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
        disc = df**j
        for i in range(n_paths):
            r = rights[i]
            c = cool_state[i]
            q_before = q_used[i]
            if r > 0 and exercise[c, r, i, j]:
                q = min(qmax, contract.Q_max - q_before)
                payoff_gross = q * max(price[i] - strike, 0.0)
                pay_cost = cost_coeff * (q**cost_exp)
                pay = payoff_gross - pay_cost
                rights[i] -= 1
                q_used[i] += q
                path_payoffs[i] += disc * pay
                cool_state[i] = cooldown  # reset cooldown after exercising
            else:
                q = 0.0
                payoff_gross = 0.0
                pay_cost = 0.0
                pay = 0.0
                # countdown cooldown if in effect
                if cool_state[i] > 0:
                    cool_state[i] -= 1
            records.append(
                {
                    "path": i,
                    "time_step": j,
                    "spot": price[i],
                    "q_exercised_so_far": q_before,
                    "q_t": q,
                    "payoff": pay,
                    "payoff_gross": payoff_gross,
                    "exercise_cost": pay_cost,
                }
            )
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
        print(f"Swing option price: {price_estimate:.4f}\n95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        print(
            f"LSM settings -> basis: {basis_type}, degree: {poly_degree}, "
            f"reg: {reg_type}, alpha: {reg_alpha:.2e}"
        )
    return price_estimate, (ci_low, ci_high)
