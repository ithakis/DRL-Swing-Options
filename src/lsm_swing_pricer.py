"""Least-squares Monte Carlo pricer for swing options."""

from dataclasses import dataclass
from itertools import product
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
    profitable_T = payoff_T_net > 0.0  # gate on NET profitability, not just ITM

    for c in range(cooldown + 1):
        for r in range(1, R + 1):
            if c == 0:
                # Only exercise at terminal if net payoff is positive
                for i in range(n_paths):
                    if profitable_T[i]:
                        values[c, r, i] = payoff_T_net[i]
                        exercise[c, r, i, n_steps - 1] = True
                    else:
                        values[c, r, i] = 0.0
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
    min_obs = X.shape[1]
    if mask is not None and mask.sum() >= min_obs:
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


def _fit_beta(
    X: np.ndarray,
    y: np.ndarray,
    degree: int,
    mask: Optional[np.ndarray],
    reg_type: str,
    reg_alpha: float,
) -> np.ndarray:
    min_obs = X.shape[1]
    if mask is not None and mask.sum() >= min_obs:
        Xm = X[mask]
        ym = y[mask]
    else:
        Xm = X
        ym = y

    if reg_type == "none":
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=1e-12)
        return beta

    if reg_type == "ridge":
        model = Ridge(alpha=reg_alpha, fit_intercept=False)
        model.fit(Xm, ym)
        return model.coef_

    if reg_type == "lasso":
        model = Lasso(alpha=reg_alpha, fit_intercept=False, max_iter=10000)
        model.fit(Xm, ym)
        return model.coef_

    raise ValueError(f"Unsupported reg_type '{reg_type}'. Expected 'none', 'ridge', or 'lasso'.")


def _validate_state_mode(state_mode: str) -> str:
    mode = state_mode.lower()
    valid_modes = {"reduced", "full"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported state_mode '{state_mode}'. Expected one of {valid_modes}.")
    return mode


def _extract_state_paths(
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    state_mode: str,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[str, ...]]:
    _, S, X, Y = dataset
    if state_mode == "reduced":
        return (np.asarray(S, dtype=np.float64),), ("spot",)
    return (
        np.asarray(S, dtype=np.float64),
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
    ), ("spot", "x_factor", "y_factor")


def _enumerate_multi_indices(n_features: int, degree: int) -> np.ndarray:
    multi_indices = []
    for powers in product(range(degree + 1), repeat=n_features):
        if sum(powers) <= degree:
            multi_indices.append(powers)
    return np.asarray(multi_indices, dtype=np.int64)


def _build_univariate_basis(values: np.ndarray, basis: str, degree: int) -> np.ndarray:
    n_samples = values.shape[0]
    X_uni = np.empty((n_samples, degree + 1), dtype=np.float64)

    if basis == "power":
        X_uni[:, 0] = 1.0
        for k in range(1, degree + 1):
            X_uni[:, k] = values**k
        return X_uni

    if basis == "laguerre":
        for k in range(degree + 1):
            X_uni[:, k] = special.eval_laguerre(k, values)
        return X_uni

    if basis == "hermite":
        for k in range(degree + 1):
            X_uni[:, k] = special.eval_hermite(k, values)
        return X_uni

    if basis == "chebyshev":
        for k in range(degree + 1):
            X_uni[:, k] = special.eval_chebyt(k, values)
        return X_uni

    raise ValueError(f"Unsupported basis_type '{basis}'. Expected power, laguerre, hermite, chebyshev.")


def _build_lsm_feature_matrix(
    state: np.ndarray,
    basis_type: str,
    degree: int,
    multi_indices: np.ndarray,
    chebyshev_params: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    basis = basis_type.lower()
    n_samples, n_features = state.shape

    if chebyshev_params is None and basis == "chebyshev":
        chebyshev_params = np.zeros((n_features, 2), dtype=np.float64)
        transformed = np.empty_like(state)
        for feature_idx in range(n_features):
            values = state[:, feature_idx]
            value_min = float(np.min(values))
            value_max = float(np.max(values))
            if value_max > value_min:
                mid = 0.5 * (value_min + value_max)
                scale = value_max - value_min
                transformed[:, feature_idx] = 2.0 * (values - mid) / scale
            else:
                mid = value_min
                scale = 0.0
                transformed[:, feature_idx] = 0.0
            chebyshev_params[feature_idx, 0] = mid
            chebyshev_params[feature_idx, 1] = scale
    elif basis == "chebyshev":
        transformed = np.empty_like(state)
        for feature_idx in range(n_features):
            mid = chebyshev_params[feature_idx, 0]
            scale = chebyshev_params[feature_idx, 1]
            if scale > 0.0:
                transformed[:, feature_idx] = 2.0 * (state[:, feature_idx] - mid) / scale
            else:
                transformed[:, feature_idx] = 0.0
    else:
        transformed = state

    univariate_bases = [
        _build_univariate_basis(transformed[:, feature_idx], basis, degree) for feature_idx in range(n_features)
    ]

    X_basis = np.empty((n_samples, len(multi_indices)), dtype=np.float64)
    for col_idx, powers in enumerate(multi_indices):
        feature_col = np.ones(n_samples, dtype=np.float64)
        for feature_idx, power in enumerate(powers):
            feature_col *= univariate_bases[feature_idx][:, power]
        X_basis[:, col_idx] = feature_col

    return X_basis, chebyshev_params


def _build_lsm_basis(
    prices: np.ndarray,
    basis_type: str,
    degree: int,
    chebyshev_params: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    basis = basis_type.lower()
    n_samples = prices.shape[0]
    X_poly = np.empty((n_samples, degree + 1), dtype=np.float64)

    if basis == "power":
        X_poly[:, 0] = 1.0
        for k in range(1, degree + 1):
            X_poly[:, k] = prices**k
        return X_poly, chebyshev_params

    if basis == "laguerre":
        for k in range(degree + 1):
            X_poly[:, k] = special.eval_laguerre(k, prices)
        return X_poly, chebyshev_params

    if basis == "hermite":
        for k in range(degree + 1):
            X_poly[:, k] = special.eval_hermite(k, prices)
        return X_poly, chebyshev_params

    if basis == "chebyshev":
        if chebyshev_params is None:
            p_min = float(np.min(prices))
            p_max = float(np.max(prices))
            if p_max > p_min:
                mid = 0.5 * (p_min + p_max)
                scale = p_max - p_min
                x_norm = 2.0 * (prices - mid) / scale
            else:
                mid = p_min
                scale = 0.0
                x_norm = np.zeros_like(prices)
            chebyshev_params = (float(mid), float(scale))
        else:
            mid, scale = chebyshev_params
            if scale > 0.0:
                x_norm = 2.0 * (prices - mid) / scale
            else:
                x_norm = np.zeros_like(prices)

        for k in range(degree + 1):
            X_poly[:, k] = special.eval_chebyt(k, x_norm)
        return X_poly, chebyshev_params

    raise ValueError(f"Unsupported basis_type '{basis_type}'. Expected power, laguerre, hermite, chebyshev.")


@dataclass
class LSMEstimators:
    """Frozen regression coefficients from in-sample LSM training."""

    state_mode: str
    feature_names: Tuple[str, ...]
    basis_type: str
    poly_degree: int
    reg_type: str
    reg_alpha: float
    n_steps: int
    cooldown: int
    rights: int
    betas_keep: np.ndarray
    betas_ex: np.ndarray
    multi_indices: np.ndarray
    chebyshev_params: Optional[np.ndarray]
    n_actions: int = 2
    action_grid: Optional[np.ndarray] = None
    n_cap_levels: int = 0


def _compute_projection_matrix(
    X: np.ndarray,
    mask: Optional[np.ndarray],
    reg_type: str = "none",
    reg_alpha: float = 0.0,
) -> np.ndarray:
    """Precompute projection matrix P such that beta = P @ y for any target y.

    For OLS: P = (X^T X)^{-1} X^T  (via SVD pseudoinverse).
    For Ridge: P = (X^T X + alpha*I)^{-1} X^T.
    Returns P of shape (n_features, n_samples) applicable to the full X.
    """
    min_obs = X.shape[1]
    if mask is not None and mask.sum() >= min_obs:
        Xm = X[mask]
    else:
        Xm = X
        mask = None  # use full matrix

    if reg_type == "none":
        # SVD-based pseudoinverse: P = V S^{-1} U^T
        U, S, Vt = np.linalg.svd(Xm, full_matrices=False)
        threshold = np.max(S) * 1e-12 if len(S) > 0 else 0.0
        S_inv = np.zeros_like(S)
        valid = S > threshold
        S_inv[valid] = 1.0 / S[valid]
        # P_masked = Vt.T @ diag(S_inv) @ U.T  — shape (n_features, n_masked)
        P_masked = (Vt.T * S_inv[np.newaxis, :]) @ U.T
    elif reg_type == "ridge":
        XtX = Xm.T @ Xm
        XtX += reg_alpha * np.eye(XtX.shape[0])
        P_masked = np.linalg.solve(XtX, Xm.T)
    else:
        raise ValueError(f"Projection matrix not supported for reg_type='{reg_type}'")

    return P_masked, mask


def _fit_betas_batch(
    P_masked: np.ndarray,
    mask: Optional[np.ndarray],
    Y_targets: np.ndarray,
) -> np.ndarray:
    """Fit betas for multiple targets at once using precomputed projection matrix.

    P_masked: (n_features, n_masked) projection matrix
    Y_targets: (n_targets, n_paths) target values
    Returns: (n_targets, n_features) beta coefficients
    """
    if mask is not None:
        return (P_masked @ Y_targets[:, mask].T).T
    return (P_masked @ Y_targets.T).T


def price_swing_option_lsm(
    contract: SwingContract,
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    poly_degree: int = 2,
    basis_type: str = "power",
    state_mode: str = "reduced",
    reg_type: str = "none",
    reg_alpha: float = 1e-6,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
    csv_path: Optional[str] = "swing_option_lsm_paths.parquet",
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
        Destination for Parquet log of optimal exercises (set to None to disable writing).
    """
    state_mode = _validate_state_mode(state_mode)
    t, S, _, _ = dataset
    # Upcast to float64 for numerical stability and numba JIT friendliness
    prices = np.asarray(S, dtype=np.float64)  # all decision prices (including initial spot at t=0)
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

    if reg_type == "none" and state_mode == "reduced":
        basis_map = {"power": 0, "laguerre": 1, "hermite": 2, "chebyshev": 3}
        basis_code = basis_map[basis_type]

        exercise = _lsm_core_ols(
            prices, strike, qmax, R, cooldown, cost_coeff, cost_exp, df, poly_degree, basis_code
        )
    else:
        state_paths, _ = _extract_state_paths(dataset, state_mode)
        multi_indices = _enumerate_multi_indices(len(state_paths), poly_degree)
        chebyshev_params = (
            np.zeros((n_steps - 1, len(state_paths), 2), dtype=np.float64)
            if basis_type == "chebyshev"
            else None
        )
        # Add cooldown state c in [0..cooldown]. c>0 means must wait c more periods to be allowed to exercise
        values = np.zeros((cooldown + 1, R + 1, n_paths))
        exercise = np.zeros((cooldown + 1, R + 1, n_paths, n_steps), dtype=bool)

        payoff_T_gross = qmax * np.maximum(prices[:, -1] - strike, 0.0)
        payoff_T_net = payoff_T_gross - exercise_cost_qmax
        profitable_T = payoff_T_net > 0.0  # gate on NET profitability, not just ITM
        # Terminal step: can exercise only if cooldown state c==0 and r>=1
        for c in range(cooldown + 1):
            for r in range(1, R + 1):
                if c == 0:
                    # Only exercise at terminal if net payoff is positive
                    values[c, r] = np.where(profitable_T, payoff_T_net, 0.0)
                    exercise[c, r, profitable_T, n_steps - 1] = True
                else:
                    # Cannot exercise at terminal if in cooldown; value is zero (no future)
                    values[c, r] = 0.0

        for j in range(n_steps - 2, -1, -1):
            price = prices[:, j]
            state_t = np.column_stack([state_path[:, j] for state_path in state_paths])
            payoff_gross = qmax * np.maximum(price - strike, 0.0)
            payoff_net = payoff_gross - exercise_cost_qmax
            profitable_now = payoff_net > 0.0
            X_poly, params = _build_lsm_feature_matrix(state_t, basis_type, poly_degree, multi_indices)
            if chebyshev_params is not None:
                chebyshev_params[j] = params
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
                        exc = (payoff_net + cont_ex > cont_keep) & profitable_now
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
    if csv_path:
        # if _print_results: print(f'csv_path: {csv_path}')
        pd.DataFrame(records).to_parquet(
            csv_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
            compression_level=22,
        )

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
            f"LSM settings -> state: {state_mode}, basis: {basis_type}, degree: {poly_degree}, "
            f"reg: {reg_type}, alpha: {reg_alpha:.2e}"
        )
    return price_estimate, (ci_low, ci_high)


def fit_lsm_estimators(
    contract: SwingContract,
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    poly_degree: int = 2,
    basis_type: str = "power",
    state_mode: str = "reduced",
    reg_type: str = "none",
    reg_alpha: float = 1e-6,
    n_actions: int = 2,
) -> LSMEstimators:
    """Fit LSM continuation-value estimators on the in-sample dataset.

    Parameters
    ----------
    n_actions : int
        Number of candidate exercise levels including zero.
        ``n_actions=2`` reproduces the original bang-bang (0 or q_max).
        ``n_actions=5`` gives {0, q_max/4, q_max/2, 3q_max/4, q_max}.
    """
    state_mode = _validate_state_mode(state_mode)
    state_paths, feature_names = _extract_state_paths(dataset, state_mode)
    prices = state_paths[0]
    n_paths, n_steps = prices.shape
    if n_steps != contract.n_rights:
        raise ValueError("Mismatch between paths and contract rights")

    basis = basis_type.lower()
    valid_basis = {"power", "laguerre", "hermite", "chebyshev"}
    if basis not in valid_basis:
        raise ValueError(f"Unsupported basis_type '{basis_type}'. Expected one of {valid_basis}.")

    reg = reg_type.lower()
    valid_reg = {"none", "ridge", "lasso"}
    if reg not in valid_reg:
        raise ValueError(f"Unsupported reg_type '{reg_type}'. Expected one of {valid_reg}.")

    df = contract.discount_factor
    strike = contract.strike
    qmax = contract.q_max
    cooldown = max(0, int(getattr(contract, "min_refraction_periods", 0)))
    cost_coeff = contract.c_cost
    cost_exp = contract.gamma_cost
    multi_indices = _enumerate_multi_indices(len(state_paths), poly_degree)

    # Build action grid: {0, Δq, 2Δq, ..., q_max}
    if n_actions < 2:
        raise ValueError("n_actions must be >= 2")
    action_grid = np.linspace(0.0, qmax, n_actions)  # [0, ..., q_max]
    dq = action_grid[1] if n_actions > 1 else qmax  # smallest non-zero step

    # Inventory DP: capacity levels in units of dq
    n_cap_levels = int(round(contract.Q_max / dq)) + 1  # 0, dq, ..., Q_max

    # Precompute payoffs for each non-zero action
    M = n_actions  # total actions including zero
    # action_grid[0] = 0, action_grid[1..M-1] are positive exercise quantities

    # DP arrays: values[cooldown+1, n_cap_levels, n_paths]
    values = np.zeros((cooldown + 1, n_cap_levels, n_paths), dtype=np.float64)

    # Terminal step: for each capacity level, try all feasible actions
    terminal_price = prices[:, -1]
    for c in range(cooldown + 1):
        for v in range(n_cap_levels):
            if c > 0:
                values[c, v] = 0.0
                continue
            remaining_dq = (n_cap_levels - 1) - v  # remaining capacity in units of dq
            best_val = np.zeros(n_paths, dtype=np.float64)
            for m in range(1, M):
                action_dq = m  # action in units of dq
                if action_dq > remaining_dq:
                    break
                q_m = action_grid[m]
                payoff_gross = q_m * np.maximum(terminal_price - strike, 0.0)
                payoff_net = payoff_gross - cost_coeff * (q_m**cost_exp)
                profitable = payoff_net > 0.0
                best_val = np.where(profitable & (payoff_net > best_val), payoff_net, best_val)
            values[c, v] = best_val

    n_basis = len(multi_indices)
    # betas_keep[step, cooldown, cap_level, n_basis]
    betas_keep = np.zeros((n_steps - 1, cooldown + 1, n_cap_levels, n_basis), dtype=np.float64)
    # betas_ex[step, cooldown, cap_level, M-1, n_basis] — one per non-zero action
    betas_ex = np.zeros((n_steps - 1, cooldown + 1, n_cap_levels, M - 1, n_basis), dtype=np.float64)
    chebyshev_params = (
        np.zeros((n_steps - 1, len(state_paths), 2), dtype=np.float64) if basis == "chebyshev" else None
    )

    for j in range(n_steps - 2, -1, -1):
        price = prices[:, j]
        state_t = np.column_stack([state_path[:, j] for state_path in state_paths])
        X_poly, params = _build_lsm_feature_matrix(state_t, basis, poly_degree, multi_indices)
        if chebyshev_params is not None:
            chebyshev_params[j] = params

        # ITM mask for regression fitting (any non-zero exercise could be in the money)
        itm_mask = (price - strike) > 0.0

        # Precompute projection matrix once per step (shared across all DP states)
        if reg in ("none", "ridge"):
            P_masked, eff_mask = _compute_projection_matrix(X_poly, itm_mask, reg, reg_alpha)
        else:
            P_masked, eff_mask = None, None

        # Precompute gross payoff and net payoff for each non-zero action
        payoffs_net = np.zeros((M - 1, n_paths), dtype=np.float64)
        profitable = np.zeros((M - 1, n_paths), dtype=bool)
        for m in range(1, M):
            q_m = action_grid[m]
            pg = q_m * np.maximum(price - strike, 0.0)
            pn = pg - cost_coeff * (q_m**cost_exp)
            payoffs_net[m - 1] = pn
            profitable[m - 1] = pn > 0.0

        old_vals = values
        new_vals = values.copy()

        for c in range(cooldown + 1):
            for v in range(n_cap_levels):
                remaining_dq = (n_cap_levels - 1) - v

                # Keep: cooldown decrements
                c_keep = max(c - 1, 0)
                y_keep = df * old_vals[c_keep, v]

                if P_masked is not None:
                    beta_keep = _fit_betas_batch(P_masked, eff_mask, y_keep[np.newaxis, :])[0]
                else:
                    beta_keep = _fit_beta(X_poly, y_keep, poly_degree, itm_mask, reg, reg_alpha)
                betas_keep[j, c, v] = beta_keep
                cont_keep = X_poly @ beta_keep

                if c == 0:
                    # Try each non-zero action
                    best_value = y_keep.copy()  # default: keep
                    best_total = cont_keep.copy()

                    for m in range(1, M):
                        action_dq = m
                        if action_dq > remaining_dq:
                            break
                        # After exercising q_m, capacity used increases by m dq-units
                        v_after = v + action_dq
                        c_ex = cooldown
                        y_ex = df * old_vals[c_ex, v_after]

                        if P_masked is not None:
                            beta_ex = _fit_betas_batch(P_masked, eff_mask, y_ex[np.newaxis, :])[0]
                        else:
                            beta_ex = _fit_beta(X_poly, y_ex, poly_degree, itm_mask, reg, reg_alpha)
                        betas_ex[j, c, v, m - 1] = beta_ex
                        cont_ex = X_poly @ beta_ex

                        total_ex = payoffs_net[m - 1] + cont_ex
                        # Only consider this action if net payoff is positive (profitability gate)
                        improve = profitable[m - 1] & (total_ex > best_total)
                        best_total = np.where(improve, total_ex, best_total)
                        best_value = np.where(improve, payoffs_net[m - 1] + y_ex, best_value)

                    new_vals[c, v] = best_value
                else:
                    # In cooldown: cannot exercise
                    new_vals[c, v] = y_keep

        values = new_vals

    return LSMEstimators(
        state_mode=state_mode,
        feature_names=feature_names,
        basis_type=basis,
        poly_degree=poly_degree,
        reg_type=reg,
        reg_alpha=reg_alpha,
        n_steps=n_steps,
        cooldown=cooldown,
        rights=int(round(contract.Q_max / qmax)),  # keep for backward compat
        betas_keep=betas_keep,
        betas_ex=betas_ex,
        multi_indices=multi_indices,
        chebyshev_params=chebyshev_params,
        n_actions=n_actions,
        action_grid=action_grid,
        n_cap_levels=n_cap_levels,
    )


def price_swing_option_lsm_oos(
    contract: SwingContract,
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    estimators: LSMEstimators,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
    csv_path: Optional[str] = "swing_option_lsm_paths.parquet",
    _print_results: bool = False,
) -> Tuple[float, Tuple[float, float]]:
    """Price swing option out-of-sample using frozen in-sample LSM estimators."""
    state_paths, _ = _extract_state_paths(dataset, estimators.state_mode)
    prices = state_paths[0]
    n_paths, n_steps = prices.shape
    if n_steps != estimators.n_steps:
        raise ValueError("Mismatch between estimator steps and evaluation paths")
    if n_steps != contract.n_rights:
        raise ValueError("Mismatch between paths and contract rights")

    df = contract.discount_factor
    strike = contract.strike
    qmax = contract.q_max
    cooldown = estimators.cooldown
    cost_coeff = contract.c_cost
    cost_exp = contract.gamma_cost
    n_actions = estimators.n_actions
    action_grid = estimators.action_grid
    n_cap_levels = estimators.n_cap_levels
    M = n_actions

    # DP backward pass to build exercise decisions
    values = np.zeros((cooldown + 1, n_cap_levels, n_paths), dtype=np.float64)
    # exercise_action[cooldown+1, n_cap_levels, n_paths, n_steps] stores the chosen action index (0=keep)
    exercise_action = np.zeros((cooldown + 1, n_cap_levels, n_paths, n_steps), dtype=np.int8)

    # Terminal step
    terminal_price = prices[:, -1]
    for c in range(cooldown + 1):
        for v in range(n_cap_levels):
            if c > 0:
                values[c, v] = 0.0
                continue
            remaining_dq = (n_cap_levels - 1) - v
            best_val = np.zeros(n_paths, dtype=np.float64)
            best_action = np.zeros(n_paths, dtype=np.int8)
            for m in range(1, M):
                action_dq = m
                if action_dq > remaining_dq:
                    break
                q_m = action_grid[m]
                payoff_gross = q_m * np.maximum(terminal_price - strike, 0.0)
                payoff_net = payoff_gross - cost_coeff * (q_m**cost_exp)
                profitable = payoff_net > 0.0
                improve = profitable & (payoff_net > best_val)
                best_val = np.where(improve, payoff_net, best_val)
                best_action = np.where(improve, m, best_action)
            values[c, v] = best_val
            exercise_action[c, v, :, n_steps - 1] = best_action

    for j in range(n_steps - 2, -1, -1):
        price = prices[:, j]
        state_t = np.column_stack([state_path[:, j] for state_path in state_paths])

        if estimators.basis_type == "chebyshev":
            params = estimators.chebyshev_params[j]
            X_poly, _ = _build_lsm_feature_matrix(
                state_t,
                estimators.basis_type,
                estimators.poly_degree,
                estimators.multi_indices,
                params,
            )
        else:
            X_poly, _ = _build_lsm_feature_matrix(
                state_t,
                estimators.basis_type,
                estimators.poly_degree,
                estimators.multi_indices,
            )

        # Precompute payoffs for each non-zero action
        payoffs_net = np.zeros((M - 1, n_paths), dtype=np.float64)
        profitable = np.zeros((M - 1, n_paths), dtype=bool)
        for m in range(1, M):
            q_m = action_grid[m]
            pg = q_m * np.maximum(price - strike, 0.0)
            pn = pg - cost_coeff * (q_m**cost_exp)
            payoffs_net[m - 1] = pn
            profitable[m - 1] = pn > 0.0

        old_vals = values
        new_vals = values.copy()
        for c in range(cooldown + 1):
            for v in range(n_cap_levels):
                remaining_dq = (n_cap_levels - 1) - v
                c_keep = max(c - 1, 0)
                y_keep = df * old_vals[c_keep, v]
                beta_keep = estimators.betas_keep[j, c, v]
                cont_keep = X_poly @ beta_keep

                if c == 0:
                    best_value = y_keep.copy()
                    best_total = cont_keep.copy()
                    best_action = np.zeros(n_paths, dtype=np.int8)

                    for m in range(1, M):
                        action_dq = m
                        if action_dq > remaining_dq:
                            break
                        v_after = v + action_dq
                        c_ex = cooldown
                        y_ex = df * old_vals[c_ex, v_after]
                        beta_ex = estimators.betas_ex[j, c, v, m - 1]
                        cont_ex = X_poly @ beta_ex
                        total_ex = payoffs_net[m - 1] + cont_ex
                        improve = profitable[m - 1] & (total_ex > best_total)
                        best_total = np.where(improve, total_ex, best_total)
                        best_value = np.where(improve, payoffs_net[m - 1] + y_ex, best_value)
                        best_action = np.where(improve, m, best_action)

                    new_vals[c, v] = best_value
                    exercise_action[c, v, :, j] = best_action
                else:
                    new_vals[c, v] = y_keep
        values = new_vals

    # Forward pass: accumulate cash flows using exercise decisions
    dq = action_grid[1] if M > 1 else qmax
    cap_used = np.zeros(n_paths, dtype=np.int32)  # in units of dq
    cool_state = np.zeros(n_paths, dtype=int)
    path_payoffs = np.zeros(n_paths)
    records = [] if csv_path else None
    for j in range(n_steps):
        price = prices[:, j]
        disc = df**j
        for i in range(n_paths):
            v = cap_used[i]
            c = cool_state[i]
            q_before = v * dq
            action_idx = exercise_action[c, v, i, j]
            if action_idx > 0:
                q_m = action_grid[action_idx]
                # Ensure feasibility
                remaining_cap = contract.Q_max - q_before
                q = min(q_m, remaining_cap)
                payoff_gross = q * max(price[i] - strike, 0.0)
                pay_cost = cost_coeff * (q**cost_exp)
                pay = payoff_gross - pay_cost
                cap_used[i] += action_idx
                path_payoffs[i] += disc * pay
                cool_state[i] = cooldown
            else:
                q = 0.0
                payoff_gross = 0.0
                pay_cost = 0.0
                pay = 0.0
                if cool_state[i] > 0:
                    cool_state[i] -= 1
            if records is not None:
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

    if csv_path and records:
        pd.DataFrame(records).to_parquet(
            csv_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
            compression_level=22,
        )

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
            f"LSM settings -> state: {estimators.state_mode}, basis: {estimators.basis_type}, "
            f"degree: {estimators.poly_degree}, reg: {estimators.reg_type}, "
            f"alpha: {estimators.reg_alpha:.2e}, n_actions: {estimators.n_actions}"
        )

    return price_estimate, (ci_low, ci_high)
