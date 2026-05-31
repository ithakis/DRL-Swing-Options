"""Shared helpers for hedging caches, HHK forwards, and risk diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .simulate_hhk_spot import no_seasonal_function
from .swing_contract import SwingContract

RL_TRACE_COLUMNS = [
    "path",
    "time_step",
    "spot_minus_strike",
    "q_exercised_norm",
    "q_remaining_norm",
    "time_to_maturity_norm",
    "normalized_time",
    "spot",
    "X_t",
    "Y_t",
    "days_since_exercise_norm",
    "q_t",
    "exercise_cost",
    "reward_discounted",
]

LSM_RAW_TRACE_COLUMNS = [
    "path",
    "time_step",
    "spot",
    "q_exercised_so_far",
    "q_t",
    "payoff",
    "payoff_gross",
    "exercise_cost",
]

SHARED_TRACE_COLUMNS = [
    "config",
    "method",
    "run_name",
    "seed",
    "path",
    "time_step",
    "time_years",
    "spot_minus_strike",
    "q_exercised_norm",
    "q_remaining_norm",
    "time_to_maturity_norm",
    "normalized_time",
    "spot",
    "X_t",
    "Y_t",
    "days_since_exercise_norm",
    "days_since_exercise",
    "q_exercised_so_far",
    "q_remaining",
    "q_t",
    "payoff_gross",
    "exercise_cost",
    "payoff_net",
    "discount_factor_step",
    "reward_discounted",
    "forward_contract_maturity",
    "strike",
    "c_cost",
    "gamma_cost",
]

TRACE_FLOAT_COLUMNS = [
    "time_years",
    "spot_minus_strike",
    "q_exercised_norm",
    "q_remaining_norm",
    "time_to_maturity_norm",
    "normalized_time",
    "spot",
    "X_t",
    "Y_t",
    "days_since_exercise_norm",
    "days_since_exercise",
    "q_exercised_so_far",
    "q_remaining",
    "q_t",
    "payoff_gross",
    "exercise_cost",
    "payoff_net",
    "discount_factor_step",
    "reward_discounted",
    "forward_contract_maturity",
    "strike",
    "c_cost",
    "gamma_cost",
]

TRACE_INT_COLUMNS = ["seed", "path", "time_step"]


@dataclass(frozen=True)
class HedgingRiskMetrics:
    mean_pnl: float
    variance_pnl: float
    std_pnl: float
    var_pnl_95: float
    cvar_pnl_95: float
    min_pnl: float
    max_pnl: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class HedgingTraceSummary:
    config: str
    method: str
    run_name: str
    seed: int
    n_paths: int
    n_rows: int
    option_price: float
    price_std: float
    confidence_95: float
    avg_total_exercised: float
    avg_exercise_count: float
    bangbangness: float
    artifact_path: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _seasonal_value(seasonal_fn: Optional[Callable[[float], float]], delivery_time: np.ndarray) -> np.ndarray:
    fn = seasonal_fn or no_seasonal_function
    vectorized = np.vectorize(fn, otypes=[np.float64])
    return vectorized(np.asarray(delivery_time, dtype=np.float64))


def hhk_forward_adjustment(
    time_to_delivery: Sequence[float] | np.ndarray,
    *,
    alpha: float,
    sigma: float,
    beta: float,
    lam: float,
    mu_J: float,
) -> np.ndarray:
    """Closed-form HHK forward adjustment for exponential jump sizes."""
    delta = np.asarray(time_to_delivery, dtype=np.float64)
    delta = np.clip(delta, 0.0, None)
    diffusion = (sigma**2 / (4.0 * alpha)) * (1.0 - np.exp(-2.0 * alpha * delta))
    jump = (lam / beta) * np.log((1.0 - mu_J * np.exp(-beta * delta)) / (1.0 - mu_J))
    return diffusion + jump


def hhk_forward_price(
    *,
    current_time: Sequence[float] | np.ndarray,
    delivery_time: float | Sequence[float] | np.ndarray,
    X_t: Sequence[float] | np.ndarray,
    Y_t: Sequence[float] | np.ndarray,
    alpha: float,
    sigma: float,
    beta: float,
    lam: float,
    mu_J: float,
    seasonal_fn: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    """Vectorized HHK forward price under the repo's no-seasonal benchmark setup."""
    current = np.asarray(current_time, dtype=np.float64)
    delivery = np.asarray(delivery_time, dtype=np.float64)
    x_t = np.asarray(X_t, dtype=np.float64)
    y_t = np.asarray(Y_t, dtype=np.float64)
    delta = np.clip(delivery - current, 0.0, None)
    seasonal = _seasonal_value(seasonal_fn, delivery)
    adjustment = hhk_forward_adjustment(delta, alpha=alpha, sigma=sigma, beta=beta, lam=lam, mu_J=mu_J)
    return np.exp(seasonal + np.exp(-alpha * delta) * x_t + np.exp(-beta * delta) * y_t + adjustment)


def compute_pnl_risk_metrics(pnl: Sequence[float], alpha: float = 0.95) -> HedgingRiskMetrics:
    """Return mean, variance, and lower-tail PnL metrics."""
    pnl_arr = np.asarray(pnl, dtype=np.float64)
    if pnl_arr.ndim != 1:
        raise ValueError("pnl must be one-dimensional")
    if pnl_arr.size == 0:
        raise ValueError("pnl must contain at least one observation")

    tail_q = float(np.quantile(pnl_arr, 1.0 - alpha))
    tail_mask = pnl_arr <= tail_q
    cvar = float(pnl_arr[tail_mask].mean()) if np.any(tail_mask) else tail_q
    variance = float(np.var(pnl_arr, ddof=1)) if pnl_arr.size > 1 else 0.0
    std = float(np.sqrt(variance))
    return HedgingRiskMetrics(
        mean_pnl=float(pnl_arr.mean()),
        variance_pnl=variance,
        std_pnl=std,
        var_pnl_95=tail_q,
        cvar_pnl_95=cvar,
        min_pnl=float(pnl_arr.min()),
        max_pnl=float(pnl_arr.max()),
    )


def summarize_trace(
    trace_df: pd.DataFrame,
    *,
    config: str,
    method: str,
    run_name: str,
    seed: int,
    artifact_path: Path,
    q_max: float,
) -> HedgingTraceSummary:
    """Compute path-level summary metrics from a normalized shared trace."""
    if trace_df.empty:
        raise ValueError("trace_df must not be empty")

    path_returns = trace_df.groupby("path", sort=False)["reward_discounted"].sum()
    total_exercised = trace_df.groupby("path", sort=False)["q_t"].sum()
    exercise_count = trace_df.groupby("path", sort=False)["q_t"].apply(
        lambda s: int(np.count_nonzero(s > 1e-6))
    )
    full_cap_mask = (trace_df["q_t"] > 1e-6) & (trace_df["q_remaining"] >= q_max - 1e-6)
    if np.any(full_cap_mask):
        bangbangness = float((trace_df.loc[full_cap_mask, "q_t"] >= 0.95 * q_max).mean())
    else:
        bangbangness = float("nan")

    price_std = float(path_returns.std(ddof=1)) if len(path_returns) > 1 else 0.0
    ci95 = 1.96 * price_std / np.sqrt(max(1, len(path_returns)))
    return HedgingTraceSummary(
        config=config,
        method=method,
        run_name=run_name,
        seed=seed,
        n_paths=int(path_returns.size),
        n_rows=int(len(trace_df)),
        option_price=float(path_returns.mean()),
        price_std=price_std,
        confidence_95=float(ci95),
        avg_total_exercised=float(total_exercised.mean()),
        avg_exercise_count=float(exercise_count.mean()),
        bangbangness=bangbangness,
        artifact_path=str(artifact_path),
    )


def ensure_trace_dtypes(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Cast normalized trace columns to compact dtypes for cache storage."""
    df = trace_df.copy()
    for column in TRACE_INT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype(np.int32)
    for column in TRACE_FLOAT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype(np.float32)
    for column in ("config", "method", "run_name"):
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df


def _shared_metadata_frame(n_rows: int, metadata: Mapping[str, Any]) -> pd.DataFrame:
    payload = {
        "config": [str(metadata["config"])] * n_rows,
        "method": [str(metadata["method"])] * n_rows,
        "run_name": [str(metadata.get("run_name", metadata["config"]))] * n_rows,
        "seed": [int(metadata["seed"])] * n_rows,
    }
    return pd.DataFrame(payload)


def _discount_factor_series(contract: SwingContract, steps: pd.Series) -> np.ndarray:
    return np.power(contract.discount_factor, steps.to_numpy(dtype=np.float64))


def normalize_rl_trace(
    rows: Sequence[Sequence[float]],
    *,
    metadata: Mapping[str, Any],
    contract: SwingContract,
    t_grid: Sequence[float],
    hhk_params: Mapping[str, Any],
    seasonal_fn: Optional[Callable[[float], float]] = None,
) -> pd.DataFrame:
    """Convert raw RL evaluation rows into the shared hedging trace schema."""
    rl_df = pd.DataFrame(rows, columns=RL_TRACE_COLUMNS)
    if rl_df.empty:
        raise ValueError("RL rows must not be empty")

    q_exercised = rl_df["q_exercised_norm"].to_numpy(dtype=np.float64) * contract.Q_max
    q_remaining = rl_df["q_remaining_norm"].to_numpy(dtype=np.float64) * contract.Q_max
    payoff_gross = rl_df["q_t"].to_numpy(dtype=np.float64) * np.maximum(
        rl_df["spot"].to_numpy(dtype=np.float64) - contract.strike,
        0.0,
    )
    payoff_net = payoff_gross - rl_df["exercise_cost"].to_numpy(dtype=np.float64)
    steps = rl_df["time_step"].astype(np.int64)
    times = np.asarray(t_grid, dtype=np.float64)[steps.to_numpy()]
    forward = hhk_forward_price(
        current_time=times,
        delivery_time=float(contract.maturity),
        X_t=rl_df["X_t"],
        Y_t=rl_df["Y_t"],
        alpha=float(hhk_params["alpha"]),
        sigma=float(hhk_params["sigma"]),
        beta=float(hhk_params["beta"]),
        lam=float(hhk_params["lam"]),
        mu_J=float(hhk_params["mu_J"]),
        seasonal_fn=seasonal_fn,
    )

    shared = pd.concat([_shared_metadata_frame(len(rl_df), metadata), rl_df[["path", "time_step"]]], axis=1)
    shared["time_years"] = times
    shared["spot_minus_strike"] = rl_df["spot_minus_strike"]
    shared["q_exercised_norm"] = rl_df["q_exercised_norm"]
    shared["q_remaining_norm"] = rl_df["q_remaining_norm"]
    shared["time_to_maturity_norm"] = rl_df["time_to_maturity_norm"]
    shared["normalized_time"] = rl_df["normalized_time"]
    shared["spot"] = rl_df["spot"]
    shared["X_t"] = rl_df["X_t"]
    shared["Y_t"] = rl_df["Y_t"]
    shared["days_since_exercise_norm"] = rl_df["days_since_exercise_norm"]
    shared["days_since_exercise"] = (
        rl_df["days_since_exercise_norm"].to_numpy(dtype=np.float64) * contract.n_rights
    )
    shared["q_exercised_so_far"] = q_exercised
    shared["q_remaining"] = q_remaining
    shared["q_t"] = rl_df["q_t"]
    shared["payoff_gross"] = payoff_gross
    shared["exercise_cost"] = rl_df["exercise_cost"]
    shared["payoff_net"] = payoff_net
    shared["discount_factor_step"] = _discount_factor_series(contract, steps)
    shared["reward_discounted"] = rl_df["reward_discounted"]
    shared["forward_contract_maturity"] = forward
    shared["strike"] = float(contract.strike)
    shared["c_cost"] = float(contract.c_cost)
    shared["gamma_cost"] = float(contract.gamma_cost)
    return ensure_trace_dtypes(shared[SHARED_TRACE_COLUMNS])


def _prior_days_since_exercise(raw_lsm_df: pd.DataFrame) -> np.ndarray:
    exercise_time = np.where(
        raw_lsm_df["q_t"].to_numpy(dtype=np.float64) > 1e-6, raw_lsm_df["time_step"], np.nan
    )
    prior_ex = (
        pd.Series(exercise_time).groupby(raw_lsm_df["path"], sort=False).transform(lambda s: s.shift().ffill())
    )
    steps = raw_lsm_df["time_step"].to_numpy(dtype=np.float64)
    prior = prior_ex.to_numpy(dtype=np.float64)
    return np.where(np.isnan(prior), steps, steps - prior)


def normalize_lsm_trace(
    raw_lsm_df: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    contract: SwingContract,
    dataset: Sequence[np.ndarray],
    hhk_params: Mapping[str, Any],
    seasonal_fn: Optional[Callable[[float], float]] = None,
) -> pd.DataFrame:
    """Enrich raw LSM path logs so they match the shared RL hedging schema."""
    if raw_lsm_df.empty:
        raise ValueError("raw_lsm_df must not be empty")

    t_grid, _, X, Y = dataset
    lsm_df = raw_lsm_df.copy()
    lsm_df = lsm_df.sort_values(["path", "time_step"], kind="stable").reset_index(drop=True)
    payoff_net = lsm_df["payoff_net"] if "payoff_net" in lsm_df.columns else lsm_df["payoff"]
    steps = lsm_df["time_step"].to_numpy(dtype=np.int64)
    paths = lsm_df["path"].to_numpy(dtype=np.int64)
    times = np.asarray(t_grid, dtype=np.float64)[steps]
    x_t = np.asarray(X, dtype=np.float64)[paths, steps]
    y_t = np.asarray(Y, dtype=np.float64)[paths, steps]
    q_exercised = lsm_df["q_exercised_so_far"].to_numpy(dtype=np.float64)
    q_remaining = contract.Q_max - q_exercised
    prior_days = _prior_days_since_exercise(lsm_df)
    forward = hhk_forward_price(
        current_time=times,
        delivery_time=float(contract.maturity),
        X_t=x_t,
        Y_t=y_t,
        alpha=float(hhk_params["alpha"]),
        sigma=float(hhk_params["sigma"]),
        beta=float(hhk_params["beta"]),
        lam=float(hhk_params["lam"]),
        mu_J=float(hhk_params["mu_J"]),
        seasonal_fn=seasonal_fn,
    )

    shared = pd.concat([_shared_metadata_frame(len(lsm_df), metadata), lsm_df[["path", "time_step"]]], axis=1)
    shared["time_years"] = times
    shared["spot_minus_strike"] = lsm_df["spot"].to_numpy(dtype=np.float64) - float(contract.strike)
    shared["q_exercised_norm"] = q_exercised / contract.Q_max
    shared["q_remaining_norm"] = q_remaining / contract.Q_max
    shared["time_to_maturity_norm"] = ((contract.n_rights - steps) * contract.dt) / contract.maturity
    shared["normalized_time"] = steps / float(contract.n_rights)
    shared["spot"] = lsm_df["spot"]
    shared["X_t"] = x_t
    shared["Y_t"] = y_t
    shared["days_since_exercise_norm"] = prior_days / float(contract.n_rights)
    shared["days_since_exercise"] = prior_days
    shared["q_exercised_so_far"] = q_exercised
    shared["q_remaining"] = q_remaining
    shared["q_t"] = lsm_df["q_t"]
    shared["payoff_gross"] = lsm_df["payoff_gross"]
    shared["exercise_cost"] = lsm_df["exercise_cost"]
    shared["payoff_net"] = payoff_net
    shared["discount_factor_step"] = _discount_factor_series(contract, lsm_df["time_step"])
    shared["reward_discounted"] = shared["discount_factor_step"] * payoff_net.to_numpy(dtype=np.float64)
    shared["forward_contract_maturity"] = forward
    shared["strike"] = float(contract.strike)
    shared["c_cost"] = float(contract.c_cost)
    shared["gamma_cost"] = float(contract.gamma_cost)
    return ensure_trace_dtypes(shared[SHARED_TRACE_COLUMNS])


def write_trace_parquet(trace_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=22,
    )
