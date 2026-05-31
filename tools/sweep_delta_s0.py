#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent_evaluation import _evaluate_swing_batch
from src.hedging_utils import hhk_forward_price
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos
from src.simulate_hhk_spot import simulate_hhk_spot
from tools.rebuild_results_v7 import build_agent, build_contract, build_hhk_params, discover_runs, dotdict

DEFAULT_CONFIGS = [
    "SwingOption_20_c0.00_gamma1",
    "SwingOption_20_c0.04_gamma1",
    "SwingOption_20_c0.04_gamma1.5",
    "SwingOption_20_c0.04_gamma2",
]
DEFAULT_PREFERRED_RL_SEEDS = (11, 12, 13)

CONFIG_LABELS = {
    "SwingOption_20_c0.00_gamma1": "Baseline (c=0)",
    "SwingOption_20_c0.04_gamma1": "c=0.04, gamma_c=1",
    "SwingOption_20_c0.04_gamma1.5": "c=0.04, gamma_c=1.5",
    "SwingOption_20_c0.04_gamma2": "c=0.04, gamma_c=2",
}

METHOD_STYLES = {"RL": "-", "LSM": "--"}
CONFIG_COLORS = {
    "SwingOption_20_c0.00_gamma1": "#4c78a8",
    "SwingOption_20_c0.04_gamma1": "#f58518",
    "SwingOption_20_c0.04_gamma1.5": "#54a24b",
    "SwingOption_20_c0.04_gamma2": "#e45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep initial spot values, evaluate the paper-style RL/LSM benchmarks, "
            "and fit GP-smoothed PV curves for TC premium, delta, and gamma."
        )
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated configuration base names.",
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        default="SwingOption_20_c0.00_gamma1",
        help="No-cost baseline config used for the TC premium.",
    )
    parser.add_argument(
        "--runs-dir", type=str, default="runs", help="Directory containing saved run json/pth files."
    )
    parser.add_argument(
        "--output-dir", type=str, default="logs/hedging_s0_sweep", help="Output directory for CSV and figures."
    )
    parser.add_argument("--s0-min", type=float, default=0.5, help="Lower bound of the observed S0 grid.")
    parser.add_argument("--s0-max", type=float, default=2.0, help="Upper bound of the observed S0 grid.")
    parser.add_argument("--s0-count", type=int, default=31, help="Number of observed S0 grid points.")
    parser.add_argument(
        "--s0-values", type=str, default="", help="Optional explicit comma-separated observed S0 values."
    )
    parser.add_argument(
        "--gp-grid-count", type=int, default=151, help="Number of S0 points in the GP-smoothed output grid."
    )
    parser.add_argument(
        "--gp-n-restarts", type=int, default=5, help="Number of optimizer restarts per GP kernel candidate."
    )
    parser.add_argument(
        "--n-paths-eval", type=int, default=32768, help="Number of common evaluation paths per S0."
    )
    parser.add_argument(
        "--lsm-train-paths", type=int, default=32768, help="Number of LSM training paths per S0."
    )
    parser.add_argument("--eval-seed", type=int, default=999, help="Random seed for evaluation paths.")
    parser.add_argument("--lsm-train-seed", type=int, default=998, help="Random seed for LSM training paths.")
    parser.add_argument("--eval-batch-size", type=int, default=4096, help="Batch size for RL evaluation.")
    parser.add_argument("--max-seeds", type=int, default=3, help="Maximum number of RL seeds per config.")
    parser.add_argument(
        "--preferred-rl-seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_PREFERRED_RL_SEEDS),
        help="Comma-separated preferred RL seeds, matching the paper methodology.",
    )
    parser.add_argument(
        "--require-exact-rl-seeds",
        action="store_true",
        help="Fail if a config does not have all preferred RL seeds available.",
    )
    parser.add_argument("--lsm-basis", type=str, default="chebyshev", help="LSM basis type.")
    parser.add_argument("--lsm-degree", type=int, default=2, help="LSM polynomial degree.")
    parser.add_argument("--lsm-reg", type=str, default="none", help="LSM regularization type.")
    parser.add_argument("--lsm-reg-alpha", type=float, default=1e-6, help="LSM regularization alpha.")
    parser.add_argument("--lsm-n-actions", type=int, default=5, help="Number of discretized actions for LSM.")
    return parser.parse_args()


def parse_s0_grid(args: argparse.Namespace) -> np.ndarray:
    if args.s0_values.strip():
        values = [float(part) for part in args.s0_values.split(",") if part.strip()]
        grid = np.asarray(values, dtype=np.float64)
    else:
        grid = np.linspace(args.s0_min, args.s0_max, args.s0_count, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("Need at least three observed S0 points to estimate finite-difference Greeks.")
    return np.unique(np.round(grid, 8))


def parse_gp_grid(args: argparse.Namespace, observed_s0_grid: np.ndarray) -> np.ndarray:
    grid = np.linspace(
        float(observed_s0_grid.min()), float(observed_s0_grid.max()), args.gp_grid_count, dtype=np.float64
    )
    return np.unique(np.round(grid, 8))


def parse_seed_list(seed_text: str) -> Tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in seed_text.split(",") if part.strip())
    if not seeds:
        raise ValueError("preferred RL seeds must contain at least one integer seed")
    return seeds


def load_run_params(json_path: str) -> dotdict:
    with open(json_path, "r") as handle:
        return dotdict(json.load(handle))


def select_runs(
    all_runs: Dict[str, List[Tuple[int, str, str]]],
    config: str,
    *,
    max_seeds: int,
    preferred_seeds: Sequence[int],
    require_exact: bool,
) -> List[Tuple[int, str, str]]:
    if config not in all_runs:
        raise KeyError(f"Configuration not found in runs/: {config}")

    available_runs = sorted(all_runs[config], key=lambda item: item[0])
    by_seed = {seed: item for seed, *rest in available_runs for item in [(seed, rest[0], rest[1])]}  # type: ignore[misc]
    selected = [by_seed[seed] for seed in preferred_seeds if seed in by_seed]
    missing = [seed for seed in preferred_seeds if seed not in by_seed]

    if max_seeds > 0:
        selected = selected[:max_seeds]

    if missing:
        message = f"Missing preferred RL seeds for {config}: {missing}. Available seeds: {[seed for seed, _, _ in available_runs]}"
        if require_exact:
            raise KeyError(message)
        print(f"  WARNING: {message}")

    if selected:
        return selected
    return available_runs if max_seeds <= 0 else available_runs[:max_seeds]


def evaluate_rl_price(agent, contract, dataset, eval_batch_size: int) -> tuple[float, float]:
    n_paths = dataset[1].shape[0]
    all_returns: List[float] = []
    for start in range(0, n_paths, eval_batch_size):
        end = min(start + eval_batch_size, n_paths)
        batch_indices = list(range(start, end))
        returns, _, _ = _evaluate_swing_batch(
            agent=agent,
            contract=contract,
            dataset=dataset,
            batch_indices=batch_indices,
            collect_path_data=False,
        )
        all_returns.extend(returns)
    returns_arr = np.asarray(all_returns, dtype=np.float64)
    return float(returns_arr.mean()), float(1.96 * returns_arr.std(ddof=1) / np.sqrt(len(returns_arr)))


def evaluate_lsm_price(
    contract,
    hhk_params: dict,
    *,
    test_dataset,
    lsm_train_paths: int,
    lsm_train_seed: int,
    lsm_basis: str,
    lsm_degree: int,
    lsm_reg: str,
    lsm_reg_alpha: float,
    lsm_n_actions: int,
) -> tuple[float, float]:
    train_dataset = simulate_hhk_spot(
        **hhk_params,
        n_paths=lsm_train_paths,
        seed=lsm_train_seed,
        stratify=True,
        batch_size=128,
    )
    train_dataset_f64 = tuple(np.asarray(arr, dtype=np.float64) for arr in train_dataset)
    estimators = fit_lsm_estimators(
        contract=contract,
        dataset=train_dataset_f64,
        poly_degree=lsm_degree,
        basis_type=lsm_basis,
        state_mode="full",
        reg_type=lsm_reg,
        reg_alpha=lsm_reg_alpha,
        n_actions=lsm_n_actions,
    )
    test_dataset_f64 = tuple(np.asarray(arr, dtype=np.float64) for arr in test_dataset)
    price, (ci_low, ci_high) = price_swing_option_lsm_oos(
        contract=contract,
        dataset=test_dataset_f64,
        estimators=estimators,
        seed=lsm_train_seed + 1,
        csv_path=None,
        _print_results=False,
    )
    return float(price), float((ci_high - ci_low) / 2.0)


def config_label(config: str) -> str:
    return CONFIG_LABELS.get(config, config)


def initial_forward_price_grid(s0: np.ndarray, contract, hhk_params: dict) -> np.ndarray:
    seasonal_fn = hhk_params.get("f")
    x0 = np.log(s0) - float(seasonal_fn(0.0))
    y0 = np.zeros_like(s0, dtype=np.float64)
    current_time = np.zeros_like(s0, dtype=np.float64)
    delivery_time = np.full_like(s0, contract.maturity, dtype=np.float64)
    return hhk_forward_price(
        current_time=current_time,
        delivery_time=delivery_time,
        X_t=x0,
        Y_t=y0,
        alpha=hhk_params["alpha"],
        sigma=hhk_params["sigma"],
        beta=hhk_params["beta"],
        lam=hhk_params["lam"],
        mu_J=hhk_params["mu_J"],
        seasonal_fn=seasonal_fn,
    )


def annotate_tc_premium(
    results_df: pd.DataFrame, baseline_config: str, premium_column: str = "premium"
) -> pd.DataFrame:
    baseline = results_df[results_df["config"] == baseline_config][["method", "s0", premium_column]].rename(
        columns={premium_column: "baseline_premium"}
    )
    enriched = results_df.merge(baseline, on=["method", "s0"], how="left")
    enriched["tc_premium"] = enriched["baseline_premium"] - enriched[premium_column]
    # Backward-compatible alias for consumers that still look for the old name.
    enriched["value_adjustment"] = enriched["tc_premium"]
    return enriched


def add_raw_finite_differences(results_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (_, _), group in results_df.groupby(["method", "config"], sort=False):
        ordered = group.sort_values("s0").copy()
        s0 = ordered["s0"].to_numpy(dtype=np.float64)
        premium = ordered["premium"].to_numpy(dtype=np.float64)
        ordered["delta_t0_raw"] = np.gradient(premium, s0)
        ordered["gamma_t0_raw"] = np.gradient(ordered["delta_t0_raw"].to_numpy(dtype=np.float64), s0)
        frames.append(ordered)
    return pd.concat(frames, ignore_index=True)


def build_gp_kernel_candidates() -> List[Tuple[str, object]]:
    signal = ConstantKernel(1.0, (1e-3, 1e3))
    noise = WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    return [
        ("rbf", signal * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + noise),
        ("matern32", signal * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + noise),
        ("matern52", signal * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + noise),
        (
            "rational_quadratic",
            signal
            * RationalQuadratic(
                length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-2, 1e2), alpha_bounds=(1e-2, 1e2)
            )
            + noise,
        ),
    ]


def observation_noise_sigma(group: pd.DataFrame) -> np.ndarray:
    eval_sigma = np.maximum(group["premium_ci95"].to_numpy(dtype=np.float64) / 1.96, 1e-6)
    seed_sigma = np.maximum(group["premium_seed_std"].to_numpy(dtype=np.float64), 0.0)
    return np.hypot(eval_sigma, seed_sigma)


def fit_best_gp(
    s0_obs: np.ndarray,
    premium_obs: np.ndarray,
    noise_sigma: np.ndarray,
    *,
    n_restarts: int,
) -> Tuple[GaussianProcessRegressor, float, float, str, float]:
    x_mean = float(np.mean(s0_obs))
    x_scale = float(np.std(s0_obs))
    if x_scale <= 1e-10:
        x_scale = 1.0

    x_train = ((s0_obs - x_mean) / x_scale).reshape(-1, 1)
    alpha = np.square(np.clip(noise_sigma, 1e-6, None))

    best_name = ""
    best_gp: GaussianProcessRegressor | None = None
    best_score = -np.inf

    for kernel_name, kernel in build_gp_kernel_candidates():
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            random_state=0,
            n_restarts_optimizer=n_restarts,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(x_train, premium_obs)
        score = float(gp.log_marginal_likelihood_value_)
        if score > best_score:
            best_score = score
            best_name = kernel_name
            best_gp = gp

    if best_gp is None:
        raise RuntimeError("Failed to fit any Gaussian-process candidate")

    return best_gp, x_mean, x_scale, best_name, best_score


def build_smoothed_curves(
    raw_df: pd.DataFrame,
    config_meta: Dict[str, Tuple[object, dict]],
    *,
    gp_eval_s0: np.ndarray,
    baseline_config: str,
    gp_n_restarts: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    smooth_records: List[dict] = []
    gp_records: List[dict] = []

    for (method, config), group in raw_df.groupby(["method", "config"], sort=False):
        ordered = group.sort_values("s0").reset_index(drop=True)
        s0_obs = ordered["s0"].to_numpy(dtype=np.float64)
        premium_obs = ordered["premium"].to_numpy(dtype=np.float64)
        noise_sigma = observation_noise_sigma(ordered)
        gp, x_mean, x_scale, kernel_name, gp_score = fit_best_gp(
            s0_obs,
            premium_obs,
            noise_sigma,
            n_restarts=gp_n_restarts,
        )

        x_eval = ((gp_eval_s0 - x_mean) / x_scale).reshape(-1, 1)
        premium_smooth, premium_gp_std = gp.predict(x_eval, return_std=True)
        delta_t0 = np.gradient(premium_smooth, gp_eval_s0)
        gamma_t0 = np.gradient(delta_t0, gp_eval_s0)

        contract, base_hhk_params = config_meta[config]
        forward_price_t0 = initial_forward_price_grid(gp_eval_s0, contract, base_hhk_params)
        premium_ci95_interp = np.interp(gp_eval_s0, s0_obs, ordered["premium_ci95"].to_numpy(dtype=np.float64))
        premium_seed_std_interp = np.interp(
            gp_eval_s0, s0_obs, ordered["premium_seed_std"].to_numpy(dtype=np.float64)
        )
        is_observed = np.isclose(gp_eval_s0[:, None], s0_obs[None, :], atol=1e-10).any(axis=1)

        selected_rl_seeds = ordered["selected_rl_seeds"].iloc[0]
        n_rl_seeds = int(ordered["n_rl_seeds"].iloc[0])
        for idx, s0 in enumerate(gp_eval_s0):
            smooth_records.append(
                {
                    "config": config,
                    "config_label": config_label(config),
                    "method": method,
                    "s0": float(s0),
                    "forward_price_t0": float(forward_price_t0[idx]),
                    "premium": float(premium_smooth[idx]),
                    "premium_gp_std": float(premium_gp_std[idx]),
                    "premium_ci95": float(premium_ci95_interp[idx]),
                    "premium_seed_std": float(premium_seed_std_interp[idx]),
                    "n_rl_seeds": n_rl_seeds,
                    "selected_rl_seeds": selected_rl_seeds,
                    "is_observed_s0": bool(is_observed[idx]),
                    "delta_t0": float(delta_t0[idx]),
                    "gamma_t0": float(gamma_t0[idx]),
                    "gp_kernel": kernel_name,
                    "gp_kernel_fitted": str(gp.kernel_),
                    "gp_log_marginal_likelihood": gp_score,
                }
            )

        gp_records.append(
            {
                "config": config,
                "method": method,
                "gp_kernel": kernel_name,
                "gp_kernel_fitted": str(gp.kernel_),
                "gp_log_marginal_likelihood": gp_score,
                "observed_s0_min": float(s0_obs.min()),
                "observed_s0_max": float(s0_obs.max()),
                "observed_s0_count": int(s0_obs.size),
                "n_rl_seeds": n_rl_seeds,
                "selected_rl_seeds": selected_rl_seeds,
            }
        )

    smooth_df = pd.DataFrame(smooth_records)
    smooth_df = annotate_tc_premium(smooth_df, baseline_config=baseline_config, premium_column="premium")
    gp_df = pd.DataFrame(gp_records).sort_values(["method", "config"]).reset_index(drop=True)
    return smooth_df.sort_values(["method", "config", "s0"]).reset_index(drop=True), gp_df


def plot_results(raw_df: pd.DataFrame, smooth_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), sharex=True)
    panel_specs = [
        ("premium", "Premium vs S0", "option premium"),
        ("tc_premium", "TC premium vs no-cost baseline", "baseline premium - premium"),
        ("delta_t0", "t=0 delta vs S0", "dPV / dS0"),
        ("gamma_t0", "t=0 gamma vs S0", "d^2 PV / dS0^2"),
    ]

    for ax, (column, title, ylabel) in zip(axes.ravel(), panel_specs):
        for method in ["RL", "LSM"]:
            for config in smooth_df["config"].drop_duplicates().tolist():
                subset = smooth_df[
                    (smooth_df["method"] == method) & (smooth_df["config"] == config)
                ].sort_values("s0")
                if subset.empty:
                    continue
                ax.plot(
                    subset["s0"],
                    subset[column],
                    color=CONFIG_COLORS.get(config, "#444444"),
                    linestyle=METHOD_STYLES[method],
                    linewidth=2.2,
                    alpha=0.95,
                )
                if column in {"premium", "tc_premium"}:
                    observed = raw_df[(raw_df["method"] == method) & (raw_df["config"] == config)].sort_values(
                        "s0"
                    )
                    raw_column = "premium" if column == "premium" else "tc_premium"
                    ax.scatter(
                        observed["s0"],
                        observed[raw_column],
                        color=CONFIG_COLORS.get(config, "#444444"),
                        edgecolor="white",
                        linewidth=0.5,
                        s=20,
                        alpha=0.7,
                    )
        ax.set_title(title)
        ax.set_xlabel("S0")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)

    axes[1, 1].text(
        0.02,
        0.96,
        "Delta and gamma are finite differences on the GP-smoothed PV curve.",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    config_handles = [
        plt.Line2D(
            [0], [0], color=CONFIG_COLORS.get(config, "#444444"), linewidth=3, label=config_label(config)
        )
        for config in smooth_df["config"].drop_duplicates().tolist()
    ]
    method_handles = [
        plt.Line2D([0], [0], color="#333333", linestyle=METHOD_STYLES[method], linewidth=2.2, label=method)
        for method in ["RL", "LSM"]
    ]
    fig.legend(
        handles=config_handles,
        title="Configuration",
        loc="lower center",
        bbox_to_anchor=(0.42, 0.01),
        ncol=2,
        frameon=True,
    )
    fig.legend(
        handles=method_handles,
        title="Method",
        loc="lower center",
        bbox_to_anchor=(0.86, 0.01),
        ncol=2,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(output_dir / "s0_greeks_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configs = [part.strip() for part in args.configs.split(",") if part.strip()]
    preferred_rl_seeds = parse_seed_list(args.preferred_rl_seeds)
    observed_s0_grid = parse_s0_grid(args)
    gp_eval_s0 = parse_gp_grid(args, observed_s0_grid)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = discover_runs(args.runs_dir)
    raw_records: List[dict] = []
    config_meta: Dict[str, Tuple[object, dict]] = {}

    for config in configs:
        selected_runs = select_runs(
            all_runs,
            config,
            max_seeds=args.max_seeds,
            preferred_seeds=preferred_rl_seeds,
            require_exact=args.require_exact_rl_seeds,
        )
        params = load_run_params(selected_runs[0][1])
        contract = build_contract(params)
        base_hhk_params = build_hhk_params(params)
        config_meta[config] = (contract, base_hhk_params)
        forward_price_grid = initial_forward_price_grid(observed_s0_grid, contract, base_hhk_params)
        selected_seed_text = ",".join(str(seed) for seed, _, _ in selected_runs)
        print(f"Processing {config} with RL seeds [{selected_seed_text}]")

        for s0, forward_price_t0 in zip(observed_s0_grid, forward_price_grid):
            hhk_params = dict(base_hhk_params)
            hhk_params["S0"] = float(s0)
            test_dataset = simulate_hhk_spot(
                **hhk_params,
                n_paths=args.n_paths_eval,
                seed=args.eval_seed,
                stratify=True,
                batch_size=128,
            )

            rl_prices = []
            rl_cis = []
            for seed, json_path, pth_path in selected_runs:
                run_params = load_run_params(json_path)
                agent = build_agent(run_params)
                agent.actor_local.load_state_dict(torch.load(pth_path, map_location="cpu"))
                agent.actor_local.eval()
                price, ci95 = evaluate_rl_price(agent, contract, test_dataset, args.eval_batch_size)
                rl_prices.append(price)
                rl_cis.append(ci95)
                print(f"  RL seed={seed:>2} | S0={s0:.4f} | premium={price:.6f} | CI95={ci95:.6f}")

            rl_price_mean = float(np.mean(rl_prices))
            rl_price_seed_std = float(np.std(rl_prices, ddof=1)) if len(rl_prices) > 1 else 0.0
            raw_records.append(
                {
                    "config": config,
                    "config_label": config_label(config),
                    "method": "RL",
                    "s0": float(s0),
                    "forward_price_t0": float(forward_price_t0),
                    "premium": rl_price_mean,
                    "premium_ci95": float(np.mean(rl_cis)),
                    "premium_seed_std": rl_price_seed_std,
                    "n_rl_seeds": len(rl_prices),
                    "selected_rl_seeds": selected_seed_text,
                }
            )

            lsm_price, lsm_ci95 = evaluate_lsm_price(
                contract,
                hhk_params,
                test_dataset=test_dataset,
                lsm_train_paths=args.lsm_train_paths,
                lsm_train_seed=args.lsm_train_seed,
                lsm_basis=args.lsm_basis,
                lsm_degree=args.lsm_degree,
                lsm_reg=args.lsm_reg,
                lsm_reg_alpha=args.lsm_reg_alpha,
                lsm_n_actions=args.lsm_n_actions,
            )
            print(f"  LSM         | S0={s0:.4f} | premium={lsm_price:.6f} | CI95={lsm_ci95:.6f}")
            raw_records.append(
                {
                    "config": config,
                    "config_label": config_label(config),
                    "method": "LSM",
                    "s0": float(s0),
                    "forward_price_t0": float(forward_price_t0),
                    "premium": lsm_price,
                    "premium_ci95": lsm_ci95,
                    "premium_seed_std": 0.0,
                    "n_rl_seeds": 0,
                    "selected_rl_seeds": "",
                }
            )

    raw_df = pd.DataFrame(raw_records)
    raw_df = annotate_tc_premium(raw_df, baseline_config=args.baseline_config, premium_column="premium")
    raw_df = add_raw_finite_differences(raw_df)
    raw_df = raw_df.sort_values(["method", "config", "s0"]).reset_index(drop=True)

    smooth_df, gp_df = build_smoothed_curves(
        raw_df,
        config_meta,
        gp_eval_s0=gp_eval_s0,
        baseline_config=args.baseline_config,
        gp_n_restarts=args.gp_n_restarts,
    )

    raw_csv_path = output_dir / "s0_greeks_raw.csv"
    summary_csv_path = output_dir / "s0_greeks_summary.csv"
    gp_csv_path = output_dir / "s0_greeks_gp_models.csv"

    raw_df.to_csv(raw_csv_path, index=False)
    smooth_df.to_csv(summary_csv_path, index=False)
    gp_df.to_csv(gp_csv_path, index=False)
    plot_results(raw_df, smooth_df, output_dir)

    print("\nSaved outputs:")
    print(f"  Raw CSV     : {raw_csv_path}")
    print(f"  Summary CSV : {summary_csv_path}")
    print(f"  GP CSV      : {gp_csv_path}")
    print(f"  PNG         : {output_dir / 's0_greeks_summary.png'}")


if __name__ == "__main__":
    main()
