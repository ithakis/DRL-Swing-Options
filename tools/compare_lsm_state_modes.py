"""Compare legacy minimal and tuned full-state LSM prices across saved convex-cost runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos
from src.simulate_hhk_spot import no_seasonal_function, simulate_hhk_spot
from src.swing_contract import SwingContract

DEFAULT_INPUT_CSV = REPO_ROOT / "Jupyter Notebooks" / "Convex Costs Results 6.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "logs" / "lsm_state_mode_comparison.csv"
DEFAULT_FULL_PARQUET_DIR = REPO_ROOT / "logs" / "lsm_full_state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=256,
        help="Bootstrap samples for each OOS price estimate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of rows to run from the input CSV. 0 means all rows.",
    )
    parser.add_argument(
        "--minimal_state_mode",
        type=str,
        default="reduced",
        help="State mode used for the legacy minimal benchmark validation rerun.",
    )
    parser.add_argument(
        "--full_state_mode",
        type=str,
        default="full",
        help="State mode used for the tuned benchmark rerun.",
    )
    parser.add_argument(
        "--minimal_degree",
        type=int,
        default=7,
        help="Polynomial degree used for the legacy minimal benchmark validation rerun.",
    )
    parser.add_argument(
        "--full_degree",
        type=int,
        default=2,
        help="Polynomial degree used for the tuned full-state rerun.",
    )
    parser.add_argument(
        "--full_parquet_dir",
        type=Path,
        default=DEFAULT_FULL_PARQUET_DIR,
        help="Directory where tuned full-state per-path evaluation parquets are written.",
    )
    return parser.parse_args()


def load_run_config(run_name: str) -> dict:
    json_path = REPO_ROOT / "runs" / f"{run_name}.json"
    with json_path.open() as handle:
        return json.load(handle)


def build_contract(config: dict) -> SwingContract:
    return SwingContract(
        q_min=config["q_min"],
        q_max=config["q_max"],
        Q_min=config["Q_min"],
        Q_max=config["Q_max"],
        strike=config["strike"],
        maturity=config["maturity"],
        n_rights=config["n_rights"],
        r=config["risk_free_rate"],
        min_refraction_periods=config["min_refraction_periods"],
        c_cost=config["c_cost"],
        gamma_cost=config["gamma_cost"],
    )


def generate_datasets(config: dict, contract: SwingContract) -> tuple:
    np_dtype = np.float32 if config.get("fp32", 1) else np.float64
    process_params = {
        "S0": config["S0"],
        "T": contract.maturity,
        "n_steps": contract.n_rights - 1,
        "alpha": config["alpha"],
        "sigma": config["sigma"],
        "beta": config["beta"],
        "lam": config["lam"],
        "mu_J": config["mu_J"],
        "f": no_seasonal_function,
        "dtype": np_dtype,
    }
    batch_size = config["batch_size"]
    seed = config["seed"]
    train_ds = simulate_hhk_spot(
        **process_params,
        n_paths=config["n_paths"],
        seed=seed,
        stratify=True,
        batch_size=batch_size,
    )
    eval_ds = simulate_hhk_spot(
        **process_params,
        n_paths=config["n_paths_eval"],
        seed=seed + 1,
        stratify=True,
        batch_size=batch_size,
    )
    train_ds = tuple(np.asarray(arr, dtype=np.float64) for arr in train_ds)
    eval_ds = tuple(np.asarray(arr, dtype=np.float64) for arr in eval_ds)
    return train_ds, eval_ds


def price_mode(
    contract: SwingContract,
    train_ds: tuple,
    eval_ds: tuple,
    config: dict,
    state_mode: str,
    degree: int,
    bootstrap: int,
    parquet_path: Path | None = None,
) -> tuple[float, tuple[float, float], float]:
    start = time.perf_counter()
    estimators = fit_lsm_estimators(
        contract=contract,
        dataset=train_ds,
        poly_degree=degree,
        basis_type=config["lsm_basis"],
        state_mode=state_mode,
        reg_type=config["lsm_reg"],
        reg_alpha=config["lsm_reg_alpha"],
    )
    price, ci = price_swing_option_lsm_oos(
        contract=contract,
        dataset=eval_ds,
        estimators=estimators,
        n_bootstrap=bootstrap,
        seed=config["seed"] + 1,
        csv_path=str(parquet_path) if parquet_path is not None else None,
    )
    elapsed = time.perf_counter() - start
    return price, ci, elapsed


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_obs = len(values)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_idx in range(n_bootstrap):
        sample_idx = rng.integers(0, n_obs, n_obs)
        boot_means[bootstrap_idx] = values[sample_idx].mean()
    return tuple(float(x) for x in np.percentile(boot_means, [2.5, 97.5]))


def _half_width(ci: tuple[float, float]) -> float:
    return 0.5 * (ci[1] - ci[0])


def _load_rl_path_values(rl_path: Path) -> np.ndarray:
    rl_df = pd.read_parquet(rl_path, columns=["path", "reward"])
    return rl_df.groupby("path", sort=True)["reward"].sum().to_numpy(dtype=np.float64)


def _load_lsm_path_values(lsm_path: Path, contract: SwingContract) -> np.ndarray:
    lsm_df = pd.read_parquet(lsm_path, columns=["path", "time_step", "payoff"])
    lsm_df = lsm_df.copy()
    lsm_df["discounted_payoff"] = lsm_df["payoff"] * (contract.discount_factor ** lsm_df["time_step"])
    return lsm_df.groupby("path", sort=True)["discounted_payoff"].sum().to_numpy(dtype=np.float64)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    results = []
    records = df.to_dict("records")
    total_rows = len(records)
    args.full_parquet_dir.mkdir(parents=True, exist_ok=True)
    for row_idx, row in enumerate(records, start=1):
        best_seed = int(row["Best Seed"])
        run_name = f"{row['Configuration']}_{best_seed}"
        print(f"[{row_idx}/{total_rows}] Running {run_name}")
        config = load_run_config(run_name)
        contract = build_contract(config)
        train_ds, eval_ds = generate_datasets(config, contract)
        rl_parquet_path = REPO_ROOT / "logs" / run_name / "evaluations" / f"rl_episode_{int(row['Best Episode'])}.parquet"
        full_parquet_path = args.full_parquet_dir / f"{run_name}_full_d{args.full_degree}.parquet"

        minimal_csv_price = float(row.get("LSM_minimal", row.get("LSM Price")))
        reduced_price, reduced_ci, reduced_time = price_mode(
            contract=contract,
            train_ds=train_ds,
            eval_ds=eval_ds,
            config=config,
            state_mode=args.minimal_state_mode,
            degree=args.minimal_degree,
            bootstrap=args.bootstrap,
        )
        full_price, full_ci, full_time = price_mode(
            contract=contract,
            train_ds=train_ds,
            eval_ds=eval_ds,
            config=config,
            state_mode=args.full_state_mode,
            degree=args.full_degree,
            bootstrap=args.bootstrap,
            parquet_path=full_parquet_path,
        )

        rl_price = float(row["RL Price"])
        pctdiff_minimal = 100.0 * (rl_price - minimal_csv_price) / minimal_csv_price
        pctdiff_full = 100.0 * (rl_price - full_price) / full_price
        rl_path_values = _load_rl_path_values(rl_parquet_path)
        lsm_full_path_values = _load_lsm_path_values(full_parquet_path, contract)
        diff_path_values = rl_path_values - lsm_full_path_values

        rl_ci = _bootstrap_ci(rl_path_values, args.bootstrap, seed=config["seed"] + 101)
        full_lsm_ci = _bootstrap_ci(lsm_full_path_values, args.bootstrap, seed=config["seed"] + 202)
        diff_ci = _bootstrap_ci(diff_path_values, args.bootstrap, seed=config["seed"] + 303)
        pctdiff_ci = (
            100.0 * diff_ci[0] / full_price,
            100.0 * diff_ci[1] / full_price,
        )
        results.append(
            {
                "Configuration": row["Configuration"],
                "Best Seed": best_seed,
                "Best Episode": int(row["Best Episode"]),
                "c": float(row["c"]),
                "gamma": float(row["gamma"]),
                "Basis": config["lsm_basis"],
                "Legacy Degree": args.minimal_degree,
                "Tuned Degree": args.full_degree,
                "Reg": config["lsm_reg"],
                "LSM_minimal_csv": minimal_csv_price,
                "LSM_minimal_rerun": reduced_price,
                "LSM_minimal_rerun_CI_low": reduced_ci[0],
                "LSM_minimal_rerun_CI_high": reduced_ci[1],
                "LSM_full": full_price,
                "LSM_full_CI_low": full_ci[0],
                "LSM_full_CI_high": full_ci[1],
                "MeanDiff_full": float(diff_path_values.mean()),
                "CI_Lower_full": diff_ci[0],
                "CI_Upper_full": diff_ci[1],
                "CI_95_full": _half_width(diff_ci),
                "RL_95CI_rerun": _half_width(rl_ci),
                "LSM_full_95CI": _half_width(full_lsm_ci),
                "PctDiff_95CI_full": _half_width(pctdiff_ci),
                "LSM_full_minus_minimal_csv": full_price - minimal_csv_price,
                "LSM_full_pct_change_vs_minimal_csv": 100.0 * (full_price - minimal_csv_price) / minimal_csv_price,
                "LSM_minimal_rerun_minus_csv": reduced_price - minimal_csv_price,
                "RL Price": rl_price,
                "RL_minus_LSM_minimal": rl_price - minimal_csv_price,
                "RL_minus_LSM_full": rl_price - full_price,
                "PctDiff_minimal": pctdiff_minimal,
                "PctDiff_full": pctdiff_full,
                "Minimal Runtime Sec": reduced_time,
                "Full Runtime Sec": full_time,
            }
        )

    results_df = pd.DataFrame(results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)

    summary = {
        "rows": len(results_df),
        "mean_full_minus_minimal_csv": float(results_df["LSM_full_minus_minimal_csv"].mean()),
        "median_full_minus_minimal_csv": float(results_df["LSM_full_minus_minimal_csv"].median()),
        "num_full_gt_minimal_csv": int((results_df["LSM_full_minus_minimal_csv"] > 0).sum()),
        "num_full_lt_minimal_csv": int((results_df["LSM_full_minus_minimal_csv"] < 0).sum()),
        "mean_abs_minimal_rerun_minus_csv": float(results_df["LSM_minimal_rerun_minus_csv"].abs().mean()),
        "output_csv": str(args.output_csv),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
