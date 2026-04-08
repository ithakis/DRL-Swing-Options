"""Run a representative full-state LSM degree sweep over a 6-config sample."""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos
from src.simulate_hhk_spot import no_seasonal_function, simulate_hhk_spot
from src.swing_contract import SwingContract


SELECTED_CONFIGS = [
    ("SwingOption_20_c0.04_gamma1", 11, 0.04, 1.0),
    ("SwingOption_20_c0.04_gamma1.5", 12, 0.04, 1.5),
    ("SwingOption_20_c0.04_gamma2", 13, 0.04, 2.0),
    ("SwingOption_20_c0.10_gamma1", 13, 0.10, 1.0),
    ("SwingOption_20_c0.10_gamma1.5", 13, 0.10, 1.5),
    ("SwingOption_20_c0.10_gamma2", 13, 0.10, 2.0),
]
DEGREES = [1, 2, 3, 4, 5, 6, 7]


def load_config(run_name: str) -> dict:
    with (REPO_ROOT / "runs" / f"{run_name}.json").open() as handle:
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
    train_ds = simulate_hhk_spot(
        **process_params,
        n_paths=config["n_paths"],
        seed=config["seed"],
        stratify=True,
        batch_size=config["batch_size"],
    )
    eval_ds = simulate_hhk_spot(
        **process_params,
        n_paths=config["n_paths_eval"],
        seed=config["seed"] + 1,
        stratify=True,
        batch_size=config["batch_size"],
    )
    return (
        tuple(np.asarray(arr, dtype=np.float64) for arr in train_ds),
        tuple(np.asarray(arr, dtype=np.float64) for arr in eval_ds),
    )


def main() -> None:
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    all_out = logs_dir / "lsm_estimator_sample_sweep.csv"
    summary_out = logs_dir / "lsm_estimator_sample_summary.csv"
    text_out = logs_dir / "lsm_estimator_sample_report.txt"

    all_rows = []
    report_lines = ["SELECTED CONFIGS"]
    for config_name, seed, c_val, gamma_val in SELECTED_CONFIGS:
        report_lines.append(f"{config_name}, seed={seed}, c={c_val}, gamma={gamma_val}")

        config = load_config(f"{config_name}_{seed}")
        contract = build_contract(config)
        train_ds, eval_ds = generate_datasets(config, contract)

        reduced_estimators = fit_lsm_estimators(
            contract=contract,
            dataset=train_ds,
            poly_degree=7,
            basis_type=config["lsm_basis"],
            state_mode="reduced",
            reg_type=config["lsm_reg"],
            reg_alpha=config["lsm_reg_alpha"],
        )
        reduced_price, _ = price_swing_option_lsm_oos(
            contract=contract,
            dataset=eval_ds,
            estimators=reduced_estimators,
            n_bootstrap=64,
            seed=config["seed"] + 1,
            csv_path=None,
        )

        for degree in DEGREES:
            full_estimators = fit_lsm_estimators(
                contract=contract,
                dataset=train_ds,
                poly_degree=degree,
                basis_type=config["lsm_basis"],
                state_mode="full",
                reg_type=config["lsm_reg"],
                reg_alpha=config["lsm_reg_alpha"],
            )
            full_price, _ = price_swing_option_lsm_oos(
                contract=contract,
                dataset=eval_ds,
                estimators=full_estimators,
                n_bootstrap=64,
                seed=config["seed"] + 1,
                csv_path=None,
            )
            all_rows.append(
                {
                    "Configuration": config_name,
                    "c": c_val,
                    "gamma": gamma_val,
                    "seed": seed,
                    "degree": degree,
                    "n_estimators": int(comb(3 + degree, degree)),
                    "reduced_price": reduced_price,
                    "full_price": full_price,
                    "pct_change_vs_reduced": 100.0 * (full_price - reduced_price) / reduced_price,
                }
            )

    all_df = pd.DataFrame(all_rows)
    summary_rows = []
    for degree in DEGREES:
        degree_df = all_df[all_df["degree"] == degree]
        summary_rows.append(
            {
                "degree": degree,
                "n_estimators": int(comb(3 + degree, degree)),
                "mean_pct_change": degree_df["pct_change_vs_reduced"].mean(),
                "median_pct_change": degree_df["pct_change_vs_reduced"].median(),
                "min_pct_change": degree_df["pct_change_vs_reduced"].min(),
                "max_pct_change": degree_df["pct_change_vs_reduced"].max(),
                "positive_count": int((degree_df["pct_change_vs_reduced"] > 0).sum()),
                "nonnegative_count": int((degree_df["pct_change_vs_reduced"] >= 0).sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_pct_change", ascending=False)

    all_df.to_csv(all_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    report_lines.append("")
    report_lines.append("AGGREGATE BY DEGREE")
    report_lines.extend(summary_df.to_string(index=False).splitlines())
    report_lines.append("")
    report_lines.append("PER CONFIG RESULTS")
    report_lines.extend(
        all_df[["Configuration", "degree", "n_estimators", "pct_change_vs_reduced"]].to_string(index=False).splitlines()
    )
    report_lines.append("")
    report_lines.append("BEST AGGREGATE")
    report_lines.append(str(summary_df.iloc[0].to_dict()))
    text_out.write_text("\n".join(report_lines))

    print(summary_df.to_string(index=False))
    print(f"Wrote {all_out}")
    print(f"Wrote {summary_out}")
    print(f"Wrote {text_out}")


if __name__ == "__main__":
    main()