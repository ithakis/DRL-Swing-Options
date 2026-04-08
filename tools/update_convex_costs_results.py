"""Rewrite the main convex-cost results CSV to expose minimal and full LSM benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = REPO_ROOT / "Jupyter Notebooks" / "Convex Costs Results 6.csv"
DEFAULT_COMPARISON_CSV = REPO_ROOT / "logs" / "lsm_state_mode_comparison.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--comparison_csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    return parser.parse_args()


def rounded(series: pd.Series, digits: int = 4) -> pd.Series:
    return series.astype(float).round(digits)


def rounded_pct(series: pd.Series, digits: int = 2) -> pd.Series:
    return series.astype(float).round(digits)


def main() -> None:
    args = parse_args()
    results_df = pd.read_csv(args.results_csv)
    comparison_df = pd.read_csv(args.comparison_csv)

    merged = results_df.merge(
        comparison_df,
        on=["Configuration", "Best Seed", "Best Episode", "c", "gamma", "RL Price"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_cmp"),
    )

    if merged["LSM_full"].isna().any():
        missing = merged.loc[merged["LSM_full"].isna(), "Configuration"].tolist()
        raise ValueError(f"Missing comparison rows for configurations: {missing}")

    updated = merged.copy()

    updated["LSM_minimal"] = rounded(updated["LSM_minimal_csv"])
    updated["LSM_full"] = rounded(updated["LSM_full"])
    updated["PctDiff_minimal"] = rounded_pct(updated["PctDiff_minimal"])
    updated["MeanDiff_minimal"] = rounded(updated["RL_minus_LSM_minimal"])
    updated["PctDiff_full"] = rounded_pct(updated["PctDiff_full"])
    updated["CI_95_full"] = rounded(updated["CI_95_full"])
    updated["MeanDiff_full"] = rounded(updated["MeanDiff_full"])
    updated["CI_Lower_full"] = rounded(updated["CI_Lower_full"])
    updated["CI_Upper_full"] = rounded(updated["CI_Upper_full"])
    updated["RL_95CI_full"] = rounded(updated["RL_95CI_rerun"])
    updated["LSM_full_95CI"] = rounded(updated["LSM_full_95CI"])
    updated["PctDiff_95CI_full"] = rounded_pct(updated["PctDiff_95CI_full"])

    preferred_columns = [
        "Configuration",
        "c",
        "gamma",
        "Best Seed",
        "Best Episode",
        "LSM_minimal",
        "LSM_full",
        "RL Price",
        "PctDiff_minimal",
        "MeanDiff_minimal",
        "PctDiff_full",
        "CI_95_full",
        "MeanDiff_full",
        "CI_Lower_full",
        "CI_Upper_full",
        "RL_95CI_full",
        "LSM_full_95CI",
        "PctDiff_95CI_full",
    ]

    optional_columns = [
        "RL_BangBangness",
        "LSM_BangBangness",
        "LSM_full_minus_minimal_csv",
        "LSM_full_pct_change_vs_minimal_csv",
        "Basis",
        "Legacy Degree",
        "Tuned Degree",
        "Reg",
    ]
    final_columns = preferred_columns + [column for column in optional_columns if column in updated.columns]

    updated = updated[final_columns].sort_values(["c", "gamma"], kind="stable")
    updated.to_csv(args.results_csv, index=False)

    summary = updated[[
        "Configuration",
        "LSM_minimal",
        "LSM_full",
        "RL Price",
        "PctDiff_minimal",
        "PctDiff_full",
    ]]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()