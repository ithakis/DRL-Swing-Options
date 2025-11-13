#!/usr/bin/env python3
"""
TensorBoard run analyzer for swing option pricing experiments.

This utility loads one or more TensorBoard event files, extracts the core
evaluation and training metrics, and produces aggregate statistics that help
diagnose why Pricing/Delta_Percent may deteriorate across runs.

Example usage
-------------
python analysis/tensorboard_run_analysis.py \\
    runs/SwingOption_20_RegimeSwitching_wRegLab_32_11 \\
    runs/SwingOption_20_RegimeSwitching_wRegLab_32_12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


EVAL_TAG_PREFIX = "Pricing/"
TRAINING_TAGS = [
    "Episode_Return",
    "Average100",
    "Critic_loss",
    "Actor_loss",
    "Policy/Action_variance_mean",
    "Policy/Actions_at_lower_pct",
    "Policy/Actions_at_upper_pct",
    "TD_Error/p90",
    "TD_Error/p99",
    "PER/priority_entropy",
]


def resolve_event_files(paths: Iterable[str]) -> List[Path]:
    """Expand directories/globs into explicit event file paths."""
    event_paths: List[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {raw}")
        if path.is_dir():
            event_paths.extend(sorted(path.rglob("events.out.tfevents.*")))
        elif path.is_file():
            event_paths.append(path)
        else:
            raise ValueError(f"Unsupported path type: {raw}")

    deduped = []
    seen = set()
    for p in event_paths:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    if not deduped:
        raise FileNotFoundError("No TensorBoard event files found.")
    return deduped


def load_scalars(event_file: Path, tags: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load scalars from a single TensorBoard event file into a DataFrame."""
    ea = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    ea.Reload()
    available_tags = ea.Tags().get("scalars", [])
    selected_tags = list(tags) if tags else available_tags

    rows = []
    for tag in selected_tags:
        if tag not in available_tags:
            continue
        for event in ea.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "value": event.value,
                }
            )
    df = pd.DataFrame(rows)
    df["event_file"] = str(event_file)
    return df


def pivot_eval_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Convert evaluation metrics into a wide table indexed by step."""
    eval_df = df[df["tag"].str.startswith(EVAL_TAG_PREFIX)]
    if eval_df.empty:
        return pd.DataFrame()

    pivot = (
        eval_df.pivot_table(index="step", columns="tag", values="value", aggfunc="last")
        .sort_index()
        .reset_index()
    )
    pivot.columns.name = None
    rename_map = {
        "Pricing/RL_Price": "rl_price",
        "Pricing/LSM_Price": "lsm_price",
        "Pricing/Delta_Price": "delta_price",
        "Pricing/Delta_Percent": "delta_percent",
    }
    pivot = pivot.rename(columns=rename_map)
    return pivot


def compute_trend(x: pd.Series, y: pd.Series) -> float:
    """Return slope from a simple linear regression (value per path)."""
    clean = x.notna() & y.notna()
    if clean.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(x[clean], y[clean], 1)
    return float(slope)


def summarize_run(run_name: str, eval_df: pd.DataFrame, train_df: pd.DataFrame) -> Dict:
    """Generate summary statistics for a single run."""
    summary: Dict[str, float] = {
        "run": run_name,
        "eval_points": int(len(eval_df)),
    }

    if not eval_df.empty and "delta_percent" in eval_df:

        delta = eval_df["delta_percent"]
        steps = eval_df["step"]
        summary.update(
            {
                "delta_final": float(delta.iloc[-1]),
                "delta_best": float(delta.max()),
                "delta_worst": float(delta.min()),
                "delta_positive_frac": float((delta > 0).mean()),
                "delta_std": float(delta.std(ddof=0) if len(delta) else np.nan),
                "delta_slope_per_1k_paths": float(compute_trend(steps, delta) * 1000.0),
            }
        )

        best_idx = delta.idxmax()
        worst_idx = delta.idxmin()
        summary["delta_best_step"] = int(eval_df.loc[best_idx, "step"])
        summary["delta_worst_step"] = int(eval_df.loc[worst_idx, "step"])

        if {"rl_price", "lsm_price"} <= set(eval_df.columns):
            summary["rl_price_final"] = float(eval_df["rl_price"].iloc[-1])
            summary["lsm_price_final"] = float(eval_df["lsm_price"].iloc[-1])
            summary["final_price_gap"] = summary["rl_price_final"] - summary["lsm_price_final"]

    # Training signals (Episode return + critic/actor losses + action stats)
    train_summary = {}
    for tag in TRAINING_TAGS:
        tag_df = train_df[train_df["tag"] == tag]
        if tag_df.empty:
            continue
        values = tag_df["value"]
        steps = tag_df["step"]
        train_summary[tag] = {
            "final": float(values.iloc[-1]),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "slope_per_1k_paths": float(compute_trend(steps, values) * 1000.0),
        }
    summary["training_signals"] = train_summary

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TensorBoard event logs.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Event files or directories containing TensorBoard event files.",
    )
    parser.add_argument(
        "--out-json",
        default="analysis/tensorboard_summary.json",
        help="Path to write JSON summary (default: analysis/tensorboard_summary.json)",
    )
    parser.add_argument(
        "--out-csv",
        default="analysis/tensorboard_eval_metrics.csv",
        help="Path to write concatenated evaluation metrics CSV.",
    )
    args = parser.parse_args()

    event_files = resolve_event_files(args.paths)
    all_eval_tables = []
    summaries = []

    for event_file in event_files:
        run_name = event_file.parent.name
        df = load_scalars(event_file)
        eval_df = pivot_eval_metrics(df)
        eval_df.insert(0, "run", run_name)
        all_eval_tables.append(eval_df)
        summaries.append(summarize_run(run_name, eval_df, df))

    combined_eval = pd.concat(all_eval_tables, ignore_index=True, sort=False)
    combined_eval.to_csv(args.out_csv, index=False)

    with open(args.out_json, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Wrote evaluation metrics to {args.out_csv}")
    print(f"Wrote summary diagnostics to {args.out_json}")

    for summary in summaries:
        run = summary.pop("run", "unknown")
        print(f"\nRun: {run}")
        for key, value in summary.items():
            if key == "training_signals":
                print("  training_signals:")
                for tag, stats in value.items():
                    slope = stats["slope_per_1k_paths"]
                    print(
                        f"    {tag}: final={stats['final']:.4f}, mean={stats['mean']:.4f}, "
                        f"std={stats['std']:.4f}, slope/1k={slope:.4f}"
                    )
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
