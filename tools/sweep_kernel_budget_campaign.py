"""Expanded kernel-size and training-budget campaign for the semi-analytical study.

This script is meant to be run directly from the terminal by the user. It does
two things:

1. Runs a denser kernel-size sweep at a higher training budget so the kernel
   sizing plot is less confounded by early-training noise.
2. Runs a convergence study over training budgets for a smaller set of kernel
   sizes, then bootstraps both the mean Delta% and the across-seed std.

Why this design:
  * The current evidence suggests the interesting region is below and around
    M in [21, 36], so the grid is denser there than above it.
  * There is no evidence that going beyond M=78 is useful, so 78 is the high
    anchor and we stop there by default.
  * The kernel-size curve is evaluated at a higher default training budget
    (8192) than the earlier notebook anchor plot (4096), because the 8k runs
    already showed materially tighter seed dispersion.

Usage:
    cd /Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate EP11
    python tools/sweep_kernel_budget_campaign.py run --max_workers 4
    python tools/sweep_kernel_budget_campaign.py plot
    python tools/sweep_kernel_budget_campaign.py run --dry_run --seeds 11 12

Outputs:
  logs/_sweep_h1/sweep_kernel_budget_campaign.csv
  logs/_sweep_h1/sweep_kernel_budget_campaign.summary.json
  logs/_sweep_h1/kernel_size_bootstrap_curve.png
  logs/_sweep_h1/training_budget_convergence.png
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.stats_analysis import bootstrap_ci
from tools.sweep_expected_target import LOG_DIR, SweepConfig, base_args, run_one


CAMPAIGN_CSV = LOG_DIR / "sweep_kernel_budget_campaign.csv"
SUMMARY_JSON = LOG_DIR / "sweep_kernel_budget_campaign.summary.json"
KERNEL_PLOT = LOG_DIR / "kernel_size_bootstrap_curve.png"
CONVERGENCE_PLOT = LOG_DIR / "training_budget_convergence.png"

DEFAULT_SEEDS = list(range(11, 23))
DEFAULT_PATH_BUDGETS = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_KERNEL_CURVE_BUDGETS = [8192]

KERNEL_ON_BASE = {
    "--use_expected_target": "1",
    "--critic_warmup_episodes": "0",
}


def kernel_overrides(m_x: int, m_per_k: int, n_max: int) -> Dict[str, str]:
    return {
        **KERNEL_ON_BASE,
        "--kernel_M_x": str(m_x),
        "--kernel_M_per_k": str(m_per_k),
        "--kernel_N_max": str(n_max),
    }


# Denser where the current results suggest the elbow lives, plus modest upper
# anchors. Every point corresponds to a real quadrature composition.
BASE_KERNEL_CURVE_CONFIGS: List[Tuple[str, int, int, int]] = [
    ("K_M6", 2, 2, 1),
    ("K_M8", 2, 3, 1),
    ("K_M10", 2, 4, 1),
    ("K_M12", 3, 3, 1),
    ("K_M15", 3, 2, 2),
    ("K_M18", 3, 5, 1),
    ("K_M21", 3, 3, 2),
    ("K_M24", 6, 3, 1),
    ("K_M28", 4, 3, 2),
    ("K_M30", 6, 2, 2),
    ("K_M36", 4, 4, 2),
    ("K_M45", 5, 4, 2),
    ("K_M54", 6, 4, 2),
    ("K_M78", 6, 4, 3),
]

# Fill the low-M gaps so the kernel-size curve is not only an anchor plot.
# These are all exact quadrature compositions satisfying
#   M = M_x * (1 + N_max * M_per_k).
# Some prime totals necessarily require M_x = 1.
LOW_M_GAP_CONFIGS: List[Tuple[str, int, int, int]] = [
    ("K_M2", 1, 1, 1),
    ("K_M3", 1, 2, 1),
    ("K_M4", 2, 1, 1),
    ("K_M5", 1, 2, 2),
    ("K_M7", 1, 3, 2),
    ("K_M11", 1, 5, 2),
    ("K_M13", 1, 4, 3),
    ("K_M14", 2, 3, 2),
    ("K_M16", 4, 3, 1),
    ("K_M17", 1, 8, 2),
    ("K_M19", 1, 6, 3),
]

# Smaller subset for the budget-convergence curves. This keeps runtime lower
# while still spanning stupid-small, plausible default, and upper-anchor sizes.
CONVERGENCE_CONFIGS: List[Tuple[str, Optional[int], Optional[int], Optional[int]]] = [
    ("B0_baseline", None, None, None),
    ("K_M6", 2, 2, 1),
    ("K_M12", 3, 3, 1),
    ("K_M21", 3, 3, 2),
    ("K_M36", 4, 4, 2),
    ("K_M54", 6, 4, 2),
    ("K_M78", 6, 4, 3),
]


def infer_kernel_size(label: str) -> Optional[int]:
    if label.startswith("K_M"):
        try:
            return int(label.split("K_M", 1)[1])
        except ValueError:
            return None
    return None


def kernel_curve_configs(include_low_m_gap_fill: bool = True) -> List[Tuple[str, int, int, int]]:
    configs = list(BASE_KERNEL_CURVE_CONFIGS)
    if include_low_m_gap_fill:
        configs.extend(LOW_M_GAP_CONFIGS)
    unique: Dict[str, Tuple[str, int, int, int]] = {}
    for config in configs:
        unique[config[0]] = config
    return sorted(unique.values(), key=lambda item: infer_kernel_size(item[0]) or 0)


def filter_kernel_curve_configs(
    configs: Sequence[Tuple[str, int, int, int]],
    allowed_m_values: Optional[Sequence[int]] = None,
    min_m: Optional[int] = None,
    max_m: Optional[int] = None,
) -> List[Tuple[str, int, int, int]]:
    allowed = None if allowed_m_values is None else {int(val) for val in allowed_m_values}
    filtered: List[Tuple[str, int, int, int]] = []
    for config in configs:
        kernel_m = infer_kernel_size(config[0])
        if kernel_m is None:
            continue
        if allowed is not None and kernel_m not in allowed:
            continue
        if min_m is not None and kernel_m < min_m:
            continue
        if max_m is not None and kernel_m > max_m:
            continue
        filtered.append(config)
    return filtered


def delta_pct(eval_price: str, lsm_price: str) -> Optional[float]:
    try:
        eval_val = float(eval_price)
        lsm_val = float(lsm_price)
    except Exception:
        return None
    if not math.isfinite(eval_val) or not math.isfinite(lsm_val) or lsm_val <= 0:
        return None
    return (eval_val / lsm_val - 1.0) * 100.0


def mean_stat(arr: np.ndarray) -> float:
    return float(np.mean(arr))


def std_stat(arr: np.ndarray) -> float:
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def summarize_deltas(values: Sequence[float], n_bootstrap: int) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean_ci = bootstrap_ci(arr, stat_fn=mean_stat, n_bootstrap=n_bootstrap, seed=0)
    std_ci = bootstrap_ci(arr, stat_fn=std_stat, n_bootstrap=n_bootstrap, seed=1)
    return {
        "n_seeds": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "mean_ci_low": float(mean_ci.get("ci_low", np.nan)),
        "mean_ci_high": float(mean_ci.get("ci_high", np.nan)),
        "std_ci_low": float(std_ci.get("ci_low", np.nan)),
        "std_ci_high": float(std_ci.get("ci_high", np.nan)),
    }


def load_campaign_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Campaign CSV not found: {csv_path}")
    rows: List[Dict[str, str]] = []
    with open(csv_path) as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def run_key(run: Dict[str, object]) -> Tuple[str, str, int, int, str]:
    cfg = run["cfg"]
    assert isinstance(cfg, SweepConfig)
    return (
        str(run["study"]),
        cfg.label,
        int(cfg.seed),
        int(run["n_paths"]),
        "focal",
    )


def row_key(row: Dict[str, str]) -> Tuple[str, str, int, int, str]:
    return (
        row.get("study", ""),
        row.get("label", ""),
        int(row.get("seed", "0") or 0),
        int(row.get("n_paths", "0") or 0),
        row.get("contract", "focal") or "focal",
    )


def load_resume_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    deduped: Dict[Tuple[str, str, int, int, str], Dict[str, str]] = {}
    with open(csv_path) as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            deduped[row_key(row)] = row
    return list(deduped.values())


def label_for(study: str, kernel_label: str, n_paths: int) -> str:
    if study == "kernel_curve":
        return f"KC_{kernel_label}_N{n_paths}"
    return f"CV_{kernel_label}_N{n_paths}"


def make_run_specs(
    seeds: Sequence[int],
    path_budgets: Sequence[int],
    kernel_curve_budgets: Sequence[int],
    include_low_m_gap_fill: bool,
    kernel_curve_m_values: Optional[Sequence[int]],
    kernel_curve_min_m: Optional[int],
    kernel_curve_max_m: Optional[int],
    skip_budget_convergence: bool,
) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []

    kernel_configs = filter_kernel_curve_configs(
        kernel_curve_configs(include_low_m_gap_fill=include_low_m_gap_fill),
        allowed_m_values=kernel_curve_m_values,
        min_m=kernel_curve_min_m,
        max_m=kernel_curve_max_m,
    )

    for kernel_curve_budget in kernel_curve_budgets:
        for kernel_label, m_x, m_per_k, n_max in kernel_configs:
            kernel_m = infer_kernel_size(kernel_label)
            assert kernel_m is not None
            overrides = kernel_overrides(m_x, m_per_k, n_max)
            for seed in seeds:
                cfg = SweepConfig(
                    label=label_for("kernel_curve", kernel_label, int(kernel_curve_budget)),
                    overrides=dict(overrides),
                    seed=int(seed),
                    note=f"kernel_curve M={kernel_m} at n_paths={int(kernel_curve_budget)}",
                )
                runs.append(
                    {
                        "study": "kernel_curve",
                        "cfg": cfg,
                        "n_paths": int(kernel_curve_budget),
                        "kernel_label": kernel_label,
                        "kernel_M": kernel_m,
                        "kernel_M_x": m_x,
                        "kernel_M_per_k": m_per_k,
                        "kernel_N_max": n_max,
                    }
                )

    if not skip_budget_convergence:
        for n_paths in path_budgets:
            for kernel_label, m_x, m_per_k, n_max in CONVERGENCE_CONFIGS:
                if kernel_label == "B0_baseline":
                    overrides = {"--use_expected_target": "0"}
                    kernel_m = None
                else:
                    overrides = kernel_overrides(int(m_x), int(m_per_k), int(n_max))
                    kernel_m = infer_kernel_size(kernel_label)
                for seed in seeds:
                    cfg = SweepConfig(
                        label=label_for("budget_convergence", kernel_label, int(n_paths)),
                        overrides=dict(overrides),
                        seed=int(seed),
                        note=f"budget_convergence {kernel_label} at n_paths={n_paths}",
                    )
                    runs.append(
                        {
                            "study": "budget_convergence",
                            "cfg": cfg,
                            "n_paths": int(n_paths),
                            "kernel_label": kernel_label,
                            "kernel_M": kernel_m,
                            "kernel_M_x": m_x,
                            "kernel_M_per_k": m_per_k,
                            "kernel_N_max": n_max,
                        }
                    )

    return runs


def run_campaign(args: argparse.Namespace) -> None:
    runs = make_run_specs(
        args.seeds,
        args.path_budgets,
        args.kernel_curve_budgets,
        include_low_m_gap_fill=not args.skip_low_m_gap_fill,
        kernel_curve_m_values=args.kernel_curve_m_values,
        kernel_curve_min_m=args.kernel_curve_min_m,
        kernel_curve_max_m=args.kernel_curve_max_m,
        skip_budget_convergence=args.skip_budget_convergence,
    )
    kernel_configs = filter_kernel_curve_configs(
        kernel_curve_configs(include_low_m_gap_fill=not args.skip_low_m_gap_fill),
        allowed_m_values=args.kernel_curve_m_values,
        min_m=args.kernel_curve_min_m,
        max_m=args.kernel_curve_max_m,
    )
    print(
        f"Kernel/budget campaign: {len(runs)} runs at max_workers={args.max_workers} "
        f"({len(args.seeds)} seeds, {len(kernel_configs)} kernel points x {len(args.kernel_curve_budgets)} curve budgets, "
        f"{0 if args.skip_budget_convergence else len(CONVERGENCE_CONFIGS)} convergence configs x {len(args.path_budgets)} budgets)"
    )

    fieldnames = [
        "tag",
        "study",
        "label",
        "kernel_label",
        "kernel_M",
        "kernel_M_x",
        "kernel_M_per_k",
        "kernel_N_max",
        "seed",
        "n_paths",
        "contract",
        "status",
        "eval_price",
        "eval_price_se",
        "lsm_price",
        "final_avg100",
        "final_paths_per_sec",
        "training_time",
        "wall_seconds",
        "note",
        "overrides",
    ]
    existing_rows = load_resume_rows(CAMPAIGN_CSV) if args.resume else []
    completed_keys = {row_key(row) for row in existing_rows}
    pending_runs = [run for run in runs if run_key(run) not in completed_keys]

    if args.resume and existing_rows:
        print(f"Resume: found {len(existing_rows)} completed rows in {CAMPAIGN_CSV}")
        print(f"Resume: skipping {len(runs) - len(pending_runs)} completed runs, launching {len(pending_runs)} pending runs")

    rows: List[Dict[str, str]] = list(existing_rows)

    def execute(run: Dict[str, object]) -> Dict[str, str]:
        cfg = run["cfg"]
        assert isinstance(cfg, SweepConfig)
        base = base_args(n_paths=int(run["n_paths"]), contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix="_kbc")
        row.update(
            {
                "tag": "_kbc",
                "study": str(run["study"]),
                "kernel_label": str(run["kernel_label"]),
                "kernel_M": "" if run["kernel_M"] is None else str(run["kernel_M"]),
                "kernel_M_x": "" if run["kernel_M_x"] is None else str(run["kernel_M_x"]),
                "kernel_M_per_k": "" if run["kernel_M_per_k"] is None else str(run["kernel_M_per_k"]),
                "kernel_N_max": "" if run["kernel_N_max"] is None else str(run["kernel_N_max"]),
                "n_paths": str(run["n_paths"]),
                "contract": "focal",
            }
        )
        return row

    if args.dry_run:
        for run in pending_runs:
            cfg = run["cfg"]
            assert isinstance(cfg, SweepConfig)
            print(
                f"[DRY] {run['study']:<18} {cfg.label:<18} seed={cfg.seed:>3} "
                f"n={run['n_paths']} kernel={run['kernel_label']}"
            )
        return

    if not pending_runs:
        print(f"\nResults already complete in {CAMPAIGN_CSV} ({len(rows)} rows)")
        return

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(execute, run) for run in pending_runs]
        for future in cf.as_completed(futures):
            try:
                row = future.result()
                rows.append(row)
                with open(CAMPAIGN_CSV, "w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for written_row in rows:
                        writer.writerow({key: written_row.get(key, "") for key in fieldnames})
            except Exception as exc:
                print(f"FAIL: {exc}", file=sys.stderr)

    print(f"\nResults -> {CAMPAIGN_CSV} ({len(rows)} rows)")


def build_summary(rows: Iterable[Dict[str, str]], n_bootstrap: int) -> Dict[str, object]:
    kernel_curve_groups: Dict[Tuple[int, int], Dict[str, object]] = {}
    convergence_groups: Dict[Tuple[str, int], Dict[str, object]] = {}

    for row in rows:
        delta = delta_pct(row.get("eval_price", ""), row.get("lsm_price", ""))
        if delta is None:
            continue
        study = row.get("study", "")
        kernel_label = row.get("kernel_label", "")
        n_paths = int(row.get("n_paths", "0") or 0)
        wall = float(row.get("wall_seconds", "nan") or "nan")

        if study == "kernel_curve":
            kernel_m = int(row.get("kernel_M", "0") or 0)
            group = kernel_curve_groups.setdefault(
                (n_paths, kernel_m),
                {
                    "kernel_label": kernel_label,
                    "kernel_M": kernel_m,
                    "n_paths": n_paths,
                    "kernel_M_x": int(row.get("kernel_M_x", "0") or 0),
                    "kernel_M_per_k": int(row.get("kernel_M_per_k", "0") or 0),
                    "kernel_N_max": int(row.get("kernel_N_max", "0") or 0),
                    "deltas": [],
                    "walls": [],
                },
            )
            group["deltas"].append(delta)
            if math.isfinite(wall):
                group["walls"].append(wall)
        elif study == "budget_convergence":
            group = convergence_groups.setdefault(
                (kernel_label, n_paths),
                {
                    "kernel_label": kernel_label,
                    "kernel_M": None if row.get("kernel_M", "") == "" else int(row["kernel_M"]),
                    "n_paths": n_paths,
                    "deltas": [],
                    "walls": [],
                },
            )
            group["deltas"].append(delta)
            if math.isfinite(wall):
                group["walls"].append(wall)

    kernel_curve = []
    for (_, kernel_m), group in sorted(kernel_curve_groups.items(), key=lambda item: (item[0][0], item[0][1])):
        stats = summarize_deltas(group["deltas"], n_bootstrap=n_bootstrap)
        kernel_curve.append(
            {
                "kernel_label": group["kernel_label"],
                "kernel_M": kernel_m,
                "n_paths": group["n_paths"],
                "kernel_M_x": group["kernel_M_x"],
                "kernel_M_per_k": group["kernel_M_per_k"],
                "kernel_N_max": group["kernel_N_max"],
                "wall_mean": float(np.mean(group["walls"])) if group["walls"] else float("nan"),
                **stats,
            }
        )

    convergence = []
    for (_, _), group in sorted(convergence_groups.items(), key=lambda item: (item[0][0], item[0][1])):
        stats = summarize_deltas(group["deltas"], n_bootstrap=n_bootstrap)
        convergence.append(
            {
                "kernel_label": group["kernel_label"],
                "kernel_M": group["kernel_M"],
                "n_paths": group["n_paths"],
                "wall_mean": float(np.mean(group["walls"])) if group["walls"] else float("nan"),
                **stats,
            }
        )

    return {"kernel_curve": kernel_curve, "budget_convergence": convergence}


def plot_kernel_curve(summary: Dict[str, object]) -> None:
    kernel_curve = summary["kernel_curve"]
    if not kernel_curve:
        raise RuntimeError("No kernel-curve rows available for plotting.")

    budgets = sorted({int(row["n_paths"]) for row in kernel_curve})
    grouped: Dict[int, List[Dict[str, object]]] = {budget: [] for budget in budgets}
    for row in kernel_curve:
        grouped[int(row["n_paths"])].append(row)

    cmap = plt.get_cmap("viridis")
    color_positions = np.linspace(0.15, 0.9, len(budgets))
    colors = {budget: cmap(color_positions[idx]) for idx, budget in enumerate(budgets)}

    fig, (ax_mean, ax_wall) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for budget in budgets:
        rows = sorted(grouped[budget], key=lambda row: int(row["kernel_M"]))
        x = [int(row["kernel_M"]) for row in rows]
        y_mean = [float(row["mean"]) for row in rows]
        y_mean_low = [float(row["mean"]) - float(row["mean_ci_low"]) for row in rows]
        y_mean_high = [float(row["mean_ci_high"]) - float(row["mean"]) for row in rows]
        y_wall = [float(row["wall_mean"]) for row in rows]
        color = colors[budget]
        ax_mean.errorbar(
            x,
            y_mean,
            yerr=np.vstack([y_mean_low, y_mean_high]),
            fmt="o-",
            capsize=3,
            lw=1.8,
            ms=5,
            label=f"n_paths={budget}",
            color=color,
        )
        ax_wall.plot(x, y_wall, "s--", lw=1.6, ms=4.5, label=f"n_paths={budget}", color=color)

    xticks = sorted({int(row["kernel_M"]) for row in kernel_curve})
    ax_mean.axhline(0.0, color="k", lw=0.5, ls="--")
    ax_mean.set_ylabel("Delta% mean (bootstrap CI)")
    ax_mean.set_title("Kernel-size curve across training budgets")
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend(loc="best", fontsize=8)

    ax_wall.set_ylabel("Wall-clock per run (s)")
    ax_wall.set_xlabel("Kernel size M")
    ax_wall.grid(True, alpha=0.3)
    ax_wall.set_xscale("log")
    ax_wall.set_xticks(xticks)
    ax_wall.set_xticklabels([str(int(val)) for val in xticks])

    fig.tight_layout()
    fig.savefig(KERNEL_PLOT, dpi=180)
    plt.close(fig)


def plot_budget_convergence(summary: Dict[str, object]) -> None:
    convergence = summary["budget_convergence"]
    if not convergence:
        return

    order = [label for label, _, _, _ in CONVERGENCE_CONFIGS]
    budgets = sorted({int(row["n_paths"]) for row in convergence})
    palette = {
        "B0_baseline": "#444444",
        "K_M6": "#d62728",
        "K_M12": "#ff7f0e",
        "K_M21": "#2ca02c",
        "K_M36": "#1f77b4",
        "K_M54": "#9467bd",
        "K_M78": "#8c564b",
    }

    grouped: Dict[str, List[Dict[str, object]]] = {label: [] for label in order}
    for row in convergence:
        grouped.setdefault(str(row["kernel_label"]), []).append(row)

    fig, (ax_mean, ax_std) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label in order:
        rows = sorted(grouped.get(label, []), key=lambda row: row["n_paths"])
        if not rows:
            continue
        x = [row["n_paths"] for row in rows]
        y_mean = [row["mean"] for row in rows]
        y_mean_low = [row["mean"] - row["mean_ci_low"] for row in rows]
        y_mean_high = [row["mean_ci_high"] - row["mean"] for row in rows]
        y_std = [row["std"] for row in rows]
        y_std_low = [row["std"] - row["std_ci_low"] for row in rows]
        y_std_high = [row["std_ci_high"] - row["std"] for row in rows]
        color = palette.get(label, None)
        label_text = label if label == "B0_baseline" else f"{label} (M={rows[0]['kernel_M']})"

        ax_mean.errorbar(
            x,
            y_mean,
            yerr=np.vstack([y_mean_low, y_mean_high]),
            fmt="o-",
            capsize=4,
            lw=2,
            ms=6,
            label=label_text,
            color=color,
        )
        ax_std.errorbar(
            x,
            y_std,
            yerr=np.vstack([y_std_low, y_std_high]),
            fmt="o-",
            capsize=4,
            lw=2,
            ms=6,
            label=label_text,
            color=color,
        )

    ax_mean.axhline(0.0, color="k", lw=0.5, ls="--")
    ax_mean.set_ylabel("Delta% mean")
    ax_mean.set_title("Training-budget convergence with bootstrap mean and std estimates")
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend(loc="best", fontsize=8)

    ax_std.set_ylabel("Across-seed std")
    ax_std.set_xlabel("Training budget n_paths")
    ax_std.grid(True, alpha=0.3)
    ax_std.set_xscale("log")
    ax_std.set_xticks(budgets)
    ax_std.set_xticklabels([str(val) for val in budgets])

    fig.tight_layout()
    fig.savefig(CONVERGENCE_PLOT, dpi=180)
    plt.close(fig)


def write_summary(summary: Dict[str, object]) -> None:
    with open(SUMMARY_JSON, "w") as handle:
        json.dump(summary, handle, indent=2)


def plot_campaign(args: argparse.Namespace) -> None:
    rows = load_campaign_rows(Path(args.csv))
    summary = build_summary(rows, n_bootstrap=args.bootstrap)
    write_summary(summary)
    plot_kernel_curve(summary)
    plot_budget_convergence(summary)
    print(f"Summary -> {SUMMARY_JSON}")
    print(f"Kernel plot -> {KERNEL_PLOT}")
    if summary.get("budget_convergence"):
        print(f"Convergence plot -> {CONVERGENCE_PLOT}")
    else:
        print("Convergence plot -> skipped (no budget_convergence rows in CSV)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--max_workers", type=int, default=4)
    run_parser.add_argument("--kernel_curve_budgets", type=int, nargs="+", default=DEFAULT_KERNEL_CURVE_BUDGETS)
    run_parser.add_argument("--path_budgets", type=int, nargs="+", default=DEFAULT_PATH_BUDGETS)
    run_parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    run_parser.add_argument("--skip_low_m_gap_fill", action="store_true")
    run_parser.add_argument("--kernel_curve_m_values", type=int, nargs="+")
    run_parser.add_argument("--kernel_curve_min_m", type=int)
    run_parser.add_argument("--kernel_curve_max_m", type=int)
    run_parser.add_argument("--skip_budget_convergence", action="store_true")
    run_parser.add_argument("--resume", action="store_true", default=True)
    run_parser.add_argument("--no_resume", action="store_false", dest="resume")
    run_parser.add_argument("--dry_run", action="store_true")

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--csv", type=str, default=str(CAMPAIGN_CSV))
    plot_parser.add_argument("--bootstrap", type=int, default=4000)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_campaign(args)
    elif args.command == "plot":
        plot_campaign(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()