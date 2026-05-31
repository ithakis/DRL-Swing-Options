#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIGS = (
    "SwingOption_20_c0.04_gamma1",
    "SwingOption_20_c0.04_gamma1.5",
    "SwingOption_20_c0.04_gamma2",
)
DEFAULT_SEEDS = (11, 12, 13)
PRICE_LINE_RE = re.compile(
    r"Option Price:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*±\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)

CONFIG_SPECS = {
    "SwingOption_20_c0.04_gamma1": {
        "label": "c=0.04, gamma_c=1",
        "gamma_cost": "1",
    },
    "SwingOption_20_c0.04_gamma1.5": {
        "label": "c=0.04, gamma_c=1.5",
        "gamma_cost": "1.5",
    },
    "SwingOption_20_c0.04_gamma2": {
        "label": "c=0.04, gamma_c=2",
        "gamma_cost": "2",
    },
}

COMMON_TRAIN_ARGS = (
    "-n_paths=32768",
    "-eval_every=-1",
    "-n_paths_eval=65536",
    "-munchausen=0",
    "-nstep=1",
    "--per_alpha=0.1",
    "--per_beta_start=1.0",
    "--per_beta_frames=120000",
    "--per_priority_floor=5e-6",
    "--per_priority_clip_pct=99.7",
    "--per_alpha_final=0.20",
    "--per_alpha_ramp_start=5000",
    "--per_alpha_ramp_end=25000",
    "--per_beta_final=0.98",
    "--gamma=1",
    "-learn_every=2",
    "-learn_number=1",
    "-iqn=0",
    "-noise_sigma0=1.30",
    "-noise_floor=0.26",
    "-noise_plateau=3200",
    "-per=1",
    "--min_replay_size=18000",
    "--max_replay_size=200000",
    "-t=0.0032",
    "-bs=128",
    "-layer_size=64",
    "--activation=silu",
    "--norm=layernorm",
    "--init_method=orthogonal",
    "-lr_a=1.6e-4",
    "-lr_c=9.0e-5",
    "--final_lr_fraction=0.20",
    "--warmup_episodes=1024",
    "--lr_schedule_episodes=40000",
    "--min_lr=1e-6",
    "--actor_grad_clip=1.0",
    "--critic_grad_clip=2.5",
    "--actor_grad_clip_type=norm",
    "--critic_grad_clip_type=norm",
    "--grad_clip_norm_type=2.0",
    "--weight_decay_actor=5e-5",
    "--weight_decay_critic=1.2e-4",
    "--critic_ema_decay=0.0",
    "--target_policy_noise=0.15",
    "--target_policy_clip=0.25",
    "--compile=0",
    "-n_cores=4",
    "--disable_csv_logging=1",
    "--limit_logging_frequency=1",
    "--critic_warmup_episodes=1024",
    "--adaptive_noise_scale=0.6",
    "--actor_output_activation=beta_sigmoid_3.0",
    "--warmup_noise_fraction=0.4",
    "--target_noise_decay_start=20000",
    "--target_noise_floor=0.04",
    "--use_robust_normalization=1",
    "--strike=1.0",
    "--maturity=0.0833",
    "--n_rights=22",
    "--q_min=0.0",
    "--q_max=2.0",
    "--Q_min=0.0",
    "--Q_max=20.0",
    "--risk_free_rate=0.05",
    "--min_refraction_periods=0",
    "--c_cost=0.04",
    "--lsm_basis=chebyshev",
    "--lsm_degree=7",
    "--lsm_reg=none",
    "--lsm_reg_alpha=1e-6",
    "--alpha=12.0",
    "--sigma=1.2",
    "--beta=150.0",
    "--lam=6.0",
    "--mu_J=0.3",
)


@dataclass(frozen=True)
class RunRecord:
    config: str
    config_label: str
    s0: float
    seed: int
    run_name: str
    option_price: float
    confidence_95: float
    log_path: str


@dataclass(frozen=True)
class PendingRun:
    config: str
    s0: float
    seed: int
    run_name: str
    command: tuple[str, ...]
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the three convex-cost what-if RL configurations from scratch across an S0 grid, "
            "and write fresh price-curve CSVs."
        )
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated config names to retrain. Defaults to the three convex-cost what-if cases.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated RL seeds to retrain for each S0 point.",
    )
    parser.add_argument("--s0-min", type=float, default=0.5, help="Lower bound of the S0 grid.")
    parser.add_argument("--s0-max", type=float, default=2.0, help="Upper bound of the S0 grid.")
    parser.add_argument("--s0-count", type=int, default=15, help="Number of S0 grid points.")
    parser.add_argument(
        "--s0-values",
        type=str,
        default="",
        help="Optional explicit comma-separated S0 values. Overrides --s0-min/--s0-max/--s0-count.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="",
        help="Optional prefix for generated run names. Defaults to a timestamped prefix.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/retrain_s0_price_curve",
        help="Directory where raw CSV, summary CSV, and per-run logs are written.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch run.py. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without starting training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs already present in the raw output CSV.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum number of concurrent run.py training jobs to execute. Defaults to 1.",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_seed_list(text: str) -> List[int]:
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("Need at least one seed.")
    return seeds


def linspace(start: float, stop: float, count: int) -> List[float]:
    if count < 1:
        raise ValueError("s0-count must be >= 1")
    if count == 1:
        return [float(start)]
    step = (stop - start) / float(count - 1)
    return [round(start + index * step, 8) for index in range(count)]


def parse_s0_grid(args: argparse.Namespace) -> List[float]:
    if args.s0_values.strip():
        values = [float(part.strip()) for part in args.s0_values.split(",") if part.strip()]
    else:
        values = linspace(args.s0_min, args.s0_max, args.s0_count)
    unique_values = sorted({round(value, 8) for value in values})
    if not unique_values:
        raise ValueError("Need at least one S0 value.")
    return unique_values


def safe_s0_token(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def build_run_name(prefix: str, config: str, s0: float, seed: int) -> str:
    return f"{prefix}_{config}_s0_{safe_s0_token(s0)}_seed{seed}"


def build_command(python_executable: str, config: str, s0: float, seed: int, run_name: str) -> List[str]:
    if config not in CONFIG_SPECS:
        raise KeyError(f"Unsupported config for full retraining script: {config}")
    gamma_cost = CONFIG_SPECS[config]["gamma_cost"]
    command = [python_executable, "run.py", *COMMON_TRAIN_ARGS]
    command.extend(
        [
            f"--gamma_cost={gamma_cost}",
            f"--S0={s0:.8f}",
            f"-name={run_name}",
            f"-seed={seed}",
        ]
    )
    return command


def print_command(command: Sequence[str]) -> None:
    print("$", " ".join(command))


def run_and_extract_price(command: Sequence[str], log_path: Path, *, echo_output: bool) -> tuple[float, float]:
    last_price: float | None = None
    last_ci95: float | None = None
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if echo_output:
                print(line, end="")
            log_file.write(line)
            match = PRICE_LINE_RE.search(line)
            if match:
                last_price = float(match.group(1))
                last_ci95 = float(match.group(2))
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if last_price is None or last_ci95 is None:
        raise RuntimeError(f"Could not parse final option price from {log_path}")
    return last_price, last_ci95


def write_raw_csv(records: Sequence[RunRecord], raw_csv_path: Path) -> None:
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config",
                "config_label",
                "s0",
                "seed",
                "run_name",
                "option_price",
                "confidence_95",
                "log_path",
            ],
        )
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item.config, item.s0, item.seed)):
            writer.writerow(
                {
                    "config": record.config,
                    "config_label": record.config_label,
                    "s0": f"{record.s0:.8f}",
                    "seed": record.seed,
                    "run_name": record.run_name,
                    "option_price": f"{record.option_price:.8f}",
                    "confidence_95": f"{record.confidence_95:.8f}",
                    "log_path": record.log_path,
                }
            )


def write_summary_csv(records: Sequence[RunRecord], summary_csv_path: Path) -> None:
    grouped: Dict[tuple[str, float], List[RunRecord]] = {}
    for record in records:
        grouped.setdefault((record.config, record.s0), []).append(record)

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config",
                "config_label",
                "s0",
                "n_seeds",
                "selected_seeds",
                "price_mean",
                "price_seed_std",
                "confidence_95_mean",
            ],
        )
        writer.writeheader()
        for (config, s0), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            prices = [record.option_price for record in group]
            ci95_values = [record.confidence_95 for record in group]
            price_mean = sum(prices) / len(prices)
            if len(prices) > 1:
                variance = sum((value - price_mean) ** 2 for value in prices) / float(len(prices) - 1)
                seed_std = math.sqrt(variance)
            else:
                seed_std = 0.0
            writer.writerow(
                {
                    "config": config,
                    "config_label": CONFIG_SPECS[config]["label"],
                    "s0": f"{s0:.8f}",
                    "n_seeds": len(group),
                    "selected_seeds": ",".join(
                        str(record.seed) for record in sorted(group, key=lambda item: item.seed)
                    ),
                    "price_mean": f"{price_mean:.8f}",
                    "price_seed_std": f"{seed_std:.8f}",
                    "confidence_95_mean": f"{sum(ci95_values) / len(ci95_values):.8f}",
                }
            )


def load_existing_records(raw_csv_path: Path) -> Dict[tuple[str, float, int], RunRecord]:
    if not raw_csv_path.exists():
        return {}
    records: Dict[tuple[str, float, int], RunRecord] = {}
    with raw_csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["config"], round(float(row["s0"]), 8), int(row["seed"]))
            records[key] = RunRecord(
                config=row["config"],
                config_label=row["config_label"],
                s0=round(float(row["s0"]), 8),
                seed=int(row["seed"]),
                run_name=row["run_name"],
                option_price=float(row["option_price"]),
                confidence_95=float(row["confidence_95"]),
                log_path=row["log_path"],
            )
    return records


def validate_configs(configs: Iterable[str]) -> List[str]:
    validated = []
    for config in configs:
        if config not in CONFIG_SPECS:
            supported = ", ".join(CONFIG_SPECS)
            raise ValueError(f"Unsupported config '{config}'. Supported configs: {supported}")
        validated.append(config)
    if not validated:
        raise ValueError("Need at least one config.")
    return validated


def build_pending_run(
    python_executable: str,
    run_prefix: str,
    log_dir: Path,
    config: str,
    s0: float,
    seed: int,
) -> PendingRun:
    run_name = build_run_name(run_prefix, config, s0, seed)
    command = tuple(build_command(python_executable, config, s0, seed, run_name))
    log_path = log_dir / f"{run_name}.log"
    return PendingRun(
        config=config,
        s0=round(s0, 8),
        seed=seed,
        run_name=run_name,
        command=command,
        log_path=log_path,
    )


def execute_pending_run(pending_run: PendingRun, *, echo_output: bool) -> RunRecord:
    price, ci95 = run_and_extract_price(pending_run.command, pending_run.log_path, echo_output=echo_output)
    return RunRecord(
        config=pending_run.config,
        config_label=str(CONFIG_SPECS[pending_run.config]["label"]),
        s0=pending_run.s0,
        seed=pending_run.seed,
        run_name=pending_run.run_name,
        option_price=price,
        confidence_95=ci95,
        log_path=str(pending_run.log_path),
    )


def main() -> None:
    args = parse_args()
    configs = validate_configs(parse_csv_list(args.configs))
    seeds = parse_seed_list(args.seeds)
    s0_grid = parse_s0_grid(args)
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = args.run_prefix.strip() or f"fullopt_s0_curve_{timestamp}"
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    log_dir = output_dir / "run_logs"
    raw_csv_path = output_dir / "price_curve_raw.csv"
    summary_csv_path = output_dir / "price_curve_summary.csv"
    log_dir.mkdir(parents=True, exist_ok=True)

    existing_records = load_existing_records(raw_csv_path) if args.resume else {}
    records = dict(existing_records)

    print(f"Repository root: {REPO_ROOT}")
    print(f"Output directory: {output_dir}")
    print(f"Configs: {', '.join(configs)}")
    print(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"S0 grid ({len(s0_grid)} points): {', '.join(f'{value:.4f}' for value in s0_grid)}")
    print(f"Parallel jobs: {args.jobs}")
    if args.dry_run:
        print("Dry-run mode enabled. No training jobs will be started.")
    if args.resume:
        print(f"Resume mode enabled. Found {len(existing_records)} existing raw records.")

    pending_runs: List[PendingRun] = []

    for config in configs:
        print(f"\n=== {config} ===")
        for s0 in s0_grid:
            print(f"\n--- S0={s0:.4f} ---")
            for seed in seeds:
                key = (config, round(s0, 8), seed)
                if key in records:
                    print(f"Skipping existing result for {config} | S0={s0:.4f} | seed={seed}")
                    continue

                pending_run = build_pending_run(args.python, run_prefix, log_dir, config, s0, seed)
                print_command(pending_run.command)
                if args.dry_run:
                    continue
                pending_runs.append(pending_run)

    if args.dry_run:
        return

    if not pending_runs:
        print("\nNo pending runs remain.")
        write_raw_csv(records.values(), raw_csv_path)
        write_summary_csv(records.values(), summary_csv_path)
        return

    print(f"\nLaunching {len(pending_runs)} pending runs with up to {args.jobs} concurrent workers.")

    if args.jobs == 1:
        for index, pending_run in enumerate(pending_runs, start=1):
            print(
                f"[{index}/{len(pending_runs)}] Starting {pending_run.config} | "
                f"S0={pending_run.s0:.4f} | seed={pending_run.seed}"
            )
            record = execute_pending_run(pending_run, echo_output=True)
            records[(record.config, record.s0, record.seed)] = record
            write_raw_csv(records.values(), raw_csv_path)
            write_summary_csv(records.values(), summary_csv_path)
            print(
                f"Recorded {record.config} | S0={record.s0:.4f} | seed={record.seed} | "
                f"price={record.option_price:.6f} | CI95=±{record.confidence_95:.6f}"
            )
    else:
        future_to_run: Dict[Future[RunRecord], PendingRun] = {}
        pending_iter = iter(pending_runs)
        completed = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            while len(future_to_run) < args.jobs:
                try:
                    pending_run = next(pending_iter)
                except StopIteration:
                    break
                print(
                    f"[launch] {pending_run.config} | S0={pending_run.s0:.4f} | seed={pending_run.seed} | "
                    f"log={pending_run.log_path}"
                )
                future = executor.submit(execute_pending_run, pending_run, echo_output=False)
                future_to_run[future] = pending_run

            while future_to_run:
                done, _ = wait(tuple(future_to_run), return_when=FIRST_COMPLETED)
                for future in done:
                    pending_run = future_to_run.pop(future)
                    record = future.result()
                    completed += 1
                    records[(record.config, record.s0, record.seed)] = record
                    write_raw_csv(records.values(), raw_csv_path)
                    write_summary_csv(records.values(), summary_csv_path)
                    print(
                        f"[done {completed}/{len(pending_runs)}] {record.config} | S0={record.s0:.4f} | "
                        f"seed={record.seed} | price={record.option_price:.6f} | "
                        f"CI95=±{record.confidence_95:.6f}"
                    )
                    try:
                        next_run = next(pending_iter)
                    except StopIteration:
                        continue
                    print(
                        f"[launch] {next_run.config} | S0={next_run.s0:.4f} | seed={next_run.seed} | "
                        f"log={next_run.log_path}"
                    )
                    next_future = executor.submit(execute_pending_run, next_run, echo_output=False)
                    future_to_run[next_future] = next_run

    write_raw_csv(records.values(), raw_csv_path)
    write_summary_csv(records.values(), summary_csv_path)
    print("\nFinished full re-optimization sweep.")
    print(f"Raw results: {raw_csv_path}")
    print(f"Summary curve: {summary_csv_path}")


if __name__ == "__main__":
    main()
