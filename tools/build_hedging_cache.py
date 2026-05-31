#!/usr/bin/env python3
"""Build a shared RL and LSM hedging cache on common HHK evaluation paths.

This script is the producer-side foundation for the future `Hedging.ipynb`.
It reuses saved pricing runs, evaluates RL and LSM on shared paths, normalizes
their per-path traces into one schema, and writes compressed Parquet outputs
that are fast to load in notebook workflows.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent_evaluation import _evaluate_swing_batch
from src.hedging_utils import (
    HedgingTraceSummary,
    normalize_lsm_trace,
    normalize_rl_trace,
    summarize_trace,
    write_trace_parquet,
)
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos
from src.simulate_hhk_spot import no_seasonal_function, simulate_hhk_spot
from tools.rebuild_results_v7 import (
    LSM_N_ACTIONS,
    LSM_TRAIN_N_PATHS,
    LSM_TRAIN_SEED,
    TEST_N_PATHS,
    TEST_SEED,
    build_agent,
    build_contract,
    build_hhk_params,
    discover_runs,
    dotdict,
    parse_config_key,
)

DEFAULT_RESULTS_CSV = REPO_ROOT / "Jupyter Notebooks" / "Convex Costs Results 7.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "hedging_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="Comma-separated config base names to process. Default: all discovered runs.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="RL,LSM",
        help="Comma-separated subset of methods to build from {RL,LSM}.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional config limit for debugging.")
    parser.add_argument(
        "--max_seeds",
        type=int,
        default=0,
        help="Optional number of RL seeds per config to keep (0 = all available seeds).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated RL seeds to include for every config.",
    )
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Canonical results CSV used only to preserve config ordering when present.",
    )
    parser.add_argument("--runs_dir", type=Path, default=REPO_ROOT / "runs", help="Directory with saved runs.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for normalized hedging cache outputs.",
    )
    parser.add_argument(
        "--n_paths_eval",
        type=int,
        default=TEST_N_PATHS,
        help="Number of common evaluation paths per config.",
    )
    parser.add_argument(
        "--lsm_train_paths",
        type=int,
        default=LSM_TRAIN_N_PATHS,
        help="Number of paths used to fit LSM estimators.",
    )
    parser.add_argument(
        "--test_seed",
        type=int,
        default=TEST_SEED,
        help="Seed for the shared evaluation dataset.",
    )
    parser.add_argument(
        "--lsm_train_seed",
        type=int,
        default=LSM_TRAIN_SEED,
        help="Seed for the separate LSM training dataset.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4096,
        help="Batch size for batched RL evaluation.",
    )
    parser.add_argument(
        "--lsm_basis",
        type=str,
        default="chebyshev",
        help="LSM basis used for cache generation.",
    )
    parser.add_argument("--lsm_degree", type=int, default=2, help="LSM polynomial degree.")
    parser.add_argument("--lsm_reg", type=str, default="none", help="LSM regularization mode.")
    parser.add_argument(
        "--lsm_reg_alpha",
        type=float,
        default=1e-6,
        help="LSM regularization strength.",
    )
    parser.add_argument(
        "--lsm_n_actions",
        type=int,
        default=LSM_N_ACTIONS,
        help="Discrete LSM action grid size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache artifacts if they already exist.",
    )
    return parser.parse_args()


def load_run_params(json_path: Path) -> dotdict:
    with json_path.open("r") as handle:
        return dotdict(json.load(handle))


def dataset_cache_key(
    *,
    hhk_params: Dict[str, Any],
    n_paths: int,
    seed: int,
) -> Tuple[Any, ...]:
    return (
        float(hhk_params["S0"]),
        float(hhk_params["T"]),
        int(hhk_params["n_steps"]),
        float(hhk_params["alpha"]),
        float(hhk_params["sigma"]),
        float(hhk_params["beta"]),
        float(hhk_params["lam"]),
        float(hhk_params["mu_J"]),
        str(np.dtype(hhk_params["dtype"])),
        int(n_paths),
        int(seed),
    )


def get_or_generate_dataset(
    cache: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    hhk_params: Dict[str, Any],
    n_paths: int,
    seed: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = dataset_cache_key(hhk_params=hhk_params, n_paths=n_paths, seed=seed)
    if key not in cache:
        cache[key] = simulate_hhk_spot(
            **hhk_params,
            n_paths=n_paths,
            seed=seed,
            stratify=True,
            batch_size=batch_size,
        )
    return cache[key]


def ordered_configs(all_runs: Dict[str, List[Tuple[int, str, str]]], results_csv: Path) -> List[str]:
    discovered = set(all_runs.keys())
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        if "Configuration" in df.columns:
            ordered = [cfg for cfg in df["Configuration"].tolist() if cfg in discovered]
            remainder = sorted(discovered.difference(ordered), key=lambda k: parse_config_key(k + "_0"))
            return ordered + remainder
    return sorted(discovered, key=lambda k: parse_config_key(k + "_0"))


def select_configs(
    *,
    all_runs: Dict[str, List[Tuple[int, str, str]]],
    results_csv: Path,
    requested: Sequence[str],
    limit: int,
) -> List[str]:
    if requested:
        missing = [cfg for cfg in requested if cfg not in all_runs]
        if missing:
            raise ValueError(f"Unknown configs requested: {missing}")
        configs = list(requested)
    else:
        configs = ordered_configs(all_runs, results_csv)
    if limit > 0:
        configs = configs[:limit]
    return configs


def select_rl_runs(
    runs: Sequence[Tuple[int, str, str]],
    *,
    requested_seeds: Sequence[int],
    max_seeds: int,
) -> List[Tuple[int, str, str]]:
    sorted_runs = sorted(runs, key=lambda item: item[0])
    if requested_seeds:
        seed_set = set(requested_seeds)
        sorted_runs = [record for record in sorted_runs if record[0] in seed_set]
    if max_seeds > 0:
        sorted_runs = sorted_runs[:max_seeds]
    return sorted_runs


def rl_trace_output_path(output_root: Path, config: str, seed: int) -> Path:
    return output_root / f"config={config}" / "method=RL" / f"seed={seed}" / "traces.parquet"


def lsm_trace_output_path(output_root: Path, config: str, seed: int) -> Path:
    return output_root / f"config={config}" / "method=LSM" / f"seed={seed}" / "traces.parquet"


def evaluate_rl_trace(
    *,
    params: dotdict,
    pth_path: Path,
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: str,
    seed: int,
    eval_batch_size: int,
) -> pd.DataFrame:
    contract = build_contract(params)
    hhk_params = build_hhk_params(params)
    agent = build_agent(params)
    agent.actor_local.load_state_dict(torch.load(pth_path, map_location="cpu"))  # type: ignore[arg-type]
    agent.actor_local.eval()
    if hasattr(agent, "actor_target"):
        agent.actor_target.eval()

    all_rows: List[List[float]] = []
    n_paths = dataset[1].shape[0]
    batch_size = max(1, min(int(eval_batch_size), n_paths))
    for start in range(0, n_paths, batch_size):
        end = min(start + batch_size, n_paths)
        _, _, batch_rows = _evaluate_swing_batch(
            agent=agent,
            contract=contract,
            dataset=dataset,
            batch_indices=list(range(start, end)),
            collect_path_data=True,
        )
        all_rows.extend(batch_rows)

    metadata = {
        "config": config,
        "method": "RL",
        "run_name": config,
        "seed": seed,
    }
    return normalize_rl_trace(
        all_rows,
        metadata=metadata,
        contract=contract,
        t_grid=dataset[0],
        hhk_params=hhk_params,
        seasonal_fn=no_seasonal_function,
    )


def evaluate_lsm_trace(
    *,
    params: dotdict,
    config: str,
    test_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    train_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_seed: int,
    lsm_basis: str,
    lsm_degree: int,
    lsm_reg: str,
    lsm_reg_alpha: float,
    lsm_n_actions: int,
) -> pd.DataFrame:
    contract = build_contract(params)
    hhk_params = build_hhk_params(params)
    estimators = fit_lsm_estimators(
        contract=contract,
        dataset=tuple(np.asarray(arr, dtype=np.float64) for arr in train_dataset),
        poly_degree=lsm_degree,
        basis_type=lsm_basis,
        state_mode="full",
        reg_type=lsm_reg,
        reg_alpha=lsm_reg_alpha,
        n_actions=lsm_n_actions,
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
        temp_path = Path(handle.name)

    try:
        price_swing_option_lsm_oos(
            contract=contract,
            dataset=tuple(np.asarray(arr, dtype=np.float64) for arr in test_dataset),
            estimators=estimators,
            seed=output_seed + 1,
            csv_path=str(temp_path),
            _print_results=False,
        )
        raw_df = pd.read_parquet(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    metadata = {
        "config": config,
        "method": "LSM",
        "run_name": f"{config}_lsm",
        "seed": output_seed,
    }
    return normalize_lsm_trace(
        raw_df,
        metadata=metadata,
        contract=contract,
        dataset=test_dataset,
        hhk_params=hhk_params,
        seasonal_fn=no_seasonal_function,
    )


def write_manifest(output_root: Path, summaries: Sequence[HedgingTraceSummary]) -> Path:
    manifest_path = output_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary.as_dict() for summary in summaries]).to_csv(manifest_path, index=False)
    return manifest_path


def main() -> None:
    args = parse_args()
    methods = {item.strip().upper() for item in args.methods.split(",") if item.strip()}
    if not methods.issubset({"RL", "LSM"}):
        raise ValueError("methods must be drawn from {RL,LSM}")
    requested_configs = [cfg.strip() for cfg in args.configs.split(",") if cfg.strip()]
    requested_seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    all_runs = discover_runs(str(args.runs_dir))
    configs = select_configs(
        all_runs=all_runs,
        results_csv=args.results_csv,
        requested=requested_configs,
        limit=args.limit,
    )

    if not configs:
        raise ValueError("No configs selected for hedging cache build")

    summaries: List[HedgingTraceSummary] = []
    dataset_cache: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    output_root = args.output_root.resolve()

    print("=" * 72)
    print("BUILD HEDGING CACHE")
    print(f"  Configs: {len(configs)}")
    print(f"  Methods: {sorted(methods)}")
    print(f"  Eval paths: {args.n_paths_eval} | Test seed: {args.test_seed}")
    print(f"  Output root: {output_root}")
    print("=" * 72)

    for config in configs:
        rl_runs = select_rl_runs(
            all_runs.get(config, []), requested_seeds=requested_seeds, max_seeds=args.max_seeds
        )
        if not rl_runs:
            print(f"Skipping {config}: no matching RL seeds")
            continue

        first_params = load_run_params(Path(rl_runs[0][1]))
        hhk_params = build_hhk_params(first_params)
        test_dataset = get_or_generate_dataset(
            dataset_cache,
            hhk_params=hhk_params,
            n_paths=args.n_paths_eval,
            seed=args.test_seed,
            batch_size=args.eval_batch_size,
        )
        train_dataset = get_or_generate_dataset(
            dataset_cache,
            hhk_params=hhk_params,
            n_paths=args.lsm_train_paths,
            seed=args.lsm_train_seed,
            batch_size=128,
        )

        print(f"\n[{config}]")
        print(f"  RL seeds: {[seed for seed, _, _ in rl_runs]}")

        if "RL" in methods:
            for seed, json_path, pth_path in rl_runs:
                output_path = rl_trace_output_path(output_root, config, seed)
                if output_path.exists() and not args.overwrite:
                    print(f"  RL seed {seed}: skipping existing {output_path}")
                    continue
                print(f"  RL seed {seed}: evaluating and writing {output_path}")
                params = load_run_params(Path(json_path))
                trace_df = evaluate_rl_trace(
                    params=params,
                    pth_path=Path(pth_path),
                    dataset=test_dataset,
                    config=config,
                    seed=seed,
                    eval_batch_size=args.eval_batch_size,
                )
                write_trace_parquet(trace_df, output_path)
                summaries.append(
                    summarize_trace(
                        trace_df,
                        config=config,
                        method="RL",
                        run_name=config,
                        seed=seed,
                        artifact_path=output_path,
                        q_max=build_contract(params).q_max,
                    )
                )

        if "LSM" in methods:
            lsm_seed = args.lsm_train_seed
            output_path = lsm_trace_output_path(output_root, config, lsm_seed)
            if output_path.exists() and not args.overwrite:
                print(f"  LSM: skipping existing {output_path}")
            else:
                print(f"  LSM: fitting, evaluating, and writing {output_path}")
                trace_df = evaluate_lsm_trace(
                    params=first_params,
                    config=config,
                    test_dataset=test_dataset,
                    train_dataset=train_dataset,
                    output_seed=lsm_seed,
                    lsm_basis=args.lsm_basis,
                    lsm_degree=args.lsm_degree,
                    lsm_reg=args.lsm_reg,
                    lsm_reg_alpha=args.lsm_reg_alpha,
                    lsm_n_actions=args.lsm_n_actions,
                )
                write_trace_parquet(trace_df, output_path)
                summaries.append(
                    summarize_trace(
                        trace_df,
                        config=config,
                        method="LSM",
                        run_name=f"{config}_lsm",
                        seed=lsm_seed,
                        artifact_path=output_path,
                        q_max=build_contract(first_params).q_max,
                    )
                )

    if summaries:
        manifest_path = write_manifest(output_root, summaries)
        print(f"\nManifest written to {manifest_path}")
        print(pd.DataFrame([summary.as_dict() for summary in summaries]).to_string(index=False))
    else:
        print("\nNo new artifacts were written.")


if __name__ == "__main__":
    main()
