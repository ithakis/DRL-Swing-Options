#!/usr/bin/env python
"""
Rebuild the convex-cost results table using a statistically correct evaluation protocol.

Protocol:
  1. ONE common test set (seed=999, 65536 paths) that no model has ever seen.
  2. Every saved RL model (final-episode actor weights) evaluated on this test set.
  3. LSM baseline fitted on a separate training set (seed=998) and evaluated on the same test set.
  4. Cross-seed mean ± std reported per configuration.

Usage:
  python tools/rebuild_results_v7.py              # full grid
  python tools/rebuild_results_v7.py --limit 1    # one config for debugging
  python tools/rebuild_results_v7.py --focal_config "SwingOption_20_c0.04_gamma2" \
         --focal_seeds "11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import Agent
from src.agent_evaluation import _evaluate_swing_batch
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos
from src.simulate_hhk_spot import no_seasonal_function, simulate_hhk_spot
from src.swing_contract import SwingContract

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_SEED = 999  # Common test set seed — never used in training or validation
LSM_TRAIN_SEED = 998  # Separate LSM training set seed
TEST_N_PATHS = 65536  # Number of test paths
LSM_TRAIN_N_PATHS = 65536  # LSM training set size (tuned for best performance)
LSM_N_ACTIONS = 5  # Discretized action grid: {0, q_max/4, q_max/2, 3q_max/4, q_max}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class dotdict(dict):
    """Attribute-accessible dict (matches evaluate_saved_agent.py pattern)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def parse_config_key(name: str) -> Tuple[float, float]:
    """Extract (c, gamma) from a run name like 'SwingOption_20_c0.04_gamma2_11'."""
    # Strip the seed suffix
    parts = name.rsplit("_", 1)
    base = parts[0]  # e.g. SwingOption_20_c0.04_gamma2
    # Extract c and gamma
    c_part = base.split("_c")[1]  # e.g. 0.04_gamma2
    c_str, gamma_part = c_part.split("_gamma")
    return float(c_str), float(gamma_part)


def config_base_name(name: str) -> str:
    """Get the config base (without seed) from a run name."""
    return name.rsplit("_", 1)[0]


def build_contract(params: dotdict) -> SwingContract:
    """Reconstruct SwingContract from JSON parameters."""
    return SwingContract(
        q_min=params.q_min,
        q_max=params.q_max,
        Q_min=params.Q_min,
        Q_max=params.Q_max,
        strike=params.strike,
        maturity=params.maturity,
        n_rights=params.n_rights,
        r=params.risk_free_rate,
        min_refraction_periods=params.min_refraction_periods,
        c_cost=params.c_cost,
        gamma_cost=params.gamma_cost,
    )


def build_hhk_params(params: dotdict) -> Dict[str, Any]:
    """Build HHK simulation parameters from JSON config."""
    contract = build_contract(params)
    return {
        "S0": params.S0,
        "T": contract.maturity,
        "n_steps": contract.n_rights - 1,
        "alpha": params.alpha,
        "sigma": params.sigma,
        "beta": params.beta,
        "lam": params.lam,
        "mu_J": params.mu_J,
        "f": no_seasonal_function,
        "dtype": np.float32,
    }


def build_agent(params: dotdict) -> Agent:
    """Reconstruct Agent from JSON parameters (matches evaluate_saved_agent.py)."""
    state_size = 9
    action_size = 1
    return Agent(
        state_size=state_size,
        action_size=action_size,
        n_step=params.nstep,
        per=params.per,
        munchausen=params.munchausen,
        distributional=params.iqn,
        random_seed=params.seed,
        hidden_size=params.layer_size,
        BATCH_SIZE=params.batch_size,
        BUFFER_SIZE=getattr(params, "max_replay_size", 200000),
        GAMMA=params.gamma,
        LR_ACTOR=params.lr_a,
        LR_CRITIC=params.lr_c,
        t=getattr(params, "t", getattr(params, "tau", 0.002)),
        LEARN_EVERY=params.learn_every,
        LEARN_NUMBER=params.learn_number,
        noise_sigma0=getattr(params, "noise_sigma0", getattr(params, "pre_noise_sigma0", 1.0)),
        noise_floor=getattr(params, "noise_floor", getattr(params, "pre_noise_floor", 0.05)),
        noise_plateau=getattr(params, "noise_plateau", getattr(params, "pre_noise_plateau", 0)),
        device="cpu",
        paths=0,
        min_replay_size=getattr(params, "min_replay_size", params.batch_size * 10),
        per_priority_floor=getattr(params, "per_priority_floor", 1e-6),
        per_priority_clip_pct=getattr(params, "per_priority_clip_pct", 99.5),
        critic_ema_decay=getattr(params, "critic_ema_decay", 0.0),
        per_alpha_final=getattr(params, "per_alpha_final", None),
        per_alpha_ramp_start=getattr(params, "per_alpha_ramp_start", 0),
        per_alpha_ramp_end=getattr(params, "per_alpha_ramp_end", 0),
        per_beta_final=getattr(params, "per_beta_final", None),
        per_alpha_sigmoid=bool(getattr(params, "per_alpha_sigmoid", 0)),
        activation=getattr(params, "activation", "silu"),
        action_output=getattr(params, "actor_output_activation", "tanh01"),
        use_robust_normalization=bool(getattr(params, "use_robust_normalization", 0)),
        strike=getattr(params, "strike", 100.0),
    )


def evaluate_rl_on_test_set(
    agent: Agent,
    contract: SwingContract,
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    eval_batch_size: int = 4096,
) -> Dict[str, float]:
    """Evaluate a single RL agent on the common test set. Returns price stats."""
    n_paths = dataset[1].shape[0]

    all_returns = []
    total_bangbang = 0
    total_full_cap_exercises = 0

    # Process in batches to limit memory (path_data for 65k paths is large)
    for start in range(0, n_paths, eval_batch_size):
        end = min(start + eval_batch_size, n_paths)
        batch_indices = list(range(start, end))

        returns, _, path_data = _evaluate_swing_batch(
            agent=agent,
            contract=contract,
            dataset=dataset,
            batch_indices=batch_indices,
            collect_path_data=True,
        )
        all_returns.extend(returns)

        # Accumulate bang-bangness stats from path data
        if path_data:
            data = np.array(path_data)
            q_actual = data[:, 11]
            q_remaining_frac = data[:, 4]
            q_remaining = q_remaining_frac * contract.Q_max

            exercised_mask = q_actual > 1e-6
            full_cap_mask = exercised_mask & (q_remaining >= contract.q_max - 1e-6)
            total_full_cap_exercises += int(np.sum(full_cap_mask))
            total_bangbang += int(np.sum(q_actual[full_cap_mask] >= 0.95 * contract.q_max))

    returns_arr = np.array(all_returns)
    price = float(np.mean(returns_arr))
    std = float(np.std(returns_arr, ddof=1))
    ci95 = 1.96 * std / np.sqrt(n_paths)

    bangbangness = (
        float(total_bangbang / total_full_cap_exercises) if total_full_cap_exercises > 0 else float("nan")
    )

    return {
        "test_price": price,
        "test_std": std,
        "test_CI95": ci95,
        "bangbangness": bangbangness,
    }


def evaluate_lsm_on_test_set(
    contract: SwingContract,
    hhk_params: Dict[str, Any],
    lsm_basis: str = "chebyshev",
    lsm_degree: int = 2,
    lsm_reg: str = "none",
    lsm_reg_alpha: float = 1e-6,
    n_actions: int = LSM_N_ACTIONS,
    test_dataset: Optional[Tuple] = None,
    parquet_dir: Optional[str] = None,
    config_name: Optional[str] = None,
) -> Dict[str, float]:
    """Fit LSM on separate training set and evaluate on common test set."""
    # Generate LSM training set (seed=998)
    lsm_train_ds = simulate_hhk_spot(
        **hhk_params,
        n_paths=LSM_TRAIN_N_PATHS,
        seed=LSM_TRAIN_SEED,
        stratify=True,
        batch_size=128,
    )
    lsm_train_ds_f64 = tuple(np.asarray(arr, dtype=np.float64) for arr in lsm_train_ds)

    # Fit estimators on LSM training set with full state
    estimators = fit_lsm_estimators(
        contract=contract,
        dataset=lsm_train_ds_f64,
        poly_degree=lsm_degree,
        basis_type=lsm_basis,
        state_mode="full",
        reg_type=lsm_reg,
        reg_alpha=lsm_reg_alpha,
        n_actions=n_actions,
    )

    # Evaluate on common test set
    csv_path = None
    if parquet_dir and config_name:
        os.makedirs(parquet_dir, exist_ok=True)
        csv_path = os.path.join(parquet_dir, f"{config_name}_lsm.parquet")

    test_ds_f64 = tuple(np.asarray(arr, dtype=np.float64) for arr in test_dataset)
    lsm_price, (ci_low, ci_high) = price_swing_option_lsm_oos(
        contract=contract,
        dataset=test_ds_f64,
        estimators=estimators,
        seed=TEST_SEED + 1,
        csv_path=csv_path,
        _print_results=False,
    )

    ci95 = (ci_high - ci_low) / 2.0

    # Compute actual LSM BangBangness from exercise decisions
    if csv_path and os.path.exists(csv_path):
        import pandas as pd

        df = pd.read_parquet(csv_path)
        exercised = df[df["q_t"] > 1e-6]
        qmax = contract.q_max
        if len(exercised) > 0:
            bangbang = float((exercised["q_t"] >= 0.95 * qmax).sum() / len(exercised))
        else:
            bangbang = float("nan")
    elif n_actions <= 2:
        bangbang = 1.0  # bang-bang by construction
    else:
        bangbang = float("nan")  # unknown without parquet

    return {
        "lsm_price": lsm_price,
        "lsm_CI95": ci95,
        "lsm_bangbangness": bangbang,
    }


def discover_runs(runs_dir: str = "runs") -> Dict[str, List[Tuple[int, str, str]]]:
    """
    Discover all SwingOption runs, grouped by config base name.
    Returns: { config_base: [(seed, json_path, pth_path), ...] }
    """
    json_files = sorted(glob.glob(os.path.join(runs_dir, "SwingOption_20_c*.json")))
    grouped: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)

    for jf in json_files:
        name = os.path.basename(jf).replace(".json", "")
        pth = jf.replace(".json", ".pth")
        if not os.path.exists(pth):
            print(f"  WARNING: Missing .pth for {name}, skipping")
            continue
        seed = int(name.rsplit("_", 1)[1])
        base = config_base_name(name)
        grouped[base].append((seed, jf, pth))

    return dict(grouped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Rebuild results with common-test-set protocol")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N configs (0 = all)")
    parser.add_argument(
        "--focal_config", type=str, default=None, help="Config base name for focal robustness study"
    )
    parser.add_argument("--focal_seeds", type=str, default=None, help="Comma-separated seeds for focal study")
    parser.add_argument(
        "--output", type=str, default="Jupyter Notebooks/Convex Costs Results 7.csv", help="Output CSV path"
    )
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory with .pth/.json files")
    args = parser.parse_args()

    print("=" * 70)
    print("REBUILD RESULTS v7 — Common Test Set Protocol")
    print(f"  Test seed: {TEST_SEED}  |  Test paths: {TEST_N_PATHS}")
    print(f"  LSM train seed: {LSM_TRAIN_SEED}  |  LSM train paths: {LSM_TRAIN_N_PATHS}")
    print("=" * 70)

    # Discover runs
    all_runs = discover_runs(args.runs_dir)
    configs = sorted(all_runs.keys(), key=lambda k: parse_config_key(k + "_0"))

    if args.focal_config and args.focal_seeds:
        # Focal mode: only evaluate the specified config with specified seeds
        focal_seeds = [int(s) for s in args.focal_seeds.split(",")]
        if args.focal_config not in all_runs:
            print(f"ERROR: focal config '{args.focal_config}' not found in runs/")
            sys.exit(1)
        available = {seed: (jf, pth) for seed, jf, pth in all_runs[args.focal_config]}
        missing = [s for s in focal_seeds if s not in available]
        if missing:
            print(f"WARNING: Missing seeds for {args.focal_config}: {missing}")
        configs = [args.focal_config]
    elif args.limit > 0:
        configs = configs[: args.limit]

    print(f"\nProcessing {len(configs)} configuration(s)...\n")

    # We'll generate the test set once (all configs share the same HHK params)
    # Load first config to get HHK params (they're identical across all configs)
    first_run = all_runs[configs[0]]
    first_json = first_run[0][1]
    with open(first_json, "r") as f:
        first_params = dotdict(json.load(f))

    # Build HHK params and generate common test set ONCE
    hhk_params = build_hhk_params(first_params)
    print(f"Generating common test set (seed={TEST_SEED}, {TEST_N_PATHS} paths)...")
    t0 = time.time()
    test_dataset = simulate_hhk_spot(
        **hhk_params, n_paths=TEST_N_PATHS, seed=TEST_SEED, stratify=True, batch_size=128
    )
    print(f"  Test set generated in {time.time() - t0:.1f}s  (S shape: {test_dataset[1].shape})")

    # Results storage
    results = []

    for ci, cfg in enumerate(configs):
        runs = all_runs[cfg]
        c_cost, gamma_cost = parse_config_key(cfg + "_0")
        print(f"\n[{ci + 1}/{len(configs)}] {cfg}  (c={c_cost}, γ={gamma_cost}, {len(runs)} seed(s))")

        # Load any run's JSON for contract/LSM params (they're identical across seeds)
        with open(runs[0][1], "r") as f:
            params = dotdict(json.load(f))

        contract = build_contract(params)

        # --- LSM Evaluation (once per config) ---
        print(
            f"  Fitting & evaluating LSM (state_mode=full, degree=2, basis=chebyshev, n_actions={LSM_N_ACTIONS})..."
        )
        t0 = time.time()
        lsm_result = evaluate_lsm_on_test_set(
            contract=contract,
            hhk_params=hhk_params,
            lsm_basis="chebyshev",
            lsm_degree=2,
            lsm_reg="none",
            lsm_reg_alpha=1e-6,
            n_actions=LSM_N_ACTIONS,
            test_dataset=test_dataset,
            parquet_dir="logs/lsm_full_state",
            config_name=cfg,
        )
        print(
            f"  LSM price: {lsm_result['lsm_price']:.4f} ± {lsm_result['lsm_CI95']:.4f}  ({time.time() - t0:.1f}s)"
        )

        # --- RL Evaluation (per seed) ---
        seed_results = {}

        # In focal mode, filter to requested seeds
        if args.focal_config and args.focal_seeds:
            focal_seeds_set = set(int(s) for s in args.focal_seeds.split(","))
            runs_to_eval = [(s, j, p) for s, j, p in runs if s in focal_seeds_set]
        else:
            runs_to_eval = runs

        for seed, json_path, pth_path in sorted(runs_to_eval):
            with open(json_path, "r") as f:
                seed_params = dotdict(json.load(f))

            print(f"  Evaluating seed {seed}...", end=" ", flush=True)
            t0 = time.time()

            agent = build_agent(seed_params)
            agent.actor_local.load_state_dict(torch.load(pth_path, map_location="cpu"))
            agent.actor_local.eval()

            rl_result = evaluate_rl_on_test_set(agent, contract, test_dataset)
            seed_results[seed] = rl_result
            print(
                f"price={rl_result['test_price']:.4f}, BB={rl_result['bangbangness']:.3f}  ({time.time() - t0:.1f}s)"
            )

            # Free memory
            del agent

        # --- Aggregate ---
        seed_prices = {s: r["test_price"] for s, r in seed_results.items()}
        seed_bb = {s: r["bangbangness"] for s, r in seed_results.items()}
        prices_arr = np.array(list(seed_prices.values()))
        bb_arr = np.array([v for v in seed_bb.values() if not np.isnan(v)])

        rl_mean = float(np.mean(prices_arr))
        rl_std = float(np.std(prices_arr, ddof=1)) if len(prices_arr) > 1 else 0.0
        rl_best = float(np.max(prices_arr))
        rl_best_seed = max(seed_prices, key=seed_prices.get)
        rl_bb_mean = float(np.mean(bb_arr)) if len(bb_arr) > 0 else float("nan")

        lsm_price = lsm_result["lsm_price"]
        pct_diff_mean = 100.0 * (rl_mean - lsm_price) / lsm_price if lsm_price != 0 else float("nan")
        pct_diff_best = 100.0 * (rl_best - lsm_price) / lsm_price if lsm_price != 0 else float("nan")

        # CI for PctDiff: propagate the pathwise CI
        # Use cross-seed std as uncertainty measure
        pct_diff_ci95 = (
            100.0 * (1.96 * rl_std / np.sqrt(len(prices_arr))) / lsm_price
            if (lsm_price != 0 and len(prices_arr) > 1)
            else float("nan")
        )

        row = {
            "Configuration": cfg,
            "c": c_cost,
            "gamma": gamma_cost,
            "LSM_full": lsm_price,
            "LSM_full_CI95": lsm_result["lsm_CI95"],
        }

        # Per-seed columns
        for seed in sorted(seed_prices.keys()):
            row[f"RL_seed{seed}"] = seed_prices[seed]

        row.update(
            {
                "RL_mean": rl_mean,
                "RL_std": rl_std,
                "RL_best": rl_best,
                "RL_best_seed": int(rl_best_seed),
                "PctDiff_mean": pct_diff_mean,
                "PctDiff_best": pct_diff_best,
                "PctDiff_CI95": pct_diff_ci95,
                "RL_BangBangness_mean": rl_bb_mean,
                "LSM_BangBangness": lsm_result["lsm_bangbangness"],
            }
        )

        results.append(row)

        print(f"  => RL mean={rl_mean:.4f} ± {rl_std:.4f}  |  Δ%={pct_diff_mean:+.2f}%  |  LSM={lsm_price:.4f}")

    # --- Write CSV ---
    import csv

    # Build column order
    # Find all RL_seed columns
    all_seed_cols = sorted({k for row in results for k in row if k.startswith("RL_seed")})
    columns = [
        "Configuration",
        "c",
        "gamma",
        "LSM_full",
        "LSM_full_CI95",
        *all_seed_cols,
        "RL_mean",
        "RL_std",
        "RL_best",
        "RL_best_seed",
        "PctDiff_mean",
        "PctDiff_best",
        "PctDiff_CI95",
        "RL_BangBangness_mean",
        "LSM_BangBangness",
    ]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 70}")

    # --- Print summary table ---
    print(f"\n{'Configuration':<35} {'LSM':>8} {'RL_mean':>8} {'RL_std':>7} {'Δ%':>7} {'BB':>5}")
    print("-" * 75)
    for row in results:
        print(
            f"{row['Configuration']:<35} "
            f"{row['LSM_full']:8.4f} "
            f"{row['RL_mean']:8.4f} "
            f"{row['RL_std']:7.4f} "
            f"{row['PctDiff_mean']:+7.2f}% "
            f"{row['RL_BangBangness_mean']:5.3f}"
        )


if __name__ == "__main__":
    main()
