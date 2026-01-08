#!/usr/bin/env python3
"""
v59 vs v60 vs v61 Comprehensive Comparison Analysis

Analyzes TensorBoard logs to compare performance metrics across versions.
Separates analysis by cost regime (CC = convex costs, non-CC = no costs).
Focuses on: Delta_Percent, RL_Price, Policy metrics, and learning dynamics.
"""

import os
import numpy as np
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Configuration
RUNS_DIR = "/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/runs"

# Separate CC (convex costs) and non-CC runs
VERSIONS_CC = {
    "v59CC": ["SwingOption_20_v59CC_1_11", "SwingOption_20_v59CC_1_12", "SwingOption_20_v59CC_1_13"],
    "v60CC": ["SwingOption_20_v60CC_1_11", "SwingOption_20_v60CC_1_12", "SwingOption_20_v60CC_1_13"],
    "v61CC": ["SwingOption_20_v61CC_1_11", "SwingOption_20_v61CC_1_12", "SwingOption_20_v61CC_1_13"],
}

VERSIONS_NO_COST = {
    "v59": ["SwingOption_20_v59_1_11", "SwingOption_20_v59_1_12", "SwingOption_20_v59_1_13"],
    "v60": ["SwingOption_20_v60_1_11", "SwingOption_20_v60_1_12", "SwingOption_20_v60_1_13"],
    "v61": ["SwingOption_20_v61_1_11", "SwingOption_20_v61_1_12", "SwingOption_20_v61_1_13"],
}

# Key metrics to analyze
METRICS = [
    "Pricing/Delta_Percent",
    "Pricing/RL_Price",
    "Pricing/Average100",
    "Policy/Action_variance_mean",
    "Policy/Actions_at_upper_pct",
    "Policy/Actions_at_lower_pct",
    "Pricing/Avg_Exercise_Count",
    "Pricing/Avg_Total_Exercised",
    "Training/Critic_loss",
    "Training/Actor_loss",
]


def load_tensorboard_data(run_path: str) -> dict:
    """Load all scalar data from a TensorBoard log directory."""
    accumulator = EventAccumulator(run_path)
    accumulator.Reload()
    
    data = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {"steps": np.array(steps), "values": np.array(values)}
    return data


def extract_metric(data: dict, metric: str) -> tuple:
    """Extract steps and values for a specific metric."""
    if metric in data:
        return data[metric]["steps"], data[metric]["values"]
    return np.array([]), np.array([])


def compute_stats_at_episodes(all_values: list, all_steps: list, target_episodes: list) -> dict:
    """Compute mean, std, min, max across seeds at specific episode counts."""
    stats = {}
    for ep in target_episodes:
        values_at_ep = []
        for steps, values in zip(all_steps, all_values):
            if len(steps) == 0:
                continue
            idx = np.argmin(np.abs(steps - ep))
            if abs(steps[idx] - ep) < 500:
                values_at_ep.append(values[idx])
        
        if values_at_ep:
            stats[ep] = {
                "mean": np.mean(values_at_ep),
                "std": np.std(values_at_ep),
                "min": np.min(values_at_ep),
                "max": np.max(values_at_ep),
                "spread": np.max(values_at_ep) - np.min(values_at_ep),
                "n": len(values_at_ep),
            }
    return stats


def analyze_convergence(all_values: list, all_steps: list) -> dict:
    """Analyze convergence speed and stability."""
    results = {}
    
    for seed_idx, (steps, values) in enumerate(zip(all_steps, all_values)):
        if len(steps) == 0:
            continue
        
        above_80 = np.where(values > 80)[0]
        first_80 = steps[above_80[0]] if len(above_80) > 0 else np.nan
        
        above_90 = np.where(values > 90)[0]
        first_90 = steps[above_90[0]] if len(above_90) > 0 else np.nan
        
        above_100 = np.where(values > 100)[0]
        first_100 = steps[above_100[0]] if len(above_100) > 0 else np.nan
        
        n = len(values)
        late_values = values[int(0.75 * n):]
        late_mean = np.mean(late_values) if len(late_values) > 0 else np.nan
        late_std = np.std(late_values) if len(late_values) > 0 else np.nan
        
        early_values = values[:int(0.25 * n)]
        early_std = np.std(early_values) if len(early_values) > 0 else np.nan
        
        results[f"seed_{seed_idx}"] = {
            "first_80_pct": first_80,
            "first_90_pct": first_90,
            "first_100_pct": first_100,
            "late_mean": late_mean,
            "late_std": late_std,
            "early_std": early_std,
            "final_value": values[-1] if len(values) > 0 else np.nan,
        }
    
    return results


def analyze_version_set(versions: dict, regime_name: str):
    """Analyze a set of versions (CC or non-CC)."""
    print("\n" + "=" * 80)
    print(f"  {regime_name} REGIME ANALYSIS")
    print("=" * 80)
    
    # Load all data
    version_data = {}
    for version, runs in versions.items():
        version_data[version] = []
        for run_name in runs:
            run_path = os.path.join(RUNS_DIR, run_name)
            if os.path.exists(run_path):
                data = load_tensorboard_data(run_path)
                version_data[version].append({"name": run_name, "data": data})
                print(f"Loaded {run_name}: {len(data)} metrics")
            else:
                print(f"WARNING: {run_path} not found")
    
    target_episodes = [1024, 2048, 3072, 4096, 8192, 16384, 24576, 32768]
    
    # Delta Percent Analysis
    print("\n" + "-" * 80)
    print("DELTA PERCENT ANALYSIS")
    print("-" * 80)
    
    for version in versions.keys():
        print(f"\n### {version} ###")
        
        all_steps, all_values = [], []
        for run in version_data[version]:
            steps, values = extract_metric(run["data"], "Pricing/Delta_Percent")
            all_steps.append(steps)
            all_values.append(values)
            if len(values) > 0:
                print(f"  {run['name']}: final={values[-1]:.2f}%, max={np.max(values):.2f}%")
        
        stats = compute_stats_at_episodes(all_values, all_steps, target_episodes)
        print(f"\n  Episode-wise (Mean ± Std [spread]):")
        for ep in [2048, 4096, 8192, 16384, 32768]:
            if ep in stats:
                s = stats[ep]
                print(f"    Ep {ep:>5}: {s['mean']:>6.2f}% ± {s['std']:>5.2f}% [{s['spread']:>5.2f}%]")
    
    # Convergence Speed Comparison
    print("\n" + "-" * 80)
    print("CONVERGENCE SPEED (First episode to reach threshold)")
    print("-" * 80)
    
    for version in versions.keys():
        print(f"\n### {version} ###")
        all_steps, all_values = [], []
        for run in version_data[version]:
            steps, values = extract_metric(run["data"], "Pricing/Delta_Percent")
            all_steps.append(steps)
            all_values.append(values)
        
        conv = analyze_convergence(all_values, all_steps)
        first_80_list = [v["first_80_pct"] for v in conv.values() if not np.isnan(v["first_80_pct"])]
        first_100_list = [v["first_100_pct"] for v in conv.values() if not np.isnan(v["first_100_pct"])]
        
        if first_80_list:
            print(f"  First 80%: min={min(first_80_list):.0f}, max={max(first_80_list):.0f}, mean={np.mean(first_80_list):.0f}")
        if first_100_list:
            print(f"  First 100%: min={min(first_100_list):.0f}, max={max(first_100_list):.0f}, mean={np.mean(first_100_list):.0f}")
    
    # RL Price Analysis
    print("\n" + "-" * 80)
    print("RL PRICE ANALYSIS")
    print("-" * 80)
    
    for version in versions.keys():
        print(f"\n### {version} ###")
        all_steps, all_values = [], []
        for run in version_data[version]:
            steps, values = extract_metric(run["data"], "Pricing/RL_Price")
            all_steps.append(steps)
            all_values.append(values)
            if len(values) > 0:
                print(f"  {run['name']}: final={values[-1]:.6f}")
        
        stats = compute_stats_at_episodes(all_values, all_steps, [8192, 16384, 32768])
        if stats:
            print(f"  Episode-wise (Mean ± Std):")
            for ep, s in stats.items():
                print(f"    Ep {ep:>5}: {s['mean']:.6f} ± {s['std']:.6f}")
    
    # Policy Metrics
    print("\n" + "-" * 80)
    print("POLICY ANALYSIS (Action Variance)")
    print("-" * 80)
    
    for version in versions.keys():
        print(f"\n### {version} ###")
        all_steps, all_values = [], []
        for run in version_data[version]:
            steps, values = extract_metric(run["data"], "Policy/Action_variance_mean")
            all_steps.append(steps)
            all_values.append(values)
        
        if all_values and len(all_values[0]) > 0:
            stats = compute_stats_at_episodes(all_values, all_steps, [2048, 8192, 16384, 32768])
            print(f"  Action_variance_mean (Mean ± Std):")
            for ep, s in stats.items():
                print(f"    Ep {ep:>5}: {s['mean']:.4f} ± {s['std']:.4f}")
    
    # Early Training Dynamics
    print("\n" + "-" * 80)
    print("EARLY TRAINING (Avg_Total_Exercised)")
    print("-" * 80)
    
    early_eps = [1024, 2048, 3072, 4096]
    for version in versions.keys():
        print(f"\n### {version} ###")
        all_steps, all_values = [], []
        for run in version_data[version]:
            steps, values = extract_metric(run["data"], "Pricing/Avg_Total_Exercised")
            all_steps.append(steps)
            all_values.append(values)
        
        if all_values and len(all_values[0]) > 0:
            stats = compute_stats_at_episodes(all_values, all_steps, early_eps)
            for ep, s in stats.items():
                print(f"    Ep {ep:>4}: mean={s['mean']:>6.2f}, std={s['std']:>5.2f}, spread={s['spread']:>5.2f}")
    
    # Summary Table
    print("\n" + "-" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("-" * 80)
    
    print(f"\n{'Version':<8} | {'Final%':>8} | {'Std':>6} | {'Spread':>7} | {'1st80':>6} | {'Late Std':>8}")
    print("-" * 60)
    
    for version in versions.keys():
        all_steps, all_values = [], []
        for run in version_data[version]:
            steps, values = extract_metric(run["data"], "Pricing/Delta_Percent")
            all_steps.append(steps)
            all_values.append(values)
        
        stats = compute_stats_at_episodes(all_values, all_steps, [32768])
        conv = analyze_convergence(all_values, all_steps)
        
        final_mean = stats[32768]["mean"] if 32768 in stats else np.nan
        final_std = stats[32768]["std"] if 32768 in stats else np.nan
        final_spread = stats[32768]["spread"] if 32768 in stats else np.nan
        
        first_80_list = [v["first_80_pct"] for v in conv.values() if not np.isnan(v["first_80_pct"])]
        first_80_min = min(first_80_list) if first_80_list else np.nan
        
        late_std_list = [v["late_std"] for v in conv.values() if not np.isnan(v["late_std"])]
        late_std_mean = np.mean(late_std_list) if late_std_list else np.nan
        
        print(f"{version:<8} | {final_mean:>7.2f}% | {final_std:>5.2f}% | {final_spread:>6.2f}% | {first_80_min:>6.0f} | {late_std_mean:>7.2f}%")
    
    return version_data


def main():
    print("=" * 80)
    print("v59 vs v60 vs v61 COMPREHENSIVE COMPARISON")
    print("=" * 80)
    
    # Analyze Convex Costs (CC) regime
    cc_data = analyze_version_set(VERSIONS_CC, "CONVEX COSTS (CC)")
    
    # Analyze No-Cost regime
    nc_data = analyze_version_set(VERSIONS_NO_COST, "NO COSTS")
    
    print("\n" + "=" * 80)
    print("CROSS-REGIME OBSERVATIONS")
    print("=" * 80)
    print("""
Key observations:
1. CC regime has different optimal policy (partial exercise is optimal)
2. Non-CC regime should converge to bang-bang policy (0 or max)
3. Profitability gate behavior differs between regimes
""")


if __name__ == "__main__":
    main()
