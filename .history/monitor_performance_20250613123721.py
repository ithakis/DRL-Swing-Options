#!/usr/bin/env python3
"""
Performance monitoring script for D4PG training runs.
Compares different configurations and provides recommendations.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_stats(run_name):
    """Load training statistics from json file."""
    json_path = f"runs/{run_name}.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None

def compare_runs():
    """Compare different training runs and analyze performance."""
    runs_to_compare = [
        ("1_128_w1", "Original w=1, bs=128"),
        ("optimized_fixed_bs", "Optimized w=1, bs=128, fixed scaling"),
        ("optimized_fixed_bs_w4", "Optimized w=4, bs=128, fixed scaling"),
    ]
    
    print("=== D4PG Performance Comparison ===\n")
    
    results = {}
    for run_name, description in runs_to_compare:
        stats = load_training_stats(run_name)
        if stats:
            results[run_name] = {
                'description': description,
                'stats': stats
            }
            print(f"✓ {description}: Loaded successfully")
        else:
            print(f"✗ {description}: No data found")
    
    if len(results) >= 2:
        analyze_performance(results)
    else:
        print("\nNeed at least 2 runs to compare. Continue training...")

def analyze_performance(results):
    """Analyze and compare performance metrics."""
    print("\n=== Performance Analysis ===")
    
    for run_name, data in results.items():
        stats = data['stats']
        desc = data['description']
        
        # Extract key metrics (these would depend on your logging format)
        print(f"\n{desc}:")
        print(f"  - Total training time: {stats.get('training_time', 'N/A')}")
        print(f"  - Final average score: {stats.get('final_score', 'N/A')}")
        print(f"  - Convergence speed: {stats.get('convergence_frame', 'N/A')} frames")

def plot_training_curves():
    """Plot training curves for comparison (requires tensorboard logs)."""
    print("\n=== Training Curves ===")
    print("To view detailed training curves, run:")
    print("tensorboard --logdir=runs/")
    print("Then open http://localhost:6006 in your browser")

def recommendations():
    """Provide optimization recommendations."""
    print("\n=== Optimization Recommendations ===")
    print("1. ✓ Fixed batch size scaling issue")
    print("2. ✓ Added worker-aware learning frequency")
    print("3. ✓ Optimized network size (256 layers)")
    print("4. ✓ Set proper tau value (1e-3)")
    print("\nAdditional improvements to consider:")
    print("- Try different learning rates: 1e-4, 3e-4, 1e-3")
    print("- Experiment with different tau values: 1e-3, 5e-3, 1e-2")
    print("- Test different network architectures (D2RL=1)")
    print("- Consider different environments for validation")

if __name__ == "__main__":
    compare_runs()
    plot_training_curves()
    recommendations()
