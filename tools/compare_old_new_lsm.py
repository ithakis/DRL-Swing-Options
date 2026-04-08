#!/usr/bin/env python
"""Compare old (bang-bang) vs new (discretized-action) LSM values."""
import csv, sys

old_file = "Jupyter Notebooks/Convex Costs Results 7.csv.bak"  # We'll need to save the old one first
new_file = "Jupyter Notebooks/Convex Costs Results 7.csv"

# Old LSM values from the terminal output (hardcoded from Results 7 before rebuild)
old_lsm = {
    ("0.0", "1.0"): 2.6661,
    ("0.01", "1.0"): 2.5534,
    ("0.01", "1.5"): 2.5079,
    ("0.01", "2.0"): 2.4442,
    ("0.01", "3.0"): 2.2375,
    ("0.02", "1.0"): 2.4442,
    ("0.02", "1.5"): 2.3567,
    ("0.02", "2.0"): 2.2375,
    ("0.02", "3.0"): 1.8634,
    ("0.04", "1.0"): 2.2375,
    ("0.04", "1.5"): 2.0751,
    ("0.04", "2.0"): 1.8634,
    ("0.04", "3.0"): 1.2652,
    ("0.05", "1.0"): 2.1382,
    ("0.05", "1.5"): 1.9455,
    ("0.05", "2.0"): 1.6948,
    ("0.05", "3.0"): 1.0357,
    ("0.08", "1.0"): 1.8634,
    ("0.08", "1.5"): 1.5916,
    ("0.08", "2.0"): 1.2652,
    ("0.10", "1.0"): 1.6948,
    ("0.10", "1.5"): 1.3875,
    ("0.10", "2.0"): 1.0357,
    ("0.15", "1.0"): 1.3299,
    ("0.15", "1.5"): 0.9737,
    ("0.15", "2.0"): 0.6226,
}

with open(new_file) as f:
    reader = csv.DictReader(f)
    print(f"{'Config':<40} {'Old LSM':>10} {'New LSM':>10} {'Change':>8} {'Δ%':>8} {'Status':>8}")
    print("-" * 90)
    all_ok = True
    improvements = []
    for row in reader:
        c = row["c"]
        gamma = row["gamma"]
        key = (c, gamma)
        new_val = float(row["LSM_full"])
        old_val = old_lsm.get(key, 0)
        if old_val > 0:
            change = new_val - old_val
            pct = 100 * change / old_val
            status = "OK" if change >= -0.01 else "FAIL"
            if status == "FAIL":
                all_ok = False
            improvements.append(pct)
            print(f"{row['Configuration']:<40} {old_val:10.4f} {new_val:10.4f} {change:+8.4f} {pct:+7.2f}% {status:>8}")

    print(f"\nAll LSM values ≥ old: {'YES' if all_ok else 'NO'}")
    print(f"Mean improvement: {sum(improvements)/len(improvements):+.2f}%")
    print(f"Max improvement: {max(improvements):+.2f}%")
    print(f"Min improvement: {min(improvements):+.2f}%")
