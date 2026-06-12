#!/usr/bin/env python
"""Merge v61 (paper) + v64-4k + v64-32k convex-cost results into one comparison CSV.

Inputs
------
- ``--v64_raw``  : the CSV emitted by ``tools/rebuild_results_v7.py`` when run over the v64
                   sweep agents (rows named ``..._v64_4k`` and/or ``..._v64_32k``).
- ``--v61_csv``  : the existing paper results ``Convex Costs Results 7.csv`` (the v61 agents
                   are gone from runs/, so their numbers are reused, never retrained).

Output (one row per (c, gamma) cell)
------------------------------------
    Configuration, c, gamma, LSM_full,
    Delta_pct_v61_old, Delta_pct_v64_4k, Delta_pct_v64_32k,
    RL_mean_v61, RL_mean_v64_4k, RL_mean_v64_32k,
    RL_std_v61,  RL_std_v64_4k,  RL_std_v64_32k

Δ% columns are sourced from each version's ``PctDiff_mean`` (= 100*(RL_mean/LSM-1)). The
merge is graceful: if the v64-raw has no 32k rows yet (phase-1, 4k only), the 32k columns
are filled with NaN. Rows are ordered by (c, gamma) ascending to match the paper table.

Usage:
    python tools/merge_v64_comparison.py \
        --v64_raw "Jupyter Notebooks/Convex Costs Results v64 raw.csv" \
        --v61_csv "Jupyter Notebooks/Convex Costs Results 7.csv" \
        --output  "Jupyter Notebooks/Convex Costs Results v64 comparison.csv"
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

DEFAULT_V64_RAW = "Jupyter Notebooks/Convex Costs Results v64 raw.csv"
DEFAULT_V61 = "Jupyter Notebooks/Convex Costs Results 7.csv"
DEFAULT_OUT = "Jupyter Notebooks/Convex Costs Results v64 comparison.csv"


def _key(df: pd.DataFrame) -> pd.DataFrame:
    """Add rounded (c, gamma) join keys to avoid float-equality mismatches."""
    df = df.copy()
    df["_c"] = df["c"].astype(float).round(4)
    df["_g"] = df["gamma"].astype(float).round(4)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v64_raw", default=DEFAULT_V64_RAW, help="Raw v64 eval CSV from rebuild_results_v7")
    ap.add_argument("--v61_csv", default=DEFAULT_V61, help="Existing paper (v61) results CSV")
    ap.add_argument("--output", default=DEFAULT_OUT, help="Output comparison CSV path")
    args = ap.parse_args()

    if not os.path.exists(args.v64_raw):
        raise SystemExit(f"ERROR: v64 raw CSV not found: {args.v64_raw}\n"
                         f"Run tools/rebuild_results_v7.py --output '{args.v64_raw}' first.")
    if not os.path.exists(args.v61_csv):
        raise SystemExit(f"ERROR: v61 CSV not found: {args.v61_csv}")

    v64 = _key(pd.read_csv(args.v64_raw))
    v61 = _key(pd.read_csv(args.v61_csv))

    # Split v64 raw rows by budget infix in the Configuration label.
    cfg = v64["Configuration"].astype(str)
    v64_4k = v64[cfg.str.contains("_v64_4k")].copy()
    v64_32k = v64[cfg.str.contains("_v64_32k")].copy()

    # Build the cell universe from whatever v64 we have (union of 4k/32k keys), fall back to v61.
    keys = pd.concat([v64_4k[["_c", "_g"]], v64_32k[["_c", "_g"]]], ignore_index=True)
    if keys.empty:
        keys = v64[["_c", "_g"]].copy()
    if keys.empty:
        keys = v61[["_c", "_g"]].copy()
    cells = keys.drop_duplicates().sort_values(["_c", "_g"]).reset_index(drop=True)

    def lookup(df: pd.DataFrame, c: float, g: float, col: str):
        hit = df[(df["_c"] == c) & (df["_g"] == g)]
        if hit.empty or col not in df.columns:
            return np.nan
        return hit.iloc[0][col]

    rows = []
    for _, row in cells.iterrows():
        c, g = row["_c"], row["_g"]
        # LSM_full: identical protocol across versions; prefer v64-4k, then v64-32k, then v61.
        lsm = lookup(v64_4k, c, g, "LSM_full")
        if pd.isna(lsm):
            lsm = lookup(v64_32k, c, g, "LSM_full")
        if pd.isna(lsm):
            lsm = lookup(v61, c, g, "LSM_full")
        rows.append(
            {
                "Configuration": f"SwingOption_20_c{c:g}_gamma{g:g}",
                "c": c,
                "gamma": g,
                "LSM_full": lsm,
                "Delta_pct_v61_old": lookup(v61, c, g, "PctDiff_mean"),
                "Delta_pct_v64_4k": lookup(v64_4k, c, g, "PctDiff_mean"),
                "Delta_pct_v64_32k": lookup(v64_32k, c, g, "PctDiff_mean"),
                "RL_mean_v61": lookup(v61, c, g, "RL_mean"),
                "RL_mean_v64_4k": lookup(v64_4k, c, g, "RL_mean"),
                "RL_mean_v64_32k": lookup(v64_32k, c, g, "RL_mean"),
                "RL_std_v61": lookup(v61, c, g, "RL_std"),
                "RL_std_v64_4k": lookup(v64_4k, c, g, "RL_std"),
                "RL_std_v64_32k": lookup(v64_32k, c, g, "RL_std"),
            }
        )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    n_4k = out["Delta_pct_v64_4k"].notna().sum()
    n_32k = out["Delta_pct_v64_32k"].notna().sum()
    print(f"Wrote {len(out)} cells -> {args.output}")
    print(f"  v61 populated:     {out['Delta_pct_v61_old'].notna().sum()}/{len(out)}")
    print(f"  v64-4k populated:  {n_4k}/{len(out)}")
    print(f"  v64-32k populated: {n_32k}/{len(out)}")
    if n_32k:
        ge0 = out[out["Delta_pct_v64_32k"] >= 0]
        print(f"  v64-32k reaches Δ% >= 0 in {len(ge0)}/{n_32k} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
