#!/usr/bin/env python3
"""Convert a price_swing --trace binary blob into the per-(path,step) parquet that the
analysis notebooks (NB6) consume.

Blob layout (little-endian): int32 magic('SWTR'=0x53575452), int32 n_paths, int32 T,
then 4 float32[n_paths*T] arrays in order q, reward, cost(undiscounted), gross(undiscounted).

Output parquet columns: path, time_step, q_t, reward, exercise_cost, payoff, payoff_gross.
(`payoff` == `payoff_gross`; NB6 reads payoff_gross for the gross PV and reward for the net.)
"""
import argparse
import struct

import numpy as np
import pandas as pd


def read_trace(path):
    with open(path, "rb") as f:
        magic, n, T = struct.unpack("<iii", f.read(12))
        if magic != 0x53575452:
            raise ValueError(f"bad magic {magic:#x} in {path!r}")
        buf = np.fromfile(f, dtype="<f4")
    sz = n * T
    if buf.size != 4 * sz:
        raise ValueError(f"expected {4*sz} floats, got {buf.size}")
    q, reward, cost, gross = (buf[i * sz:(i + 1) * sz].reshape(n, T) for i in range(4))
    return n, T, q, reward, cost, gross


def to_dataframe(path):
    n, T, q, reward, cost, gross = read_trace(path)
    paths = np.repeat(np.arange(n, dtype=np.int32), T)
    steps = np.tile(np.arange(T, dtype=np.int32), n)
    return pd.DataFrame({
        "path": paths,
        "time_step": steps,
        "q_t": q.reshape(-1),
        "reward": reward.reshape(-1),
        "exercise_cost": cost.reshape(-1),
        "payoff": gross.reshape(-1),
        "payoff_gross": gross.reshape(-1),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("out_parquet")
    ap.add_argument("--validate", action="store_true",
                    help="print mean per-path summed reward (should equal the run price)")
    a = ap.parse_args()
    df = to_dataframe(a.blob)
    if a.validate:
        price = df.groupby("path")["reward"].sum().mean()
        print(f"[validate] paths={df['path'].nunique()} steps={df['time_step'].max()+1} "
              f"mean(sum reward)={price:.6f}")
    df.to_parquet(a.out_parquet, index=False)
    print(f"wrote {a.out_parquet} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
