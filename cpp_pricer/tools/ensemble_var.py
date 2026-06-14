#!/usr/bin/env python3
"""H-N9 ensemble variance analysis: how seed-to-seed std shrinks when the published
method is an ENSEMBLE of K independently-seeded agents (price-averaged).

For a config's per-seed prices, the K-ensemble estimator (mean of K agents) has the same
expected price but std ~ sigma_single/sqrt(K). We report both the analytic sigma/sqrt(K)
and an empirical bootstrap (random K-subsets), plus the K needed to match an LSM std bar.

Usage:
  ensemble_var.py --pat 'data/research/hn4_swa_seed{s}.json' --seeds 11-25 [--lsm_std 0.0012]
"""
import argparse, glob, json, math, random, statistics as st


def load(pat, seeds):
    out = []
    for s in seeds:
        try:
            out.append(json.load(open(pat.replace("{s}", str(s))))["price"])
        except FileNotFoundError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pat", required=True)
    ap.add_argument("--seeds", default="11-25")
    ap.add_argument("--lsm_std", type=float, default=0.0012)
    ap.add_argument("--lsm_price", type=float, default=1.9809)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-")) if "-" in a.seeds else (None, None)
    seeds = list(range(lo, hi + 1)) if lo else [int(x) for x in a.seeds.split(",")]
    p = load(a.pat, seeds)
    n = len(p)
    mean, sd = st.mean(p), st.pstdev(p)
    print(f"single-agent: n={n}  price={mean:.5f}  seed-std={sd:.5f}")
    print(f"LSM bar     : price={a.lsm_price:.5f}  seed-std={a.lsm_std:.5f}")
    print(f"{'K':>3} {'std~sig/sqrtK':>13} {'boot-std':>9} {'vs LSM':>8}")
    random.seed(7)
    for K in (1, 2, 3, 4, 5, 9, 16, 25):
        analytic = sd / math.sqrt(K)
        # bootstrap: mean of K random draws (with replacement), std over 4000 trials
        boots = [st.mean(random.choices(p, k=K)) for _ in range(4000)]
        bstd = st.pstdev(boots)
        tag = "BEATS" if analytic < a.lsm_std else ""
        print(f"{K:>3} {analytic:>13.5f} {bstd:>9.5f} {analytic/a.lsm_std:>7.2f}x {tag}")
    kmatch = (sd / a.lsm_std) ** 2
    print(f"\nK to match LSM std ({a.lsm_std}): ~{kmatch:.1f} agents")


if __name__ == "__main__":
    main()
