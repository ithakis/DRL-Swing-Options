#!/usr/bin/env python
"""Report realized (X, Y) support of the HHK decision-date factors, to justify the DP grid
truncation ranges.  Uses the repo's own simulator (src/simulate_hhk_spot).

    conda run -n EP11 python grid_dp_pricer/tools/set_grid_ranges.py [--paths 200000]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from src.simulate_hhk_spot import simulate_hhk_spot  # noqa: E402
from src.simulate_hhk_spot import no_seasonal_function as no_seasonal  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=12345)
    # v64 focal HHK + contract
    ap.add_argument("--S0", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=12.0)
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--beta", type=float, default=150.0)
    ap.add_argument("--lam", type=float, default=6.0)
    ap.add_argument("--mu_J", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=0.0833)
    ap.add_argument("--n_rights", type=int, default=22)
    args = ap.parse_args()

    n_steps = args.n_rights - 1
    t, S, X, Y = simulate_hhk_spot(
        S0=args.S0, T=args.T, n_steps=n_steps, n_paths=args.paths,
        alpha=args.alpha, sigma=args.sigma, beta=args.beta, lam=args.lam, mu_J=args.mu_J,
        f=no_seasonal, seed=args.seed, dtype=np.float64, stratify=False,
    )
    Xf, Yf = X.ravel(), Y.ravel()
    qs = [0, 1e-6, 1e-5, 1e-4, 1e-3, 0.5, 1 - 1e-3, 1 - 1e-4, 1 - 1e-5, 1 - 1e-6, 1.0]
    print(f"paths={args.paths}  decision-date factor support ({S.shape[0]}x{S.shape[1]} samples):")
    print("  quantile        X            Y            S=exp(X+Y)")
    Sf = np.exp(Xf + Yf)
    for q in qs:
        print(f"  {q:<10.6g} {np.quantile(Xf, q):>10.4f}  {np.quantile(Yf, q):>10.4f}  "
              f"{np.quantile(Sf, q):>10.4f}")
    print(f"\n  X in [{Xf.min():.4f}, {Xf.max():.4f}]   Y in [{Yf.min():.4f}, {Yf.max():.4f}]")
    print(f"  stationary std(X) ~ {Xf.std():.4f}   max S = {Sf.max():.3f}")
    print("\nSuggested DP ranges (cover ~ to the 1-1e-5 quantile with margin):")
    print(f"  --Xlo {np.floor(np.quantile(Xf,1e-6)*10)/10:.1f} "
          f"--Xhi {np.ceil(np.quantile(Xf,1-1e-6)*10)/10:.1f} "
          f"--Ylo 0.0 --Yhi {np.ceil(np.quantile(Yf,1-1e-6)*10)/10:.1f}")


if __name__ == "__main__":
    main()
