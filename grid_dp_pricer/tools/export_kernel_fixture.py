#!/usr/bin/env python
"""Export the src/transition_kernel.py mesh + reference integrals as a binary fixture
for the C++ FP64 parity test (grid_dp_pricer/tests/test_kernel_parity.cpp).

The C++ test loads the SAME float64 nodes/weights and recomputes the SAME analytic
test integrals; identical inputs + identical arithmetic => agreement to ~machine
precision (bar: <= 1e-10).  Scrambled-Sobol jump nodes are not reproduced in C++ —
they are loaded from here, mirroring cpp_pricer's kernel_v64.bin discipline.

Run from the EP11 env:
    conda run -n EP11 python grid_dp_pricer/tools/export_kernel_fixture.py
"""
import math
import os
import struct
import sys

import numpy as np

# Resolve repo root (…/DRL-Swing-Options) so `import src.…` works from the worktree.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from src.transition_kernel import KernelParams, precompute_kernel  # noqa: E402


def no_seasonal(_t: float) -> float:
    return 0.0


# Test functions g(X', Y'). MUST match tests/test_kernel_parity.cpp exactly.
def g0(Xp, Yp):  # payoff-like, K=1
    return np.maximum(np.exp(Xp + Yp) - 1.0, 0.0)


def g1(Xp, Yp):  # smooth, mildly growing
    return np.exp(0.5 * Xp) * (1.0 + Yp)


def g2(Xp, Yp):  # smooth, bounded
    return 1.0 / (1.0 + Xp * Xp + Yp)


TEST_FUNCS = [g0, g1, g2]


def main():
    # v64 focal HHK + contract dt.
    alpha, sigma, beta, lam, mu_J = 12.0, 1.2, 150.0, 6.0, 0.3
    maturity, n_rights = 0.0833, 22
    dt = maturity / (n_rights - 1)

    params = KernelParams(
        alpha=alpha, sigma=sigma, beta=beta, lam=lam, mu_J=mu_J, dt=dt,
        M_x=6, N_max=3, M_per_k=8, f_id="no_seasonal",
    )
    cache_dir = os.path.join(HERE, "..", "data", "_kernel_cache")
    k = precompute_kernel(params, n_rights, no_seasonal, maturity=maturity,
                          cache_dir=cache_dir, force_rebuild=True)

    z_X = np.asarray(k.z_X, dtype=np.float64)
    w_X = np.asarray(k.w_X, dtype=np.float64)
    delta_Y = np.asarray(k.delta_Y, dtype=np.float64)
    w_Y = np.asarray(k.w_Y, dtype=np.float64)
    decay_X, sigma_X, decay_Y = float(k.decay_X), float(k.sigma_X), float(k.decay_Y)

    x_base, y_base = 0.2, 0.5
    Xp = decay_X * x_base + sigma_X * z_X[:, None]   # (M_x, 1)
    Yp = decay_Y * y_base + delta_Y[None, :]         # (1, M_y)
    W = w_X[:, None] * w_Y[None, :]                  # (M_x, M_y)

    I_ref = []
    for g in TEST_FUNCS:
        vals = g(Xp, Yp)                              # (M_x, M_y)
        I_ref.append(float(np.sum(W * vals)))

    out_path = os.path.join(HERE, "..", "data", "kernel_fixture.bin")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", 0x47444B31))                 # magic "GDK1"
        f.write(struct.pack("<ii", z_X.size, delta_Y.size))    # M_x, M_y
        f.write(struct.pack("<ddd", decay_X, sigma_X, decay_Y))
        f.write(struct.pack("<dd", x_base, y_base))
        f.write(z_X.astype("<f8").tobytes())
        f.write(w_X.astype("<f8").tobytes())
        f.write(delta_Y.astype("<f8").tobytes())
        f.write(w_Y.astype("<f8").tobytes())
        f.write(struct.pack("<i", len(I_ref)))
        f.write(np.asarray(I_ref, dtype="<f8").tobytes())

    print(f"wrote {out_path}")
    print(f"  M_x={z_X.size} M_y={delta_Y.size}  decay_X={decay_X:.12g} "
          f"sigma_X={sigma_X:.12g} decay_Y={decay_Y:.12g}")
    print(f"  sum w_X={w_X.sum():.16g}  sum w_Y={w_Y.sum():.16g}")
    for i, v in enumerate(I_ref):
        print(f"  I_ref[g{i}] = {v:.16g}")


if __name__ == "__main__":
    main()
