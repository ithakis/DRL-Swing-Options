// config.hpp — compile-time precision switch + problem/grid parameter structs.
//
// Precision: GRID_DP_FP64 selects the bulk-value / GEMM scalar type. The reference
// build is FP64 (default ON); the FP32 fast path (=OFF) is studied in the Pareto
// analysis. Structural math (parameters, grid coordinates, quadrature build) always
// uses double; only the value arrays U/W and the BLAS GEMMs use `Scalar`.
#pragma once

#include <cmath>
#include <cstddef>

namespace grid_dp {

#ifndef GRID_DP_FP64
#define GRID_DP_FP64 1
#endif

#if GRID_DP_FP64
using Scalar = double;
#else
using Scalar = float;
#endif

// Hambly–Howison–Kluge spot model:  S_t = exp(f(t) + X_t + Y_t),  f ≡ 0.
//   dX = -alpha X dt + sigma dW                          (Gaussian OU)
//   dY = -beta  Y dt + J dN,  N~Poisson(lam), J~Exp(mean mu_J)   (positive jumps)
struct HHKParams {
    double S0    = 1.0;
    double alpha = 12.0;
    double sigma = 1.2;
    double beta  = 150.0;
    double lam   = 6.0;
    double mu_J  = 0.3;
    // f ≡ 0 (no seasonal). Kept as a documented hook only.
};

// Discrete swing contract with convex exercise cost. Matches src/swing_contract.py
// + src/swing_env.py semantics (0-based discrete discounting, net-profit gate).
struct SwingContract {
    double K       = 1.0;       // strike
    double maturity = 0.0833;   // T (years)
    int    n_rights = 22;       // N decision dates j = 0..N-1
    double q_min   = 0.0;       // per-date lower lift
    double q_max   = 2.0;       // per-date upper lift
    double Q_min   = 0.0;       // global floor (hook; 0 supported)
    double Q_max   = 20.0;      // global cap (cumulative volume)
    double r       = 0.05;      // risk-free rate
    int    min_refraction = 0;  // cooldown periods (hook; 0 supported)
    double c_cost  = 0.04;      // convex cost coefficient c
    double gamma_cost = 2.0;    // convex cost exponent gamma

    // dt = maturity / (n_rights - 1):  t_j = j*dt, exactly as SwingContract.dt.
    double dt() const { return maturity / static_cast<double>(n_rights - 1); }
    // df = e^{-r dt}; per-date contribution to V_0 is df^j * pi_j (0-based exponent).
    double discount_factor() const { return std::exp(-r * dt()); }
};

// Tensor-product grid in (X, Y, Q) + analytical-kernel quadrature orders.
struct GridParams {
    // Spatial resolution (converged reference defaults; see RESULTS convergence study).
    int n_X = 121;     // nodes on X (OU log-factor); X support is tiny (std~0.18)
    int n_Y = 121;     // nodes on Y (jump log-factor, >= 0)
    int n_Q = 151;     // nodes on Q (cumulative volume in [0, Q_max])
    // Truncation ranges (set from realized HHK support; see tools/set_grid_ranges.py:
    // X in [-1.14,1.05], Y max ~2.87 — these add safety margin).
    double X_lo = -1.6, X_hi = 1.6;
    double Y_lo = 0.0,  Y_hi = 4.0;
    // Quadrature orders for the one-step expectation E[U_{j+1}].
    int M_x   = 24;    // Gauss-Hermite nodes for the Gaussian X' step
    int N_max = 3;     // Poisson jump-count truncation (tail folded into N_max)
    int gl_U  = 16;    // Gauss-Legendre nodes for arrival times U in (0, dt)
    int glag_J = 16;   // Gauss-Laguerre nodes for Exp jump sizes J
    // Interpolation in (X, Y): 0 = multilinear, 1 = not-a-knot cubic spline.
    int interp_xy = 1;
    // Threads for the inner-control sweep (0 = hardware concurrency). The GEMM stays single
    // BLAS call; only the per-(x,y) inner maximization is parallelized over X-rows.
    int n_threads = 0;
};

} // namespace grid_dp
