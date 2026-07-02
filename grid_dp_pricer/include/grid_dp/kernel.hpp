// kernel.hpp — analytical one-step HHK transition kernel as a quadrature mesh.
//
// The transition (X,Y) -> (X',Y') is EXACT in time (no Euler discretization):
//   X' = decay_X * X + sigma_X * Z,   Z ~ N(0,1)          (Gaussian OU)
//   Y' = decay_Y * Y + Delta,         Delta ⟂ Y           (jump increment)
// where Delta = sum_{i=1}^{k} J_i * exp(-beta (dt - U_i)),  k ~ Poisson(lam*dt),
//       U_i ~ Unif(0,dt),  J_i ~ Exp(mean mu_J).
//
// We represent E[g(X',Y')] = sum_m sum_n w_X[m] w_Y[n] g(decay_X x + sigma_X z_X[m],
// decay_Y y + delta_Y[n]).  The X' nodes are Gauss–Hermite (spectral, exact for the
// Gaussian); the Delta law is built by tensoring Gauss-Legendre (arrival times U) with
// Gauss-Laguerre (Exp marks J) per jump count k=0..N_max, the Poisson tail folded into
// N_max (mirrors src/transition_kernel.py, but deterministic — no scrambled Sobol).
#pragma once

#include "grid_dp/config.hpp"
#include "grid_dp/gauss.hpp"
#include "grid_dp/spline.hpp"

#include <cmath>
#include <vector>

namespace grid_dp {

struct TransitionKernel {
    double decay_X = 0.0, sigma_X = 0.0, decay_Y = 0.0;
    std::vector<double> z_X, w_X;       // Gaussian step, E_{Z~N(0,1)} (sum w_X = 1)
    std::vector<double> delta_Y, w_Y;   // jump-increment law (sum w_Y = 1)

    int M_x() const { return static_cast<int>(z_X.size()); }
    int M_y() const { return static_cast<int>(delta_Y.size()); }
};

// Single-jump increment law xi = mu_J * E * exp(-beta (dt - U)),  E~Exp(1), U~Unif(0,dt),
// returned as (node, prob) pairs over an (mU x mJ) Gauss-Legendre x Gauss-Laguerre tensor.
inline void single_jump_mesh(double beta, double mu_J, double dt, int mU, int mJ,
                             std::vector<double>& s, std::vector<double>& p) {
    const Quadrature gU = gauss_legendre_unit(mU);   // U/dt ~ Unif(0,1)
    const Quadrature gJ = gauss_laguerre_exp(mJ);    // E ~ Exp(1)
    s.clear();
    p.clear();
    s.reserve(static_cast<size_t>(mU) * mJ);
    p.reserve(static_cast<size_t>(mU) * mJ);
    for (int iu = 0; iu < mU; ++iu) {
        const double U = dt * gU.x[iu];
        const double decay = std::exp(-beta * (dt - U));
        for (int ij = 0; ij < mJ; ++ij) {
            s.push_back(mu_J * gJ.x[ij] * decay);
            p.push_back(gU.w[iu] * gJ.w[ij]);
        }
    }
}

// Build the full transition kernel mesh for a given HHK config, step dt, grid orders.
inline TransitionKernel build_kernel(const HHKParams& h, double dt, const GridParams& g) {
    TransitionKernel k;
    k.decay_X = std::exp(-h.alpha * dt);
    const double varX = h.sigma * h.sigma * (1.0 - k.decay_X * k.decay_X) / (2.0 * h.alpha);
    k.sigma_X = std::sqrt(varX > 0.0 ? varX : 0.0);
    k.decay_Y = std::exp(-h.beta * dt);

    // --- X' Gaussian nodes (Gauss–Hermite, N(0,1)) ---
    const Quadrature gh = gauss_hermite_prob(g.M_x);
    k.z_X = gh.x;
    k.w_X = gh.w;

    // --- Poisson pmf over jump count k=0..N_max, tail folded into N_max ---
    const int N_max = g.N_max;
    const double rate = h.lam * dt;
    std::vector<double> pmf(N_max + 1);
    pmf[0] = std::exp(-rate);
    for (int kk = 1; kk <= N_max; ++kk) pmf[kk] = pmf[kk - 1] * rate / kk;
    double psum = 0.0;
    for (double v : pmf) psum += v;
    pmf[N_max] += 1.0 - psum;  // fold the truncated Poisson tail into the last bin

    // --- jump-increment mesh: k=0 atom + tensor-of-single-jumps for k=1..N_max ---
    k.delta_Y.assign(1, 0.0);
    k.w_Y.assign(1, pmf[0]);

    constexpr int CAP = 4096;  // cap on tensor nodes per jump count (k>=2)
    for (int kk = 1; kk <= N_max; ++kk) {
        if (pmf[kk] <= 0.0) continue;
        int mU = g.gl_U, mJ = g.glag_J;
        if (kk >= 2) {
            // shrink per-jump order so total tensor (mU*mJ)^kk stays <= CAP
            const int m1_target = static_cast<int>(std::floor(std::pow(
                static_cast<double>(CAP), 1.0 / kk)));
            int per = static_cast<int>(std::round(std::sqrt(
                static_cast<double>(std::min(g.gl_U * g.glag_J, std::max(m1_target, 4))))));
            if (per < 2) per = 2;
            mU = mJ = per;
        }
        std::vector<double> s, p;
        single_jump_mesh(h.beta, h.mu_J, dt, mU, mJ, s, p);
        const int m1 = static_cast<int>(s.size());

        // Odometer over kk single-jump indices: delta = sum s, weight = pmf[k]*prod p.
        std::vector<int> idx(kk, 0);
        const std::size_t total = static_cast<std::size_t>(std::llround(std::pow(
            static_cast<double>(m1), kk)));
        for (std::size_t t = 0; t < total; ++t) {
            double d = 0.0, w = pmf[kk];
            for (int j = 0; j < kk; ++j) {
                d += s[idx[j]];
                w *= p[idx[j]];
            }
            k.delta_Y.push_back(d);
            k.w_Y.push_back(w);
            // advance odometer
            for (int j = 0; j < kk; ++j) {
                if (++idx[j] < m1) break;
                idx[j] = 0;
            }
        }
    }

    // renormalize against tiny floating drift
    double wsum = 0.0;
    for (double w : k.w_Y) wsum += w;
    if (wsum > 0.0)
        for (double& w : k.w_Y) w /= wsum;
    return k;
}

// Constant transfer matrices for the separable expectation E[U_{j+1}]:
//   A_X[i,a] = sum_m w_X[m] * L_a(decay_X x_i + sigma_X z_X[m])   (n_X x n_X)
//   A_Y[k,b] = sum_n w_Y[n] * L_b(decay_Y y_k + delta_Y[n])      (n_Y x n_Y)
// where L_* are cardinal interpolation weights. Then per Q'-slice:
//   W[:,:,Q'] = A_X * U[:,:,Q'] * A_Y^T   (two GEMMs).
// Built once (in double, then cast to Scalar). Each row sums to 1 (partition of unity).
struct TransferMatrices {
    std::vector<Scalar> A_X, A_Y;  // row-major, n_X x n_X and n_Y x n_Y
    int nX = 0, nY = 0;
};

inline TransferMatrices build_transfer_matrices(const TransitionKernel& k,
                                                const std::vector<double>& gx,
                                                const std::vector<double>& gy,
                                                InterpMode mode) {
    TransferMatrices T;
    T.nX = static_cast<int>(gx.size());
    T.nY = static_cast<int>(gy.size());
    std::vector<double> AX(static_cast<size_t>(T.nX) * T.nX, 0.0);
    std::vector<double> AY(static_cast<size_t>(T.nY) * T.nY, 0.0);

    CubicSpline sx, sy;
    if (mode == InterpMode::Cubic) { sx.build(gx); sy.build(gy); }

    for (int i = 0; i < T.nX; ++i) {
        double* row = &AX[static_cast<size_t>(i) * T.nX];
        const double mu = k.decay_X * gx[i];
        for (int m = 0; m < k.M_x(); ++m) {
            const double p = mu + k.sigma_X * k.z_X[m];
            if (mode == InterpMode::Cubic) sx.accumulate_weights(p, k.w_X[m], row);
            else linear_accumulate_weights(gx, p, k.w_X[m], row);
        }
    }
    for (int kk = 0; kk < T.nY; ++kk) {
        double* row = &AY[static_cast<size_t>(kk) * T.nY];
        const double base = k.decay_Y * gy[kk];
        for (int n = 0; n < k.M_y(); ++n) {
            const double p = base + k.delta_Y[n];
            if (mode == InterpMode::Cubic) sy.accumulate_weights(p, k.w_Y[n], row);
            else linear_accumulate_weights(gy, p, k.w_Y[n], row);
        }
    }

    T.A_X.assign(AX.begin(), AX.end());  // cast double -> Scalar
    T.A_Y.assign(AY.begin(), AY.end());
    return T;
}

// Closed-form one-step increment moments for validation.
//   E[X' | x] = decay_X x,  Var[X'] = sigma_X^2
//   E[Delta] = lam mu_J (1 - decay_Y) / beta
//   Var[Delta] = lam mu_J^2 (1 - decay_Y^2) / beta
struct KernelMoments {
    double EdX, VarX, EdDelta, VarDelta;
};
inline KernelMoments kernel_moments_closed_form(const HHKParams& h, double dt) {
    const double decay_X = std::exp(-h.alpha * dt);
    const double varX = h.sigma * h.sigma * (1.0 - decay_X * decay_X) / (2.0 * h.alpha);
    const double decay_Y = std::exp(-h.beta * dt);
    KernelMoments m;
    m.EdX = decay_X;  // E[X'] per unit x
    m.VarX = varX;
    m.EdDelta = h.lam * h.mu_J * (1.0 - decay_Y) / h.beta;
    m.VarDelta = h.lam * h.mu_J * h.mu_J * (1.0 - decay_Y * decay_Y) / h.beta;
    return m;
}

} // namespace grid_dp
