// dp.hpp — backward dynamic-programming driver.
//
// Value layout U[ix][q][iy] (row-major, ix outer): index ix*(nQ*nY) + q*nY + iy.
// This makes the separable one-step expectation E[U_{j+1}] two large GEMMs per step:
//   tmp(nX x nQ*nY) = A_X(nX x nX) * U(nX x nQ*nY)        [contract X]
//   W(nX*nQ x nY)   = tmp(nX*nQ x nY) * A_Y^T(nY x nY)    [contract Y]
// W[ix][q][iy] = E[U_{j+1}(X',Y',Q'=gQ[q]) | x=gx[ix], y=gy[iy]].  Then per (ix,iy) the
// continuation along Q' is a PCHIP and the inner control solve gives U_j (see payoff.hpp).
//
//   U_{N-1}(x,y,Q) = max_q pi(q; S)                 (terminal, W=0)
//   U_j(x,y,Q)     = max_q [ pi(q; S) + df*W_j(x,y,Q+q) ]
//   V_0 = U_0(X0, Y0, 0),  X0 = log(S0), Y0 = 0.
#pragma once

#include "grid_dp/config.hpp"
#include "grid_dp/hhk_sim.hpp"
#include "grid_dp/kernel.hpp"
#include "grid_dp/linalg.hpp"
#include "grid_dp/payoff.hpp"
#include "grid_dp/spline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace grid_dp {

// Split [0, n) across `nt` threads (nt<=0 -> hardware concurrency); body(i) per index.
template <class Body>
inline void parallel_for(int n, int nt, Body&& body) {
    if (nt <= 0) nt = std::max(1u, std::thread::hardware_concurrency());
    nt = std::min(nt, std::max(1, n));
    if (nt == 1) { for (int i = 0; i < n; ++i) body(i); return; }
    std::vector<std::thread> pool;
    pool.reserve(nt - 1);
    const int chunk = (n + nt - 1) / nt;
    for (int t = 1; t < nt; ++t) {
        const int lo = t * chunk, hi = std::min(n, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([lo, hi, &body] { for (int i = lo; i < hi; ++i) body(i); });
    }
    for (int i = 0; i < std::min(n, chunk); ++i) body(i);  // chunk 0 on this thread
    for (auto& th : pool) th.join();
}

inline std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    if (n == 1) { v[0] = lo; return v; }
    for (int i = 0; i < n; ++i) v[i] = lo + (hi - lo) * i / static_cast<double>(n - 1);
    return v;
}

struct DPResult {
    double price = 0.0;
    std::vector<double> gx, gy, gQ;          // grids
    std::vector<std::vector<Scalar>> policy; // policy[j] = optimal lift q*(grid) at date j
    InterpMode mode = InterpMode::Cubic;
    int M_y = 0;                             // jump-mesh size (diagnostic)
    double t_build_ms = 0.0, t_backward_ms = 0.0, t_total_ms = 0.0;
};

// Fast 8-point trilinear interpolation of a value array (layout [ix][q][iy]) at (x,y,Q).
// Used for the greedy-policy forward rollout (cheap; clamps out-of-range).
inline double interp3_linear(const std::vector<Scalar>& A, const std::vector<double>& gx,
                             const std::vector<double>& gy, const std::vector<double>& gQ,
                             double x, double y, double Q) {
    const int nX = (int)gx.size(), nY = (int)gy.size(), nQ = (int)gQ.size();
    auto cell = [](const std::vector<double>& g, double v, int& i, double& t) {
        const int n = (int)g.size();
        if (v <= g[0]) { i = 0; t = 0.0; return; }
        if (v >= g[n - 1]) { i = n - 2; t = 1.0; return; }
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) { const int m = (lo + hi) >> 1; if (v < g[m]) hi = m; else lo = m; }
        i = lo; t = (v - g[lo]) / (g[lo + 1] - g[lo]);
    };
    int ix, iy, iq; double tx, ty, tq;
    cell(gx, x, ix, tx); cell(gy, y, iy, ty); cell(gQ, Q, iq, tq);
    auto at = [&](int a, int q, int b) -> double {
        return A[(size_t)a * nQ * nY + (size_t)q * nY + b];
    };
    double v = 0.0;
    for (int dxi = 0; dxi < 2; ++dxi)
        for (int dqi = 0; dqi < 2; ++dqi)
            for (int dyi = 0; dyi < 2; ++dyi) {
                const double w = (dxi ? tx : 1 - tx) * (dqi ? tq : 1 - tq) * (dyi ? ty : 1 - ty);
                v += w * at(ix + dxi, iq + dqi, iy + dyi);
            }
    return v;
}

// Tensor (X,Y,Q) evaluation of a value array with the chosen interpolation.
inline double eval_xyq(const std::vector<Scalar>& U, const std::vector<double>& gx,
                       const std::vector<double>& gy, const std::vector<double>& gQ,
                       double x, double y, double Q, InterpMode mode) {
    const int nX = (int)gx.size(), nY = (int)gy.size(), nQ = (int)gQ.size();
    std::vector<double> wx(nX, 0.0), wy(nY, 0.0), wq(nQ, 0.0);
    if (mode == InterpMode::Cubic) {
        CubicSpline sx, sy, sq;
        sx.build(gx); sy.build(gy); sq.build(gQ);
        sx.accumulate_weights(x, 1.0, wx.data());
        sy.accumulate_weights(y, 1.0, wy.data());
        sq.accumulate_weights(Q, 1.0, wq.data());
    } else {
        linear_accumulate_weights(gx, x, 1.0, wx.data());
        linear_accumulate_weights(gy, y, 1.0, wy.data());
        linear_accumulate_weights(gQ, Q, 1.0, wq.data());
    }
    double v = 0.0;
    for (int ix = 0; ix < nX; ++ix) {
        if (wx[ix] == 0.0) continue;
        for (int q = 0; q < nQ; ++q) {
            if (wq[q] == 0.0) continue;
            const double wxq = wx[ix] * wq[q];
            const size_t base = (size_t)ix * nQ * nY + (size_t)q * nY;
            double acc = 0.0;
            for (int iy = 0; iy < nY; ++iy) acc += wy[iy] * U[base + iy];
            v += wxq * acc;
        }
    }
    return v;
}

inline DPResult price_backward(const SwingContract& c, const HHKParams& h,
                               const GridParams& g, bool store_policy = true) {
    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    const int nX = g.n_X, nY = g.n_Y, nQ = g.n_Q;
    const int N = c.n_rights;
    const double dt = c.dt();
    const double df = c.discount_factor();
    const double K = c.K, cc = c.c_cost, gam = c.gamma_cost;
    const double Qmax = c.Q_max, qmax = c.q_max;

    DPResult R;
    R.mode = static_cast<InterpMode>(g.interp_xy);
    R.gx = linspace(g.X_lo, g.X_hi, nX);
    R.gy = linspace(g.Y_lo, g.Y_hi, nY);
    R.gQ = linspace(0.0, Qmax, nQ);

    const TransitionKernel kern = build_kernel(h, dt, g);
    R.M_y = kern.M_y();
    const TransferMatrices T = build_transfer_matrices(kern, R.gx, R.gy, R.mode);

    // S grid: S = exp(x + y) (f == 0).
    std::vector<double> Sg((size_t)nX * nY);
    for (int ix = 0; ix < nX; ++ix)
        for (int iy = 0; iy < nY; ++iy)
            Sg[(size_t)ix * nY + iy] = std::exp(R.gx[ix] + R.gy[iy]);

    const size_t NSZ = (size_t)nX * nQ * nY;
    std::vector<Scalar> U(NSZ), Wbuf(NSZ), tmp(NSZ);
    if (store_policy) R.policy.assign(N, {});

    // Per-state residual budget cap and q-cap.
    auto qcap_at = [&](double Ql) { return std::min(qmax, std::max(0.0, Qmax - Ql)); };

    const auto t1 = clock::now();

    // ---- Terminal U_{N-1}: max_q pi(q) (W = 0) ----
    if (store_policy) R.policy[N - 1].assign(NSZ, Scalar(0));
    parallel_for(nX, g.n_threads, [&](int ix) {
        for (int q = 0; q < nQ; ++q) {
            const double qc = qcap_at(R.gQ[q]);
            const size_t bse = (size_t)ix * nQ * nY + (size_t)q * nY;
            for (int iy = 0; iy < nY; ++iy) {
                const InnerResult r = solve_terminal(Sg[(size_t)ix * nY + iy], K, cc, gam, qc);
                U[bse + iy] = static_cast<Scalar>(r.value);
                if (store_policy) R.policy[N - 1][bse + iy] = static_cast<Scalar>(r.q);
            }
        }
    });

    // ---- Backward j = N-2 .. 0 ----
    for (int j = N - 2; j >= 0; --j) {
        // W = A_X * U * A_Y^T  (two GEMMs)
        gemm(false, false, nX, nQ * nY, nX, Scalar(1), T.A_X.data(), nX, U.data(),
             nQ * nY, Scalar(0), tmp.data(), nQ * nY);
        gemm(false, true, nX * nQ, nY, nY, Scalar(1), tmp.data(), nY, T.A_Y.data(), nY,
             Scalar(0), Wbuf.data(), nY);

        if (store_policy) R.policy[j].assign(NSZ, Scalar(0));

        // U_j = max_q [ pi(q) + df * W_j(Q+q) ] — inner control parallel over X-rows
        // (each row writes disjoint U entries; Wbuf is read-only this pass).
        parallel_for(nX, g.n_threads, [&](int ix) {
            std::vector<double> Wcol(nQ);
            PCHIP pc;
            for (int iy = 0; iy < nY; ++iy) {
                const double S = Sg[(size_t)ix * nY + iy];
                for (int q = 0; q < nQ; ++q)
                    Wcol[q] = Wbuf[(size_t)ix * nQ * nY + (size_t)q * nY + iy];
                pc.build(R.gQ, Wcol);
                for (int q = 0; q < nQ; ++q) {
                    const double qc = qcap_at(R.gQ[q]);
                    const InnerResult r = solve_inner(S, K, cc, gam, qc, df, R.gQ, Wcol, pc, q);
                    const size_t idx = (size_t)ix * nQ * nY + (size_t)q * nY + iy;
                    U[idx] = static_cast<Scalar>(r.value);
                    if (store_policy) R.policy[j][idx] = static_cast<Scalar>(r.q);
                }
            }
        });
    }

    const auto t2 = clock::now();

    // ---- Price = U_0(X0, Y0=0, Q0=0) ----
    const double X0 = std::log(h.S0);  // f(0)=0
    R.price = eval_xyq(U, R.gx, R.gy, R.gQ, X0, 0.0, 0.0, R.mode);

    R.t_build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    R.t_backward_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    R.t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    return R;
}

struct MCResult {
    double price = 0.0;   // mean discounted payoff of the DP greedy policy
    double stderr_ = 0.0; // standard error of the mean
    double ci_lo = 0.0, ci_hi = 0.0;  // 95% CI
    int n_paths = 0;
};

// Self-consistency check (GOAL 2.2): roll the stored greedy policy forward on independent
// OOS HHK paths, accumulate sum_j df^j pi_j (env semantics incl. the profitability gate).
// The mean is an unbiased estimate of the policy value and a true lower bound on the optimum;
// it should approach backward U_0 from below as the grid/quadrature refine.
inline MCResult forward_mc(const DPResult& R, const SwingContract& c, const HHKParams& h,
                           int n_paths, std::uint64_t seed) {
    const int N = c.n_rights;
    const double dt = c.dt();
    const double df = c.discount_factor();
    const double K = c.K, cc = c.c_cost, gam = c.gamma_cost;
    const double Qmax = c.Q_max, qmax = c.q_max;
    Xoshiro rng(seed);

    double sum = 0.0, sum2 = 0.0;
    for (int p = 0; p < n_paths; ++p) {
        const HHKPath path = simulate_path(h, dt, N, rng);
        double Q = 0.0, disc = 1.0, pv = 0.0;
        for (int j = 0; j < N; ++j) {
            const double qc = std::min(qmax, std::max(0.0, Qmax - Q));
            if (qc > 1e-12) {
                const double S = std::exp(path.X[j] + path.Y[j]);
                double q = interp3_linear(R.policy[j], R.gx, R.gy, R.gQ, path.X[j], path.Y[j], Q);
                q = std::min(std::max(q, 0.0), qc);
                const double mpos = std::fmax(S - K, 0.0);
                double pi = q * mpos - cc * std::pow(q, gam);
                if (pi <= 0.0) { q = 0.0; pi = 0.0; }  // env profitability gate
                pv += disc * pi;
                Q += q;
            }
            disc *= df;
        }
        sum += pv;
        sum2 += pv * pv;
    }
    MCResult m;
    m.n_paths = n_paths;
    m.price = sum / n_paths;
    const double var = std::max(0.0, sum2 / n_paths - m.price * m.price);
    m.stderr_ = std::sqrt(var / n_paths);
    m.ci_lo = m.price - 1.959963985 * m.stderr_;
    m.ci_hi = m.price + 1.959963985 * m.stderr_;
    return m;
}

} // namespace grid_dp
