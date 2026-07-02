// test_kernel_moments — the native C++ kernel reproduces the closed-form one-step
// moments of (X', Y') derived analytically from the HHK transition.
#include "grid_dp/kernel.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace grid_dp;
using namespace gdp_test;

int main() {
    HHKParams h;  // v64 focal defaults
    SwingContract c;
    const double dt = c.dt();
    GridParams g;
    g.M_x = 24;
    g.N_max = 6;  // push the Poisson-tail fold below 1e-12 so this test isolates
                  // the *quadrature* accuracy (production uses N_max=3; tail ~1e-8,
                  // price impact << 1e-4 — see RESULTS convergence study).
    g.gl_U = 24;
    g.glag_J = 24;

    const TransitionKernel k = build_kernel(h, dt, g);
    const KernelMoments mm = kernel_moments_closed_form(h, dt);

    // process constants
    expect_near(k.decay_X, std::exp(-h.alpha * dt), 1e-14, "decay_X");
    expect_near(k.decay_Y, std::exp(-h.beta * dt), 1e-14, "decay_Y");
    expect_near(k.sigma_X * k.sigma_X, mm.VarX, 1e-14, "sigma_X^2 == VarX");

    // weights normalized
    double swx = 0.0, swy = 0.0;
    for (double w : k.w_X) swx += w;
    for (double w : k.w_Y) swy += w;
    expect_near(swx, 1.0, 1e-13, "sum w_X == 1");
    expect_near(swy, 1.0, 1e-13, "sum w_Y == 1");

    // X' Gaussian moments: E[Z]=0, E[Z^2]=1, E[Z^3]=0, E[Z^4]=3
    double m1 = 0, m2 = 0, m3 = 0, m4 = 0;
    for (int i = 0; i < k.M_x(); ++i) {
        const double z = k.z_X[i], w = k.w_X[i];
        m1 += w * z;
        m2 += w * z * z;
        m3 += w * z * z * z;
        m4 += w * z * z * z * z;
    }
    expect_near(m1, 0.0, 1e-13, "E[Z]=0");
    expect_near(m2, 1.0, 1e-13, "E[Z^2]=1");
    expect_near(m3, 0.0, 1e-12, "E[Z^3]=0");
    expect_near(m4, 3.0, 1e-12, "E[Z^4]=3");

    // jump increment Delta moments vs closed form
    double e1 = 0, e2 = 0;
    for (int i = 0; i < k.M_y(); ++i) {
        e1 += k.w_Y[i] * k.delta_Y[i];
        e2 += k.w_Y[i] * k.delta_Y[i] * k.delta_Y[i];
    }
    const double varDelta = e2 - e1 * e1;
    expect_near(e1, mm.EdDelta, 1e-9, "E[Delta]");
    expect_near(varDelta, mm.VarDelta, 1e-9, "Var[Delta]");

    // ---- transfer matrices A_X / A_Y (the GEMM operators) ----
    auto linspace = [](double lo, double hi, int n) {
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) v[i] = lo + (hi - lo) * i / (n - 1);
        return v;
    };
    const std::vector<double> gx = linspace(-1.6, 1.6, 121);
    const std::vector<double> gy = linspace(0.0, 4.0, 97);
    const TransferMatrices T = build_transfer_matrices(k, gx, gy, InterpMode::Cubic);

    // Partition of unity: every row sums to 1 (constant function -> constant).
    double max_rowsum_err_X = 0.0, max_rowsum_err_Y = 0.0;
    for (int i = 0; i < T.nX; ++i) {
        double s = 0.0;
        for (int a = 0; a < T.nX; ++a) s += T.A_X[i * T.nX + a];
        max_rowsum_err_X = std::max(max_rowsum_err_X, std::fabs(s - 1.0));
    }
    for (int kk = 0; kk < T.nY; ++kk) {
        double s = 0.0;
        for (int b = 0; b < T.nY; ++b) s += T.A_Y[kk * T.nY + b];
        max_rowsum_err_Y = std::max(max_rowsum_err_Y, std::fabs(s - 1.0));
    }
    expect_le(max_rowsum_err_X, 1e-12, "A_X rows sum to 1");
    expect_le(max_rowsum_err_Y, 1e-12, "A_Y rows sum to 1");

    // Mean reproduction: (A_X g_x)[i] = decay_X x_i for interior i (no boundary clamp).
    for (int i : {40, 60, 80}) {  // |x_i| <= ~0.6 so shifted nodes stay in-grid
        double mx = 0.0;
        for (int a = 0; a < T.nX; ++a) mx += T.A_X[i * T.nX + a] * gx[a];
        expect_near(mx, k.decay_X * gx[i], 1e-9, "A_X reproduces conditional mean");
    }
    // (A_Y g_y)[0] = decay_Y*0 + E[Delta] = E[Delta] at y=0.
    {
        double my = 0.0;
        for (int b = 0; b < T.nY; ++b) my += T.A_Y[0 * T.nY + b] * gy[b];
        expect_near(my, mm.EdDelta, 1e-5, "A_Y reproduces E[Delta] at y=0");
    }

    return summary("test_kernel_moments");
}
