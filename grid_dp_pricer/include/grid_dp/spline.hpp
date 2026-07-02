// spline.hpp — 1-D interpolation primitives.
//
//  * CubicSpline (not-a-knot, natural fallback): linear-in-values, so its cardinal
//    weights feed the constant transfer matrices A_X / A_Y (the GEMM key).
//  * linear_cardinal_weights: multilinear alternative for the convergence study.
//  * PCHIP: shape-preserving Hermite cubic for the Q axis (concave-respecting,
//    overshoot-free) used in the inner control solve; provides value + derivative.
//
// Out-of-range evaluation clamps to the boundary knot (constant extrapolation) — the
// grids are sized so the truncated probability mass is negligible (~1e-6).
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace grid_dp {

// Solve A X = B for X (A is n×n, B is n×m), Gaussian elimination + partial pivoting.
// Row-major; A is modified in place. Returns X (n×m).
inline std::vector<double> solve_dense(std::vector<double> A, std::vector<double> B,
                                       int n, int m) {
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::fabs(A[col * n + col]);
        for (int r = col + 1; r < n; ++r) {
            const double v = std::fabs(A[r * n + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best == 0.0) throw std::runtime_error("solve_dense: singular");
        if (piv != col) {
            for (int c = 0; c < n; ++c) std::swap(A[col * n + c], A[piv * n + c]);
            for (int c = 0; c < m; ++c) std::swap(B[col * m + c], B[piv * m + c]);
        }
        const double inv = 1.0 / A[col * n + col];
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = A[r * n + col] * inv;
            if (f == 0.0) continue;
            for (int c = col; c < n; ++c) A[r * n + c] -= f * A[col * n + c];
            for (int c = 0; c < m; ++c) B[r * m + c] -= f * B[col * m + c];
        }
    }
    std::vector<double> X(static_cast<size_t>(n) * m);
    for (int r = 0; r < n; ++r) {
        const double inv = 1.0 / A[r * n + r];
        for (int c = 0; c < m; ++c) X[r * m + c] = B[r * m + c] * inv;
    }
    return X;
}

// Interpolation mode for the (X,Y) transfer matrices.
enum class InterpMode { Linear = 0, Cubic = 1 };

// Not-a-knot cubic spline. Stores the moment matrix C (n×n): M = C·y maps node values
// to second derivatives.  cardinal_weights(p) returns the dense length-n weights w with
// s(p) = sum_a w_a y_a.
struct CubicSpline {
    std::vector<double> x;  // knots (ascending)
    std::vector<double> C;  // moment matrix, row-major n×n
    int n = 0;

    void build(const std::vector<double>& knots) {
        x = knots;
        n = static_cast<int>(x.size());
        if (n < 4) {
            // Not-a-knot needs >=4 knots; fall back to natural spline (M_0=M_{n-1}=0).
            build_natural();
            return;
        }
        std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
        std::vector<double> B(static_cast<size_t>(n) * n, 0.0);
        auto h = [&](int i) { return x[i + 1] - x[i]; };
        for (int i = 1; i <= n - 2; ++i) {
            const double hm = h(i - 1), hi = h(i);
            A[i * n + (i - 1)] = hm / 6.0;
            A[i * n + i] = (hm + hi) / 3.0;
            A[i * n + (i + 1)] = hi / 6.0;
            B[i * n + (i - 1)] = 1.0 / hm;
            B[i * n + i] = -(1.0 / hm + 1.0 / hi);
            B[i * n + (i + 1)] = 1.0 / hi;
        }
        // not-a-knot: s''' continuous across x_1 and x_{n-2}
        const double h0 = h(0), h1 = h(1);
        A[0 * n + 0] = -h1;
        A[0 * n + 1] = h0 + h1;
        A[0 * n + 2] = -h0;
        const double hl = h(n - 2), hl2 = h(n - 3);
        A[(n - 1) * n + (n - 3)] = -hl;
        A[(n - 1) * n + (n - 2)] = hl2 + hl;
        A[(n - 1) * n + (n - 1)] = -hl2;
        C = solve_dense(std::move(A), std::move(B), n, n);
    }

    void build_natural() {
        std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
        std::vector<double> B(static_cast<size_t>(n) * n, 0.0);
        auto h = [&](int i) { return x[i + 1] - x[i]; };
        A[0] = 1.0;                        // M_0 = 0
        A[(n - 1) * n + (n - 1)] = 1.0;    // M_{n-1} = 0
        for (int i = 1; i <= n - 2; ++i) {
            const double hm = h(i - 1), hi = h(i);
            A[i * n + (i - 1)] = hm / 6.0;
            A[i * n + i] = (hm + hi) / 3.0;
            A[i * n + (i + 1)] = hi / 6.0;
            B[i * n + (i - 1)] = 1.0 / hm;
            B[i * n + i] = -(1.0 / hm + 1.0 / hi);
            B[i * n + (i + 1)] = 1.0 / hi;
        }
        C = solve_dense(std::move(A), std::move(B), n, n);
    }

    int interval(double p) const {
        if (p <= x[0]) return 0;
        if (p >= x[n - 1]) return n - 2;
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (p < x[mid]) hi = mid; else lo = mid;
        }
        return lo;
    }

    // Accumulate the cardinal weights of point p into w (length n), scaled by `scale`.
    // (Used to assemble A_X / A_Y without per-call allocation.)
    void accumulate_weights(double p, double scale, double* w) const {
        const double pc = std::min(std::max(p, x[0]), x[n - 1]);  // clamp / const-extrap
        const int j = interval(pc);
        const double hh = x[j + 1] - x[j];
        const double t = (pc - x[j]) / hh;
        const double a = 1.0 - t, b = t;
        const double cMj = (a * a * a - a) * hh * hh / 6.0;
        const double cMj1 = (b * b * b - b) * hh * hh / 6.0;
        w[j] += scale * a;
        w[j + 1] += scale * b;
        const double* Cj = &C[static_cast<size_t>(j) * n];
        const double* Cj1 = &C[static_cast<size_t>(j + 1) * n];
        const double s0 = scale * cMj, s1 = scale * cMj1;
        for (int a2 = 0; a2 < n; ++a2) w[a2] += s0 * Cj[a2] + s1 * Cj1[a2];
    }
};

// Multilinear cardinal weights of point p on ascending knots, accumulated into w.
inline void linear_accumulate_weights(const std::vector<double>& x, double p, double scale,
                                      double* w) {
    const int n = static_cast<int>(x.size());
    if (p <= x[0]) { w[0] += scale; return; }
    if (p >= x[n - 1]) { w[n - 1] += scale; return; }
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (p < x[mid]) hi = mid; else lo = mid;
    }
    const double t = (p - x[lo]) / (x[lo + 1] - x[lo]);
    w[lo] += scale * (1.0 - t);
    w[lo + 1] += scale * t;
}

// Shape-preserving piecewise-cubic Hermite (PCHIP), MATLAB-compatible slopes.
struct PCHIP {
    std::vector<double> x, v, d;  // knots, values, slopes
    int n = 0;

    static double endpoint_slope(double h0, double h1, double del0, double del1) {
        double d = ((2.0 * h0 + h1) * del0 - h0 * del1) / (h0 + h1);
        if (d * del0 <= 0.0) d = 0.0;
        else if ((del0 * del1 <= 0.0) && (std::fabs(d) > 3.0 * std::fabs(del0)))
            d = 3.0 * del0;
        return d;
    }

    void build(const std::vector<double>& knots, const std::vector<double>& vals) {
        x = knots;
        v = vals;
        n = static_cast<int>(x.size());
        d.assign(n, 0.0);
        if (n == 1) { d[0] = 0.0; return; }
        std::vector<double> h(n - 1), del(n - 1);
        for (int i = 0; i < n - 1; ++i) {
            h[i] = x[i + 1] - x[i];
            del[i] = (v[i + 1] - v[i]) / h[i];
        }
        if (n == 2) { d[0] = d[1] = del[0]; return; }
        for (int i = 1; i < n - 1; ++i) {
            if (del[i - 1] * del[i] <= 0.0) {
                d[i] = 0.0;  // local extremum -> flat (no overshoot)
            } else {
                const double w1 = 2.0 * h[i] + h[i - 1];
                const double w2 = h[i] + 2.0 * h[i - 1];
                d[i] = (w1 + w2) / (w1 / del[i - 1] + w2 / del[i]);  // weighted harmonic mean
            }
        }
        d[0] = endpoint_slope(h[0], h[1], del[0], del[1]);
        d[n - 1] = endpoint_slope(h[n - 2], h[n - 3], del[n - 2], del[n - 3]);
    }

    int interval(double p) const {
        if (p <= x[0]) return 0;
        if (p >= x[n - 1]) return n - 2;
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (p < x[mid]) hi = mid; else lo = mid;
        }
        return lo;
    }

    double eval(double p) const {
        if (n == 1) return v[0];
        const double pc = std::min(std::max(p, x[0]), x[n - 1]);
        const int j = interval(pc);
        const double hh = x[j + 1] - x[j];
        const double t = (pc - x[j]) / hh;
        const double t2 = t * t, t3 = t2 * t;
        const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        const double h10 = t3 - 2.0 * t2 + t;
        const double h01 = -2.0 * t3 + 3.0 * t2;
        const double h11 = t3 - t2;
        return h00 * v[j] + hh * h10 * d[j] + h01 * v[j + 1] + hh * h11 * d[j + 1];
    }

    // d/dp of the interpolant (0 outside the knot range, matching constant extrapolation).
    double deriv(double p) const {
        if (n == 1 || p <= x[0] || p >= x[n - 1]) return 0.0;
        const int j = interval(p);
        const double hh = x[j + 1] - x[j];
        const double t = (p - x[j]) / hh;
        const double t2 = t * t;
        const double dh00 = 6.0 * t2 - 6.0 * t;
        const double dh10 = 3.0 * t2 - 4.0 * t + 1.0;
        const double dh01 = -6.0 * t2 + 6.0 * t;
        const double dh11 = 3.0 * t2 - 2.0 * t;
        return (dh00 * v[j] + dh01 * v[j + 1]) / hh + dh10 * d[j] + dh11 * d[j + 1];
    }
};

} // namespace grid_dp
