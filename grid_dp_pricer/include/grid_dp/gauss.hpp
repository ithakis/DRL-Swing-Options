// gauss.hpp — Gaussian quadrature via the Golub–Welsch algorithm.
//
// All rules are returned already normalized to a *probability* weight so the user
// integrates an expectation directly:
//   * gauss_hermite_prob(n)  -> nodes z, weights w with  sum w_i g(z_i) ≈ E_{Z~N(0,1)}[g(Z)]
//   * gauss_legendre_unit(n) -> nodes u, weights w with  sum w_i g(u_i) ≈ E_{U~Unif(0,1)}[g(U)]
//   * gauss_laguerre_exp(n)  -> nodes x, weights w with  sum w_i g(x_i) ≈ E_{E~Exp(1)}[g(E)]
//
// The eigenvalues of the symmetric Jacobi (tridiagonal) matrix are the nodes; the
// weights are mu0 * (first component of each unit eigenvector)^2.  This matches
// numpy.polynomial.hermite_e.hermegauss (then /sqrt(2*pi)) to ~1e-13.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace grid_dp {

struct Quadrature {
    std::vector<double> x;  // nodes
    std::vector<double> w;  // weights (sum to 1 for the probability-normalized rules)
};

// Symmetric tridiagonal eigensolver (implicit-shift QL with eigenvectors).
// d[0..n-1] diagonal; e[i] couples (i, i+1) for i=0..n-2, e[n-1] unused.
// z is n*n row-major, initialized to the identity; on return its columns hold the
// orthonormal eigenvectors (z[k*n + j] = component k of eigenvector j).
inline void symtridiag_ql(int n, std::vector<double>& d, std::vector<double>& e,
                          std::vector<double>& z) {
    if (n == 1) return;
    const double eps = std::numeric_limits<double>::epsilon();
    for (int l = 0; l < n; ++l) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; ++m) {
                const double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
                if (std::fabs(e[m]) <= eps * dd) break;
            }
            if (m != l) {
                if (++iter > 60) throw std::runtime_error("symtridiag_ql: no convergence");
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = std::hypot(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
                double s = 1.0, c = 1.0, p = 0.0;
                int i = m - 1;
                for (; i >= l; --i) {
                    double f = s * e[i];
                    const double b = c * e[i];
                    r = std::hypot(f, g);
                    e[i + 1] = r;
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;
                    for (int k = 0; k < n; ++k) {
                        f = z[k * n + i + 1];
                        z[k * n + i + 1] = s * z[k * n + i] + c * f;
                        z[k * n + i] = c * z[k * n + i] - s * f;
                    }
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
}

// Golub–Welsch from a Jacobi matrix: diagonal `a` (size n) and sub-diagonal `b`
// (size n-1, the actual off-diagonal entries J[i,i+1]); mu0 = integral of the weight.
inline Quadrature golub_welsch(std::vector<double> a, const std::vector<double>& b,
                               double mu0) {
    const int n = static_cast<int>(a.size());
    std::vector<double> d = std::move(a);
    std::vector<double> e(n, 0.0);
    for (int i = 0; i < n - 1; ++i) e[i] = b[i];
    std::vector<double> z(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) z[i * n + i] = 1.0;

    symtridiag_ql(n, d, e, z);

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int p, int q) { return d[p] < d[q]; });

    Quadrature out;
    out.x.resize(n);
    out.w.resize(n);
    for (int j = 0; j < n; ++j) {
        const int col = idx[j];
        out.x[j] = d[col];
        const double v0 = z[0 * n + col];  // first component of eigenvector `col`
        out.w[j] = mu0 * v0 * v0;
    }
    return out;
}

// Probabilist Gauss–Hermite normalized to N(0,1): monic recurrence
//   He_{k+1} = x He_k - k He_{k-1}  =>  a_k = 0, off-diagonal sqrt(k).
inline Quadrature gauss_hermite_prob(int n) {
    std::vector<double> a(n, 0.0);
    std::vector<double> b(std::max(n - 1, 0));
    for (int i = 0; i < n - 1; ++i) b[i] = std::sqrt(static_cast<double>(i + 1));
    constexpr double mu0 = 2.5066282746310002;  // sqrt(2*pi)
    Quadrature q = golub_welsch(std::move(a), b, mu0);
    for (double& w : q.w) w /= mu0;  // normalize to a probability measure (sum -> 1)
    return q;
}

// Gauss–Legendre on (0,1) for E_{U~Unif(0,1)}:  a_k=0, b_k = k/sqrt(4k^2-1) on [-1,1].
inline Quadrature gauss_legendre_unit(int n) {
    std::vector<double> a(n, 0.0);
    std::vector<double> b(std::max(n - 1, 0));
    for (int i = 0; i < n - 1; ++i) {
        const double k = static_cast<double>(i + 1);
        b[i] = k / std::sqrt(4.0 * k * k - 1.0);
    }
    Quadrature q = golub_welsch(std::move(a), b, 2.0);  // mu0 = length of [-1,1]
    for (int i = 0; i < n; ++i) {
        q.x[i] = 0.5 * (q.x[i] + 1.0);  // map [-1,1] -> [0,1]
        q.w[i] *= 0.5;                  // weights -> sum to 1
    }
    return q;
}

// Gauss–Laguerre (weight e^{-x} on [0,inf)) for E_{E~Exp(1)}:
//   a_k = 2k+1, b_k = k.  mu0 = 1, weights already sum to 1.
inline Quadrature gauss_laguerre_exp(int n) {
    std::vector<double> a(n);
    for (int i = 0; i < n; ++i) a[i] = 2.0 * i + 1.0;
    std::vector<double> b(std::max(n - 1, 0));
    for (int i = 0; i < n - 1; ++i) b[i] = static_cast<double>(i + 1);
    return golub_welsch(std::move(a), b, 1.0);
}

} // namespace grid_dp
