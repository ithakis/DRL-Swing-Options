// test_gauss — Gaussian-quadrature correctness:
//   (1) exact integration of polynomials up to degree 2n-1,
//   (2) analytic small-n node/weight values,
//   (3) reproduction of N(0,1) / Exp(1) / Unif(0,1) raw moments.
#include "grid_dp/gauss.hpp"
#include "test_util.hpp"

#include <cmath>

using namespace grid_dp;
using namespace gdp_test;

// double-factorial moments of N(0,1): E[Z^{2k}] = (2k-1)!!
static double normal_moment(int p) {
    if (p % 2 == 1) return 0.0;
    double m = 1.0;
    for (int k = 1; k <= p - 1; k += 2) m *= k;
    return m;
}
static double factorial(int n) {
    double f = 1.0;
    for (int k = 2; k <= n; ++k) f *= k;
    return f;
}

// Relative-error check of a quadrature moment against its true value, normalized by
// the natural scale E[|x|^p] so it works for both nonzero (even) and zero (odd)
// moments regardless of magnitude. Exactness of an n-point rule holds to ~machine
// precision in *relative* terms; absolute error grows with the moment magnitude.
static void check_moment(const Quadrature& q, int p, double want, double tol,
                         const std::string& msg) {
    double I = 0.0, scale = 0.0;
    for (size_t i = 0; i < q.x.size(); ++i) {
        I += q.w[i] * std::pow(q.x[i], p);
        scale += q.w[i] * std::pow(std::fabs(q.x[i]), p);
    }
    if (scale < 1.0) scale = 1.0;  // p=0 / tiny-node guard
    gdp_test::expect_le(std::fabs(I - want) / scale, tol, msg);
}

int main() {
    // ---- Gauss–Hermite (probabilist, N(0,1)) ----
    for (int n : {2, 4, 6, 12, 24}) {
        Quadrature q = gauss_hermite_prob(n);
        double sw = 0.0;
        for (double w : q.w) sw += w;
        expect_near(sw, 1.0, 1e-13, "GH weights sum to 1 (n=" + std::to_string(n) + ")");
        // exact up to degree 2n-1 (relative, normalized by the moment scale)
        for (int p = 0; p <= 2 * n - 1; ++p) {
            check_moment(q, p, normal_moment(p), 1e-11,
                         "GH moment p=" + std::to_string(p) + " n=" + std::to_string(n));
        }
    }
    // analytic n=2: nodes +/-1, weights 1/2
    {
        Quadrature q = gauss_hermite_prob(2);
        expect_near(q.x[0], -1.0, 1e-13, "GH n=2 node0");
        expect_near(q.x[1], 1.0, 1e-13, "GH n=2 node1");
        expect_near(q.w[0], 0.5, 1e-13, "GH n=2 w0");
    }
    // analytic n=3: nodes 0, +/-sqrt(3); weights 2/3, 1/6, 1/6
    {
        Quadrature q = gauss_hermite_prob(3);
        expect_near(q.x[0], -std::sqrt(3.0), 1e-12, "GH n=3 node0");
        expect_near(q.x[1], 0.0, 1e-12, "GH n=3 node1");
        expect_near(q.x[2], std::sqrt(3.0), 1e-12, "GH n=3 node2");
        expect_near(q.w[1], 2.0 / 3.0, 1e-12, "GH n=3 w-center");
        expect_near(q.w[0], 1.0 / 6.0, 1e-12, "GH n=3 w-wing");
    }

    // ---- Gauss–Legendre on (0,1), E_{Unif(0,1)} ----
    for (int n : {2, 4, 8, 16}) {
        Quadrature q = gauss_legendre_unit(n);
        double sw = 0.0;
        for (double w : q.w) sw += w;
        expect_near(sw, 1.0, 1e-13, "GL weights sum to 1 (n=" + std::to_string(n) + ")");
        for (int p = 0; p <= 2 * n - 1; ++p) {
            check_moment(q, p, 1.0 / (p + 1), 1e-12,
                         "GL moment p=" + std::to_string(p) + " n=" + std::to_string(n));
        }
    }

    // ---- Gauss–Laguerre, E_{Exp(1)}[x^p] = p! ----
    for (int n : {2, 4, 8, 16}) {
        Quadrature q = gauss_laguerre_exp(n);
        double sw = 0.0;
        for (double w : q.w) sw += w;
        expect_near(sw, 1.0, 1e-13, "GLag weights sum to 1 (n=" + std::to_string(n) + ")");
        for (int p = 0; p <= 2 * n - 1; ++p) {
            check_moment(q, p, factorial(p), 1e-10,
                         "GLag moment p=" + std::to_string(p) + " n=" + std::to_string(n));
        }
    }

    return summary("test_gauss");
}
