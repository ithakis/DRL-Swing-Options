// test_spline — interpolation correctness:
//   * not-a-knot cubic spline reproduces global cubics exactly + partition of unity,
//   * linear cardinal weights reproduce affine functions + partition of unity,
//   * PCHIP interpolates knots, reproduces affine, is monotone on monotone data,
//     and its analytic derivative matches finite differences.
#include "grid_dp/spline.hpp"
#include "test_util.hpp"

#include <cmath>
#include <vector>

using namespace grid_dp;
using namespace gdp_test;

static double cubic(double x) { return 2.0 * x * x * x - 3.0 * x * x + x + 5.0; }

int main() {
    std::vector<double> knots = {0.0, 0.5, 1.3, 2.0, 2.7, 3.5, 4.0};
    const int n = static_cast<int>(knots.size());

    // ---- cubic spline reproduces a global cubic ----
    CubicSpline sp;
    sp.build(knots);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = cubic(knots[i]);
    for (double p : {0.1, 0.7, 1.0, 1.9, 2.5, 3.1, 3.9}) {
        std::vector<double> w(n, 0.0);
        sp.accumulate_weights(p, 1.0, w.data());
        double s = 0.0, sw = 0.0;
        for (int a = 0; a < n; ++a) { s += w[a] * y[a]; sw += w[a]; }
        expect_near(s, cubic(p), 1e-9, "cubic spline reproduces cubic at p=" + std::to_string(p));
        expect_near(sw, 1.0, 1e-12, "cubic partition of unity at p=" + std::to_string(p));
    }
    // node interpolation
    for (int i = 0; i < n; ++i) {
        std::vector<double> w(n, 0.0);
        sp.accumulate_weights(knots[i], 1.0, w.data());
        double s = 0.0;
        for (int a = 0; a < n; ++a) s += w[a] * y[a];
        expect_near(s, y[i], 1e-9, "cubic spline hits knot " + std::to_string(i));
    }

    // ---- linear weights reproduce an affine function ----
    auto affine = [](double x) { return 1.5 * x - 0.7; };
    std::vector<double> ya(n);
    for (int i = 0; i < n; ++i) ya[i] = affine(knots[i]);
    for (double p : {0.2, 1.1, 2.4, 3.7}) {
        std::vector<double> w(n, 0.0);
        linear_accumulate_weights(knots, p, 1.0, w.data());
        double s = 0.0, sw = 0.0;
        for (int a = 0; a < n; ++a) { s += w[a] * ya[a]; sw += w[a]; }
        expect_near(s, affine(p), 1e-12, "linear reproduces affine at p=" + std::to_string(p));
        expect_near(sw, 1.0, 1e-13, "linear partition of unity at p=" + std::to_string(p));
    }

    // ---- PCHIP: interpolation, affine reproduction, monotonicity, derivative ----
    {
        PCHIP pc;
        pc.build(knots, ya);
        for (int i = 0; i < n; ++i)
            expect_near(pc.eval(knots[i]), ya[i], 1e-12, "PCHIP hits knot " + std::to_string(i));
        for (double p : {0.2, 1.1, 2.4, 3.7})
            expect_near(pc.eval(p), affine(p), 1e-12, "PCHIP reproduces affine");
    }
    {
        // monotone increasing, concave-ish data -> monotone interpolant, no overshoot
        std::vector<double> vm(n);
        for (int i = 0; i < n; ++i) vm[i] = std::sqrt(knots[i] + 0.1);
        PCHIP pc;
        pc.build(knots, vm);
        double prev = pc.eval(knots[0]);
        bool mono = true;
        double pmax = vm[n - 1] + 1e-12, pmin = vm[0] - 1e-12;
        for (int s = 1; s <= 400; ++s) {
            const double p = knots[0] + (knots[n - 1] - knots[0]) * s / 400.0;
            const double val = pc.eval(p);
            if (val < prev - 1e-12) mono = false;
            if (val > pmax + 1e-9 || val < pmin - 1e-9) mono = false;  // no overshoot
            prev = val;
        }
        expect_true(mono, "PCHIP monotone + overshoot-free on monotone data");

        // analytic derivative vs central finite difference
        const double eps = 1e-6;
        for (double p : {0.3, 1.0, 2.2, 3.3}) {
            const double fd = (pc.eval(p + eps) - pc.eval(p - eps)) / (2.0 * eps);
            expect_near(pc.deriv(p), fd, 1e-5, "PCHIP deriv vs FD at p=" + std::to_string(p));
        }
    }

    return summary("test_spline");
}
