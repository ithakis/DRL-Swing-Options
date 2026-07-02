// test_kernel_parity — FP64 parity vs src/transition_kernel.py.
//
// Loads the exact mesh exported by tools/export_kernel_fixture.py and recomputes the
// SAME analytic test integrals; identical float64 inputs + arithmetic must agree to
// <= 1e-10.  This validates the C++ weighted-sum integrator against the Python kernel.
#include "grid_dp/io.hpp"
#include "test_util.hpp"

#include <cmath>
#include <string>

using namespace grid_dp;
using namespace gdp_test;

// Test functions g(X', Y'). MUST match tools/export_kernel_fixture.py exactly.
static double g0(double Xp, double Yp) { return std::fmax(std::exp(Xp + Yp) - 1.0, 0.0); }
static double g1(double Xp, double Yp) { return std::exp(0.5 * Xp) * (1.0 + Yp); }
static double g2(double Xp, double Yp) { return 1.0 / (1.0 + Xp * Xp + Yp); }

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";
    const std::string path = data_dir + "/kernel_fixture.bin";

    KernelFixture k;
    try {
        k = load_kernel_fixture(path);
    } catch (const std::exception& e) {
        std::printf("  [FAIL] %s\n", e.what());
        std::printf("  (run: conda run -n EP11 python tools/export_kernel_fixture.py)\n");
        return 1;
    }

    expect_true(k.M_x > 0 && k.M_y > 0, "fixture has nodes");
    double swx = 0.0, swy = 0.0;
    for (double w : k.w_X) swx += w;
    for (double w : k.w_Y) swy += w;
    expect_near(swx, 1.0, 1e-12, "sum w_X == 1");
    expect_near(swy, 1.0, 1e-12, "sum w_Y == 1");

    double (*funcs[])(double, double) = {g0, g1, g2};
    for (int t = 0; t < static_cast<int>(k.I_ref.size()); ++t) {
        double I = 0.0;
        for (int m = 0; m < k.M_x; ++m) {
            const double Xp = k.decay_X * k.x_base + k.sigma_X * k.z_X[m];
            for (int n = 0; n < k.M_y; ++n) {
                const double Yp = k.decay_Y * k.y_base + k.delta_Y[n];
                I += k.w_X[m] * k.w_Y[n] * funcs[t](Xp, Yp);
            }
        }
        expect_near(I, k.I_ref[t], 1e-10, "kernel integral g" + std::to_string(t));
    }

    return summary("test_kernel_parity");
}
