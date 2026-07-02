// io.hpp — small binary/JSON helpers: load the Python kernel-parity fixture and
// emit pricer results. No external dependencies.
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace grid_dp {

// Mesh + reference integrals exported by tools/export_kernel_fixture.py from
// src/transition_kernel.py.  Used only by the parity test (not at runtime).
struct KernelFixture {
    int M_x = 0, M_y = 0;
    double decay_X = 0, sigma_X = 0, decay_Y = 0;
    double x_base = 0, y_base = 0;
    std::vector<double> z_X, w_X, delta_Y, w_Y;
    std::vector<double> I_ref;  // reference integral per test function
};

inline KernelFixture load_kernel_fixture(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("load_kernel_fixture: cannot open " + path);
    auto rd_i32 = [&](const char* what) {
        std::int32_t v = 0;
        if (std::fread(&v, sizeof(v), 1, f) != 1)
            throw std::runtime_error(std::string("fixture read failed: ") + what);
        return v;
    };
    auto rd_f64 = [&](const char* what) {
        double v = 0;
        if (std::fread(&v, sizeof(v), 1, f) != 1)
            throw std::runtime_error(std::string("fixture read failed: ") + what);
        return v;
    };
    auto rd_vec = [&](int n, const char* what) {
        std::vector<double> v(n);
        if (n > 0 && std::fread(v.data(), sizeof(double), n, f) != static_cast<size_t>(n))
            throw std::runtime_error(std::string("fixture read failed: ") + what);
        return v;
    };

    const std::int32_t magic = rd_i32("magic");
    if (magic != 0x47444B31)  // "GDK1"
        throw std::runtime_error("load_kernel_fixture: bad magic");
    KernelFixture k;
    k.M_x = rd_i32("M_x");
    k.M_y = rd_i32("M_y");
    k.decay_X = rd_f64("decay_X");
    k.sigma_X = rd_f64("sigma_X");
    k.decay_Y = rd_f64("decay_Y");
    k.x_base = rd_f64("x_base");
    k.y_base = rd_f64("y_base");
    k.z_X = rd_vec(k.M_x, "z_X");
    k.w_X = rd_vec(k.M_x, "w_X");
    k.delta_Y = rd_vec(k.M_y, "delta_Y");
    k.w_Y = rd_vec(k.M_y, "w_Y");
    const std::int32_t n_tests = rd_i32("n_tests");
    k.I_ref = rd_vec(n_tests, "I_ref");
    std::fclose(f);
    return k;
}

} // namespace grid_dp
