// config.hpp — precision switch, problem constants, and parameter structs.
#pragma once
#include <cmath>
#include <cstdint>
#include <string>

namespace pricer {

// Headline price uses float32 (+ fast-math); FP64 build is for parity/gradient checks.
#ifdef CPP_PRICER_FP64
using Real = double;
#else
using Real = float;
#endif

// ---- Swing contract (SwingOption_20 focal, v64) --------------------------
struct SwingContract {
    double q_min = 0.0;
    double q_max = 2.0;
    double Q_min = 0.0;
    double Q_max = 20.0;
    double strike = 1.0;
    double maturity = 0.0833;
    int    n_rights = 22;
    int    min_refraction_periods = 0;
    double r = 0.05;
    double c_cost = 0.04;
    double gamma_cost = 2.0;

    double dt() const { return maturity / (n_rights - 1); }
    double discount_factor() const { return std::exp(-r * dt()); }
    int    n_steps() const { return n_rights - 1; }            // simulation steps
    double denormalize_action(double a) const { return q_min + a * (q_max - q_min); }
};

// ---- HHK two-factor OU-with-jumps process (v64 focal) --------------------
struct HHKParams {
    double S0 = 1.0;
    double alpha = 12.0;   // OU mean reversion
    double sigma = 1.2;    // OU volatility
    double beta = 150.0;   // jump decay
    double lam = 6.0;      // jump intensity / year
    double mu_J = 0.3;     // mean jump size (Exp)
    // f(t) == 0 (no seasonal) throughout.
};

// ---- Kernel build sizes (fast M1 default => M=4) -------------------------
struct KernelParams {
    int M_x = 2;
    int N_max = 1;
    int M_per_k = 1;
    int y_mesh_seed = 20260525;
};

} // namespace pricer
