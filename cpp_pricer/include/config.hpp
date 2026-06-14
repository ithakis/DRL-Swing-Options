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

// Convex-cost power q^gamma. gamma is a runtime value, so std::pow cannot be
// const-folded by -ffast-math; special-case the sweep's integer/half exponents
// (gamma in {1,1.5,2,3}) to skip the libm call in the rollout/eval/calibrate paths.
inline double cost_pow(double q, double gamma) {
    if (gamma == 1.0) return q;
    if (gamma == 2.0) return q * q;
    if (gamma == 3.0) return q * q * q;
    return std::pow(q, gamma);
}
// Myopic-FOC inverse exponent: base^(1/(gamma-1)).
inline double foc_qstar(double base, double gamma) {
    const double e = 1.0 / (gamma - 1.0);
    if (e == 1.0) return base;          // gamma == 2
    if (e == 2.0) return base * base;   // gamma == 1.5
    if (e == 0.5) return std::sqrt(base);  // gamma == 3
    return std::pow(base, e);
}

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

// ---- Kernel build sizes (fast M1 default => M=2 after R2/H-K1) -----------
// H-K1 (2026-06-13): N_max=0 folds the jump into its unconditional mean (M_y=1),
// giving M=2. Measured +23.5% learn-step speedup with a TOST-equivalent price
// (paired Δ=-5e-5, 90% CI ⊂ ±0.5%, 15 seeds). Set N_max=1 to reproduce the old
// M=4 mesh. NOTE: this default only affects the build_fast fallback; to adopt in
// the production pipeline, re-export kernel_v64.bin with N_max=0.
struct KernelParams {
    int M_x = 2;
    int N_max = 0;
    int M_per_k = 1;
    int y_mesh_seed = 20260525;
};

} // namespace pricer
