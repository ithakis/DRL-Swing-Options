// hhk_sim.hpp — native HHK path simulation for forward-MC self-consistency.
//
// Samples the SAME exact one-step kernel the DP integrates by quadrature:
//   X_k = decay_X X_{k-1} + sigma_X Z,           Z ~ N(0,1)
//   Y_k = decay_Y Y_{k-1} + sum_i J_i e^{-beta(dt-U_i)},  #i ~ Poisson(lam*dt),
//                                                 U_i ~ Unif(0,dt), J_i ~ Exp(mean mu_J)
// So the MC of any fixed (DP-derived) policy is an unbiased estimate of that policy's value
// and a true lower bound on the optimum — it should approach the backward U_0 from below as
// the grid/quadrature refine.  RNG is splitmix64 -> xoshiro256** (independent of the Python
// Sobol stream; the self-consistency bar is statistical, not bit-identical).
#pragma once

#include "grid_dp/config.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace grid_dp {

class Xoshiro {
public:
    explicit Xoshiro(std::uint64_t seed) {
        std::uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < 4; ++i) {
            z += 0x9E3779B97F4A7C15ULL;
            std::uint64_t t = z;
            t = (t ^ (t >> 30)) * 0xBF58476D1CE4E5B9ULL;
            t = (t ^ (t >> 27)) * 0x94D049BB133111EBULL;
            s_[i] = t ^ (t >> 31);
        }
    }
    std::uint64_t next_u64() {
        const std::uint64_t r = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
        s_[2] ^= t; s_[3] = rotl(s_[3], 45);
        return r;
    }
    // uniform in (0,1)
    double uniform() { return (next_u64() >> 11) * (1.0 / 9007199254740992.0) + 1e-300; }
    double normal() {
        // Box–Muller (cache the second deviate)
        if (have_spare_) { have_spare_ = false; return spare_; }
        double u1 = uniform(), u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double th = 6.283185307179586 * u2;
        spare_ = r * std::sin(th);
        have_spare_ = true;
        return r * std::cos(th);
    }
    int poisson(double lambda) {
        // Knuth's algorithm (lambda is tiny here, so this is ~1 iteration)
        const double L = std::exp(-lambda);
        int k = 0;
        double p = 1.0;
        do { ++k; p *= uniform(); } while (p > L);
        return k - 1;
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t s_[4];
    bool have_spare_ = false;
    double spare_ = 0.0;
};

// One simulated HHK path of decision-date factors (X_j, Y_j), j=0..N-1.  S_j = exp(X+Y).
struct HHKPath {
    std::vector<double> X, Y;
};

inline HHKPath simulate_path(const HHKParams& h, double dt, int N, Xoshiro& rng) {
    const double decay_X = std::exp(-h.alpha * dt);
    const double sigma_X = std::sqrt(h.sigma * h.sigma * (1.0 - decay_X * decay_X) /
                                     (2.0 * h.alpha));
    const double decay_Y = std::exp(-h.beta * dt);
    HHKPath p;
    p.X.assign(N, 0.0);
    p.Y.assign(N, 0.0);
    p.X[0] = std::log(h.S0);  // f(0)=0
    p.Y[0] = 0.0;
    for (int k = 1; k < N; ++k) {
        p.X[k] = decay_X * p.X[k - 1] + sigma_X * rng.normal();
        double delta = 0.0;
        const int nj = rng.poisson(h.lam * dt);
        for (int i = 0; i < nj; ++i) {
            const double U = dt * rng.uniform();
            const double J = -h.mu_J * std::log(rng.uniform());  // Exp(mean mu_J)
            delta += J * std::exp(-h.beta * (dt - U));
        }
        p.Y[k] = decay_Y * p.Y[k - 1] + delta;
    }
    return p;
}

} // namespace grid_dp
