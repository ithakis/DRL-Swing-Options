// hhk_sim.hpp — HHK two-factor OU-with-jumps spot simulation.
// S_t = exp(X_t + Y_t) (f==0). Antithetic pairs on OU driver and jump marks,
// optional terminal stratification (matches the Python variance-reduction design).
#pragma once
#include "config.hpp"
#include "rng.hpp"
#include <vector>
#include <cstdint>

namespace pricer {

struct Paths {
    int n_paths = 0;
    int T = 0;                       // = n_rights = n_steps + 1
    std::vector<Real> S, X, Y;       // row-major (n_paths * T)

    const Real* Srow(int i) const { return S.data() + (size_t)i * T; }
    const Real* Xrow(int i) const { return X.data() + (size_t)i * T; }
    const Real* Yrow(int i) const { return Y.data() + (size_t)i * T; }
};

// Simulate `n_paths` (must be even) paths. Multi-threaded over antithetic pairs.
void simulate_hhk(const HHKParams& hhk, const SwingContract& c, int n_paths,
                  uint64_t seed, bool stratify, int batch_size, int n_threads,
                  Paths& out);

} // namespace pricer
