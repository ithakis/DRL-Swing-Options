// hhk_sim.hpp — HHK two-factor OU-with-jumps spot simulation.
// S_t = exp(X_t + Y_t) (f==0). Antithetic pairs on OU driver and jump marks,
// optional terminal stratification (matches the Python variance-reduction design).
#pragma once
#include "config.hpp"
#include "rng.hpp"
#include <vector>
#include <cstdint>

namespace pricer {

// Scenario generator for the HHK training paths (Task 3 — sample-efficiency study).
//   MC    : forward Monte-Carlo (antithetic OU + jumps, terminal stratification) — v65 canonical.
//   ARQMC : array-RQMC (L'Ecuyer-Lecot-Tuffin) — sort chains by state each step, advance the OU
//           leg with a rank-stratified, randomly-shifted point set (jumps stay pseudo-random).
enum class Sampler { MC, ARQMC };

struct Paths {
    int n_paths = 0;
    int T = 0;                       // = n_rights = n_steps + 1
    std::vector<Real> S, X, Y;       // row-major (n_paths * T)

    const Real* Srow(int i) const { return S.data() + (size_t)i * T; }
    const Real* Xrow(int i) const { return X.data() + (size_t)i * T; }
    const Real* Yrow(int i) const { return Y.data() + (size_t)i * T; }
};

// Simulate `n_paths` (must be even) paths. Multi-threaded over antithetic pairs.
// `sampler` selects the scenario generator; MC (default) is bit-identical to v65.
void simulate_hhk(const HHKParams& hhk, const SwingContract& c, int n_paths,
                  uint64_t seed, bool stratify, int batch_size, int n_threads,
                  Paths& out, Sampler sampler = Sampler::MC);

} // namespace pricer
