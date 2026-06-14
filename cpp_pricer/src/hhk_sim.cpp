#include "hhk_sim.hpp"
#include <algorithm>
#include <numeric>
#include <thread>
#include <cmath>

namespace pricer {

void simulate_hhk(const HHKParams& hhk, const SwingContract& c, int n_paths,
                  uint64_t seed, bool stratify, int batch_size, int n_threads,
                  Paths& out) {
    const int T = c.n_rights;          // time points 0..n_steps
    const int n_steps = c.n_steps();   // = n_rights - 1
    const double dt = c.dt();

    const double e_m   = std::exp(-hhk.alpha * dt);
    const double var_m = hhk.sigma * hhk.sigma * (1.0 - e_m * e_m) / (2.0 * hhk.alpha);
    const double sqrt_vm = std::sqrt(var_m);
    const double decay_Y = std::exp(-hhk.beta * dt);
    const double logS0 = std::log(hhk.S0);

    out.n_paths = n_paths; out.T = T;
    out.S.assign((size_t)n_paths * T, 0);
    out.X.assign((size_t)n_paths * T, 0);
    out.Y.assign((size_t)n_paths * T, 0);

    const int n_pairs = n_paths / 2;
    if (n_threads < 1) n_threads = 1;

    auto worker = [&](int pair_lo, int pair_hi, uint64_t wseed) {
        Rng rng(wseed);
        std::vector<double> z(n_steps);
        for (int p = pair_lo; p < pair_hi; ++p) {
            const int i0 = 2 * p, i1 = 2 * p + 1;
            Real* X0 = out.X.data() + (size_t)i0 * T; Real* X1 = out.X.data() + (size_t)i1 * T;
            Real* Y0 = out.Y.data() + (size_t)i0 * T; Real* Y1 = out.Y.data() + (size_t)i1 * T;
            Real* S0 = out.S.data() + (size_t)i0 * T; Real* S1 = out.S.data() + (size_t)i1 * T;
            X0[0] = (Real)logS0; X1[0] = (Real)logS0;
            Y0[0] = 0; Y1[0] = 0;
            S0[0] = (Real)hhk.S0; S1[0] = (Real)hhk.S0;
            for (int k = 0; k < n_steps; ++k) z[k] = rng.normal();

            for (int k = 1; k <= n_steps; ++k) {
                // OU diffusion: antithetic (+z, -z)
                double zk = z[k-1];
                double x0 = e_m * X0[k-1] + sqrt_vm * zk;
                double x1 = e_m * X1[k-1] - sqrt_vm * zk;
                X0[k] = (Real)x0; X1[k] = (Real)x1;

                // Jumps: shared Poisson count, antithetic exponential marks.
                double jc0 = 0.0, jc1 = 0.0;
                int cnt = rng.poisson(hhk.lam * dt);
                for (int j = 0; j < cnt; ++j) {
                    double U = rng.uniform() * dt;          // arrival in (0,dt)
                    double V = rng.uniform();
                    if (V < 1e-12) V = 1e-12; if (V > 1.0 - 1e-12) V = 1.0 - 1e-12;
                    double J0 = -hhk.mu_J * std::log(V);
                    double J1 = -hhk.mu_J * std::log(1.0 - V);
                    double decay = std::exp(-hhk.beta * (dt - U));
                    jc0 += J0 * decay;
                    jc1 += J1 * decay;
                }
                double y0 = decay_Y * Y0[k-1] + jc0;
                double y1 = decay_Y * Y1[k-1] + jc1;
                Y0[k] = (Real)y0; Y1[k] = (Real)y1;
                S0[k] = (Real)std::exp(x0 + y0);
                S1[k] = (Real)std::exp(x1 + y1);
            }
        }
    };

    int nthreads = std::min(n_threads, std::max(1, n_pairs));
    if (nthreads == 1) {
        worker(0, n_pairs, seed);
    } else {
        std::vector<std::thread> pool;
        int chunk = (n_pairs + nthreads - 1) / nthreads;
        for (int t = 0; t < nthreads; ++t) {
            int lo = t * chunk, hi = std::min(n_pairs, lo + chunk);
            if (lo >= hi) break;
            // Distinct stream per worker (independent paths => statistically equivalent).
            pool.emplace_back(worker, lo, hi, seed + 0x9E3779B9ULL * (uint64_t)(t + 1));
        }
        for (auto& th : pool) th.join();
    }

    // ---- terminal stratification (systematic reorder by S_T) -------------
    if (stratify && batch_size > 0 && n_paths >= batch_size) {
        std::vector<int> idx(n_paths);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){
            return out.S[(size_t)a*T + (T-1)] < out.S[(size_t)b*T + (T-1)];
        });
        int num_batches = (n_paths + batch_size - 1) / batch_size;
        std::vector<int> order; order.reserve(n_paths);
        for (int b = 0; b < num_batches; ++b)
            for (int j = b; j < n_paths; j += num_batches) order.push_back(idx[j]);

        Paths tmp; tmp.n_paths = n_paths; tmp.T = T;
        tmp.S.resize(out.S.size()); tmp.X.resize(out.X.size()); tmp.Y.resize(out.Y.size());
        for (int i = 0; i < n_paths; ++i) {
            int src = order[i];
            std::copy(out.Srow(src), out.Srow(src)+T, tmp.S.data()+(size_t)i*T);
            std::copy(out.Xrow(src), out.Xrow(src)+T, tmp.X.data()+(size_t)i*T);
            std::copy(out.Yrow(src), out.Yrow(src)+T, tmp.Y.data()+(size_t)i*T);
        }
        out.S.swap(tmp.S); out.X.swap(tmp.X); out.Y.swap(tmp.Y);
    }
}

} // namespace pricer
