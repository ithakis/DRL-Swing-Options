#include "hhk_sim.hpp"
#include <algorithm>
#include <numeric>
#include <thread>
#include <cmath>

namespace pricer {

// Inverse standard-normal CDF (Acklam's rational approximation, |err| < 1.2e-9).
// Used by array-RQMC to map low-discrepancy points in (0,1) to OU increments.
static double inv_norm_cdf(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double cc[] = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    const double plow = 0.02425, phigh = 1.0 - 0.02425;
    if (p < plow) {
        double q = std::sqrt(-2.0 * std::log(p));
        return (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= phigh) {
        double q = p - 0.5, r = q*q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

// Array-RQMC fill (A1). Advance all chains in lock-step: at each step sort the chains by
// their OU state X[k-1], drive the OU increment of rank-r with a randomly-shifted, rank-
// stratified point z_r = Phi^{-1}(frac((r + U_k)/n)). Jumps stay pseudo-random per chain
// (rare, secondary leg). Single-threaded — generation is ~ms vs the training hot loop.
static void fill_arqmc(const HHKParams& hhk, const SwingContract& c, int n_paths,
                       uint64_t seed, Paths& out) {
    const int T = c.n_rights, n_steps = c.n_steps();
    const double dt = c.dt();
    const double e_m   = std::exp(-hhk.alpha * dt);
    const double sqrt_vm = std::sqrt(hhk.sigma * hhk.sigma * (1.0 - e_m * e_m) / (2.0 * hhk.alpha));
    const double decay_Y = std::exp(-hhk.beta * dt);
    const double logS0 = std::log(hhk.S0);

    Rng rng(seed);
    for (int i = 0; i < n_paths; ++i) {
        out.X[(size_t)i*T] = (Real)logS0; out.Y[(size_t)i*T] = 0; out.S[(size_t)i*T] = (Real)hhk.S0;
    }
    std::vector<int> ord(n_paths);
    const double inv_n = 1.0 / (double)n_paths;
    for (int k = 1; k <= n_steps; ++k) {
        // rank chains by current OU state X[k-1]
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b){
            return out.X[(size_t)a*T + (k-1)] < out.X[(size_t)b*T + (k-1)];
        });
        const double U = rng.uniform();   // common random shift for this step (RQMC randomization)
        for (int r = 0; r < n_paths; ++r) {
            int i = ord[r];
            double frac = (r + U) * inv_n; frac -= std::floor(frac);
            if (frac < 1e-12) frac = 1e-12; if (frac > 1.0 - 1e-12) frac = 1.0 - 1e-12;
            double zk = inv_norm_cdf(frac);
            double xk = e_m * out.X[(size_t)i*T + (k-1)] + sqrt_vm * zk;
            // pseudo-random jumps (non-antithetic)
            double jc = 0.0;
            int cnt = rng.poisson(hhk.lam * dt);
            for (int j = 0; j < cnt; ++j) {
                double Uj = rng.uniform() * dt, V = rng.uniform();
                if (V < 1e-12) V = 1e-12; if (V > 1.0 - 1e-12) V = 1.0 - 1e-12;
                jc += (-hhk.mu_J * std::log(V)) * std::exp(-hhk.beta * (dt - Uj));
            }
            double yk = decay_Y * out.Y[(size_t)i*T + (k-1)] + jc;
            out.X[(size_t)i*T + k] = (Real)xk; out.Y[(size_t)i*T + k] = (Real)yk;
            out.S[(size_t)i*T + k] = (Real)std::exp(xk + yk);
        }
    }
}

void simulate_hhk(const HHKParams& hhk, const SwingContract& c, int n_paths,
                  uint64_t seed, bool stratify, int batch_size, int n_threads,
                  Paths& out, Sampler sampler) {
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

    if (sampler == Sampler::ARQMC) {
        fill_arqmc(hhk, c, n_paths, seed, out);
    } else {

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
    } // end MC branch

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
