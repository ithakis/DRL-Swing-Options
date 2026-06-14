#include "kernel.hpp"
#include <cstdio>
#include <cmath>
#include <stdexcept>

namespace pricer {

void TransitionKernel::finalize() {
    const int Mx = M_x(), My = M_y();
    w_flat.resize((size_t)Mx * My);
    for (int ix = 0; ix < Mx; ++ix)
        for (int iy = 0; iy < My; ++iy)
            w_flat[(size_t)ix*My + iy] = (Real)(w_X[ix] * w_Y[iy]);
}

TransitionKernel TransitionKernel::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("kernel mesh file not found: " + path);
    auto rd_d = [&](double& d){ if (std::fread(&d,sizeof(double),1,f)!=1) throw std::runtime_error("kernel read"); };
    auto rd_i = [&](int32_t& i){ if (std::fread(&i,sizeof(int32_t),1,f)!=1) throw std::runtime_error("kernel read"); };
    int32_t magic; rd_i(magic);
    if (magic != 0x4B524E4C) throw std::runtime_error("bad kernel magic");
    TransitionKernel k;
    rd_d(k.decay_X); rd_d(k.sigma_X); rd_d(k.decay_Y);
    int32_t nr, mx, my; rd_i(nr); rd_i(mx); rd_i(my);
    k.n_rights = nr;
    auto rd_arr = [&](std::vector<double>& v, int n){ v.resize(n); if ((int)std::fread(v.data(),sizeof(double),n,f)!=n) throw std::runtime_error("kernel arr"); };
    rd_arr(k.z_X, mx); rd_arr(k.w_X, mx);
    rd_arr(k.delta_Y, my); rd_arr(k.w_Y, my);
    rd_arr(k.exp_f_grid, nr + 1);
    std::fclose(f);
    k.finalize();
    return k;
}

TransitionKernel TransitionKernel::build_fast(const HHKParams& hhk, const SwingContract& c,
                                              const KernelParams& kp) {
    TransitionKernel k;
    const double dt = c.dt();
    k.decay_X = std::exp(-hhk.alpha * dt);
    double var_X = hhk.sigma*hhk.sigma*(1.0 - k.decay_X*k.decay_X)/(2.0*hhk.alpha);
    k.sigma_X = std::sqrt(std::max(var_X, 0.0));
    k.decay_Y = std::exp(-hhk.beta * dt);
    k.n_rights = c.n_rights;
    // Gauss-Hermite (probabilist) M_x=2: nodes ±1, weights 0.5.
    if (kp.M_x == 2) { k.z_X = {-1.0, 1.0}; k.w_X = {0.5, 0.5}; }
    else throw std::runtime_error("build_fast only supports M_x=2; supply a mesh file");
    double rate = hhk.lam * dt;
    double Edecay = (1.0 - std::exp(-hhk.beta*dt)) / (hhk.beta*dt);  // E[exp(-beta(dt-U))]
    if (kp.N_max <= 0) {
        // H-K1: M_y=1 — fold the *unconditional* mean jump increment into a single
        // deterministic Y node. E[ΔY_jump] = E[N]·E[J]·E[decay] = (λdt)·μ_J·Edecay.
        // Halves the mesh (M=4→2) and the kernel-target GEMM rows.
        double mean_incr = rate * hhk.mu_J * Edecay;
        k.delta_Y = {mean_incr}; k.w_Y = {1.0};
    } else {
        // N_max=1: node0 = no jump; node1 = conditional mean increment.
        double pmf0 = std::exp(-rate), pmf1 = pmf0 * rate;
        double tail = 1.0 - (pmf0 + pmf1); if (tail > 0) pmf1 += tail;
        double d1 = hhk.mu_J * Edecay;
        k.delta_Y = {0.0, d1}; k.w_Y = {pmf0, pmf1};
    }
    k.exp_f_grid.assign(c.n_rights + 1, 1.0);   // f == 0
    k.finalize();
    return k;
}

void expected_critic_target(Actor& actor_t, Critic& critic_t,
                            const Real* states, const Real* next_states, int B,
                            const TransitionKernel& k, double strike,
                            KernelWorkspace& ws, Real* q_next) {
    const int Mx = k.M_x(), My = k.M_y(), M = Mx * My;
    ws.ensure(B * M);
    Real* flat = ws.flat.data();

    for (int b = 0; b < B; ++b) {
        const Real* ns = next_states + (size_t)b * STATE_DIM;
        double X_t = states[(size_t)b*STATE_DIM + 6];
        double Y_t = states[(size_t)b*STATE_DIM + 7];
        double t_norm = ns[4];
        int t_idx = (int)std::lround(t_norm * k.n_rights);
        if (t_idx < 0) t_idx = 0; if (t_idx > k.n_rights) t_idx = k.n_rights;
        double exp_f = k.exp_f_grid[t_idx];
        double mu_x = k.decay_X * X_t;
        double y_base = k.decay_Y * Y_t;
        for (int ix = 0; ix < Mx; ++ix) {
            double x_val = mu_x + k.sigma_X * k.z_X[ix];
            double exp_x = std::exp(x_val);
            for (int iy = 0; iy < My; ++iy) {
                int m = ix*My + iy;
                double y_val = y_base + k.delta_Y[iy];
                double S = exp_f * exp_x * std::exp(y_val);
                Real* row = flat + (size_t)(b*M + m) * STATE_DIM;
                for (int d = 0; d < STATE_DIM; ++d) row[d] = ns[d];   // carry forward
                row[0] = (Real)(S - strike);
                row[5] = (Real)S;
                row[6] = (Real)x_val;
                row[7] = (Real)y_val;
            }
        }
    }

    actor_t.forward(flat, B * M, ws.act.data());
    critic_t.forward(flat, ws.act.data(), B * M, ws.q.data());

    const Real* w = k.w_flat.data();
    for (int b = 0; b < B; ++b) {
        Real acc = 0;
        const Real* qb = ws.q.data() + (size_t)b * M;
        for (int m = 0; m < M; ++m) acc += w[m] * qb[m];
        q_next[b] = acc;
    }
}

} // namespace pricer
