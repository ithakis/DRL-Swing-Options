// price_dp — CLI for the grid backward-DP swing-option pricer.
//
//   ./price_dp [--c 0.04 --gamma 2 --nX 121 --nY 97 --nQ 101 --Mx 24 ...] [--csv]
//
// No flags reproduces the v64 focal contract/HHK at the default reference grid.
// Emits a JSON line (or CSV with --csv) of price + per-phase timings.
#include "grid_dp/dp.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace grid_dp;

namespace {
const char* arg_val(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return nullptr;
}
bool has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}
double getd(int argc, char** argv, const char* key, double def) {
    const char* v = arg_val(argc, argv, key);
    return v ? std::atof(v) : def;
}
int geti(int argc, char** argv, const char* key, int def) {
    const char* v = arg_val(argc, argv, key);
    return v ? std::atoi(v) : def;
}
}  // namespace

int main(int argc, char** argv) {
    SwingContract c;
    c.K = getd(argc, argv, "--K", c.K);
    c.maturity = getd(argc, argv, "--T", c.maturity);
    c.n_rights = geti(argc, argv, "--n_rights", c.n_rights);
    c.q_min = getd(argc, argv, "--q_min", c.q_min);
    c.q_max = getd(argc, argv, "--q_max", c.q_max);
    c.Q_min = getd(argc, argv, "--Q_min", c.Q_min);
    c.Q_max = getd(argc, argv, "--Q_max", c.Q_max);
    c.r = getd(argc, argv, "--r", c.r);
    c.c_cost = getd(argc, argv, "--c", c.c_cost);
    c.gamma_cost = getd(argc, argv, "--gamma", c.gamma_cost);
    c.min_refraction = geti(argc, argv, "--refraction", c.min_refraction);

    HHKParams h;
    h.S0 = getd(argc, argv, "--S0", h.S0);
    h.alpha = getd(argc, argv, "--alpha", h.alpha);
    h.sigma = getd(argc, argv, "--sigma", h.sigma);
    h.beta = getd(argc, argv, "--beta", h.beta);
    h.lam = getd(argc, argv, "--lam", h.lam);
    h.mu_J = getd(argc, argv, "--mu_J", h.mu_J);

    GridParams g;
    g.n_X = geti(argc, argv, "--nX", g.n_X);
    g.n_Y = geti(argc, argv, "--nY", g.n_Y);
    g.n_Q = geti(argc, argv, "--nQ", g.n_Q);
    g.X_lo = getd(argc, argv, "--Xlo", g.X_lo);
    g.X_hi = getd(argc, argv, "--Xhi", g.X_hi);
    g.Y_lo = getd(argc, argv, "--Ylo", g.Y_lo);
    g.Y_hi = getd(argc, argv, "--Yhi", g.Y_hi);
    g.M_x = geti(argc, argv, "--Mx", g.M_x);
    g.N_max = geti(argc, argv, "--Nmax", g.N_max);
    g.gl_U = geti(argc, argv, "--glU", g.gl_U);
    g.glag_J = geti(argc, argv, "--glagJ", g.glag_J);
    g.interp_xy = geti(argc, argv, "--interp", g.interp_xy);
    g.n_threads = geti(argc, argv, "--threads", g.n_threads);
    g.lattice_q = has_flag(argc, argv, "--latticeQ") ? 1 : 0;

    if (g.lattice_q) {
        const bool linear_payoff = (c.gamma_cost == 1.0) || (c.c_cost == 0.0);
        const double spacing = c.Q_max / static_cast<double>(g.n_Q - 1);
        if (!linear_payoff || std::fabs(spacing - c.q_max) > 1e-12) {
            std::fprintf(stderr,
                         "error: --latticeQ requires a payoff linear in q (gamma==1 or c==0) "
                         "and nQ == Q_max/q_max + 1 (got gamma=%g c=%g nQ=%d)\n",
                         c.gamma_cost, c.c_cost, g.n_Q);
            return 2;
        }
    }

    if (c.min_refraction != 0) {
        std::fprintf(stderr, "error: --refraction>0 not yet implemented (cooldown axis is a hook)\n");
        return 2;
    }

    const int mc = geti(argc, argv, "--mc", 0);                 // forward-MC paths (0 = off)
    const long mc_seed = (long)getd(argc, argv, "--mc_seed", 999);
    const bool store = mc > 0;  // policy storage only needed for the forward-MC rollout
    const DPResult R = price_backward(c, h, g, store);

    MCResult M;
    if (mc > 0) M = forward_mc(R, c, h, mc, (std::uint64_t)mc_seed);

    if (has_flag(argc, argv, "--csv")) {
        std::printf("c,gamma,nX,nY,nQ,Mx,Nmax,interp,M_y,price,t_build_ms,t_backward_ms,t_total_ms\n");
        std::printf("%.4g,%.4g,%d,%d,%d,%d,%d,%d,%d,%.12g,%.3f,%.3f,%.3f\n",
                    c.c_cost, c.gamma_cost, g.n_X, g.n_Y, g.n_Q, g.M_x, g.N_max, g.interp_xy,
                    R.M_y, R.price, R.t_build_ms, R.t_backward_ms, R.t_total_ms);
    } else {
        std::printf(
            "{\"price\": %.12g, \"c\": %.4g, \"gamma\": %.4g, \"nX\": %d, \"nY\": %d, "
            "\"nQ\": %d, \"Mx\": %d, \"Nmax\": %d, \"glU\": %d, \"glagJ\": %d, "
            "\"interp\": %d, \"M_y\": %d, \"fp64\": %d, "
            "\"t_build_ms\": %.3f, \"t_backward_ms\": %.3f, \"t_total_ms\": %.3f",
            R.price, c.c_cost, c.gamma_cost, g.n_X, g.n_Y, g.n_Q, g.M_x, g.N_max, g.gl_U,
            g.glag_J, g.interp_xy, R.M_y, (int)GRID_DP_FP64, R.t_build_ms, R.t_backward_ms,
            R.t_total_ms);
        if (mc > 0)
            std::printf(", \"mc_price\": %.12g, \"mc_stderr\": %.3e, \"mc_ci_lo\": %.12g, "
                        "\"mc_ci_hi\": %.12g, \"mc_paths\": %d, \"backward_minus_mc\": %.6g",
                        M.price, M.stderr_, M.ci_lo, M.ci_hi, M.n_paths, R.price - M.price);
        std::printf("}\n");
    }
    return 0;
}
