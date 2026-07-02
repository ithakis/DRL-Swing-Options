// test_dp_limits — limiting-case / sanity checks for the backward DP (GOAL 2.4):
//   (1) zero-vol deterministic path: the DP equals an exact Lagrangian budget allocation,
//   (2) monotonicities: price up in q_max, Q_max, sigma; down in c, gamma,
//   (3) gamma=1 / c=0 corner is (near) bang-bang: ITM lifts cluster at {0, q_max}.
#include "grid_dp/dp.hpp"
#include "test_util.hpp"

#include <cmath>
#include <vector>

using namespace grid_dp;
using namespace gdp_test;

// Exact optimum for a DETERMINISTIC price path: maximize sum_j w_j (q mpos_j - c q^gamma)
// s.t. 0<=q_j<=q_max, sum q_j <= Q_max  (w_j = df^j).  Separable convex with one budget
// multiplier mu>=0 found by bisection.  Valid for gamma>1.
static double deterministic_reference(const SwingContract& c, const HHKParams& h) {
    const int N = c.n_rights;
    const double dt = c.dt(), df = c.discount_factor();
    const double decay_X = std::exp(-h.alpha * dt);
    std::vector<double> w(N), mpos(N);
    double x = std::log(h.S0);
    for (int j = 0; j < N; ++j) {
        const double S = std::exp(x);
        mpos[j] = std::fmax(S - c.K, 0.0);
        w[j] = std::pow(df, j);
        x *= decay_X;  // next deterministic X
    }
    auto q_at = [&](double mu) {
        std::vector<double> q(N, 0.0);
        for (int j = 0; j < N; ++j) {
            const double num = w[j] * mpos[j] - mu;
            if (num <= 0.0) continue;
            const double val = num / (w[j] * c.c_cost * c.gamma_cost);
            q[j] = std::min(std::pow(val, 1.0 / (c.gamma_cost - 1.0)), c.q_max);
        }
        return q;
    };
    auto total = [&](const std::vector<double>& q) {
        double s = 0;
        for (double v : q) s += v;
        return s;
    };
    std::vector<double> q = q_at(0.0);
    if (total(q) > c.Q_max) {  // budget binds -> bisection on mu
        double lo = 0.0, hi = 1.0;
        while (total(q_at(hi)) > c.Q_max) hi *= 2.0;
        for (int it = 0; it < 200; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (total(q_at(mid)) > c.Q_max) lo = mid; else hi = mid;
        }
        q = q_at(0.5 * (lo + hi));
    }
    double price = 0.0;
    for (int j = 0; j < N; ++j)
        price += w[j] * (q[j] * mpos[j] - c.c_cost * std::pow(q[j], c.gamma_cost));
    return price;
}

static double dp_price(SwingContract c, HHKParams h, GridParams g) {
    return price_backward(c, h, g, false).price;
}

int main() {
    // ---- (1) zero-vol deterministic cross-check ----
    {
        SwingContract c;            // focal contract
        c.c_cost = 0.04; c.gamma_cost = 2.0;
        HHKParams h;
        h.S0 = 1.5; h.sigma = 1e-7; h.lam = 0.0;  // (near) deterministic, no jumps
        GridParams g;
        g.n_X = 161; g.n_Y = 9; g.n_Q = 161; g.M_x = 4; g.N_max = 1;
        g.Y_lo = 0.0; g.Y_hi = 0.5;               // Y stays 0
        const double dp = dp_price(c, h, g);
        const double ref = deterministic_reference(c, h);
        expect_near(dp, ref, 3e-3, "zero-vol DP == Lagrangian reference");
        expect_true(dp > 0.0, "zero-vol price positive");
    }

    // ---- (2) monotonicities (coarse grid for speed) ----
    auto base_grid = []() {
        GridParams g; g.n_X = 81; g.n_Y = 61; g.n_Q = 81; g.M_x = 16; return g;
    };
    {
        SwingContract c; HHKParams h; GridParams g = base_grid();
        const double p0 = dp_price(c, h, g);

        SwingContract c_hi_c = c; c_hi_c.c_cost = 0.10;
        expect_true(dp_price(c_hi_c, h, g) < p0, "price decreasing in c");

        SwingContract c_hi_g = c; c_hi_g.gamma_cost = 3.0;
        expect_true(dp_price(c_hi_g, h, g) < p0, "price decreasing in gamma");

        SwingContract c_hi_Q = c; c_hi_Q.Q_max = 24.0;
        expect_true(dp_price(c_hi_Q, h, g) > p0 - 1e-9, "price non-decreasing in Q_max");

        SwingContract c_hi_qm = c; c_hi_qm.q_max = 2.5;
        expect_true(dp_price(c_hi_qm, h, g) > p0 - 1e-9, "price non-decreasing in q_max");

        HHKParams h_hi_s = h; h_hi_s.sigma = 1.6;
        GridParams g_wide = g; g_wide.X_lo = -2.0; g_wide.X_hi = 2.0;
        expect_true(dp_price(c, h_hi_s, g_wide) > dp_price(c, h, g_wide), "price increasing in sigma");
    }

    // ---- (3) gamma=1 / c=0 corner is (near) bang-bang ----
    {
        SwingContract c; c.c_cost = 0.0; c.gamma_cost = 1.0;
        HHKParams h; GridParams g; g.n_X = 81; g.n_Y = 61; g.n_Q = 81; g.M_x = 16;
        const DPResult R = price_backward(c, h, g, true);
        // examine the date-0 policy slice (q=0 budget level): ITM lifts should be ~0 or ~q_max
        const int j = 0, nQ = g.n_Q, nY = g.n_Y;
        long itm = 0, bang = 0;
        for (int ix = 0; ix < g.n_X; ++ix) {
            for (int iy = 0; iy < nY; ++iy) {
                const double S = std::exp(R.gx[ix] + R.gy[iy]);
                if (S <= c.K + 1e-9) continue;  // OTM
                const double q = R.policy[j][(size_t)ix * nQ * nY + 0 * nY + iy];
                ++itm;
                if (q < 0.05 * c.q_max || q > 0.95 * c.q_max) ++bang;
            }
        }
        expect_true(itm > 0, "bang-bang case has ITM states");
        expect_le(1.0 - double(bang) / double(itm), 0.05, "gamma=1 lifts are >=95% bang-bang");
    }

    return summary("test_dp_limits");
}
