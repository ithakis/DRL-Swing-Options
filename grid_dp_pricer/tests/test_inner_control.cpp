// test_inner_control — the robust inner solve matches a fine brute-force grid argmax of
// g(q) = pi(q) + df*W(Q+q) on the SAME interpolant (isolates the optimizer), across regimes
// incl. gamma=1 bang-bang, OTM gate, budget cap, a deliberately NON-concave continuation
// (the failure mode that broke the naive solver), and the terminal (W=0) case.
#include "grid_dp/payoff.hpp"
#include "test_util.hpp"

#include <cmath>
#include <vector>

using namespace grid_dp;
using namespace gdp_test;

// Env-gated brute-force maximizer of g over [0, q_cap] on the PCHIP (the reference for the
// optimizer): only profitable lifts pi(q)>0 may be exercised, else q=0 (matches solve_inner).
static InnerResult brute(double S, double K, double c, double gamma, double q_cap, double df,
                         const PCHIP& W, double Q) {
    const double mpos = std::fmax(S - K, 0.0);
    const double base = df * W.eval(Q);
    InnerResult best{0.0, base};
    const int M = 400000;
    for (int i = 1; i <= M; ++i) {
        const double q = q_cap * i / M;
        const double pn = payoff_net(q, mpos, c, gamma);
        if (pn <= 0.0) continue;  // env profitability gate
        const double v = pn + df * W.eval(Q + q);
        if (v > best.value) best = {q, v};
    }
    return best;
}

int main() {
    const double Qmax = 20.0, h = 0.2;
    const int nQ = 101;
    std::vector<double> Qk(nQ);
    for (int i = 0; i < nQ; ++i) Qk[i] = h * i;
    const double K = 1.0, df = std::exp(-0.05 * (0.0833 / 21.0));

    auto run_case = [&](const std::vector<double>& Wv, double S, double c, double gamma,
                        double q_cap_req, int l, const std::string& tag) {
        PCHIP W;
        W.build(Qk, Wv);
        const double Ql = Qk[l];
        const double q_cap = std::min(q_cap_req, Qmax - Ql);
        const InnerResult got = solve_inner(S, K, c, gamma, q_cap, df, Qk, Wv, W, l);
        const InnerResult ref = brute(S, K, c, gamma, q_cap, df, W, Ql);
        expect_near(got.value, ref.value, 1e-7, tag + " value");
        expect_near(got.q, ref.q, 2e-3, tag + " q");
    };

    // smooth, concave, decreasing continuation
    std::vector<double> Wsmooth(nQ);
    for (int i = 0; i < nQ; ++i) Wsmooth[i] = 2.0 * std::sqrt(Qmax - Qk[i]);
    run_case(Wsmooth, 1.5, 0.04, 2.0, 2.0, 0, "smooth interior");
    run_case(Wsmooth, 1.2, 0.04, 2.0, 2.0, 40, "smooth mid-budget");
    run_case(Wsmooth, 3.0, 0.10, 3.0, 2.0, 10, "deep ITM steep cost");
    run_case(Wsmooth, 1.05, 0.15, 1.5, 2.0, 90, "near budget cap");
    run_case(Wsmooth, 1.3, 0.0, 1.0, 2.0, 20, "linear no-cost bang-bang");
    run_case(Wsmooth, 1.3, 0.02, 1.0, 2.0, 20, "linear with-cost bang-bang");
    run_case(Wsmooth, 0.9, 0.04, 2.0, 2.0, 0, "OTM gate");
    run_case(Wsmooth, 1.5, 0.04, 2.0, 0.5, 98, "residual budget < q_max");

    // NON-concave continuation (mimics signed-spline-transfer wiggle): the global scan must
    // not be fooled into a local optimum the way a monotone-g' root finder was.
    std::vector<double> Wwig(nQ);
    for (int i = 0; i < nQ; ++i)
        Wwig[i] = 2.0 * std::sqrt(Qmax - Qk[i]) + 0.01 * std::sin(6.0 * Qk[i]);
    run_case(Wwig, 1.5, 0.04, 2.0, 2.0, 0, "wiggly interior");
    run_case(Wwig, 1.4, 0.04, 2.0, 2.0, 30, "wiggly mid");
    run_case(Wwig, 2.0, 0.08, 2.0, 2.0, 50, "wiggly deeper");

    // bang-bang structure for gamma=1
    {
        PCHIP W; W.build(Qk, Wsmooth);
        const InnerResult r = solve_inner(1.3, K, 0.02, 1.0, 2.0, df, Qk, Wsmooth, W, 20);
        expect_true(r.q < 1e-3 || std::fabs(r.q - 2.0) < 1e-3, "gamma=1 bang-bang");
    }
    // OTM gated to q=0
    {
        PCHIP W; W.build(Qk, Wsmooth);
        const InnerResult r = solve_inner(0.8, K, 0.04, 2.0, 2.0, df, Qk, Wsmooth, W, 0);
        expect_near(r.q, 0.0, 1e-12, "OTM gate q=0");
    }

    // Terminal (W=0): max of pi(q); matches FOC clip
    {
        const double S = 1.6, c = 0.04, gamma = 2.0, q_cap = 2.0;
        const InnerResult t = solve_terminal(S, K, c, gamma, q_cap);
        const double qf = foc_q(S, K, c, gamma, 0.0, q_cap);
        const double vref = payoff_net(qf, std::fmax(S - K, 0.0), c, gamma);
        expect_near(t.value, vref, 1e-12, "terminal value == pi(FOC)");
        expect_near(t.q, qf, 1e-12, "terminal q == FOC");
        expect_true(t.value > 0.0, "terminal ITM positive");
        expect_near(solve_terminal(0.7, K, c, gamma, q_cap).value, 0.0, 1e-12, "terminal OTM gate");
    }

    return summary("test_inner_control");
}
