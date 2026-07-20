// payoff.hpp — per-step net payoff, FOC seed, and a globally-robust 1-D inner control.
//
// pi(q) = q*(S-K)^+ - c*q^gamma  (the env's net payoff before the profitability gate).
// Bellman inner step maximizes  g(q) = pi(q) + df * W(Q+q)  over q in [0, q_cap].
//
// In exact arithmetic g is concave (pi concave for gamma>=1, W concave in Q', df>0).  But
// the continuation comes from W = A_X U A_Y^T with *signed* cubic-spline transfer weights,
// so the grid continuation is only "almost" concave and its PCHIP wiggles slightly between
// nodes — enough to fool a monotone-g' root finder and inject a per-step bias that COMPOUNDS
// over the backward recursion.  We therefore use a globally-robust maximizer: scan the
// continuation at exact grid nodes (no between-node interp error) to bracket the global
// optimum, then golden-section polish the winning cell on the PCHIP.  This converges to the
// genuine interpolation error (-> 0 with refinement) instead of a compounding optimizer bias.
//
// The profitability gate is enforced by comparing the optimum against q=0: since W is
// non-increasing in Q', any q>0 that beats q=0 necessarily has pi(q)>0.
#pragma once

#include "grid_dp/spline.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace grid_dp {

inline double moneyness_pos(double S, double K) { return std::fmax(S - K, 0.0); }

// q^gamma with fast paths for the integer exponents used by the sweep (gamma in {1,1.5,2,3}).
inline double pow_q(double q, double gamma) {
    if (gamma == 2.0) return q * q;
    if (gamma == 1.0) return q;
    if (gamma == 3.0) return q * q * q;
    return std::pow(q, gamma);
}

inline double payoff_net(double q, double mpos, double c, double gamma) {
    return q * mpos - c * pow_q(q, gamma);
}

// Myopic first-order-condition optimum of the immediate payoff (the LSM/RL warm-start):
//   q*(S) = clip( ((S-K)^+/(c*gamma))^{1/(gamma-1)}, q_min, q_max ).
// gamma==1: immediate payoff linear -> bang-bang.  c==0: linear-in-q -> push to q_max.
inline double foc_q(double S, double K, double c, double gamma, double q_min, double q_max) {
    const double mpos = moneyness_pos(S, K);
    if (mpos <= 0.0) return q_min;
    if (gamma == 1.0) return (mpos > c) ? q_max : q_min;
    if (c <= 0.0) return q_max;
    const double q = std::pow(mpos / (c * gamma), 1.0 / (gamma - 1.0));
    return std::min(std::max(q, q_min), q_max);
}

struct InnerResult {
    double q = 0.0;      // optimal lift
    double value = 0.0;  // U_j(x,y,Q) at this state
};

// Golden-section maximizer of f on [a,b] (assumes near-unimodal over the small bracket).
template <class F>
inline InnerResult golden_max(F&& f, double a, double b, int iters = 32) {
    constexpr double gr = 0.6180339887498949;
    double c = b - gr * (b - a), d = a + gr * (b - a);
    double fc = f(c), fd = f(d);
    InnerResult best{a, f(a)};
    auto consider = [&](double q, double v) { if (v > best.value) best = {q, v}; };
    consider(b, f(b));
    consider(c, fc);
    consider(d, fd);
    for (int i = 0; i < iters; ++i) {
        if (fc >= fd) {
            b = d; d = c; fd = fc;
            c = b - gr * (b - a); fc = f(c);
            consider(c, fc);
        } else {
            a = c; c = d; fc = fd;
            d = a + gr * (b - a); fd = f(d);
            consider(d, fd);
        }
    }
    return best;
}

// Terminal step: U_{N-1}(x,y,Q) = max_q pi(q; S),  q in [0, q_cap] (no continuation).
inline InnerResult solve_terminal(double S, double K, double c, double gamma, double q_cap) {
    const double mpos = moneyness_pos(S, K);
    if (mpos <= 0.0 || q_cap <= 0.0) return {0.0, 0.0};  // OTM -> gated to 0
    const double qf = std::min(std::max(foc_q(S, K, c, gamma, 0.0, q_cap), 0.0), q_cap);
    const double v = payoff_net(qf, mpos, c, gamma);
    return v > 0.0 ? InnerResult{qf, v} : InnerResult{0.0, 0.0};  // gate
}

// Scan-only inner control for the exact Q-lattice (bang-bang) mode: payoff linear in q
// (gamma==1 or c==0), gQ spacing == q_max.  Optimal q at lattice states is node-aligned,
// so the maximization uses EXACT node values only — no PCHIP, no sub-cell polish.
inline InnerResult solve_inner_lattice(double S, double K, double c, double gamma,
                                       double q_cap, double df,
                                       const std::vector<double>& gQ,
                                       const std::vector<double>& Wgrid, int l) {
    const double mpos = moneyness_pos(S, K);
    const double base = df * Wgrid[l];
    if (q_cap <= 0.0 || mpos <= 0.0) return {0.0, base};
    const int nQ = static_cast<int>(gQ.size());
    const double Ql = gQ[l];
    double best_q = 0.0, best_v = base;
    for (int m = l + 1; m < nQ; ++m) {
        const double q = gQ[m] - Ql;
        if (q > q_cap + 1e-12) break;
        const double pi = payoff_net(q, mpos, c, gamma);
        if (pi <= 0.0) continue;                     // env profitability gate
        const double v = pi + df * Wgrid[m];
        if (v > best_v) { best_v = v; best_q = q; }
    }
    return {best_q, best_v};
}

// Robust inner control.  Continuation is given by exact grid values Wgrid on knots gQ
// (uniform), plus a PCHIP W for sub-cell polish.  l = index of the current state Q_l=gQ[l].
inline InnerResult solve_inner(double S, double K, double c, double gamma, double q_cap,
                               double df, const std::vector<double>& gQ,
                               const std::vector<double>& Wgrid, const PCHIP& W, int l) {
    const double mpos = moneyness_pos(S, K);
    const double base = df * Wgrid[l];  // value of q=0 (exact node)
    if (q_cap <= 0.0 || mpos <= 0.0) return {0.0, base};  // OTM/full -> no exercise

    // Strict profitability cap: pi(q)>0 only for 0 < q < q_zero.  Restricting the whole search to
    // [0, q_hi] makes the env gate hold by construction, even when the signed-spline continuation is
    // locally non-monotone in Q' (which could otherwise pull the optimum into the pi<=0 region).
    // q_zero solves pi(q)=0, i.e. q^{gamma-1} = mpos/c.  Fast paths avoid a per-node std::pow
    // for the canonical exponents (gamma in {1,2,3}); gamma=2 is the focal case.
    double q_zero;
    if (c <= 0.0) q_zero = q_cap;                              // no cost: pi=q*mpos>0 for q>0
    else if (gamma == 1.0) q_zero = (mpos > c) ? q_cap : 0.0;  // linear: bang-bang threshold
    else if (gamma == 2.0) q_zero = mpos / c;
    else if (gamma == 3.0) q_zero = std::sqrt(mpos / c);
    else q_zero = std::pow(mpos / c, 1.0 / (gamma - 1.0));     // convex: pi crosses 0 here
    const double q_hi = std::min(q_cap, q_zero);
    if (q_hi <= 0.0) return {0.0, base};

    const int nQ = static_cast<int>(gQ.size());
    const double Ql = gQ[l];
    const double Qtop = Ql + q_hi;
    const double h = (nQ > 1) ? (gQ[1] - gQ[0]) : q_hi;
    auto pinet = [&](double q) { return q * mpos - c * pow_q(q, gamma); };
    auto g = [&](double q) { return pinet(q) + df * W.eval(Ql + q); };

    // (1) scan grid-aligned candidates using EXACT node values (immune to interp wiggle)
    double best_q = 0.0, best_v = base;
    int m_star = l, m_last = l;
    for (int m = l; m < nQ; ++m) {
        if (gQ[m] > Qtop + 1e-12) break;
        m_last = m;
        const double q = gQ[m] - Ql;
        const double v = pinet(q) + df * Wgrid[m];
        if (v > best_v) { best_v = v; best_q = q; m_star = m; }
    }
    // (2) parabolic vertex through the winning node and its neighbours (exact node g-values)
    // to localize the sub-cell optimum, then a short golden polish around it.
    double qv = best_q;
    if (m_star > l && m_star < m_last) {
        const double gL = pinet(best_q - h) + df * Wgrid[m_star - 1];
        const double gR = pinet(best_q + h) + df * Wgrid[m_star + 1];
        const double denom = gL - 2.0 * best_v + gR;  // < 0 for a concave triple
        if (denom < 0.0) {
            double delta = 0.5 * h * (gL - gR) / denom;
            qv = std::min(std::max(best_q + delta, 0.0), q_hi);
        }
    }
    const double lo = std::max(0.0, qv - h), hi = std::min(q_hi, qv + h);
    if (hi > lo) {
        const InnerResult pol = golden_max(g, lo, hi, 16);
        if (pol.value > best_v) { best_v = pol.value; best_q = pol.q; }
    }
    // (3) myopic FOC seed + the (off-grid) profitable endpoint
    for (double qcand : {foc_q(S, K, c, gamma, 0.0, q_hi), q_hi}) {
        const double v = g(qcand);
        if (v > best_v) { best_v = v; best_q = qcand; }
    }

    // Profitability gate (env semantics): exercise only if it beats q=0 AND the chosen lift is
    // itself profitable.  The pi>0 check is normally implied by W being non-increasing in Q', but
    // the signed cubic-spline transfer can make the grid continuation locally non-monotone — so we
    // enforce it explicitly to stay bit-consistent with calculate_standardized_reward.
    if (best_q <= 0.0 || best_v <= base || pinet(best_q) <= 0.0) return {0.0, base};
    return {best_q, best_v};
}

} // namespace grid_dp
