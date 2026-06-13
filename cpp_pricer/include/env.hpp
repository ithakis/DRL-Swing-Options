// env.hpp — SwingOptionEnv dynamics (observation, feasibility, reward).
// Header-only; shared by the training episode loop and the vectorized rollout.
#pragma once
#include "config.hpp"
#include "mlp.hpp"
#include <algorithm>
#include <cmath>

namespace pricer {

struct EpisodeState {
    int step = 0;
    double q_exercised = 0.0;
    int last_exercise_step = -1;
};

// Build the 9-dim observation at ep.step on a given path (Srow/Xrow/Yrow length n_rights).
inline void build_obs(const SwingContract& c, const Real* Srow, const Real* Xrow,
                      const Real* Yrow, const EpisodeState& ep, Real* o) {
    int step = ep.step;
    if (step >= c.n_rights) step = c.n_rights - 1;
    double S = Srow[step];
    double q_rem = c.Q_max - ep.q_exercised;
    double ttm = (c.n_rights - step) * c.dt();
    double norm_t = (double)step / c.n_rights;
    double days_since = (ep.last_exercise_step >= 0) ? (step - ep.last_exercise_step) : step;
    o[0] = (Real)(S - c.strike);
    o[1] = (Real)(ep.q_exercised / c.Q_max);
    o[2] = (Real)(q_rem / c.Q_max);
    o[3] = (Real)(ttm / c.maturity);
    o[4] = (Real)norm_t;
    o[5] = (Real)S;
    o[6] = (Real)Xrow[step];
    o[7] = (Real)Yrow[step];
    o[8] = (Real)(days_since / (double)c.n_rights);
}

// Feasible exercise from a proposed contract-unit quantity (matches _get_feasible_action).
inline double feasible_action(const SwingContract& c, double q_proposed, const EpisodeState& ep) {
    double q = std::clamp(q_proposed, c.q_min, c.q_max);
    q = std::min(q, c.Q_max - ep.q_exercised);
    if (c.min_refraction_periods > 0 && ep.last_exercise_step >= 0 &&
        (ep.step - ep.last_exercise_step) <= c.min_refraction_periods)
        q = 0.0;
    int remaining = c.n_rights - ep.step - 1;
    if (remaining > 0) {
        double min_needed = std::max(0.0, c.Q_min - ep.q_exercised - q);
        double max_later = c.q_max * remaining;
        if (min_needed > max_later) {
            double req = min_needed - max_later;
            q = std::max(q, req);
            q = std::min(q, c.q_max);
        }
    }
    return std::max(0.0, q);
}

// One env step given normalized action a in [0,1]. Returns discounted reward and
// advances ep. (Net-profitability gate applied: q forced to 0 if net<=0.)
inline double env_step(const SwingContract& c, const Real* Srow, double a, EpisodeState& ep,
                       bool* terminated = nullptr) {
    double q_proposed = c.denormalize_action(a);
    double q = feasible_action(c, q_proposed, ep);
    int step = ep.step;
    double spot = Srow[step];
    double payoff = std::max(spot - c.strike, 0.0) * q;
    double cost = c.c_cost * std::pow(q, c.gamma_cost);
    double net = payoff - cost;
    double reward;
    if (net <= 0.0) { q = 0.0; net = 0.0; reward = 0.0; }
    else reward = std::pow(c.discount_factor(), step) * net;
    if (q > 1e-6) ep.last_exercise_step = step;
    ep.q_exercised += q;
    ep.step += 1;
    bool term = (ep.step >= c.n_rights) || (ep.q_exercised >= c.Q_max - 1e-6);
    if (terminated) *terminated = term;
    return reward;
}

} // namespace pricer
