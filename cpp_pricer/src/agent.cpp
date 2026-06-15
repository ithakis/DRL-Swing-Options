#include "agent.hpp"
#include "env.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>

namespace pricer {

// ---- parameter lerp helpers (soft-update / EMA) --------------------------
static inline void lerp(std::vector<Real>& d, const std::vector<Real>& s, Real keep, Real add) {
    for (size_t i = 0; i < d.size(); ++i) d[i] = keep * d[i] + add * s[i];
}
static void actor_lerp(Actor& d, const Actor& s, Real keep, Real add) {
    for (size_t i = 0; i < s.lin.size(); ++i) { lerp(d.lin[i].W,s.lin[i].W,keep,add); lerp(d.lin[i].b,s.lin[i].b,keep,add);
                                                lerp(d.ln[i].g,s.ln[i].g,keep,add); lerp(d.ln[i].b,s.ln[i].b,keep,add); }
    lerp(d.fc4.W,s.fc4.W,keep,add); lerp(d.fc4.b,s.fc4.b,keep,add);
}
static void critic_lerp(Critic& d, const Critic& s, Real keep, Real add) {
    lerp(d.se_lin.W,s.se_lin.W,keep,add); lerp(d.se_lin.b,s.se_lin.b,keep,add); lerp(d.se_ln.g,s.se_ln.g,keep,add); lerp(d.se_ln.b,s.se_ln.b,keep,add);
    lerp(d.al_lin.W,s.al_lin.W,keep,add); lerp(d.al_lin.b,s.al_lin.b,keep,add); lerp(d.al_ln.g,s.al_ln.g,keep,add); lerp(d.al_ln.b,s.al_ln.b,keep,add);
    for (size_t i = 0; i < s.pl_lin.size(); ++i) { lerp(d.pl_lin[i].W,s.pl_lin[i].W,keep,add); lerp(d.pl_lin[i].b,s.pl_lin[i].b,keep,add);
                                                   lerp(d.pl_ln[i].g,s.pl_ln[i].g,keep,add); lerp(d.pl_ln[i].b,s.pl_ln[i].b,keep,add); }
    lerp(d.fc4.W,s.fc4.W,keep,add); lerp(d.fc4.b,s.fc4.b,keep,add);
}

static inline Real gate(Real q_raw, Real s_minus_k, double c, double g, double qmin, double qmax) {
    if (c == 0.0) return q_raw;
    Real payoff = std::max(s_minus_k, (Real)0);
    if (g == 1.0) return (payoff > (Real)c) ? q_raw : (Real)0;
    Real qa = (Real)foc_qstar((double)payoff/std::max(c,1e-9), g);
    Real qn = std::clamp((qa-(Real)qmin)/(Real)(qmax-qmin), (Real)0, (Real)1e9);
    return std::min(q_raw, qn);
}

// ---- orthogonal-equivalent (He-normal) init ------------------------------
static void he_init(Linear& L, Rng& rng, double gain) {
    double std_ = gain / std::sqrt((double)L.in);
    for (auto& w : L.W) w = (Real)(rng.normal() * std_);
    std::fill(L.b.begin(), L.b.end(), Real(0));
}
static void fc_out_init(Linear& L, Rng& rng, double scale = 3e-3) {
    for (auto& w : L.W) w = (Real)((rng.uniform()*2.0 - 1.0) * scale);
    std::fill(L.b.begin(), L.b.end(), Real(0));
}

// TRUE (semi-)orthogonal init via modified Gram-Schmidt — the dynamical-isometry init
// (Saxe; "All you need is a good init"). The existing he_init is iid-Gaussian (mis-named
// "orthogonal"); this de-correlates the layer's rows/cols so W is norm-preserving.
// For W[out,in]: orthonormalize k=min(out,in) Gaussian vectors in R^{max(out,in)}, place
// them as rows (out<=in) or columns (out>in), then scale by `gain`.
static void orthogonal_init(Linear& L, Rng& rng, double gain) {
    const int out = L.out, in = L.in;
    const int d = std::max(out, in), k = std::min(out, in);
    std::vector<std::vector<double>> v(k, std::vector<double>(d));
    for (int i = 0; i < k; ++i) for (int j = 0; j < d; ++j) v[i][j] = rng.normal();
    // modified Gram-Schmidt
    for (int i = 0; i < k; ++i) {
        for (int p = 0; p < i; ++p) {
            double dot = 0; for (int j = 0; j < d; ++j) dot += v[i][j]*v[p][j];
            for (int j = 0; j < d; ++j) v[i][j] -= dot*v[p][j];
        }
        double nrm = 0; for (int j = 0; j < d; ++j) nrm += v[i][j]*v[i][j];
        nrm = std::sqrt(std::max(nrm, 1e-30));
        for (int j = 0; j < d; ++j) v[i][j] /= nrm;
    }
    std::fill(L.b.begin(), L.b.end(), Real(0));
    if (out <= in)  // rows orthonormal: W[i][:] = gain * v[i]
        for (int i = 0; i < out; ++i) for (int j = 0; j < in; ++j) L.W[(size_t)i*in+j] = (Real)(gain*v[i][j]);
    else            // columns orthonormal: W[:,j] = gain * v[j]
        for (int i = 0; i < out; ++i) for (int j = 0; j < in; ++j) L.W[(size_t)i*in+j] = (Real)(gain*v[j][i]);
}

Agent::Agent(const AgentConfig& cfg, const SwingContract& c, const HHKParams& hhk,
             const TransitionKernel& kernel, uint64_t seed)
    : cfg_(cfg), c_(c), hhk_(hhk), kernel_(kernel), rng_(seed),
      actor_local_ (cfg.hidden_actor  > 0 ? cfg.hidden_actor  : cfg.hidden, cfg.actor_layers),
      actor_target_(cfg.hidden_actor  > 0 ? cfg.hidden_actor  : cfg.hidden, cfg.actor_layers),
      actor_ema_   (cfg.hidden_actor  > 0 ? cfg.hidden_actor  : cfg.hidden, cfg.actor_layers),
      critic_local_ (cfg.hidden_critic > 0 ? cfg.hidden_critic : cfg.hidden, cfg.critic_layers),
      critic_target_(cfg.hidden_critic > 0 ? cfg.hidden_critic : cfg.hidden, cfg.critic_layers) {

    init_orthogonal(seed);

    const double gate_args[5] = {c_.strike, c_.q_min, c_.q_max, c_.c_cost, c_.gamma_cost};
    actor_local_.set_gate(gate_args[0],gate_args[1],gate_args[2],gate_args[3],gate_args[4]);
    actor_target_.set_gate(gate_args[0],gate_args[1],gate_args[2],gate_args[3],gate_args[4]);
    actor_ema_.set_gate(gate_args[0],gate_args[1],gate_args[2],gate_args[3],gate_args[4]);
    critic_local_.set_strike(c_.strike);
    critic_target_.set_strike(c_.strike);

    actor_target_.copy_from(actor_local_);
    actor_ema_.copy_from(actor_local_);
    critic_target_.copy_from(critic_local_);

    std::vector<ParamRef> pa; actor_local_.collect_params(pa, cfg_.wd_a);
    opt_a_.init(pa, cfg_.lr_a, cfg_.b1_a, cfg_.b2_a);
    std::vector<ParamRef> pc; critic_local_.collect_params(pc, cfg_.wd_c);
    opt_c_.init(pc, cfg_.lr_c, cfg_.b1_c, cfg_.b2_c);

    replay_.init(cfg_.max_replay);

    int B = cfg_.batch;
    sb_.resize((size_t)B*STATE_DIM); nsb_.resize((size_t)B*STATE_DIM);
    ab_.resize(B); rb_.resize(B); db_.resize(B);
    qexp_.resize(B); qtgt_.resize(B); qnext_.resize(B);
    apred_.resize(B); qval_.resize(B); dqa_.resize(B); dq_.resize(B); dqv_.resize(B);
}

void Agent::init_orthogonal(uint64_t seed) {
    Rng r(seed ^ 0xABCDEF01ULL);
    const double gain = (cfg_.init_gain > 0) ? cfg_.init_gain : std::sqrt(2.0);   // SiLU default
    const double a_out = (cfg_.actor_out_init  >= 0) ? cfg_.actor_out_init  : 3e-3;
    const double c_out = (cfg_.critic_out_init >= 0) ? cfg_.critic_out_init : 3e-3;
    // H-I3: hidden-layer init method. He (0, current/iid-Gaussian) preserved bit-identical;
    // orthogonal (1) and Xavier (2) only change the hidden weights, not the fc4 / RNG-for-data.
    auto hidden = [&](Linear& L) {
        if      (cfg_.init_method == 1) orthogonal_init(L, r, gain);
        else if (cfg_.init_method == 2) he_init(L, r, std::sqrt(2.0*L.in/(double)(L.in+L.out))); // Xavier: he_init std=gain/sqrt(in)=sqrt(2/(in+out))
        else                            he_init(L, r, gain);
    };
    // Draw order preserved (lin[0..],fc4 ; se,al,pl[0..],fc4) so He (default) is bit-identical to v64.
    for (size_t i = 0; i < actor_local_.lin.size(); ++i) hidden(actor_local_.lin[i]);
    fc_out_init(actor_local_.fc4, r, a_out);
    hidden(critic_local_.se_lin);
    hidden(critic_local_.al_lin);
    for (size_t i = 0; i < critic_local_.pl_lin.size(); ++i) hidden(critic_local_.pl_lin[i]);
    fc_out_init(critic_local_.fc4, r, c_out);
}

void Agent::load_weights(const double* actor_flat, const double* critic_flat) {
    actor_local_.load_flat(actor_flat);
    critic_local_.load_flat(critic_flat);
    actor_target_.copy_from(actor_local_);
    actor_ema_.copy_from(actor_local_);
    critic_target_.copy_from(critic_local_);
}

double Agent::pre_noise_sigma() const {
    int e = std::max(1, episode_count_);
    int plateau = std::max(0, cfg_.noise_plateau);
    double sigma;
    int horizon = std::max(plateau + 1, noise_decay_episodes_);   // linear schedule
    double frac = std::min(1.0, std::max(0.0, (double)(e - plateau) / std::max(1, horizon - plateau)));
    sigma = cfg_.noise_sigma0 + (cfg_.noise_floor - cfg_.noise_sigma0) * frac;
    if (e <= cfg_.critic_warmup && cfg_.warmup_noise_fraction < 1.0) {
        double prog = (double)e / std::max(1, cfg_.critic_warmup);
        double cur = cfg_.warmup_noise_fraction + (1.0 - cfg_.warmup_noise_fraction) * prog;
        sigma *= cur;
    }
    return sigma;
}

Real Agent::act_single(const Real* state, bool add_noise) {
    Real u; actor_local_.forward_preact(state, 1, &u);
    if (add_noise) {
        double sigma = pre_noise_sigma();
        if (sigma > 0) {
            double scale = 1.0 + cfg_.adaptive_noise_scale * std::abs((double)u);
            u += (Real)(rng_.normal() * sigma * scale);
        }
    }
    Real qraw = sigmoidf(ACTOR_BETA * u);
    Real q = gate(qraw, state[0], c_.c_cost, c_.gamma_cost, c_.q_min, c_.q_max);
    return std::clamp(q, (Real)0, (Real)1);
}

#ifdef PRICER_PROFILE
  #define PROF_T0 auto _pt=std::chrono::high_resolution_clock::now()
  #define PROF_ADD(acc) acc += std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-_pt).count()
#else
  #define PROF_T0
  #define PROF_ADD(acc)
#endif

// ---- R9/H-R1: ELM critic readout via ridge regression --------------------
// In-house Cholesky solve of an SPD P×P system (P=H+1 ≤ 65); overwrites A with its
// factor and b with the solution. Returns false if A is not positive-definite.
static bool chol_solve(std::vector<double>& A, std::vector<double>& b, int P) {
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[(size_t)i*P+j];
            for (int k = 0; k < j; ++k) s -= A[(size_t)i*P+k]*A[(size_t)j*P+k];
            if (i == j) { if (s <= 0) return false; A[(size_t)i*P+i] = std::sqrt(s); }
            else A[(size_t)i*P+j] = s / A[(size_t)j*P+j];
        }
    }
    for (int i = 0; i < P; ++i) {                 // forward solve L y = b
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= A[(size_t)i*P+k]*b[k];
        b[i] = s / A[(size_t)i*P+i];
    }
    for (int i = P-1; i >= 0; --i) {              // back solve L^T x = y
        double s = b[i];
        for (int k = i+1; k < P; ++k) s -= A[(size_t)k*P+i]*b[k];
        b[i] = s / A[(size_t)i*P+i];
    }
    return true;
}

void Agent::elm_update_readout() {
    const int B = cfg_.batch;
    const int H = critic_local_.H;
    const int P = H + 1;                                 // [features, bias]
    if ((int)elm_A_.size() != P*P) { elm_A_.assign((size_t)P*P, 0.0); elm_c_.assign(P, 0.0); }
    const Real* phi = critic_local_.last_features();     // B×H, filled by the forward above
    const double rho = cfg_.elm_forget, invB = 1.0 / B;
    for (double& a : elm_A_) a *= rho;                   // exponential forgetting (track moving target)
    for (double& c : elm_c_) c *= rho;
    for (int b = 0; b < B; ++b) {                        // A += (1/B) Σ f fᵀ ,  c += (1/B) Σ f·y
        const Real* p = phi + (size_t)b*H;
        const double y = qtgt_[b];
        for (int i = 0; i < H; ++i) {
            const double fi = invB * (double)p[i];
            double* Ai = elm_A_.data() + (size_t)i*P;
            for (int j = 0; j < H; ++j) Ai[j] += fi * (double)p[j];
            Ai[H] += fi;                                 // feature × bias
            elm_c_[i] += fi * y;
        }
        double* AH = elm_A_.data() + (size_t)H*P;        // bias row
        for (int j = 0; j < H; ++j) AH[j] += invB * (double)p[j];
        AH[H] += invB;                                   // bias × bias
        elm_c_[H] += invB * y;
    }
    std::vector<double> M = elm_A_, w = elm_c_;          // solve (A+λI) w = c on a copy
    const double lam = cfg_.elm_ridge;
    for (int i = 0; i < P; ++i) M[(size_t)i*P+i] += lam;
    if (!chol_solve(M, w, P)) return;                    // keep previous readout if ill-conditioned
    std::vector<Real> wf(H);
    for (int i = 0; i < H; ++i) wf[i] = (Real)w[i];
    critic_local_.set_readout(wf.data(), (Real)w[H]);
}

void Agent::learn_step(bool refresh_target) {
    const int B = cfg_.batch;

    // ---- sample minibatch + kernel target ----
    // H-C1 (stale-target reuse): when refresh_target=false the previous minibatch
    // (sb_/ab_/rb_/nsb_/db_) and its target (qtgt_) are reused for another critic+actor
    // update.  Sound because the target nets are frozen within an interaction's learn
    // group (they only soft-update at end of each step by tau=0.0032), so recomputing
    // would barely move the target — we skip the dominant kernel-target forward instead.
    if (refresh_target) {
    if (cfg_.replay_mode != 0)
        replay_.sample_weighted(B, rng_, sb_.data(), ab_.data(), rb_.data(), nsb_.data(), db_.data());
    else
        replay_.sample(B, rng_, sb_.data(), ab_.data(), rb_.data(), nsb_.data(), db_.data());

    // ---- target ----
    { PROF_T0;
    if (kernel_.M() > 0) {
        expected_critic_target(actor_target_, critic_target_, sb_.data(), nsb_.data(), B,
                               kernel_, c_.strike, kws_, qnext_.data());
    } else {
        actor_target_.forward(nsb_.data(), B, apred_.data());
        critic_target_.forward(nsb_.data(), apred_.data(), B, qnext_.data());
    }
    for (int i = 0; i < B; ++i)
        qtgt_[i] = rb_[i] + (Real)cfg_.gamma * qnext_[i] * (Real(1) - db_[i]);
    PROF_ADD(prof_kernel); }
    }

    // ---- critic step ----
    { PROF_T0;
    if (cfg_.elm_critic) {
        // R9/H-R1: forward fills the random-feature map; ridge-solve the readout in closed form.
        critic_local_.forward(sb_.data(), ab_.data(), B, qexp_.data());
        elm_update_readout();
    } else {
        critic_local_.forward(sb_.data(), ab_.data(), B, qexp_.data());
        for (int i = 0; i < B; ++i) dq_[i] = (Real)(2.0/B) * (qexp_[i] - qtgt_[i]);
        critic_local_.zero_grad();
        critic_local_.backward(dq_.data(), B);
        opt_c_.step();
    }
    PROF_ADD(prof_critic); }

    // ---- actor step (after critic warmup) ----
    if (episode_count_ > cfg_.critic_warmup) {
        PROF_T0;
        actor_local_.forward(sb_.data(), B, apred_.data());
        critic_local_.forward(sb_.data(), apred_.data(), B, qval_.data());
        std::fill(dqv_.begin(), dqv_.end(), (Real)(-1.0/B));   // d(-mean q)/dq
        // H-S6: only the action gradient is consumed here; skip the discarded critic
        // param grads (and the dead state-encoder branch). Bit-identical, cheaper.
        critic_local_.backward(dqv_.data(), B, dqa_.data(), /*accum_params=*/false);
        actor_local_.zero_grad();
        actor_local_.backward(dqa_.data(), B);
        opt_a_.step();
        actor_lerp(actor_target_, actor_local_, (Real)(1.0-cfg_.tau), (Real)cfg_.tau);
        // Eval-actor weight averaging (H-N4): EMA (default) or SWA equal-weight tail average.
        if (cfg_.weight_avg == 1) {                       // SWA
            if (episode_count_ < swa_start_) {
                actor_ema_.copy_from(actor_local_);        // track local until the SWA window opens
            } else if (swa_count_ == 0) {
                actor_ema_.copy_from(actor_local_); swa_count_ = 1;
            } else {
                Real nn = (Real)swa_count_;
                actor_lerp(actor_ema_, actor_local_, nn/(nn+1), (Real)1/(nn+1));
                swa_count_++;
            }
        } else {                                          // EMA (canonical)
            actor_lerp(actor_ema_, actor_local_, (Real)cfg_.ema_decay, (Real)(1.0-cfg_.ema_decay));
        }
        PROF_ADD(prof_actor);
    }
    // ---- critic soft update (always, at end) ----
    { PROF_T0;
    critic_lerp(critic_target_, critic_local_, (Real)(1.0-cfg_.tau), (Real)cfg_.tau);
    PROF_ADD(prof_soft); }
}

// Task 3 (A4): coverage-flattening replay bins the (step, log-spot) plane.
static constexpr int COV_SPOT_BINS = 16;
static inline int cov_spot_bin(double spot) {
    // log-spot mapped to [0,COV_SPOT_BINS) over roughly [exp(-1), exp(1)] ~ S in [0.37, 2.72].
    double u = (std::log(std::max(spot, 1e-6)) + 1.0) / 2.0;   // [0,1] for log-spot in [-1,1]
    int b = (int)(u * COV_SPOT_BINS);
    return std::clamp(b, 0, COV_SPOT_BINS - 1);
}

void Agent::train(const Paths& tp, int n_episodes) {
    noise_decay_episodes_ = n_episodes;
    total_steps_ = 0;
    swa_start_ = (cfg_.swa_start >= 0) ? cfg_.swa_start : (int)(0.75 * n_episodes);  // H-N4
    swa_count_ = 0;
    const int T = c_.n_rights;
    if (cfg_.replay_mode != 0) replay_.enable_weighted();
    if (cfg_.replay_mode == 2) cov_count_.assign((size_t)T * COV_SPOT_BINS, 0);
    Real next_obs[STATE_DIM];
    std::vector<Real> s(STATE_DIM);
    for (int ep = 1; ep <= n_episodes; ++ep) {
        episode_count_ = ep;
        int pidx = (ep - 1) % tp.n_paths;
        const Real* Sr = tp.Srow(pidx); const Real* Xr = tp.Xrow(pidx); const Real* Yr = tp.Yrow(pidx);
        EpisodeState es;
        build_obs(c_, Sr, Xr, Yr, es, s.data());
        while (true) {
            Real a = act_single(s.data(), true);
            EpisodeState before = es;
            bool term=false;
            double reward = env_step(c_, Sr, a, es, &term);
            build_obs(c_, Sr, Xr, Yr, es, next_obs);
            total_steps_++;
            if (cfg_.replay_mode == 1) {                       // A2: time-graded
                double w = std::pow((double)(before.step + 1) / (double)T, cfg_.step_density);
                replay_.add_w(s.data(), a, (Real)reward, next_obs, term, w);
            } else if (cfg_.replay_mode == 2) {                // A4: coverage-flattening
                size_t bin = (size_t)before.step * COV_SPOT_BINS + cov_spot_bin(Sr[before.step]);
                long cnt = ++cov_count_[bin];
                replay_.add_w(s.data(), a, (Real)reward, next_obs, term, 1.0 / std::sqrt((double)cnt));
            } else {
                replay_.add(s.data(), a, (Real)reward, next_obs, term);
            }
            if (replay_.size() >= cfg_.min_replay && replay_.size() > cfg_.batch &&
                (total_steps_ % cfg_.learn_every == 0)) {
                // H-C1: refresh the minibatch+target only on the first inner step when reuse_target.
                for (int k = 0; k < cfg_.learn_number; ++k)
                    learn_step(/*refresh_target=*/!cfg_.reuse_target || k == 0);
            }
            std::copy(next_obs, next_obs+STATE_DIM, s.data());
            (void)before;
            if (term) break;
        }
    }
}

// ---- R10/H-R2: critic-free direct-policy REINFORCE (likelihood-ratio) ----
// Treats the actor's pre-squash exploration as a Gaussian policy on the mean u_θ(s):
// the sampled pre-activation is u' = u + σ_eff·ε, so ∇_θ log π(u') = (ε/σ_eff)·∂u/∂θ.
// REINFORCE with a moving-average return baseline; no critic, no kernel target.
void Agent::train_direct_policy(const Paths& tp, int n_episodes) {
    noise_decay_episodes_ = n_episodes;
    double baseline = 0.0; const double bl_decay = 0.99; bool bl_init = false;
    std::vector<Real> states; std::vector<Real> eps_sig; std::vector<double> rewards;
    std::vector<Real> tmp_u, du;
    Real s[STATE_DIM], u1;
    for (int ep = 1; ep <= n_episodes; ++ep) {
        episode_count_ = ep;
        int pidx = (ep - 1) % tp.n_paths;
        const Real* Sr = tp.Srow(pidx); const Real* Xr = tp.Xrow(pidx); const Real* Yr = tp.Yrow(pidx);
        EpisodeState es; build_obs(c_, Sr, Xr, Yr, es, s);
        states.clear(); eps_sig.clear(); rewards.clear();
        const double sigma = pre_noise_sigma();
        while (true) {
            actor_local_.forward_preact(s, 1, &u1);
            double scale = 1.0 + cfg_.adaptive_noise_scale * std::abs((double)u1);
            double seff = sigma * scale;
            double eps = rng_.normal();
            Real u_sample = u1 + (Real)(seff > 0 ? seff * eps : 0.0);
            Real qraw = sigmoidf(ACTOR_BETA * u_sample);
            Real q = std::clamp(gate(qraw, s[0], c_.c_cost, c_.gamma_cost, c_.q_min, c_.q_max), (Real)0, (Real)1);
            for (int d = 0; d < STATE_DIM; ++d) states.push_back(s[d]);
            eps_sig.push_back((Real)(seff > 0 ? eps / seff : 0.0));   // ∂logπ/∂u at the mean
            bool term = false; double r = env_step(c_, Sr, q, es, &term);
            rewards.push_back(r);
            build_obs(c_, Sr, Xr, Yr, es, s);
            if (term) break;
        }
        int T = (int)rewards.size();
        double R = 0; for (double r : rewards) R += r;            // rewards already discounted
        double adv = bl_init ? (R - baseline) : 0.0;
        baseline = bl_init ? bl_decay * baseline + (1 - bl_decay) * R : R; bl_init = true;
        if (T == 0) continue;
        tmp_u.resize(T); du.resize(T);
        actor_local_.forward_preact(states.data(), T, tmp_u.data());  // repopulate caches at B=T
        for (int t = 0; t < T; ++t) du[t] = (Real)(-(adv / T) * (double)eps_sig[t]);  // ascend J ⇒ descend -J
        actor_local_.zero_grad();
        actor_local_.backward_from_u(du.data(), T);
        opt_a_.step();
        actor_lerp(actor_ema_, actor_local_, (Real)cfg_.ema_decay, (Real)(1.0 - cfg_.ema_decay));
    }
}

EvalResult Agent::evaluate(const Paths& ep, int eval_batch, EvalTrace* trace) {
    const int n = ep.n_paths;
    const int T = c_.n_rights;
    if (eval_batch > n) eval_batch = n;
    std::vector<double> returns(n, 0.0), exercised(n, 0.0);
    int nthreads = std::max(1, cfg_.n_threads);
    if (trace) trace->init(n, T);
    // Bang-bangness (matches rebuild_results_v7): among exercises with full remaining room
    // (q_remaining >= q_max), the fraction with q >= 0.95*q_max. Per-thread accumulators.
    std::vector<long> th_fullcap(nthreads, 0), th_bang(nthreads, 0);

    auto worker = [&](int lo, int hi, Actor actor, int tid) {
        long fullcap = 0, bang = 0;
        for (int start = lo; start < hi; start += eval_batch) {
            int B = std::min(eval_batch, hi - start);
            std::vector<EpisodeState> es(B);
            std::vector<char> done(B, 0);
            std::vector<Real> state((size_t)B*STATE_DIM), qg(B);
            for (int step = 0; step < T; ++step) {
                for (int b = 0; b < B; ++b)
                    build_obs(c_, ep.Srow(start+b), ep.Xrow(start+b), ep.Yrow(start+b), es[b],
                              state.data()+(size_t)b*STATE_DIM);
                actor.forward(state.data(), B, qg.data());
                for (int b = 0; b < B; ++b) {
                    if (done[b]) continue;
                    double q_rem = c_.Q_max - es[b].q_exercised;   // room BEFORE this exercise
                    Real a = std::clamp(qg[b], (Real)0, (Real)1);
                    bool term=false; double q=0, gross=0, cost=0;
                    double r = env_step(c_, ep.Srow(start+b), a, es[b], &term, &q, &gross, &cost);
                    returns[start+b] += r;
                    if (q > 1e-6 && q_rem >= c_.q_max - 1e-6) {
                        ++fullcap;
                        if (q >= 0.95 * c_.q_max) ++bang;
                    }
                    if (trace) {
                        size_t idx = (size_t)(start+b) * T + step;
                        trace->q[idx] = (float)q; trace->reward[idx] = (float)r;
                        trace->cost[idx] = (float)cost; trace->gross[idx] = (float)gross;
                    }
                    if (term) done[b] = 1;
                }
            }
            for (int b = 0; b < B; ++b) exercised[start+b] = es[b].q_exercised;
        }
        th_fullcap[tid] = fullcap; th_bang[tid] = bang;
    };

    if (nthreads == 1) {
        worker(0, n, actor_ema_, 0);
    } else {
        std::vector<std::thread> pool;
        int chunk = (n + nthreads - 1) / nthreads;
        // round chunk to multiple of eval_batch for clean batching
        chunk = ((chunk + eval_batch - 1) / eval_batch) * eval_batch;
        int tid = 0;
        for (int t = 0; t < nthreads; ++t) {
            int lo = t*chunk, hi = std::min(n, lo+chunk);
            if (lo >= hi) break;
            pool.emplace_back(worker, lo, hi, actor_ema_, tid++);
        }
        for (auto& th : pool) th.join();
    }

    double mean=0; for (double x : returns) mean += x; mean /= n;
    double var=0; for (double x : returns) { double d=x-mean; var += d*d; } var /= (n>1?(n-1):1);
    double sd = std::sqrt(var);
    double avg_ex=0; for (double x : exercised) avg_ex += x; avg_ex /= n;
    long fullcap=0, bang=0;
    for (int t=0;t<nthreads;++t){ fullcap+=th_fullcap[t]; bang+=th_bang[t]; }
    double bb = (fullcap > 0) ? (double)bang / (double)fullcap : std::nan("");
    return { mean, 1.96*sd/std::sqrt((double)n), sd, avg_ex, bb };
}

void Agent::bench_prepare(const Paths& tp) {
    // roll random episodes to fill replay beyond min_replay, then arm actor updates.
    Real next_obs[STATE_DIM]; std::vector<Real> s(STATE_DIM);
    int ep = 0;
    while (replay_.size() < cfg_.min_replay * 4 + cfg_.batch + 10) {
        int pidx = (ep++) % tp.n_paths;
        const Real* Sr = tp.Srow(pidx); const Real* Xr = tp.Xrow(pidx); const Real* Yr = tp.Yrow(pidx);
        EpisodeState es; build_obs(c_, Sr, Xr, Yr, es, s.data());
        while (true) {
            Real a = act_single(s.data(), true);
            bool term=false; double r = env_step(c_, Sr, a, es, &term);
            build_obs(c_, Sr, Xr, Yr, es, next_obs);
            replay_.add(s.data(), a, (Real)r, next_obs, term);
            std::copy(next_obs, next_obs+STATE_DIM, s.data());
            if (term) break;
        }
    }
    episode_count_ = cfg_.critic_warmup + 1;   // ensure actor updates run
    noise_decay_episodes_ = 4096;
}

double Agent::bench_learn(int K) {
    // Mirror the train loop's refresh cadence so H-C1 (reuse_target) is benchmarked
    // faithfully: refresh the minibatch+target only at the start of each learn_number group.
    const int ln = std::max(1, cfg_.learn_number);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < K; ++i)
        learn_step(/*refresh_target=*/!cfg_.reuse_target || (i % ln == 0));
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// ---- closed-form myopic warm-start (matches _calibrate_bias_closed_form) -
void Agent::calibrate_bias(const Paths& w) {
    int n = std::min(cfg_.calib_warmup_episodes, w.n_paths);
    auto batch_stats = [&](double& std_q, double& mean_q) {
        // vectorized rollout over n warmup paths; collect normalized exercised q stats.
        std::vector<double> qs; qs.reserve((size_t)n * c_.n_rights);
        double denom = (c_.q_max > c_.q_min) ? (c_.q_max - c_.q_min) : 1.0;
        std::vector<EpisodeState> es(n);
        std::vector<Real> state((size_t)n*STATE_DIM), qg(n);
        for (int step = 0; step < c_.n_rights; ++step) {
            for (int b = 0; b < n; ++b)
                build_obs(c_, w.Srow(b), w.Xrow(b), w.Yrow(b), es[b], state.data()+(size_t)b*STATE_DIM);
            actor_local_.forward(state.data(), n, qg.data());
            for (int b = 0; b < n; ++b) {
                double q_prop = c_.denormalize_action(std::clamp((double)qg[b],0.0,1.0));
                double q = std::clamp(q_prop, c_.q_min, c_.q_max);
                q = std::min(q, c_.Q_max - es[b].q_exercised);
                double spot = w.Srow(b)[std::min(step,c_.n_rights-1)];
                double net = q*std::max(spot-c_.strike,0.0) - c_.c_cost*cost_pow(q,c_.gamma_cost);
                if (net <= 0.0) q = 0.0;
                es[b].q_exercised += q;
                if (q > 1e-6) es[b].last_exercise_step = step;
                es[b].step += 1;
                qs.push_back((q - c_.q_min)/denom);
            }
        }
        double m=0; for(double x:qs) m+=x; m/=qs.size();
        double v=0; for(double x:qs){double d=x-m; v+=d*d;} v/= (qs.size()>1?qs.size()-1:1);
        mean_q=m; std_q=std::sqrt(v);
    };

    // Step 1: variance scaling
    double std_q, mean_q; batch_stats(std_q, mean_q);
    if (std_q > 1e-9) {
        Real sf = (Real)(cfg_.calib_target_std / std_q);
        for (auto& x : actor_local_.fc4.W) x *= sf;
        for (auto& x : actor_target_.fc4.W) x *= sf;
    }
    // Step 2: myopic FOC target mean action
    double denom = (c_.q_max > c_.q_min) ? (c_.q_max - c_.q_min) : 1.0;
    double sum_a = 0; long cnt = 0;
    for (int b = 0; b < n; ++b) {
        const Real* Sr = w.Srow(b);
        for (int k = 0; k < c_.n_rights; ++k) {
            double itm = std::max((double)Sr[k] - c_.strike, 0.0);
            double qstar;
            if (c_.c_cost <= 0.0) qstar = (Sr[k] > c_.strike) ? c_.q_max : c_.q_min;
            else if (std::abs(c_.gamma_cost-1.0) < 1e-9) qstar = (itm > c_.c_cost) ? c_.q_max : c_.q_min;
            else { double cg = std::max(c_.c_cost*c_.gamma_cost, 1e-12);
                   qstar = std::clamp(foc_qstar(itm/cg, c_.gamma_cost), c_.q_min, c_.q_max); }
            sum_a += (qstar - c_.q_min)/denom; ++cnt;
        }
    }
    double a_myopic = cnt ? sum_a/cnt : 0.5;
    double target_q = c_.Q_max / c_.n_rights;
    double target_budget = (target_q - c_.q_min)/denom;
    double a_target = std::min(a_myopic, target_budget);
    a_target = std::clamp(a_target, 0.05, 0.95);
    // Step 3: single bias shift via local squash inversion (beta_sigmoid_1.5)
    double dummy, m0; batch_stats(dummy, m0);
    double mm = std::min(std::max(m0, 1e-3), 1.0-1e-3);
    double slope = std::max((double)ACTOR_BETA * mm * (1.0-mm), 0.05);
    double delta = std::clamp((a_target - m0)/slope, -4.0, 4.0);
    actor_local_.fc4.b[0] += (Real)delta;
    actor_target_.fc4.b[0] += (Real)delta;
    actor_ema_.copy_from(actor_local_);
}

} // namespace pricer
