#include "mlp.hpp"
#include <algorithm>

namespace pricer {

// ---- HHK input preprocessing (median=0, iqr=1 fixed; clip=10) ------------
void hhk_preprocess(const Real* state, Real* out, int B, double strike) {
    const Real K = (Real)strike;
    const Real inv_iqr = (Real)(1.0 / (1.0 + 1e-6));
    for (int r = 0; r < B; ++r) {
        const Real* s = state + (size_t)r * STATE_DIM;
        Real* o = out + (size_t)r * STATE_DIM;
        for (int i = 0; i < STATE_DIM; ++i) o[i] = s[i];
        Real s_safe = std::max(s[5], (Real)1e-4);
        o[5] = std::log(s_safe / K);
        o[6] = std::clamp(s[6] * inv_iqr, (Real)-10, (Real)10);
        o[7] = std::clamp(s[7] * inv_iqr, (Real)-10, (Real)10);
    }
}

// ============================ Actor ======================================
Actor::Actor() {
    lin[0].init(STATE_DIM, H); ln[0].init(H);
    lin[1].init(H, H);         ln[1].init(H);
    lin[2].init(H, H);         ln[2].init(H);
    fc4.init(H, 1);
}

void Actor::ensure(int B) {
    if (B_ == B) return;
    B_ = B;
    pre_.resize((size_t)B * STATE_DIM);
    for (int i = 0; i < 3; ++i) { lin_out_[i].resize((size_t)B*H); ln_out_[i].resize((size_t)B*H); act_[i].resize((size_t)B*H); }
    u_.resize((size_t)B); qraw_.resize((size_t)B);
    bw_du_.resize((size_t)B); bw_gact_.resize((size_t)B*H); bw_gln_.resize((size_t)B*H); bw_glin_.resize((size_t)B*H);
}

static inline Real actor_gate(Real q_raw, Real s_minus_k, double c_cost,
                              double gamma_cost, double q_min, double q_max) {
    if (c_cost == 0.0) return q_raw;
    Real payoff = std::max(s_minus_k, (Real)0);
    if (gamma_cost == 1.0) return (payoff > (Real)c_cost) ? q_raw : (Real)0;
    Real qlim_abs = std::pow(payoff / (Real)std::max(c_cost, 1e-9),
                             (Real)(1.0 / (gamma_cost - 1.0)));
    Real qlim_norm = (qlim_abs - (Real)q_min) / (Real)(q_max - q_min);
    qlim_norm = std::clamp(qlim_norm, (Real)0, (Real)1e9);
    return std::min(q_raw, qlim_norm);
}

void Actor::forward_preact(const Real* states, int B, Real* u) {
    ensure(B);
    hhk_preprocess(states, pre_.data(), B, strike);
    const Real* in = pre_.data(); int in_dim = STATE_DIM;
    for (int i = 0; i < 3; ++i) {
        linear_forward(lin[i].W.data(), lin[i].b.data(), in, lin_out_[i].data(), B, in_dim, H);
        layernorm_forward(lin_out_[i].data(), ln[i].g.data(), ln[i].b.data(), ln_out_[i].data(), B, H, ln[i].cache);
        silu_forward(ln_out_[i].data(), act_[i].data(), B*H);
        in = act_[i].data(); in_dim = H;
    }
    linear_forward(fc4.W.data(), fc4.b.data(), act_[2].data(), u, B, H, 1);
}

void Actor::forward(const Real* states, int B, Real* q_gated, Real* q_raw_out) {
    ensure(B);                       // size u_ before taking its pointer (forward_preact resizes members)
    forward_preact(states, B, u_.data());
    for (int r = 0; r < B; ++r) {
        Real qr = sigmoidf(ACTOR_BETA * u_[r]);
        qraw_[r] = qr;
        if (q_raw_out) q_raw_out[r] = qr;
        q_gated[r] = actor_gate(qr, states[(size_t)r*STATE_DIM+0], c_cost, gamma_cost, q_min, q_max);
    }
}

void Actor::backward(const Real* dq_gated, int B) {
    // STE: gradient flows as if q = q_raw.
    Real* du = bw_du_.data();
    for (int r = 0; r < B; ++r) {
        Real qr = qraw_[r];
        du[r] = dq_gated[r] * ACTOR_BETA * qr * (Real(1) - qr);   // d sigmoid(beta*u)/du
    }
    Real* g_act = bw_gact_.data(); Real* g_ln = bw_gln_.data(); Real* g_lin = bw_glin_.data();
    // fc4
    linear_backward(fc4.W.data(), act_[2].data(), du, g_act, fc4.gW.data(), fc4.gb.data(), B, H, 1);
    for (int i = 2; i >= 0; --i) {
        silu_backward(ln_out_[i].data(), g_act, g_ln, B*H);
        std::fill(g_lin, g_lin + (size_t)B*H, Real(0));
        layernorm_backward(g_ln, ln[i].g.data(), ln[i].cache, g_lin, ln[i].gg.data(), ln[i].gb.data(), B, H);
        const Real* layer_in = (i == 0) ? pre_.data() : act_[i-1].data();
        int in_dim = (i == 0) ? STATE_DIM : H;
        Real* g_in = (i == 0) ? nullptr : g_act;   // reuse g_act as next grad-out
        linear_backward(lin[i].W.data(), layer_in, g_lin, g_in, lin[i].gW.data(), lin[i].gb.data(), B, in_dim, H);
    }
}

void Actor::zero_grad() {
    for (int i = 0; i < 3; ++i) { lin[i].zero_grad(); ln[i].zero_grad(); }
    fc4.zero_grad();
}

void Actor::collect_params(std::vector<ParamRef>& o, double wd) {
    for (int i = 0; i < 3; ++i) {
        o.push_back({lin[i].W.data(), lin[i].gW.data(), (int)lin[i].W.size(), wd});
        o.push_back({lin[i].b.data(), lin[i].gb.data(), (int)lin[i].b.size(), 0});
        o.push_back({ln[i].g.data(), ln[i].gg.data(), (int)ln[i].g.size(), 0});
        o.push_back({ln[i].b.data(), ln[i].gb.data(), (int)ln[i].b.size(), 0});
    }
    o.push_back({fc4.W.data(), fc4.gW.data(), (int)fc4.W.size(), wd});
    o.push_back({fc4.b.data(), fc4.gb.data(), (int)fc4.b.size(), 0});
}

void Actor::load_flat(const double* p) {
    size_t k = 0;
    auto rd = [&](std::vector<Real>& v){ for (auto& x : v) x = (Real)p[k++]; };
    for (int i = 0; i < 3; ++i) { rd(lin[i].W); rd(lin[i].b); rd(ln[i].g); rd(ln[i].b); }
    rd(fc4.W); rd(fc4.b);
}

size_t Actor::num_scalars() const {
    size_t n = 0;
    for (int i = 0; i < 3; ++i) n += lin[i].W.size()+lin[i].b.size()+ln[i].g.size()+ln[i].b.size();
    n += fc4.W.size()+fc4.b.size();
    return n;
}

void Actor::copy_from(const Actor& o) {
    for (int i = 0; i < 3; ++i) { lin[i].W=o.lin[i].W; lin[i].b=o.lin[i].b; ln[i].g=o.ln[i].g; ln[i].b=o.ln[i].b; }
    fc4.W=o.fc4.W; fc4.b=o.fc4.b;
    strike=o.strike; q_min=o.q_min; q_max=o.q_max; c_cost=o.c_cost; gamma_cost=o.gamma_cost;
}

// ============================ Critic =====================================
Critic::Critic() {
    se_lin.init(STATE_DIM, H); se_ln.init(H);
    al_lin.init(H+1, H);       al_ln.init(H);
    pl_lin.init(H, H);         pl_ln.init(H);
    fc4.init(H, 1);
}

void Critic::ensure(int B) {
    if (B_ == B) return;
    B_ = B;
    pre_.resize((size_t)B*STATE_DIM);
    se_lin_out_.resize((size_t)B*H); se_ln_out_.resize((size_t)B*H); se_act_.resize((size_t)B*H);
    cat_.resize((size_t)B*(H+1));
    al_lin_out_.resize((size_t)B*H); al_ln_out_.resize((size_t)B*H); al_act_.resize((size_t)B*H);
    pl_lin_out_.resize((size_t)B*H); pl_ln_out_.resize((size_t)B*H); pl_act_.resize((size_t)B*H);
    bw_gplact_.resize((size_t)B*H); bw_gln_.resize((size_t)B*H); bw_glin_.resize((size_t)B*H);
    bw_galact_.resize((size_t)B*H); bw_gcat_.resize((size_t)B*(H+1)); bw_gseact_.resize((size_t)B*H);
}

void Critic::forward(const Real* states, const Real* actions, int B, Real* q) {
    ensure(B);
    hhk_preprocess(states, pre_.data(), B, strike);
    linear_forward(se_lin.W.data(), se_lin.b.data(), pre_.data(), se_lin_out_.data(), B, STATE_DIM, H);
    layernorm_forward(se_lin_out_.data(), se_ln.g.data(), se_ln.b.data(), se_ln_out_.data(), B, H, se_ln.cache);
    silu_forward(se_ln_out_.data(), se_act_.data(), B*H);
    // concat [se_act, action]
    for (int r = 0; r < B; ++r) {
        Real* c = cat_.data() + (size_t)r*(H+1);
        const Real* a = se_act_.data() + (size_t)r*H;
        for (int i = 0; i < H; ++i) c[i] = a[i];
        c[H] = actions[r];
    }
    linear_forward(al_lin.W.data(), al_lin.b.data(), cat_.data(), al_lin_out_.data(), B, H+1, H);
    layernorm_forward(al_lin_out_.data(), al_ln.g.data(), al_ln.b.data(), al_ln_out_.data(), B, H, al_ln.cache);
    silu_forward(al_ln_out_.data(), al_act_.data(), B*H);
    linear_forward(pl_lin.W.data(), pl_lin.b.data(), al_act_.data(), pl_lin_out_.data(), B, H, H);
    layernorm_forward(pl_lin_out_.data(), pl_ln.g.data(), pl_ln.b.data(), pl_ln_out_.data(), B, H, pl_ln.cache);
    silu_forward(pl_ln_out_.data(), pl_act_.data(), B*H);
    linear_forward(fc4.W.data(), fc4.b.data(), pl_act_.data(), q, B, H, 1);
}

void Critic::backward(const Real* dq, int B, Real* g_action) {
    Real* g_plact = bw_gplact_.data(); Real* g_ln = bw_gln_.data(); Real* g_lin = bw_glin_.data();
    Real* g_alact = bw_galact_.data(); Real* g_cat = bw_gcat_.data(); Real* g_seact = bw_gseact_.data();
    // fc4
    linear_backward(fc4.W.data(), pl_act_.data(), dq, g_plact, fc4.gW.data(), fc4.gb.data(), B, H, 1);
    // post layer  (SiLU input is the LayerNorm output)
    silu_backward(pl_ln_out_.data(), g_plact, g_ln, B*H);
    std::fill(g_lin, g_lin + (size_t)B*H, Real(0));
    layernorm_backward(g_ln, pl_ln.g.data(), pl_ln.cache, g_lin, pl_ln.gg.data(), pl_ln.gb.data(), B, H);
    linear_backward(pl_lin.W.data(), al_act_.data(), g_lin, g_alact, pl_lin.gW.data(), pl_lin.gb.data(), B, H, H);
    // action layer
    silu_backward(al_ln_out_.data(), g_alact, g_ln, B*H);
    std::fill(g_lin, g_lin + (size_t)B*H, Real(0));
    layernorm_backward(g_ln, al_ln.g.data(), al_ln.cache, g_lin, al_ln.gg.data(), al_ln.gb.data(), B, H);
    linear_backward(al_lin.W.data(), cat_.data(), g_lin, g_cat, al_lin.gW.data(), al_lin.gb.data(), B, H+1, H);
    // split g_cat -> se_act grad + action grad
    for (int r = 0; r < B; ++r) {
        const Real* gc = g_cat + (size_t)r*(H+1);
        Real* gs = g_seact + (size_t)r*H;
        for (int i = 0; i < H; ++i) gs[i] = gc[i];
        if (g_action) g_action[r] = gc[H];
    }
    // state encoder
    silu_backward(se_ln_out_.data(), g_seact, g_ln, B*H);
    std::fill(g_lin, g_lin + (size_t)B*H, Real(0));
    layernorm_backward(g_ln, se_ln.g.data(), se_ln.cache, g_lin, se_ln.gg.data(), se_ln.gb.data(), B, H);
    linear_backward(se_lin.W.data(), pre_.data(), g_lin, nullptr, se_lin.gW.data(), se_lin.gb.data(), B, STATE_DIM, H);
}

void Critic::zero_grad() {
    se_lin.zero_grad(); se_ln.zero_grad(); al_lin.zero_grad(); al_ln.zero_grad();
    pl_lin.zero_grad(); pl_ln.zero_grad(); fc4.zero_grad();
}

void Critic::collect_params(std::vector<ParamRef>& o, double wd) {
    auto add_lin = [&](Linear& L){ o.push_back({L.W.data(),L.gW.data(),(int)L.W.size(),wd}); o.push_back({L.b.data(),L.gb.data(),(int)L.b.size(),0}); };
    auto add_ln  = [&](LayerNorm& N){ o.push_back({N.g.data(),N.gg.data(),(int)N.g.size(),0}); o.push_back({N.b.data(),N.gb.data(),(int)N.b.size(),0}); };
    add_lin(se_lin); add_ln(se_ln);
    add_lin(al_lin); add_ln(al_ln);
    add_lin(pl_lin); add_ln(pl_ln);
    add_lin(fc4);
}

void Critic::load_flat(const double* p) {
    size_t k = 0;
    auto rd = [&](std::vector<Real>& v){ for (auto& x : v) x = (Real)p[k++]; };
    rd(se_lin.W); rd(se_lin.b); rd(se_ln.g); rd(se_ln.b);
    rd(al_lin.W); rd(al_lin.b); rd(al_ln.g); rd(al_ln.b);
    rd(pl_lin.W); rd(pl_lin.b); rd(pl_ln.g); rd(pl_ln.b);
    rd(fc4.W); rd(fc4.b);
}

size_t Critic::num_scalars() const {
    return se_lin.W.size()+se_lin.b.size()+se_ln.g.size()+se_ln.b.size()
         + al_lin.W.size()+al_lin.b.size()+al_ln.g.size()+al_ln.b.size()
         + pl_lin.W.size()+pl_lin.b.size()+pl_ln.g.size()+pl_ln.b.size()
         + fc4.W.size()+fc4.b.size();
}

void Critic::copy_from(const Critic& o) {
    se_lin.W=o.se_lin.W; se_lin.b=o.se_lin.b; se_ln.g=o.se_ln.g; se_ln.b=o.se_ln.b;
    al_lin.W=o.al_lin.W; al_lin.b=o.al_lin.b; al_ln.g=o.al_ln.g; al_ln.b=o.al_ln.b;
    pl_lin.W=o.pl_lin.W; pl_lin.b=o.pl_lin.b; pl_ln.g=o.pl_ln.g; pl_ln.b=o.pl_ln.b;
    fc4.W=o.fc4.W; fc4.b=o.fc4.b; strike=o.strike;
}

} // namespace pricer
