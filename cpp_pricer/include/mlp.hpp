// mlp.hpp — Actor (9->64->64->64->1) and Critic ((s,a)->64..->1) for v64.
// Hand-derived forward/backward; parameter layout mirrors the PyTorch state_dict
// so exported weights load in a fixed order. STE profitability gate on the actor.
#pragma once
#include "config.hpp"
#include "linalg.hpp"
#include <vector>

namespace pricer {

constexpr int STATE_DIM = 9;
constexpr int H = 64;            // hidden width
constexpr Real ACTOR_BETA = static_cast<Real>(1.5);  // beta_sigmoid_1.5

// A weight-decayable parameter block (Adam iterates over these).
struct ParamRef { Real* data; Real* grad; int n; double weight_decay; };

struct Linear {
    int in, out;
    std::vector<Real> W, b, gW, gb;
    void init(int in_, int out_) {
        in = in_; out = out_;
        W.assign((size_t)out * in, 0); b.assign(out, 0);
        gW.assign((size_t)out * in, 0); gb.assign(out, 0);
    }
    void zero_grad() { std::fill(gW.begin(), gW.end(), Real(0)); std::fill(gb.begin(), gb.end(), Real(0)); }
};

struct LayerNorm {
    int D;
    std::vector<Real> g, b, gg, gb;
    LNCache cache;
    void init(int D_) { D = D_; g.assign(D, 1); b.assign(D, 0); gg.assign(D, 0); gb.assign(D, 0); }
    void zero_grad() { std::fill(gg.begin(), gg.end(), Real(0)); std::fill(gb.begin(), gb.end(), Real(0)); }
};

// HHK input preprocessing (fixed: log-moneyness at idx5, clamp X,Y at idx6,7).
void hhk_preprocess(const Real* state, Real* out, int B, double strike);

// ----------------------------- Actor -------------------------------------
class Actor {
public:
    Linear lin[3];
    LayerNorm ln[3];
    Linear fc4;
    // gate params (from contract)
    double strike=1.0, q_min=0.0, q_max=2.0, c_cost=0.04, gamma_cost=2.0;

    Actor();
    void set_gate(double strike_, double qmin, double qmax, double c, double g)
        { strike=strike_; q_min=qmin; q_max=qmax; c_cost=c; gamma_cost=g; }

    // Forward to gated action q_gated[B]. If q_raw_out given, also returns raw (post-squash, pre-gate).
    void forward(const Real* states, int B, Real* q_gated, Real* q_raw_out=nullptr);

    // Forward only to pre-activation u[B] (used by act() so noise is added pre-squash).
    void forward_preact(const Real* states, int B, Real* u);

    // Backward: dq_gated[B] (STE => dq_raw = dq_gated). Accumulates param grads.
    void backward(const Real* dq_gated, int B);

    void zero_grad();
    void collect_params(std::vector<ParamRef>& out, double wd);
    void load_flat(const double* p);                 // load weights in canonical order
    size_t num_scalars() const;
    void copy_from(const Actor& o);                  // copy weights (target init / EMA swap)

    // weight access for calibrate_bias (fc4)
    Real* fc4_weight() { return fc4.W.data(); }
    Real* fc4_bias()   { return fc4.b.data(); }

private:
    // forward caches (sized to B on demand)
    int B_=0;
    std::vector<Real> pre_;            // preprocessed input (B*9)
    std::vector<Real> lin_out_[3];     // linear outputs (B*64)
    std::vector<Real> ln_out_[3];      // layernorm outputs (B*64) = silu input
    std::vector<Real> act_[3];         // silu outputs (B*64)
    std::vector<Real> u_;              // fc4 output (B*1)
    std::vector<Real> qraw_;           // sigmoid(beta*u) (B)
    std::vector<Real> bw_du_, bw_gact_, bw_gln_, bw_glin_;   // backward scratch
    void ensure(int B);
};

// ----------------------------- Critic ------------------------------------
class Critic {
public:
    Linear se_lin; LayerNorm se_ln;        // state encoder 9->64
    Linear al_lin; LayerNorm al_ln;        // action layer 65->64
    Linear pl_lin; LayerNorm pl_ln;        // 1 post layer 64->64
    Linear fc4;                            // 64->1
    double strike=1.0;

    Critic();
    void set_strike(double s){ strike=s; }

    // q[B] = Q(states, actions). caches for backward.
    void forward(const Real* states, const Real* actions, int B, Real* q);

    // Backward from dq[B].  If g_action given, returns grad wrt action[B].
    // Param grads are accumulated (discarded by caller during actor update).
    void backward(const Real* dq, int B, Real* g_action=nullptr);

    void zero_grad();
    void collect_params(std::vector<ParamRef>& out, double wd);
    void load_flat(const double* p);
    size_t num_scalars() const;
    void copy_from(const Critic& o);

private:
    int B_=0;
    std::vector<Real> pre_;            // preprocessed state (B*9)
    std::vector<Real> se_lin_out_, se_ln_out_, se_act_;   // B*64
    std::vector<Real> cat_;            // B*65  [se_act, action]
    std::vector<Real> al_lin_out_, al_ln_out_, al_act_;   // B*64
    std::vector<Real> pl_lin_out_, pl_ln_out_, pl_act_;   // B*64
    std::vector<Real> bw_gplact_, bw_gln_, bw_glin_, bw_galact_, bw_gcat_, bw_gseact_; // backward scratch
    void ensure(int B);
};

} // namespace pricer
