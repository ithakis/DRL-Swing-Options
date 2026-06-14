// mlp.hpp — Actor (9->64->64->64->1) and Critic ((s,a)->64..->1) for v64.
// Hand-derived forward/backward; parameter layout mirrors the PyTorch state_dict
// so exported weights load in a fixed order. STE profitability gate on the actor.
#pragma once
#include "config.hpp"
#include "linalg.hpp"
#include <vector>

namespace pricer {

constexpr int STATE_DIM = 9;
constexpr int H_DEFAULT = 64;    // canonical v64 hidden width
// NOTE (R5/H-A1): hidden width is now a runtime member of Actor/Critic (`H`),
// set at construction from AgentConfig::hidden.  Default 64 reproduces v64 exactly
// (PyTorch parity tests load 64-wide weights).  Smaller widths are C++-native archs
// (train from scratch, no parity), gated by TOST-equivalence in the speedup study.
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
    int H = H_DEFAULT;           // hidden width (runtime; set at construction)
    int n_layers = 3;            // hidden depth (runtime; H-N1/H-N2). 3 = v64 canonical.
    std::vector<Linear> lin;     // n_layers blocks (first 9->H, rest H->H)
    std::vector<LayerNorm> ln;   // n_layers
    Linear fc4;
    // gate params (from contract)
    double strike=1.0, q_min=0.0, q_max=2.0, c_cost=0.04, gamma_cost=2.0;

    explicit Actor(int hidden = H_DEFAULT, int layers = 3);
    void set_gate(double strike_, double qmin, double qmax, double c, double g)
        { strike=strike_; q_min=qmin; q_max=qmax; c_cost=c; gamma_cost=g; }

    // Forward to gated action q_gated[B]. If q_raw_out given, also returns raw (post-squash, pre-gate).
    void forward(const Real* states, int B, Real* q_gated, Real* q_raw_out=nullptr);

    // Forward only to pre-activation u[B] (used by act() so noise is added pre-squash).
    void forward_preact(const Real* states, int B, Real* u);

    // Backward: dq_gated[B] (STE => dq_raw = dq_gated). Accumulates param grads.
    void backward(const Real* dq_gated, int B);

    // R10/H-R2: backward from a gradient defined at the pre-activation u (bypasses the
    // squash/STE). Used by the likelihood-ratio direct-policy trainer, whose REINFORCE
    // gradient is g_u = (return-baseline)·(ε/σ) at the policy mean u. Accumulates param grads.
    void backward_from_u(const Real* du, int B);

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
    std::vector<Real> pre_;                                  // preprocessed input (B*9)
    std::vector<std::vector<Real>> lin_out_, ln_out_, act_;  // n_layers each (B*H): lin, LN, silu out
    std::vector<Real> u_;              // fc4 output (B*1)
    std::vector<Real> qraw_;           // sigmoid(beta*u) (B)
    std::vector<Real> bw_du_, bw_gact_, bw_gln_, bw_glin_;   // backward scratch
    void ensure(int B);
};

// ----------------------------- Critic ------------------------------------
class Critic {
public:
    int H = H_DEFAULT;                     // hidden width (runtime; set at construction)
    int n_layers = 3;                      // hidden depth (>=2): se + al + (n_layers-2) post.
    Linear se_lin; LayerNorm se_ln;        // state encoder 9->H
    Linear al_lin; LayerNorm al_ln;        // action layer (H+1)->H
    std::vector<Linear> pl_lin;            // (n_layers-2) post layers H->H
    std::vector<LayerNorm> pl_ln;          // (n_layers-2)
    Linear fc4;                            // H->1
    double strike=1.0;

    explicit Critic(int hidden = H_DEFAULT, int layers = 3);
    void set_strike(double s){ strike=s; }

    // q[B] = Q(states, actions). caches for backward.
    void forward(const Real* states, const Real* actions, int B, Real* q);

    // R9/H-R1: the last hidden activation (B×H), i.e. the random feature map φ(s,a) that
    // the linear readout fc4 consumes. Valid after forward(). Used by the ELM ridge solve.
    const Real* last_features() const { return pl_act_.empty() ? al_act_.data() : pl_act_.back().data(); }
    void set_readout(const Real* w, Real b) { for (int i=0;i<H;++i) fc4.W[i]=w[i]; fc4.b[0]=b; }

    // Backward from dq[B].  If g_action given, returns grad wrt action[B].
    // H-S6: accum_params=false skips all param-grad accumulation (used by the DPG actor
    // update, where only g_action is consumed and the critic param grads are discarded).
    void backward(const Real* dq, int B, Real* g_action=nullptr, bool accum_params=true);

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
    std::vector<Real> al_lin_out_, al_ln_out_, al_act_;   // B*H
    std::vector<std::vector<Real>> pl_lin_out_, pl_ln_out_, pl_act_;  // (n_layers-2) each, B*H
    std::vector<Real> bw_gplact_, bw_gln_, bw_glin_, bw_galact_, bw_gcat_, bw_gseact_; // backward scratch
    void ensure(int B);
};

} // namespace pricer
