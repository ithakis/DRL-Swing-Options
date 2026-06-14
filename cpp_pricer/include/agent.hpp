// agent.hpp — D4PG agent (kernel-on, v64): act, learn, soft-update, EMA,
// closed-form warm-start, training driver, and vectorized 65k evaluation.
#pragma once
#include "config.hpp"
#include "mlp.hpp"
#include "adam.hpp"
#include "replay.hpp"
#include "kernel.hpp"
#include "hhk_sim.hpp"
#include "rng.hpp"
#include <vector>
#include <cstdint>

namespace pricer {

struct AgentConfig {
    double lr_a = 3e-4, lr_c = 6e-4;
    double b1_a = 0.9, b2_a = 0.99, b1_c = 0.85, b2_c = 0.99;
    double wd_a = 5e-5, wd_c = 1e-4;
    int batch = 128;
    double tau = 0.0032;
    double gamma = 1.0;
    int max_replay = 100000, min_replay = 1000;
    int learn_every = 2, learn_number = 2;
    int critic_warmup = 512;
    double noise_sigma0 = 1.30, noise_floor = 0.26;
    int noise_plateau = 0;
    double adaptive_noise_scale = 0.6;
    double warmup_noise_fraction = 0.3;
    double ema_decay = 0.999;
    int calib_warmup_episodes = 1024;
    double calib_target_std = 0.005;
    int n_threads = 1;       // for evaluation/sim parallelism
    int hidden = H_DEFAULT;  // R5/H-A1: actor+critic hidden width (64 = v64 canonical)
    int hidden_actor = -1;   // H-W1: per-net actor width override (-1 => use `hidden`)
    int hidden_critic = -1;  // H-W1: per-net critic width override (-1 => use `hidden`)
    int actor_layers = 3;    // H-N1/H-N2: actor hidden depth (3 = v64 canonical)
    int critic_layers = 3;   // H-N1/H-N2: critic hidden depth, >=2 (3 = v64 canonical)
    bool reuse_target = false; // R6/H-C1: reuse one minibatch+kernel-target across the
                               // learn_number inner updates (1 kernel eval per interaction)
    int  weight_avg = 0;       // H-N4: eval-actor averaging — 0=EMA (default), 1=SWA equal-weight tail avg
    int  swa_start = -1;       // H-N4: episode to begin SWA accumulation (-1 => 0.75*n_episodes)
    // H-I (init): knobs that LayerNorm does NOT normalize away. init_gain sets the orthogonal/He
    // gain (default sqrt(2)); critic_out_init / actor_out_init set the fc4 output-weight scale
    // (default 3e-3).  -1 => keep the canonical default (bit-identical).
    double init_gain = -1;
    double critic_out_init = -1;
    double actor_out_init = -1;
    int    init_method = 0;   // H-I3: hidden-layer init — 0=He(current,iid-Gaussian), 1=TRUE orthogonal
                              // (Gram-Schmidt, dynamical isometry), 2=Xavier/Glorot. fc4 unchanged.
    // R9/H-R1: ELM critic — freeze the critic's hidden layers at random init, learn ONLY the
    // linear readout (fc4) by closed-form ridge regression against the kernel target.
    bool   elm_critic = false;
    double elm_ridge  = 1e-2;   // ridge lambda on the (H+1)x(H+1) normal equations
    double elm_forget = 0.99;   // forgetting factor on the running Phi^T Phi accumulator
};

struct EvalResult { double price; double ci95; double std; double avg_exercised; double bangbang; };

// Optional per-(path,step) capture for the analysis notebooks (NB6). Row-major [path*T + step].
// q = post-gate exercised qty; reward = discounted net; cost = undiscounted c*q^gamma;
// gross = undiscounted q*(S-K)+.  Steps after early termination stay 0 (NB6 reindexes with 0).
struct EvalTrace {
    int n_paths = 0, T = 0;
    std::vector<float> q, reward, cost, gross;
    void init(int n, int t) {
        n_paths = n; T = t;
        size_t sz = (size_t)n * t;
        q.assign(sz, 0.f); reward.assign(sz, 0.f); cost.assign(sz, 0.f); gross.assign(sz, 0.f);
    }
};

class Agent {
public:
    Agent(const AgentConfig& cfg, const SwingContract& c, const HHKParams& hhk,
          const TransitionKernel& kernel, uint64_t seed);

    void calibrate_bias(const Paths& warmup);                 // closed-form warm-start
    void train(const Paths& train_paths, int n_episodes);     // 0 -> n_episodes
    // R10/H-R2: critic-free likelihood-ratio (REINFORCE) direct-policy trainer (Warin-style).
    void train_direct_policy(const Paths& train_paths, int n_episodes);
    EvalResult evaluate(const Paths& eval_paths, int eval_batch, EvalTrace* trace = nullptr);

    // Load PyTorch-exported weights into local+target (for parity tests).
    void load_weights(const double* actor_flat, const double* critic_flat);

    // --- benchmarking hooks (isolate the training hot loop) ---
    void bench_prepare(const Paths& p);    // prefill replay + arm actor updates
    double bench_learn(int K);             // time K learn_steps (seconds)
    double prof_kernel=0, prof_critic=0, prof_actor=0, prof_soft=0;  // accumulated seconds

    Actor& actor_local_ref() { return actor_local_; }
    Actor& actor_eval_ref() { return actor_ema_; }   // EMA/SWA eval actor (used by evaluate + hedging)
    Critic& critic_local_ref() { return critic_local_; }
    Actor& actor_target_ref() { return actor_target_; }
    Critic& critic_target_ref() { return critic_target_; }

private:
    AgentConfig cfg_;
    SwingContract c_;
    HHKParams hhk_;
    TransitionKernel kernel_;
    Rng rng_;

    Actor actor_local_, actor_target_, actor_ema_;
    Critic critic_local_, critic_target_;
    AdamW opt_a_, opt_c_;
    ReplayBuffer replay_;
    KernelWorkspace kws_;

    int episode_count_ = 0;
    int noise_decay_episodes_ = 4096;
    long total_steps_ = 0;
    int swa_start_ = 1 << 30;   // H-N4: effective SWA start episode (set in train())
    long swa_count_ = 0;        // H-N4: # snapshots accumulated into actor_ema_

    // batch scratch
    std::vector<Real> sb_, ab_, rb_, nsb_, db_, qexp_, qtgt_, qnext_, apred_, qval_, dqa_, dq_, dqv_;

    double pre_noise_sigma() const;
    Real act_single(const Real* state, bool add_noise);
    void learn_step(bool refresh_target = true);
    void init_orthogonal(uint64_t seed);

    // R9/H-R1: ELM critic ridge state (running normal equations, double precision).
    std::vector<double> elm_A_, elm_c_;   // (H+1)^2 and (H+1)
    void elm_update_readout();            // accumulate φ from critic_local_ + solve ridge
};

} // namespace pricer
