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
};

struct EvalResult { double price; double ci95; double std; double avg_exercised; };

class Agent {
public:
    Agent(const AgentConfig& cfg, const SwingContract& c, const HHKParams& hhk,
          const TransitionKernel& kernel, uint64_t seed);

    void calibrate_bias(const Paths& warmup);                 // closed-form warm-start
    void train(const Paths& train_paths, int n_episodes);     // 0 -> n_episodes
    EvalResult evaluate(const Paths& eval_paths, int eval_batch);

    // Load PyTorch-exported weights into local+target (for parity tests).
    void load_weights(const double* actor_flat, const double* critic_flat);

    // --- benchmarking hooks (isolate the training hot loop) ---
    void bench_prepare(const Paths& p);    // prefill replay + arm actor updates
    double bench_learn(int K);             // time K learn_steps (seconds)
    double prof_kernel=0, prof_critic=0, prof_actor=0, prof_soft=0;  // accumulated seconds

    Actor& actor_local_ref() { return actor_local_; }
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

    // batch scratch
    std::vector<Real> sb_, ab_, rb_, nsb_, db_, qexp_, qtgt_, qnext_, apred_, qval_, dqa_, dq_, dqv_;

    double pre_noise_sigma() const;
    Real act_single(const Real* state, bool add_noise);
    void learn_step();
    void init_orthogonal(uint64_t seed);
};

} // namespace pricer
