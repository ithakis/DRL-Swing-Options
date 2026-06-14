// bench_train.cpp — time K learn_steps in isolation (fast optimization iteration).
#include "agent.hpp"
#include "hhk_sim.hpp"
#include "kernel.hpp"
#include <cstdio>
#include <string>

using namespace pricer;

int main(int argc, char** argv) {
    int K = 20000;
    int n_max = 0;   // H-K1 canonical M=2; --n_max 1 for old M=4
    int hidden = H_DEFAULT;   // R5/H-A1: hidden width. 64 = v64 canonical.
    int actor_layers = 3;     // H-N1/H-N2: actor hidden depth. 3 = v64 canonical.
    int critic_layers = 3;    // H-N1/H-N2: critic hidden depth (>=2). 3 = v64 canonical.
    int reuse_target = 0;     // R6/H-C1: 1 = reuse minibatch+target across learn_number inner steps.
    int elm_critic = 0;       // R9/H-R1: 1 = ELM critic (frozen random features + ridge readout).
    std::string kpath = "data/kernel_v64.bin";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--K") K = std::stoi(argv[++i]);
        else if (a == "--kernel") kpath = argv[++i];
        else if (a == "--n_max") n_max = std::stoi(argv[++i]);
        else if (a == "--hidden") hidden = std::stoi(argv[++i]);
        else if (a == "--actor_layers") actor_layers = std::stoi(argv[++i]);
        else if (a == "--critic_layers") critic_layers = std::stoi(argv[++i]);
        else if (a == "--reuse_target") reuse_target = std::stoi(argv[++i]);
        else if (a == "--elm_critic") elm_critic = std::stoi(argv[++i]);
    }
    SwingContract c; HHKParams hhk; KernelParams kp; kp.N_max = n_max;
    TransitionKernel kernel;
    try { kernel = TransitionKernel::load(kpath); } catch (...) { kernel = TransitionKernel::build_fast(hhk, c, kp); }

    AgentConfig cfg; cfg.n_threads = 1; cfg.hidden = hidden; cfg.reuse_target = (reuse_target != 0);
    cfg.actor_layers = actor_layers; cfg.critic_layers = critic_layers;
    cfg.elm_critic = (elm_critic != 0);
    Paths paths; simulate_hhk(hhk, c, 4096, 11, true, cfg.batch, 4, paths);
    Agent agent(cfg, c, hhk, kernel, 11);
    agent.calibrate_bias(paths);
    agent.bench_prepare(paths);

    double warm = agent.bench_learn(2000);          // warm caches
    agent.prof_kernel = agent.prof_critic = agent.prof_actor = agent.prof_soft = 0;
    double t = agent.bench_learn(K);
    std::printf("learn_steps=%d  time=%.4fs  us/step=%.2f  (warm %.3fs)\n",
                K, t, 1e6 * t / K, warm);
#ifdef PRICER_PROFILE
    double tot = agent.prof_kernel + agent.prof_critic + agent.prof_actor + agent.prof_soft;
    std::printf("  kernel-target: %.1f%%  critic: %.1f%%  actor: %.1f%%  soft/ema: %.1f%%  (sum %.2fs)\n",
        100*agent.prof_kernel/tot, 100*agent.prof_critic/tot, 100*agent.prof_actor/tot,
        100*agent.prof_soft/tot, tot);
#endif
    return 0;
}
