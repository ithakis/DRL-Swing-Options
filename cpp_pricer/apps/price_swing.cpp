// price_swing.cpp — standalone CLI: simulate, train 0->4k, evaluate 65k, emit JSON.
// Timings: t_zero_to_4k (sim train paths + warm-start + train) and
//          t_4k_to_65k (sim OOS paths + greedy rollout) — the two costs the user minimizes.
#include "agent.hpp"
#include "kernel.hpp"
#include "hhk_sim.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>

using namespace pricer;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    uint64_t seed = 11;
    int n_train = 4096, n_eval = 65536, eval_batch = 512;
    int threads = (int)std::thread::hardware_concurrency();
    if (threads <= 0) threads = 4;
    std::string kernel_path = "data/kernel_v64.bin";
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]{ return std::string(argv[++i]); };
        if (a == "--seed") seed = std::stoull(next());
        else if (a == "--n_train") n_train = std::stoi(next());
        else if (a == "--n_eval") n_eval = std::stoi(next());
        else if (a == "--eval_batch") eval_batch = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--kernel") kernel_path = next();
        else if (a == "--quiet") quiet = true;
    }

    SwingContract c; HHKParams hhk; KernelParams kp;

    TransitionKernel kernel;
    try { kernel = TransitionKernel::load(kernel_path); }
    catch (...) {
        if (!quiet) std::fprintf(stderr, "[warn] no kernel mesh at %s; building fast mesh in C++\n", kernel_path.c_str());
        kernel = TransitionKernel::build_fast(hhk, c, kp);
    }

    AgentConfig cfg; cfg.n_threads = threads;

    // ============ 0 -> 4k ============
    auto t0 = clk::now();
    Paths train_paths;
    simulate_hhk(hhk, c, n_train, seed, /*stratify*/true, cfg.batch, threads, train_paths);
    auto t_sim_train = clk::now();

    Agent agent(cfg, c, hhk, kernel, seed);
    agent.calibrate_bias(train_paths);
    auto t_calib = clk::now();

    agent.train(train_paths, n_train);
    auto t1 = clk::now();

    // ============ 4k -> 65k ============
    Paths eval_paths;
    simulate_hhk(hhk, c, n_eval, seed + 777, /*stratify*/false, 0, threads, eval_paths);
    auto t_sim_eval = clk::now();

    EvalResult res = agent.evaluate(eval_paths, eval_batch);
    auto t2 = clk::now();

    double t_zero_to_4k = secs(t0, t1);
    double t_4k_to_65k  = secs(t1, t2);

    std::printf(
        "{\n"
        "  \"price\": %.6f,\n"
        "  \"ci95\": %.6f,\n"
        "  \"std\": %.6f,\n"
        "  \"avg_exercised\": %.4f,\n"
        "  \"seed\": %llu,\n"
        "  \"n_train\": %d,\n"
        "  \"n_eval\": %d,\n"
        "  \"threads\": %d,\n"
        "  \"precision\": \"%s\",\n"
        "  \"t_sim_train\": %.4f,\n"
        "  \"t_calibrate\": %.4f,\n"
        "  \"t_train\": %.4f,\n"
        "  \"t_sim_eval\": %.4f,\n"
        "  \"t_eval\": %.4f,\n"
        "  \"t_zero_to_4k\": %.4f,\n"
        "  \"t_4k_to_65k\": %.4f,\n"
        "  \"t_total\": %.4f\n"
        "}\n",
        res.price, res.ci95, res.std, res.avg_exercised,
        (unsigned long long)seed, n_train, n_eval, threads,
        (sizeof(Real)==8 ? "fp64" : "fp32"),
        secs(t0, t_sim_train), secs(t_sim_train, t_calib), secs(t_calib, t1),
        secs(t1, t_sim_eval), secs(t_sim_eval, t2),
        t_zero_to_4k, t_4k_to_65k, t_zero_to_4k + t_4k_to_65k);
    return 0;
}
