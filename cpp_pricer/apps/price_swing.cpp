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
#include <fstream>
#include <sys/resource.h>

using namespace pricer;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
// User-mode CPU-seconds. Load-INDEPENDENT: the train loop is single-threaded compute on
// pre-allocated buffers, so user CPU time = the actual work done and stays constant under
// CPU contention (unlike wall-clock t_train). System time is excluded (it picks up
// scheduling/context-switch noise under oversubscription).
static double cpu_secs() {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    uint64_t seed = 11;
    long long agent_seed = -1;   // H-N5b: init+exploration RNG seed (-1 => = data seed). Fix it to
                                 // isolate DATA-only seed variance (LSM-matched seed semantics).
    int n_train = 4096, n_eval = 65536, eval_batch = 512;
    int threads = (int)std::thread::hardware_concurrency();
    if (threads <= 0) threads = 4;
    std::string kernel_path = "data/kernel_v64.bin";
    std::string trace_path;   // if set, dump per-(path,step) eval trace as a binary blob for NB6
    std::string eval_paths_out; // if set, dump the eval S/X/Y paths so Python LSM prices the SAME set
    bool quiet = false;
    bool kernel_off = false;   // NB3/Hedging "RL (no kernel)": empty mesh => single-sample TD target
    int n_max = 0;   // H-K1 canonical: M_y=1 (jump folded into mean). --n_max 1 reproduces old M=4.
    int hidden = H_DEFAULT;   // R5/H-A1: hidden width. 64 = v64 canonical.
    int hidden_actor = -1;    // H-W1: per-net actor width (-1 => use --hidden)
    int hidden_critic = -1;   // H-W1: per-net critic width (-1 => use --hidden)
    int actor_layers = 3;     // H-N1/H-N2: actor hidden depth. 3 = v64 canonical.
    int critic_layers = 3;    // H-N1/H-N2: critic hidden depth (>=2). 3 = v64 canonical.
    int reuse_target = 0;     // R6/H-C1: 1 = reuse minibatch+target across learn_number inner steps.
    int elm_critic = 0;       // R9/H-R1: 1 = ELM critic (frozen random features + ridge readout).
    double elm_ridge = 1e-2, elm_forget = 0.99;
    int direct_policy = 0;    // R10/H-R2: 1 = critic-free REINFORCE direct-policy trainer.
    int batch = -1;           // H-N3: minibatch size (-1 = AgentConfig default 128)
    double lr_a = -1, lr_c = -1, wd_c = -1, ema_decay = -1;   // H-N3/H-N6/H-N4 (-1 = keep default)
    double tau = -1, noise_sigma0 = -1, noise_floor = -1, adaptive_noise_scale = -1, warmup_noise_fraction = -1; // H-RL: actor/critic RL space
    int weight_avg = -1;      // H-N4: 0=EMA (default), 1=SWA tail-average
    int swa_start = -1;       // H-N4: episode to start SWA accumulation (-1 = 75% of n_train)
    double init_gain = -1, critic_out_init = -1, actor_out_init = -1;  // H-I: init knobs (-1 = default)
    int learn_number = -1, learn_every = -1, critic_warmup = -1;  // H-T: train-cadence (-1 = default)
    double c_cost = -1, gamma_cost = -1;   // cross-regime: nocost(c=0)/linear(g=1)/convex(g=2). -1=default focal.
    int init_method = 0;   // H-I3: 0=He(default), 1=orthogonal, 2=Xavier
    std::string sampler_s = "mc";   // Task 3: training scenario generator {mc, arqmc}
    std::string replay_s = "uniform"; // Task 3: replay distribution {uniform, time, coverage}
    double step_density = 1.0;       // Task 3 (replay=time): weight ∝ ((t+1)/T)^step_density

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]{ return std::string(argv[++i]); };
        if (a == "--seed") seed = std::stoull(next());
        else if (a == "--agent_seed") agent_seed = std::stoll(next());
        else if (a == "--n_train") n_train = std::stoi(next());
        else if (a == "--n_eval") n_eval = std::stoi(next());
        else if (a == "--eval_batch") eval_batch = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--kernel") kernel_path = next();
        else if (a == "--trace") trace_path = next();
        else if (a == "--dump_eval_paths") eval_paths_out = next();
        else if (a == "--kernel_off") kernel_off = true;
        else if (a == "--n_max") n_max = std::stoi(next());
        else if (a == "--hidden") hidden = std::stoi(next());
        else if (a == "--hidden_actor") hidden_actor = std::stoi(next());
        else if (a == "--hidden_critic") hidden_critic = std::stoi(next());
        else if (a == "--actor_layers") actor_layers = std::stoi(next());
        else if (a == "--critic_layers") critic_layers = std::stoi(next());
        else if (a == "--reuse_target") reuse_target = std::stoi(next());
        else if (a == "--elm_critic") elm_critic = std::stoi(next());
        else if (a == "--elm_ridge") elm_ridge = std::stod(next());
        else if (a == "--elm_forget") elm_forget = std::stod(next());
        else if (a == "--direct_policy") direct_policy = std::stoi(next());
        else if (a == "--batch") batch = std::stoi(next());
        else if (a == "--lr_a") lr_a = std::stod(next());
        else if (a == "--lr_c") lr_c = std::stod(next());
        else if (a == "--wd_c") wd_c = std::stod(next());
        else if (a == "--ema_decay") ema_decay = std::stod(next());
        else if (a == "--weight_avg") weight_avg = std::stoi(next());
        else if (a == "--swa_start") swa_start = std::stoi(next());
        else if (a == "--init_gain") init_gain = std::stod(next());
        else if (a == "--critic_out_init") critic_out_init = std::stod(next());
        else if (a == "--actor_out_init") actor_out_init = std::stod(next());
        else if (a == "--learn_number") learn_number = std::stoi(next());
        else if (a == "--learn_every") learn_every = std::stoi(next());
        else if (a == "--critic_warmup") critic_warmup = std::stoi(next());
        else if (a == "--c_cost") c_cost = std::stod(next());
        else if (a == "--gamma_cost") gamma_cost = std::stod(next());
        else if (a == "--init_method") init_method = std::stoi(next());
        else if (a == "--tau") tau = std::stod(next());
        else if (a == "--noise_sigma0") noise_sigma0 = std::stod(next());
        else if (a == "--noise_floor") noise_floor = std::stod(next());
        else if (a == "--adaptive_noise_scale") adaptive_noise_scale = std::stod(next());
        else if (a == "--warmup_noise_fraction") warmup_noise_fraction = std::stod(next());
        else if (a == "--sampler") sampler_s = next();
        else if (a == "--replay") replay_s = next();
        else if (a == "--step_density") step_density = std::stod(next());
        else if (a == "--quiet") quiet = true;
    }

    SwingContract c; HHKParams hhk; KernelParams kp; kp.N_max = n_max;
    if (c_cost >= 0) c.c_cost = c_cost;          // cross-regime cost override
    if (gamma_cost >= 0) c.gamma_cost = gamma_cost;

    TransitionKernel kernel;   // default => M()==0 => single-sample TD (kernel-off)
    if (!kernel_off) {
        try { kernel = TransitionKernel::load(kernel_path); }
        catch (...) {
            if (!quiet) std::fprintf(stderr, "[warn] no kernel mesh at %s; building fast mesh in C++\n", kernel_path.c_str());
            kernel = TransitionKernel::build_fast(hhk, c, kp);
        }
    }

    AgentConfig cfg; cfg.n_threads = threads; cfg.hidden = hidden;
    cfg.hidden_actor = hidden_actor; cfg.hidden_critic = hidden_critic;
    cfg.actor_layers = actor_layers; cfg.critic_layers = critic_layers;
    cfg.reuse_target = (reuse_target != 0);
    if (batch > 0) cfg.batch = batch;
    if (lr_a >= 0) cfg.lr_a = lr_a;
    if (lr_c >= 0) cfg.lr_c = lr_c;
    if (wd_c >= 0) cfg.wd_c = wd_c;
    if (ema_decay >= 0) cfg.ema_decay = ema_decay;
    if (weight_avg >= 0) cfg.weight_avg = weight_avg;
    if (swa_start >= 0) cfg.swa_start = swa_start;
    cfg.init_gain = init_gain; cfg.critic_out_init = critic_out_init; cfg.actor_out_init = actor_out_init;
    cfg.init_method = init_method;
    if (tau >= 0) cfg.tau = tau;
    if (noise_sigma0 >= 0) cfg.noise_sigma0 = noise_sigma0;
    if (noise_floor >= 0) cfg.noise_floor = noise_floor;
    if (adaptive_noise_scale >= 0) cfg.adaptive_noise_scale = adaptive_noise_scale;
    if (warmup_noise_fraction >= 0) cfg.warmup_noise_fraction = warmup_noise_fraction;
    if (learn_number > 0) cfg.learn_number = learn_number;
    if (learn_every > 0) cfg.learn_every = learn_every;
    if (critic_warmup >= 0) cfg.critic_warmup = critic_warmup;
    cfg.elm_critic = (elm_critic != 0); cfg.elm_ridge = elm_ridge; cfg.elm_forget = elm_forget;
    // Task 3 (sample efficiency): scenario generator + replay-distribution arms.
    Sampler sampler = (sampler_s == "arqmc") ? Sampler::ARQMC : Sampler::MC;
    cfg.replay_mode = (replay_s == "time") ? 1 : (replay_s == "coverage") ? 2 : 0;
    cfg.step_density = step_density;

    // ============ 0 -> 4k ============
    auto t0 = clk::now();
    Paths train_paths;
    simulate_hhk(hhk, c, n_train, seed, /*stratify*/true, cfg.batch, threads, train_paths, sampler);
    auto t_sim_train = clk::now();

    Agent agent(cfg, c, hhk, kernel, (agent_seed >= 0 ? (uint64_t)agent_seed : seed));
    agent.calibrate_bias(train_paths);
    auto t_calib = clk::now();

    double cpu0 = cpu_secs();
    if (direct_policy) agent.train_direct_policy(train_paths, n_train);
    else               agent.train(train_paths, n_train);
    double cpu_train = cpu_secs() - cpu0;   // load-independent train cost
    auto t1 = clk::now();

    // ============ 4k -> 65k ============
    Paths eval_paths;
    simulate_hhk(hhk, c, n_eval, seed + 777, /*stratify*/false, 0, threads, eval_paths);
    auto t_sim_eval = clk::now();

    // Dump the eval S/X/Y so the Python LSM can price the IDENTICAL OOS set (valid per-path
    // RL-vs-LSM merge in NB6). Layout: int32 magic('SXYP'), n_paths, T, then float32 S,X,Y [n*T].
    if (!eval_paths_out.empty()) {
        std::ofstream f(eval_paths_out, std::ios::binary);
        int32_t magic = 0x53585950 /*'SXYP'*/, np = eval_paths.n_paths, tt = eval_paths.T;
        f.write((char*)&magic, 4); f.write((char*)&np, 4); f.write((char*)&tt, 4);
        size_t sz = (size_t)np * tt;
        f.write((char*)eval_paths.S.data(), sz*sizeof(Real));
        f.write((char*)eval_paths.X.data(), sz*sizeof(Real));
        f.write((char*)eval_paths.Y.data(), sz*sizeof(Real));
    }

    EvalTrace trace;
    EvalResult res = agent.evaluate(eval_paths, eval_batch, trace_path.empty() ? nullptr : &trace);
    auto t2 = clk::now();

    // Per-(path,step) trace blob for the analysis notebooks (NB6). Layout:
    //   int32 magic('SWTR'), int32 n_paths, int32 T, then 4 float32[n_paths*T] arrays
    //   in order q, reward, cost(undisc.), gross(undisc.). Python: tools/cpp_trace_to_parquet.py
    if (!trace_path.empty()) {
        std::ofstream f(trace_path, std::ios::binary);
        int32_t magic = 0x53575452 /*'SWTR'*/, np = trace.n_paths, tt = trace.T;
        f.write((char*)&magic, 4); f.write((char*)&np, 4); f.write((char*)&tt, 4);
        size_t sz = (size_t)np * tt;
        f.write((char*)trace.q.data(),      sz*sizeof(float));
        f.write((char*)trace.reward.data(), sz*sizeof(float));
        f.write((char*)trace.cost.data(),   sz*sizeof(float));
        f.write((char*)trace.gross.data(),  sz*sizeof(float));
    }

    double t_zero_to_4k = secs(t0, t1);
    double t_4k_to_65k  = secs(t1, t2);

    std::printf(
        "{\n"
        "  \"price\": %.6f,\n"
        "  \"ci95\": %.6f,\n"
        "  \"std\": %.6f,\n"
        "  \"avg_exercised\": %.4f,\n"
        "  \"bangbang\": %.6f,\n"
        "  \"sampler\": \"%s\",\n"
        "  \"replay\": \"%s\",\n"
        "  \"step_density\": %.4f,\n"
        "  \"seed\": %llu,\n"
        "  \"n_train\": %d,\n"
        "  \"n_eval\": %d,\n"
        "  \"hidden\": %d,\n"
        "  \"actor_layers\": %d,\n"
        "  \"critic_layers\": %d,\n"
        "  \"threads\": %d,\n"
        "  \"precision\": \"%s\",\n"
        "  \"t_sim_train\": %.4f,\n"
        "  \"t_calibrate\": %.4f,\n"
        "  \"t_train\": %.4f,\n"
        "  \"cpu_train\": %.4f,\n"
        "  \"t_sim_eval\": %.4f,\n"
        "  \"t_eval\": %.4f,\n"
        "  \"t_zero_to_4k\": %.4f,\n"
        "  \"t_4k_to_65k\": %.4f,\n"
        "  \"t_total\": %.4f\n"
        "}\n",
        res.price, res.ci95, res.std, res.avg_exercised, res.bangbang,
        sampler_s.c_str(), replay_s.c_str(), step_density,
        (unsigned long long)seed, n_train, n_eval, hidden, actor_layers, critic_layers, threads,
        (sizeof(Real)==8 ? "fp64" : "fp32"),
        secs(t0, t_sim_train), secs(t_sim_train, t_calib), secs(t_calib, t1), cpu_train,
        secs(t1, t_sim_eval), secs(t_sim_eval, t2),
        t_zero_to_4k, t_4k_to_65k, t_zero_to_4k + t_4k_to_65k);
    return 0;
}
