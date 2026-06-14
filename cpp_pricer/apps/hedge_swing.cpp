// hedge_swing.cpp — v65 hedging export. Trains the v65 agent ONCE, then:
//   (1) PV-vs-S0 grid: mean OOS PV at 9 spot nodes (shared CRN seed) -> Delta/Gamma by np.gradient.
//   (2) Daily delta hedge: base roll (cf, q_before, pv) + per-date spot-bump-and-re-roll (Vp/Vm),
//       with the OU bump propagated in closed form (CRN), mirroring src/greeks.py exactly.
// All policy-dependent rolls run here (where the v65 net lives); the forward prices, conditioning
// regression, hedge ratios and P&L are finished in Python (tools/cpp_hedge.py) reusing src/greeks.py.
#include "agent.hpp"
#include "kernel.hpp"
#include "hhk_sim.hpp"
#include "env.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace pricer;

// Roll the EMA actor from `start_step` to maturity over all n paths (batched), on the supplied
// (possibly bumped) S/X/Y arrays. Returns per-path continuation PV discounted to t=0. With
// collect=true also fills cf (discounted net per step) and qbefore (pre-decision q_exercised).
static void roll_continuation(Actor& actor, const SwingContract& c,
                              const Real* S, const Real* X, const Real* Y, int n, int T,
                              int start_step, const double* q0, const long* lastex0,
                              double* pv, bool collect, float* cf, float* qbefore) {
    std::vector<EpisodeState> es(n);
    std::vector<char> done(n, 0);
    for (int p = 0; p < n; ++p) {
        es[p].step = start_step;
        es[p].q_exercised = q0[p];
        es[p].last_exercise_step = (int)lastex0[p];
        pv[p] = 0.0;
        done[p] = (start_step >= c.n_rights) || (es[p].q_exercised >= c.Q_max - 1e-6);
    }
    std::vector<Real> state((size_t)n * STATE_DIM), qg(n);
    for (int step = start_step; step < c.n_rights; ++step) {
        for (int p = 0; p < n; ++p) {
            if (collect) {
                cf[(size_t)p * T + step] = 0.f;
                qbefore[(size_t)p * T + step] = (float)es[p].q_exercised;
            }
            build_obs(c, S + (size_t)p * T, X + (size_t)p * T, Y + (size_t)p * T, es[p],
                      state.data() + (size_t)p * STATE_DIM);
        }
        actor.forward(state.data(), n, qg.data());
        for (int p = 0; p < n; ++p) {
            if (done[p]) continue;
            Real a = std::clamp(qg[p], (Real)0, (Real)1);
            bool term = false;
            double r = env_step(c, S + (size_t)p * T, a, es[p], &term);
            pv[p] += r;
            if (collect) cf[(size_t)p * T + step] = (float)r;
            if (term) done[p] = 1;
        }
    }
}

int main(int argc, char** argv) {
    uint64_t seed = 999;
    int n_train = 4096, n_rl = 8192, n_hedge = 4096, n_grid = 9, threads = 8;
    double h = 0.01, S0 = 1.0, c_cost = -1, gamma_cost = -1;
    std::string kernel_path = "data/kernel_v64.bin", out_path = "hedge.bin";
    bool kernel_off = false;
    int hidden = 48, hidden_actor = 32, hidden_critic = -1, actor_layers = 2, critic_layers = 4;
    int batch = 64, learn_number = 3; double lr_c = 5e-4;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]{ return std::string(argv[++i]); };
        if (a == "--seed") seed = std::stoull(nx());
        else if (a == "--n_train") n_train = std::stoi(nx());
        else if (a == "--n_rl") n_rl = std::stoi(nx());
        else if (a == "--n_hedge") n_hedge = std::stoi(nx());
        else if (a == "--n_grid") n_grid = std::stoi(nx());
        else if (a == "--threads") threads = std::stoi(nx());
        else if (a == "--h") h = std::stod(nx());
        else if (a == "--c_cost") c_cost = std::stod(nx());
        else if (a == "--gamma_cost") gamma_cost = std::stod(nx());
        else if (a == "--kernel") kernel_path = nx();
        else if (a == "--kernel_off") kernel_off = true;
        else if (a == "--out") out_path = nx();
        else if (a == "--hidden") hidden = std::stoi(nx());
        else if (a == "--hidden_actor") hidden_actor = std::stoi(nx());
        else if (a == "--hidden_critic") hidden_critic = std::stoi(nx());
        else if (a == "--actor_layers") actor_layers = std::stoi(nx());
        else if (a == "--critic_layers") critic_layers = std::stoi(nx());
        else if (a == "--batch") batch = std::stoi(nx());
        else if (a == "--learn_number") learn_number = std::stoi(nx());
        else if (a == "--lr_c") lr_c = std::stod(nx());
    }

    SwingContract c; HHKParams hhk; KernelParams kp; hhk.S0 = S0;
    if (c_cost >= 0) c.c_cost = c_cost;
    if (gamma_cost >= 0) c.gamma_cost = gamma_cost;

    TransitionKernel kernel;   // default M()==0 => kernel-off
    if (!kernel_off) {
        try { kernel = TransitionKernel::load(kernel_path); }
        catch (...) { kernel = TransitionKernel::build_fast(hhk, c, kp); }
    }

    AgentConfig cfg; cfg.n_threads = threads;
    cfg.hidden = hidden; cfg.hidden_actor = hidden_actor; cfg.hidden_critic = hidden_critic;
    cfg.actor_layers = actor_layers; cfg.critic_layers = critic_layers;
    cfg.batch = batch; cfg.learn_number = learn_number; cfg.lr_c = lr_c;

    // ---- train once ----
    Paths train_paths;
    simulate_hhk(hhk, c, n_train, seed, /*stratify*/true, cfg.batch, threads, train_paths);
    Agent agent(cfg, c, hhk, kernel, seed);
    agent.calibrate_bias(train_paths);
    agent.train(train_paths, n_train);
    Actor& actor = agent.actor_eval_ref();

    const int T = c.n_rights;
    const double dt = c.dt();

    // ---- (1) PV-vs-S0 grid (shared seed across nodes => CRN) ----
    std::vector<double> grid(n_grid), grid_pv(n_grid);
    for (int g = 0; g < n_grid; ++g)
        grid[g] = (0.5 + 1.5 * (double)g / (n_grid - 1)) * S0;   // linspace(0.5,2.0,n_grid)*S0
    for (int g = 0; g < n_grid; ++g) {
        HHKParams hg = hhk; hg.S0 = grid[g];
        Paths pg;
        simulate_hhk(hg, c, n_rl, seed, /*stratify*/false, 0, threads, pg);
        EvalResult r = agent.evaluate(pg, 512);
        grid_pv[g] = r.price;
    }

    // ---- (2) daily delta hedge at base S0 ----
    Paths hp;
    simulate_hhk(hhk, c, n_hedge, seed, /*stratify*/false, 0, threads, hp);
    const int n = n_hedge;
    std::vector<float> cf((size_t)n * T, 0.f), qbefore((size_t)n * T, 0.f);
    std::vector<float> Vp((size_t)n * T, 0.f), Vm((size_t)n * T, 0.f);
    std::vector<double> pv(n, 0.0);
    std::vector<double> q0(n, 0.0); std::vector<long> lastex0(n, -1);

    // base roll: realised PV + per-step cf + pre-decision q_exercised. Recover last_ex by tracking.
    // We need q_before AND last_ex per (path,t) to seed continuation rolls. roll_continuation fills
    // cf+qbefore; re-derive last_ex on the fly via a parallel collect pass.
    {
        std::vector<EpisodeState> es(n);
        std::vector<char> done(n, 0);
        std::vector<Real> state((size_t)n * STATE_DIM), qg(n);
        // lastex_before[p*T+step] captured into a temp; store into Vm temporarily? Use separate buffer.
        std::vector<long> lastex_before((size_t)n * T, -1);
        for (int p = 0; p < n; ++p) { es[p].step = 0; es[p].q_exercised = 0; es[p].last_exercise_step = -1; }
        for (int step = 0; step < T; ++step) {
            for (int p = 0; p < n; ++p) {
                qbefore[(size_t)p * T + step] = (float)es[p].q_exercised;
                lastex_before[(size_t)p * T + step] = es[p].last_exercise_step;
                build_obs(c, hp.Srow(p), hp.Xrow(p), hp.Yrow(p), es[p], state.data() + (size_t)p * STATE_DIM);
            }
            actor.forward(state.data(), n, qg.data());
            for (int p = 0; p < n; ++p) {
                if (done[p]) continue;
                Real a = std::clamp(qg[p], (Real)0, (Real)1);
                bool term = false;
                double r = env_step(c, hp.Srow(p), a, es[p], &term);
                pv[p] += r;
                cf[(size_t)p * T + step] = (float)r;
                if (term) done[p] = 1;
            }
        }

        // per-date bump-and-re-roll: bump X_t (+/- h*decay) over [t:], propagate to S, roll from t.
        std::vector<Real> Sp((size_t)n * T), Xp((size_t)n * T), Sm((size_t)n * T), Xm((size_t)n * T);
        std::vector<double> q_t(n); std::vector<long> lx_t(n);
        std::vector<double> vpp(n), vmm(n);
        for (int t = 0; t < T - 1; ++t) {
            // build bumped copies (only k>=t differ; copy whole rows for simplicity)
            for (int p = 0; p < n; ++p) {
                const Real* Sr = hp.Srow(p); const Real* Xr = hp.Xrow(p);
                Real* sp = Sp.data() + (size_t)p * T; Real* xp = Xp.data() + (size_t)p * T;
                Real* sm = Sm.data() + (size_t)p * T; Real* xm = Xm.data() + (size_t)p * T;
                for (int k = 0; k < t; ++k) { sp[k]=Sr[k]; xp[k]=Xr[k]; sm[k]=Sr[k]; xm[k]=Xr[k]; }
                for (int k = t; k < T; ++k) {
                    double decay = std::exp(-hhk.alpha * (k - t) * dt);
                    sp[k] = (Real)(Sr[k] * std::exp(h * decay));
                    sm[k] = (Real)(Sr[k] * std::exp(-h * decay));
                    xp[k] = (Real)(Xr[k] + h * decay);
                    xm[k] = (Real)(Xr[k] - h * decay);
                }
                q_t[p] = qbefore[(size_t)p * T + t];
                lx_t[p] = lastex_before[(size_t)p * T + t];
            }
            roll_continuation(actor, c, Sp.data(), Xp.data(), hp.Y.data(), n, T, t, q_t.data(), lx_t.data(), vpp.data(), false, nullptr, nullptr);
            roll_continuation(actor, c, Sm.data(), Xm.data(), hp.Y.data(), n, T, t, q_t.data(), lx_t.data(), vmm.data(), false, nullptr, nullptr);
            for (int p = 0; p < n; ++p) {
                Vp[(size_t)p * T + t] = (float)vpp[p];
                Vm[(size_t)p * T + t] = (float)vmm[p];
            }
        }
    }

    // ---- write blob ----
    std::ofstream f(out_path, std::ios::binary);
    int32_t magic = 0x48454447 /*'HEDG'*/, nn = n, tt = T, ng = n_grid;
    f.write((char*)&magic,4); f.write((char*)&nn,4); f.write((char*)&tt,4); f.write((char*)&ng,4);
    f.write((char*)&h,8); f.write((char*)&S0,8);
    f.write((char*)grid.data(), (size_t)ng*8);
    f.write((char*)grid_pv.data(), (size_t)ng*8);
    auto wf = [&](std::vector<float>& v){ f.write((char*)v.data(), v.size()*sizeof(float)); };
    // S/X/Y as float32 (hp.* are Real=float in the fp32 build)
    f.write((char*)hp.S.data(), (size_t)n*T*sizeof(Real));
    f.write((char*)hp.X.data(), (size_t)n*T*sizeof(Real));
    f.write((char*)hp.Y.data(), (size_t)n*T*sizeof(Real));
    wf(cf); wf(qbefore); wf(Vp); wf(Vm);
    std::vector<float> pvf(pv.begin(), pv.end()); wf(pvf);

    std::printf("{\"price\": %.6f, \"n_hedge\": %d, \"n_grid\": %d, \"grid_pv0\": %.4f, \"grid_pvN\": %.4f}\n",
                pv.size()? (std::accumulate(pv.begin(),pv.end(),0.0)/n):0.0, n, n_grid, grid_pv.front(), grid_pv.back());
    return 0;
}
