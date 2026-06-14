// test_parity.cpp — assert C++ forward + kernel target match PyTorch references.
// Build this with -DCPP_PRICER_FP64=ON for a tight (<1e-4) comparison.
#include "mlp.hpp"
#include "kernel.hpp"
#include "io.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace pricer;

static std::string DATA = "data/";   // run tests from cpp_pricer/

static double max_abs_diff(const std::vector<Real>& a, const std::vector<double>& b) {
    double m = 0;
    for (size_t i = 0; i < b.size(); ++i) m = std::max(m, std::abs((double)a[i] - b[i]));
    return m;
}

int main(int argc, char** argv) {
    if (argc > 1) DATA = std::string(argv[1]) + "/";
    int fails = 0;
    try {
        // ---- load weights ----
        auto actor_w = pio::load_flat(DATA + "actor.bin");
        auto critic_w = pio::load_flat(DATA + "critic.bin");
        Actor actor; Critic critic;
        actor.load_flat(actor_w.data());
        critic.load_flat(critic_w.data());

        // ---- forward parity ----
        {
            FILE* f = pio::open_rd(DATA + "ref_fwd.bin");
            int B = pio::rd_i32(f);
            double strike = 1.0, qmin=0.0, qmax=2.0, c=0.04, g=2.0;
            actor.set_gate(strike, qmin, qmax, c, g); critic.set_strike(strike);
            auto states_d = pio::rd_doubles(f, B*STATE_DIM);
            auto acts_d   = pio::rd_doubles(f, B);
            auto ao_ref   = pio::rd_doubles(f, B);
            auto co_ref   = pio::rd_doubles(f, B);
            std::fclose(f);
            std::vector<Real> states(B*STATE_DIM), acts(B);
            for (int i=0;i<B*STATE_DIM;++i) states[i]=(Real)states_d[i];
            for (int i=0;i<B;++i) acts[i]=(Real)acts_d[i];
            std::vector<Real> aout(B), cout(B);
            actor.forward(states.data(), B, aout.data());
            critic.forward(states.data(), acts.data(), B, cout.data());
            double da = max_abs_diff(aout, ao_ref);
            double dc = max_abs_diff(cout, co_ref);
            std::printf("[fwd] actor max|d|=%.3e  critic max|d|=%.3e\n", da, dc);
            if (da > 1e-4 || dc > 1e-4) { std::printf("  FAIL forward parity\n"); ++fails; }
        }

        // ---- kernel target parity ----
        {
            auto kernel = TransitionKernel::load(DATA + "kernel_v64.bin");
            FILE* f = pio::open_rd(DATA + "ref_kernel.bin");
            int B = pio::rd_i32(f);
            auto states_d = pio::rd_doubles(f, B*STATE_DIM);
            auto ns_d     = pio::rd_doubles(f, B*STATE_DIM);
            auto qn_ref   = pio::rd_doubles(f, B);
            std::fclose(f);
            std::vector<Real> states(B*STATE_DIM), ns(B*STATE_DIM);
            for (int i=0;i<B*STATE_DIM;++i){ states[i]=(Real)states_d[i]; ns[i]=(Real)ns_d[i]; }
            // target nets = the loaded weights
            Actor at; Critic ct; at.load_flat(actor_w.data()); ct.load_flat(critic_w.data());
            at.set_gate(1.0,0.0,2.0,0.04,2.0); ct.set_strike(1.0);
            KernelWorkspace ws; std::vector<Real> qn(B);
            expected_critic_target(at, ct, states.data(), ns.data(), B, kernel, 1.0, ws, qn.data());
            double dk = max_abs_diff(qn, qn_ref);
            std::printf("[kernel] max|d|=%.3e\n", dk);
            if (dk > 1e-4) { std::printf("  FAIL kernel parity\n"); ++fails; }
        }
    } catch (const std::exception& e) {
        std::printf("ERROR: %s\n(did you run tools/export_reference.py first?)\n", e.what());
        return 2;
    }
    std::printf(fails ? "PARITY: FAIL (%d)\n" : "PARITY: PASS\n", fails);
    return fails ? 1 : 0;
}
