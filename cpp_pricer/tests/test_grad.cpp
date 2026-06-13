// test_grad.cpp — finite-difference gradcheck of the hand-derived backward.
// Given forward parity vs PyTorch (test_parity), matching FD grads => the
// analytic grads match torch.autograd. Build with -DCPP_PRICER_FP64=ON.
#include "mlp.hpp"
#include "io.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

using namespace pricer;
static std::string DATA = "data/";

static double critic_loss(Critic& cr, const std::vector<Real>& s, const std::vector<Real>& a, int B) {
    std::vector<Real> q(B); cr.forward(s.data(), a.data(), B, q.data());
    double L = 0; for (int i = 0; i < B; ++i) L += q[i]; return L;     // dL/dq = 1
}
static double actor_loss(Actor& ac, Critic& cr, const std::vector<Real>& s, int B) {
    std::vector<Real> ap(B), q(B);
    ac.forward(s.data(), B, ap.data());
    cr.forward(s.data(), ap.data(), B, q.data());
    double L = 0; for (int i = 0; i < B; ++i) L += q[i]; return L;
}

static double check(std::vector<ParamRef>& params, const double* analytic_src_unused,
                    std::function<double()> loss, std::vector<ParamRef>& grad_params) {
    (void)analytic_src_unused;
    const double eps = 1e-5;
    double maxd = 0;
    int gi = 0;
    for (auto& blk : grad_params) {
        for (int i = 0; i < blk.n; ++i) {
            Real save = blk.data[i];
            blk.data[i] = save + (Real)eps; double Lp = loss();
            blk.data[i] = save - (Real)eps; double Lm = loss();
            blk.data[i] = save;
            double fd = (Lp - Lm) / (2 * eps);
            double an = blk.grad[i];
            maxd = std::max(maxd, std::abs(fd - an));
            ++gi;
        }
    }
    (void)params; (void)gi;
    return maxd;
}

int main(int argc, char** argv) {
    if (argc > 1) DATA = std::string(argv[1]) + "/";
    // FD in float32 is only ~1e-3 accurate (eps=1e-5, float precision ~1e-7); the rigorous
    // gradcheck is the FP64 build. Use a precision-appropriate tolerance.
    const double TOL = (sizeof(Real) == 8) ? 1e-5 : 5e-3;
    auto aw = pio::load_flat(DATA + "actor.bin");
    auto cw = pio::load_flat(DATA + "critic.bin");
    FILE* f = pio::open_rd(DATA + "ref_fwd.bin");
    int Bfull = pio::rd_i32(f);
    auto sd = pio::rd_doubles(f, Bfull*STATE_DIM);
    auto ad = pio::rd_doubles(f, Bfull);
    std::fclose(f);

    int B = 16;                          // small batch keeps FD cheap
    std::vector<Real> s(B*STATE_DIM), a(B);
    for (int i=0;i<B*STATE_DIM;++i) s[i]=(Real)sd[i];
    for (int i=0;i<B;++i) a[i]=(Real)ad[i];

    Actor actor; Critic critic;
    actor.load_flat(aw.data()); critic.load_flat(cw.data());
    actor.set_gate(1.0,0.0,2.0,0.04,2.0); critic.set_strike(1.0);

    int fails = 0;

    // ---- critic grad ----
    {
        std::vector<Real> q(B); critic.forward(s.data(), a.data(), B, q.data());
        std::vector<Real> dq(B, (Real)1);
        critic.zero_grad(); critic.backward(dq.data(), B);
        std::vector<ParamRef> gp; critic.collect_params(gp, 0.0);
        double md = check(gp, nullptr, [&]{ return critic_loss(critic, s, a, B); }, gp);
        std::printf("[critic grad] max|fd-analytic|=%.3e\n", md);
        if (md > TOL) { std::printf("  FAIL critic grad\n"); ++fails; }
    }
    // ---- actor grad (DPG path through critic) ----
    // Disable the profitability gate (c=0 => identity) so the STE clamp does not
    // mask the true derivative; the STE wrapper is identity-by-construction and is
    // covered by the gate-ON forward parity test.
    actor.set_gate(1.0, 0.0, 2.0, 0.0, 2.0);
    {
        std::vector<Real> ap(B), q(B);
        actor.forward(s.data(), B, ap.data());
        critic.forward(s.data(), ap.data(), B, q.data());
        std::vector<Real> dqv(B, (Real)1), gact(B);
        critic.zero_grad(); critic.backward(dqv.data(), B, gact.data());
        actor.zero_grad(); actor.backward(gact.data(), B);
        std::vector<ParamRef> gp; actor.collect_params(gp, 0.0);
        double md = check(gp, nullptr, [&]{ return actor_loss(actor, critic, s, B); }, gp);
        std::printf("[actor grad]  max|fd-analytic|=%.3e\n", md);
        if (md > TOL) { std::printf("  FAIL actor grad\n"); ++fails; }
    }
    std::printf(fails ? "GRADCHECK: FAIL\n" : "GRADCHECK: PASS\n");
    return fails ? 1 : 0;
}
