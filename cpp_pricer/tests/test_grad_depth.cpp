// test_grad_depth.cpp — fixture-free finite-difference gradcheck of the
// runtime-depth Actor/Critic (H-N1/H-N2 / O2). Validates the NEW code paths the
// PyTorch-fixture gradcheck (test_grad) cannot reach in this worktree:
//   - critic with 0 post-layers (depth-2: fc4 grad feeds the action layer directly)
//   - the depth loops at depth 2/3/4
//   - asymmetric actor/critic depths
// Random-inits the nets (no fixtures needed); FD vs analytic must match in FP64.
// Build with -DCPP_PRICER_FP64=ON (FP32 FD is only ~1e-3 accurate).
#include "mlp.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <functional>

using namespace pricer;

static void fill_net(Actor& a, Critic& c, uint64_t seed) {
    std::mt19937_64 g(seed);
    std::normal_distribution<double> N(0.0, 1.0);
    auto rand_lin = [&](Linear& L){
        double sd = std::sqrt(2.0) / std::sqrt((double)L.in);
        for (auto& w : L.W) w = (Real)(N(g) * sd);
        for (auto& b : L.b) b = (Real)(N(g) * 0.01);
    };
    auto rand_ln = [&](LayerNorm& Nn){
        for (auto& x : Nn.g) x = (Real)(1.0 + 0.05 * N(g));
        for (auto& x : Nn.b) x = (Real)(0.02 * N(g));
    };
    for (size_t i = 0; i < a.lin.size(); ++i) { rand_lin(a.lin[i]); rand_ln(a.ln[i]); }
    rand_lin(a.fc4);
    rand_lin(c.se_lin); rand_ln(c.se_ln);
    rand_lin(c.al_lin); rand_ln(c.al_ln);
    for (size_t i = 0; i < c.pl_lin.size(); ++i) { rand_lin(c.pl_lin[i]); rand_ln(c.pl_ln[i]); }
    rand_lin(c.fc4);
}

static double fd_check(std::function<double()> loss, std::vector<ParamRef>& gp) {
    const double eps = 1e-5;
    double maxd = 0;
    for (auto& blk : gp)
        for (int i = 0; i < blk.n; ++i) {
            Real save = blk.data[i];
            blk.data[i] = save + (Real)eps; double Lp = loss();
            blk.data[i] = save - (Real)eps; double Lm = loss();
            blk.data[i] = save;
            maxd = std::max(maxd, std::abs((Lp - Lm) / (2 * eps) - (double)blk.grad[i]));
        }
    return maxd;
}

static int one_case(int hidden, int a_layers, int c_layers, uint64_t seed, double tol) {
    int B = 12;
    Actor actor(hidden, a_layers); Critic critic(hidden, c_layers);
    fill_net(actor, critic, seed);
    actor.set_gate(1.0, 0.0, 2.0, 0.0, 2.0);   // c=0 => gate is identity (don't mask the derivative)
    critic.set_strike(1.0);

    std::mt19937_64 g(seed ^ 0x5151ULL);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<Real> s(B*STATE_DIM), a(B);
    for (auto& x : s) x = (Real)U(g);
    for (int i = 0; i < B; ++i) { s[(size_t)i*STATE_DIM+5] = (Real)(1.0 + 0.3*U(g)); a[i] = (Real)(0.5 + 0.3*U(g)); }

    int fails = 0;
    // critic params
    {
        std::vector<Real> q(B); critic.forward(s.data(), a.data(), B, q.data());
        std::vector<Real> dq(B, (Real)1);
        critic.zero_grad(); critic.backward(dq.data(), B);
        std::vector<ParamRef> gp; critic.collect_params(gp, 0.0);
        double md = fd_check([&]{ std::vector<Real> qq(B); critic.forward(s.data(),a.data(),B,qq.data());
                                  double L=0; for(int i=0;i<B;++i) L+=qq[i]; return L; }, gp);
        std::printf("  [critic aL=%d cL=%d H=%d] max|fd-an|=%.3e %s\n", a_layers, c_layers, hidden, md, md>tol?"FAIL":"ok");
        if (md > tol) ++fails;
    }
    // actor params (DPG path through critic; also exercises critic.backward g_action, ap=false implicitly off here)
    {
        std::vector<Real> ap(B), q(B);
        actor.forward(s.data(), B, ap.data());
        critic.forward(s.data(), ap.data(), B, q.data());
        std::vector<Real> dqv(B, (Real)1), gact(B);
        critic.zero_grad(); critic.backward(dqv.data(), B, gact.data());
        actor.zero_grad(); actor.backward(gact.data(), B);
        std::vector<ParamRef> gp; actor.collect_params(gp, 0.0);
        double md = fd_check([&]{ std::vector<Real> aa(B),qq(B); actor.forward(s.data(),B,aa.data());
                                  critic.forward(s.data(),aa.data(),B,qq.data());
                                  double L=0; for(int i=0;i<B;++i) L+=qq[i]; return L; }, gp);
        std::printf("  [actor  aL=%d cL=%d H=%d] max|fd-an|=%.3e %s\n", a_layers, c_layers, hidden, md, md>tol?"FAIL":"ok");
        if (md > tol) ++fails;
    }
    return fails;
}

int main() {
    const double TOL = (sizeof(Real) == 8) ? 1e-5 : 5e-3;
    int fails = 0;
    std::printf("gradcheck (depth/asymmetry), TOL=%.0e (Real=%zuB)\n", TOL, sizeof(Real));
    fails += one_case(48, 3, 3, 1, TOL);   // canonical depth-3
    fails += one_case(48, 2, 2, 2, TOL);   // depth-2 both (critic n_post=0 path)
    fails += one_case(48, 4, 4, 3, TOL);   // depth-4 both (critic n_post=2)
    fails += one_case(48, 2, 3, 4, TOL);   // asymmetric: shallow actor, deeper critic (H-N1)
    fails += one_case(32, 2, 3, 5, TOL);   // asymmetric + narrow
    std::printf(fails ? "GRADCHECK-DEPTH: FAIL\n" : "GRADCHECK-DEPTH: PASS\n");
    return fails ? 1 : 0;
}
