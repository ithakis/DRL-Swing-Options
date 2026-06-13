// kernel.hpp — semi-analytical HHK transition kernel + expected critic target.
// The tiny quadrature mesh is loaded from a Python-exported binary so the C++
// target is bit-faithful to transition_kernel.expected_critic_target.
#pragma once
#include "config.hpp"
#include "mlp.hpp"
#include <vector>
#include <string>

namespace pricer {

struct TransitionKernel {
    double decay_X = 0, sigma_X = 0, decay_Y = 0;
    int n_rights = 0;
    std::vector<double> z_X, w_X;        // (M_x)
    std::vector<double> delta_Y, w_Y;    // (M_y)
    std::vector<double> exp_f_grid;      // (n_rights+1)
    std::vector<Real>   w_flat;          // (M) = w_X (outer) w_Y, row-major m=ix*M_y+iy

    int M_x() const { return (int)z_X.size(); }
    int M_y() const { return (int)delta_Y.size(); }
    int M()   const { return M_x() * M_y(); }

    void finalize();                     // builds w_flat from w_X,w_Y
    static TransitionKernel load(const std::string& path);
    // Build the mesh analytically in C++ (M_x=2 fast config), approximating the
    // single jump node by its conditional mean — used when no mesh file is given.
    static TransitionKernel build_fast(const HHKParams&, const SwingContract&, const KernelParams&);
};

// Reusable scratch to avoid per-call allocation in the hot training loop.
struct KernelWorkspace {
    std::vector<Real> flat;     // (B*M * 9)
    std::vector<Real> act;      // (B*M)
    std::vector<Real> q;        // (B*M)
    void ensure(int BM) { flat.resize((size_t)BM*STATE_DIM); act.resize(BM); q.resize(BM); }
};

// q_next[B] = sum_m w_m * critic_target(s'_m, actor_target(s'_m)).
void expected_critic_target(Actor& actor_t, Critic& critic_t,
                            const Real* states, const Real* next_states, int B,
                            const TransitionKernel& k, double strike,
                            KernelWorkspace& ws, Real* q_next);

} // namespace pricer
