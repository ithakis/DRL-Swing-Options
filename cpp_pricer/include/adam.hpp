// adam.hpp — decoupled AdamW over a flat list of ParamRef blocks.
#pragma once
#include "config.hpp"
#include "mlp.hpp"
#include <vector>
#include <cmath>

namespace pricer {

class AdamW {
public:
    void init(const std::vector<ParamRef>& params, double lr, double b1, double b2, double eps = 1e-8) {
        params_ = params; lr_ = lr; b1_ = b1; b2_ = b2; eps_ = eps; t_ = 0;
        size_t n = 0; for (auto& p : params_) n += p.n;
        m_.assign(n, 0); v_.assign(n, 0);
        base_lr_ = lr;
    }

    // Runtime LR override (v61 cosine/linear LR schedule). base_lr() returns the init LR
    // captured at init() so the schedule can scale relative to it.
    void set_lr(double lr) { lr_ = lr; }
    double base_lr() const { return base_lr_; }

    void step() {
        ++t_;
        const double bc1 = 1.0 - std::pow(b1_, (double)t_);
        const double bc2 = 1.0 - std::pow(b2_, (double)t_);
        const double inv_bc1 = 1.0 / bc1, inv_bc2 = 1.0 / bc2;
        size_t off = 0;
        for (auto& blk : params_) {
            const double wd = blk.weight_decay;
            for (int i = 0; i < blk.n; ++i) {
                const double g = blk.grad[i];
                double m = b1_ * m_[off+i] + (1.0 - b1_) * g;
                double v = b2_ * v_[off+i] + (1.0 - b2_) * g * g;
                m_[off+i] = m; v_[off+i] = v;
                double mhat = m * inv_bc1;
                double vhat = v * inv_bc2;
                double p = blk.data[i];
                if (wd != 0.0) p -= lr_ * wd * p;              // decoupled weight decay
                p -= lr_ * mhat / (std::sqrt(vhat) + eps_);
                blk.data[i] = (Real)p;
            }
            off += blk.n;
        }
    }

private:
    std::vector<ParamRef> params_;
    std::vector<double> m_, v_;
    double lr_=0, base_lr_=0, b1_=0.9, b2_=0.999, eps_=1e-8;
    long t_ = 0;
};

} // namespace pricer
