// replay.hpp — circular uniform replay buffer (SoA), batched sampling.
#pragma once
#include "config.hpp"
#include "mlp.hpp"
#include "rng.hpp"
#include <vector>

namespace pricer {

class ReplayBuffer {
public:
    void init(int capacity) {
        cap_ = capacity; size_ = 0; head_ = 0;
        s_.assign((size_t)cap_*STATE_DIM, 0);
        ns_.assign((size_t)cap_*STATE_DIM, 0);
        a_.assign(cap_, 0); r_.assign(cap_, 0); d_.assign(cap_, 0);
    }
    int size() const { return size_; }

    void add(const Real* s, Real a, Real r, const Real* ns, bool done) {
        Real* sd = s_.data()+(size_t)head_*STATE_DIM;
        Real* nd = ns_.data()+(size_t)head_*STATE_DIM;
        for (int i=0;i<STATE_DIM;++i){ sd[i]=s[i]; nd[i]=ns[i]; }
        a_[head_]=a; r_[head_]=r; d_[head_]= done?Real(1):Real(0);
        head_ = (head_+1)%cap_;
        if (size_ < cap_) ++size_;
    }

    // Fill batch buffers (B rows). Caller provides storage.
    void sample(int B, Rng& rng, Real* sb, Real* ab, Real* rb, Real* nsb, Real* db) {
        for (int j=0;j<B;++j){
            int i = (int)(rng.next_u64() % (uint64_t)size_);
            std::copy(s_.data()+(size_t)i*STATE_DIM, s_.data()+(size_t)i*STATE_DIM+STATE_DIM, sb+(size_t)j*STATE_DIM);
            std::copy(ns_.data()+(size_t)i*STATE_DIM, ns_.data()+(size_t)i*STATE_DIM+STATE_DIM, nsb+(size_t)j*STATE_DIM);
            ab[j]=a_[i]; rb[j]=r_[i]; db[j]=d_[i];
        }
    }

private:
    int cap_=0, size_=0, head_=0;
    std::vector<Real> s_, ns_, a_, r_, d_;
};

} // namespace pricer
