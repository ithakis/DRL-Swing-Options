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
        cap_ = capacity; size_ = 0; head_ = 0; weighted_ = false; wtot_ = 0.0;
        s_.assign((size_t)cap_*STATE_DIM, 0);
        ns_.assign((size_t)cap_*STATE_DIM, 0);
        a_.assign(cap_, 0); r_.assign(cap_, 0); d_.assign(cap_, 0);
    }
    // Task 3 (A2/A4): enable per-transition weighted sampling via a Fenwick tree.
    // Bit-identical to uniform when never enabled (separate add/sample paths).
    void enable_weighted() {
        weighted_ = true; wtot_ = 0.0;
        w_.assign(cap_, 0.0);
        fen_.assign(cap_ + 1, 0.0);
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

    // Weighted insert (A2/A4): same storage as add() plus a per-slot sampling weight.
    void add_w(const Real* s, Real a, Real r, const Real* ns, bool done, double weight) {
        int slot = head_;
        Real* sd = s_.data()+(size_t)slot*STATE_DIM;
        Real* nd = ns_.data()+(size_t)slot*STATE_DIM;
        for (int i=0;i<STATE_DIM;++i){ sd[i]=s[i]; nd[i]=ns[i]; }
        a_[slot]=a; r_[slot]=r; d_[slot]= done?Real(1):Real(0);
        if (weight < 1e-12) weight = 1e-12;
        fen_update(slot, weight - w_[slot]);   // overwrite-aware
        wtot_ += weight - w_[slot];
        w_[slot] = (Real)weight;
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

    // Weighted batch sampling (proportional to per-slot weight; with replacement).
    void sample_weighted(int B, Rng& rng, Real* sb, Real* ab, Real* rb, Real* nsb, Real* db) {
        for (int j=0;j<B;++j){
            double target = rng.uniform() * wtot_;
            int i = fen_find(target);
            if (i < 0) i = 0; if (i >= size_) i = size_ - 1;
            std::copy(s_.data()+(size_t)i*STATE_DIM, s_.data()+(size_t)i*STATE_DIM+STATE_DIM, sb+(size_t)j*STATE_DIM);
            std::copy(ns_.data()+(size_t)i*STATE_DIM, ns_.data()+(size_t)i*STATE_DIM+STATE_DIM, nsb+(size_t)j*STATE_DIM);
            ab[j]=a_[i]; rb[j]=r_[i]; db[j]=d_[i];
        }
    }

private:
    int cap_=0, size_=0, head_=0;
    std::vector<Real> s_, ns_, a_, r_, d_;

    // --- Fenwick (BIT) over slot weights for O(log cap) weighted sampling ---
    bool weighted_=false;
    double wtot_=0.0;
    std::vector<Real> w_;       // per-slot current weight
    std::vector<double> fen_;   // 1-indexed Fenwick of weights

    void fen_update(int slot, double delta) {
        for (int i = slot + 1; i <= cap_; i += i & (-i)) fen_[i] += delta;
    }
    // smallest 0-based slot whose prefix-weight strictly exceeds `target`.
    int fen_find(double target) const {
        int pos = 0; double rem = target;
        int LOG = 1; while ((1 << LOG) <= cap_) ++LOG;
        for (int pw = (1 << LOG); pw > 0; pw >>= 1) {
            int nxt = pos + pw;
            if (nxt <= cap_ && fen_[nxt] <= rem) { rem -= fen_[nxt]; pos = nxt; }
        }
        return pos;   // 0-based slot index = pos (Fenwick pos points past accumulated cells)
    }
};

} // namespace pricer
