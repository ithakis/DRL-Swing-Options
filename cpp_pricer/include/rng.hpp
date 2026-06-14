// rng.hpp — fast splittable RNG (xoshiro256**) + normal/uniform draws.
// Each path-stream or worker gets an independent seed; reproducible per seed.
#pragma once
#include <cstdint>
#include <cmath>

namespace pricer {

class Rng {
public:
    explicit Rng(uint64_t seed = 0) { reseed(seed); }

    void reseed(uint64_t seed) {
        // SplitMix64 to fill the xoshiro state from a single seed.
        uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < 4; ++i) {
            z += 0x9E3779B97F4A7C15ULL;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            s_[i] = x ^ (x >> 31);
        }
        has_spare_ = false;
    }

    inline uint64_t next_u64() {
        const uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform in (0,1) — 53-bit mantissa, never exactly 0 or 1.
    inline double uniform() {
        const uint64_t x = (next_u64() >> 11) + 1;        // [1, 2^53]
        return x * (1.0 / 9007199254740993.0);            // (0,1)
    }

    // Standard normal via Marsaglia polar (caches the spare).
    inline double normal() {
        if (has_spare_) { has_spare_ = false; return spare_; }
        double u, v, s;
        do {
            u = 2.0 * uniform() - 1.0;
            v = 2.0 * uniform() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        const double f = std::sqrt(-2.0 * std::log(s) / s);
        spare_ = v * f;
        has_spare_ = true;
        return u * f;
    }

    inline int poisson(double mean) {
        // Knuth's algorithm — fine for the tiny means here (lam*dt ~ 0.024).
        const double L = std::exp(-mean);
        int k = 0;
        double p = 1.0;
        do { ++k; p *= uniform(); } while (p > L);
        return k - 1;
    }

private:
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t s_[4];
    bool has_spare_ = false;
    double spare_ = 0.0;
};

} // namespace pricer
