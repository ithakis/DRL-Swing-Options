// linalg.hpp — minimal row-major dense ops + fused NN primitives (forward/backward).
// Hand-rolled for the 9..65-wide layers; the hot 64-wide GEMV dominates and a tiny
// hand kernel beats a BLAS call (no dispatch overhead) for these sizes.
#pragma once
#include "config.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#ifdef CPP_PRICER_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace pricer {

constexpr Real LN_EPS = static_cast<Real>(1e-5);   // matches torch.nn.LayerNorm default

// Fast float exp (~1e-5 rel), FP32 path only; the FP64 parity build uses std::exp.
inline float fast_expf(float x) {
    x = x < -87.0f ? -87.0f : (x > 87.0f ? 87.0f : x);
    float z = x * 1.4426950408889634f;        // log2(e)
    float zi = std::floor(z);
    float zf = z - zi;
    float p = 1.0f + zf*(0.6931471805f + zf*(0.2402265069f + zf*(0.0555041087f
              + zf*(0.0096181291f + zf*0.0013333558f))));   // 2^zf Taylor (deg 5)
    union { float f; int32_t i; } u;
    u.i = (int32_t)(((int)zi + 127) << 23);   // 2^zi
    return p * u.f;
}

inline Real sigmoidf(Real x) {
#ifdef CPP_PRICER_FP64
    return Real(1) / (Real(1) + std::exp(-x));
#else
    return 1.0f / (1.0f + fast_expf(-x));
#endif
}
inline Real siluf(Real x) { return x * sigmoidf(x); }

// ---- Linear: Y[B,out] = X[B,in] · W[out,in]^T + b[out] -------------------
inline void linear_forward(const Real* __restrict W, const Real* __restrict b,
                           const Real* __restrict X, Real* __restrict Y,
                           int B, int in, int out) {
#ifdef CPP_PRICER_ACCELERATE
    // Y = X (B×in) · W^T (in×out) via tuned BLAS, then add bias.
    const Real alpha = 1, beta = 0;
  #ifdef CPP_PRICER_FP64
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, out, in,
                alpha, X, in, W, in, beta, Y, out);
  #else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, out, in,
                alpha, X, in, W, in, beta, Y, out);
  #endif
    if (b) for (int r = 0; r < B; ++r) { Real* yr = Y + (size_t)r*out; for (int o = 0; o < out; ++o) yr[o] += b[o]; }
#else
    for (int r = 0; r < B; ++r) {
        const Real* __restrict xr = X + (size_t)r * in;
        Real* __restrict yr = Y + (size_t)r * out;
        for (int o = 0; o < out; ++o) {
            const Real* __restrict wo = W + (size_t)o * in;
            Real acc = b ? b[o] : Real(0);
            for (int i = 0; i < in; ++i) acc += wo[i] * xr[i];
            yr[o] = acc;
        }
    }
#endif
}

// grad wrt input, plus (optionally) accumulate grads wrt W,b.  gX may be null.
// H-S6 (2026-06-13): when accum_params=false, skip the gW/gb GEMM+reduction entirely.
// Used by the DPG actor update, where the critic's param grads are computed only to be
// discarded (the critic zero_grads before its own step) — only the action gradient (gX)
// is consumed.  Skipping the second GEMM is bit-identical and ~halves the critic backward.
inline void linear_backward(const Real* W, const Real* X, const Real* gY,
                            Real* gX, Real* gW, Real* gb,
                            int B, int in, int out, bool accum_params = true) {
#ifdef CPP_PRICER_ACCELERATE
    const Real one = 1, zero = 0;
  #ifdef CPP_PRICER_FP64
    #define GEMM cblas_dgemm
  #else
    #define GEMM cblas_sgemm
  #endif
    if (gX)  // gX(B×in) = gY(B×out) · W(out×in)
        GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, B, in, out, one, gY, out, W, in, zero, gX, in);
    if (accum_params) {
        // gW(out×in) += gY^T(out×B) · X(B×in)
        GEMM(CblasRowMajor, CblasTrans, CblasNoTrans, out, in, B, one, gY, out, X, in, one, gW, in);
        if (gb) for (int r = 0; r < B; ++r) { const Real* gyr = gY + (size_t)r*out; for (int o = 0; o < out; ++o) gb[o] += gyr[o]; }
    }
    #undef GEMM
#else
    if (gX) std::memset(gX, 0, sizeof(Real) * (size_t)B * in);
    for (int r = 0; r < B; ++r) {
        const Real* xr = X + (size_t)r * in;
        const Real* gyr = gY + (size_t)r * out;
        Real* gxr = gX ? gX + (size_t)r * in : nullptr;
        for (int o = 0; o < out; ++o) {
            const Real g = gyr[o];
            if (accum_params && gb) gb[o] += g;
            const Real* wo = W + (size_t)o * in;
            Real* gwo = gW + (size_t)o * in;
            for (int i = 0; i < in; ++i) {
                if (accum_params) gwo[i] += g * xr[i];
                if (gxr) gxr[i] += g * wo[i];
            }
        }
    }
#endif
}

// ---- LayerNorm over last dim D (affine), caches mean/rstd/xhat -----------
struct LNCache { std::vector<Real> mean, rstd, xhat; };

inline void layernorm_forward(const Real* x, const Real* gamma, const Real* beta,
                              Real* y, int B, int D, LNCache& c) {
    c.mean.resize(B); c.rstd.resize(B); c.xhat.resize((size_t)B * D);
    for (int r = 0; r < B; ++r) {
        const Real* xr = x + (size_t)r * D;
        Real m = 0; for (int i = 0; i < D; ++i) m += xr[i]; m /= D;
        Real v = 0; for (int i = 0; i < D; ++i) { Real d = xr[i] - m; v += d * d; } v /= D;
        Real rs = Real(1) / std::sqrt(v + LN_EPS);
        c.mean[r] = m; c.rstd[r] = rs;
        Real* yr = y + (size_t)r * D;
        Real* xh = c.xhat.data() + (size_t)r * D;
        for (int i = 0; i < D; ++i) { Real h = (xr[i] - m) * rs; xh[i] = h; yr[i] = h * gamma[i] + beta[i]; }
    }
}

inline void layernorm_backward(const Real* gy, const Real* gamma, const LNCache& c,
                               Real* gx, Real* ggamma, Real* gbeta, int B, int D,
                               bool accum_params = true) {
    for (int r = 0; r < B; ++r) {
        const Real* gyr = gy + (size_t)r * D;
        const Real* xh = c.xhat.data() + (size_t)r * D;
        Real rs = c.rstd[r];
        Real sum_dy = 0, sum_dy_xhat = 0;
        for (int i = 0; i < D; ++i) {
            Real dy = gyr[i] * gamma[i];
            sum_dy += dy;
            sum_dy_xhat += dy * xh[i];
            if (accum_params) { ggamma[i] += gyr[i] * xh[i]; gbeta[i] += gyr[i]; }
        }
        Real* gxr = gx + (size_t)r * D;
        Real invD = Real(1) / D;
        for (int i = 0; i < D; ++i) {
            Real dy = gyr[i] * gamma[i];
            gxr[i] = rs * (dy - invD * sum_dy - xh[i] * invD * sum_dy_xhat);
        }
    }
}

// NOTE (dead end, R1 2026-06-13): fusing LayerNorm+SiLU into one row sweep was
// MEASURED ~9% SLOWER (517 vs 476 us/step). The standalone silu_forward over a
// flat B*H array auto-vectorizes fast_expf far better than when interleaved into
// the row-serial LayerNorm reduction. Keep them separate.

// ---- hidden activation (in place semantics handled by caller via separate buffers) ----
// Default = SiLU.  -DPRICER_GELU swaps in exact (erf-form) GELU *under the same names*
// (so no mlp.cpp call-site edits; test_grad_depth gradchecks whichever is compiled).
// H-A2: one-shot accuracy probe of the smooth-gated-unit cluster; GELU is strictly
// costlier than SiLU (erf) so it can only move accuracy, never wall-clock.
#ifdef PRICER_GELU
inline void silu_forward(const Real* x, Real* y, int n) {
    for (int i = 0; i < n; ++i)
        y[i] = x[i] * Real(0.5) * (Real(1) + std::erf(x[i] * Real(0.7071067811865476)));
}
inline void silu_backward(const Real* x, const Real* gy, Real* gx, int n) {
    for (int i = 0; i < n; ++i) {
        Real cdf = Real(0.5) * (Real(1) + std::erf(x[i] * Real(0.7071067811865476)));
        Real pdf = Real(0.3989422804014327) * std::exp(Real(-0.5) * x[i] * x[i]);
        gx[i] = gy[i] * (cdf + x[i] * pdf);
    }
}
#elif defined(PRICER_GELU_FAST)
// H-A2: swish-with-slope  g(x) = x·σ(β·x).  β=1.702 ≈ GELU; β=1 = SiLU.  Reuses the
// fast_expf sigmoid (≈SiLU-cost).  β is the tunable gate steepness — set -DGELU_SLOPE=… to sweep.
#ifndef GELU_SLOPE
#define GELU_SLOPE 1.702
#endif
inline void silu_forward(const Real* x, Real* y, int n) {
    const Real beta = Real(GELU_SLOPE);
    for (int i = 0; i < n; ++i) y[i] = x[i] * sigmoidf(beta * x[i]);
}
inline void silu_backward(const Real* x, const Real* gy, Real* gx, int n) {
    const Real beta = Real(GELU_SLOPE);
    for (int i = 0; i < n; ++i) {
        Real s = sigmoidf(beta * x[i]);
        gx[i] = gy[i] * (s + beta * x[i] * s * (Real(1) - s));
    }
}
#else
inline void silu_forward(const Real* x, Real* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = siluf(x[i]);
}
// grad wrt pre-activation given x cached:  d/dx (x*sig(x)) = sig(x)*(1+x*(1-sig(x)))
inline void silu_backward(const Real* x, const Real* gy, Real* gx, int n) {
    for (int i = 0; i < n; ++i) {
        Real s = sigmoidf(x[i]);
        gx[i] = gy[i] * (s * (Real(1) + x[i] * (Real(1) - s)));
    }
}
#endif

} // namespace pricer
