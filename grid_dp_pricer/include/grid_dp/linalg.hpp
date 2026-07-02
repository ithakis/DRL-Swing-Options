// linalg.hpp — row-major GEMM wrapper over Apple Accelerate (cblas_{d,s}gemm),
// with a portable naive fallback when GRID_DP_ACCELERATE=0.
#pragma once

#include "grid_dp/config.hpp"

#if defined(GRID_DP_ACCELERATE) && GRID_DP_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace grid_dp {

// C(M x N) = alpha * op(A) * op(B) + beta * C, all row-major.
//   op(A) is (M x K), op(B) is (K x N).  lda/ldb/ldc are the leading (row) strides
//   of the *stored* (untransposed) matrices.
inline void gemm(bool transA, bool transB, int M, int N, int K,
                 Scalar alpha, const Scalar* A, int lda,
                 const Scalar* B, int ldb, Scalar beta, Scalar* C, int ldc) {
#if defined(GRID_DP_ACCELERATE) && GRID_DP_ACCELERATE
    const CBLAS_TRANSPOSE ta = transA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE tb = transB ? CblasTrans : CblasNoTrans;
#if GRID_DP_FP64
    cblas_dgemm(CblasRowMajor, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    cblas_sgemm(CblasRowMajor, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#else
    // Portable reference (correctness only; not tuned).
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            Scalar acc = 0;
            for (int k = 0; k < K; ++k) {
                const Scalar a = transA ? A[k * lda + i] : A[i * lda + k];
                const Scalar b = transB ? B[j * ldb + k] : B[k * ldb + j];
                acc += a * b;
            }
            C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
        }
    }
#endif
}

} // namespace grid_dp
