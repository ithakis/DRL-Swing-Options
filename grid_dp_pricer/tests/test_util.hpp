// test_util.hpp — tiny assert-based test harness (no external framework).
#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace gdp_test {

inline int g_failures = 0;
inline int g_checks = 0;

inline void expect_true(bool cond, const std::string& msg) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::printf("  [FAIL] %s\n", msg.c_str());
    }
}

inline void expect_near(double got, double want, double tol, const std::string& msg) {
    ++g_checks;
    const double err = std::fabs(got - want);
    if (!(err <= tol)) {
        ++g_failures;
        std::printf("  [FAIL] %s : got=%.16g want=%.16g |err|=%.3e tol=%.3e\n",
                    msg.c_str(), got, want, err, tol);
    }
}

inline void expect_le(double got, double bound, const std::string& msg) {
    ++g_checks;
    if (!(got <= bound)) {
        ++g_failures;
        std::printf("  [FAIL] %s : got=%.16g must be <= %.16g\n", msg.c_str(), got, bound);
    }
}

inline int summary(const char* suite) {
    std::printf("[%s] %d checks, %d failures\n", suite, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

} // namespace gdp_test
