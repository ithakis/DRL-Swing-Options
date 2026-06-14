// test_sim.cpp — validate HHK simulation against closed-form moments at T.
#include "hhk_sim.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace pricer;

// Closed-form E[S_T], Std[S_T] for the HHK model (f==0), from theoretical_moments.
static void theo(const HHKParams& h, double T, double& ES, double& sS) {
    double X0 = std::log(h.S0);
    double mX = X0 * std::exp(-h.alpha*T);
    double vX = h.sigma*h.sigma*(1.0-std::exp(-2*h.alpha*T))/(2*h.alpha);
    auto M = [&](double th){
        double mxp = std::exp(th*mX + 0.5*th*th*vX);
        double yp = std::pow((1.0 - th*h.mu_J*std::exp(-h.beta*T))/(1.0 - th*h.mu_J), h.lam/h.beta);
        return mxp*yp;
    };
    ES = M(1.0); double E2 = M(2.0); sS = std::sqrt(E2 - ES*ES);
}

int main() {
    HHKParams h; SwingContract c;
    int n_paths = 1<<18;   // 262144
    Paths p;
    simulate_hhk(h, c, n_paths, /*seed*/11, /*stratify*/false, 0, 4, p);

    int T = p.T;
    double ES, sS; theo(h, c.maturity, ES, sS);
    // empirical terminal moments
    double m=0; for (int i=0;i<n_paths;++i) m += p.Srow(i)[T-1]; m/=n_paths;
    double v=0; for (int i=0;i<n_paths;++i){ double d=p.Srow(i)[T-1]-m; v+=d*d; } v/=(n_paths-1);
    double sd=std::sqrt(v);

    double err_mean = std::abs(m-ES)/ES;
    double err_std  = std::abs(sd-sS)/sS;
    std::printf("E[S_T]:  emp=%.5f  theo=%.5f  relerr=%.3e\n", m, ES, err_mean);
    std::printf("Std[S_T]:emp=%.5f  theo=%.5f  relerr=%.3e\n", sd, sS, err_std);

    // also check S_0 and X_0
    double s0 = p.Srow(0)[0];
    std::printf("S_0=%.5f (expect %.5f)\n", s0, h.S0);

    int fails = (err_mean > 0.02 || err_std > 0.05) ? 1 : 0;   // MC tolerance
    std::printf(fails ? "SIM: FAIL\n" : "SIM: PASS\n");
    return fails;
}
