#!/usr/bin/env python3
"""Equivalence + speed analysis for cpp_pricer research rounds.

Compares two sets of per-seed price_swing JSONs (baseline vs variant), paired by
seed. Reports paired t, Welch two-sample t, and the TOST equivalence verdict
(90% CI of the paired mean difference must lie within +/- delta_eq).

Usage:
  research_equiv.py --base 'data/research/base_seed{s}.json' \
                    --var  'data/research/hk1_seed{s}.json' \
                    --seeds 11-25 [--delta_eq_frac 0.005]
"""
import argparse, glob, json, math, random, re, statistics as st


def load(pattern, seeds):
    out = {}
    for s in seeds:
        try:
            out[s] = json.load(open(pattern.replace("{s}", str(s))))
        except FileNotFoundError:
            pass
    return out


def t_cdf_sf(t, df):
    # two-sided p = P(|T|>|t|) = I_x(df/2, 1/2) with x = df/(df+t^2)  (no scipy dependency).
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def _betacf(a, b, x):
    # Lentz continued fraction for the regularized incomplete beta (Numerical Recipes).
    TINY = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < TINY: d = TINY
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY: d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY: c = TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY: d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY: c = TINY
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-12: break
    return h


def _betai(a, b, x):
    # regularized incomplete beta I_x(a,b); uses the converging CF in each region.
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_inv_90(df):
    # 95th percentile of t (for a 90% two-sided CI) via bisection on the sf.
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        # one-sided upper tail = 0.05  => two-sided = 0.10
        if t_cdf_sf(mid, df) / 2.0 > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--var", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--delta_eq_frac", type=float, default=0.005)
    a = ap.parse_args()

    m = re.match(r"(\d+)-(\d+)", a.seeds)
    seeds = list(range(int(m.group(1)), int(m.group(2)) + 1)) if m else [int(x) for x in a.seeds.split(",")]
    B, V = load(a.base, seeds), load(a.var, seeds)
    common = [s for s in seeds if s in B and s in V]

    bp = [B[s]["price"] for s in common]
    vp = [V[s]["price"] for s in common]
    diffs = [v - b for b, v in zip(bp, vp)]
    n = len(common)

    mb, mv = st.mean(bp), st.mean(vp)
    delta_eq = a.delta_eq_frac * mb
    md = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    df = n - 1
    tcrit = t_inv_90(df) if n > 1 else 0.0
    ci_lo, ci_hi = md - tcrit * se, md + tcrit * se
    equivalent = (ci_lo > -delta_eq) and (ci_hi < delta_eq)

    # paired t (two-sided) and Welch two-sample
    t_paired = md / se if se > 0 else float("inf")
    p_paired = t_cdf_sf(t_paired, df) if se > 0 else 0.0
    sb, sv = (st.stdev(bp) if n > 1 else 0), (st.stdev(vp) if n > 1 else 0)
    seW = math.sqrt(sb * sb / n + sv * sv / n) if n > 1 else 0
    tW = (mv - mb) / seW if seW > 0 else 0.0
    dfW = ((sb*sb/n + sv*sv/n)**2) / ((sb*sb/n)**2/(n-1) + (sv*sv/n)**2/(n-1)) if n > 1 else 1
    pW = t_cdf_sf(tW, dfW) if seW > 0 else 1.0

    print(f"n seeds paired      : {n}  ({common[0]}..{common[-1]})")
    print(f"baseline price      : {mb:.6f} +/- {sb:.6f} (sd)")
    print(f"variant  price      : {mv:.6f} +/- {sv:.6f} (sd)")
    print(f"paired mean diff    : {md:+.6f}  (sd {sd:.6f}, se {se:.6f})")
    print(f"  per-seed diffs    : min {min(diffs):+.6f}  max {max(diffs):+.6f}")
    print(f"delta_eq (+/-{a.delta_eq_frac*100:.2f}%): +/- {delta_eq:.6f}")
    print(f"90% CI of diff      : [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"TOST verdict        : {'EQUIVALENT' if equivalent else 'NOT equivalent'}")
    print(f"paired t-test       : t={t_paired:.3f} p={p_paired:.4f} (two-sided; shift test)")
    print(f"Welch two-sample    : t={tW:.3f} p={pW:.4f} df={dfW:.1f}")

    # ---- ACCURACY front: one-sided paired SUPERIORITY test (H1: variant price > baseline) ----
    if se > 0:
        sf = t_cdf_sf(abs(t_paired), df)           # two-sided P(|T| > |t|)
        p_sup = (sf / 2.0) if md > 0 else (1.0 - sf / 2.0)   # upper-tail P(T > t_paired)
    else:
        p_sup = 0.0 if md > 0 else 1.0
    sup_verdict = "SUPERIOR (price up)" if (md > 0 and p_sup < 0.05) else \
                  ("INFERIOR (price down)" if (md < 0 and (1 - p_sup) < 0.05) else "n.s.")
    print(f"accuracy 1-sided    : H1 variant>base  p={p_sup:.4f}  -> {sup_verdict}")

    # ---- VARIANCE front: Pitman-Morgan paired-variance test + bootstrap std-ratio CI ----
    print(f"seed-std base/var   : {sb:.6f} / {sv:.6f}  (ratio {sv/sb if sb else float('nan'):.3f}; <1 => variant tighter)")
    if n > 2 and sb > 0 and sv > 0:
        U = [v - b for b, v in zip(bp, vp)]        # = diffs
        Vv = [v + b for b, v in zip(bp, vp)]
        mu_, mv_ = st.mean(U), st.mean(Vv)
        cov = sum((u - mu_) * (w - mv_) for u, w in zip(U, Vv)) / (n - 1)
        su, sw = st.stdev(U), st.stdev(Vv)
        r = cov / (su * sw) if su > 0 and sw > 0 else 0.0
        r = max(-0.999999, min(0.999999, r))
        t_pm = r * math.sqrt((n - 2) / (1 - r * r))
        p_pm = t_cdf_sf(abs(t_pm), n - 2)
        # cov(U,V) = cov(v-b, v+b) = var(variant) - var(base); cov>0 => variant variance LARGER (wider)
        print(f"Pitman-Morgan var   : t={t_pm:.3f} p={p_pm:.4f}  "
              f"({'variant wider' if cov>0 else 'variant tighter'}; sig@.05={'yes' if p_pm<0.05 else 'no'})")
        # bootstrap 90% CI of the std-ratio (paired resample over seeds)
        random.seed(12345)
        ratios = []
        for _ in range(5000):
            idx = [random.randrange(n) for _ in range(n)]
            bb = [bp[i] for i in idx]; vv = [vp[i] for i in idx]
            sbb = st.pstdev(bb); svv = st.pstdev(vv)
            if sbb > 0:
                ratios.append(svv / sbb)
        ratios.sort()
        lo = ratios[int(0.05 * len(ratios))]; hi = ratios[int(0.95 * len(ratios)) - 1]
        var_sup = hi < 1.0
        print(f"std-ratio 90% CI    : [{lo:.3f}, {hi:.3f}]  -> "
              f"{'VARIANCE SUPERIOR (CI<1)' if var_sup else 'n.s.'}")

    # ---- TIME front (if t_train present) ----
    if all("t_train" in B[s] for s in common) and all("t_train" in V[s] for s in common):
        tb = st.mean([B[s]["t_train"] for s in common]); tv = st.mean([V[s]["t_train"] for s in common])
        print(f"t_train base/var    : {tb:.2f}s / {tv:.2f}s  (speedup {tb/tv:.2f}x, {100*(1-tv/tb):+.1f}%)")


if __name__ == "__main__":
    main()
