#!/usr/bin/env python
"""GOAL 4/5 — accuracy/time Pareto frontier for the DP, and the recommended production config.

Sweeps a ladder of grid resolutions, records (price, wall-clock), measures error against a
high-resolution reference, finds the Pareto frontier and the cheapest config hitting the 1e-4
and 1e-3 accuracy bars.  Writes data/pareto.csv (+ docs/figs/pareto.png if matplotlib).

    python grid_dp_pricer/tools/pareto.py [--bin build/price_dp] [--threads 8]
"""
import argparse
import csv
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GDP = os.path.abspath(os.path.join(HERE, ".."))

# (nX, nY, nQ, Mx) ladder, coarse -> fine.
LADDER = [
    (41, 41, 51, 8), (51, 51, 61, 10), (61, 61, 76, 12), (81, 81, 101, 16),
    (101, 101, 121, 20), (121, 121, 151, 24), (141, 141, 176, 28), (161, 161, 201, 32),
]
REFERENCE_CFG = (201, 201, 251, 36)  # high-res reference for the error metric


def run(binpath, cfg, threads, interp=1):
    nX, nY, nQ, Mx = cfg
    args = [binpath, "--no-store", "--threads", str(threads), "--interp", str(interp),
            "--nX", str(nX), "--nY", str(nY), "--nQ", str(nQ), "--Mx", str(Mx),
            "--Yhi", "4.0", "--Xhi", "1.6", "--Xlo", "-1.6"]
    j = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    return j["price"], j["t_total_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(GDP, "build", "price_dp"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--interp", type=int, default=1)
    args = ap.parse_args()

    ref_price, _ = run(args.bin, REFERENCE_CFG, args.threads, args.interp)
    print(f"reference price ({REFERENCE_CFG}) = {ref_price:.6f}\n")

    rows = []
    print(f"{'nX':>4} {'nY':>4} {'nQ':>4} {'Mx':>3} | {'price':>10} {'|err|':>9} {'t_total(s)':>10}")
    print("-" * 56)
    for cfg in LADDER:
        p, t = run(args.bin, cfg, args.threads, args.interp)
        err = abs(p - ref_price)
        rows.append(dict(nX=cfg[0], nY=cfg[1], nQ=cfg[2], Mx=cfg[3], price=p, err=err,
                         t_s=t / 1000.0))
        print(f"{cfg[0]:>4} {cfg[1]:>4} {cfg[2]:>4} {cfg[3]:>3} | {p:>10.6f} {err:>9.2e} {t/1000:>10.3f}")

    # Pareto frontier: configs not dominated (lower time AND lower error) by another.
    pareto = [r for r in rows if not any(o["t_s"] <= r["t_s"] and o["err"] < r["err"] for o in rows)]
    print("\nPareto-optimal configs (err, time):")
    for r in sorted(pareto, key=lambda r: r["t_s"]):
        print(f"  nX{r['nX']} nY{r['nY']} nQ{r['nQ']} Mx{r['Mx']}: err={r['err']:.2e}  t={r['t_s']:.3f}s")

    for bar in (1e-3, 1e-4):
        ok = [r for r in rows if r["err"] <= bar]
        if ok:
            best = min(ok, key=lambda r: r["t_s"])
            print(f"cheapest <= {bar:.0e} abs: nX{best['nX']} nY{best['nY']} nQ{best['nQ']} "
                  f"Mx{best['Mx']}  t={best['t_s']:.3f}s")
        else:
            print(f"no config in the ladder reached <= {bar:.0e}")

    os.makedirs(os.path.join(GDP, "data"), exist_ok=True)
    with open(os.path.join(GDP, "data", "pareto.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.loglog([r["t_s"] for r in rows], [r["err"] + 1e-12 for r in rows], "o", color="#888")
        pf = sorted(pareto, key=lambda r: r["t_s"])
        ax.loglog([r["t_s"] for r in pf], [r["err"] + 1e-12 for r in pf], "-o", color="#1f77b4",
                  label="Pareto frontier")
        for bar in (1e-3, 1e-4):
            ax.axhline(bar, ls="--", color="#d62728", alpha=0.5)
        ax.set_xlabel("wall-clock (s)"); ax.set_ylabel("|price - reference|")
        ax.set_title("DP accuracy / time Pareto frontier"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.join(GDP, "docs", "figs"), exist_ok=True)
        fig.savefig(os.path.join(GDP, "docs", "figs", "pareto.png"), dpi=130)
        print(f"wrote {os.path.join(GDP, 'docs', 'figs', 'pareto.png')}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
