#!/usr/bin/env python
"""GOAL 3 — convergence analysis.  Sweep each grid/quadrature axis independently (others held
at a fine reference), fit the observed order error ~ C h^p, Richardson-extrapolate a converged
price + error bar, and identify the controlling axis.  Writes data/convergence.csv and, if
matplotlib is available, docs/figs/convergence.png.

    python grid_dp_pricer/tools/convergence_analysis.py            # uses ./build/price_dp
"""
import csv
import json
import math
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GDP = os.path.abspath(os.path.join(HERE, ".."))
BIN = os.path.join(GDP, "build", "price_dp")

# Fine reference held on the non-swept axes (each is already converged to ~1e-4, so it does
# not contaminate the order estimate of the swept axis, while keeping runtime tractable).
REF = dict(nX=121, nY=121, nQ=151, Mx=24, Yhi=4.0, Xhi=1.6, Xlo=-1.6)


def price(**kw):
    args = [BIN, "--no-store", "--threads", "0"]
    for k, v in {**REF, **kw}.items():
        args += [f"--{k}", str(v)]
    j = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    return j["price"], j["t_backward_ms"]


# (axis flag, label, h(n) mapping, list of resolutions). h is the *grid spacing* (algebraic
# axes) — for Mx (spectral) we just track node count.
SWEEPS = {
    "nX": ("X nodes", lambda n: 3.2 / (n - 1), [41, 61, 81, 121, 161, 241]),
    "nY": ("Y nodes", lambda n: 4.0 / (n - 1), [41, 61, 81, 121, 161, 241]),
    "nQ": ("Q nodes", lambda n: 20.0 / (n - 1), [51, 76, 101, 151, 201, 301]),
    "Mx": ("GH nodes", lambda n: 1.0 / n, [6, 8, 12, 16, 24, 32]),
}


def richardson(hs, ps):
    """Estimate order p and extrapolated limit from the three finest points."""
    (h0, p0), (h1, p1), (h2, p2) = zip(hs[-3:], ps[-3:])
    d01, d12 = p1 - p0, p2 - p1
    if d01 == 0 or d12 == 0 or (d12 / d01) <= 0:
        return float("nan"), ps[-1]
    r = h0 / h1  # ~2
    p_order = math.log(abs(d01 / d12)) / math.log(r)
    extr = p2 + d12 / (r ** p_order - 1.0)
    return p_order, extr


def main():
    rows = []
    summary = {}
    for axis, (label, hf, res) in SWEEPS.items():
        hs, ps, ts = [], [], []
        for n in res:
            p, t = price(**{axis: n})
            hs.append(hf(n)); ps.append(p); ts.append(t)
            rows.append(dict(axis=axis, n=n, h=hf(n), price=p, t_backward_ms=t))
        order, extr = richardson(hs, ps)
        summary[axis] = dict(label=label, order=order, extr=extr,
                             finest=ps[-1], err_at_ref=abs(ps[-1] - extr))
        print(f"[{axis:>3}] {label:<9} order p~{order:>5.2f}  extrap={extr:.6f}  "
              f"finest({res[-1]})={ps[-1]:.6f}  |finest-extrap|={abs(ps[-1]-extr):.2e}")

    ctrl = max(summary, key=lambda a: summary[a]["err_at_ref"])
    print(f"\nControlling axis at the reference resolution: {ctrl} "
          f"(residual {summary[ctrl]['err_at_ref']:.2e})")
    # Combined Richardson estimate: finest full-reference price +/- summed residuals.
    ref_price, _ = price()
    band = sum(s["err_at_ref"] for s in summary.values())
    print(f"Reference-grid price {ref_price:.6f}  ~converged {ref_price:.6f} +/- {band:.1e}")

    os.makedirs(os.path.join(GDP, "data"), exist_ok=True)
    with open(os.path.join(GDP, "data", "convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["axis", "n", "h", "price", "t_backward_ms"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {os.path.join(GDP, 'data', 'convergence.csv')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for axis, (label, hf, res) in SWEEPS.items():
            sub = [r for r in rows if r["axis"] == axis]
            extr = summary[axis]["extr"]
            err = [abs(r["price"] - extr) + 1e-12 for r in sub]
            ax[0].loglog([r["h"] for r in sub], err, "o-", label=f"{axis} (p~{summary[axis]['order']:.1f})")
            ax[1].semilogy([r["n"] for r in sub], err, "o-", label=axis)
        ax[0].set_xlabel("grid spacing h"); ax[0].set_ylabel("|price - extrapolated|")
        ax[0].set_title("Convergence order (log-log)"); ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)
        ax[1].set_xlabel("nodes / quadrature order"); ax[1].set_ylabel("|error|")
        ax[1].set_title("Error vs resolution"); ax[1].legend(); ax[1].grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.join(GDP, "docs", "figs"), exist_ok=True)
        out = os.path.join(GDP, "docs", "figs", "convergence.png")
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
