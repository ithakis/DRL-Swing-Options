#!/usr/bin/env python3
"""gen_table5_dp.py — regenerate the full body of Table 4 (tab:results).

Column layout (LSM-D benchmark FIRST, then DP treated like a priced method):

    c | gamma | LSM-D (M=5) price+-CI | DP: price , Delta% [CI] | AC-sample: price+-CI, Delta% [CI]
                                                                 | AC-kernel: price+-CI, Delta% [CI]

Every method (DP, AC-sample, AC-kernel) is reported the same way against the LSM-D
benchmark: a mean price and Delta% = 100*(V/V_LSM - 1) with an asymmetric 95% interval.

  * LSM-D / AC-sample / AC-kernel prices, half-widths, Delta% and Delta%-CI come from
    "Convex Costs Results 9.csv" (8-seed BCa bootstrap, produced by gen_results9_v67.py).
  * V^DP comes from grid_dp_pricer/results/dp_publication_sweep.csv.
  * DP is deterministic, so its Delta% interval is obtained by propagating the LSM-D price
    interval through the monotone map Delta%(L) = 100*(V_DP/L - 1) (endpoints swap):
        Delta%_lo = 100*(V_DP/L_hi - 1),  Delta%_hi = 100*(V_DP/L_lo - 1).
    This is exactly the "CI inherited from LSM" statement in the caption.

Cell shading of the Delta% columns replicates tools/colorize_results_table.py
(white at 0, green positive, red negative, intensity = min(|d|/CAP,1)*MAXMIX).

Run:  python tools/gen_table5_dp.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "Paper" / "DRL_Swing_Options.tex"
DPCSV = ROOT / "grid_dp_pricer" / "results" / "dp_publication_sweep.csv"
R9CSV = ROOT / "Jupyter Notebooks" / "Convex Costs Results 9.csv"

CAP = 5.2        # Delta% shading saturates at the most extreme cell
MAXMIX = 80      # cap the white-mix so saturated cells stay legible under black text

# Grid order in the table: zero-cost reference row, then c ascending x gamma ascending.
GAMMAS = [1.0, 1.5, 2.0, 3.0]
CS = [0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15]
ROWS = [(0.0, 1.0)] + [(c, g) for c in CS for g in GAMMAS]


def colour(v: float) -> str:
    pct = round(min(abs(v) / CAP, 1.0) * MAXMIX)
    base = "dgrnpos" if v >= 0 else "dredneg"
    return f"{base}!{pct}!white"


def load_dp() -> dict[tuple[float, float], float]:
    out = {}
    with DPCSV.open() as f:
        for r in csv.DictReader(f):
            out[(float(r["c"]), float(r["gamma"]))] = float(r["V_dp"])
    return out


def load_r9() -> dict[tuple[float, float], dict[str, float]]:
    out = {}
    with R9CSV.open() as f:
        for r in csv.DictReader(f):
            out[(float(r["c"]), float(r["gamma"]))] = {k: float(v) for k, v in r.items()
                                                       if k not in ("Configuration",)}
    return out


def pmci(mean: float, lo: float, hi: float) -> str:
    hw = (hi - lo) / 2.0
    return f"\\pmci{{{mean:.4f}}}{{{hw:.3f}}}"


def dpct(v: float, lo: float, hi: float) -> str:
    return f"\\dpct{{{colour(v)}}}{{{v:+.2f}}}{{{lo:+.2f}}}{{{hi:+.2f}}}"


def cfmt(c: float) -> str:
    return f"{c:.2f}"


def gfmt(g: float) -> str:
    return f"{g:.0f}" if g == int(g) else f"{g:g}"


def build_body() -> str:
    dp = load_dp()
    r9 = load_r9()
    lines: list[str] = []
    prev_c = None
    for c, g in ROWS:
        if prev_c is not None and c != prev_c:
            lines.append("            \\midrule")
        prev_c = c
        d = r9[(c, g)]
        vdp = dp[(c, g)]
        L_mean, L_lo, L_hi = d["LSM_mean"], d["LSM_lo"], d["LSM_hi"]
        # DP treated like a method vs LSM: price (no CI) + Delta% [CI inherited from LSM].
        dp_delta = 100.0 * (vdp / L_mean - 1.0)
        dp_lo = 100.0 * (vdp / L_hi - 1.0)   # monotone decreasing -> endpoints swap
        dp_hi = 100.0 * (vdp / L_lo - 1.0)
        cells = [
            cfmt(c), gfmt(g),
            pmci(L_mean, L_lo, L_hi),                                   # LSM-D benchmark (first)
            f"\\dpv{{{vdp:.4f}}}", dpct(dp_delta, dp_lo, dp_hi),        # DP: price, Delta%[CI]
            pmci(d["ACsample_mean"], d["ACsample_lo"], d["ACsample_hi"]),
            dpct(d["ACsample_delta_pct"], d["ACsample_delta_lo"], d["ACsample_delta_hi"]),
            pmci(d["ACkernel_mean"], d["ACkernel_lo"], d["ACkernel_hi"]),
            dpct(d["ACkernel_delta_pct"], d["ACkernel_delta_lo"], d["ACkernel_delta_hi"]),
        ]
        lines.append("            " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


HEADER = (
    "            \\toprule\n"
    "                 &            & LSM-D ($M\\!=\\!5$) & \\multicolumn{2}{c}{DP reference} "
    "& \\multicolumn{2}{c}{AC-sample (model-agnostic)} & \\multicolumn{2}{c}{AC-kernel (HHK target)} \\\\\n"
    "            \\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
    "            $c$  & $\\gamma_c$ & price\\,\\footnotesize$\\pm$CI "
    "& $V^{\\mathrm{DP}}$ & $\\Delta\\%$ \\footnotesize[95\\% CI] "
    "& price\\,\\footnotesize$\\pm$CI & $\\Delta\\%$ \\footnotesize[95\\% CI] "
    "& price\\,\\footnotesize$\\pm$CI & $\\Delta\\%$ \\footnotesize[95\\% CI] \\\\\n"
    "            \\midrule"
)
COLSPEC = "cc r r l r l r l"


def main() -> None:
    text = TEX.read_text()
    m = re.search(r"(\\label\{tab:results\}.*?\\begin\{tabular\}\{)([^}]*)(\}\s*\n)"
                  r"(.*?)(\n\s*\\bottomrule)", text, re.S)
    if not m:
        raise SystemExit("tab:results tabular not found")
    new = (text[:m.start()]
           + m.group(1) + COLSPEC + m.group(3)
           + HEADER + "\n" + build_body()
           + m.group(5) + text[m.end():])
    TEX.write_text(new)
    print(f"rebuilt tab:results body ({len(ROWS)} rows), colspec -> {COLSPEC!r}")


if __name__ == "__main__":
    main()
