#!/usr/bin/env python3
"""gen_table5_dp.py — inject the DP reference column into Table 5 (tab:results).

Reads grid_dp_pricer/results/dp_publication_sweep.csv and rewrites the tab:results body
in Paper/DRL_Swing_Options.tex, inserting a "DP" value column right after the (c, gamma)
key columns.  The existing LSM-D / AC-sample / AC-kernel entries (BCa numbers baked by
gen_results9_v67.py + colorize_results_table.py) are preserved verbatim.

Idempotent: an already-injected DP cell (marked \\dpv{...}) is replaced, not duplicated.
Run:  python tools/gen_table5_dp.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "Paper" / "DRL_Swing_Options.tex"
DPCSV = ROOT / "grid_dp_pricer" / "results" / "dp_publication_sweep.csv"


def load_dp() -> dict[tuple[float, float], float]:
    out = {}
    with DPCSV.open() as f:
        for r in csv.DictReader(f):
            out[(float(r["c"]), float(r["gamma"]))] = float(r["V_dp"])
    return out


def main() -> None:
    dp = load_dp()
    text = TEX.read_text()

    # Locate the tab:results tabular body.
    m = re.search(r"(\\label\{tab:results\}.*?\\begin\{tabular\}\{)([^}]*)(\}.*?)\\end\{tabular\}",
                  text, re.S)
    if not m:
        raise SystemExit("tab:results tabular not found")
    colspec, body = m.group(2), m.group(3)

    # Column spec: cc r r l r l  ->  cc r r r l r l (one extra r after the key pair).
    if colspec.strip() == "cc r r l r l":
        new_colspec = "cc r r r l r l"
    elif colspec.strip() == "cc r r r l r l":
        new_colspec = colspec  # already injected
    else:
        raise SystemExit(f"unexpected tab:results colspec: {colspec!r}")

    new_body = body
    # Header rows: extend the group header and the column-name row.
    new_body = new_body.replace(
        "&            & LSM-D ($M\\!=\\!5$) & \\multicolumn{2}{c}{AC-sample (model-agnostic)} & \\multicolumn{2}{c}{AC-kernel (HHK target)} \\\\",
        "&            & DP reference & LSM-D ($M\\!=\\!5$) & \\multicolumn{2}{c}{AC-sample (model-agnostic)} & \\multicolumn{2}{c}{AC-kernel (HHK target)} \\\\",
        1)
    new_body = new_body.replace(
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}", 1)
    new_body = new_body.replace(
        "$c$  & $\\gamma_c$ & price\\,\\footnotesize$\\pm$CI & price\\,\\footnotesize$\\pm$CI & $\\Delta\\%$ \\footnotesize[95\\% CI] & price\\,\\footnotesize$\\pm$CI & $\\Delta\\%$ \\footnotesize[95\\% CI] \\\\",
        "$c$  & $\\gamma_c$ & $V^{\\mathrm{DP}}$ & price\\,\\footnotesize$\\pm$CI & price\\,\\footnotesize$\\pm$CI & $\\Delta\\%$ \\footnotesize[95\\% CI] & price\\,\\footnotesize$\\pm$CI & $\\Delta\\%$ \\footnotesize[95\\% CI] \\\\",
        1)

    # Body rows: "  0.04 & 2 & \pmci{...}" (optionally with an existing \dpv cell to replace).
    row_pat = re.compile(
        r"^(\s*)([0-9.]+) & ([0-9.]+) & (?:\\dpv\{[^{}]*\} & )?(\\pmci)",
        re.M)
    n = 0

    def repl(mo: re.Match) -> str:
        nonlocal n
        c, g = float(mo.group(2)), float(mo.group(3))
        v = dp.get((c, g))
        if v is None:
            raise SystemExit(f"no DP value for cell ({c}, {g})")
        n += 1
        return f"{mo.group(1)}{mo.group(2)} & {mo.group(3)} & \\dpv{{{v:.4f}}} & {mo.group(4)}"

    new_body = row_pat.sub(repl, new_body)
    if n != 29:
        raise SystemExit(f"expected 29 rows, rewrote {n}")

    text = text[:m.start(2)] + new_colspec + new_body + "\\end{tabular}" + text[m.end():]

    # Ensure the \dpv macro exists (plain value; band documented in caption/subsection).
    if "\\newcommand{\\dpv}" not in text:
        text = text.replace("\\newcommand{\\pmci}[2]",
                            "\\newcommand{\\dpv}[1]{$#1$}\n\\newcommand{\\pmci}[2]", 1)

    TEX.write_text(text)
    print(f"injected DP column into {n} rows of tab:results")


if __name__ == "__main__":
    main()
