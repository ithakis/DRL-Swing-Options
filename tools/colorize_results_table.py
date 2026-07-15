#!/usr/bin/env python3
"""Bake diverging cell-background colours into the \\dpct cells of Table 5 (tab:results).

The two \\Delta\\% columns (AC-sample, AC-kernel) are shaded on a single shared linear scale:
white at 0, green for positive, red for negative, intensity = min(|Delta|/CAP, 1) * MAXMIX.
The colour is precomputed here and written as the first argument of \\dpct, so the LaTeX build
performs no runtime floating-point evaluation (which proved fragile across latexmk passes).

Idempotent: it normalises any existing 4-arg \\dpct{color}{v}{lo}{hi} back to 3 args first, then
re-colours. Run:  python tools/colorize_results_table.py
"""
import re
from pathlib import Path

TEX = Path(__file__).resolve().parents[1] / "Paper" / "DRL_Swing_Options.tex"
CAP = 5.2        # saturates at the most extreme cell (+5.14)
MAXMIX = 80      # cap the white-mix so saturated cells stay legible under black text

def colour(v: float) -> str:
    pct = round(min(abs(v) / CAP, 1.0) * MAXMIX)
    base = "dgrnpos" if v >= 0 else "dredneg"
    return f"{base}!{pct}!white"

def main() -> None:
    text = TEX.read_text()
    # Normalise any prior 4-arg form back to 3-arg so re-runs are stable.
    text = re.sub(r"\\dpct\{[^{}]*\}(\{[^{}]*\}\{[^{}]*\}\{[^{}]*\})", r"\\dpct\1", text)
    # 3-arg form: \dpct{value}{lo}{hi}  ->  \dpct{color}{value}{lo}{hi}
    pat = re.compile(r"\\dpct\{([+-]?[0-9.]+)\}(\{[^{}]*\}\{[^{}]*\})")
    n = 0
    def repl(m):
        nonlocal n
        n += 1
        v = float(m.group(1))
        return f"\\dpct{{{colour(v)}}}{{{m.group(1)}}}{m.group(2)}"
    text = pat.sub(repl, text)
    TEX.write_text(text)
    print(f"coloured {n} \\dpct cells (CAP={CAP}, MAXMIX={MAXMIX})")

if __name__ == "__main__":
    main()
