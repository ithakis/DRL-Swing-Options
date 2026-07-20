"""Regenerate the §6 hedging figures from the v67 hedge blobs (logs/hedging_v67/).

Re-execs Hedging.ipynb cells with cell 1 patched: HEDGE_DIR -> hedging_v67, SEED -> 788 (train 11 +
777, matching the new OOS eval split in hedge_swing). Reuses the EXACT figure code so there is zero
plotting divergence. Regenerates figs/hedging_greeks_grid_rl_lsm.png (cell 3) and
figs/hedging_pnl_var_es.png (cell 11), plus prints the σ-reduction / VaR / ES stats for the prose.

Run from repo root:  python "Jupyter Notebooks/3: Validation 3 .../render_hedging_v67.py"
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "Jupyter Notebooks" / "Hedging.ipynb"
NBDIR = ROOT / "Jupyter Notebooks"

nb = json.loads(NB.read_text())
cells = nb["cells"]

def patch_cell1(src):
    src = src.replace('"logs", "hedging_v65"', '"logs", "hedging_v67"')
    src = src.replace("SEED    = 999        # common CRN/test seed",
                      "SEED    = 788        # OOS eval seed (train 11 + 777), matches hedge_swing")
    src = src.replace('"missing hedge blobs — run: EP11python tools/gen_hedging_v65.py"',
                      '"missing hedge blobs — run: EP11python tools/gen_hedging_v67.py"')
    return src

os.chdir(NBDIR)   # cell 1 uses os.path.join("..", "logs", ...) and sys.path.insert(0, abspath(".."))
import matplotlib.pyplot as plt
g = {"__name__": "__nb__", "display": lambda *a, **k: None}   # display() shim outside Jupyter
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    code = patch_cell1("".join(c.get("source", []))) if i == 1 else "".join(c.get("source", []))
    try:
        exec(compile(code, f"<hedging cell {i}>", "exec"), g)
    except Exception as e:
        print(f"   [cell {i}] skipped: {type(e).__name__}: {str(e)[:160]}")
    finally:
        plt.close("all")

print("\n[render] hedging figures:")
for f in ("hedging_greeks_grid_rl_lsm", "hedging_pnl_var_es", "hedging_daily_mechanics"):
    p = ROOT / "figs" / f"{f}.png"
    print(f"   {'OK ' if p.exists() else 'MISS'} figs/{f}.png")
