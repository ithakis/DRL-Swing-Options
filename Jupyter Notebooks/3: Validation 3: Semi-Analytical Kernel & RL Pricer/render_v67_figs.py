"""A3 render — regenerate Figs 2 & 3 (hist_exercise, spot_income_pv_hist) from the v67 parquets.

Reuses NB6's EXACT plotting code: pulls the source of cells [1, 3, 4, 5, 21, 24] live from the
notebook, swaps cell 2's RUN_SPECS for the v67 runs (written by gen_figures_v67.py), and execs the
concatenation. Skips the intermediate analysis cells (6-20, 22-29) -- only what Figs 2 & 3 need.
Also writes the 4 minimal runs/<name>.json configs that cell 4 loads.

Run from repo root:  python "Jupyter Notebooks/3: Validation 3 .../render_v67_figs.py"
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "Jupyter Notebooks" / "6: Convex costs 0.04 Analysis.ipynb"
RUNS = ROOT / "runs"
SEED = 11
CELLS = [(0.0, 1.0, "1", "nocost"), (0.04, 1.0, "1", "g1"),
         (0.04, 1.5, "1p5", "g15"), (0.04, 2.0, "2", "g2")]

# 1) minimal config JSONs (cell 3/4/24 read maturity, n_rights, risk_free_rate, Q_max, q_max)
for c, gamma, gl, _ in CELLS:
    name = f"SwingOption_20_c{c:.2f}_gamma{gl}_v67_{SEED}"
    cfg = dict(c_cost=c, gamma_cost=gamma, strike=1.0, maturity=0.0833, n_rights=22,
               risk_free_rate=0.05, q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, S0=1.0,
               alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3)
    (RUNS / f"{name}.json").write_text(json.dumps(cfg, indent=2))

# 2) patched cell 2 (v67 RUN_SPECS, same attach_paths logic)
specs = ",\n    ".join(
    f'{{"label": "{tag}_seed{SEED}", "c": {c}, "gamma": {gamma}, "seed": {SEED}, "step": 4096, '
    f'"run_name": "SwingOption_20_c{c:.2f}_gamma{gl}_v67_{SEED}"}}'
    for c, gamma, gl, tag in CELLS)
CELL2 = f'''RUN_SPECS = [
    {specs}
]
FULL_LSM_DIR = REPO_ROOT / "logs/lsm_full_state"
def attach_paths(spec):
    eval_dir = LOGS_DIR / spec["run_name"] / "evaluations"
    config_base = spec["run_name"].rsplit("_", 1)[0]
    return {{**spec,
        "rl_path": eval_dir / f"rl_episode_{{spec['step']}}.parquet",
        "lsm_path": FULL_LSM_DIR / f"{{config_base}}_lsm.parquet",
        "config_path": RUNS_DIR / f"{{spec['run_name']}}.json"}}
RUN_SPECS = [attach_paths(s) for s in RUN_SPECS]
'''

# 3) exec cells 1..24 in order (patched cell 2), per-cell try/except so an intermediate plotting
#    cell that chokes on the v67 data doesn't block the figure cells (21=Fig2, 24=Fig3).
nb = json.loads(NB.read_text())
cells = nb["cells"]
os.chdir(ROOT)
import matplotlib.pyplot as plt
g = {"__name__": "__nb__"}
for i in range(1, 25):
    c = cells[i]
    if c["cell_type"] != "code":
        continue
    code = CELL2 if i == 2 else "".join(c.get("source", []))
    try:
        exec(compile(code, f"<nb6 cell {i}>", "exec"), g)
    except Exception as e:
        print(f"   [cell {i}] skipped: {type(e).__name__}: {str(e)[:120]}")
    finally:
        plt.close("all")
print("\n[render] regenerated:")
for f in ("hist_exercise", "spot_income_pv_hist"):
    for ext in ("png", "pdf"):
        p = ROOT / "figs" / "convex_costs_0p04" / f"{f}.{ext}"
        print(f"   {'OK ' if p.exists() else 'MISS'} {p.relative_to(ROOT)}")
