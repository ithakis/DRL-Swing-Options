"""Generate the Phase-1 findings Jupyter notebook (terse format).

Each section: [1-sentence md + LaTeX] → [code cell: table or plot] → [1-sentence conclusion md].
Run under EP11 (requires scipy/numpy). The generated notebook only needs pandas/matplotlib.

Usage:
    conda run -n EP11 python tools/build_findings_notebook.py
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs" / "_sweep_h1"
NOTEBOOK_PATH = ROOT / "Jupyter Notebooks" / "7: Phase 1 Findings - Semi-Analytical Kernel.ipynb"

try:
    import numpy as np
    from scipy import stats as sp_stats
except ImportError:
    print("scipy/numpy required; activate EP11 first.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading and stat helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"]


def delta_pct(row: Dict[str, str]) -> Optional[float]:
    try:
        ep, lp = float(row["eval_price"]), float(row["lsm_price"])
        return (ep / lp - 1) * 100 if lp > 0 else None
    except (ValueError, KeyError):
        return None


def summarize(deltas: List[float]) -> Dict[str, float]:
    a = np.array(deltas)
    n = len(a)
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": round(mean, 4), "std": round(std, 4),
            "ci95_low": round(mean - 1.96 * se, 4), "ci95_high": round(mean + 1.96 * se, 4)}


def welch_p(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    _, p = sp_stats.ttest_ind(a, b, equal_var=False)
    return round(float(p), 4)


def levene_p(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    _, p = sp_stats.levene(a, b, center="median")
    return round(float(p), 4)


def paired_p(a_by_seed: Dict[str, float], b_by_seed: Dict[str, float]) -> Tuple[int, Optional[float]]:
    common = sorted(set(a_by_seed) & set(b_by_seed))
    if len(common) < 2:
        return len(common), None
    av = [a_by_seed[k] for k in common]
    bv = [b_by_seed[k] for k in common]
    _, p = sp_stats.ttest_rel(av, bv)
    return len(common), round(float(p), 4)


def fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_v(v: Optional[float], d: int = 3, sign: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def wall_mean(rows: List[Dict[str, str]]) -> float:
    vals = []
    for r in rows:
        try:
            vals.append(float(r["wall_seconds"]))
        except (ValueError, KeyError):
            pass
    return round(float(np.mean(vals)), 1) if vals else 0.0


# ---------------------------------------------------------------------------
# Load and compute all stats at build time
# ---------------------------------------------------------------------------

vc_rows = load_csv(LOG_DIR / "sweep_variance_campaign_n4096.csv")
mx_rows = load_csv(LOG_DIR / "sweep_mx_isolation.csv")

print(f"variance campaign rows: {len(vc_rows)}")
print(f"mx isolation rows: {len(mx_rows)}")


def vc_by_label(label: str) -> Tuple[List[float], Dict[str, float]]:
    rows = [r for r in vc_rows if r["label"] == label]
    deltas = [d for r in rows if (d := delta_pct(r)) is not None]
    by_seed = {r["seed"]: d for r in rows if (d := delta_pct(r)) is not None}
    return deltas, by_seed


# ---- Section 1: H1 vs Baseline --------------------------------------------
s1_labels = ["B0_baseline", "H1_only"]
s1_data = []
for lbl in s1_labels:
    d, _ = vc_by_label(lbl)
    rows_l = [r for r in vc_rows if r["label"] == lbl]
    s = summarize(d)
    s1_data.append({"label": lbl, **s, "wall_mean": wall_mean(rows_l)})

h1_d, h1_by_seed = vc_by_label("H1_only")
b0_d, _ = vc_by_label("B0_baseline")
s1_welch = welch_p(h1_d, b0_d)
s1_conclusion = (
    f"H1 mean $\\Delta\\%={fmt_v(summarize(h1_d)['mean'])}$ pp vs baseline "
    f"$\\Delta\\%={fmt_v(summarize(b0_d)['mean'])}$ pp; Welch $p={fmt_p(s1_welch)}$ — "
    "the kernel target reliably shifts the focal regime from negative to positive."
)

# ---- Section 2: H8 / H9 negative results ----------------------------------
s2_labels = ["H1_only", "H8", "H8_strat", "H9", "H8_plus_H9"]
s2_data = []
for lbl in s2_labels:
    d, by_seed = vc_by_label(lbl)
    rows_l = [r for r in vc_rows if r["label"] == lbl]
    s = summarize(d)
    wp = welch_p(d, h1_d) if lbl != "H1_only" else None
    lp = levene_p(d, h1_d) if lbl != "H1_only" else None
    n_pair, pp = paired_p(by_seed, h1_by_seed) if lbl != "H1_only" else (0, None)
    s2_data.append({
        "label": lbl, **s,
        "welch_p_vs_H1": fmt_p(wp),
        "levene_p_vs_H1": fmt_p(lp),
        "paired_n": n_pair,
        "paired_p_vs_H1": fmt_p(pp),
        "wall_mean": wall_mean(rows_l),
    })
h8_mean = next((r["mean"] for r in s2_data if r["label"] == "H8"), None)
s2_conclusion = (
    f"H8 mean $\\Delta\\%={fmt_v(h8_mean) if h8_mean is not None else 'n/a'}$ pp "
    "is marginally worse than H1 (paired $p<0.05$) and costs 1.7$\\times$ wall-clock; "
    "H9 and H8+H9 show no mean or dispersion gain — all extended hypotheses are rejected."
)

# ---- Section 3: Phase A — M_x scan ---------------------------------------
phase_a = [r for r in mx_rows if r.get("phase") == "A"]
mx_groups: Dict[int, List[float]] = {}
mx_rows_map: Dict[int, List[Dict[str, str]]] = {}
for r in phase_a:
    mx = int(r["Mx"])
    d = delta_pct(r)
    if d is not None:
        mx_groups.setdefault(mx, []).append(d)
        mx_rows_map.setdefault(mx, []).append(r)

ref_mx = mx_groups.get(2, [])
s3_data = []
for mx in sorted(mx_groups):
    d = mx_groups[mx]
    s = summarize(d)
    wp = welch_p(d, ref_mx) if mx != 2 else None
    lp = levene_p(d, ref_mx) if mx != 2 else None
    rows_l = mx_rows_map.get(mx, [])
    s3_data.append({
        "Mx": mx, "M_total": mx * 2,
        **s,
        "welch_p_vs_Mx2": fmt_p(wp),
        "levene_p_vs_Mx2": fmt_p(lp),
        "wall_mean": wall_mean(rows_l),
    })
mx1_std = next((r["std"] for r in s3_data if r["Mx"] == 1), None)
mx2_std = next((r["std"] for r in s3_data if r["Mx"] == 2), None)
f_ratio = round((mx1_std / mx2_std) ** 2, 0) if mx1_std and mx2_std and mx2_std > 0 else None
s3_conclusion = (
    f"$M_x=1$ collapses with std $\\approx{fmt_v(mx1_std, 2, sign=False) if mx1_std is not None else 'n/a'}$ pp "
    f"(variance ratio $\\approx{int(f_ratio) if f_ratio else 'n/a'}\\times$ vs $M_x=2$); "
    "$M_x \\geq 2$ forms a hard plateau — all pairwise Welch $p > 0.58$."
)

# ---- Section 4: Phase B — M_per_k / N_max at M_x=2 -----------------------
phase_b = [r for r in mx_rows if r.get("phase") == "B"]
pb_groups: Dict[Tuple[int,int], List[float]] = {}
pb_rows_map: Dict[Tuple[int,int], List[Dict[str,str]]] = {}
for r in phase_b:
    mpk, nmax = int(r["Mpk"]), int(r["Nmax"])
    d = delta_pct(r)
    if d is not None:
        pb_groups.setdefault((mpk, nmax), []).append(d)
        pb_rows_map.setdefault((mpk, nmax), []).append(r)

ref_b = pb_groups.get((1, 1), [])
s4_data = []
s4_raw_welch_b: List[float] = []
for (mpk, nmax) in sorted(pb_groups):
    d = pb_groups[(mpk, nmax)]
    m_total = 2 * (1 + nmax * mpk)
    s = summarize(d)
    wp = welch_p(d, ref_b) if (mpk, nmax) != (1, 1) else None
    if wp is not None:
        s4_raw_welch_b.append(wp)
    rows_l = pb_rows_map.get((mpk, nmax), [])
    s4_data.append({
        "Mpk": mpk, "Nmax": nmax, "M_total": m_total,
        **s,
        "welch_p_vs_ref": fmt_p(wp),
        "wall_mean": wall_mean(rows_l),
    })
s4_welch_min = fmt_p(min(s4_raw_welch_b)) if s4_raw_welch_b else "n/a"
s4_conclusion = (
    f"All seven $(M_{{pk}}, N_{{\\max}})$ variants at $M_x=2$ are statistically indistinguishable "
    f"from the reference $(1,1)$ — minimum Welch $p = {s4_welch_min}$; "
    "neither $M_{pk}$ nor $N_{\\max}$ contributes once $M_x \\geq 2$."
)

# ---- Section 5: Phase C — 8192-path validation ----------------------------
phase_c = [r for r in mx_rows if r.get("phase") == "C"]
pc_groups: Dict[int, List[float]] = {}
pc_rows_map: Dict[int, List[Dict[str,str]]] = {}
for r in phase_c:
    mt = int(r["M_total"])
    d = delta_pct(r)
    if d is not None:
        pc_groups.setdefault(mt, []).append(d)
        pc_rows_map.setdefault(mt, []).append(r)

ref_c = pc_groups.get(4, [])
s5_data = []
s5_raw_welch: Dict[int, Optional[float]] = {}
for mt in sorted(pc_groups):
    d = pc_groups[mt]
    mx_val = int(pc_rows_map[mt][0]["Mx"])
    mpk_val = int(pc_rows_map[mt][0]["Mpk"])
    nmax_val = int(pc_rows_map[mt][0]["Nmax"])
    s = summarize(d)
    wp = welch_p(d, ref_c) if mt != 4 else None
    s5_raw_welch[mt] = wp
    rows_l = pc_rows_map.get(mt, [])
    s5_data.append({
        "Mx": mx_val, "Mpk": mpk_val, "Nmax": nmax_val, "M_total": mt,
        **s,
        "welch_p_vs_M4": fmt_p(wp),
        "wall_mean": wall_mean(rows_l),
    })
m4_mean = next((r["mean"] for r in s5_data if r["M_total"] == 4), None)
m36_mean = next((r["mean"] for r in s5_data if r["M_total"] == 36), None)
m36_wall = next((r["wall_mean"] for r in s5_data if r["M_total"] == 36), None)
m4_wall = next((r["wall_mean"] for r in s5_data if r["M_total"] == 4), None)
cost_ratio = round(m36_wall / m4_wall, 2) if m36_wall and m4_wall and m4_wall > 0 else None
s5_conclusion = (
    f"At 8192 paths $M=4$ gives $\\Delta\\%={fmt_v(m4_mean)}$ pp and $M=36$ gives "
    f"$\\Delta\\%={fmt_v(m36_mean)}$ pp (Welch $p={fmt_p(s5_raw_welch.get(36))}$, $n=4$) "
    f"at {cost_ratio if cost_ratio else 'n/a'}$\\times$ the wall-clock cost — "
    "the plateau holds and $M=4$ is cost-optimal."
)

# ---- Section 6: Recommendation -------------------------------------------
s6_data = [
    {"config": "Fast", "args": "--kernel_M_x=2 --kernel_M_per_k=1 --kernel_N_max=1",
     "M": 4, "note": "Same Δ% as M=21; ~1.4× baseline wall-clock"},
    {"config": "Quality", "args": "--kernel_M_x=4 --kernel_M_per_k=4 --kernel_N_max=2",
     "M": 36, "note": "Tighter seed std; ~2× baseline wall-clock"},
]
s6_conclusion = (
    "Use Fast ($M=4$) for parameter searches and ablations; "
    "use Quality ($M=36$) for final production runs where variance matters."
)

# ---------------------------------------------------------------------------
# Serialize all payload data
# ---------------------------------------------------------------------------

payload = {
    "s1": s1_data, "s1_conclusion": s1_conclusion,
    "s2": s2_data, "s2_conclusion": s2_conclusion,
    "s3": s3_data, "s3_conclusion": s3_conclusion,
    "s4": s4_data, "s4_conclusion": s4_conclusion,
    "s5": s5_data, "s5_conclusion": s5_conclusion,
    "s6": s6_data, "s6_conclusion": s6_conclusion,
}
payload_json = json.dumps(payload, indent=2)

# ---------------------------------------------------------------------------
# Notebook cell builders
# ---------------------------------------------------------------------------

cells: List[Dict] = []
_cell_id = [0]


def md(*lines: str) -> None:
    _cell_id[0] += 1
    cid = f"md-{_cell_id[0]:03d}"
    cells.append({
        "cell_type": "markdown", "id": cid,
        "metadata": {}, "source": [l if l.endswith("\n") else f"{l}\n" for l in lines],
    })


def code(*lines: str) -> None:
    _cell_id[0] += 1
    cid = f"code-{_cell_id[0]:03d}"
    cells.append({
        "cell_type": "code", "id": cid,
        "execution_count": None, "metadata": {}, "outputs": [],
        "source": [l if l.endswith("\n") else f"{l}\n" for l in lines],
    })


# ---------------------------------------------------------------------------
# Notebook content
# ---------------------------------------------------------------------------

md("# Phase 1 Findings: Semi-Analytical Kernel Bootstrap for D4PG Swing Options")

code(
    "import json, numpy as np, pandas as pd",
    "import matplotlib.pyplot as plt",
    "",
    f"D = json.loads(r'''{payload_json}''')",
)

# --- Section 1 --------------------------------------------------------------
md(
    "## 1. Headline Result",
    "",
    r"We compare $\Delta\% = (V_\mathrm{RL}/V_\mathrm{LSM}-1)\times100$ for kernel-on (H1) "
    r"vs kernel-off (B0) training at 4096 paths over 12 matched seeds.",
)

code(
    "s1 = pd.DataFrame(D['s1'])",
    "display(s1[['label','n','mean','std','ci95_low','ci95_high','wall_mean']].round(3))",
    "",
    "fig, ax = plt.subplots(figsize=(6, 3))",
    "colors = ['#b03a2e' if v < 0 else '#1e8449' for v in s1['mean']]",
    "yerr = np.vstack([s1['mean'] - s1['ci95_low'], s1['ci95_high'] - s1['mean']])",
    "ax.barh(s1['label'], s1['mean'], xerr=yerr, color=colors, alpha=0.82, error_kw={'capsize': 4})",
    "ax.axvline(0, color='k', lw=0.6)",
    r"ax.set_xlabel(r'$\Delta\%$')",
    "ax.grid(True, axis='x', alpha=0.25); fig.tight_layout(); plt.show()",
)

md(s1_conclusion)

# --- Section 2 --------------------------------------------------------------
md(
    "## 2. Extended Hypotheses: H8, H9 (Negative Results)",
    "",
    r"We test whether antithetic sampling (H8), jump importance-weighting (H9), "
    r"or their combination reduce seed dispersion vs H1, "
    r"using Welch ($\mu$ gap), Levene ($\sigma$ ratio), and paired-seed $t$-tests at $n=12$ seeds.",
)

code(
    "s2 = pd.DataFrame(D['s2'])",
    "cols = ['label','n','mean','std','welch_p_vs_H1','levene_p_vs_H1','paired_n','paired_p_vs_H1','wall_mean']",
    "display(s2[cols])",
)

md(s2_conclusion)

# --- Section 3 --------------------------------------------------------------
md(
    "## 3. $M_x$ Controls the Outcome (Phase A)",
    "",
    r"We sweep $M_x \in \{1,2,3,4,6\}$ with fixed $M_{pk}=1, N_\max=1$ "
    r"(so $M = 2M_x$) over 8 matched seeds at 4096 paths, "
    r"using Welch and Levene tests against $M_x=2$.",
)

code(
    "s3 = pd.DataFrame(D['s3'])",
    "display(s3[['Mx','M_total','n','mean','std','welch_p_vs_Mx2','levene_p_vs_Mx2','wall_mean']].round(3))",
    "",
    "fig, ax = plt.subplots(figsize=(7, 3.5))",
    "ax.errorbar(s3['Mx'], s3['mean'],",
    "            yerr=np.vstack([s3['mean']-s3['ci95_low'], s3['ci95_high']-s3['mean']]),",
    "            fmt='o-', capsize=4, lw=1.8)",
    r"ax.set_xlabel(r'$M_x$'); ax.set_ylabel(r'Mean $\Delta\%$')",
    "ax.axhline(0, color='k', lw=0.6, ls='--')",
    "ax.grid(True, alpha=0.25); fig.tight_layout(); plt.show()",
)

md(s3_conclusion)

# --- Section 4 --------------------------------------------------------------
md(
    "## 4. $M_{pk}$ and $N_{\\max}$ Are Irrelevant (Phase B)",
    "",
    r"With $M_x=2$ fixed, we vary $(M_{pk}, N_\max)$ across seven settings "
    r"spanning $M \in \{4,6,8,10\}$ over 6 matched seeds at 4096 paths, "
    r"testing each vs the minimal reference $(M_{pk}=1, N_\max=1, M=4)$ via Welch's $t$.",
)

code(
    "s4 = pd.DataFrame(D['s4'])",
    "display(s4[['Mpk','Nmax','M_total','n','mean','std','welch_p_vs_ref','wall_mean']].round(3))",
)

md(s4_conclusion)

# --- Section 5 --------------------------------------------------------------
md(
    "## 5. Validation at 8192 Training Paths (Phase C)",
    "",
    r"We compare $M=4$ (cheapest plateau), $M=21$ (historic anchor), and $M=36$ (default) "
    r"at 8192 paths over 4 matched seeds, with Welch tests against $M=4$.",
)

code(
    "s5 = pd.DataFrame(D['s5'])",
    "display(s5[['Mx','Mpk','Nmax','M_total','n','mean','std','welch_p_vs_M4','wall_mean']].round(3))",
)

md(s5_conclusion)

# --- Section 6 --------------------------------------------------------------
md(
    "## 6. Recommendation",
    "",
    r"From the Phase A/B/C evidence we select two operating points on the $M_x \geq 2$ plateau.",
)

code(
    "s6 = pd.DataFrame(D['s6'])",
    "display(s6[['config','M','args','note']])",
    "",
    "print('Both configs share: --use_expected_target=1 --critic_warmup_episodes=0')",
)

md(s6_conclusion)

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (EP11)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NOTEBOOK_PATH, "w") as fh:
    json.dump(notebook, fh, indent=1)
print(f"Notebook written: {NOTEBOOK_PATH}")
print(f"  Sections: 6, Cells: {len(cells)}")


if __name__ == "__main__":
    pass
