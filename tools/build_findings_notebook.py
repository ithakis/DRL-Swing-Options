"""Generate the Phase-1 findings Jupyter notebook from the sweep CSVs.

Writes 'Jupyter Notebooks/7: Phase 1 Findings - Semi-Analytical Kernel.ipynb'
as a self-contained tour for someone new to the project.

Approach: build the ipynb JSON directly, with embedded result data so the
notebook can be re-executed standalone.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs" / "_sweep_h1"
NOTEBOOK_PATH = ROOT / "Jupyter Notebooks" / "7: Phase 1 Findings - Semi-Analytical Kernel.ipynb"

# ---------------------------------------------------------------------------
# Load + aggregate data
# ---------------------------------------------------------------------------

SOURCES = [
    "sweep_results_n4096.csv",
    "sweep_results_n3072_wide2.csv",
    "sweep_h1_phase2.csv",
    "sweep_h1_phase3_n8192.csv",
    "sweep_h4_n4096.csv",
    "sweep_h4_v2_n4096.csv",
    "sweep_h6_n2048.csv",
    "sweep_h789_n4096.csv",
    "sweep_h789_resume_n4096.csv",
    "sweep_h8strat_n4096.csv",
    "sweep_param_study.csv",
    "sweep_param_study_resume.csv",
]


def load_all():
    rows = []
    for s in SOURCES:
        path = LOG_DIR / s
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                r["_source"] = s
                rows.append(r)
    return rows


def per_seed(rows):
    return [
        (r["label"], int(r.get("seed", 0) or 0), int(r.get("n_paths", 0) or 0),
         r.get("contract", "focal") or "focal",
         (float(r["eval_price"]) / float(r["lsm_price"]) - 1.0) * 100.0,
         float(r["eval_price"]), float(r["lsm_price"]),
         float(r.get("wall_seconds", 0) or 0))
        for r in rows
        if r.get("eval_price") and r.get("lsm_price")
        and float(r.get("lsm_price", 0) or 0) > 0
    ]


def group_stats(rows, filt=lambda r: True):
    by_key: Dict[Tuple, List] = {}
    for r in rows:
        if not filt(r):
            continue
        key = (r[0], r[2], r[3])  # (label, n_paths, contract)
        by_key.setdefault(key, []).append(r)
    out = []
    for (label, n_p, contract), grp in by_key.items():
        deltas = [g[4] for g in grp]
        walls = [g[7] for g in grp if g[7] > 0]
        n = len(deltas)
        m = sum(deltas) / n
        s = statistics.stdev(deltas) if n > 1 else 0.0
        se = s / math.sqrt(n) if n > 1 else 0.0
        out.append({
            "label": label, "n_paths": n_p, "contract": contract,
            "n_seeds": n, "mean": m, "std": s, "se": se,
            "ci_low": m - 1.96 * se, "ci_high": m + 1.96 * se,
            "conservative": m - 1.96 * se,
            "wall_mean": (sum(walls) / len(walls)) if walls else float("nan"),
        })
    return sorted(out, key=lambda g: -g["mean"])


def main():
    rows = per_seed(load_all())
    print(f"Loaded {len(rows)} ok rows")
    groups = group_stats(rows, filt=lambda r: r[3] == "focal" and r[2] == 4096)
    nocost = group_stats(rows, filt=lambda r: r[3] == "nocost")
    paths_groups = group_stats(rows, filt=lambda r: r[3] == "focal" and r[0] in ("H1_only", "K36_no_warmup", "B0_baseline", "CTRL"))

    # Kernel-size data: M6, M21, M36 (=H1_only), M78. Use only 4096-ep groups.
    M_map = {"K_M6": 6, "K_M21": 21, "H1_only": 36, "K_M78": 78}
    kernel_pts = []
    for g in groups:
        if g["label"] in M_map and g["n_paths"] == 4096:
            kernel_pts.append({
                "M": M_map[g["label"]],
                "label": g["label"],
                "mean": g["mean"], "std": g["std"], "se": g["se"],
                "n_seeds": g["n_seeds"], "wall_mean": g["wall_mean"],
            })
    kernel_pts.sort(key=lambda d: d["M"])

    # Embed data into notebook code cells
    payload = {
        "groups_focal_4096": groups,
        "groups_nocost": nocost,
        "kernel_points": kernel_pts,
        "groups_focal_paths": paths_groups,
    }

    # Pretty-print embedded JSON for the notebook
    payload_json = json.dumps(payload, indent=2)

    cells = []

    def md(*lines: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [ln + "\n" if not ln.endswith("\n") else ln for ln in lines],
        })

    def code(*lines: str, exec_count=None):
        cells.append({
            "cell_type": "code",
            "execution_count": exec_count,
            "metadata": {},
            "outputs": [],
            "source": [ln + "\n" if not ln.endswith("\n") else ln for ln in lines],
        })

    # ------------------------- title + setup ----------------------------
    md(
        "# Phase 1 Findings: Semi-Analytical Kernel Bootstrap for D4PG Swing Options",
        "",
        "**Branch:** `feat/semi-analytical-bootstrap`  ·  **Contract:** focal $c=0.04, \\gamma=2$  ·  **Horizon:** 4096 ep unless noted",
        "",
        "This notebook is a self-contained tour for someone new to the project. It assumes you know:",
        "- The basics of swing option pricing (multi-exercise American-style)",
        "- The Hambly-Howison-Kluge (HHK) two-factor OU + jump-diffusion spot model",
        "- Stochastic optimal control / reinforcement learning at a conceptual level",
        "",
        "By the end you will know:",
        "1. **What the kernel idea is** and why it's the right level of analytical leverage.",
        "2. **What works** (H1) and **what doesn't** (H4–H9), with proper statistics.",
        "3. **How to choose the kernel size $M$** based on accuracy vs wall-clock.",
        "4. **What to port to C++** for the next phase.",
    )

    code(
        "import json, math, numpy as np, pandas as pd",
        "import matplotlib.pyplot as plt",
        "from scipy import stats as sps",
        "",
        "# Embedded result data so this notebook is self-contained",
        f"DATA = json.loads(r'''{payload_json}''')",
        "",
        "print('Groups loaded for focal/4096:', len(DATA['groups_focal_4096']))",
        "print('Kernel-size points:', len(DATA['kernel_points']))",
    )

    # ------------------------- §1 the problem ---------------------------
    md(
        "## 1. The Problem",
        "",
        "A **swing option** on electricity with strike $K$ and maturity $T$ gives the holder the right to exercise a continuous quantity $q_i \\in [q_{\\min}, q_{\\max}]$ at each of $n$ discrete decision dates $t_i$, subject to a global cap $\\sum_i q_i \\le Q_{\\max}$.  The per-step payoff is",
        "",
        "$$\\pi_i \\;=\\; q_i\\,(S_{t_i} - K)^{+} - c\\,q_i^{\\gamma}$$",
        "",
        "where the convex cost $c q^{\\gamma}$ (with $\\gamma=2$ here) penalises large lifts.  The fair value is",
        "",
        "$$V_0 \\;=\\; \\sup_{(q_i) \\in \\mathcal{A}} \\;\\mathbb{E}\\!\\left[ \\sum_{i=1}^{n} e^{-r t_i}\\,\\pi_i \\right]$$",
        "",
        "under the HHK spot dynamics",
        "$S_t = \\exp(f(t) + X_t + Y_t)$ where $X_t$ is a mean-reverting OU diffusion and $Y_t$ is a mean-reverting compound-Poisson jump process.",
        "",
        "The classical benchmark is **Least-Squares Monte Carlo (LSM)**.  We use D4PG (a deterministic-policy actor-critic) to learn a continuous-action exercise policy and beat LSM in the convex-cost regime.",
    )

    # ------------------------- §2 hypothesis -----------------------------
    md(
        "## 2. The Hypothesis: Kernel-Expected TD Target (H1)",
        "",
        "Standard TD learning bootstraps the critic with **one** sampled next-state per transition:",
        "",
        "$$Q_{\\text{target}}(s_t, a_t) \\;=\\; r_t + \\gamma \\, Q_\\theta^-\\bigl(s_{t+1}, \\pi_\\phi^-(s_{t+1})\\bigr)$$",
        "",
        "The variance of this estimator comes entirely from the random draw $s_{t+1} \\sim p(\\cdot \\mid s_t, a_t)$.  For HHK with rare-but-heavy jumps, that variance is the dominant noise source at low training-data regimes.",
        "",
        "**H1 idea:** the HHK transition is *analytically tractable*.",
        "- $X_{t+1} | X_t \\sim \\mathcal{N}(e^{-\\alpha\\Delta t} X_t, \\sigma_X^2)$ — exact OU.",
        "- $Y_{t+1} | Y_t$ is compound-Poisson with closed-form characteristic function.",
        "",
        "So we can replace the single-sample bootstrap with the **analytical expectation**:",
        "",
        "$$Q_{\\text{target}}(s_t, a_t) \\;=\\; r_t + \\gamma \\, \\mathbb{E}_{s_{t+1} \\mid s_t, a_t}\\!\\bigl[ Q_\\theta^-(s_{t+1}, \\pi_\\phi^-(s_{t+1})) \\bigr]$$",
        "",
        "computed via tensor-product Gauss-Hermite × Poisson-truncated quadrature with $M$ nodes:",
        "",
        "$$\\hat Q_{\\text{target}}(s_t, a_t) \\;=\\; r_t + \\gamma \\sum_{m=1}^{M} w_m \\, Q_\\theta^-(s^{(m)}_{t+1}, \\pi_\\phi^-(s^{(m)}_{t+1}))$$",
        "",
        "where $(s^{(m)}_{t+1}, w_m)$ are deterministic quadrature points/weights chosen so that $\\sum_m w_m\\,g(s^{(m)}) \\approx \\int g(s)\\,p(s|s_t,a_t)\\,ds$ for smooth $g$.",
        "",
        "**Critically**, the kernel acts only on the *target* for transitions actually visited by the agent.  It does *not* introduce off-policy training signals (this turns out to matter — see §5).",
    )

    # ------------------------- §3 implementation/validation --------------
    md(
        "## 3. Implementation & Kernel Accuracy",
        "",
        "Implementation in `src/transition_kernel.py`.  The quadrature kernel uses:",
        "- **$X$:** $M_x$ standardised Gauss-Hermite nodes (probabilist's), shifted+scaled by the OU conditional mean and std.",
        "- **$Y$:** truncated-Poisson on the number of jumps $N \\le N_{\\max}$ × QMC quadrature on the jump-amount/time joint.  With $\\beta = 150$ and $\\Delta t \\approx 1/22$, $\\lambda\\Delta t \\approx 0.024$ so $P(N \\ge 3) \\le 10^{-5}$.",
        "",
        "Total grid size $M = M_x \\cdot (1 + N_{\\max} \\cdot M_{\\text{per-k}})$.  Validation (in `tools/validate_transition_kernel.py`, 5M-path MC ground truth + analytical MGF for $\\mathbb{E}[S]$):",
        "",
        "| Grid | $M$ | Max relative error on smooth $\\sigma(S-K)$ |",
        "|---|---:|---:|",
        "| $M_x=2, M_{pk}=2, N_{\\max}=1$ | 6   | $\\sim 10^{-3}$ |",
        "| $M_x=4, M_{pk}=4, N_{\\max}=2$ | 36  | $\\sim 5\\times 10^{-4}$ |",
        "| $M_x=6, M_{pk}=8, N_{\\max}=3$ | 150 | $\\sim 3\\times 10^{-4}$ |",
        "",
        "Per-call kernel grid build: ~4 µs single-state, ~150 µs batched at $B=128$ (all `@numba.njit`, plain `float64`, ready to port to C++).",
    )

    # ------------------------- §4 main results --------------------------
    md(
        "## 4. Main Results: H1 Wins",
        "",
        "All numbers below are per-seed $\\Delta\\% = (\\text{eval\\_price} / \\text{LSM\\_price} - 1) \\times 100$ on out-of-sample evaluation paths, focal regime, 4096 ep.",
        "",
        "Reference comparison: kernel-on (`H1_only`) vs baseline (`CTRL`).",
    )

    code(
        "groups = pd.DataFrame(DATA['groups_focal_4096'])",
        "ref = groups[groups['label'].isin(['H1_only', 'K36_no_warmup', 'CTRL', 'B0_baseline'])].copy()",
        "ref = ref.sort_values('mean', ascending=False).reset_index(drop=True)",
        "ref[['label','n_seeds','mean','std','se','ci_low','ci_high','wall_mean']]",
    )

    code(
        "# Welch's t-test: H1_only vs B0_baseline at 4096 ep",
        "h1 = groups[groups['label'] == 'H1_only'].iloc[0]",
        "b0 = groups[groups['label'] == 'B0_baseline'].iloc[0]",
        "diff = h1['mean'] - b0['mean']",
        "se_diff = math.sqrt(h1['se']**2 + b0['se']**2)",
        "t = diff / se_diff",
        "df = (h1['se']**2 + b0['se']**2)**2 / (h1['se']**4/(h1['n_seeds']-1) + b0['se']**4/(b0['n_seeds']-1))",
        "p = 2 * (1 - sps.t.cdf(abs(t), df))",
        "print(f'H1 mean = {h1[\"mean\"]:+.3f} pp  ({h1[\"n_seeds\"]} seeds, std {h1[\"std\"]:.3f})')",
        "print(f'B0 mean = {b0[\"mean\"]:+.3f} pp  ({b0[\"n_seeds\"]} seeds, std {b0[\"std\"]:.3f})')",
        "print(f'Gap = {diff:+.3f} pp, SE = {se_diff:.3f}, t = {t:+.2f}, df = {df:.1f}, p = {p:.2e}')",
    )

    md(
        "### Δ% by configuration (4096 ep, 95% CI bars)",
        "",
        "Bar plot of mean Δ% with 95% CI bars for every group with $n \\ge 2$ seeds at the focal regime, 4096 ep.",
    )

    code(
        "g = pd.DataFrame(DATA['groups_focal_4096'])",
        "g = g[(g['n_seeds'] >= 2)].sort_values('mean', ascending=True).reset_index(drop=True)",
        "fig, ax = plt.subplots(figsize=(11, max(4, 0.35*len(g))))",
        "yerr = (g['mean'] - g['ci_low'], g['ci_high'] - g['mean'])",
        "colors = ['#d9534f' if m < 0 else ('#5cb85c' if m > 0 else '#777') for m in g['mean']]",
        "ax.barh(g['label'], g['mean'], xerr=yerr, color=colors, alpha=0.75, error_kw={'capsize':3})",
        "ax.axvline(0, color='k', lw=0.5)",
        "ax.set_xlabel(r'$\\Delta\\%$  (eval / LSM - 1) $\\times$ 100,  mean $\\pm$ 1.96 SE')",
        "ax.set_title('Per-config Δ% with 95% CI (focal regime, 4096 ep)')",
        "ax.grid(True, axis='x', alpha=0.3)",
        "fig.tight_layout()",
        "plt.show()",
    )

    md(
        "**Read of this plot:** the only configurations on the *positive* side are kernel-on variants.  Every kernel-off configuration (`CTRL`, `B0_baseline`, `B1_no_target_noise`) sits at $\\sim -2$% Δ.  Within the kernel-on cluster, all variants have overlapping CIs at $n=3$ seeds — at this sample size we **cannot** distinguish them.",
    )

    # ------------------------- §5 negatives ----------------------------
    md(
        "## 5. The Negative Results (H4–H9)",
        "",
        "We tested 9 hypotheses total.  Only H1 wins.  The remaining 8 either don't help or actively hurt.  The pattern is clear:",
        "",
        "> **The kernel adds value when it operates on real transitions; it hurts when it operates on synthetic/off-policy state distributions.**",
        "",
        "| Hypothesis | Mechanism | Verdict | Why it failed / didn't help |",
        "|---|---|---|---|",
        "| **H1** kernel-expected TD target | Replace 1-sample bootstrap with weighted-sum over kernel quadrature | ✅ **WIN** (+0.47% vs −2.05%, z = 10.5) | — |",
        "| H4 backward-induction critic warm-start | Build $V(X,Y,Q,t)$ on a grid, supervise $Q^*$ before D4PG | ❌ null (refined) / catastrophic (naive) | Supervised step on uniform synthetic states biases critic |",
        "| H5 Dyna synthetic experience | Augment replay with kernel-sampled $(s,a,r,s')$ | ❌ catastrophic across $\\lambda\\in[10^{-3},1]$ | Same: uniform-state Q* dominates the real TD signal |",
        "| H6 analytical IQN quantile target | Kernel-weighted average of per-quantile predictions | ❌ catastrophic, one seed −17% | IQN + kernel together create a wrong attractor |",
        "| H7 twin critics (TD3) | Two critics, min of kernel-expected targets | ⚪ no different (z = −0.55) | Overestimation not the bottleneck here |",
        "| H8 antithetic-pair averaging | Average kernel-target on antithetic partner paths | ⚪ no different (no-strat: lower std but lower mean; strat-preserved: same as H1) | Kernel *already* integrates the Y distribution analytically |",
        "| H9 jump-event importance weighting | Upweight transitions where a jump fired | ⚪ no different (z = −0.24) | PER already implicitly prioritises high-TD-error jumps |",
        "| H8+H9 stacked | Both | ⚪ slightly worse | Combined drag from H8's mean loss |",
        "",
        "The H4/H5/H6 failures share a common mechanism: they all introduce *kernel-derived training signals on synthetic/uniform state distributions*.  The 2×64 critic MLP cannot reconcile a Q-function calibrated to that distribution with the one needed for realistic rollouts, and falls into a wrong attractor (eval price typically collapses to $\\sim 1.63$ across seeds).",
    )

    # ------------------------- §6 statistics -----------------------------
    md(
        "## 6. Proper Statistical Comparison",
        "",
        "**Welch's two-sample t-test** for mean differences (unequal-variance) and **Levene's test** for variance equality, applied pairwise against `H1_only` as reference.",
        "",
        "With $n=3$ seeds per config the SE is large; almost no kernel-on variants are statistically distinguishable from each other.  This is by itself a meaningful finding: **at the data we have, H1 is the unique decisive winner over baseline, and the choice between kernel-on variants is empirically a coin flip.**",
    )

    code(
        "g = pd.DataFrame(DATA['groups_focal_4096'])",
        "h1 = g[g['label'] == 'H1_only'].iloc[0]",
        "rows = []",
        "for _, gi in g.iterrows():",
        "    if gi['label'] == 'H1_only' or gi['n_seeds'] < 2: continue",
        "    diff = gi['mean'] - h1['mean']",
        "    se = math.sqrt(gi['se']**2 + h1['se']**2)",
        "    if se == 0: continue",
        "    t = diff/se",
        "    df_w = (gi['se']**2 + h1['se']**2)**2 / (gi['se']**4/max(gi['n_seeds']-1,1) + h1['se']**4/max(h1['n_seeds']-1,1) + 1e-30)",
        "    p = 2*(1 - sps.t.cdf(abs(t), df_w))",
        "    rows.append({'config': gi['label'], 'n': gi['n_seeds'], 'mean': gi['mean'],",
        "                  'gap_vs_H1': diff, 't': t, 'p_two_sided': p,",
        "                  'verdict': ('DECISIVELY better' if t > 1.96 else ('weakly better' if t > 1.0",
        "                              else ('no different' if t > -1.0 else ('weakly worse' if t > -1.96 else 'DECISIVELY worse'))))})",
        "pd.DataFrame(rows).sort_values('gap_vs_H1', ascending=False).reset_index(drop=True)",
    )

    # ------------------------- §7 kernel size --------------------------
    md(
        "## 7. Choosing the Kernel Size $M$ (Accuracy vs Wall-Clock)",
        "",
        "The kernel grid size $M$ controls how accurately we approximate the expectation $\\int g(s)\\,p(s|s_t,a_t)\\,ds$.  Larger $M$ → tighter quadrature accuracy at the cost of more critic forward passes per update.",
        "",
        "We swept $M \\in \\{6, 21, 36, 78\\}$ at the focal regime, 4096 ep, 3 seeds each.  Below: mean Δ% with 1 SE bars (left axis) and wall-clock per run (right axis) vs $M$.",
    )

    code(
        "kp = pd.DataFrame(DATA['kernel_points']).sort_values('M').reset_index(drop=True)",
        "fig, ax1 = plt.subplots(figsize=(8, 5))",
        "ax1.errorbar(kp['M'], kp['mean'], yerr=kp['se'], fmt='o-', color='#1f77b4', label='mean Δ% ± SE', capsize=5, lw=2, ms=8)",
        "ax1.axhline(0, color='k', lw=0.5, ls='--')",
        "ax1.set_xlabel('Kernel size $M$ (quadrature nodes)')",
        "ax1.set_ylabel('Mean Δ% over LSM', color='#1f77b4')",
        "ax1.set_xscale('log')",
        "ax1.set_xticks(kp['M']); ax1.set_xticklabels([str(int(x)) for x in kp['M']])",
        "ax1.grid(True, alpha=0.3)",
        "ax1.tick_params(axis='y', labelcolor='#1f77b4')",
        "",
        "ax2 = ax1.twinx()",
        "ax2.plot(kp['M'], kp['wall_mean'], 's--', color='#d62728', label='wall-clock per run (s)', lw=2, ms=8)",
        "ax2.set_ylabel('Wall-clock per run (s)', color='#d62728')",
        "ax2.tick_params(axis='y', labelcolor='#d62728')",
        "",
        "ax1.set_title('Kernel-size accuracy/speed trade-off (focal, 4096 ep, 3 seeds each)')",
        "fig.tight_layout()",
        "plt.show()",
        "",
        "kp[['M', 'label', 'n_seeds', 'mean', 'std', 'se', 'wall_mean']]",
    )

    md(
        "**Reading this curve:**",
        "- $M=6$ (the bare minimum): still beats baseline by $\\sim 2.3$ pp.  Mean Δ% slightly lower ($+0.30$ vs $+0.47$) but the gap is well within the 3-seed SE.",
        "- $M=21$ gives essentially the same Δ% as $M=36$ at **half the wall-clock**.",
        "- $M=78$ is **slower without measurable benefit** at $n=3$ seeds.",
        "",
        "### Recommended kernel sizing",
        "",
        "| Use case | $M_x$ | $M_{pk}$ | $N_{\\max}$ | $M$ | Wall (s/run @ 4096 ep) | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
        "| **M1-friendly default** | 3 | 3 | 2 | **21** | ~285 | **Best trade-off.**  Empirically equivalent to $M=36$. |",
        "| Conservative high-accuracy | 4 | 4 | 2 | 36 | ~555 | Original H1 default.  Use if precision matters more than wall-clock. |",
        "| Cheap exploration | 2 | 2 | 1 | 6 | ~170 | Still solidly beats baseline; useful for early hyperparameter scans. |",
        "| Diminishing returns | 6 | 4–8 | 3 | 78–150 | ~660–1000 | No measurable gain over $M=36$ at $n=3$ seeds. |",
        "",
        "**Decision rule for new HHK regimes:** start with $M=21$ ($M_x=3, M_{pk}=3, N_{\\max}=2$).  If $\\lambda \\Delta t$ is large (more frequent jumps), bump $N_{\\max}$ to 3.  If volatility $\\sigma_X$ is high, bump $M_x$ to 4.",
    )

    # ------------------------- §8 antithetic head-to-head ----------------
    md(
        "## 8. Antithetic Head-to-Head (H1 vs H8 vs H8-strat)",
        "",
        "H8 was the most plausible competitor to H1.  At $n=3$ seeds:",
        "",
        "- **H1_only**: mean $+0.474\\%$ ± 0.204",
        "- **H8 (no-strat)**: mean $+0.305\\%$ ± **0.130** (lower std but lower mean — lost stratification's benefit)",
        "- **H8_strat**: mean $+0.439\\%$ ± 0.246  (recovered mean, lost the variance edge)",
        "",
        "Welch's t-test shows neither H8 variant differs significantly from H1 (all $|t| < 1.5$, $p > 0.20$).  The 'low std' of H8 no-strat is most likely a *compensating* effect (the antithetic averaging halves the variance increase caused by losing stratification); when both pieces of variance reduction are kept, the benefits don't compose.",
        "",
        "**Recommendation:** the antithetic mechanism is theoretically motivated but empirically *redundant* with the kernel's analytical integration of the same $Y$ distribution.  **Don't ship it.**",
    )

    # ------------------------- §9 C++ porting ----------------------------
    md(
        "## 9. Recommendations and C++ Porting",
        "",
        "### Final recommended training configuration",
        "",
        "```bash",
        "python run.py \\",
        "  --use_expected_target=1 \\",
        "  --kernel_M_x=3 --kernel_M_per_k=3 --kernel_N_max=2 \\",
        "  --critic_warmup_episodes=0 \\",
        "  # ... all other flags as v61 focal default",
        "```",
        "",
        "Bit-identical to v61 if `--use_expected_target=0`.",
        "",
        "### What to port to C++ for the deployment phase",
        "",
        "1. **`src/transition_kernel.py`** (~300 LOC) is the highest-value, lowest-effort port.  It's pure `numpy` + `@numba.njit`, no Python objects in the hot path.  Translation to C++ is mechanical: arrays-of-doubles, three nested loops in `_build_sxy_grid_batched`, and the bilinear / quadrature helpers.",
        "2. **The 2×64 critic MLP** is small enough to hand-roll in C++ if you don't want LibTorch.  $9 \\to 64 \\to 64 \\to 1$ with SiLU + LayerNorm.  ~20k parameters total.",
        "3. **The HHK simulator** (`src/simulate_hhk_spot.py`) is also amenable: the diffusive OU step is exact Gaussian, the jump component is Poisson + exponential.  Drop the QMC if you want bit-portability and use a standard PRNG.",
        "",
        "### Simplifications that the data justify",
        "",
        "- **Drop H4/H5/H6 code** (`src/critic_warmstart.py`, `src/dyna_augment.py`, the IQN-target machinery).  None help.",
        "- **Drop H7/H8/H9 code paths** if you want a minimal port (none beat H1).  Keep just the H1 expected-target wiring.",
        "- The actor / critic networks could be **further simplified**: linear baseline + kernel-expected target might already capture most of the gain.  This is the next experimental step (\"ditch the NNs\" per the project plan).",
        "",
        "### What the data does NOT yet justify",
        "",
        "- A larger kernel ($M > 36$): no measurable benefit.",
        "- A distributional critic (IQN): hurts at this setup.",
        "- Synthetic-state training augmentation: catastrophically hurts.",
        "",
        "### Where to push next",
        "",
        "- **Higher seed count** ($n \\ge 8$ at 8k+ ep) — to detect a $0.1$ pp effect we'd need many more seeds.",
        "- **Linear value function** with kernel-expected target — could be much faster and still capture the H1 effect.",
        "- **Full 32k-ep validation** — establish whether the H1 advantage is convergence-speed only or also a higher plateau.",
    )

    # ------------------------- finale ----------------------------------
    md(
        "## Summary table",
        "",
        "| Item | Value | Notes |",
        "|---|---|---|",
        "| **Winner** | **H1: kernel-expected TD target** | Wins decisively over baseline (z = 10.5 at 4k ep, z = 16 at 8k ep) |",
        "| Best kernel size | $M = 21$ | Half the wall-clock of the default $M=36$, indistinguishable mean Δ% |",
        "| Wall-clock cost | ~2× baseline | Acceptable given the OOS gain |",
        "| No-cost regression | $+5.2$ pp better than baseline | No regression in the no-cost regime |",
        "| Seed std (8k ep) | **0.022** pp | Kernel-on converges to a remarkably tight asymptote |",
        "| Failed hypotheses | H4 (warm-start), H5 (Dyna), H6 (IQN) | All introduce off-policy training signals; critic can't reconcile |",
        "| Neutral hypotheses | H7 (twin), H8 (antithetic), H9 (jump-IW) | None statistically beat H1 alone |",
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Notebook written: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
