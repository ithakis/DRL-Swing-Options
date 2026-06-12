"""Builds '3: Validation 3: Semi-Analytical Kernel & RL Pricer.ipynb' from the
companion CSVs. Re-run to regenerate the notebook skeleton; cells may then be
tuned in-place. Kept in the companion folder for reproducibility."""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "3: Validation 3: Semi-Analytical Kernel & RL Pricer.ipynb"

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ====================================================================== TITLE
md(r"""
# Validation 3 — Semi-Analytical Kernel & RL Pricer

**Validating the model-based expected-backup actor–critic for swing-option pricing under the Hambly–Howison–Kluge (HHK) spot model.**

The D4PG pricer replaces the single-sample TD bootstrap with an **analytical expectation over the HHK one-step transition kernel**,
$Q_{\text{target}}(s,a)=r(s,a)+\mathbb{E}\!\left[Q(s',\pi(s'))\mid s,a\right]$, where the expectation is evaluated by a deterministic quadrature mesh (Gauss–Hermite on the OU factor $\times$ a compound-Poisson jump mesh) rather than by Monte-Carlo rollout.

This notebook validates the method as a **layered cake — the mathematics first, then the reinforcement learning on top:**

> **Part I · Mathematics of the kernel.** Is the quadrature target *correct*? We check the OU/Gaussian leg against its closed-form moment-generating function, the convergence of the Gauss–Hermite and jump meshes, agreement with brute-force nested Monte-Carlo for fixed state–action pairs, and the cost/accuracy frontier.
>
> **Part II · The RL pricer.** Does the trained policy *price well*? Every policy is evaluated by **forward rollout on one common, fresh Monte-Carlo test set** (seed 999, 65 536 paths) and benchmarked against a **strengthened full-state LSM-D baseline at $M\in\{5,9,17\}$ exercise levels** — directly testing whether the RL edge is merely an artefact of a coarse five-action discretisation. We report **wall-budget in episodes**, seed robustness, economic correctness, and the end-to-end role of the quadrature resolution $M_x$.

*Scope (per project decision):* the kernel-on agents are evaluated on the four regimes with a full saved seed family — **no-cost** $(c=0,\gamma=1)$ and **$c=0.04$** at $\gamma\in\{1,1.5,2\}$ (focal $\gamma=2$). All data are produced by `gen_kernel_validation.py` / `gen_rl_validation.py` in the companion folder; this notebook only loads and narrates them.
""")

# ====================================================================== SETUP
code(r"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import seaborn as sns
warnings.filterwarnings("ignore")

# --- locate companion folder (CSVs) and repo (for stats_analysis) ---
NB_DIR = Path.cwd()
D = NB_DIR / "3: Validation 3: Semi-Analytical Kernel & RL Pricer"
if not (D / "kernel_moments.csv").exists():
    D = Path("3: Validation 3: Semi-Analytical Kernel & RL Pricer")
ROOT = (D.resolve().parents[1])
FIGS = D / "figs"; FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
import stats_analysis as sa   # Welch / Levene / F-ratio / MDE

# --- house style: clean, publication-grade ---
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 9.5, "axes.edgecolor": "#444",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
COL = {"kernel_on": "#0b7285", "kernel_off": "#d6336c",
       "lsm_M5": "#c0c6cc", "lsm_M9": "#7b848c", "lsm_M17": "#2b2f33",
       "accent": "#e8590c", "ok": "#2b8a3e"}
REG_LABEL = {"nocost": "no cost (gamma=1)", "g1": "c=0.04, gamma=1",
             "g15": "c=0.04, gamma=1.5", "g2": "c=0.04, gamma=2 (focal)"}
REG_ORDER = ["nocost", "g1", "g15", "g2"]

def save(fig, name):
    fig.savefig(FIGS / f"{name}.png"); fig.savefig(FIGS / f"{name}.pdf")

# --- load everything ---
KM  = pd.read_csv(D / "kernel_moments.csv")
KQ  = pd.read_csv(D / "kernel_quadrature_convergence.csv")
KJ  = pd.read_csv(D / "kernel_jump_mesh_convergence.csv")
KN  = pd.read_csv(D / "kernel_nested_mc_agreement.csv")
KC  = pd.read_csv(D / "kernel_cost_accuracy.csv")
RL  = pd.read_csv(D / "rl_lsm_pricing.csv")
EE  = pd.read_csv(D / "episode_efficiency.csv")
MX  = pd.read_csv(D / "mx_isolation.csv")
RL["method"] = RL["method"].replace("v61_paper", "kernel_off")
EE["method"] = EE["method"].replace("v61_paper", "kernel_off")
print("Loaded:", {k: v.shape for k, v in
      dict(moments=KM, quad=KQ, jump=KJ, nested=KN, cost=KC,
           pricing=RL, episodes=EE, mx=MX).items()})
""")

# ====================================================================== PART I
md(r"""
---
# Part I · Mathematics of the HHK transition kernel

Under the HHK model $S_t=\exp\!\big(f(t)+X_t+Y_t\big)$ with an Ornstein–Uhlenbeck diffusion
$dX=-\alpha X\,dt+\sigma\,dW$ and a mean-reverting compound-Poisson jump factor
$dY=-\beta Y\,dt+J\,dN$, the one-step transition factorises into an **exact Gaussian leg** (the OU
increment) and a **jump leg** (a decayed compound-Poisson sum). The kernel integrates a continuation
function against this transition using $M_x$ Gauss–Hermite nodes on $X'$ and a stratified-QMC mesh on
$Y'$ (parameters `N_max`, `M_per_k`); the total node count is $M=M_x\,(1+N_{\max} M_{\text{per }k})$.

Two reference settings recur below:
**FAST** $=(M_x{=}2, M_{\text{per }k}{=}1, N_{\max}{=}1)\Rightarrow M=4$ (the production kernel) and
**ACCURATE** $=(M_x{=}4, M_{\text{per }k}{=}4, N_{\max}{=}2)\Rightarrow M=36$.
""")

# ---- F1 -------------------------------------------------------------------
md(r"""
### F1 · The OU/Gaussian leg is *exact* (Gauss–Hermite spectral convergence)

With the jump intensity switched off ($\lambda=0$) the transition is purely Gaussian, so the conditional
moments $\mathbb{E}[S_{t+1}\mid X_t]$ and $\mathbb{E}[S_{t+1}^2\mid X_t]$ have a closed form (the OU
moment-generating function). Gauss–Hermite quadrature integrates these against the Gaussian density and
should converge **spectrally** in $M_x$.
""")
code(r"""
g = KM[KM.leg == "gaussian"].sort_values("M_x")
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax[0].semilogy(g.M_x, g.relerr_ES.clip(1e-16), "o-", color=COL["kernel_on"], lw=2, label=r"$E[S_{t+1}]$")
ax[0].semilogy(g.M_x, g.relerr_ES2.clip(1e-16), "s--", color=COL["accent"], lw=2, label=r"$E[S_{t+1}^2]$")
ax[0].axhline(2.2e-16, ls=":", color="grey"); ax[0].text(7.5, 4e-16, "machine $\\epsilon$", color="grey", fontsize=8)
ax[0].set(xlabel=r"Gauss–Hermite nodes $M_x$", ylabel="relative error vs closed-form MGF",
          title="(a) OU/Gaussian leg — spectral convergence ($\\lambda=0$)")
ax[0].legend(frameon=True)

f = KM[KM.leg.str.startswith("full")].groupby("leg")[["relerr_ES", "relerr_ES2"]].first()
f = f.reindex(["full_fast", "full_accurate"])
x = np.arange(2); w = 0.36
ax[1].bar(x - w/2, f.relerr_ES, w, color=COL["kernel_on"], label=r"$E[S_{t+1}]$")
ax[1].bar(x + w/2, f.relerr_ES2, w, color=COL["accent"], label=r"$E[S_{t+1}^2]$")
ax[1].set_yscale("log"); ax[1].set_xticks(x, ["FAST (M=4)", "ACCURATE (M=36)"])
ax[1].set(ylabel="relative error vs closed-form MGF",
          title="(b) Full kernel (with jumps) — moment error")
ax[1].legend(frameon=True)
fig.tight_layout(); save(fig, "F1_gaussian_leg"); plt.show()
print("Gaussian leg reaches machine precision by M_x=4; with jumps the error settles at the "
      "jump-mesh floor (~1e-3 on E[S]) — the subject of F2/F3.")
""")

# ---- F2 -------------------------------------------------------------------
md(r"""
### F2 · Convergence in $M_x$ vs brute-force nested Monte-Carlo

We now compare the **full** kernel expectation against a 2-million-path nested Monte-Carlo reference for
four integrands spanning the relevant function class — $E[S]$, $E[S^2]$, the option kink $E[(S-K)^+]$,
and a smooth payoff proxy $E[\,(S-K)\,\sigma(S-K)\,]$ — as $M_x$ increases (jump mesh held at ACCURATE).
The Gaussian leg is already converged by $M_x=2$, so the curve should **plateau at the jump-mesh floor**;
$M_x=1$ (a single node) is degenerate.
""")
code(r"""
qs = KQ[(KQ.X == 0.10) & (KQ.Y == 0.50)]
order = ["E[S]", "E[S^2]", "E[(S-K)+]", "E[smooth]"]
cmap = dict(zip(order, sns.color_palette("viridis", 4)))
fig, ax = plt.subplots(figsize=(7.4, 4.6))
for fn in order:
    d = qs[qs.func == fn].sort_values("M_x")
    ax.semilogy(d.M_x, d.relerr, "o-", lw=1.8, color=cmap[fn], label=fn)
ax.axvspan(0.5, 1.5, color="#ffe3e3", alpha=.7, zorder=0)
ax.text(1.0, ax.get_ylim()[1]*0.4, "degenerate\n$M_x=1$", ha="center", color=COL["kernel_off"], fontsize=8)
ax.axvline(2, ls="--", color=COL["kernel_on"]); ax.text(2.1, 2e-3, "FAST", color=COL["kernel_on"], fontsize=9)
ax.set(xlabel=r"Gauss–Hermite nodes $M_x$", ylabel="relative error vs 2M-path nested MC",
       title="F2 · $M_x$ convergence — hard plateau once $M_x\\geq2$  (state $X$=0.1, $Y$=0.5)")
ax.legend(frameon=True, ncol=2)
fig.tight_layout(); save(fig, "F2_Mx_convergence"); plt.show()
""")

# ---- F3 -------------------------------------------------------------------
md(r"""
### F3 · Convergence of the compound-Poisson jump mesh

The residual error in F2 is the **jump-mesh discretisation**, controlled by the Poisson truncation
`N_max` and the per-count QMC resolution `M_per_k`. Because $\lambda\,\Delta t\approx0.29$, almost all mass
sits on $0$–$2$ jumps; the truncation tail beyond `N_max` is shown for reference.
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
order = ["E[S]", "E[S^2]", "E[(S-K)+]", "E[smooth]"]
cmap = dict(zip(order, sns.color_palette("viridis", 4)))
for a, axis, xl in [(ax[0], "N_max", r"Poisson truncation $N_{\max}$"),
                    (ax[1], "M_per_k", r"QMC nodes per jump count $M_{\text{per }k}$")]:
    s = KJ[KJ.axis == axis]
    for fn in order:
        d = s[s.func == fn].sort_values("value")
        a.semilogy(d.value, d.relerr, "o-", lw=1.8, color=cmap[fn], label=fn)
    a.set(xlabel=xl, title=f"({'a' if axis=='N_max' else 'b'}) vs {xl.split('$')[0].strip()}")
ax[0].set_ylabel("relative error vs 2M-path nested MC")
tail = KJ[(KJ.axis == "N_max")].groupby("value").poisson_tail.first()
ax[0].text(0.55, 0.05, "\n".join(f"$N_{{\\max}}$={int(k)}: tail={v:.1e}" for k, v in tail.items()),
           transform=ax[0].transAxes, fontsize=7.5, va="bottom",
           bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
ax[1].legend(frameon=True, ncol=2)
fig.suptitle("F3 · Jump-mesh convergence", fontweight="bold", y=1.02)
fig.tight_layout(); save(fig, "F3_jump_mesh"); plt.show()
""")

# ---- F4 -------------------------------------------------------------------
md(r"""
### F4 · Agreement with nested Monte-Carlo for fixed state–action pairs

This is the reviewer's central check. For fixed states we compare the **deterministic** kernel target
against a *gold* 4-million-path nested-MC reference, for the payoff/value integrands and for the
**actual critic network** $Q(s',\pi(s'))$ (architecture-matched, frozen).

Two messages: **(a)** the kernel is a deterministic quadrature whose bias is **sub-percent at every mesh
setting** and is driven lowest ($\lesssim0.1\%$) by the RICH mesh — the mid "accurate" tier is not strictly
monotone, a known QMC-stratification artefact also visible in F3(b); and **(b)** that bias is
**1–2 orders of magnitude smaller** ($\approx$6–120×) than the standard deviation `boot_std` of the integrand — i.e. the noise that a
single-sample ($n{=}1$) TD bootstrap target injects on every update, and which the expected backup removes.
""")
code(r"""
# keep states whose option payoff is non-negligible (deep-OTM states have vanishing
# denominators that blow up relative error); applied at the STATE level so every
# integrand -- including the small-magnitude critic Q -- is retained.
pay = KN[KN.func == "E[(S-K)+]"].groupby(["X", "Y"]).mc_ref.first()
itm_xy = set(pay[pay > 0.05].index)
KN["_xy"] = list(zip(KN.X, KN.Y))
itm = KN[KN["_xy"].isin(itm_xy)].copy()
itm["abs_err"] = (itm.kernel - itm.mc_ref).abs()
tier_order = ["fast", "accurate", "rich"]

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) rel-error convergence across tiers (mean over ITM states), per function
funcs = ["E[(S-K)+]", "E[smooth]", "E[S]", "E[critic_Q]"]
cmap = dict(zip(funcs, sns.color_palette("rocket", 4)))
for fn in funcs:
    d = itm[itm.func == fn].groupby("kernel_tag").rel_err.mean().reindex(tier_order)
    ax[0].plot(tier_order, 100*d.values, "o-", lw=1.9, color=cmap[fn], label=fn)
ax[0].set(ylabel="mean relative error vs gold MC (%)",
          title="(a) kernel vs gold nested-MC — sub-percent at every setting")
ax[0].set_yscale("log"); ax[0].legend(frameon=True, fontsize=8)

# (b) variance reduction: deterministic kernel bias vs single-sample bootstrap noise
d = itm[itm.kernel_tag == "fast"].groupby("func").agg(abs_err=("abs_err", "mean"),
                                                      boot=("boot_std", "mean")).reindex(funcs)
x = np.arange(len(funcs)); w = 0.38
ax[1].bar(x - w/2, d.boot, w, color=COL["kernel_off"], label="single-sample bootstrap std (removed)")
ax[1].bar(x + w/2, d.abs_err, w, color=COL["kernel_on"], label="FAST kernel |bias| (incurred)")
ax[1].set_yscale("log"); ax[1].set_xticks(x, funcs, rotation=20, ha="right")
ax[1].set(ylabel="magnitude", title=r"(b) bias incurred $\ll$ variance removed")
ax[1].legend(frameon=True, fontsize=8)
fig.tight_layout(); save(fig, "F4_nested_mc"); plt.show()

ratio = (d.boot / d.abs_err).replace([np.inf], np.nan).dropna()
print(f"FAST kernel: single-sample noise / kernel bias ≈ {ratio.min():.0f}–{ratio.max():.0f}× "
      "across integrands (the variance-reduction factor of the expected backup).")
""")

# ---- F5 -------------------------------------------------------------------
md(r"""
### F5 · Cost / accuracy frontier — why $M=4$ is the production choice

Per-call build time (a $B{=}128$ batch, warm njit cache) against accuracy on the smooth continuation
integrand. The degenerate $M_x{=}1$ kernel is cheap but inaccurate; the **FAST $M=4$** kernel sits at the
knee — sub-percent on the smooth integrand at a few microseconds — while richer meshes buy little on the
quantities the critic actually integrates.
""")
code(r"""
fig, ax = plt.subplots(figsize=(7.4, 4.6))
for _, r in KC.iterrows():
    is_fast = r.label.startswith("FAST"); is_deg = r.M_x == 1
    c = COL["kernel_on"] if is_fast else (COL["kernel_off"] if is_deg else "#495057")
    ax.scatter(r.build_us_b128, 100*r.relerr_smooth, s=140 if is_fast else 80, color=c, zorder=3,
               edgecolor="white", linewidth=1.2)
    ax.annotate(f"{r.label}\n(M={int(r.M)})", (r.build_us_b128, 100*r.relerr_smooth),
                textcoords="offset points", xytext=(8, 6), fontsize=8)
ax.set_yscale("log")
ax.set_ylim(top=ax.get_ylim()[1] * 2.2)   # headroom so annotations clear the title
ax.set(xlabel=r"kernel build time  [$\mu$s / batch of 128]",
       ylabel="mean rel. error, smooth integrand (%)",
       title="F5 · Cost / accuracy frontier — FAST $M{=}4$ at the knee")
fig.tight_layout(); save(fig, "F5_pareto"); plt.show()
display(KC[["label","M","relerr_ES","relerr_smooth","worst_relerr","build_us_b128"]]
        .round({"relerr_ES":5,"relerr_smooth":4,"worst_relerr":3,"build_us_b128":1}))
""")

md(r"""
> **Part I takeaway.** The kernel is mathematically sound: the OU leg is exact to machine precision
> (spectral GH convergence), the jump mesh converges by $N_{\max}{=}2$, and the deterministic target agrees
> with brute-force nested MC to $\lesssim0.1\%$ — a bias ~100× smaller than the single-sample bootstrap
> variance it eliminates. The production **FAST $M=4$** kernel is the cost/accuracy optimum. With the
> backup validated, we turn to the learned policy.
""")

# ===================================================================== PART II
md(r"""
---
# Part II · Validation of the RL pricer

**Protocol.** Every saved policy is evaluated by **forward rollout on one common fresh test set**
(`seed=999`, 65 536 HHK paths, never seen in training). The benchmark is the **full-state LSM-D** pricer
(Chebyshev degree-2 regression, the canonical paper configuration) fit on a separate set (`seed=998`) and
evaluated on the same test paths, at **$M\in\{5,9,17\}$ exercise levels**. The headline comparison is
**kernel-on D4PG @ 4 096 episodes** vs the **kernel-off paper baseline @ 32 768 episodes** (identical
network and hyper-parameters; they differ only in the expected backup and the training budget).
""")

# ---- R1 -------------------------------------------------------------------
md(r"""
### R1 · Forward-rollout pricing vs a strengthened LSM-D baseline

Does the kernel-on policy hold up against a *stronger* LSM than the five-action one — i.e. is the RL edge
real or just a discretisation artefact? We compare mean out-of-sample price (across seeds) for each method
and report the percentage gap $\Delta\%=(V_{\text{RL}}/V_{\text{LSM}}-1)\times100$ against **each** $M$.
""")
code(r"""
def agg_price(df):
    g = df.groupby("seed").price.first()
    return g.mean(), (g.std(ddof=1) if len(g) > 1 else 0.0), len(g)

methods = ["lsm_M5", "lsm_M9", "lsm_M17", "kernel_off", "kernel_on"]
mlabel = {"lsm_M5": "LSM M=5", "lsm_M9": "LSM M=9", "lsm_M17": "LSM M=17",
          "kernel_off": "RL kernel-off\n(32 768 ep)", "kernel_on": "RL kernel-on\n(4 096 ep)"}
mcol = {"lsm_M5": COL["lsm_M5"], "lsm_M9": COL["lsm_M9"], "lsm_M17": COL["lsm_M17"],
        "kernel_off": COL["kernel_off"], "kernel_on": COL["kernel_on"]}

fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0), sharey=False)
delta_rows = []
for ax, reg in zip(axes, REG_ORDER):
    sub = RL[RL.regime == reg]
    means, errs = [], []
    for m in methods:
        d = sub[sub.method == m]
        if m.startswith("lsm"):
            means.append(d.price.iloc[0]); errs.append(d.ci95.iloc[0])
        else:
            mu, sd, n = agg_price(d); means.append(mu); errs.append(1.96*sd/np.sqrt(max(n,1)))
    ax.bar(range(5), means, color=[mcol[m] for m in methods], edgecolor="white",
           yerr=errs, capsize=3, error_kw=dict(lw=1))
    ax.set_xticks(range(5), [mlabel[m].split("\n")[0] for m in methods], rotation=40, ha="right", fontsize=8)
    ax.set_title(REG_LABEL[reg], fontsize=10)
    lo = min(means)*0.985; hi = max(means)*1.01; ax.set_ylim(lo, hi)
    # delta% of kernel_on / kernel_off vs each LSM M
    on_mu = agg_price(sub[sub.method == "kernel_on"])[0]
    off_mu = agg_price(sub[sub.method == "kernel_off"])[0]
    row = {"regime": REG_LABEL[reg]}
    for M in (5, 9, 17):
        lp = sub[sub.method == f"lsm_M{M}"].price.iloc[0]
        row[f"on vs M{M}"] = 100*(on_mu/lp-1); row[f"off vs M{M}"] = 100*(off_mu/lp-1)
    delta_rows.append(row)
axes[0].set_ylabel("out-of-sample price")
fig.suptitle("R1 · Forward-rollout price: RL vs strengthened LSM-D (mean ± 95% CI across seeds)",
             fontweight="bold", y=1.04)
fig.tight_layout(); save(fig, "R1_pricing"); plt.show()

dd = pd.DataFrame(delta_rows).set_index("regime").round(2)
print("Δ% vs LSM at each discretisation M (positive = RL above LSM):")
display(dd[["on vs M5","on vs M9","on vs M17","off vs M5","off vs M9","off vs M17"]])
# Welch test: kernel-on vs kernel-off per regime (per-seed prices)
print("\nWelch test  kernel-on vs kernel-off (per-seed OOS price):")
for reg in REG_ORDER:
    s = RL[RL.regime == reg]
    a = s[s.method == "kernel_on"].groupby("seed").price.first().values
    b = s[s.method == "kernel_off"].groupby("seed").price.first().values
    w = sa.welch_t_test(a, b)
    if w.get("valid", False):
        print(f"  {REG_LABEL[reg]:<24} Δprice={w['mean_diff']:+.4f}  p={w['p']:.3f}  "
              f"(n_on={len(a)}, n_off={len(b)})")
    else:
        print(f"  {REG_LABEL[reg]:<24} (n_on={len(a)}, n_off={len(b)} — too few seeds for Welch)")
""")

# ---- R2 -------------------------------------------------------------------
md(r"""
### R2 · Sample efficiency — kernel-on vs the kernel-off paper baseline @ 32 768

The expected backup makes the TD target deterministic, so the critic learns the continuation value in far
fewer episodes. We track the out-of-sample price against the training budget for the FAST kernel (now run
out to the **same 32 768-episode horizon** as the paper baseline) against the LSM $M{=}5$ benchmark (solid
line, shaded $\pm 1$ Monte-Carlo s.e.). The kernel-on curve **matches** the kernel-off paper policy at
$\approx$2 048 episodes (16×), **surpasses** it by 4 096 (8×), and at the common 32 768 horizon the
kernel-on agent sits at the benchmark while the kernel-off agent is still visibly below it.
""")
code(r"""
on = EE[EE.method == "kernel_on"]
off = EE[EE.method == "kernel_off"]
lsm5 = float(EE.lsm_M5_price.iloc[0])
lsm5_se = float(EE.get("lsm_M5_ci95", pd.Series([np.nan])).iloc[0]) / 1.96  # MC s.e. of the LSM price
agg = on.groupby("episodes").price.agg(["mean", "std", "count"]).reset_index()
off_mu, off_sd = off.price.mean(), off.price.std(ddof=1)

fig, ax = plt.subplots(figsize=(8.4, 4.9))
# LSM benchmark: horizontal line + shaded ±1 s.e. band (request: line + std bounds)
ax.axhline(lsm5, color=COL["lsm_M17"], lw=2.2, ls="-", zorder=2,
           label=f"LSM-D benchmark (M=5) = {lsm5:.4f}")
if np.isfinite(lsm5_se):
    ax.axhspan(lsm5-lsm5_se, lsm5+lsm5_se, color=COL["lsm_M17"], alpha=.13, zorder=0,
               label=r"LSM $\pm 1$ s.e.")
# kernel-on episode-efficiency curve (512 -> 32 768), with seed band
ax.fill_between(agg.episodes, agg["mean"]-agg["std"], agg["mean"]+agg["std"],
                color=COL["kernel_on"], alpha=.16, zorder=1)
ax.plot(agg.episodes, agg["mean"], "o-", color=COL["kernel_on"], lw=2.2, ms=7, zorder=3,
        label=r"RL kernel-on ($M_x{=}2$)")
for _, r in agg.iterrows():
    ax.annotate(f"{100*(r['mean']/lsm5-1):+.2f}%\n(n={int(r['count'])})",
                (r.episodes, r["mean"]), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=7, color=COL["kernel_on"])
# kernel-off paper baseline @ 32 768 ep — dodged slightly in x so it does not sit on
# top of the kernel-on 32 768 point (request: keep the old 32k while extending the new RL).
ax.errorbar([32768*1.16], [off_mu], yerr=[off_sd], fmt="*", ms=20, color=COL["kernel_off"],
            capsize=4, zorder=4, label=f"RL kernel-off @ 32 768 ep ({100*(off_mu/lsm5-1):+.2f}%)")
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_xticks([512, 2048, 4096, 8192, 16384, 32768])
ax.set_xlim(430, 5.0e4)
ax.set(xlabel="training episodes (log scale)", ylabel="out-of-sample price",
       title="R2 · Sample efficiency vs the LSM benchmark (focal $c$=0.04, $\\gamma$=2)")
ax.legend(frameon=True, loc="lower right", fontsize=8.5)
fig.tight_layout(); save(fig, "R2_episode_efficiency"); plt.show()

on32 = agg[agg.episodes == 32768]
match = agg[agg["mean"] >= off_mu].episodes.min()
if len(on32):
    print(f"Kernel-on @32 768 ep: {100*(on32['mean'].iloc[0]/lsm5-1):+.2f}% of LSM "
          f"(n={int(on32['count'].iloc[0])}); kernel-off @32 768 ep: {100*(off_mu/lsm5-1):+.2f}%.")
if np.isfinite(match):
    print(f"Kernel-on matches the 32 768-episode kernel-off price by ~{int(match)} episodes "
          f"({32768/match:.0f}× fewer).")
""")

# ---- R3 -------------------------------------------------------------------
md(r"""
### R3 · Seed robustness — variance collapse

Beyond the mean, the deterministic target sharply reduces **seed-to-seed dispersion**. We compare the
per-seed $\Delta\%$ distributions of the kernel-on and kernel-off policies at the focal regime and test the
spread with Levene's and an $F$ variance-ratio test.
""")
code(r"""
g2 = RL[RL.regime == "g2"]
lsm5 = g2[g2.method == "lsm_M5"].price.iloc[0]
on = 100*(g2[g2.method=="kernel_on"].groupby("seed").price.first()/lsm5 - 1)
off = 100*(g2[g2.method=="kernel_off"].groupby("seed").price.first()/lsm5 - 1)
dfp = pd.concat([pd.DataFrame({"Δ%": on, "method": "kernel-on\n(4 096 ep)"}),
                pd.DataFrame({"Δ%": off, "method": "kernel-off\n(32 768 ep)"})])

fig, ax = plt.subplots(figsize=(6.6, 4.6))
pal = {"kernel-on\n(4 096 ep)": COL["kernel_on"], "kernel-off\n(32 768 ep)": COL["kernel_off"]}
sns.boxplot(data=dfp, x="method", y="Δ%", palette=pal, width=.5, fliersize=0, ax=ax)
sns.stripplot(data=dfp, x="method", y="Δ%", color="#222", size=5, alpha=.7, jitter=.12, ax=ax)
ax.axhline(0, ls=":", color="grey")
ax.set(xlabel="", ylabel=r"$\Delta\%$ vs LSM (M=5)", title="R3 · Seed robustness (focal $c$=0.04, $\\gamma$=2)")
fig.tight_layout(); save(fig, "R3_seed_robustness"); plt.show()

lev = sa.scale_test(on.values, off.values, center="median")
fr = sa.f_var_ratio(off.values, on.values)   # off/on: how much wider the baseline spread is
w = sa.welch_t_test(on.values, off.values)
print(f"kernel-on : mean Δ%={on.mean():+.2f}, std={on.std(ddof=1):.2f} (n={len(on)})")
print(f"kernel-off: mean Δ%={off.mean():+.2f}, std={off.std(ddof=1):.2f} (n={len(off)})")
print(f"variance ratio (off/on) = {fr['var_ratio']:.1f}×   F-test p={fr['p']:.1e}   "
      f"Brown–Forsythe p={lev['p']:.1e}")
print(f"mean difference: Welch p={w['p']:.3f}")
""")

# ---- R4 -------------------------------------------------------------------
md(r"""
### R4 · How much of the RL–LSM gap is the action discretisation?

The central review question: is the RL edge merely an artefact of beating a coarse LSM action grid?
We isolate the discretisation's contribution by sweeping the LSM action grid from the **coarsest
bang-bang** level up to a very fine one, $M\in\{2,3,4,5,9,17\}$ ($M{=}1$ is degenerate — no exercise-size
choice — and is omitted). The key economic fact is that **discretisation only matters where intermediate
exercise is optimal — under a convex cost ($\gamma>1$)**; for $\gamma\le1$ the optimal policy is bang-bang
and $M$ is irrelevant.
Panel (a) shows exactly this — the LSM price is flat in $M$ for no-cost / $\gamma{=}1$ and rises only for
$\gamma{=}1.5,2$ (shaded $\pm$ 95% Monte-Carlo CI) — while the **right axis** tracks the price-calculation
cost, which grows with $M$: refining the grid is not free. Panel (b) then pits the kernel-on RL against the
*strongest* ($M{=}17$) LSM: the RL tracks it within a few tenths of a percent, so the edge is **not** a
five-action artefact — the residual gap is a *policy-quality* gap (addressed in the discussion that
follows), not a benchmarking one.
""")
code(r"""
LSMD = pd.read_csv(D / "lsm_discretisation.csv")
MREF = 2  # rebase the price-gain to the coarsest (bang-bang) grid
cmap = dict(zip(REG_ORDER, sns.color_palette("flare", 4)))
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.7))

# (a) LSM relative price gain from finer discretisation, per regime, with 95% CI band
for reg in REG_ORDER:
    s = LSMD[LSMD.regime == reg].sort_values("M")
    base = s[s.M == MREF].price.iloc[0]
    gain = 100*(s.price.values/base - 1)
    band = 100*s.ci95.values/base                     # marginal 95% MC CI in relative units
    ax[0].plot(s.M, gain, "o-", lw=2, color=cmap[reg], label=REG_LABEL[reg], zorder=3)
    ax[0].fill_between(s.M, gain-band, gain+band, color=cmap[reg], alpha=.10, zorder=1)
ax[0].axhline(0, ls=":", color="grey", lw=1)
ax[0].set_xscale("log"); ax[0].set_xticks([2, 3, 4, 5, 9, 17])
ax[0].xaxis.set_major_formatter(mticker.ScalarFormatter())
ax[0].set(xlabel="LSM action levels $M$", ylabel="LSM price gain vs $M{=}2$ (bang-bang)  (%)",
          title="(a) finer LSM-D helps only under convex cost")
ax[0].legend(frameon=True, fontsize=8, loc="upper left")
# secondary axis: price-calculation time vs M (regime-averaged; ~regime-independent)
axt = ax[0].twinx()
tt = LSMD.groupby("M").wall_total.mean().reset_index().sort_values("M")
axt.plot(tt.M, tt.wall_total, "s--", color="#555", lw=1.5, ms=5, alpha=.85, zorder=2,
         label="LSM price-calc time")
axt.set_ylabel("LSM price-calc time  (s)", color="#555")
axt.tick_params(axis="y", colors="#555"); axt.set_ylim(0, tt.wall_total.max()*1.3)
axt.legend(frameon=False, fontsize=7.5, loc="lower right")

# (b) kernel-on RL vs the strengthened LSM at each M (convex regimes, where M matters)
for reg in ["g15", "g2"]:
    g = RL[(RL.regime == reg) & (RL.method == "kernel_on")].groupby("seed").price.first()
    on_mu, on_se = g.mean(), g.std(ddof=1)/np.sqrt(len(g))
    s = LSMD[LSMD.regime == reg].sort_values("M")
    delt = 100*(on_mu/s.price.values - 1)
    se = np.sqrt(on_se**2 + (s.ci95.values/1.96)**2)          # combine RL-seed + LSM MC error
    band = 100*1.96*se/s.price.values
    ax[1].plot(s.M, delt, "s-", lw=2, color=cmap[reg], ms=8, label=REG_LABEL[reg], zorder=3)
    ax[1].fill_between(s.M, delt-band, delt+band, color=cmap[reg], alpha=.10, zorder=1)
ax[1].axhline(0, ls=":", color="grey")
ax[1].set_xscale("log"); ax[1].set_xticks([2, 3, 4, 5, 9, 17])
ax[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
ax[1].set(xlabel="LSM action levels $M$", ylabel=r"kernel-on $\Delta\%$ vs LSM($M$)",
          title="(b) RL kernel-on tracks even the $M{=}17$ LSM")
ax[1].legend(frameon=True, fontsize=8)
fig.tight_layout(); save(fig, "R4_discretisation"); plt.show()

g2on = RL[(RL.regime=="g2") & (RL.method=="kernel_on")].groupby("seed").price.first().mean()
print("Focal kernel-on Δ% vs strengthening LSM:",
      {f"M={int(r.M)}": round(100*(g2on/r.price-1), 2) for _, r in LSMD[LSMD.regime=="g2"].sort_values("M").iterrows()})
print("LSM price gain M=2→17 (bang-bang → fine):",
      {reg: round(100*(LSMD[(LSMD.regime==reg)&(LSMD.M==17)].price.iloc[0]/
                       LSMD[(LSMD.regime==reg)&(LSMD.M==2)].price.iloc[0]-1), 2) for reg in REG_ORDER})
print("LSM price-calc time (s) vs M:",
      {int(r.M): round(r.wall_total, 1) for _, r in LSMD.groupby("M").wall_total.mean().reset_index().iterrows()})
""")

# ---- R5 -------------------------------------------------------------------
md(r"""
### R5 · End-to-end role of $M_x$ — the math plateau in the *learned* outcome

F2 showed the quadrature target plateaus once $M_x\geq2$. Here we confirm the same threshold survives all
the way to the **trained policy's price**: sweeping $M_x\in\{1,2,3,4,6\}$ (fixed budget, 4 096 episodes),
$M_x{=}1$ collapses while $M_x\geq2$ is statistically flat — tying the kernel mathematics directly to
pricing performance.
""")
code(r"""
agg = MX.groupby("M_x").delta_pct.agg(["mean", "std", "count"]).reset_index()
fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.errorbar(agg.M_x, agg["mean"], yerr=agg["std"], fmt="o-", color=COL["kernel_on"], lw=2,
            capsize=4, ms=8, label=r"trained $\Delta\%$ (mean ± seed std)")
ax.axvspan(0.5, 1.5, color="#ffe3e3", alpha=.7, zorder=0)
ax.text(1.0, agg["mean"].min(), "collapse", ha="center", color=COL["kernel_off"], fontsize=9)
ax.axhline(0, ls=":", color="grey")
ax.set(xlabel=r"Gauss–Hermite nodes $M_x$ (total $M=2M_x$)", ylabel=r"$\Delta\%$ vs LSM (M=5)",
       title="R5 · $M_x$ isolation in the trained policy (focal, 4 096 ep)")
ax.legend(frameon=True)
fig.tight_layout(); save(fig, "R5_mx_isolation"); plt.show()

a1 = MX[MX.M_x == 1].delta_pct.values
a2 = MX[MX.M_x == 2].delta_pct.values
w = sa.welch_t_test(a2, a1); fr = sa.f_var_ratio(a1, a2)
print(f"M_x=1: mean Δ%={a1.mean():+.2f} (std {a1.std(ddof=1):.2f}) | "
      f"M_x=2: mean Δ%={a2.mean():+.2f} (std {a2.std(ddof=1):.2f})")
print(f"M_x=2 vs M_x=1: Welch p={w['p']:.1e}; variance ratio (1/2)={fr['var_ratio']:.0f}×")
print("Pairwise Welch among M_x>=2 (expect all non-significant):")
import itertools
for i, j in itertools.combinations([2, 3, 4, 6], 2):
    wj = sa.welch_t_test(MX[MX.M_x==i].delta_pct.values, MX[MX.M_x==j].delta_pct.values)
    print(f"  M_x={i} vs {j}: p={wj['p']:.2f}")
""")

# ---- Closing --------------------------------------------------------------
md(r"""
---
### Summary — headline table and hypothesis tests

The focal-regime replication of the headline result, with the full statistical battery.
""")
code(r"""
g2 = RL[RL.regime == "g2"]
lsm5 = g2[g2.method == "lsm_M5"].price.iloc[0]
rows = []
for m, ep in [("kernel_on", 4096), ("kernel_off", 32768)]:
    g = g2[g2.method == m].groupby("seed").price.first()
    d = 100*(g/lsm5 - 1)
    rows.append({"method": m, "episodes": ep, "n_seeds": len(g),
                 "mean Δ%": d.mean(), "seed std": d.std(ddof=1),
                 "best Δ%": d.max(), "worst Δ%": d.min()})
tbl = pd.DataFrame(rows).set_index("method").round(3)
print("Focal regime  c=0.04, γ=2  —  Δ% vs full-state LSM-D (M=5):")
display(tbl)

on = 100*(g2[g2.method=="kernel_on"].groupby("seed").price.first()/lsm5 - 1)
off = 100*(g2[g2.method=="kernel_off"].groupby("seed").price.first()/lsm5 - 1)
w = sa.welch_t_test(on.values, off.values); fr = sa.f_var_ratio(off.values, on.values)
mde = sa.minimum_detectable_effect(len(on), len(off),
                                   pooled_std=np.sqrt((on.var(ddof=1)+off.var(ddof=1))/2))
print(f"\nkernel-on − kernel-off:  Δmean = {w['mean_diff']:+.2f} pp  (Welch p={w['p']:.3f})")
print(f"variance collapse:       {fr['var_ratio']:.1f}× tighter  (F-test p={fr['p']:.1e})")
print(f"minimum detectable effect (n={len(on)}/{len(off)}, 80% power): {mde['mde']:.2f} pp")
""")
md(r"""
---
### Discussion · Closing the remaining gap to LSM (within the same budget)

Both LSM-D and the RL rollout are *lower bounds* on the true price — each follows a sub-optimal exercise
policy, so a **higher** out-of-sample value is the better estimate. The kernel-on policy still sits a few
tenths of a percent below LSM at the focal regime: a **policy-quality** gap, not a benchmarking one. The
dominant cause is that the agents above use the **v61 hyper-parameters tuned for the high-variance
single-sample TD target**; the kernel has since made that target *deterministic*, so the regularisation that
once fought sampling noise (very low critic LR, heavy/long exploration noise, soft PER, slow target
tracking) is now mostly counter-productive. Re-tuning the *learner* for the deterministic target closes part
of the gap **at no extra episode cost** — the figure shows the hyper-parameter lever at a fixed 4 096
episodes; a full-horizon linear noise decay + eval-time EMA lifts focal $\Delta\%$ from $-0.66\%$ to
$-0.42\%$ and cuts seed std $\sim1.7\times$. (The orthogonal *budget* lever is visible in R2: the same kernel
at 16 384 episodes already reaches $-0.10\%$.) The three highest-ROI directions, all within budget:
**(1)** adopt the deterministic-target retune (higher critic LR, linear-noise→floor, eval-EMA);
**(2)** revisit replay — soften/disable PER (the kernel removed the target variance PER compensated for) with
faster target-network tracking; **(3)** extract more gradient steps per episode (an exact target makes extra
updates pure signal, not amplified variance) and soften the actor's $\beta$-sigmoid so it can resolve the
intermediate exercise that convex costs reward.
""")
code(r"""
cg = pd.read_csv(D / "r6_closing_gap.csv").set_index("config")
order = ["v61 paper (32k)", "v64 (4k)", "v64 (32k)"]
cg = cg.reindex([o for o in order if o in cg.index]).reset_index()

fig, ax = plt.subplots(figsize=(7.6, 4.4))
label_map = {
    "v61 paper (32k)": "v61 paper\n(32k ep)",
    "v64 (4k)": "v64\n(4k ep)",
    "v64 (32k)": "v64\n(32k ep)"
}
labels = [label_map[c] for c in cg["config"]]

color_map = {
    "v61 paper (32k)": COL["kernel_off"],
    "v64 (4k)": COL["kernel_on"],
    "v64 (32k)": COL["ok"]
}
colors = [color_map[c] for c in cg["config"]]

ax.bar(range(len(cg)), cg.mean_delta, yerr=cg.seed_std, capsize=5,
       color=colors, edgecolor="white", zorder=3,
       error_kw=dict(zorder=4, ecolor="#222"))

for i, r in cg.iterrows():
    y_pos = -0.04 if r.mean_delta < 0 else 0.01
    va_dir = "top" if r.mean_delta < 0 else "bottom"
    text_color = "white" if r.mean_delta < 0 else "black"
    ax.text(i, y_pos, f"{r.mean_delta:+.2f}%", ha="center", va=va_dir, fontsize=11,
            fontweight="bold", color=text_color, zorder=5)
    
    err_pos = r.mean_delta - r.seed_std - 0.015 if r.mean_delta < 0 else r.mean_delta + r.seed_std + 0.015
    err_va = "top" if r.mean_delta < 0 else "bottom"
    ax.text(i, err_pos, f"$\\pm${r.seed_std:.2f}", ha="center",
            va=err_va, fontsize=8.5, color="#444", zorder=5)

ax.axhline(0, color=COL["lsm_M17"], lw=2.2, zorder=2, label="LSM (M=5) benchmark")
ax.set_xticks(range(len(cg)), labels, fontsize=9)
ax.set(ylabel=r"focal $\Delta\%$ vs LSM (mean ± seed std)",
       title="Discussion · v61 paper vs v64 re-baseline (focal, c=0.04, gamma=2)")

ymin = float(cg.mean_delta.min() - cg.seed_std.max() - 0.16)
ymax = float(max(0.05, cg.mean_delta.max() + cg.seed_std.max() + 0.16))
ax.set_ylim(ymin, ymax)

ax.legend(frameon=True, loc="lower center")
fig.tight_layout(); save(fig, "D_closing_gap"); plt.show()

row_61 = cg[cg.config == "v61 paper (32k)"].iloc[0]
row_64 = cg[cg.config == "v64 (4k)"].iloc[0]
print(f"v61 paper (32k) mean delta: {row_61.mean_delta:+.2f}% | v64 (4k) mean delta: {row_64.mean_delta:+.2f}%")
if "v64 (32k)" in cg.config.values:
    row_64_32k = cg[cg.config == "v64 (32k)"].iloc[0]
    print(f"v64 (32k) mean delta: {row_64_32k.mean_delta:+.2f}%")
""")
md(r"""
> **Part II takeaway.** On the common fresh test set the kernel-on policy prices **within $\approx$0.5–1% of
> the full-state LSM-D** at every regime — competitive even with the strengthened **$M{=}17$** benchmark.
> Tellingly, raising the LSM resolution from $M{=}5$ to $M{=}17$ **closes part of the nominal RL-vs-LSM gap at
> convex costs**: the apparent edge over the five-action grid is partly a *discretisation* effect, exactly the
> reviewer's concern — which is why we report against all three resolutions rather than $M{=}5$ alone. The
> kernel's decisive contribution is **efficiency and stability**: kernel-on @ 4 096 episodes **matches or
> beats the kernel-off paper baseline @ 32 768** (focal $-0.69\%$ vs $-1.06\%$) at **8× fewer episodes** and
> with markedly **tighter seed dispersion**; the same kernel reaches the benchmark ($-0.10\%$) by 16 384
> episodes, and the residual gap is a within-budget *tuning* gap (Discussion above). The quadrature threshold
> $M_x\geq2$ from Part I governs the trained outcome end-to-end.

---
#### Reproducibility
All figures and tables derive from the CSVs in the companion folder, produced by
`gen_kernel_validation.py` (Part I), `gen_rl_validation.py` (R1–R5), and `gen_r4_discretisation.py`
(R4 action-grid sweep, with LSM compute time) over saved agents. Test set: `seed=999`, 65 536 paths;
LSM-train: `seed=998`; LSM: full-state Chebyshev degree-2, action grid $M\in\{2,3,4,5,9,17\}$. The R2
kernel-on point at 32 768 episodes was trained with the code version contemporaneous with the rest of the
R2 curve (a `git worktree` at the matching commit) and evaluated through the same common-test-set pipeline.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3 (EP11)", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}})

# ── Safety guard: refuse to overwrite a hand-edited notebook ─────────────────
# If the existing notebook has MORE cells than this template generates, it means
# the user has added cells manually and running this script would destroy that work.
# Pass --force on the CLI to override.
import sys as _sys
if NB.exists():
    _existing = nbf.read(str(NB), as_version=4)
    if len(_existing.cells) > len(cells):
        if "--force" not in _sys.argv:
            print(
                f"⛔  REFUSED: {NB.name} already has {len(_existing.cells)} cells "
                f"but this template only generates {len(cells)}.\n"
                f"   Looks like you've hand-edited it.  Pass --force to overwrite anyway."
            )
            _sys.exit(1)
        else:
            import shutil as _sh, datetime as _dt
            bak = NB.with_suffix(f'.bak_{_dt.datetime.now():%Y%m%d_%H%M%S}.ipynb')
            _sh.copy2(NB, bak)
            print(f"⚠️  --force: backed up existing notebook → {bak.name}")

nbf.write(nb, str(NB))
print(f"Wrote {NB} with {len(cells)} cells.")
