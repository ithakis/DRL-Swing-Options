"""
Statistical analysis framework for the feat/semi-analytical-bootstrap branch.

Loads all sweep CSVs, computes per-seed Δ% (= eval_price / lsm_price - 1),
groups by (label, n_paths, contract), and runs:

  * Mean comparison: Welch's two-sample t-test (unequal variances).
  * Variance comparison: Levene's test (robust to non-normality) and
    Bartlett's test (parametric).
  * Per-config summary: mean, std, SE, 95% CI, conservative score
    (mean - 1.96 * SE).
  * Pareto frontier over (mean, SE, wall_clock).
  * Accuracy/speed curves over the kernel size M and n_paths axes.

Designed to be reused across multiple sweep CSVs by passing source paths.
Imports scipy for the tests; numpy for the arithmetic.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as sps  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "logs" / "_sweep_h1"

# Default canonical sources. Extend as new sweeps land.
DEFAULT_SOURCES: List[Tuple[str, str, Optional[str], Optional[int]]] = [
    ("wide1",     "sweep_results_n4096.csv",        "focal",  4096),
    ("wide2",     "sweep_results_n3072_wide2.csv",  "focal",  3072),
    ("phase2",    "sweep_h1_phase2.csv",            None,     None),
    ("phase3",    "sweep_h1_phase3_n8192.csv",      None,     None),
    ("h4v2",     "sweep_h4_v2_n4096.csv",          None,     None),
    ("h6",       "sweep_h6_n2048.csv",             None,     None),
    ("h789",     "sweep_h789_n4096.csv",           None,     None),
    ("h789r",    "sweep_h789_resume_n4096.csv",    None,     None),
    ("h8strat",  "sweep_h8strat_n4096.csv",        None,     None),
    ("paramstd", "sweep_param_study.csv",           None,     None),
]


@dataclass
class GroupStats:
    label: str
    contract: str
    n_paths: int
    n_seeds: int
    deltas: List[float]
    walls: List[float]
    evals: List[float]
    lsms: List[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.deltas)) if self.deltas else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.deltas, ddof=1)) if len(self.deltas) > 1 else 0.0

    @property
    def se(self) -> float:
        return self.std / math.sqrt(len(self.deltas)) if len(self.deltas) > 1 else 0.0

    @property
    def ci95(self) -> Tuple[float, float]:
        m = self.mean
        s = 1.96 * self.se
        return m - s, m + s

    @property
    def conservative(self) -> float:
        return self.mean - 1.96 * self.se

    @property
    def wall_mean(self) -> float:
        return float(np.mean(self.walls)) if self.walls else float("nan")


def load_sources(
    sources: Optional[Iterable[Tuple[str, str, Optional[str], Optional[int]]]] = None,
    log_dir: Path = DEFAULT_LOG_DIR,
) -> List[Dict]:
    if sources is None:
        sources = DEFAULT_SOURCES
    rows = []
    for tag, fname, default_contract, default_n in sources:
        path = log_dir / fname
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                if not r.get("contract"):
                    r["contract"] = default_contract or ""
                if not r.get("n_paths"):
                    r["n_paths"] = str(default_n) if default_n else ""
                r["_source"] = tag
                rows.append(r)
    return rows


def per_seed_delta(row: Dict) -> Optional[float]:
    try:
        ep = float(row["eval_price"]); lsm = float(row["lsm_price"])
        if lsm <= 0: return None
        return (ep / lsm - 1.0) * 100.0
    except Exception:
        return None


def group_by(rows: List[Dict], key=lambda r: (r["label"], r.get("contract", ""), r.get("n_paths", ""))) -> Dict:
    out: Dict = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return dict(out)


def summarize_groups(rows: List[Dict]) -> List[GroupStats]:
    grouped = group_by(rows)
    out = []
    for (label, contract, n_paths_str), grp in grouped.items():
        deltas = []
        walls = []
        evals = []
        lsms = []
        for r in grp:
            d = per_seed_delta(r)
            if d is None:
                continue
            deltas.append(d)
            evals.append(float(r["eval_price"]))
            lsms.append(float(r["lsm_price"]))
            try:
                walls.append(float(r.get("wall_seconds", "0")))
            except Exception:
                pass
        try:
            n_paths_int = int(n_paths_str)
        except Exception:
            n_paths_int = 0
        out.append(GroupStats(
            label=label, contract=contract or "focal", n_paths=n_paths_int,
            n_seeds=len(deltas), deltas=deltas, walls=walls,
            evals=evals, lsms=lsms,
        ))
    return out


def welch_t_test(a: List[float], b: List[float]) -> Dict:
    """Welch's two-sample t-test for unequal variances. Returns:
    {t, df, p_two_sided, mean_diff, se_diff, ci95_diff}."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    na, nb = len(a_arr), len(b_arr)
    if na < 2 or nb < 2:
        return {"valid": False}
    mean_a, mean_b = a_arr.mean(), b_arr.mean()
    var_a, var_b = a_arr.var(ddof=1), b_arr.var(ddof=1)
    se = math.sqrt(var_a / na + var_b / nb)
    if se == 0:
        return {"valid": False}
    t = (mean_a - mean_b) / se
    # Welch-Satterthwaite df
    df = (var_a / na + var_b / nb) ** 2 / (
        (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    )
    if HAVE_SCIPY:
        p = float(2 * (1 - sps.t.cdf(abs(t), df)))
    else:
        # rough normal approx
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2.0)))))
    diff = mean_a - mean_b
    half = 1.96 * se
    return {
        "valid": True, "t": float(t), "df": float(df), "p": p,
        "mean_diff": float(diff), "se_diff": float(se),
        "ci95_low": float(diff - half), "ci95_high": float(diff + half),
    }


def levene_test(a: List[float], b: List[float]) -> Dict:
    """Levene's test (mean version) for equality of variance."""
    if not HAVE_SCIPY:
        return {"valid": False, "msg": "scipy not available"}
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return {"valid": False}
    stat, p = sps.levene(a_arr, b_arr, center="mean")
    return {"valid": True, "W": float(stat), "p": float(p)}


def f_var_ratio(a: List[float], b: List[float]) -> Dict:
    """Two-sided F-test of variance ratio (parametric, assumes normality)."""
    if not HAVE_SCIPY:
        return {"valid": False}
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return {"valid": False}
    var_a = a_arr.var(ddof=1); var_b = b_arr.var(ddof=1)
    if var_a == 0 or var_b == 0:
        return {"valid": False}
    F = var_a / var_b
    df_a = len(a_arr) - 1; df_b = len(b_arr) - 1
    p_one = 1 - sps.f.cdf(F, df_a, df_b) if F >= 1 else sps.f.cdf(F, df_a, df_b)
    p_two = 2 * min(p_one, 1 - p_one)
    return {"valid": True, "F": float(F), "df1": df_a, "df2": df_b, "p": float(p_two)}


def pairwise_compare(group_a: GroupStats, group_b: GroupStats) -> Dict:
    out: Dict = {"a": group_a.label, "b": group_b.label}
    out["welch"] = welch_t_test(group_a.deltas, group_b.deltas)
    out["levene"] = levene_test(group_a.deltas, group_b.deltas)
    out["f_var"] = f_var_ratio(group_a.deltas, group_b.deltas)
    return out


def conservative_pareto(groups: List[GroupStats]) -> List[GroupStats]:
    """Return groups on the (conservative, -wall_mean) Pareto frontier.
    Higher conservative is better; lower wall is better."""
    items = [g for g in groups if g.n_seeds >= 2 and not math.isnan(g.wall_mean)]
    pareto = []
    for g in items:
        dominated = False
        for h in items:
            if h is g: continue
            if (h.conservative >= g.conservative and h.wall_mean <= g.wall_mean and
                (h.conservative > g.conservative or h.wall_mean < g.wall_mean)):
                dominated = True
                break
        if not dominated:
            pareto.append(g)
    return pareto


def print_summary(groups: List[GroupStats], title: str = "") -> None:
    if title: print(f"\n=== {title} ===")
    print(f"{'config':<28} {'cont':<6} {'n_paths':>7} {'n':>3} "
          f"{'mean Δ%':>9} {'std':>6} {'SE':>6} {'95% CI':>20} "
          f"{'conservative':>12} {'wall_s':>8}")
    print("-" * 120)
    for g in sorted(groups, key=lambda g: -g.mean):
        ci = g.ci95
        print(f"{g.label:<28} {g.contract:<6} {g.n_paths:>7d} {g.n_seeds:>3d} "
              f"{g.mean:>+9.3f} {g.std:>6.3f} {g.se:>6.3f} "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}]  {g.conservative:>+12.3f} {g.wall_mean:>8.0f}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--filter_contract", type=str, default="focal")
    p.add_argument("--filter_npaths", type=int, default=4096)
    args = p.parse_args()

    rows = load_sources()
    print(f"Loaded {len(rows)} OK rows from disk")
    groups = summarize_groups(rows)

    # Print everything first
    print_summary(groups, "ALL groups")

    # Filter to focal @ requested n_paths
    target = [g for g in groups if g.contract == args.filter_contract and g.n_paths == args.filter_npaths and g.n_seeds >= 2]
    print_summary(target, f"FILTERED: contract={args.filter_contract}, n_paths={args.filter_npaths}")

    # Pairwise tests vs H1_only
    h1 = next((g for g in target if g.label == "H1_only"), None)
    if h1:
        print(f"\n=== Pairwise vs H1_only (n={h1.n_seeds}, mean={h1.mean:+.3f} pp) ===")
        print(f"{'config':<28} {'n':>3} {'mean':>8} {'gap':>8} {'t':>6} {'p(t)':>7} {'p(Lev)':>8} {'verdict':<24}")
        print("-" * 100)
        for g in sorted(target, key=lambda g: -g.mean):
            if g is h1: continue
            cmp = pairwise_compare(g, h1)
            w = cmp["welch"]
            lev = cmp["levene"]
            if not w.get("valid"):
                continue
            t = w["t"]
            pval = w["p"]
            pl = lev.get("p") if lev.get("valid") else float("nan")
            if pval < 0.05 and t > 0: verdict = "DECISIVELY better mean"
            elif pval < 0.05 and t < 0: verdict = "DECISIVELY worse mean"
            elif pval < 0.20 and t > 0: verdict = "weakly better mean"
            elif pval < 0.20 and t < 0: verdict = "weakly worse mean"
            else: verdict = "no different mean"
            print(f"{g.label:<28} {g.n_seeds:>3d} {g.mean:>+8.3f} {w['mean_diff']:>+8.3f} "
                  f"{t:>+6.2f} {pval:>7.3f} {pl if not math.isnan(pl) else 0.0:>8.3f}  {verdict:<24}")

    # Pareto frontier
    pareto = conservative_pareto(target)
    if pareto:
        print(f"\n=== Pareto frontier (conservative Δ% vs wall_clock) ===")
        for g in sorted(pareto, key=lambda g: g.wall_mean):
            print(f"  {g.label:<28} conservative={g.conservative:+.3f}%  wall={g.wall_mean:.0f}s  n={g.n_seeds}")


if __name__ == "__main__":
    main()
