"""Pure-statistics utilities for D4PG sweep analysis.

Provides mean/variance inference, paired-seed tests, Holm–Bonferroni correction,
and conservative Pareto selection over GroupStats objects. Consumes rows with
columns: label, contract, n_paths, seed, eval_price, lsm_price, wall_seconds, status.

Delta% = 100 * (eval_price / lsm_price - 1) computed per row.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as sps  # type: ignore

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False



def _norm_ppf(prob: float) -> float:
    if HAVE_SCIPY:
        return float(sps.norm.ppf(prob))
    return statistics.NormalDist().inv_cdf(prob)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _percentile_interval(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return low, high


def bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 4000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"valid": False}
    if arr.size == 1:
        val = float(stat_fn(arr))
        return {"valid": True, "estimate": val, "ci_low": val, "ci_high": val}
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    idx = np.arange(arr.size)
    for boot_idx in range(n_bootstrap):
        draw = rng.choice(idx, size=arr.size, replace=True)
        samples[boot_idx] = stat_fn(arr[draw])
    ci_low, ci_high = _percentile_interval(samples, alpha=alpha)
    return {
        "valid": True,
        "estimate": float(stat_fn(arr)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


@dataclass
class GroupStats:
    label: str
    contract: str
    n_paths: int
    seeds: List[int]
    deltas: List[float]
    walls: List[float]
    evals: List[float]
    lsms: List[float]

    def __post_init__(self) -> None:
        self.delta_by_seed = {seed: delta for seed, delta in zip(self.seeds, self.deltas)}

    @property
    def n_seeds(self) -> int:
        return len(self.deltas)

    @property
    def mean(self) -> float:
        return float(np.mean(self.deltas)) if self.deltas else float("nan")

    @property
    def median(self) -> float:
        return float(np.median(self.deltas)) if self.deltas else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.deltas, ddof=1)) if self.n_seeds > 1 else 0.0

    @property
    def var(self) -> float:
        return self.std**2

    @property
    def se(self) -> float:
        return self.std / math.sqrt(self.n_seeds) if self.n_seeds > 1 else 0.0

    @property
    def ci95(self) -> Tuple[float, float]:
        half = 1.96 * self.se
        return self.mean - half, self.mean + half

    @property
    def conservative(self) -> float:
        return self.mean - 1.96 * self.se

    @property
    def wall_mean(self) -> float:
        return float(np.mean(self.walls)) if self.walls else float("nan")

    @property
    def matched_seed_count(self) -> int:
        return len(self.delta_by_seed)

    def bootstrap_std_ci(self, n_bootstrap: int = 4000) -> Dict[str, float]:
        return bootstrap_ci(
            self.deltas,
            stat_fn=lambda arr: float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            n_bootstrap=n_bootstrap,
            seed=abs(hash((self.label, self.contract, self.n_paths, "std"))) % (2**32),
        )

    def bootstrap_var_ci(self, n_bootstrap: int = 4000) -> Dict[str, float]:
        return bootstrap_ci(
            self.deltas,
            stat_fn=lambda arr: float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0,
            n_bootstrap=n_bootstrap,
            seed=abs(hash((self.label, self.contract, self.n_paths, "var"))) % (2**32),
        )

    def to_payload(self, n_bootstrap: int = 4000) -> Dict[str, object]:
        ci_low, ci_high = self.ci95
        std_ci = self.bootstrap_std_ci(n_bootstrap=n_bootstrap)
        var_ci = self.bootstrap_var_ci(n_bootstrap=n_bootstrap)
        return {
            "label": self.label,
            "contract": self.contract,
            "n_paths": self.n_paths,
            "n_seeds": self.n_seeds,
            "seeds": list(self.seeds),
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "var": self.var,
            "se": self.se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "conservative": self.conservative,
            "wall_mean": self.wall_mean,
            "std_ci_low": std_ci.get("ci_low"),
            "std_ci_high": std_ci.get("ci_high"),
            "var_ci_low": var_ci.get("ci_low"),
            "var_ci_high": var_ci.get("ci_high"),
        }



def per_seed_delta(row: Dict[str, str]) -> Optional[float]:
    eval_price = _safe_float(row.get("eval_price"))
    lsm_price = _safe_float(row.get("lsm_price"))
    if not math.isfinite(eval_price) or not math.isfinite(lsm_price) or lsm_price <= 0:
        return None
    return (eval_price / lsm_price - 1.0) * 100.0


def summarize_groups(rows: List[Dict[str, str]]) -> List[GroupStats]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row.get("label", ""), row.get("contract", "") or "focal", _safe_int(row.get("n_paths")))
        ].append(row)

    out: List[GroupStats] = []
    for (label, contract, n_paths), group_rows in grouped.items():
        seeds: List[int] = []
        deltas: List[float] = []
        walls: List[float] = []
        evals: List[float] = []
        lsms: List[float] = []
        for row in sorted(group_rows, key=lambda item: _safe_int(item.get("seed"))):
            delta = per_seed_delta(row)
            if delta is None:
                continue
            seeds.append(_safe_int(row.get("seed")))
            deltas.append(delta)
            evals.append(_safe_float(row.get("eval_price"), 0.0))
            lsms.append(_safe_float(row.get("lsm_price"), 0.0))
            wall = _safe_float(row.get("wall_seconds"))
            if math.isfinite(wall) and wall > 0:
                walls.append(wall)
        out.append(
            GroupStats(
                label=label,
                contract=contract,
                n_paths=n_paths,
                seeds=seeds,
                deltas=deltas,
                walls=walls,
                evals=evals,
                lsms=lsms,
            )
        )
    return sorted(out, key=lambda group: (group.contract, group.n_paths, group.label))


def group_lookup(groups: Sequence[GroupStats]) -> Dict[Tuple[str, str, int], GroupStats]:
    return {(group.label, group.contract, group.n_paths): group for group in groups}


def filter_groups(
    groups: Sequence[GroupStats],
    contract: Optional[str] = None,
    n_paths: Optional[int] = None,
    min_seeds: int = 1,
) -> List[GroupStats]:
    filtered = []
    for group in groups:
        if contract is not None and group.contract != contract:
            continue
        if n_paths is not None and group.n_paths != n_paths:
            continue
        if group.n_seeds < min_seeds:
            continue
        filtered.append(group)
    return filtered


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    na = a_arr.size
    nb = b_arr.size
    if na < 2 or nb < 2:
        return {"valid": False}
    mean_a = float(a_arr.mean())
    mean_b = float(b_arr.mean())
    var_a = float(a_arr.var(ddof=1))
    var_b = float(b_arr.var(ddof=1))
    se_diff = math.sqrt(var_a / na + var_b / nb)
    if se_diff == 0:
        return {"valid": False}
    t_stat = (mean_a - mean_b) / se_diff
    df = (var_a / na + var_b / nb) ** 2 / (((var_a / na) ** 2) / (na - 1) + ((var_b / nb) ** 2) / (nb - 1))
    if HAVE_SCIPY:
        p_value = float(2 * (1 - sps.t.cdf(abs(t_stat), df)))
    else:
        p_value = float(2 * (1 - statistics.NormalDist().cdf(abs(t_stat))))
    half = 1.96 * se_diff
    return {
        "valid": True,
        "mean_diff": mean_a - mean_b,
        "se_diff": se_diff,
        "t": t_stat,
        "df": float(df),
        "p": p_value,
        "ci95_low": mean_a - mean_b - half,
        "ci95_high": mean_a - mean_b + half,
    }


def scale_test(a: Sequence[float], b: Sequence[float], center: str) -> Dict[str, float]:
    if not HAVE_SCIPY:
        return {"valid": False}
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.size < 2 or b_arr.size < 2:
        return {"valid": False}
    stat, p_value = sps.levene(a_arr, b_arr, center=center)
    return {"valid": True, "stat": float(stat), "p": float(p_value), "center": center}


def f_var_ratio(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    if not HAVE_SCIPY:
        return {"valid": False}
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.size < 2 or b_arr.size < 2:
        return {"valid": False}
    var_a = float(a_arr.var(ddof=1))
    var_b = float(b_arr.var(ddof=1))
    if var_a <= 0 or var_b <= 0:
        return {"valid": False}
    f_stat = var_a / var_b
    df1 = a_arr.size - 1
    df2 = b_arr.size - 1
    lower_tail = float(sps.f.cdf(f_stat, df1, df2))
    upper_tail = float(1 - lower_tail)
    p_value = 2 * min(lower_tail, upper_tail)
    return {
        "valid": True,
        "F": f_stat,
        "df1": float(df1),
        "df2": float(df2),
        "p": float(min(p_value, 1.0)),
        "var_ratio": f_stat,
        "std_ratio": math.sqrt(f_stat),
    }


def paired_seed_test(group_a: GroupStats, group_b: GroupStats) -> Dict[str, object]:
    shared = sorted(set(group_a.delta_by_seed).intersection(group_b.delta_by_seed))
    if len(shared) < 2:
        return {"valid": False, "n_pairs": len(shared), "shared_seeds": shared}
    diffs = np.asarray(
        [group_a.delta_by_seed[seed] - group_b.delta_by_seed[seed] for seed in shared], dtype=np.float64
    )
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    se_diff = std_diff / math.sqrt(diffs.size) if diffs.size > 1 else 0.0
    if se_diff == 0:
        return {
            "valid": True,
            "n_pairs": int(diffs.size),
            "shared_seeds": shared,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "se_diff": se_diff,
            "t": 0.0,
            "df": float(diffs.size - 1),
            "p": 1.0,
            "ci95_low": mean_diff,
            "ci95_high": mean_diff,
        }
    t_stat = mean_diff / se_diff
    df = diffs.size - 1
    if HAVE_SCIPY:
        p_value = float(2 * (1 - sps.t.cdf(abs(t_stat), df)))
    else:
        p_value = float(2 * (1 - statistics.NormalDist().cdf(abs(t_stat))))
    half = 1.96 * se_diff
    return {
        "valid": True,
        "n_pairs": int(diffs.size),
        "shared_seeds": shared,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "t": t_stat,
        "df": float(df),
        "p": p_value,
        "ci95_low": mean_diff - half,
        "ci95_high": mean_diff + half,
    }


def minimum_detectable_effect(
    n_a: int,
    n_b: int,
    pooled_std: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, float]:
    if n_a < 2 or n_b < 2 or pooled_std <= 0:
        return {"valid": False}
    z_alpha = _norm_ppf(1.0 - alpha / 2.0)
    z_beta = _norm_ppf(power)
    se = pooled_std * math.sqrt(1.0 / n_a + 1.0 / n_b)
    return {
        "valid": True,
        "alpha": alpha,
        "power": power,
        "mde": (z_alpha + z_beta) * se,
    }


def paired_minimum_detectable_effect(
    std_diff: float,
    n_pairs: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, float]:
    if n_pairs < 2 or std_diff <= 0:
        return {"valid": False}
    z_alpha = _norm_ppf(1.0 - alpha / 2.0)
    z_beta = _norm_ppf(power)
    return {
        "valid": True,
        "alpha": alpha,
        "power": power,
        "mde": (z_alpha + z_beta) * std_diff / math.sqrt(n_pairs),
    }


def holm_bonferroni(
    pvalues: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, object]]:
    """Holm–Bonferroni step-down correction over a family of raw p-values.

    Returns, keyed by the same labels, dicts with:
      p_raw, p_adj (monotone-enforced adjusted p), reject (bool at FWER=alpha), rank.
    NaN p-values are carried through as non-rejections (rank last) so a missing test
    never spuriously "passes". No behaviour change to existing functions.
    """
    items = list(pvalues.items())
    # Sort ascending by p; NaN sorts last.
    def _key(kv):
        p = kv[1]
        return (1, 0.0) if (p is None or not math.isfinite(p)) else (0, float(p))

    ordered = sorted(items, key=_key)
    m = len(ordered)
    out: Dict[str, Dict[str, object]] = {}
    running_max = 0.0
    still_rejecting = True
    for rank, (label, p) in enumerate(ordered):  # rank 0-based
        finite = p is not None and math.isfinite(p)
        if finite:
            p_adj = min(1.0, (m - rank) * float(p))
            p_adj = max(p_adj, running_max)  # enforce monotonicity
            running_max = p_adj
            reject = still_rejecting and (p_adj < alpha)
            if not reject:
                still_rejecting = False  # Holm stops at the first non-rejection
        else:
            p_adj = float("nan")
            reject = False
            still_rejecting = False
        out[label] = {
            "p_raw": (float(p) if finite else float("nan")),
            "p_adj": p_adj,
            "reject": bool(reject),
            "rank": rank + 1,
        }
    return out


def pairwise_compare(
    group_a: GroupStats,
    group_b: GroupStats,
    n_bootstrap: int = 4000,
) -> Dict[str, object]:
    std_a = group_a.bootstrap_std_ci(n_bootstrap=n_bootstrap)
    std_b = group_b.bootstrap_std_ci(n_bootstrap=n_bootstrap)
    var_ratio = group_a.var / group_b.var if group_b.var > 0 else float("nan")
    variance_reduction_pct = (1.0 - var_ratio) * 100.0 if math.isfinite(var_ratio) else float("nan")
    pooled_std = math.sqrt(max(group_a.var + group_b.var, 0.0) / 2.0)
    paired = paired_seed_test(group_a, group_b)
    return {
        "a": group_a.label,
        "b": group_b.label,
        "welch": welch_t_test(group_a.deltas, group_b.deltas),
        "levene_mean": scale_test(group_a.deltas, group_b.deltas, center="mean"),
        "brown_forsythe": scale_test(group_a.deltas, group_b.deltas, center="median"),
        "f_var": f_var_ratio(group_a.deltas, group_b.deltas),
        "paired": paired,
        "var_ratio": var_ratio,
        "std_ratio": math.sqrt(var_ratio) if math.isfinite(var_ratio) and var_ratio >= 0 else float("nan"),
        "variance_reduction_pct": variance_reduction_pct,
        "std_ci_a": std_a,
        "std_ci_b": std_b,
        "mde_independent": minimum_detectable_effect(group_a.n_seeds, group_b.n_seeds, pooled_std),
        "mde_paired": paired_minimum_detectable_effect(
            std_diff=float(paired.get("std_diff", float("nan"))) if paired.get("valid") else float("nan"),
            n_pairs=int(paired.get("n_pairs", 0)),
        ),
    }


def conservative_pareto(groups: Sequence[GroupStats]) -> List[GroupStats]:
    candidates = [group for group in groups if group.n_seeds >= 2 and math.isfinite(group.wall_mean)]
    frontier: List[GroupStats] = []
    for group in candidates:
        dominated = False
        for other in candidates:
            if other is group:
                continue
            if (
                other.conservative >= group.conservative
                and other.wall_mean <= group.wall_mean
                and (other.conservative > group.conservative or other.wall_mean < group.wall_mean)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(group)
    return sorted(frontier, key=lambda group: group.wall_mean)


def comparisons_against_reference(
    groups: Sequence[GroupStats],
    reference_label: str = "H1_only",
    n_bootstrap: int = 4000,
) -> List[Dict[str, object]]:
    reference = next((group for group in groups if group.label == reference_label), None)
    if reference is None:
        return []
    rows: List[Dict[str, object]] = []
    for group in sorted(groups, key=lambda item: (-item.mean, item.label)):
        if group.label == reference_label:
            continue
        comparison = pairwise_compare(group, reference, n_bootstrap=n_bootstrap)
        rows.append({"group": group, "comparison": comparison})
    return rows



def print_summary(groups: Sequence[GroupStats], title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    print(
        f"{'config':<28} {'cont':<6} {'n_paths':>7} {'n':>3} "
        f"{'mean Δ%':>9} {'std':>7} {'SE':>7} {'95% CI':>22} {'std boot CI':>24} {'wall_s':>8}"
    )
    print("-" * 140)
    for group in sorted(groups, key=lambda item: (-item.mean, item.label)):
        ci_low, ci_high = group.ci95
        std_ci = group.bootstrap_std_ci(n_bootstrap=2000)
        if std_ci.get("valid"):
            std_ci_str = f"[{std_ci['ci_low']:+.3f},{std_ci['ci_high']:+.3f}]"
        else:
            std_ci_str = "n/a"
        print(
            f"{group.label:<28} {group.contract:<6} {group.n_paths:>7d} {group.n_seeds:>3d} "
            f"{group.mean:>+9.3f} {group.std:>7.3f} {group.se:>7.3f} "
            f"[{ci_low:+.3f},{ci_high:+.3f}] {std_ci_str:>24} {group.wall_mean:>8.0f}"
        )


def print_pairwise(groups: Sequence[GroupStats], reference_label: str, n_bootstrap: int) -> None:
    reference = next((group for group in groups if group.label == reference_label), None)
    if reference is None:
        print(f"\nReference '{reference_label}' not found.")
        return
    print(f"\n=== Pairwise vs {reference.label} (n={reference.n_seeds}, mean={reference.mean:+.3f} pp) ===")
    print(
        f"{'config':<24} {'gap':>8} {'p(mean)':>8} {'var red%':>9} {'p(BF)':>8} {'pairs':>5} {'p(pair)':>8} {'MDE ind':>8} {'MDE pair':>9}"
    )
    print("-" * 110)
    for row in comparisons_against_reference(groups, reference_label=reference_label, n_bootstrap=n_bootstrap):
        group = row["group"]
        comparison = row["comparison"]
        welch = comparison["welch"]
        bf = comparison["brown_forsythe"]
        paired = comparison["paired"]
        mde_ind = comparison["mde_independent"]
        mde_pair = comparison["mde_paired"]
        gap = welch.get("mean_diff", float("nan")) if welch.get("valid") else float("nan")
        p_mean = welch.get("p", float("nan")) if welch.get("valid") else float("nan")
        p_bf = bf.get("p", float("nan")) if bf.get("valid") else float("nan")
        pairs = int(paired.get("n_pairs", 0)) if paired else 0
        p_pair = paired.get("p", float("nan")) if paired.get("valid") else float("nan")
        mde_ind_val = mde_ind.get("mde", float("nan")) if mde_ind.get("valid") else float("nan")
        mde_pair_val = mde_pair.get("mde", float("nan")) if mde_pair.get("valid") else float("nan")
        print(
            f"{group.label:<24} {gap:>+8.3f} {p_mean:>8.3f} {comparison['variance_reduction_pct']:>+9.1f} {p_bf:>8.3f} "
            f"{pairs:>5d} {p_pair:>8.3f} {mde_ind_val:>8.3f} {mde_pair_val:>9.3f}"
        )
