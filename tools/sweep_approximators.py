"""Approximator comparison sweep: poly / rff / rbf / tiny_nn vs the NN baseline.

All runs use the semi-analytical kernel (the going-forward basis). Three stages:

  Stage A (tune)    -- per-estimator hyperparameter grid, 4 regimes x 4 seeds,
                       4096 ep, FAST kernel. Pick the single most-robust config
                       per estimator (best mean Delta% at acceptable seed-std).
  Stage B (screen)  -- one winner config per estimator + the NN baseline,
                       4 regimes x 24 seeds, 4096 ep, FAST kernel, 4 workers.
                       High-power head-to-head.
  Stage C (final)   -- finalists, 4 regimes x 6 seeds x {8192, 32768} ep,
                       ACCURATE kernel (M_x=4, M_per_k=4, N_max=2), 3 workers.

The 4 contract regimes (HHK process identical across all):
  g1   : c=0.04, gamma=1
  g15  : c=0.04, gamma=1.5
  g2   : c=0.04, gamma=2   (focal)
  nocost: c=0.0, gamma=1

CSV schema is consumed directly by tools/stats_analysis.py (label, seed,
n_paths, contract, eval_price, lsm_price, wall_seconds, status, ...).

Usage:
  python tools/sweep_approximators.py --stage A --max_workers 4 --dry_run
  python tools/sweep_approximators.py --stage A --max_workers 4
  python tools/sweep_approximators.py --stage B --max_workers 4
  python tools/sweep_approximators.py --stage C --max_workers 3
  python tools/sweep_approximators.py --stage B --only nn,poly --regimes g2 nocost
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sweep_expected_target import SweepConfig, base_args, run_one  # noqa: E402

LOG_DIR = ROOT / "logs" / "_sweep_approx"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- contract regimes ----------------------------

REGIMES: Dict[str, Dict[str, str]] = {
    "g1":     {"--c_cost": "0.04", "--gamma_cost": "1"},
    "g15":    {"--c_cost": "0.04", "--gamma_cost": "1.5"},
    "g2":     {"--c_cost": "0.04", "--gamma_cost": "2"},
    "nocost": {"--c_cost": "0.0",  "--gamma_cost": "1"},
}

# ----------------------------- kernel presets -------------------------------

FAST_KERNEL = {
    "--use_expected_target": "1", "--critic_warmup_episodes": "0",
    "--kernel_M_x": "2", "--kernel_M_per_k": "1", "--kernel_N_max": "1",
}
ACCURATE_KERNEL = {
    "--use_expected_target": "1", "--critic_warmup_episodes": "0",
    "--kernel_M_x": "4", "--kernel_M_per_k": "4", "--kernel_N_max": "2",
}

# ----------------------------- approximator knobs ---------------------------


def approx_overrides(kind: str, **knobs) -> Dict[str, str]:
    """Build run.py overrides for one approximator config."""
    ov: Dict[str, str] = {"--approximator": kind}
    mapping = {
        "degree": "--poly_degree",
        "rff_dim": "--rff_dim",
        "rff_lengthscale": "--rff_lengthscale",
        "rff_learnable": "--rff_learnable",
        "rbf_centers": "--rbf_centers",
        "rbf_bandwidth": "--rbf_bandwidth",
        "rbf_learnable_bandwidth": "--rbf_learnable_bandwidth",
        "tiny_width": "--tiny_width",
        "tiny_activation": "--tiny_activation",
        "feature_use_cross": "--feature_use_cross",
    }
    for k, v in knobs.items():
        ov[mapping[k]] = str(v)
    return ov


def _grid_label(kind: str, **knobs) -> str:
    parts = [kind] + [f"{k}{v}" for k, v in knobs.items()]
    return "_".join(str(p) for p in parts)


# Stage A tuning grids (one entry -> one config). Kept modest.
def stage_a_configs() -> List[Dict[str, object]]:
    grid: List[Dict[str, object]] = []

    def add(kind, **knobs):
        grid.append({"kind": kind, "label": _grid_label(kind, **knobs),
                     "overrides": approx_overrides(kind, **knobs)})

    for degree in (2, 3, 4):
        add("poly", degree=degree)
    for dim in (128, 256):
        add("rff", rff_dim=dim, rff_lengthscale="1.0")
    add("rff", rff_dim=256, rff_lengthscale="0.5")
    add("rff", rff_dim=256, rff_lengthscale="2.0")
    for nc in (64, 128):
        add("rbf", rbf_centers=nc, rbf_bandwidth="1.0")
    add("rbf", rbf_centers=128, rbf_bandwidth="0.75")
    add("rbf", rbf_centers=128, rbf_bandwidth="1.5")
    for width in (24, 32):
        add("tiny_nn", tiny_width=width, tiny_activation="silu")
    add("tiny_nn", tiny_width=32, tiny_activation="tanh")
    return grid


# Stage B/C winner configs (edit after Stage A picks robust winners).
STAGE_B_CONFIGS: List[Dict[str, object]] = [
    # Paper version (v61): NN + single-sample TD (kernel OFF). Reference bar to beat.
    {"kind": "nn",      "label": "nn_paper", "overrides": {"--approximator": "nn", "--use_expected_target": "0"}},
    # Analytical-reward version: NN + semi-analytical kernel ON (the going-forward basis).
    {"kind": "nn",      "label": "nn",      "overrides": {"--approximator": "nn"}},
    {"kind": "poly",    "label": "poly",    "overrides": approx_overrides("poly", degree=3)},
    {"kind": "rff",     "label": "rff",     "overrides": approx_overrides("rff", rff_dim=256, rff_lengthscale="1.0")},
    {"kind": "rbf",     "label": "rbf",     "overrides": approx_overrides("rbf", rbf_centers=128, rbf_bandwidth="1.0")},
    {"kind": "tiny_nn", "label": "tiny_nn", "overrides": approx_overrides("tiny_nn", tiny_width=32, tiny_activation="silu")},
]

# Finalists for Stage C (edit after Stage B). Default: NN + the two best basis nets.
STAGE_C_LABELS = ["nn", "poly", "tiny_nn"]


# Stage D — head-LR diagnostic. Stage A showed every approximator losing to LSM
# and the NN at 4096 ep with the v61 LRs (lr_a=1.6e-4, lr_c=9e-5), which were
# tuned for the deep 2x64 net. The fixed-feature heads (poly/rff/rbf) are linear
# regression and likely undertrained at that LR. This sweeps a head-LR multiplier
# on the best-per-family Stage A config + the NN anchor, on the focal regime.
_LR_A0, _LR_C0 = 1.6e-4, 9.0e-5


def _with_lr(ov: Dict[str, str], mult: float) -> Dict[str, str]:
    return {**ov, "-lr_a": f"{_LR_A0 * mult:.4e}", "-lr_c": f"{_LR_C0 * mult:.4e}"}


STAGE_D_CONFIGS: List[Dict[str, object]] = [
    {"kind": "nn",      "label": "nn_lr1x",   "overrides": {"--approximator": "nn"}},
    {"kind": "poly",    "label": "poly_lr3x",  "overrides": _with_lr(approx_overrides("poly", degree=2), 3)},
    {"kind": "poly",    "label": "poly_lr10x", "overrides": _with_lr(approx_overrides("poly", degree=2), 10)},
    {"kind": "rff",     "label": "rff_lr3x",   "overrides": _with_lr(approx_overrides("rff", rff_dim=256, rff_lengthscale="2.0"), 3)},
    {"kind": "rff",     "label": "rff_lr10x",  "overrides": _with_lr(approx_overrides("rff", rff_dim=256, rff_lengthscale="2.0"), 10)},
    {"kind": "rbf",     "label": "rbf_lr10x",  "overrides": _with_lr(approx_overrides("rbf", rbf_centers=128, rbf_bandwidth="0.75"), 10)},
    {"kind": "tiny_nn", "label": "tiny_lr10x", "overrides": _with_lr(approx_overrides("tiny_nn", tiny_width=32, tiny_activation="silu"), 10)},
]

STAGE_DEFAULTS = {
    "A": {"seeds": list(range(11, 15)), "n_paths": [4096], "kernel": FAST_KERNEL, "workers": 4},
    "B": {"seeds": list(range(11, 35)), "n_paths": [4096], "kernel": FAST_KERNEL, "workers": 4},
    "C": {"seeds": list(range(11, 17)), "n_paths": [8192, 32768], "kernel": ACCURATE_KERNEL, "workers": 3},
    "D": {"seeds": list(range(11, 15)), "n_paths": [4096], "kernel": FAST_KERNEL, "workers": 4},
}

# ----------------------------- CSV / resume ---------------------------------

CSV_FIELDS = ["stage", "approximator", "label", "regime", "contract", "seed",
              "n_paths", "status", "eval_price", "eval_price_se", "lsm_price",
              "final_avg100", "final_paths_per_sec", "training_time",
              "wall_seconds", "note", "overrides"]


def csv_path(stage: str) -> Path:
    return LOG_DIR / f"sweep_approx_stage{stage}.csv"


def row_key(r: Dict[str, str]):
    return (r.get("label", ""), r.get("regime", ""),
            int(r.get("seed", "0") or 0), int(r.get("n_paths", "0") or 0))


def load_resume(stage: str) -> List[Dict[str, str]]:
    path = csv_path(stage)
    if not path.exists():
        return []
    keep: Dict = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                keep[row_key(r)] = r
    return list(keep.values())


def write_csv(stage: str, rows: List[Dict[str, str]]) -> None:
    with open(csv_path(stage), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


# ----------------------------- run planning ---------------------------------


def configs_for_stage(stage: str) -> List[Dict[str, object]]:
    if stage == "A":
        return stage_a_configs()
    if stage == "B":
        return STAGE_B_CONFIGS
    if stage == "C":
        return [c for c in STAGE_B_CONFIGS if c["label"] in STAGE_C_LABELS]
    if stage == "D":
        return STAGE_D_CONFIGS
    raise ValueError(stage)


def build_runs(stage: str, regimes: Sequence[str], seeds: Sequence[int],
               n_paths_list: Sequence[int], kernel: Dict[str, str],
               only: Sequence[str] | None) -> List[Dict[str, object]]:
    cfgs = configs_for_stage(stage)
    if only:
        cfgs = [c for c in cfgs if c["label"] in set(only)]
    runs: List[Dict[str, object]] = []
    for cfg in cfgs:
        for regime in regimes:
            for n_paths in n_paths_list:
                for seed in seeds:
                    overrides = {**kernel, **REGIMES[regime], **cfg["overrides"]}
                    label = f"{cfg['label']}_{regime}"
                    runs.append({
                        "kind": cfg["kind"], "label": cfg["label"], "regime": regime,
                        "seed": seed, "n_paths": n_paths, "overrides": overrides,
                        "sweep_cfg": SweepConfig(label=label, overrides=overrides,
                                                 seed=seed, note=f"stage={stage} {cfg['label']} {regime}"),
                    })
    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["A", "B", "C", "D"], required=True)
    p.add_argument("--max_workers", type=int, default=None,
                   help="Parallel subprocess cap (default per-stage: A/B=4, C=3).")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Override the per-stage seed set.")
    p.add_argument("--n_paths", type=int, nargs="+", default=None,
                   help="Override the per-stage training horizons.")
    p.add_argument("--regimes", nargs="+", default=list(REGIMES.keys()),
                   choices=list(REGIMES.keys()))
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated config-label whitelist (e.g. nn,poly).")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    args = p.parse_args()

    d = STAGE_DEFAULTS[args.stage]
    seeds = args.seeds or d["seeds"]
    n_paths_list = args.n_paths or d["n_paths"]
    kernel = d["kernel"]
    workers = args.max_workers or d["workers"]
    only = args.only.split(",") if args.only else None

    runs = build_runs(args.stage, args.regimes, seeds, n_paths_list, kernel, only)
    existing = load_resume(args.stage) if args.resume else []
    done = {row_key(r) for r in existing}
    pending = [r for r in runs
               if (r["label"] + "_" + r["regime"], r["regime"], r["seed"], r["n_paths"]) not in done
               and (r["label"], r["regime"], r["seed"], r["n_paths"]) not in done]

    print(f"Stage {args.stage}: regimes={args.regimes} seeds={seeds} "
          f"n_paths={n_paths_list} | total={len(runs)} done={len(existing)} "
          f"pending={len(pending)} workers={workers}")
    if args.dry_run:
        for r in pending:
            print(f"[DRY] {r['label']:<14} regime={r['regime']:<6} seed={r['seed']} "
                  f"n_paths={r['n_paths']}  overrides={r['overrides']}")
        return
    if not pending:
        print("Nothing pending.")
        return

    rows: List[Dict[str, str]] = list(existing)

    def execute(run: Dict[str, object]) -> Dict[str, str]:
        base = base_args(n_paths=int(run["n_paths"]))
        row = run_one(run["sweep_cfg"], base, dry_run=False, tag_suffix=f"_approx{args.stage}")
        row.update({
            "stage": args.stage,
            "approximator": str(run["kind"]),
            "label": str(run["label"]),
            "regime": str(run["regime"]),
            "contract": str(run["regime"]),     # stats_analysis groups on 'contract'
            "n_paths": str(run["n_paths"]),
        })
        return row

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(execute, r) for r in pending]
        for fut in cf.as_completed(futs):
            try:
                row = fut.result()
                rows.append(row)
                write_csv(args.stage, rows)
            except Exception as e:  # noqa: BLE001
                print(f"FAIL: {e}", file=sys.stderr)

    print(f"\nResults -> {csv_path(args.stage)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
