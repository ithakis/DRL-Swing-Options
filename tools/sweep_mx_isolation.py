"""Isolate the M_x effect: confirm M_x is the only controlling axis.

The dense kernel-budget campaign suggested that the apparent "M=21 sweet
spot" was actually an M_x>=2 effect, with M_per_k and N_max contributing
little once M_x>=2. This sweep tests that hypothesis directly with a
controlled factorial design.

Phase A (M_x scan at fixed minimal M_per_k=1, N_max=1):
    M_x in {1, 2, 3, 4, 6} x seeds {11..18} -> 40 runs at 4096 paths.

Phase B (refinement at M_x=2: does M_per_k or N_max matter?):
    (M_per_k, N_max) in {(1,1), (1,2), (2,1), (2,2), (3,1), (4,2)}
    x seeds {11..16} -> 36 runs at 4096 paths.

Phase C (sanity check at 8192 paths, seeds {11..14}):
    A small subset of the winners at a higher budget -> 12 runs.

Usage:
    python tools/sweep_mx_isolation.py --max_workers 3
    python tools/sweep_mx_isolation.py --max_workers 3 --phases A B
    python tools/sweep_mx_isolation.py --dry_run
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, csv, sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sweep_expected_target import LOG_DIR, SweepConfig, base_args, run_one


KERNEL_BASE = {"--use_expected_target": "1", "--critic_warmup_episodes": "0"}


def overrides(mx: int, mpk: int, nmax: int) -> Dict[str, str]:
    return {
        **KERNEL_BASE,
        "--kernel_M_x": str(mx),
        "--kernel_M_per_k": str(mpk),
        "--kernel_N_max": str(nmax),
    }


def label(mx: int, mpk: int, nmax: int) -> str:
    m = mx * (1 + nmax * mpk)
    return f"Mx{mx}_pk{mpk}_n{nmax}_M{m}"


# Phase A: pure M_x axis (M_per_k=1, N_max=1 -> M = 2*M_x)
PHASE_A_GRID: List[Tuple[int, int, int]] = [
    (1, 1, 1),  # M=2 baseline of "too cheap"
    (2, 1, 1),  # M=4 cheapest plateau candidate
    (3, 1, 1),  # M=6
    (4, 1, 1),  # M=8
    (6, 1, 1),  # M=12
]
PHASE_A_SEEDS = list(range(11, 19))   # 8 seeds


# Phase B: refinement at M_x=2 (cheap), do M_per_k and N_max move anything?
PHASE_B_GRID: List[Tuple[int, int, int]] = [
    (2, 1, 1),  # M=4
    (2, 1, 2),  # M=6
    (2, 1, 3),  # M=8
    (2, 2, 1),  # M=6
    (2, 2, 2),  # M=10
    (2, 3, 1),  # M=8
    (2, 4, 2),  # M=18
]
PHASE_B_SEEDS = list(range(11, 17))   # 6 seeds


# Phase C: sanity check at 8192 paths
PHASE_C_GRID: List[Tuple[int, int, int]] = [
    (2, 1, 1),  # M=4 -- the conjectured cheapest plateau point
    (3, 3, 2),  # M=21 -- historic anchor
    (4, 4, 2),  # M=36 -- old default
]
PHASE_C_SEEDS = list(range(11, 15))   # 4 seeds


PHASES = {
    "A": (PHASE_A_GRID, PHASE_A_SEEDS, 4096),
    "B": (PHASE_B_GRID, PHASE_B_SEEDS, 4096),
    "C": (PHASE_C_GRID, PHASE_C_SEEDS, 8192),
}


def build_runs(active: Sequence[str]) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for phase in active:
        grid, seeds, n_paths = PHASES[phase]
        for (mx, mpk, nmax) in grid:
            for seed in seeds:
                lbl = f"{phase}_{label(mx, mpk, nmax)}"
                cfg = SweepConfig(label=lbl, overrides=overrides(mx, mpk, nmax),
                                  seed=seed, note=f"phase={phase} Mx={mx} Mpk={mpk} Nmax={nmax}")
                runs.append({"cfg": cfg, "n_paths": n_paths, "phase": phase,
                             "Mx": mx, "Mpk": mpk, "Nmax": nmax,
                             "M_total": mx * (1 + nmax * mpk)})
    return runs


CSV_PATH = LOG_DIR / "sweep_mx_isolation.csv"
FIELDS = ["tag", "phase", "label", "Mx", "Mpk", "Nmax", "M_total",
          "seed", "n_paths", "contract", "status",
          "eval_price", "eval_price_se", "lsm_price",
          "final_avg100", "final_paths_per_sec", "training_time",
          "wall_seconds", "note", "overrides"]


def row_key(row):
    return (row.get("phase",""), row.get("label",""),
            int(row.get("seed", "0") or 0),
            int(row.get("n_paths", "0") or 0))


def load_resume() -> List[Dict[str, str]]:
    if not CSV_PATH.exists():
        return []
    keep: Dict = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                keep[row_key(r)] = r
    return list(keep.values())


def write_csv(rows: List[Dict[str, str]]) -> None:
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_workers", type=int, default=3)
    p.add_argument("--phases", nargs="+", default=["A", "B", "C"],
                   choices=list(PHASES.keys()))
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    args = p.parse_args()

    runs = build_runs(args.phases)
    existing = load_resume() if args.resume else []
    done_keys = {row_key(r) for r in existing}
    pending = [r for r in runs if (r["phase"], r["cfg"].label, r["cfg"].seed, r["n_paths"]) not in done_keys]

    print(f"M_x isolation sweep: phases={args.phases} "
          f"total={len(runs)} done={len(existing)} pending={len(pending)} "
          f"max_workers={args.max_workers}")
    if args.dry_run:
        for r in pending:
            cfg = r["cfg"]
            print(f"[DRY] phase={r['phase']} {cfg.label:<24} seed={cfg.seed} "
                  f"M={r['M_total']} n_paths={r['n_paths']}")
        return
    if not pending:
        print("Nothing pending.")
        return

    rows: List[Dict[str, str]] = list(existing)

    def execute(run):
        cfg = run["cfg"]
        base = base_args(n_paths=int(run["n_paths"]), contract="focal")
        row = run_one(cfg, base, args.dry_run, tag_suffix="_mxiso")
        row.update({
            "tag": "_mxiso", "phase": run["phase"],
            "Mx": str(run["Mx"]), "Mpk": str(run["Mpk"]), "Nmax": str(run["Nmax"]),
            "M_total": str(run["M_total"]),
            "n_paths": str(run["n_paths"]), "contract": "focal",
        })
        return row

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(execute, r) for r in pending]
        for fut in cf.as_completed(futs):
            try:
                row = fut.result()
                rows.append(row)
                write_csv(rows)
            except Exception as e:
                print(f"FAIL: {e}", file=sys.stderr)

    print(f"\nResults -> {CSV_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
