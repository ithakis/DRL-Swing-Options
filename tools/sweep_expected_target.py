"""
Wide sweep over (kernel size, target_policy_noise, critic_warmup, PER alpha) for
the feat/semi-analytical-bootstrap H1 hypothesis.

Runs each configuration as a subprocess of run.py, captures stdout, parses
final eval price + running-mean trajectory + wall-clock, and emits a CSV
summary plus per-config full logs under logs/_sweep_h1/.

Designed for the M1 8 GB constraint: parallelism is configurable (default 2)
and each subprocess pins n_cores=1.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "_sweep_h1"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- config base ---------------------------------

CONTRACT_FOCAL = {"--c_cost": "0.04", "--gamma_cost": "2"}
CONTRACT_NOCOST = {"--c_cost": "0.0", "--gamma_cost": "1"}


def base_args(n_paths: int = 4096, contract: str = "focal") -> Dict[str, str]:
    """Common-base args matching the focal c=0.04, gamma=2 config but at a
    smaller training horizon.  All settings here are held fixed; only the
    swept fields override these.

    contract: 'focal' (c=0.04, gamma=2) or 'nocost' (c=0, gamma=1).
    """
    # Choose schedule horizons to fit within the n_paths budget.
    return {
        # training horizon
        "-n_paths": str(n_paths),
        "-n_paths_eval": "8192",
        "-eval_every": "-1",
        # algorithm core
        "-nstep": "1",
        "--gamma": "1",
        # PER (focal defaults)
        "-per": "1",
        "--per_alpha": "0.1",
        "--per_beta_start": "1.0",
        "--per_beta_frames": "100000",
        "--per_priority_floor": "5e-6",
        "--per_priority_clip_pct": "99.7",
        "--per_alpha_final": "0.20",
        "--per_alpha_ramp_start": "1024",
        "--per_alpha_ramp_end": "3000",
        "--per_beta_final": "0.98",
        # learn cadence
        "-learn_every": "2",
        "-learn_number": "1",
        # exploration noise
        "-noise_sigma0": "1.30",
        "-noise_floor": "0.26",
        "-noise_plateau": "512",
        # replay
        "--min_replay_size": "2048",
        "--max_replay_size": "100000",
        # network
        "-t": "0.0032",
        "-bs": "128",
        "-layer_size": "64",
        "--activation": "silu",
        "--norm": "layernorm",
        "--init_method": "orthogonal",
        # LR schedule
        "-lr_a": "1.6e-4",
        "-lr_c": "9.0e-5",
        "--final_lr_fraction": "0.20",
        "--warmup_episodes": "256",
        "--lr_schedule_episodes": "5000",
        "--min_lr": "1e-6",
        # grad clip
        "--actor_grad_clip": "1.0",
        "--critic_grad_clip": "2.5",
        "--actor_grad_clip_type": "norm",
        "--critic_grad_clip_type": "norm",
        "--grad_clip_norm_type": "2.0",
        # regularisation
        "--weight_decay_actor": "5e-5",
        "--weight_decay_critic": "1.2e-4",
        # target-policy smoothing (SWEPT BELOW)
        "--target_policy_noise": "0.15",
        "--target_policy_clip": "0.25",
        # misc
        "--compile": "0",
        "-n_cores": "1",
        "--disable_csv_logging": "1",
        "--limit_logging_frequency": "1",
        # warmup & noise schedule (SWEPT BELOW)
        "--critic_warmup_episodes": "256",
        "--adaptive_noise_scale": "0.6",
        "--actor_output_activation": "beta_sigmoid_3.0",
        "--warmup_noise_fraction": "0.4",
        "--target_noise_decay_start": "3500",
        "--target_noise_floor": "0.04",
        "--use_robust_normalization": "1",
        # contract (focal c=0.04, gamma=2)
        "--strike": "1.0",
        "--maturity": "0.0833",
        "--n_rights": "22",
        "--q_min": "0.0",
        "--q_max": "2.0",
        "--Q_min": "0.0",
        "--Q_max": "20.0",
        "--risk_free_rate": "0.05",
        "--min_refraction_periods": "0",
        # contract regime (overridden below)
        "--c_cost": "0.04",
        "--gamma_cost": "2",
        # LSM benchmark
        "--lsm_basis": "chebyshev",
        "--lsm_degree": "7",
        # HHK process
        "--S0": "1.0",
        "--alpha": "12.0",
        "--sigma": "1.2",
        "--beta": "150.0",
        "--lam": "6.0",
        "--mu_J": "0.3",
        # kernel defaults (only used when --use_expected_target=1)
        "--use_expected_target": "0",
        "--kernel_M_x": "4",
        "--kernel_M_per_k": "4",
        "--kernel_N_max": "2",
        "--kernel_chunk_M": "0",
    }


def _apply_contract(base: Dict[str, str], contract: str) -> Dict[str, str]:
    """Overlay contract-regime overrides onto the base dict."""
    if contract == "nocost":
        return {**base, **CONTRACT_NOCOST}
    return base


@dataclass
class SweepConfig:
    label: str               # short tag (used in run name)
    overrides: Dict[str, str] = field(default_factory=dict)
    seed: int = 11
    note: str = ""


def make_configs() -> List[SweepConfig]:
    """The 12-config wide sweep.  Each varies only a handful of fields off the
    common base so attribution is clean."""
    cfgs: List[SweepConfig] = []

    # --- Baselines (kernel off, v61-style) -----------------------------------
    cfgs.append(SweepConfig(
        label="B0_baseline",
        overrides={"--use_expected_target": "0"},
        note="v61-style baseline (kernel off, target_noise=0.15, warmup=256)",
    ))
    cfgs.append(SweepConfig(
        label="B1_no_target_noise",
        overrides={"--use_expected_target": "0", "--target_policy_noise": "0.00"},
        note="baseline w/ no target-policy noise smoothing",
    ))

    # --- Kernel default (M=36) sweep over interactions -----------------------
    K = {"--use_expected_target": "1",
         "--kernel_M_x": "4", "--kernel_M_per_k": "4", "--kernel_N_max": "2"}
    cfgs.append(SweepConfig(label="K36_default", overrides=dict(K),
                            note="kernel M=36, default smoothing/warmup"))
    cfgs.append(SweepConfig(
        label="K36_no_noise",
        overrides={**K, "--target_policy_noise": "0.00"},
        note="kernel M=36 + no target-policy noise (kernel already smooths)",
    ))
    cfgs.append(SweepConfig(
        label="K36_low_noise",
        overrides={**K, "--target_policy_noise": "0.05"},
        note="kernel M=36 + low target-policy noise",
    ))
    cfgs.append(SweepConfig(
        label="K36_no_warmup",
        overrides={**K, "--critic_warmup_episodes": "0"},
        note="kernel M=36 + zero critic warmup (test if kernel stabilises faster)",
    ))
    cfgs.append(SweepConfig(
        label="K36_long_warmup",
        overrides={**K, "--critic_warmup_episodes": "1024"},
        note="kernel M=36 + 1024 ep warmup (focal default)",
    ))
    cfgs.append(SweepConfig(
        label="K36_per_high",
        overrides={**K, "--per_alpha": "0.3", "--per_alpha_final": "0.4"},
        note="kernel M=36 + higher PER alpha (sharper prioritisation)",
    ))
    cfgs.append(SweepConfig(
        label="K36_per_low",
        overrides={**K, "--per_alpha": "0.05", "--per_alpha_final": "0.10"},
        note="kernel M=36 + lower PER alpha (more uniform replay)",
    ))

    # --- Larger kernels (accuracy vs cost) -----------------------------------
    cfgs.append(SweepConfig(
        label="K78",
        overrides={"--use_expected_target": "1",
                   "--kernel_M_x": "6", "--kernel_M_per_k": "4", "--kernel_N_max": "3"},
        note="kernel M=78 (M_x=6, M_per_k=4, N_max=3)",
    ))
    cfgs.append(SweepConfig(
        label="K150",
        overrides={"--use_expected_target": "1",
                   "--kernel_M_x": "6", "--kernel_M_per_k": "8", "--kernel_N_max": "3"},
        note="kernel M=150 (default plan size; ~3x cost vs K36)",
    ))

    # --- K36 + smaller batch (test if low-variance target lets us use bs=64) -
    cfgs.append(SweepConfig(
        label="K36_bs64",
        overrides={**K, "-bs": "64"},
        note="kernel M=36 + batch_size=64 (test if low-var target needs less smoothing via batch averaging)",
    ))

    return cfgs


# ----------------------------- runner --------------------------------------


PARSE_PRICE = re.compile(r"Option Price:\s*([\d.\-eE]+)\s*\xb1\s*([\d.\-eE]+)")
PARSE_LSM = re.compile(r"LSM Benchmark Price:\s*([\d.\-eE]+)")
PARSE_LAST_AVG100 = re.compile(r"Average100\s*=\s*([\d.\-eE]+)")
PARSE_PATHS_SEC = re.compile(r"Paths/sec\s*=\s*([\d.\-eE]+)")
PARSE_TIME = re.compile(r"Training Time:\s*(\d{2}:\d{2}:\d{2}\.\d+)")


def parse_stdout(text: str) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {
        "eval_price": None,
        "eval_price_se": None,
        "lsm_price": None,
        "final_avg100": None,
        "final_paths_per_sec": None,
        "training_time": None,
    }
    m = PARSE_PRICE.search(text)
    if m:
        out["eval_price"], out["eval_price_se"] = m.group(1), m.group(2)
    m = PARSE_LSM.search(text)
    if m:
        out["lsm_price"] = m.group(1)
    avgs = PARSE_LAST_AVG100.findall(text)
    if avgs:
        out["final_avg100"] = avgs[-1]
    pps = PARSE_PATHS_SEC.findall(text)
    if pps:
        out["final_paths_per_sec"] = pps[-1]
    m = PARSE_TIME.search(text)
    if m:
        out["training_time"] = m.group(1)
    return out


def run_one(cfg: SweepConfig, base: Dict[str, str], dry_run: bool,
            tag_suffix: str = "") -> Dict[str, str]:
    args = dict(base)
    args.update(cfg.overrides)
    run_name = f"_sw_h1{tag_suffix}_{cfg.label}_s{cfg.seed}"
    args["-name"] = run_name
    args["-seed"] = str(cfg.seed)

    cmd = ["python", "run.py"]
    for k, v in args.items():
        if k.startswith("--"):
            cmd.extend([k, v])
        else:
            cmd.extend([k, v])

    log_path = LOG_DIR / f"{cfg.label}{tag_suffix}_s{cfg.seed}.log"

    if dry_run:
        print(f"[DRY] {cfg.label}: {' '.join(shlex.quote(c) for c in cmd)}")
        return {"label": cfg.label, "status": "dry"}

    print(f"[{cfg.label}] launching (log -> {log_path})")
    t0 = time.time()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    wall = time.time() - t0

    text = log_path.read_text(errors="replace")
    parsed = parse_stdout(text)
    row = {
        "label": cfg.label,
        "seed": str(cfg.seed),
        "status": "ok" if result.returncode == 0 else f"exit_{result.returncode}",
        "wall_seconds": f"{wall:.1f}",
        **{k: (v if v is not None else "") for k, v in parsed.items()},
        "note": cfg.note,
        "overrides": json.dumps(cfg.overrides),
    }
    print(f"[{cfg.label}] done in {wall:.0f}s | price={parsed.get('eval_price')} "
          f"avg100={parsed.get('final_avg100')} status={row['status']}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_paths", type=int, default=4096,
                        help="Training episodes per config (default 4096).")
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Parallel subprocess cap (default 2 for M1 8 GB).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11],
                        help="Seeds to run each config with.")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated label whitelist (e.g. B0_baseline,K36_default).")
    parser.add_argument("--contract", type=str, choices=["focal", "nocost"], default="focal",
                        help="Which contract regime to use ('focal' = c=0.04, gamma=2; "
                             "'nocost' = c=0, gamma=1).")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix appended to CSV/log filenames (e.g. '_nocost').")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base = base_args(n_paths=args.n_paths)
    base = _apply_contract(base, args.contract)
    cfgs = make_configs()
    if args.only:
        wanted = set(args.only.split(","))
        cfgs = [c for c in cfgs if c.label in wanted]

    # Multi-seed expansion
    all_runs: List[SweepConfig] = []
    for c in cfgs:
        for s in args.seeds:
            cc = SweepConfig(label=c.label, overrides=dict(c.overrides),
                              seed=s, note=c.note)
            all_runs.append(cc)

    print(f"Running {len(all_runs)} runs with max_workers={args.max_workers} "
          f"({len(cfgs)} configs x {len(args.seeds)} seeds, n_paths={args.n_paths})")

    csv_path = LOG_DIR / f"sweep_results_n{args.n_paths}{args.output_suffix}.csv"
    fieldnames = ["label", "seed", "status", "eval_price", "eval_price_se",
                  "lsm_price", "final_avg100", "final_paths_per_sec",
                  "training_time", "wall_seconds", "note", "overrides"]
    rows: List[Dict[str, str]] = []
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(run_one, r, base, args.dry_run, args.output_suffix) for r in all_runs]
        for fut in cf.as_completed(futs):
            try:
                row = fut.result()
                if "status" in row and row["status"] != "dry":
                    rows.append(row)
                    # write CSV incrementally so we don't lose progress on crash
                    with open(csv_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        for r in rows:
                            w.writerow({k: r.get(k, "") for k in fieldnames})
            except Exception as e:
                print(f"FUTURE FAILED: {e}", file=sys.stderr)

    print(f"\nResults written to {csv_path} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
