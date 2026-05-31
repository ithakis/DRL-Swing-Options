#!/usr/bin/env python
"""v63 Runtime Feature Audit sweep (E1/E3/E4/E5).

Screens the EDIT-test candidates from the v63 runtime audit (HPT.md → "v63 Runtime
Feature Audit") against the v63-canonical anchor, one feature change at a time.

Methodology (accuracy-first; see HPT.md):
  * Run each arm at 3 seeds {11,12,13} FIRST.
  * Adopt a change only if it is >= anchor on Δ% mean AND not worse on std
    (Welch p>=0.05 on the mean, Levene p>=0.05 on the spread).
  * If a removal looks computationally worth it but is statistically ambiguous at
    3 seeds, escalate THAT arm to 12 seeds (`--seeds 11 12 ... 22`).
  * A change that needs retuning to break even is only "safe" once the *retuned*
    variant wins — never adopt a bare removal on "not obviously worse".

The candidates:
  E1  critic_warmup 1024 -> 512            (cc_g1 only; the regime that needs warmup)
  E3  tau 0.0032 -> 0.005                  (cc_g2, nocost)
  E4  drop double critic.step()            (cc_g2, nocost) — two arms:
        E4_drop_lrc3e4 : --single_critic_step 1 (un-retuned; shows the halving)
        E4_drop_lrc6e4 : --single_critic_step 1 -lr_c 6e-4 (principled compensator)
  E5  learn_every 2 -> 1                   (cc_g1, nocost; costs ~2x wall-clock)

Each regime also runs its `anchor` so the comparison is within-regime.

Usage (env: conda activate EP11):
  python tools/sweep_v63_audit.py --dry_run                 # print commands only
  python tools/sweep_v63_audit.py                           # 3 seeds, all arms/regimes
  python tools/sweep_v63_audit.py --arms E4 --seeds 11 12 13 14 15 16 17 18 19 20 21 22
  python tools/sweep_v63_audit.py --analyze_only            # re-run stats on existing CSV

Writes logs/_sweep_v63_audit/sweep_v63_audit.csv in the tools/stats_analysis.py schema
(columns: label, seed, status, eval_price, lsm_price, wall_seconds, contract, n_paths, ...).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "_sweep_v63_audit"
LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))


# --------------------------------------------------------------------------- #
# v63-canonical base config (focal SwingOption_20 contract + HHK process).
# NOTE: this is a CLEAN v63 harness — it does NOT pass the args v63 removed
# (-per/--per_*, --target_policy_noise/_clip, grad clips). It mirrors the
# contract/process block of tools/sweep_expected_target.py:base_args.
# --------------------------------------------------------------------------- #
def base_args(n_paths: int) -> Dict[str, str]:
    return {
        # training horizon
        "-n_paths": str(n_paths),
        "-n_paths_eval": "8192",
        "-eval_every": "-1",
        # algorithm core
        "-nstep": "1",
        "--gamma": "1",
        # kernel ON, fast option (M_x=2 is the validated controlling axis)
        "--use_expected_target": "1",
        "--kernel_M_x": "2",
        "--kernel_M_per_k": "1",
        "--kernel_N_max": "1",
        # learn cadence
        "-learn_every": "2",
        "-learn_number": "1",
        # exploration noise — Task 2 adopted: linear sigma0->floor decay over the FULL
        # horizon (noise_schedule=linear, plateau=0). At 4k the old plateau=3200 wasted
        # the decay; reducing exploration lifted focal cc_g2 +0.34pp (Welch p=0.002).
        "-noise_sigma0": "1.30",
        "-noise_floor": "0.26",
        "-noise_plateau": "0",
        "--noise_schedule": "linear",
        "--adaptive_noise_scale": "0.6",
        "--actor_output_activation": "beta_sigmoid_3.0",
        # Task 2 adopted: eval-only EMA of actor weights (Schedule-Free spirit) — tightens
        # seed dispersion and stacks with the noise decay (combined arm N_full_ema).
        "--weight_averaging": "ema",
        "--ema_decay": "0.999",
        # replay (uniform; PER removed in v63)
        "--min_replay_size": "1000",
        "--max_replay_size": "100000",
        # network
        "-t": "0.0032",
        "-bs": "128",
        "-layer_size": "64",
        "--activation": "silu",
        "--norm": "layernorm",
        "--init_method": "orthogonal",
        # LR: CONSTANT (LR decay removed in Task 2; see HPT). Task 3 adopted lr_c=6e-4
        # (2x the inherited 3e-4) — the deterministic single-step target wants a faster
        # critic: +worst-case & tighter std in all 3 regimes, +mean on cc_g1/nocost.
        # warmup_episodes here only sizes the calibrate_bias dataset.
        "-lr_a": "3e-4",
        "-lr_c": "6e-4",
        "--final_lr_fraction": "1.0",
        "--warmup_episodes": "1024",
        "--lr_schedule_episodes": "65000",
        "--min_lr": "1e-7",
        # regularisation
        "--weight_decay_actor": "5e-5",
        "--weight_decay_critic": "1e-4",
        # critic warmup (v63 audit RESOLVED: 512 — structurally required for g1,
        # but 1024 was over-conservative; 512 has 0/12 blowups + best mean)
        "--critic_warmup_episodes": "512",
        # v63 audit E4 RESOLVED: single critic step is now canonical (was double-step bug)
        "--single_critic_step": "1",
        # misc
        "--compile": "0",
        "-n_cores": "1",
        "--disable_csv_logging": "1",
        "--limit_logging_frequency": "1",
        # contract (focal SwingOption_20)
        "--strike": "1.0",
        "--maturity": "0.0833",
        "--n_rights": "22",
        "--q_min": "0.0",
        "--q_max": "2.0",
        "--Q_min": "0.0",
        "--Q_max": "20.0",
        "--risk_free_rate": "0.05",
        "--min_refraction_periods": "0",
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
    }


# Regime = contract + regime-specific knobs (robust-norm + warmup-noise fraction).
REGIMES: Dict[str, Dict[str, str]] = {
    "cc_g1": {"--c_cost": "0.04", "--gamma_cost": "1",
              "--use_robust_normalization": "1", "--warmup_noise_fraction": "0.3"},
    "cc_g2": {"--c_cost": "0.04", "--gamma_cost": "2",
              "--use_robust_normalization": "1", "--warmup_noise_fraction": "0.3"},
    "nocost": {"--c_cost": "0.0", "--gamma_cost": "1",
               "--use_robust_normalization": "0", "--warmup_noise_fraction": "0.4"},
}


@dataclass
class Arm:
    arm_id: str                       # E1 / E3 / E4 / E5 (for --arms filtering)
    label: str                        # unique config label (grouping key for stats)
    regimes: List[str]                # which regimes this arm runs in
    overrides: Dict[str, str] = field(default_factory=dict)
    note: str = ""


# Anchor runs in every regime; each EDIT-test arm runs only where it is meaningful.
ARMS: List[Arm] = [
    Arm("anchor", "anchor", ["cc_g1", "cc_g2", "nocost"], {},
        "v63 canonical (constant LR, single critic-step, warmup=512, tau=0.0032)"),
    # E1 — shorter critic warmup, g1 only (the regime that needs it)
    Arm("E1", "E1_warmup512", ["cc_g1"], {"--critic_warmup_episodes": "512"},
        "critic_warmup 1024->512; pass only if NO g1 seed blows up"),
    Arm("E1", "E1_warmup256", ["cc_g1"], {"--critic_warmup_episodes": "256"},
        "critic_warmup 1024->256 (more aggressive); pass only if NO g1 seed blows up"),
    # E3 — faster target tracking
    Arm("E3", "E3_tau005", ["cc_g2", "nocost"], {"-t": "0.005"},
        "tau 0.0032->0.005"),
    # E4 — drop the double critic step (validate on ALL regimes incl. fragile cc_g1)
    Arm("E4", "E4_drop_lrc3e4", ["cc_g1", "cc_g2", "nocost"], {"--single_critic_step": "1"},
        "drop double-step, lr_c UNCHANGED (3e-4) -> simplest merge (must be non-inferior)"),
    Arm("E4", "E4_drop_lrc6e4", ["cc_g1", "cc_g2", "nocost"],
        {"--single_critic_step": "1", "-lr_c": "6e-4"},
        "drop double-step + lr_c 3e-4->6e-4 (retuned compensator)"),
    # E5 — denser updates (costs ~2x wall-clock)
    Arm("E5", "E5_learn1", ["cc_g1", "nocost"], {"-learn_every": "1"},
        "learn_every 2->1; adopt only if Δ%/std win justifies ~2x time"),

    # ------------------------------------------------------------------ #
    # W — critic_warmup investigation (runs under the NEW single-step default).
    # Q1 needed?  Q2 correctly tuned?  Q3 can lr_a retune make it obsolete?
    # cc_g1 is the binding regime (collapsed at warmup=0 with the OLD double-step);
    # nocost is the safety check (survived warmup=0 before). Goal: accuracy+speed+consistency.
    # ------------------------------------------------------------------ #
    # Q1/Q2 — pure warmup level (all regimes)
    Arm("W", "W_warmup0",   ["cc_g1", "cc_g2", "nocost"], {"--critic_warmup_episodes": "0"},
        "NO critic warmup — does single-step already remove the g1 collapse?"),
    Arm("W", "W_warmup256", ["cc_g1", "cc_g2", "nocost"], {"--critic_warmup_episodes": "256"},
        "warmup 1024->256"),
    Arm("W", "W_warmup512", ["cc_g1", "cc_g2", "nocost"], {"--critic_warmup_episodes": "512"},
        "warmup 1024->512"),
    # Q3 — substitute warmup with a gentler actor cold-start (cc_g1 binding + nocost check)
    Arm("W", "W_w0_lra2e4",   ["cc_g1", "nocost"],
        {"--critic_warmup_episodes": "0", "-lr_a": "2e-4"},
        "warmup=0 + lr_a 3e-4->2e-4: can a gentler actor remove the need for warmup?"),
    Arm("W", "W_w0_lra1p5e4", ["cc_g1", "nocost"],
        {"--critic_warmup_episodes": "0", "-lr_a": "1.5e-4"},
        "warmup=0 + lr_a 3e-4->1.5e-4 (gentler still)"),
    Arm("W", "W_w256_lra2e4", ["cc_g1", "nocost"],
        {"--critic_warmup_episodes": "256", "-lr_a": "2e-4"},
        "short warmup (256) + gentle actor (lr_a=2e-4) combined"),

    # ------------------------------------------------------------------ #
    # CB — Task 1: closed-form vs Rprop actor warm-start (calibrate_bias).
    # Anchor uses the new default (closed_form). CB_rprop restores the legacy
    # 20-iter Rprop loop. Goal: closed_form (anchor) >= rprop on Δ%/std/worst,
    # at much lower startup wall-time (parsed from "Calibration wall-time:").
    # ------------------------------------------------------------------ #
    Arm("CB", "CB_rprop", ["cc_g1", "cc_g2", "nocost"], {"--calibrate_bias_mode": "rprop"},
        "legacy Rprop warm-start (anchor=closed_form must be >= this)"),

    # ------------------------------------------------------------------ #
    # Task 2 Screen A — exploration noise (vs the hyperbolic-schedule anchor).
    # All 3 regimes. Floor scaled to keep floor/sigma0 ≈ 0.2 (canonical ratio).
    # ------------------------------------------------------------------ #
    Arm("N", "N_const", ["cc_g1", "cc_g2", "nocost"],
        {"--noise_schedule": "const_floor"},
        "constant sigma0 during plateau, hard step to floor after (no hyperbolic tail)"),
    Arm("N", "N_fulldecay", ["cc_g1", "cc_g2", "nocost"],
        {"--noise_schedule": "linear", "-noise_plateau": "0"},
        "linear sigma0->floor over the full 4096-ep horizon (no plateau)"),
    Arm("N", "N_lowsigma08", ["cc_g1", "cc_g2", "nocost"],
        {"-noise_sigma0": "0.8", "-noise_floor": "0.16"},
        "reduced exploration sigma0=0.8 (floor scaled 0.16)"),
    Arm("N", "N_lowsigma05", ["cc_g1", "cc_g2", "nocost"],
        {"-noise_sigma0": "0.5", "-noise_floor": "0.10"},
        "reduced exploration sigma0=0.5 (floor scaled 0.10)"),

    # ------------------------------------------------------------------ #
    # Task 2 Screen B — LR schedule / update rule (vs the constant-LR anchor).
    # Only L_ema survives: constant LR + eval-only EMA. The LR-decay arms were
    # screened, found neutral-to-harmful at 4k, and removed with their flags.
    # ------------------------------------------------------------------ #
    Arm("L", "L_ema", ["cc_g1", "cc_g2", "nocost"],
        {"--weight_averaging": "ema", "--ema_decay": "0.999"},
        "constant LR + eval-only EMA of actor weights (Schedule-Free spirit)"),
    # NOTE: the L_lindecay / L_cosine4k_* arms were removed after Task 2 Screen B found
    # LR decay neutral-to-harmful at 4k and the cosine/linear lr_lambda branches were
    # deleted (constant LR is canonical). See HPT "Task 2 — scheduler simplification".

    # ------------------------------------------------------------------ #
    # C — Task 2 combined winner: N_fulldecay (linear noise) + L_ema (eval-only EMA).
    # The "combine winners" step run AFTER the individual screens (methodology).
    # ------------------------------------------------------------------ #
    Arm("C", "N_full_ema", ["cc_g1", "cc_g2", "nocost"],
        {"--noise_schedule": "linear", "-noise_plateau": "0",
         "--weight_averaging": "ema", "--ema_decay": "0.999"},
        "combined: linear noise decay (N_fulldecay) + eval-only EMA (L_ema)"),

    # ------------------------------------------------------------------ #
    # LR — Task 3: LR magnitude re-tune on the LOCKED schedule (linear noise + EMA,
    # constant LR, now baked into base_args). lr_c x lr_a 2-D grid. The (3e-4,3e-4)
    # cell == anchor. lr_a=lr_c=3e-4 was inherited from the buggy double-step config;
    # single-step halved effective critic LR and the deterministic target removed the
    # noise that justified small LRs -> the critic likely wants to be faster.
    # ------------------------------------------------------------------ #
    Arm("LR", "LR_a15_c20",  ["cc_g1", "cc_g2", "nocost"], {"-lr_a": "1.5e-4", "-lr_c": "2e-4"},   "lr_a=1.5e-4 lr_c=2e-4"),
    Arm("LR", "LR_a15_c30",  ["cc_g1", "cc_g2", "nocost"], {"-lr_a": "1.5e-4", "-lr_c": "3e-4"},   "lr_a=1.5e-4 lr_c=3e-4"),
    Arm("LR", "LR_a15_c45",  ["cc_g1", "cc_g2", "nocost"], {"-lr_a": "1.5e-4", "-lr_c": "4.5e-4"}, "lr_a=1.5e-4 lr_c=4.5e-4"),
    Arm("LR", "LR_a15_c60",  ["cc_g1", "cc_g2", "nocost"], {"-lr_a": "1.5e-4", "-lr_c": "6e-4"},   "lr_a=1.5e-4 lr_c=6e-4"),
    Arm("LR", "LR_a30_c20",  ["cc_g1", "cc_g2", "nocost"], {"-lr_a": "3e-4",   "-lr_c": "2e-4"},   "lr_a=3e-4 lr_c=2e-4"),
    Arm("LR", "LR_a30_c45",  ["cc_g1", "cc_g2", "nocost"], {"-lr_a": "3e-4",   "-lr_c": "4.5e-4"}, "lr_a=3e-4 lr_c=4.5e-4"),
    Arm("LR", "LR_a30_c60",  ["cc_g1", "cc_g2", "nocost"], {"-lr_a": "3e-4",   "-lr_c": "6e-4"},   "lr_a=3e-4 lr_c=6e-4"),
    # (lr_a=3e-4, lr_c=3e-4) is the anchor cell — not repeated as an arm.
]


# ----------------------------- runner --------------------------------------- #
PARSE_PRICE = re.compile(r"Option Price:\s*([\d.\-eE]+)\s*\xb1\s*([\d.\-eE]+)")
PARSE_LSM = re.compile(r"LSM Benchmark Price:\s*([\d.\-eE]+)")
PARSE_LAST_AVG100 = re.compile(r"Average100\s*=\s*([\d.\-eE]+)")
PARSE_CALIB = re.compile(r"Calibration wall-time:\s*([\d.\-eE]+)s")


def parse_stdout(text: str) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {
        "eval_price": None, "eval_price_se": None, "lsm_price": None, "final_avg100": None,
        "calib_seconds": None,
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
    m = PARSE_CALIB.search(text)
    if m:
        out["calib_seconds"] = m.group(1)
    return out


def build_cmd(regime: str, arm: Arm, seed: int, n_paths: int) -> List[str]:
    args = dict(base_args(n_paths))
    args.update(REGIMES[regime])
    args.update(arm.overrides)
    args["-name"] = f"_v63audit_{regime}_{arm.label}_s{seed}"
    args["-seed"] = str(seed)
    cmd = ["python", "run.py"]
    for k, v in args.items():
        cmd.extend([k, v])
    return cmd


def run_one(regime: str, arm: Arm, seed: int, n_paths: int, dry_run: bool) -> Dict[str, str]:
    cmd = build_cmd(regime, arm, seed, n_paths)
    if dry_run:
        print(f"[DRY] {regime}/{arm.label} s{seed}: {' '.join(shlex.quote(c) for c in cmd)}")
        return {"status": "dry"}

    log_path = LOG_DIR / f"{regime}_{arm.label}_s{seed}.log"
    print(f"[{regime}/{arm.label} s{seed}] launching (log -> {log_path.name})")
    t0 = time.time()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    wall = time.time() - t0
    parsed = parse_stdout(log_path.read_text(errors="replace"))
    row = {
        "label": arm.label,
        "arm_id": arm.arm_id,
        "contract": regime,
        "n_paths": str(n_paths),
        "seed": str(seed),
        "status": "ok" if result.returncode == 0 else f"exit_{result.returncode}",
        "wall_seconds": f"{wall:.1f}",
        **{k: (v if v is not None else "") for k, v in parsed.items()},
        "note": arm.note,
        "overrides": json.dumps(arm.overrides),
    }
    print(f"[{regime}/{arm.label} s{seed}] done {wall:.0f}s | price={parsed.get('eval_price')} "
          f"lsm={parsed.get('lsm_price')} status={row['status']}")
    return row


FIELDNAMES = ["label", "arm_id", "contract", "n_paths", "seed", "status", "eval_price",
              "eval_price_se", "lsm_price", "final_avg100", "calib_seconds", "wall_seconds",
              "note", "overrides"]


def write_csv(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})


# ----------------------------- analysis ------------------------------------- #
def analyze(csv_path: Path) -> None:
    """Within each regime, Welch (mean) + Levene (spread) of each arm vs anchor."""
    import stats_analysis as sa  # local import; tools/ is on sys.path

    rows = [r for r in csv.DictReader(open(csv_path)) if r.get("status") == "ok"]
    if not rows:
        print("No 'ok' rows to analyze yet.")
        return
    groups = sa.summarize_groups(rows)
    by_regime: Dict[str, List] = {}
    for g in groups:
        by_regime.setdefault(g.contract, []).append(g)

    print("\n" + "=" * 78)
    print("v63 AUDIT — anchor vs arm (per regime).  Pass = Welch p>=0.05 & Levene p>=0.05")
    print("=" * 78)
    for regime, gs in sorted(by_regime.items()):
        anchor = next((g for g in gs if g.label == "anchor"), None)
        print(f"\n### {regime}  (n_paths={gs[0].n_paths})")
        if anchor is None or not anchor.deltas:
            print("  (no anchor data yet)")
            continue
        print(f"  anchor          Δ% = {anchor.mean:+.3f} ± {anchor.std:.3f} "
              f"(n={anchor.n_seeds}, worst={min(anchor.deltas):+.3f})")
        for g in sorted(gs, key=lambda x: x.label):
            if g.label == "anchor":
                continue
            wt = sa.welch_t_test(g.deltas, anchor.deltas)
            lev = sa.scale_test(g.deltas, anchor.deltas, center="median")
            wp = wt.get("p", float("nan")) if wt.get("valid", True) else float("nan")
            lp = lev.get("p", float("nan")) if lev.get("valid", True) else float("nan")
            d_mean = g.mean - anchor.mean
            tighter = g.std <= anchor.std
            # Regression iff mean drops materially, OR variance is significantly WIDER.
            mean_ok = g.mean >= anchor.mean - 0.30
            var_bad = (lp < 0.05) and (not tighter)
            if not mean_ok or var_bad:
                verdict = "FAIL"
            elif (lp < 0.05 and tighter) or g.mean > anchor.mean + 0.10:
                verdict = "WIN "   # significant tighter spread, or a clear mean gain
            else:
                verdict = "pass"
            var_dir = "↓tight" if tighter else "↑WIDE"
            print(f"  {g.label:16s} Δ% = {g.mean:+.3f} ± {g.std:.3f} "
                  f"(worst={min(g.deltas):+.3f}) | Δmean={d_mean:+.3f} | "
                  f"Welch p={wp:.3f} Levene p={lp:.3f} {var_dir} -> {verdict}")
    print("\nNote: E4_drop_lrc6e4 must MATCH the anchor (mean & std) to justify removing the "
          "double-step; E4_drop_lrc3e4 is the un-retuned control (expected to drop). "
          "E1 additionally requires NO g1 seed blow-up (inspect per-seed deltas).\n")


# ----------------------------- main ----------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_paths", type=int, default=4096, help="Training episodes per run (default 4096).")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13],
                   help="Seeds (default 3-seed screen {11,12,13}; escalate with 11..22).")
    p.add_argument("--arms", type=str, nargs="+", default=None,
                   help="Arm IDs to run (default all). E.g. --arms E4  (anchor always included).")
    p.add_argument("--regimes", type=str, nargs="+", default=None,
                   help="Regimes to run (default all relevant). Subset of cc_g1,cc_g2,nocost.")
    p.add_argument("--only", type=str, nargs="+", default=None,
                   help="Label whitelist (e.g. anchor W_warmup256 W_warmup512). Filters final job list.")
    p.add_argument("--max_workers", type=int, default=2, help="Parallel subprocess cap (default 2).")
    p.add_argument("--dry_run", action="store_true", help="Print commands, run nothing.")
    p.add_argument("--analyze_only", action="store_true", help="Skip running; analyze existing CSV.")
    p.add_argument("--resume", action="store_true",
                   help="Keep existing 'ok' rows in the CSV, skip already-completed "
                        "(label,contract,seed) combos, and merge new rows in (interrupt-safe).")
    args = p.parse_args()

    csv_path = LOG_DIR / "sweep_v63_audit.csv"
    if args.analyze_only:
        analyze(csv_path)
        return

    wanted_arms = set(args.arms) if args.arms else None
    wanted_regimes = set(args.regimes) if args.regimes else None

    # Build the run list: for each arm, for each of its regimes, for each seed.
    # The anchor is always included for any regime that has at least one selected arm.
    selected = [a for a in ARMS if (a.arm_id == "anchor" or wanted_arms is None or a.arm_id in wanted_arms)]
    active_regimes = set()
    for a in selected:
        if a.arm_id == "anchor":
            continue
        for r in a.regimes:
            if wanted_regimes is None or r in wanted_regimes:
                active_regimes.add(r)
    if wanted_arms is None and wanted_regimes is not None:
        active_regimes &= wanted_regimes

    only = set(args.only) if args.only else None

    # Resume support: load existing 'ok' rows, skip combos already done, merge new in.
    existing_rows: List[Dict[str, str]] = []
    done: set = set()
    if args.resume and csv_path.exists():
        for r in csv.DictReader(open(csv_path)):
            existing_rows.append(r)
            if r.get("status") == "ok":
                done.add((r.get("label"), r.get("contract"), str(r.get("seed"))))
        print(f"[resume] loaded {len(existing_rows)} existing rows ({len(done)} 'ok' combos to skip).")

    jobs = []
    skipped = 0
    for a in selected:
        if only is not None and a.label not in only:
            continue
        regimes = active_regimes if a.arm_id == "anchor" else [r for r in a.regimes if r in active_regimes]
        for r in regimes:
            for s in args.seeds:
                if (a.label, r, str(s)) in done:
                    skipped += 1
                    continue
                jobs.append((r, a, s))
    if args.resume:
        print(f"[resume] {skipped} combos already done, {len(jobs)} remaining.")

    print(f"v63 audit: {len(jobs)} runs "
          f"(arms={sorted({a.arm_id for _, a, _ in jobs})}, regimes={sorted(active_regimes)}, "
          f"seeds={args.seeds}, n_paths={args.n_paths})")
    if not jobs:
        print("Nothing to run (check --arms/--regimes).")
        return

    rows: List[Dict[str, str]] = list(existing_rows)  # carry forward prior rows when resuming
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(run_one, r, a, s, args.n_paths, args.dry_run) for (r, a, s) in jobs]
        for fut in cf.as_completed(futs):
            row = fut.result()
            if row.get("status") in (None, "dry"):
                continue
            rows.append(row)
            write_csv(csv_path, rows)  # incremental save (union of prior + new)

    if not args.dry_run and rows:
        print(f"\nWrote {len(rows)} rows -> {csv_path}")
        analyze(csv_path)


if __name__ == "__main__":
    main()
