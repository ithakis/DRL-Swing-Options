"""Task 3 — sample-efficiency study collector.

Runs the C++ v65 pricer across (arm × regime × budget N × seed) and emits a tidy CSV
    sampler, replay, step_density, arm, regime, c, gamma, N, seed,
    price, ci95, std, bangbang, cpu_train, t_total
one row per run. The OOS test set is the common 65 536-path MC set (seed+777), identical
across arms at a fixed seed, so price differences are policy-only (CRN-shared test noise).

Arms (default = the P1 cheap screen):
    A0  mc      uniform              (control, v65 canonical — bit-identical)
    A1  arqmc   uniform              (array-RQMC scenario generation)
    A2  mc      time     density=2   (time-graded replay, denser toward maturity)
    A4  mc      coverage             (coverage-flattening replay)

Env knobs (all optional):
    ARMS="A0:mc:uniform:1,A1:arqmc:uniform:1,A2:mc:time:2,A4:mc:coverage:1"
    REGIMES="g2"            # subset of {nocost,g1,g15,g2}
    NS="1024,2048,4096"
    SEEDS="11-22"           # inclusive range or comma list
    NEVAL=65536  WORKERS=8  THREADS=2  NTAG=screen
    OUT=logs/sampler_study/sampler_results.csv

Run:  EP11python tools/collect_sampler_results.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "cpp_pricer" / "build_v65" / "price_swing"
KERNEL = ROOT / "cpp_pricer" / "data" / "kernel_v64.bin"
V65 = ["--hidden", "48", "--actor_layers", "2", "--critic_layers", "4", "--hidden_actor", "32",
       "--batch", "64", "--learn_number", "3", "--lr_c", "5e-4", "--quiet"]

REGIME_CG = {"nocost": (0.0, 1.0), "g1": (0.04, 1.0), "g15": (0.04, 1.5), "g2": (0.04, 2.0)}

DEFAULT_ARMS = "A0:mc:uniform:1,A1:arqmc:uniform:1,A2:mc:time:2,A4:mc:coverage:1"


def parse_arms(spec):
    arms = []
    for tok in spec.split(","):
        name, sampler, replay, dens = tok.split(":")
        arms.append((name, sampler, replay, float(dens)))
    return arms


def parse_seeds(spec):
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def cpp_run(arm, regime, N, seed, n_eval, threads):
    name, sampler, replay, dens = arm
    c, gamma = REGIME_CG[regime]
    cmd = [str(CPP), "--seed", str(seed), "--n_train", str(N), "--n_eval", str(n_eval),
           "--c_cost", str(c), "--gamma_cost", str(gamma), "--kernel", str(KERNEL),
           "--sampler", sampler, "--replay", replay, "--step_density", str(dens),
           "--threads", str(threads), *V65]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT / "cpp_pricer"))
    if out.returncode != 0:
        raise RuntimeError(f"cpp failed ({name},{regime},N={N},s={seed}): {out.stderr[-400:]}")
    j = json.loads(out.stdout)
    return {
        "arm": name, "sampler": sampler, "replay": replay, "step_density": dens,
        "regime": regime, "c": c, "gamma": gamma, "N": N, "seed": seed,
        "price": j["price"], "ci95": j["ci95"], "std": j["std"], "bangbang": j["bangbang"],
        "cpu_train": j["cpu_train"], "t_total": j["t_total"], "wall_s": time.time() - t0,
    }


def main():
    assert CPP.exists(), f"build the v65 binary first: {CPP}"
    arms = parse_arms(os.environ.get("ARMS", DEFAULT_ARMS))
    regimes = os.environ.get("REGIMES", "g2").split(",")
    Ns = [int(x) for x in os.environ.get("NS", "1024,2048,4096").split(",")]
    seeds = parse_seeds(os.environ.get("SEEDS", "11-22"))
    n_eval = int(os.environ.get("NEVAL", 65536))
    workers = int(os.environ.get("WORKERS", 8))
    threads = int(os.environ.get("THREADS", 2))
    out_csv = Path(os.environ.get("OUT", ROOT / "logs" / "sampler_study" / "sampler_results.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(a, rg, N, s) for a in arms for rg in regimes for N in Ns for s in seeds]
    print(f"sampler study: {len(arms)} arms × {len(regimes)} regimes × {len(Ns)} N × "
          f"{len(seeds)} seeds = {len(jobs)} runs | {workers} workers, {threads} thr/run")

    rows, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(cpp_run, a, rg, N, s, n_eval, threads): (a, rg, N, s) for a, rg, N, s in jobs}
        for fut in as_completed(futs):
            rows.append(fut.result()); done += 1
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)} ({el:.0f}s, ~{el/done*(len(jobs)-done):.0f}s left)")

    df = pd.DataFrame(rows).sort_values(["regime", "N", "arm", "seed"]).reset_index(drop=True)
    if out_csv.exists():   # append to prior results, de-dup on the run key
        prev = pd.read_csv(out_csv)
        key = ["arm", "regime", "N", "seed", "sampler", "replay", "step_density"]
        df = pd.concat([prev, df]).drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} rows)")

    # quick per-(regime,N,arm) summary
    g = df.groupby(["regime", "N", "arm"])["price"].agg(["mean", "std", "count"])
    print(g.to_string())


if __name__ == "__main__":
    main()
