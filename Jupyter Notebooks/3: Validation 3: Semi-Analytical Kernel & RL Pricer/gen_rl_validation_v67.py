"""Validation 3 - Part II under **v67** (two-mode C++ pricer) — bootstrap methodology.

Three configs compared on the SAME out-of-sample paths:
  * lsm_M5     = full-state Chebyshev deg-2 LSM-D, 5 action levels (Python; src/lsm_swing_pricer)
  * kernel_on  = v67 kernel mode = build_v67_kernel (GELU-β3); R1 uses the balanced v65 recipe,
                 R2 uses the **accuracy champion** (--learn_number 4).
  * kernel_off = v67 no-kernel mode = build_v67_nokernel (optimised literal-v61 + eval-EMA; single-
                 sample TD). See cpp_pricer/docs/V61_CONFIG.md + research/V61_NOKERNEL_OPTIMIZATION_PLAN.md.

Methodology (the user's spec): **8 estimators trained per method, all evaluated on the SAME 8 OOS
65 536-path sets** (8+8+8 trained, 8 shared eval sets). OOS set k is generated ONCE by the C++ pricer
(eval seed = train_seed_k + 777) and dumped (`--dump_eval_paths`, SXYP blob); both RL modes share it
(the HHK sim is independent of the kernel) and the Python LSM prices the IDENTICAL paths
(`_extract_state_paths` ignores t, so dataset=(0,S,X,Y) is sufficient). The per-(method,seed) prices
are written long-format; the notebook derives mean ± std by **bootstrap (1000 resamples)**, pairing
each RL price with the LSM price on the *same* OOS set for a clean paired Δ%.

Produces:
  rl_lsm_pricing.csv      R1 - regime x {lsm_M5,lsm_M9,lsm_M17,kernel_on,kernel_off} x 8 seeds
  episode_efficiency.csv  R2 - focal g2, {kernel_on(champion),kernel_off} x {2048..32768} x 8 seeds
                               + lsm_M5 reference (episodes=-1), all on the shared g2 OOS sets

Run:  EP11python gen_rl_validation_v67.py            (EP11 = ~/miniforge3/envs/EP11/bin/python)
      VAL3_NTEST=16384 EP11python gen_rl_validation_v67.py     # quick pass
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)

import rebuild_results_v7 as rb  # noqa: E402
from src.simulate_hhk_spot import simulate_hhk_spot  # noqa: E402
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos  # noqa: E402

# ---- config -----------------------------------------------------------------
N = int(os.environ.get("VAL3_NTEST", 65536))      # OOS + LSM-train path count
N_EVAL = int(os.environ.get("VAL3_NEVAL", 65536))
SEEDS = [int(s) for s in os.environ.get("VAL3_SEEDS", "11,12,13,14,15,16,17,18").split(",")]
# All 8 cores, 1 thread per core: parallelism is ACROSS C++ runs (each --threads 1). The C++ GEMM
# does not scale with intra-run threads (see cpp_pricer/DEVELOPMENT_NOTES.md), so 8 single-thread
# runs in parallel ≈ 8× the throughput of sequential --threads 8 runs.
WORKERS = int(os.environ.get("VAL3_WORKERS", "8"))
THREADS = 1
LSM_M = 5
LSM_MS = [5, 9, 17]   # R1 shows the LSM-D action-discretisation ladder (M = 5/9/17)

CPP_DIR = ROOT / "cpp_pricer"
KERNEL_BIN = CPP_DIR / "build_v67_kernel" / "price_swing"
NOKERNEL_BIN = CPP_DIR / "build_v67_nokernel" / "price_swing"
KERNEL = CPP_DIR / "data" / "kernel_v64.bin"
OOS_DIR = CPP_DIR / "data" / "_v67oos"; OOS_DIR.mkdir(exist_ok=True)

# ---- resumable on-disk price cache (so a killed run resumes; shared across both generators) ----
import threading  # noqa: E402

CACHE_DIR = CPP_DIR / "data" / "_v67cache"; CACHE_DIR.mkdir(exist_ok=True)
_RL_CACHE_PATH = CACHE_DIR / "rl_prices.json"
_RL_CACHE_LOCK = threading.Lock()
_RL_CACHE = json.load(open(_RL_CACHE_PATH)) if _RL_CACHE_PATH.exists() else {}

# v67 KERNEL recipes (GELU-β3 compiled into build_v67_kernel). R1 = balanced; R2 = accuracy champion.
KON_BALANCED = ["--hidden", "48", "--actor_layers", "2", "--critic_layers", "4", "--hidden_actor", "32",
                "--batch", "64", "--learn_number", "3", "--lr_c", "5e-4"]
KON_CHAMPION = ["--hidden", "48", "--actor_layers", "2", "--critic_layers", "4", "--hidden_actor", "32",
                "--batch", "64", "--learn_number", "4", "--lr_c", "5e-4"]   # ln4 = accuracy champion


def koff_flags(n_train: int) -> list[str]:
    """v67 no-kernel = optimised literal-v61 + eval-EMA. Plateau/cosine horizon + LR-warmup scale to
    the budget (literal v61 was 1024-ep warmup at 32 768 ep = 1/32; we hold that fraction so short
    budgets don't spend a quarter/half of training ramping the LR)."""
    plateau = max(1, n_train // 10)
    lr_warm = max(1, n_train // 32)   # 32 768 -> 1024 (== literal v61); scales down for short budgets
    return ["--kernel_off",
            "--actor_layers", "2", "--critic_layers", "2", "--hidden", "64", "--init_method", "1",
            "--lr_a", "1.6e-4", "--lr_c", "9e-5", "--wd_c", "1.2e-4",
            "--learn_every", "2", "--learn_number", "1", "--batch", "128", "--tau", "0.0032",
            "--noise_sigma0", "1.30", "--noise_floor", "0.26",
            "--noise_schedule", "hyperbolic", "--noise_plateau", str(plateau),
            "--adaptive_noise_scale", "0.6", "--warmup_noise_fraction", "0.4",
            "--critic_warmup", "1024", "--weight_avg", "0", "--ema_decay", "0.999",
            "--double_critic_step", "1",
            "--target_policy_noise", "0.15", "--tpn_decay_start", str(max(1, n_train // 2)),
            "--tpn_floor", "0.04",
            "--lr_schedule", "cosine", "--lr_warmup_episodes", str(lr_warm),
            "--lr_schedule_episodes", str(n_train), "--final_lr_fraction", "0.20", "--min_lr", "1e-6",
            "--min_replay", "18000", "--max_replay", "200000"]

REGIMES = [("nocost", 0.0, 1.0), ("g1", 0.04, 1.0), ("g15", 0.04, 1.5), ("g2", 0.04, 2.0)]
R2_EPISODES = [2048, 4096, 8192, 32768]   # efficiency curve: skip 16k, jump straight to the v61 budget


def read_sxy(path):
    """Read a C++ SXYP eval-paths dump -> (S, X, Y) float64 arrays of shape (n_paths, T)."""
    with open(path, "rb") as f:
        magic, n, T = struct.unpack("<iii", f.read(12))
        assert magic == 0x53585950, f"bad SXYP magic {magic:#x}"
        buf = np.fromfile(f, dtype="<f4")
    sz = n * T
    return tuple(buf[i * sz:(i + 1) * sz].reshape(n, T).astype(np.float64) for i in range(3))


def cpp_run(c, gamma, seed, n_train, kon_flags, dump=None, tag=None):
    """Run the C++ pricer; kon_flags=None -> no-kernel build. Returns OOS price; optionally dumps OOS.
    `tag` (e.g. 'kbal'/'kchamp'/'nokernel') makes the resumable cache key unambiguous and lets the two
    generators dedup identical runs. A cached price is reused only when any required OOS dump exists."""
    if tag is None:
        tag = "nokernel" if kon_flags is None else "kernel"
    ckey = f"{tag}|{c}|{gamma}|{seed}|{n_train}|{N_EVAL}"
    if ckey in _RL_CACHE and (dump is None or Path(dump).exists()):
        return _RL_CACHE[ckey]
    binary = NOKERNEL_BIN if kon_flags is None else KERNEL_BIN
    cmd = [str(binary), "--seed", str(seed), "--n_train", str(n_train), "--n_eval", str(N_EVAL),
           "--c_cost", str(c), "--gamma_cost", str(gamma), "--threads", str(THREADS), "--quiet"]
    cmd += koff_flags(n_train) if kon_flags is None else (kon_flags + ["--kernel", str(KERNEL)])
    if dump:
        cmd += ["--dump_eval_paths", str(dump)]
    # Pin each process to a single thread so WORKERS parallel runs map to 1 thread/core (the C++ app
    # is --threads 1, but Accelerate/BLAS would otherwise spawn its own threads and oversubscribe).
    env = {**os.environ, "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(CPP_DIR), env=env)
    if out.returncode != 0:
        raise RuntimeError(f"cpp failed: {' '.join(cmd)}\n{out.stderr[-500:]}")
    price = float(json.loads(out.stdout)["price"])
    with _RL_CACHE_LOCK:                       # persist immediately so a kill loses at most 1 run
        _RL_CACHE[ckey] = price
        tmp = _RL_CACHE_PATH.with_suffix(".json.tmp")
        json.dump(_RL_CACHE, open(tmp, "w"))
        tmp.replace(_RL_CACHE_PATH)
    return price


def lsm_worker(args):
    """Process-pool worker: fit full-state LSM-D at each M on its OWN training paths (seed=700+k) and
    price every M on the dumped OOS set -- the IDENTICAL paths the RL agents were evaluated on. One
    train simulation is reused across the M ladder. Returns {M: price}. Module-level for spawn."""
    reg, c, gamma, seed, k, oos_path, Ms = args
    # Cache key is the physics, not the label: (c, gamma, seed, train-path-count N). This makes it
    # N-safe (a VAL3_NTEST quick pass won't reuse full-resolution fits) and shared across both
    # generators (gen_rl 'g2' and gen_results8 'SwingOption_..._gamma2' price the identical OOS).
    cache_f = CACHE_DIR / f"lsm_c{c}_g{gamma}_s{seed}_N{N}.json"
    if cache_f.exists():                       # resumable: reuse if all requested M are present
        cached = {int(m): v for m, v in json.load(open(cache_f)).items()}
        if all(M in cached for M in Ms):
            return reg, c, gamma, seed, {M: cached[M] for M in Ms}
    params = rb.dotdict(dict(q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0, maturity=0.0833,
        n_rights=22, risk_free_rate=0.05, min_refraction_periods=0, c_cost=c, gamma_cost=gamma,
        S0=1.0, alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3))
    contract = rb.build_contract(params); hhk = rb.build_hhk_params(params)
    oos = read_sxy(oos_path)
    with contextlib.redirect_stdout(io.StringIO()):
        train = simulate_hhk_spot(**hhk, n_paths=N, seed=700 + k, stratify=True, batch_size=128)
    out = {}
    for M in Ms:
        with contextlib.redirect_stdout(io.StringIO()):
            est = fit_lsm_estimators(contract=contract, dataset=train, poly_degree=2,
                                     basis_type="chebyshev", state_mode="full",
                                     reg_type="none", reg_alpha=1e-6, n_actions=M)
            price, _ = price_swing_option_lsm_oos(contract=contract, dataset=(0, *oos),
                                                  estimators=est, seed=12345,
                                                  csv_path=None, _print_results=False)
        out[M] = float(price)
    tmp = cache_f.with_suffix(".json.tmp")      # persist this cell's ladder atomically
    json.dump(out, open(tmp, "w"))
    tmp.replace(cache_f)
    return reg, c, gamma, seed, out


def _rl_job(job):
    """One C++ run (executed in a worker thread, --threads 1). job carries everything needed."""
    kind = job[0]
    if kind == "R1k":      # R1 kernel (balanced) + dump the shared OOS for this (regime,seed)
        _, reg, c, gamma, seed = job
        return job, cpp_run(c, gamma, seed, 4096, KON_BALANCED, dump=OOS_DIR / f"{reg}_{seed}.bin",
                            tag="kbal")
    if kind == "R1n":      # R1 no-kernel (same OOS = seed+777)
        _, reg, c, gamma, seed = job
        return job, cpp_run(c, gamma, seed, 4096, None, tag="nokernel")
    if kind == "R2k":      # R2 kernel = accuracy champion (learn_number 4)
        _, seed, ep = job
        return job, cpp_run(0.04, 2.0, seed, ep, KON_CHAMPION, tag="kchamp")
    _, seed, ep = job      # R2n: no-kernel
    return job, cpp_run(0.04, 2.0, seed, ep, None, tag="nokernel")


def main():
    t0 = time.time()
    print(f"v67 generator (bootstrap) | N={N} seeds={SEEDS} | {WORKERS} workers x {THREADS} thread\n"
          f"  kernel_on  = {KERNEL_BIN}\n  kernel_off = {NOKERNEL_BIN}\n", flush=True)
    assert KERNEL_BIN.exists() and NOKERNEL_BIN.exists(), "build both v67 binaries first"

    # ---- Phase A: ALL C++ RL runs in parallel (8 cores, 1 thread each) ----
    jobs = []
    for reg, c, gamma in REGIMES:
        for seed in SEEDS:
            jobs += [("R1k", reg, c, gamma, seed), ("R1n", reg, c, gamma, seed)]
    for ep in R2_EPISODES:
        for seed in SEEDS:
            jobs += [("R2k", seed, ep), ("R2n", seed, ep)]
    rl = {}
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for job, price in ex.map(_rl_job, jobs):
            rl[job] = price; done += 1
            if done % WORKERS == 0:
                print(f"  RL {done}/{len(jobs)} runs done [{time.time()-t0:.0f}s]", flush=True)
    print(f"  Phase A (RL) done in {time.time()-t0:.0f}s.", flush=True)

    # ---- Phase B: LSM-D (M=5/9/17 ladder) priced on the IDENTICAL dumped OOS sets (process pool) ----
    lsm_jobs = [(reg, c, gamma, seed, k, str(OOS_DIR / f"{reg}_{seed}.bin"), LSM_MS)
                for reg, c, gamma in REGIMES for k, seed in enumerate(SEEDS)]
    lsm_res = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for reg, c, gamma, seed, out in ex.map(lsm_worker, lsm_jobs):
            lsm_res[(reg, seed)] = out
    print(f"  Phase B (LSM ladder) done in {time.time()-t0:.0f}s.", flush=True)

    r1_rows, lsm_g2 = [], {}
    for reg, c, gamma in REGIMES:
        for seed in SEEDS:
            ladder = lsm_res[(reg, seed)]
            if reg == "g2":
                lsm_g2[seed] = ladder[5]
            rows = [(f"lsm_M{M}", ladder[M]) for M in LSM_MS]
            rows += [("kernel_on", rl[("R1k", reg, c, gamma, seed)]),
                     ("kernel_off", rl[("R1n", reg, c, gamma, seed)])]
            for method, price in rows:
                r1_rows.append(dict(regime=reg, c=c, gamma=gamma, method=method, seed=seed, price=price))
    pd.DataFrame(r1_rows).to_csv(HERE / "rl_lsm_pricing.csv", index=False)

    # ---- R2 assembly (reuses the g2 OOS LSM prices) ----
    r2_rows = [dict(method="lsm_M5", episodes=-1, seed=s, price=lsm_g2[s]) for s in SEEDS]
    for ep in R2_EPISODES:
        for seed in SEEDS:
            r2_rows.append(dict(method="kernel_on", episodes=ep, seed=seed, price=rl[("R2k", seed, ep)]))
            r2_rows.append(dict(method="kernel_off", episodes=ep, seed=seed, price=rl[("R2n", seed, ep)]))
    pd.DataFrame(r2_rows).to_csv(HERE / "episode_efficiency.csv", index=False)
    print(f"\nDone in {time.time() - t0:.0f}s. CSVs in {HERE}")


if __name__ == "__main__":
    main()
