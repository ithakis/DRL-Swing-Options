#!/usr/bin/env python
"""Price-vs-#eval-paths convergence study (pure actor-NN rollout, no kernel/M_x).

Answers the questions:
  (1) How does the RL price estimate and its uncertainty behave as a function of the
      number of evaluation paths?
  (2) Evaluation uses ONLY the actor network rolled out on Monte-Carlo episodes
      (src/agent_evaluation.py — zero kernel/M_x references). M_x is a *training*-time
      TD-target device; it plays no role here.
  (3)+(4) 12 trained seeds, 30 log-spaced path counts from 1 → 32768, empirical 95% CI,
      plotted against the LSM(~130k) benchmark. Publication-quality figure.

Pipeline:
  Stage 1 (train): train 12 canonical v63 agents (focal cc_g2; single-step default,
                   critic_warmup=512, kernel M_x=2 in TRAINING only) via run.py, saving
                   runs/_pvp_s{seed}.pth.  Skips seeds already trained.
  Stage 2 (eval) : build ONE shared out-of-sample pool of ~130k HHK paths (seed 999);
                   roll out each agent's actor on it -> per-path discounted returns
                   (12 x ~130k matrix). LSM benchmark fit OOS on the same pool.
  Stage 3 (plot) : log-spaced convergence curve + empirical 95% CI vs LSM line.

Usage (conda activate EP11):
  python tools/price_vs_paths.py                 # train missing seeds, then eval+plot
  python tools/price_vs_paths.py --skip-train    # eval+plot using existing runs/_pvp_s*.pth
  python tools/price_vs_paths.py --train-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "logs" / "_price_vs_paths"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(11, 23))                 # 12 training seeds
EVAL_SEED = 999                             # out-of-sample test pool
LSM_TRAIN_SEED = 998
N_EVAL = 131072                             # ~130k (2^17: clean QMC/stratification)
N_LSM_TRAIN = 16384
N_MAX_GRID = 32768                          # largest path count on the x-axis
N_GRID_POINTS = 30
BOOT = 400                                  # bootstrap reps per seed (for the CI band)

# ── Focal cc_g2 contract + HHK process (the headline regime) ──────────────────
CONTRACT = dict(q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0,
                maturity=0.0833, n_rights=22, min_refraction_periods=0,
                r=0.05, c_cost=0.04, gamma_cost=2.0)
HHK = dict(S0=1.0, alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3)
LSM = dict(basis="chebyshev", degree=7, reg="none", reg_alpha=1e-6, n_actions=5)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — train the 12 canonical agents (kernel M_x=2 in training only)
# ──────────────────────────────────────────────────────────────────────────────
def train_cmd(seed: int) -> list[str]:
    return [
        "python", "run.py", "-name", f"_pvp_s{seed}", "-seed", str(seed),
        "-n_paths", "4096", "-n_paths_eval", "4096", "-eval_every", "-1",
        "-nstep", "1", "--gamma", "1",
        "--use_expected_target", "1", "--kernel_M_x", "2",
        "--kernel_M_per_k", "1", "--kernel_N_max", "1",
        "-learn_every", "2", "-noise_sigma0", "1.30", "-noise_floor", "0.26",
        "-noise_plateau", "3200", "--adaptive_noise_scale", "0.6",
        "--actor_output_activation", "beta_sigmoid_3.0",
        "--min_replay_size", "1000", "-t", "0.0032", "-bs", "128",
        "-layer_size", "64", "--activation", "silu", "--norm", "layernorm",
        "--init_method", "orthogonal", "-lr_a", "3e-4", "-lr_c", "3e-4",
        "--final_lr_fraction", "1.0", "--critic_warmup_episodes", "512",
        "--single_critic_step", "1", "--warmup_noise_fraction", "0.3",
        "--use_robust_normalization", "1", "--compile", "0", "-n_cores", "1",
        "--disable_csv_logging", "1", "--limit_logging_frequency", "1",
        "--strike", "1.0", "--maturity", "0.0833", "--n_rights", "22",
        "--q_min", "0.0", "--q_max", "2.0", "--Q_min", "0.0", "--Q_max", "20.0",
        "--risk_free_rate", "0.05", "--min_refraction_periods", "0",
        "--c_cost", "0.04", "--gamma_cost", "2",
        "--lsm_basis", "chebyshev", "--lsm_degree", "7",
        "--S0", "1.0", "--alpha", "12.0", "--sigma", "1.2",
        "--beta", "150.0", "--lam", "6.0", "--mu_J", "0.3",
    ]


def train_agents(max_workers: int) -> None:
    todo = [s for s in SEEDS if not (ROOT / "runs" / f"_pvp_s{s}.pth").exists()]
    if not todo:
        print(f"All {len(SEEDS)} agents already trained.")
        return
    print(f"Training {len(todo)} agents (seeds {todo}) with {max_workers} workers...")
    import concurrent.futures as cf
    logs = OUT_DIR / "train_logs"
    logs.mkdir(exist_ok=True)

    def _run(seed: int):
        t0 = time.time()
        with open(logs / f"train_s{seed}.log", "w") as f:
            r = subprocess.run(train_cmd(seed), stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        print(f"  seed {seed}: exit {r.returncode} in {time.time()-t0:.0f}s")
        return seed, r.returncode

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_run, todo))


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — build the shared eval pool, roll out each actor, compute LSM
# ──────────────────────────────────────────────────────────────────────────────
def build_contract():
    from src.swing_contract import SwingContract
    return SwingContract(**CONTRACT)


def hhk_params():
    from src.simulate_hhk_spot import no_seasonal_function
    return dict(S0=HHK["S0"], T=CONTRACT["maturity"], n_steps=CONTRACT["n_rights"] - 1,
                alpha=HHK["alpha"], sigma=HHK["sigma"], beta=HHK["beta"],
                lam=HHK["lam"], mu_J=HHK["mu_J"], f=no_seasonal_function)


def make_dataset(n_paths: int, seed: int, dtype=np.float32):
    from src.simulate_hhk_spot import simulate_hhk_spot
    return simulate_hhk_spot(**hhk_params(), n_paths=n_paths, seed=seed,
                             stratify=True, batch_size=4096, dtype=dtype)


def build_agent(seed: int, contract):
    """Reconstruct the trained actor faithfully and load its weights."""
    import torch
    from src.agent import Agent
    agent = Agent(
        state_size=9, action_size=1, n_step=1, random_seed=seed,
        hidden_size=64, BATCH_SIZE=128, GAMMA=1.0, t=0.0032,
        LR_ACTOR=3e-4, LR_CRITIC=3e-4, activation="silu", norm_type="layernorm",
        init_method="orthogonal", action_output="beta_sigmoid_3.0",
        use_robust_normalization=True, strike=CONTRACT["strike"], device="cpu",
    )
    pth = ROOT / "runs" / f"_pvp_s{seed}.pth"
    agent.actor_local.load_state_dict(torch.load(pth, map_location="cpu"))
    agent.contract = contract          # drives the profitability-gate init in agent.act
    return agent


def eval_returns(agent, eval_env) -> np.ndarray:
    """Per-path discounted returns from pure actor rollout (no kernel)."""
    from src.agent_evaluation import evaluate_agent
    summary = evaluate_agent(agent=agent, eval_env=eval_env, writer=None, path=None,
                             evaluations_dir=None, lsm_price=None,
                             eval_batch_size=16384, n_episodes=eval_env.S.shape[0])
    return np.asarray(summary.returns, dtype=np.float64)


def lsm_benchmark(contract, eval_ds_f64):
    from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos
    print("Fitting LSM (in-sample 16384) + pricing OOS on the ~130k pool...")
    lsm_train = tuple(np.asarray(a, dtype=np.float64) for a in make_dataset(N_LSM_TRAIN, LSM_TRAIN_SEED))
    est = fit_lsm_estimators(contract=contract, dataset=lsm_train, poly_degree=LSM["degree"],
                             basis_type=LSM["basis"], reg_type=LSM["reg"],
                             reg_alpha=LSM["reg_alpha"], n_actions=LSM["n_actions"])
    mean, (q5, q95) = price_swing_option_lsm_oos(contract=contract, dataset=eval_ds_f64,
                                                 estimators=est, seed=EVAL_SEED, csv_path=None)
    # (q5,q95) is the bootstrap 95% CI of the LSM mean → half-width = (q95−q5)/2.
    return mean, (q95 - q5) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — convergence curve + empirical 95% CI, then plot
# ──────────────────────────────────────────────────────────────────────────────
def log_path_grid() -> np.ndarray:
    g = np.unique(np.round(np.logspace(0, np.log10(N_MAX_GRID), N_GRID_POINTS)).astype(int))
    return g[g >= 1]


def convergence_bands(R: np.ndarray, grid: np.ndarray, rng):
    """R: (n_seeds, N_EVAL) per-path returns. For each N draw BOOT random N-subsets per
    seed (without replacement) -> pooled distribution of the N-path price estimate.
    Returns median, lo(2.5%), hi(97.5%), and the std of the estimate, per grid point."""
    n_seeds, n_pool = R.shape
    med = np.empty(len(grid)); lo = np.empty(len(grid)); hi = np.empty(len(grid)); sd = np.empty(len(grid))
    for j, N in enumerate(grid):
        ests = np.empty(n_seeds * BOOT)
        k = 0
        for s in range(n_seeds):
            row = R[s]
            for _ in range(BOOT):
                idx = rng.integers(0, n_pool, size=int(N))   # bootstrap subsample of size N
                ests[k] = row[idx].mean(); k += 1
        med[j] = np.median(ests); lo[j] = np.percentile(ests, 2.5)
        hi[j] = np.percentile(ests, 97.5); sd[j] = ests.std()
    return med, lo, hi, sd


def make_plot(grid, med, lo, hi, sd, rl_price, seed_prices, lsm_mean, lsm_ci):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1.0]})

    # ── Panel A: price vs #paths ──
    ax1.fill_between(grid, lo, hi, color="#1f77b4", alpha=0.20, label="RL 95% CI (12 seeds × bootstrap)")
    ax1.plot(grid, med, color="#1f77b4", lw=2.0, label="RL price (median estimate)")
    ax1.axhline(rl_price, color="#1f77b4", ls=":", lw=1.2, alpha=0.7,
                label=f"RL converged price = {rl_price:.4f}")
    ax1.axhline(lsm_mean, color="#d62728", ls="--", lw=1.8, label=f"LSM (~130k) = {lsm_mean:.4f}")
    ax1.fill_between(grid, lsm_mean - lsm_ci, lsm_mean + lsm_ci, color="#d62728", alpha=0.18)
    ax1.set_xscale("log")
    ax1.set_ylabel("Option price")
    ax1.set_title("Swing-option price vs. number of evaluation paths\n"
                  "(focal c=0.04, γ=2; actor-NN rollout only — no kernel/M$_x$)", fontsize=11)
    # clip extreme low-N band for readability
    ax1.set_ylim(min(lsm_mean, rl_price) - 0.35, max(lsm_mean, rl_price) + 0.35)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # ── Panel B: std of the price estimate vs #paths (log-log) ──
    ax2.loglog(grid, sd, color="#2ca02c", lw=2.0, marker="o", ms=3, label="std of price estimate")
    ref = sd[0] * np.sqrt(grid[0]) / np.sqrt(grid)        # 1/√N reference anchored at first point
    ax2.loglog(grid, ref, color="gray", ls="--", lw=1.2, label=r"$\propto 1/\sqrt{N}$ reference")
    ax2.set_xlabel("Number of evaluation paths  $N$  (log scale)")
    ax2.set_ylabel("Std of price est.")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"price_vs_paths.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nFigure saved: {OUT_DIR/'price_vs_paths.png'}  (+ .pdf)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--force-eval", action="store_true",
                   help="Re-roll-out agents even if returns_matrix.npy is cached.")
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    if not args.skip_train:
        train_agents(args.max_workers)
    if args.train_only:
        return

    have = [s for s in SEEDS if (ROOT / "runs" / f"_pvp_s{s}.pth").exists()]
    print(f"Evaluating {len(have)} trained agents on a shared {N_EVAL:,}-path OOS pool...")
    contract = build_contract()

    # Shared out-of-sample pool (float32 for RL env, float64 for LSM)
    t0 = time.time()
    eval_ds_f32 = make_dataset(N_EVAL, EVAL_SEED, dtype=np.float32)
    eval_ds_f64 = tuple(np.asarray(a, dtype=np.float64) for a in eval_ds_f32)
    print(f"  pool generated in {time.time()-t0:.0f}s")

    cache = OUT_DIR / "returns_matrix.npy"
    if cache.exists() and not args.force_eval and np.load(cache).shape[0] == len(have):
        R = np.load(cache)
        print(f"  loaded cached returns matrix {R.shape} (use --force-eval to recompute)")
    else:
        from src.swing_env import SwingOptionEnv
        R = []
        for s in have:
            agent = build_agent(s, contract)
            env = SwingOptionEnv(contract=contract, hhk_params=hhk_params(), dataset=eval_ds_f32)
            r = eval_returns(agent, env)
            R.append(r)
            print(f"  seed {s}: price={r.mean():.4f}  (per-path std={r.std():.3f})")
        R = np.vstack(R)
        np.save(cache, R)

    lsm_mean, lsm_ci = lsm_benchmark(contract, eval_ds_f64)
    print(f"LSM(~130k) = {lsm_mean:.4f} ± {lsm_ci:.4f} (95% CI of mean)")

    grid = log_path_grid()
    rng = np.random.default_rng(0)
    med, lo, hi, sd = convergence_bands(R, grid, rng)
    rl_price = R.mean()
    seed_prices = R.mean(axis=1)
    delta = (rl_price / lsm_mean - 1.0) * 100.0
    print(f"\nRL converged price = {rl_price:.4f}  (seed range "
          f"{seed_prices.min():.4f}–{seed_prices.max():.4f}) | Δ% vs LSM = {delta:+.2f}%")

    make_plot(grid, med, lo, hi, sd, rl_price, seed_prices, lsm_mean, lsm_ci)


if __name__ == "__main__":
    main()
