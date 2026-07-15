# Prompt: generate "Convex Costs Results 9" (v9) and update the paper's headline table

Paste this into a fresh Claude Code chat in the `DRL-Swing-Options` repo.

---

## Task

Generate a new **"Convex Costs Results 9"** study with the v67 two-mode C++ pricer and update the
paper's headline results table (`Paper/DRL_Swing_Options.tex`, Table with `\label{tab:results}`) to
the new numbers and reporting format. The paper's **methodology (§4)** is already updated to v67; the
**results section (§5) + tables are deliberately stale** (still the old 32k study) and are yours to
regenerate so they match the v67 method.

## The three methods

1. **LSM-D** — Python full-state discretized-action LSM, Chebyshev **degree-2**, **M=5** action
   levels (`n_actions=5`). Use `src/lsm_swing_pricer.fit_lsm_estimators` /
   `price_swing_option_lsm_oos`, `state_mode="full"`, `basis_type="chebyshev"`, `poly_degree=2`,
   `reg_type="none"`. Fit on **65,536** independent training paths.
2. **AC-sample** — v67 **no-kernel** build `cpp_pricer/build_v67_nokernel/price_swing`, recipe =
   `koff_flags(n_train)`. Train **32,768 episodes** (`--n_train 32768`).
3. **AC-kernel** — v67 **kernel** build `cpp_pricer/build_v67_kernel/price_swing`, recipe =
   `KON_BALANCED` + `--kernel cpp_pricer/data/kernel_v64.bin`. Train **4,096 episodes**
   (`--n_train 4096`).

`KON_BALANCED` and `koff_flags` are defined in
`Jupyter Notebooks/3: Validation 3: Semi-Analytical Kernel & RL Pricer/gen_rl_validation_v67.py`.
If the two C++ builds don't exist, build them per `cpp_pricer/README.md` ("v67 — two-mode builds").

## Protocol

**Start from `gen_results8_v67.py`** (same folder) — it already does per-seed OOS dumping, paired LSM
pricing, and bootstrap. Modify it as follows:

1. **8 training runs per method**: `SEEDS = [11,12,13,14,15,16,17,18]`.
2. **Per-method budgets** (the key change vs Results 8): LSM-D 65,536 fit paths; AC-sample **32,768
   episodes**; AC-kernel **4,096 episodes**. Each OOS set is **65,536 paths**.
3. **8 paired OOS sets, paired by run index.** For run/seed `k`, generate ONE OOS set of 65,536
   paths at **eval seed = train_seed_k + 777**, dumped once by the C++ run (`--dump_eval_paths`,
   SXYP blob). All three methods price **OOS_k**: AC-kernel run k, AC-sample run k, and LSM-D run k
   are all evaluated on OOS_k. (HHK OOS generation is independent of kernel/episode count, so the
   no-kernel build reproduces the identical OOS deterministically, and the Python LSM prices the
   dumped paths via `read_sxy` + `dataset=(0,S,X,Y)`.)
4. **Reporting — mean ± 95% CI.** For each (cell, method), collect the 8 paired OOS prices and
   bootstrap (1000 resamples) the **mean** and **std** (the bootstrap standard error). Report
   **mean ± 95% CI** with **CI95 = 1.96 × bootstrap_std** (normal fit, z = 1.96), formatted like
   `2.1234 ± 0.034`. Also compute the paired Δ% per run (RL/LSM − 1 on the same OOS) and its
   bootstrap mean ± 95% CI.
5. **Grid** (mirror `gen_results8_v67.py`'s `GRID`, 29 cells): `(c=0, γ=1)` plus
   `c ∈ {0.01,0.02,0.04,0.05,0.08,0.10,0.15} × γ ∈ {1,1.5,2,3}`. The paper drops γ=3 for c>0.05.

**Performance:** run C++ jobs across cores with `--threads 1` each (the 512×64 GEMM doesn't scale
intra-run; see `gen_rl_validation_v67.py` WORKERS/THREADS + the BLAS env-var pinning). Use a
resumable on-disk cache keyed by `(method, c, gamma, seed, n_train, n_eval)`. Smoke-test first with
`VAL3_NTEST=16384` before the full 32k/4k run.

## Outputs

- `Jupyter Notebooks/Convex Costs Results 9.csv` (wide: per cell, each method's mean, CI95, paired
  Δ% + Δ% CI95) and `results9_pricing_long.csv` (long: `Configuration,c,gamma,method,seed,price`).
- **Update Table `\label{tab:results}`** in `Paper/DRL_Swing_Options.tex`: report each method as
  **mean ± 95% CI** (e.g. `2.1234 ± 0.034`), with a **legend at the bottom of the table** naming the
  methods (LSM-D = full-state Chebyshev deg-2 M=5, 65k fit; AC-sample = v67 no-kernel, 32k ep;
  AC-kernel = v67 kernel, 4k ep). Restructure columns from the old `price / σ_seed / Δ%` layout to
  the `mean ± CI` layout. Regenerate Table `\label{tab:lsm_grid}` (action-grid sensitivity)
  consistently if you refit the M=5/9/17 ladder.
- **Update the protocol prose to match** the new study: §5.1 Experimental Setup; the two table
  captions; §5.5 Seed Robustness (now 8 paired runs + bootstrap CI instead of 3-seed σ_seed);
  reproducibility appendix "Seeds and data" (8 runs; AC-sample 32k / AC-kernel 4k; LSM 65k fit;
  8 paired OOS at seed train+777; bootstrap mean ± 95% CI) and "Checkpoints" (both variants use the
  evaluation EMA). Re-check and update the discussion numbers in §5.2/§5.4 (e.g. "trails by 1–2%",
  "+3.21%", "two orders of magnitude") against Results 9.
- `make paper` and verify (no LaTeX errors / undefined refs; tables typeset; numbers consistent).

## Do not touch

The §4 methodology (recipe, Tables `tab:network_arch` / `tab:hyperparams`, profitability gate, STE,
Computational Efficiency) — it is already correct for v67. Only Table 3's **Episodes** row already
reflects the asymmetric 32k (AC-sample) / 4k (AC-kernel) budget; leave it. Do not change the hedging
section (§6) or its figures.
