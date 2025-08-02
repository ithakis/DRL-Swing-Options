#!/usr/bin/env bash
# --------------------------------------------------------------------
# HHK Figure-10 replication: one-year swing with 100 daily rights
# Paper: "Modelling Spikes and Pricing Swing Options in Electricity Markets"
#        Quantitative Finance 9 (8), 2009, pp 937-949
# --------------------------------------------------------------------

args=(
    # ── LSM algorithm ────────────────────────────────────────────────
    -n_paths=4096           # 4096 or 8192 or 16384
    -poly_degree=3            # cubic regression (as in the paper)
    -seed=2                   # matches notebook seed

    # ── Swing-contract parameters (Fig. 10) ──────────────────────────
    --strike=1.0              # K = 1   (paper works in relative price units)
    --maturity=1.0            # 1-year delivery period
    --n_rights=365            # daily decision opportunities (matches notebook n_steps)
    --q_min=0.0               # bang-bang: 0 or 1 each day
    --q_max=1.0
    --Q_min=0.0
    --Q_max=100.0             # up to 100 call rights
    --risk_free_rate=0.0      # r = 0 in the experiment
    --min_refraction_days=0

    # ── HHK spike-model parameters (Fig. 1 & 10, matches notebook) ───
    --S0=1.0                  # initial spot (matches strike scale)
    --alpha=7.0               # OU mean-reversion speed
    --sigma=1.4               # OU volatility
    --beta=200.0              # jump decay rate
    --lam=4.0                 # Poisson-jump intensity (4 spikes / yr)
    --mu_J=0.4                # mean jump size
)
python -m src.lsm_swing_pricer "${args[@]}" -name "HHK_Fig10_LSM" -seed 2
