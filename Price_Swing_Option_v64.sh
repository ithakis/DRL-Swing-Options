#!/bin/bash
# =============================================================================
# v64 canonical reference — kernel-on D4PG swing-option pricer (focal contract)
# =============================================================================
# v64 is the winning configuration from the "Mega campaign" (see HPT.md): the v63
# kernel-on canonical PLUS the three orthogonal Stage-C winners
#   * depth-3 actor/critic        (--actor_layers 3 --critic_layers 3)
#   * two gradient steps / step   (-learn_number 2)
#   * softer squash               (--actor_output_activation beta_sigmoid_1.5)
#
# The ENTIRE recipe + the SwingOption_20 focal contract/HHK process are baked into
# run.py's argparse DEFAULTS, so `python run.py` with no flags already reproduces a
# v64 focal run. This script makes the canonical explicit (and sets the study budget
# + seeds) for reproducibility and as living documentation of the v64 default set.
#
# NOTE: v64 changes eval-critical knobs vs the published v61 paper agents
# (beta_sigmoid 1.5 vs 3.0; 3 vs 2 layers; kernel ON). v64 results are therefore
# NOT comparable to the v61 paper agents — see HPT.md "Mega campaign" for the
# re-baseline caveat. Reproduce the paper with its own saved-run JSONs.
# -----------------------------------------------------------------------------
args=(
    # --- study budget (the only run-size overrides; recipe is all defaults) ---
    -n_paths=32768
    -eval_every=1024
    -n_paths_eval=65536
    --disable_csv_logging=0
    --limit_logging_frequency=1

    # --- focal convex-cost cell (also the run.py default) ---
    --c_cost=0.04
    --gamma_cost=2

    # The lines below are v64 DEFAULTS in run.py — listed here only to document the
    # canonical recipe. They can be omitted with identical effect.
    # kernel:   --use_expected_target=1 --kernel_M_x=2 --kernel_M_per_k=1 --kernel_N_max=1
    # net:      -layer_size=64 --actor_layers=3 --critic_layers=3 --activation=silu --norm=layernorm
    # squash:   --actor_output_activation=beta_sigmoid_1.5
    # optim:    -lr_a=3e-4 -lr_c=6e-4 -t=0.0032 -bs=128 -learn_every=2 -learn_number=2
    # noise:    -noise_sigma0=1.30 -noise_floor=0.26 --noise_schedule=linear --adaptive_noise_scale=0.6
    # warm:     --critic_warmup_episodes=512 --warmup_noise_fraction=0.3 --calibrate_bias_mode=closed_form
    # eval:     --weight_averaging=ema --ema_decay=0.999
    # norm:     --use_robust_normalization=1
    # contract: --strike=1.0 --maturity=0.0833 --n_rights=22 --q_max=2.0 --Q_max=20.0
    # process:  --S0=1.0 --alpha=12.0 --sigma=1.2 --beta=150.0 --lam=6.0 --mu_J=0.3
    # lsm:      --lsm_basis=chebyshev --lsm_degree=2
)

pids=()
for seed in 11 12 13; do
    python run.py "${args[@]}" -name "SwingOption_20_c0.04_gamma2_v64_${seed}" -seed "$seed" &
    pids+=($!)
done
exit_code=0
for pid in "${pids[@]}"; do wait "$pid" || exit_code=1; done
exit $exit_code

## conda activate EP11 && bash Price_Swing_Option_v64.sh
