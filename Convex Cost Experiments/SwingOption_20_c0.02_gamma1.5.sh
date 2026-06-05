#!/bin/bash
# v64 canonical convex-cost experiment: c_cost=0.02, gamma_cost=1.5
# v64 = kernel-on D4PG canonical (kernel M_x=2, depth-3 net, learn_number=2,
# beta_sigmoid_1.5 squash, lr_c=6e-4, linear noise + eval-EMA, closed-form warm-start,
# critic_warmup=512, single critic step, robust normalization). The full recipe + the
# SwingOption_20 focal contract/HHK process now live in run.py's v64 DEFAULTS, so this
# script only overrides the study budget, this cell's convex-cost (c, gamma), and seeds.
args=(
    -n_paths=32768            # paper study budget
    -eval_every=1024
    -n_paths_eval=65536       # paths per evaluation (stable pricing)
    --c_cost=0.02              # convex exercise-cost coefficient (this cell)
    --gamma_cost=1.5          # convex-cost exponent (this cell)
    --disable_csv_logging=0
    --limit_logging_frequency=1
)
pids=()
for seed in 11 12 13; do
    python run.py "${args[@]}" -name "SwingOption_20_c0.02_gamma1.5_${seed}" -seed "$seed" &
    pids+=($!)
done
exit_code=0
for pid in "${pids[@]}"; do wait "$pid" || exit_code=1; done
exit $exit_code
