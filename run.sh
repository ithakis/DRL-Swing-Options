#!/bin/bash

#  nohup "/Users/alexanderithakis/Documents/GitHub/D4PG-QR-FRM/run.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt
args=( 
    -env="Pendulum-v1"
    -frames=30000
    -eval_every=500
    -munchausen=1
    -eval_runs=5
    -nstep=5
    -learn_every=1
    -per=1
    -iqn=0
    -w=1
    -bs=128
    -layer_size=128
    -t=5e-4
    # -d2rl=1
    --compile=1
)
python run.py "${args[@]}" -info "Baseline4_nstep2" -seed 1


# Baseline4:
# args=( 
#     -env="Pendulum-v1"
#     -frames=30000
#     -eval_every=500
#     -munchausen=1
#     -eval_runs=5
#     -nstep=5
#     -learn_every=1
#     -per=1
#     -iqn=0
#     -w=1
#     -bs=128
#     -layer_size=128
#     -t=5e-4
#     # -d2rl=1
#     --compile=1
# )
# python run.py "${args[@]}" -info "Baseline4" -seed 1