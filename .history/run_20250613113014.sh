#!/bin/bash

#  nohup "/Users/alexanderithakis/Documents/GitHub/D4PG-QR-FRM/run.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt

args=( 
    -env="Pendulum-v1"
    -frames=30000
    -eval_every=1000
    -eval_runs=1
    -nstep=1
    -learn_every=1
    -per=1
    -iqn=0
    -w=1
    -bs=128
    -layer_size=128
    -t=1e-2
    -d2rl=0
    # --jit=1
    # --info "Baseline"
)
python run.py "${args[@]}" -info "2_te-2" -seed 1
# python run.py "${args[@]}" -info "Baseline2" -seed 2
# python run.py "${args[@]}" -info "Baseline3" -seed 3
# python run.py "${args[@]}" -info "Baseline4" -seed 4

