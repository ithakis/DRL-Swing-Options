#!/bin/bash

#  nohup "./run.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt


# Baseline4 - Benchmark:
args=( 
    -env="Pendulum-v1"
    -frames=50000
    -eval_every=500
    -munchausen=1
    -eval_runs=5
    -nstep=5
    -learn_every=2
    -per=1
    -iqn=0
    -w=1
    -bs=64
    -layer_size=128
    -t=5e-3
    --compile=0
)
python run.py "${args[@]}" -info "Baseline1" -seed 11
python run.py "${args[@]}" -info "Baseline2" -seed 12
python run.py "${args[@]}" -info "Baseline3" -seed 13
python run.py "${args[@]}" -info "Baseline4" -seed 14
python run.py "${args[@]}" -info "Baseline5" -seed 15
python run.py "${args[@]}" -info "Baseline6" -seed 16
python run.py "${args[@]}" -info "Baseline7" -seed 17
python run.py "${args[@]}" -info "Baseline8" -seed 18
python run.py "${args[@]}" -info "Baseline9" -seed 19


# Episode Return = -123.790 | Frames = 30000/30000 | Frames Per Second = 537.0734
# ============================================================
# FINAL EVALUATION RESULTS
# ============================================================
# Average Episode Return: -396.093 ± 190.691
# Min Episode Return: -715.351
# Max Episode Return: -121.875
# Number of Evaluation Episodes: 10
# ============================================================

# Training Time:  00:00:55.97

