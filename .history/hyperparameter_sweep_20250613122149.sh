#!/bin/bash

# Hyperparameter sweep script for optimized D4PG
# Tests different configurations to find optimal settings

echo "=== D4PG Hyperparameter Sweep ==="
echo "This will test different configurations to find optimal settings"

# Base configuration
base_args=(
    -env="Pendulum-v1"
    -frames=15000  # Shorter for sweep
    -eval_every=1000
    -eval_runs=1
    -nstep=1
    -learn_every=1
    -per=1
    -iqn=1
    -w=1
    -bs=128
    -layer_size=256
    -d2rl=0
)

# Test different tau values
echo "Testing different tau values..."
for tau in 1e-4 5e-4 1e-3 5e-3 1e-2; do
    echo "Testing tau=$tau"
    python run.py "${base_args[@]}" -t=$tau -info "sweep_tau_$tau" -seed 1
done

# Test different learning rates
echo "Testing different learning rates..."
for lr in 1e-4 3e-4 1e-3; do
    echo "Testing learning rate=$lr"
    python run.py "${base_args[@]}" -t=1e-3 -lr_a=$lr -lr_c=$lr -info "sweep_lr_$lr" -seed 1
done

# Test different worker configurations
echo "Testing worker configurations..."
for workers in 1 2 4; do
    echo "Testing workers=$workers"
    python run.py "${base_args[@]}" -t=1e-3 -w=$workers -info "sweep_workers_$workers" -seed 1
done

# Test D2RL architecture
echo "Testing D2RL architecture..."
python run.py "${base_args[@]}" -t=1e-3 -d2rl=1 -info "sweep_d2rl" -seed 1

echo "=== Hyperparameter sweep completed ==="
echo "Run 'python monitor_performance.py' to analyze results"
