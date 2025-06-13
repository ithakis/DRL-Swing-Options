#!/bin/bash

#  nohup "/Users/alexanderithakis/Documents/GitHub/D4PG-QR-FRM/run.sh" > .terminal_output.txt 2>&1 &
#  tail -f .terminal_output.txt

# Optimized configuration based on recommendations
args=( 
    -env="Pendulum-v1"
    -frames=30000
    -eval_every=1000
    -eval_runs=1
    -nstep=1
    -learn_every=1  # Will be automatically adjusted by worker count in agent
    -per=1
    -iqn=1
    -w=1
    -bs=128  # Now constant regardless of worker count
    -layer_size=256  # Increased from 128 for better performance
    -t=1e-3  # Changed back to 1e-3 (better for most cases than 1e-2)
    -d2rl=0
)

echo "Running optimized D4PG with fixed batch size scaling..."
python run.py "${args[@]}" -info "optimized_fixed_bs" -seed 1

# Test with 4 workers to see if parallelization now works better
echo "Testing with 4 workers (same batch size, adjusted learning frequency)..."
args_parallel=( 
    -env="Pendulum-v1"
    -frames=30000
    -eval_every=1000
    -eval_runs=1
    -nstep=1
    -learn_every=1
    -per=1
    -iqn=1
    -w=4  # 4 workers
    -bs=128  # Same batch size as w=1
    -layer_size=256
    -t=1e-3
    -d2rl=0
)
python run.py "${args_parallel[@]}" -info "optimized_fixed_bs_w4" -seed 1

echo "Experiments completed! Check tensorboard logs to compare performance."
