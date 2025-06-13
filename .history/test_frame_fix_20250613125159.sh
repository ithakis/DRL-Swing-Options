#!/bin/bash

# Test script to verify the frame counting fix
echo "=== Testing Frame Counting Fix ==="

# Test with 1 worker (should run 5000 frames)
echo "Testing w=1 with 5000 frames..."
python run.py -env="Pendulum-v1" -frames=5000 -eval_every=1000 -eval_runs=1 -nstep=1 -learn_every=1 -per=1 -iqn=1 -w=1 -bs=128 -layer_size=256 -t=1e-3 -d2rl=0 -info "frame_test_w1" -seed 1

echo ""
echo "Testing w=4 with 5000 frames..."
python run.py -env="Pendulum-v1" -frames=5000 -eval_every=1000 -eval_runs=1 -nstep=1 -learn_every=1 -per=1 -iqn=1 -w=4 -bs=128 -layer_size=256 -t=1e-3 -d2rl=0 -info "frame_test_w4" -seed 1

echo ""
echo "=== Frame counting test completed ==="
echo "Both should show the same total frame count at the end!"
