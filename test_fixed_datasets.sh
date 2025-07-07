#!/bin/bash

# Test script with minimal parameters
args=( 
    # Training parameters
    -n_paths=10 
    -eval_every=5
    -n_paths_eval=5
    -seed=42
    -name="test_fixed_datasets"
    
    # Minimal contract parameters for quick testing
    --maturity=0.1 
    --n_rights=5 
    --q_max=1.0 
    --Q_max=3.0
    --strike=100.0
    -lr_a=1e-3
    -lr_c=1e-3
    -bs=32
)

echo "Testing fixed datasets approach..."
python run.py "${args[@]}"
