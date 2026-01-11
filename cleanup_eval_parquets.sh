#!/bin/bash
# Cleanup script for evaluation parquet files
# Keeps only the highest-priced RL evaluation parquet and all LSM parquets from multiple seed runs
# Usage: bash cleanup_eval_parquets.sh "Convex Cost Experiments/SwingOption_20_c0.01_gamma1.sh"

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment_script_path>"
    echo "Example: $0 'Convex Cost Experiments/SwingOption_20_c0.01_gamma1.sh'"
    exit 1
fi

SCRIPT_PATH="$1"

# Extract base experiment name from script path
# e.g., "Convex Cost Experiments/SwingOption_20_c0.01_gamma1.sh" -> "SwingOption_20_c0.01_gamma1"
BASE_NAME=$(basename "$SCRIPT_PATH" .sh)

echo ""

# Run Python helper script to find the best episode (using conda environment)
# Run Python helper script to find the best episode (using current environment)
RESULT=$(python find_best_episode.py "$BASE_NAME" 2>&1)

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to find best episode from TensorBoard logs"
    echo "$RESULT"
    exit 1
fi

# Check if result contains error messages
if echo "$RESULT" | grep -q "ERROR:"; then
    echo "$RESULT"
    exit 1
fi

# Parse the result
BEST_SEED=$(echo "$RESULT" | cut -d',' -f1)
BEST_EPISODE=$(echo "$RESULT" | cut -d',' -f2)
MAX_PRICE=$(echo "$RESULT" | cut -d',' -f3)

echo ""
echo "Best RL Price: $MAX_PRICE"
echo "Found at: Seed $BEST_SEED, Episode $BEST_EPISODE"
echo ""

# Cleanup parquet files
SEEDS="11 12 13"
TOTAL_DELETED=0
TOTAL_KEPT=0

for SEED in $SEEDS; do
    EVAL_DIR="logs/${BASE_NAME}_${SEED}/evaluations"
    
    if [ ! -d "$EVAL_DIR" ]; then
        echo "Warning: Evaluations directory not found: $EVAL_DIR"
        continue
    fi
    
    echo "Processing: $EVAL_DIR"
    
    # Count RL episode parquet files before cleanup
    RL_PARQUETS=$(find "$EVAL_DIR" -name "rl_episode_*.parquet" 2>/dev/null | wc -l | tr -d ' ')
    
    # Keep LSM parquet
    if [ -f "$EVAL_DIR/lsm.parquet" ]; then
        echo "  ✓ Keeping: lsm.parquet"
        ((TOTAL_KEPT++)) || true
    fi
    
    # Process RL episode parquets
    for PARQUET_FILE in "$EVAL_DIR"/rl_episode_*.parquet; do
        if [ ! -f "$PARQUET_FILE" ]; then
            continue
        fi
        
        FILENAME=$(basename "$PARQUET_FILE")
        
        # Check if this is the best episode from the best seed
        if [ "$SEED" -eq "$BEST_SEED" ] && [ "$FILENAME" = "rl_episode_${BEST_EPISODE}.parquet" ]; then
            echo "  ✓ Keeping: $FILENAME (best RL price)"
            ((TOTAL_KEPT++)) || true
        else
            echo "  ✗ Deleting: $FILENAME"
            rm -f "$PARQUET_FILE"
            ((TOTAL_DELETED++)) || true
        fi
    done
    
    echo ""
done

echo "=========================================="
echo "Cleanup Summary"
echo "=========================================="
echo "Files kept:    $TOTAL_KEPT"
echo "Files deleted: $TOTAL_DELETED"
echo "=========================================="
