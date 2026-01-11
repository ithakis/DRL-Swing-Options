#!/usr/bin/env python3
"""
Helper script to find the best episode from TensorBoard logs.
This script parses TensorBoard event files to find the episode with the highest RL price.
"""
import sys
import os

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: TensorBoard is not installed. Please install it with: pip install tensorboard", file=sys.stderr)
    sys.exit(1)

def find_best_episode(base_name):
    """Find the seed and episode with the highest RL price from TensorBoard logs."""
    seeds = [11, 12, 13]
    max_price = -float('inf')
    best_seed = None
    best_episode = None
    
    for seed in seeds:
        tb_dir = f"runs/{base_name}_{seed}"
        if not os.path.exists(tb_dir):
            print(f"Warning: TensorBoard directory not found: {tb_dir}", file=sys.stderr)
            continue
        
        try:
            ea = event_accumulator.EventAccumulator(tb_dir)
            ea.Reload()
            
            # Check if the Pricing/RL_Price metric exists
            if 'scalars' in ea.Tags() and 'Pricing/RL_Price' in ea.Tags()['scalars']:
                prices = ea.Scalars('Pricing/RL_Price')
                for event in prices:
                    if event.value > max_price:
                        max_price = event.value
                        best_seed = seed
                        best_episode = event.step
            else:
                print(f"Warning: No 'Pricing/RL_Price' metric found in {tb_dir}", file=sys.stderr)
        except Exception as e:
            print(f"Error reading TensorBoard logs from {tb_dir}: {e}", file=sys.stderr)
            continue
    
    if best_seed is None or best_episode is None:
        print("ERROR: Could not find any valid RL price data in TensorBoard logs", file=sys.stderr)
        return None, None, None
    
    return best_seed, best_episode, max_price

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_best_episode.py <base_name>", file=sys.stderr)
        sys.exit(1)
    
    base_name = sys.argv[1]
    best_seed, best_episode, max_price = find_best_episode(base_name)
    
    if best_seed is not None:
        # Print results in a parseable format: seed,episode,price
        print(f"{best_seed},{best_episode},{max_price}")
    else:
        sys.exit(1)
