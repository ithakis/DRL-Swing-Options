#!/usr/bin/env python3
"""
Quick verification that frame counting fix is working correctly.
This script analyzes the training logs to verify both w=1 and w=4 run the same number of steps.
"""

import json
import os
from pathlib import Path

def check_frame_counts():
    """Check if the frame counting fix is working by examining run logs."""
    
    print("=== Frame Count Verification ===\n")
    
    # Check different run configurations
    runs_to_check = [
        ("optimized_fixed_bs", "w=1, fixed batch size"),
        ("optimized_fixed_bs_w4", "w=4, fixed batch size"), 
        ("1_128_w1", "Original w=1 (if exists)"),
        ("frame_test_w1", "Test w=1"),
        ("frame_test_w4", "Test w=4"),
    ]
    
    results = {}
    
    for run_name, description in runs_to_check:
        json_path = f"runs/{run_name}.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract frame count and timing info
                total_frames = data.get('frames', 'Unknown')
                worker_count = data.get('worker', 'Unknown') 
                batch_size = data.get('batch_size', 'Unknown')
                training_time = data.get('training_time', 'Unknown')
                
                results[run_name] = {
                    'description': description,
                    'frames': total_frames,
                    'workers': worker_count,
                    'batch_size': batch_size,
                    'time': training_time
                }
                
                print(f"✓ {description}")
                print(f"  Frames: {total_frames}")
                print(f"  Workers: {worker_count}")  
                print(f"  Batch Size: {batch_size}")
                print(f"  Training Time: {training_time}")
                print()
                
            except Exception as e:
                print(f"✗ {description}: Error reading {json_path} - {e}")
        else:
            print(f"? {description}: No data file found ({json_path})")
    
    # Analysis
    if len(results) >= 2:
        print("=== Analysis ===")
        w1_runs = [r for r in results.values() if r.get('workers') == 1]
        w4_runs = [r for r in results.values() if r.get('workers') == 4]
        
        if w1_runs and w4_runs:
            w1_frames = w1_runs[0]['frames']
            w4_frames = w4_runs[0]['frames'] 
            
            if w1_frames == w4_frames:
                print(f"✅ SUCCESS: Both w=1 and w=4 ran {w1_frames} frames!")
                print("   Frame counting fix is working correctly.")
            else:
                print(f"❌ ISSUE: w=1 ran {w1_frames} frames, w=4 ran {w4_frames} frames")
                print("   Frame counting still needs fixing.")
        else:
            print("Need both w=1 and w=4 runs to compare.")
    else:
        print("Not enough runs to analyze. Run some experiments first.")

if __name__ == "__main__":
    check_frame_counts()
