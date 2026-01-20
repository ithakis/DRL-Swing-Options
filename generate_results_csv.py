import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
import re
from multiprocessing import Pool, cpu_count

# Configuration
RUNS_DIR = "runs"
OUTPUT_DIR = "Jupyter Notebooks"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Convex Costs Results 5.2.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_run_name(run_name):
    """
    Parses directory name to extract config name and seed.
    Expected format: ConfigName_SEED
    Example: SwingOption_20_c0.01_gamma1_11
    """
    parts = run_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

def process_run(dirname):
    """
    Worker function to process a single run directory.
    Returns: (config_name, seed, stats_dict) or None
    """
    run_path = os.path.join(RUNS_DIR, dirname)
    if not os.path.isdir(run_path):
        return None

    config_name, seed = parse_run_name(dirname)
    if not config_name or not seed:
        return None
    
    # Only process if we have a valid config pattern
    if "SwingOption" not in config_name:
        return None

    try:
        ea = event_accumulator.EventAccumulator(run_path, size_guidance={
            event_accumulator.SCALARS: 0,
        })
        ea.Reload()
        
        tags = ea.Tags().get('scalars', [])
        
        # Identify tags (case-insensitive search)
        rl_tag = next((t for t in tags if 'rl' in t.lower() and 'price' in t.lower()), None)
        lsm_tag = next((t for t in tags if 'lsm' in t.lower() and 'price' in t.lower()), None)
        delta_tag = next((t for t in tags if 'delta' in t.lower() and 'percent' in t.lower()), None)

        if not rl_tag:
            return None

        rl_events = ea.Scalars(rl_tag)
        if not rl_events:
            return None
            
        # Find max RL price event
        max_rl_event = max(rl_events, key=lambda e: e.value)
        max_rl_price = max_rl_event.value
        step = max_rl_event.step
        
        # Get LSM price (usually constant, but let's try to match or get first)
        lsm_price = None
        if lsm_tag:
            lsm_events = ea.Scalars(lsm_tag)
            if lsm_events:
                lsm_price = lsm_events[0].value # Assuming constant LSM price
        
        # Get Delta Percent at that step
        delta_percent = None
        if delta_tag:
             delta_events = ea.Scalars(delta_tag)
             match = next((e for e in delta_events if e.step == step), None)
             if match:
                 delta_percent = match.value
                 
        stats = {
            "RL Price": max_rl_price,
            "LSM Price": lsm_price,
            "Delta Percent": delta_percent,
            "Best Episode": step
        }
        
        if stats:
            # Print is thread-safe enough for simple progress tracking in stdout usually
            # But with Pool it might interleave; that's fine.
            print(f"Parsed {config_name} seed {seed}: RL={max_rl_price:.4f}")
            return (config_name, seed, stats)
            
    except Exception as e:
        print(f"Error parsing {run_path}: {e}")
    
    return None

def main():
    print(f"Scanning {RUNS_DIR} with {cpu_count()} processes...")
    
    if not os.path.exists(RUNS_DIR):
        print(f"Error: {RUNS_DIR} does not exist.")
        return

    # Get list of directories
    dirs = sorted(os.listdir(RUNS_DIR))
    
    # Use multiprocessing Pool
    data = defaultdict(dict)
    
    with Pool(processes=cpu_count()) as pool:
        # Map process_run over all directories
        results = pool.map(process_run, dirs)
        
    # Aggregate results
    for res in results:
        if res:
            config_name, seed, stats = res
            data[config_name][seed] = stats
    
    # Select best seed per configuration
    final_rows = []
    
    print("\nSelecting best seed for each configuration...")
    for config_name in sorted(data.keys()):
        seeds_data = data[config_name]
        if not seeds_data:
            continue
            
        # Find seed with max RL Price
        best_seed = max(seeds_data, key=lambda s: seeds_data[s]["RL Price"])
        best_stats = seeds_data[best_seed]
        
        # Extract c and gamma from config name for sorting/columns
        # Format: SwingOption_20_cX.XX_gammaY.Y
        c_val = None
        gamma_val = None
        
        c_match = re.search(r'_c([\d\.]+)', config_name)
        gamma_match = re.search(r'_gamma([\d\.]+)', config_name)
        
        if c_match: c_val = float(c_match.group(1))
        if gamma_match: gamma_val = float(gamma_match.group(1))
        
        row = {
            "Configuration": config_name,
            "c": c_val,
            "gamma": gamma_val,
            "Best Seed": best_seed,
            "Best Episode": best_stats["Best Episode"],
            "RL Price": best_stats["RL Price"],
            "LSM Price": best_stats["LSM Price"],
            "Delta Percent": best_stats["Delta Percent"]
        }
        final_rows.append(row)
        print(f"  {config_name}: Best Seed {best_seed} (RL={best_stats['RL Price']:.4f})")

    # Create DataFrame
    df = pd.DataFrame(final_rows)
    
    # Sort nicely if possible
    df = df.sort_values(by=["c", "gamma"])
    
    # Reorder columns
    cols = ["Configuration", "c", "gamma", "Best Seed", "Best Episode", "LSM Price", "RL Price", "Delta Percent"]
    df = df[cols]
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully generated table with {len(df)} rows.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
