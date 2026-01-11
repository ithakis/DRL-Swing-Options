
import os
import re
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
from multiprocessing import Pool, cpu_count

# Regex to parse folder name
dirname_pattern = re.compile(r"SwingOption_20_c([\d\.]+)_gamma([\d\.]+)_(\d+)")

def parse_run(full_path):
    try:
        # Find tfevents file
        try:
            event_files = [f for f in os.listdir(full_path) if "events.out.tfevents" in f]
        except FileNotFoundError:
            return None
            
        if not event_files:
            return None
        
        event_file = max(event_files, key=lambda f: os.path.getsize(os.path.join(full_path, f)))
        path = os.path.join(full_path, event_file)
        
        ea = EventAccumulator(path, size_guidance={
            'compressedHistograms': 0,
            'images': 0,
            'audio': 0,
            'scalars': 0, 
            'histograms': 0,
        })
        ea.Reload()
        
        tags = ea.Tags()['scalars']
        if 'Pricing/Delta_Percent' not in tags:
            return None
            
        delta_pcts = pd.DataFrame(ea.Scalars('Pricing/Delta_Percent'))
        if delta_pcts.empty:
            return None
            
        delta_pcts = delta_pcts.rename(columns={'value': 'delta_pct', 'step': 'step'})
        
        best_idx = delta_pcts['delta_pct'].idxmax()
        best_row = delta_pcts.loc[best_idx]
        best_step = best_row['step']
        best_delta = best_row['delta_pct']
        
        def get_value_at_step(tag_name, step):
            if tag_name in tags:
                data = pd.DataFrame(ea.Scalars(tag_name))
                row = data[data['step'] == step]
                if not row.empty:
                    return row.iloc[0]['value']
            return None

        rl_price = get_value_at_step('Pricing/RL_Price', best_step)
        lsm_price = get_value_at_step('Pricing/LSM_Price', best_step)

        # Parse folder name again here or pass it?
        # Better to parse from path
        folder_name = os.path.basename(full_path)
        match = dirname_pattern.search(folder_name)
        if match:
            return {
                'c': float(match.group(1)),
                'gamma': float(match.group(2)),
                'ID': int(match.group(3)),
                'LSM': lsm_price,
                'Best RL': rl_price,
                'PctDiff': best_delta
            }
    except Exception as e:
        # print(f"Error parsing {full_path}: {e}")
        return None
    return None

def main():
    runs_dir = "runs"
    tasks = []
    
    sys.stderr.write("Enumerating runs...\n")
    
    for item in os.listdir(runs_dir):
        full_path = os.path.join(runs_dir, item)
        if not os.path.isdir(full_path):
            continue
        if dirname_pattern.search(item):
            tasks.append(full_path)
    
    sys.stderr.write(f"Found {len(tasks)} runs. Processing with {cpu_count()} cores...\n")
    
    results = []
    with Pool(processes=cpu_count()) as pool:
        for result in pool.imap_unordered(parse_run, tasks):
            if result:
                results.append(result)
                # Simple progress
                sys.stderr.write(".")
                sys.stderr.flush()
    
    sys.stderr.write("\nProcessing complete.\n")
    
    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by=['c', 'gamma'])
    
    # Select best seed per (c, gamma)
    best_df = df.loc[df.groupby(['c', 'gamma'])['PctDiff'].idxmax()]
    
    output_df = best_df[['c', 'gamma', 'ID', 'LSM', 'Best RL', 'PctDiff']]
    
    # Save to file
    output_csv_path = "results_table.csv"
    output_df.to_csv(output_csv_path, index=False)
    sys.stderr.write(f"Results saved to {output_csv_path}\n")
    
    # Also print to stdout
    print(output_df.to_csv(index=False))

if __name__ == "__main__":
    main()
