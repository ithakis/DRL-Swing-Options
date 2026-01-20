import os
import shutil
import pandas as pd

# Clean up targets based on analysis
# Format: Config -> {keep_seed, keep_episode, rl_price, lsm_price, delta_percent}
TARGETS = {
    "SwingOption_20_c0.04_gamma2": {
        "keep_seed": "13",
        "keep_episode": 29696,
        "rl_price": 1.9735312461853027,
        "lsm_price": 1.8481931686401367, # From typical values or I should re-fetch? Let's rely on consistency.
        # Wait, I didn't save LSM price in the previous analysis output log (Step 310 output rl_price but not explicitly LSM).
        # I should fetch it or just parse valid logs again to be safe.
        # Let's re-parse efficiently to get LSM and confirm values.
    },
    "SwingOption_20_c0.04_gamma3": {
        "keep_seed": "11",
        "keep_episode": 29696, 
        "rl_price": 1.7956602573394775
    }
}

RUNS_DIR = "runs"
LOGS_DIR = "logs"
CSV_FILE = "Jupyter Notebooks/Convex Costs Results 5.2.csv"

def get_lsm_price(run_path):
    from tensorboard.backend.event_processing import event_accumulator
    try:
        ea = event_accumulator.EventAccumulator(run_path, size_guidance={event_accumulator.SCALARS: 1})
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        lsm_tag = next((t for t in tags if 'lsm' in t.lower() and 'price' in t.lower()), None)
        if lsm_tag:
            return ea.Scalars(lsm_tag)[0].value
    except:
        return None
    return None

def main():
    print("Starting Cleanup...")
    
    # 1. Cleanup Files
    for config, info in TARGETS.items():
        keep_seed = info["keep_seed"]
        keep_episode = info["keep_episode"]
        
        # Cleanup Seeds
        for seed in ["11", "12", "13"]:
            run_name = f"{config}_{seed}"
            log_path = os.path.join(LOGS_DIR, run_name)
            
            if seed != keep_seed:
                if os.path.exists(log_path):
                    shutil.rmtree(log_path)
                    print(f"Deleted seed dir: {log_path}")
            else:
                # Cleanup Episodes in the kept seed
                eval_path = os.path.join(log_path, "evaluations")
                if os.path.exists(eval_path):
                    best_parquet = f"rl_episode_{keep_episode}.parquet"
                    files = [f for f in os.listdir(eval_path) if f.startswith('rl_episode_')]
                    for f in files:
                        if f != best_parquet:
                            os.remove(os.path.join(eval_path, f))
                    print(f"Cleaned up episodes in {eval_path}, kept {best_parquet}")
    
    print("\nAppending to CSV...")
    if not os.path.exists(CSV_FILE):
        print("Error: CSV file not found!")
        return

    df = pd.read_csv(CSV_FILE)
    
    new_rows = []
    for config, info in TARGETS.items():
        seed = info["keep_seed"]
        # Fetch LSM price fresh to be accurate
        run_path = os.path.join(RUNS_DIR, f"{config}_{seed}")
        lsm_val = get_lsm_price(run_path)
        
        # Extract c and gamma
        # SwingOption_20_c0.04_gamma2
        c_val = 0.04
        gamma_val = float(config.split("gamma")[1])
        
        # RL Price from analysis (or could re-fetch, but using trusted value)
        rl_val = info["rl_price"]
        
        # Delta Percent calculation to be consistent with table
        # (RL - LSM) / LSM ? No, the table has 'PctDiff' or 'Delta Percent'.
        # Previous table col was 'Delta Percent'.
        # Let's calculate it or fetch it? 
        # I'll fetch the delta tag value just to be consistent with how the table was built (from logs).
        
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(run_path, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        delta_tag = next((t for t in tags if 'delta' in t.lower() and 'percent' in t.lower()), None)
        delta_val = 0.0
        if delta_tag:
             events = ea.Scalars(delta_tag)
             # Find event at keep_episode
             # Note: step might not match exactly if there is some offset, but usually does for evaluation.
             # Actually, best_episode was derived from best delta/RL event, so it should exist.
             match = next((e for e in events if int(e.step) == info["keep_episode"]), None)
             if match: delta_val = match.value
        
        row = {
            "Configuration": config,
            "c": c_val,
            "gamma": gamma_val,
            "Best Seed": int(seed),
            "Best Episode": info["keep_episode"],
            "LSM Price": lsm_val,
            "RL Price": rl_val,
            "PctDiff": delta_val # User renamed column to PctDiff in last manual edit!
        }
        new_rows.append(row)
        print(f"Prepared row for {config}: RL={rl_val:.4f}, Delta={delta_val:.4f}")

    # Remove existing rows for these configs if any (to avoid duplicates)
    df = df[~df['Configuration'].isin(TARGETS.keys())]
    
    # Check column name compatibility
    # The user manually edited column 'Delta Percent' to 'PctDiff'.
    # I should respect the current file structure.
    
    # Append
    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Sort
    df = df.sort_values(by=["c", "gamma"])
    
    df.to_csv(CSV_FILE, index=False)
    print("CSV updated and saved.")

if __name__ == "__main__":
    main()
