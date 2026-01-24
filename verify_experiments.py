import os
import re

LOGS_DIR = "logs"
SCRIPT_FILE = "conv_cost_exps.sh"

def get_expected_configs():
    configs = []
    with open(SCRIPT_FILE, 'r') as f:
        content = f.read()
        # Extract script paths from the array
        matches = re.findall(r'"Convex Cost Experiments/(SwingOption_.*)\.sh"', content)
        configs = matches
    return configs

def verify_configs(configs):
    results = {}
    
    for config in configs:
        results[config] = {
            "seeds_found": [],
            "valid_seeds": [],
            "status": "MISSING"
        }
        
        # Find all directories in logs that start with this config name
        # The regex should match config name followed by _ and digits (seed)
        # e.g. SwingOption_20_c0.01_gamma1.5_11
        pattern = re.compile(rf"^{re.escape(config)}_(\d+)$")
        
        if not os.path.exists(LOGS_DIR):
            print(f"Logs directory {LOGS_DIR} not found.")
            return

        for item in os.listdir(LOGS_DIR):
            match = pattern.match(item)
            if match:
                seed = match.group(1)
                full_path = os.path.join(LOGS_DIR, item)
                if os.path.isdir(full_path):
                    results[config]["seeds_found"].append(seed)
                    
                    lsm_path = os.path.join(full_path, "lsm.parquet")
                
                # Check root for rl_episode*.parquet
                root_rl_files = [f for f in os.listdir(full_path) if f.startswith("rl_episode") and f.endswith(".parquet")]
                root_valid = os.path.exists(lsm_path) and len(root_rl_files) > 0
                
                # Check evaluations subdir
                eval_path = os.path.join(full_path, "evaluations")
                if os.path.exists(eval_path) and os.path.isdir(eval_path):
                    eval_lsm_path = os.path.join(eval_path, "lsm.parquet")
                    eval_rl_files = [f for f in os.listdir(eval_path) if f.startswith("rl_episode") and f.endswith(".parquet")]
                    eval_valid = os.path.exists(eval_lsm_path) and len(eval_rl_files) > 0
                else:
                    eval_valid = False

                if root_valid or eval_valid:
                    results[config]["valid_seeds"].append(seed)

        if results[config]["valid_seeds"]:
            results[config]["status"] = "COMPLETE"
        elif results[config]["seeds_found"]:
            results[config]["status"] = "INCOMPLETE" # Seeds exist but missing files
        else:
            results[config]["status"] = "NO_DATA"

    return results

def main():
    configs = get_expected_configs()
    results = verify_configs(configs)
    
    print(f"{'Config':<50} | {'Status':<12} | {'Seeds Found':<15} | {'Valid Seeds':<15}")
    print("-" * 100)
    
    missing_rl_file = []
    complete = []
    
    for config, data in results.items():
        seeds_str = ",".join(data["seeds_found"])
        valid_str = ",".join(data["valid_seeds"])
        print(f"{config:<50} | {data['status']:<12} | {seeds_str:<15} | {valid_str:<15}")
        
        if data["status"] != "COMPLETE":
            missing_rl_file.append(config)
        else:
            complete.append(config)
            
    print("\n\nConfigs to RERUN (Missing valid results):")
    for c in missing_rl_file:
        print(c)
        
    print("\nConfigs COMPLETE (Have valid results):")
    for c in complete:
        print(c)

if __name__ == "__main__":
    main()
