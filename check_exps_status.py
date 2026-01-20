
import os
import re

# List of scripts from conv_cost_exps.sh
scripts = [
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma1.5.sh", 
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma1.sh",   
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma2.sh",   
    "Convex Cost Experiments/SwingOption_20_c0.01_gamma3.sh",   
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma1.5.sh", 
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma1.sh",     
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma2.sh",     
    "Convex Cost Experiments/SwingOption_20_c0.02_gamma3.sh",   
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma1.5.sh", 
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma1.sh",   
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma2.sh",     
    "Convex Cost Experiments/SwingOption_20_c0.04_gamma3.sh",     
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma1.5.sh", 
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma1.sh",     
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma2.sh",     
    "Convex Cost Experiments/SwingOption_20_c0.05_gamma3.sh",     
    "Convex Cost Experiments/SwingOption_20_c0.08_gamma1.5.sh",   
    "Convex Cost Experiments/SwingOption_20_c0.08_gamma1.sh",     
    "Convex Cost Experiments/SwingOption_20_c0.08_gamma2.sh",   
    "Convex Cost Experiments/SwingOption_20_c0.10_gamma1.5.sh", 
    "Convex Cost Experiments/SwingOption_20_c0.10_gamma1.sh",   
    "Convex Cost Experiments/SwingOption_20_c0.10_gamma2.sh",   
    "Convex Cost Experiments/SwingOption_20_c0.15_gamma1.5.sh", 
    "Convex Cost Experiments/SwingOption_20_c0.15_gamma1.sh",   
    "Convex Cost Experiments/SwingOption_20_c0.15_gamma2.sh",   
]

seeds = [11, 12, 13]
base_log_dir = "logs"

for script_path in scripts:
    # Extract c and gamma from filename
    # Filename format: SwingOption_20_c{c}_gamma{gamma}.sh
    basename = os.path.basename(script_path)
    match = re.match(r"SwingOption_20_c([\d.]+)_gamma([\d.]+)\.sh", basename)
    if not match:
        print(f"Skipping {script_path}: Pattern not matched")
        continue

    c_val = match.group(1)
    gamma_val = match.group(2)
    
    # Check logs for all seeds
    all_seeds_done = True
    for seed in seeds:
        log_dir_name = f"SwingOption_20_c{c_val}_gamma{gamma_val}_{seed}"
        eval_dir = os.path.join(base_log_dir, log_dir_name, "evaluations")
        
        has_lsm = False
        has_rl = False
        
        if os.path.exists(eval_dir):
            files = os.listdir(eval_dir)
            if "lsm.parquet" in files:
                has_lsm = True
            for f in files:
                if f.startswith("rl_episode") and f.endswith(".parquet"):
                    has_rl = True
                    break
        
        if not (has_lsm or has_rl):
            all_seeds_done = False
            break
            
    if all_seeds_done:
        print(f"DONE: {script_path}")
    else:
        print(f"TODO: {script_path}")
