import os
import shutil
import glob

LOGS_DIR = "logs"

def cleanup():
    if not os.path.exists(LOGS_DIR):
        print("Logs directory not found.")
        return

    deleted_count = 0
    kept_count = 0

    print(f"{'Directory':<60} | {'Status':<10}")
    print("-" * 80)

    for item in os.listdir(LOGS_DIR):
        full_path = os.path.join(LOGS_DIR, item)
        
        if not os.path.isdir(full_path):
            continue

        # Check for rl_episode*.parquet in root
        root_rl = glob.glob(os.path.join(full_path, "rl_episode*.parquet"))
        
        # Check for rl_episode*.parquet in evaluations
        eval_rl = glob.glob(os.path.join(full_path, "evaluations", "rl_episode*.parquet"))
        
        if not root_rl and not eval_rl:
            print(f"{item:<60} | DELETING")
            shutil.rmtree(full_path)
            deleted_count += 1
        else:
            # print(f"{item:<60} | KEEPING") # verbose
            kept_count += 1

    print("-" * 80)
    print(f"Cleanup complete.")
    print(f"Deleted: {deleted_count}")
    print(f"Kept:    {kept_count}")

if __name__ == "__main__":
    cleanup()
