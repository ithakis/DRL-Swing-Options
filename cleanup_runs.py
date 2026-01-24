import os
import shutil

RUNS_DIR = "runs"

# List of experiment IDs to delete (based on previous logs cleanup)
TARGETS = [
    # c0.02
    "SwingOption_20_c0.02_gamma1_11", "SwingOption_20_c0.02_gamma1_12", "SwingOption_20_c0.02_gamma1_13",
    "SwingOption_20_c0.02_gamma2_11", "SwingOption_20_c0.02_gamma2_12", "SwingOption_20_c0.02_gamma2_13",
    # c0.05
    "SwingOption_20_c0.05_gamma1_11", "SwingOption_20_c0.05_gamma1_12", "SwingOption_20_c0.05_gamma1_13",
    "SwingOption_20_c0.05_gamma2_11", "SwingOption_20_c0.05_gamma2_12", "SwingOption_20_c0.05_gamma2_13",
    "SwingOption_20_c0.05_gamma3_11", "SwingOption_20_c0.05_gamma3_12", "SwingOption_20_c0.05_gamma3_13",
    # c0.08
    "SwingOption_20_c0.08_gamma1.5_11", "SwingOption_20_c0.08_gamma1.5_12", "SwingOption_20_c0.08_gamma1.5_13",
    "SwingOption_20_c0.08_gamma1_11", "SwingOption_20_c0.08_gamma1_12", "SwingOption_20_c0.08_gamma1_13",
]

def cleanup_runs():
    if not os.path.exists(RUNS_DIR):
        print("Runs directory not found.")
        return

    deleted_count = 0
    print(f"{'Item':<60} | {'Status':<10}")
    print("-" * 80)

    for target in TARGETS:
        # 1. Check for Directory
        dir_path = os.path.join(RUNS_DIR, target)
        if os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"{target + ' (DIR)':<60} | DELETED")
                deleted_count += 1
            except Exception as e:
                print(f"{target + ' (DIR)':<60} | ERROR: {e}")

        # 2. Check for JSON file
        json_path = os.path.join(RUNS_DIR, target + ".json")
        if os.path.isfile(json_path):
            try:
                os.remove(json_path)
                print(f"{target + '.json':<60} | DELETED")
                deleted_count += 1
            except Exception as e:
                print(f"{target + '.json':<60} | ERROR: {e}")

        # 3. Check for PTH file
        pth_path = os.path.join(RUNS_DIR, target + ".pth")
        if os.path.isfile(pth_path):
            try:
                os.remove(pth_path)
                print(f"{target + '.pth':<60} | DELETED")
                deleted_count += 1
            except Exception as e:
                print(f"{target + '.pth':<60} | ERROR: {e}")

    print("-" * 80)
    print(f"Cleanup of runs complete.")
    print(f"Total items deleted: {deleted_count}")

if __name__ == "__main__":
    cleanup_runs()
