#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment_script.sh>"
    exit 1
fi

exp_script="$1"
if [ ! -f "$exp_script" ]; then
    echo "Error: experiment script not found: $exp_script"
    exit 1
fi

python - "$exp_script" <<'PY'
import math
import re
import sys
from pathlib import Path

import pandas as pd
exp_script = Path(sys.argv[1]).resolve()
repo_root = exp_script.parents[1]
script_text = exp_script.read_text()

def _parse_arg(name: str, default: str | None = None) -> str:
    pattern = rf"--{re.escape(name)}=([0-9.eE+-]+)"
    match = re.search(pattern, script_text)
    if match:
        return match.group(1)
    if default is not None:
        return default
    raise SystemExit(f"Missing required arg --{name} in {exp_script}")

run_names = []
for line in script_text.splitlines():
    if "python run.py" in line and "-name" in line:
        match = re.search(r'-name\s+"([^"]+)"', line)
        if match:
            run_names.append(match.group(1))

if not run_names:
    raise SystemExit(f"No run names found in {exp_script}")

run_names = sorted(set(run_names))

risk_free_rate = float(_parse_arg("risk_free_rate"))
maturity = float(_parse_arg("maturity"))
n_rights = int(float(_parse_arg("n_rights")))
if n_rights < 2:
    raise SystemExit("n_rights must be >= 2 to compute discount factor")
dt = maturity / (n_rights - 1)
df = math.exp(-risk_free_rate * dt)

def compute_lsm_price(lsm_path: Path) -> float | None:
    if not lsm_path.exists():
        return None
    try:
        frame = pd.read_parquet(lsm_path, columns=["path", "time_step", "payoff"])
    except Exception:
        return None
    if frame.empty:
        return None
    paths = frame["path"].to_numpy()
    max_path = int(paths.max())
    time_steps = frame["time_step"].to_numpy()
    payoffs = frame["payoff"].to_numpy()
    total = (payoffs * (df ** time_steps)).sum()
    return total / (max_path + 1)

def compute_rl_price(rl_path: Path) -> float | None:
    try:
        frame = pd.read_parquet(rl_path, columns=["path", "reward"])
    except Exception:
        return None
    if frame.empty:
        return None
    total = frame["reward"].sum()
    max_path = int(frame["path"].max())
    return total / (max_path + 1)

def delta_percent(rl_price: float, lsm_price: float) -> float:
    if lsm_price == 0.0:
        if rl_price == 0.0:
            return 0.0
        return math.inf if rl_price > 0 else -math.inf
    return (rl_price - lsm_price) / lsm_price * 100.0

candidates = []
eligible_eval_dirs = []
for run_name in run_names:
    eval_dir = repo_root / "logs" / run_name / "evaluations"
    lsm_price = compute_lsm_price(eval_dir / "lsm.parquet")
    if lsm_price is None:
        continue
    eligible_eval_dirs.append(eval_dir)
    for rl_parquet in sorted(eval_dir.glob("rl_episode_*.parquet")):
        rl_price = compute_rl_price(rl_parquet)
        if rl_price is None:
            continue
        candidates.append((delta_percent(rl_price, lsm_price), rl_parquet))

if not candidates:
    print("No evaluation Parquet files found to rank; skipping cleanup.")
    sys.exit(0)

candidates.sort(key=lambda x: x[0], reverse=True)
keep = {path for _, path in candidates[:3]}

removed = 0
kept = 0
for eval_dir in eligible_eval_dirs:
    if not eval_dir.exists():
        continue
    for rl_parquet in eval_dir.glob("rl_episode_*.parquet"):
        if rl_parquet in keep:
            kept += 1
            continue
        try:
            rl_parquet.unlink()
            removed += 1
        except Exception:
            continue

print(f"Kept {kept} eval Parquet files (top 3 by Delta percent); removed {removed}.")
PY
