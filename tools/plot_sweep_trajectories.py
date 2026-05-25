"""
Quick text-mode trajectory comparison for the sweep runs.

Extracts Average100 series from each per-config log file and prints a
side-by-side summary at fixed checkpoints (every 10% of training horizon)
so we can eyeball convergence shape without matplotlib.

Usage:
    python tools/plot_sweep_trajectories.py [--suffix _wide2]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "_sweep_h1"


PARSE_AVG100 = re.compile(r"Path\s+(\d+)/\d+.*?Average100\s*=\s*([\d.\-eE]+)")


def load_trajectory(path: Path) -> List[float]:
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    out: List[float] = []
    for m in PARSE_AVG100.finditer(text):
        out.append(float(m.group(2)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", type=str, default="",
                   help="Log filename suffix (e.g. '_wide2', '_nocost').")
    p.add_argument("--checkpoints", type=int, default=10,
                   help="Number of equally-spaced trajectory checkpoints to print.")
    args = p.parse_args()

    suffix = args.suffix
    logs = sorted(LOG_DIR.glob(f"*{suffix}_s*.log"))
    if not logs:
        print(f"No logs matching *{suffix}_s*.log in {LOG_DIR}")
        return

    trajs: Dict[str, List[float]] = {}
    for lf in logs:
        # strip "_s11" and ".log"
        name = lf.stem.replace("_s11", "")
        trajs[name] = load_trajectory(lf)

    # Filter to ones with non-trivial trajectories
    trajs = {k: v for k, v in trajs.items() if len(v) > 100}
    if not trajs:
        print("No populated trajectories yet.")
        return

    max_len = max(len(v) for v in trajs.values())
    checkpoints = [int((i + 1) / args.checkpoints * max_len) for i in range(args.checkpoints)]

    # Header
    width = max(len(n) for n in trajs)
    print(f"{'config':<{width}}  ", end="")
    for c in checkpoints:
        print(f"@{c:>5}  ", end="")
    print()
    print("-" * (width + 2 + len(checkpoints) * 9))

    # Sort by final Average100 descending
    items = sorted(trajs.items(), key=lambda kv: -(kv[1][-1] if kv[1] else 0))
    for name, traj in items:
        print(f"{name:<{width}}  ", end="")
        for c in checkpoints:
            idx = min(c - 1, len(traj) - 1)
            if idx >= 0:
                print(f"{traj[idx]:>6.3f}  ", end="")
            else:
                print(f"{'-':>6}  ", end="")
        print()


if __name__ == "__main__":
    main()
