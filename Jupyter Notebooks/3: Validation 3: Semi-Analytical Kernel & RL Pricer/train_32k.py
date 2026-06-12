"""Train kernel-on (fast K_M6: M_x=2, M_per_k=2, N_max=1) focal-g2 agents at
n_paths=32768, replicating EXACTLY the base_args used for the 512..16384 R2 points
(tools/sweep_expected_target.base_args + kernel_overrides), so discover_fingerprint
in gen_rl_validation.py picks them up by hyperparameter and they extend the R2 curve
to the same 32768 horizon as kernel-off.

Usage:
    python train_32k.py 11            # single seed (times it)
    python train_32k.py 11 12 13 ...  # several seeds, sequential
"""
from __future__ import annotations
import shlex, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sweep_expected_target import base_args  # noqa: E402

N_PATHS = 32768
KERNEL = {"--use_expected_target": "1", "--critic_warmup_episodes": "0",
          "--kernel_M_x": "2", "--kernel_M_per_k": "2", "--kernel_N_max": "1"}
LOGDIR = Path(__file__).resolve().parent / "_train32k_logs"
LOGDIR.mkdir(exist_ok=True)


def train(seed: int) -> float:
    args = base_args(n_paths=N_PATHS, contract="focal")
    args.update(KERNEL)
    name = f"_sw_h1_kbc_CV_K_M6_N{N_PATHS}_s{seed}"
    args["-name"] = name
    args["-seed"] = str(seed)
    cmd = ["python", "run.py"]
    for k, v in args.items():
        cmd.extend([k, v])
    log = LOGDIR / f"{name}.log"
    t0 = time.time()
    print(f"[s{seed}] launching -> {log}")
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    wall = time.time() - t0
    print(f"[s{seed}] done in {wall:.0f}s status={r.returncode}")
    return wall


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [11]
    for s in seeds:
        train(s)
