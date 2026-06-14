"""Run the v65 C++ hedge export (hedge_swing) for the Hedging notebook's RL variants.

Produces one blob per variant under logs/hedging_v65/:
  RL-kernel.bin  — v65 kernel-on
  RL.bin         — v65 kernel-off (single-sample TD)

Each blob holds the PV-vs-S0 grid + the daily-hedge arrays (S/X/Y/cf/q_before/pv/Vp/Vm); the
Hedging notebook reconstructs Delta/Gamma + hedge P&L via tools/cpp_hedge.py (src/greeks.py math).
Focal contract c=0.04, gamma=2; common CRN seed 999.

Run:  EP11python tools/gen_hedging_v65.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "cpp_pricer" / "build_v65" / "hedge_swing"
KERNEL = ROOT / "cpp_pricer" / "data" / "kernel_v64.bin"
OUT = ROOT / "logs" / "hedging_v65"

SEED = int(os.environ.get("HEDGE_SEED", 999))
N_TRAIN = int(os.environ.get("HEDGE_NTRAIN", 4096))
N_RL = int(os.environ.get("HEDGE_NRL", 8192))      # paths for the PV-vs-S0 grid
N_HEDGE = int(os.environ.get("HEDGE_NHEDGE", 4096)) # paths for the daily hedge (O(N^2) re-rolls)
THREADS = int(os.environ.get("HEDGE_THREADS", "8"))
C_COST, GAMMA = 0.04, 2.0
V65 = ["--hidden", "48", "--actor_layers", "2", "--critic_layers", "4", "--hidden_actor", "32",
       "--batch", "64", "--learn_number", "3", "--lr_c", "5e-4"]


def run(label, kernel_on):
    out = OUT / f"{label}.bin"
    cmd = [str(CPP), "--seed", str(SEED), "--n_train", str(N_TRAIN), "--n_rl", str(N_RL),
           "--n_hedge", str(N_HEDGE), "--threads", str(THREADS), "--c_cost", str(C_COST),
           "--gamma_cost", str(GAMMA), "--out", str(out), *V65]
    cmd += (["--kernel", str(KERNEL)] if kernel_on else ["--kernel_off"])
    print(f"[{label}] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT / "cpp_pricer"))
    if r.returncode != 0:
        raise RuntimeError(f"hedge_swing failed: {r.stderr[-800:]}")
    print(f"[{label}] {r.stdout.strip()} -> {out}")


def main():
    assert CPP.exists(), f"build hedge_swing first: {CPP}"
    OUT.mkdir(parents=True, exist_ok=True)
    run("RL-kernel", kernel_on=True)
    run("RL", kernel_on=False)
    print("done.")


if __name__ == "__main__":
    main()
