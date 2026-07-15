"""Run the v67 C++ hedge export (hedge_swing) for the Hedging notebook's RL variants.

Mirrors tools/gen_hedging_v65.py but uses the v67 two-mode builds and the Table-5 recipes, and
trains with seed 11 (so the policies genuinely are the "seed-11 focal checkpoints" the paper claims):

  RL-kernel.bin  -> build_v67_kernel/hedge_swing, KON_BALANCED (a2c4/w48/actor32/b64/ln3/lr_c5e-4),
                    kernel-on, n_train=4096  == Table 5 AC-kernel (EXACT recipe).
  RL.bin         -> build_v67_nokernel/hedge_swing, --kernel_off, a2c2/w64, n_train=32768 == Table 5
                    AC-sample budget/architecture. NB: hedge_swing can't take the no-kernel runtime
                    flags (TPN/cosine-LR/double-critic), so this is the closest no-kernel the harness
                    allows; we VERIFY its PV vs Table 5's 1.9592 and escalate only if materially off.

Focal contract c=0.04, gamma=2. Run:  EP11python tools/gen_hedging_v67.py
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KBIN = ROOT / "cpp_pricer" / "build_v67_kernel" / "hedge_swing"
NBIN = ROOT / "cpp_pricer" / "build_v67_nokernel" / "hedge_swing"
KERNEL = ROOT / "cpp_pricer" / "data" / "kernel_v64.bin"
OUT = ROOT / "logs" / "hedging_v67"

SEED = int(os.environ.get("HEDGE_SEED", 11))      # seed-11 focal checkpoints (matches the paper)
N_RL = int(os.environ.get("HEDGE_NRL", 8192))     # PV-vs-S0 grid paths
N_HEDGE = int(os.environ.get("HEDGE_NHEDGE", 4096))
THREADS = int(os.environ.get("HEDGE_THREADS", "8"))
C_COST, GAMMA = 0.04, 2.0
KON = ["--hidden", "48", "--actor_layers", "2", "--critic_layers", "4", "--hidden_actor", "32",
       "--batch", "64", "--learn_number", "3", "--lr_c", "5e-4"]           # == KON_BALANCED
# Full v67 no-kernel recipe == gen_rl_validation_v67.koff_flags(32768), so the hedge AC-sample is the
# SAME policy as Table 5's AC-sample. Budget-dependent values hardwired for n_train=32768:
# noise_plateau=32768//10, lr_warmup=32768//32, tpn_decay_start=32768//2.
KOFF = ["--hidden", "64", "--hidden_actor", "64", "--actor_layers", "2", "--critic_layers", "2",
        "--init_method", "1", "--batch", "128", "--learn_every", "2", "--learn_number", "1",
        "--lr_a", "1.6e-4", "--lr_c", "9e-5", "--wd_c", "1.2e-4", "--tau", "0.0032",
        "--noise_sigma0", "1.30", "--noise_floor", "0.26",
        "--noise_schedule", "hyperbolic", "--noise_plateau", "3276",
        "--adaptive_noise_scale", "0.6", "--warmup_noise_fraction", "0.4",
        "--critic_warmup", "1024", "--weight_avg", "0", "--ema_decay", "0.999", "--double_critic_step", "1",
        "--target_policy_noise", "0.15", "--tpn_decay_start", "16384", "--tpn_floor", "0.04",
        "--lr_schedule", "cosine", "--lr_warmup_episodes", "1024", "--lr_schedule_episodes", "32768",
        "--final_lr_fraction", "0.20", "--min_lr", "1e-6", "--min_replay", "18000", "--max_replay", "200000"]


def run(label, binary, n_train, extra):
    out = OUT / f"{label}.bin"
    cmd = [str(binary), "--seed", str(SEED), "--n_train", str(n_train), "--n_rl", str(N_RL),
           "--n_hedge", str(N_HEDGE), "--threads", str(THREADS), "--c_cost", str(C_COST),
           "--gamma_cost", str(GAMMA), "--out", str(out), *extra]
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT / "cpp_pricer"))
    if r.returncode != 0:
        raise RuntimeError(f"hedge_swing failed: {r.stderr[-800:]}")
    print(f"[{label}] {r.stdout.strip()} -> {out}", flush=True)


def main():
    assert KBIN.exists() and NBIN.exists(), "build both v67 hedge_swing binaries first"
    OUT.mkdir(parents=True, exist_ok=True)
    run("RL-kernel", KBIN, 4096, KON + ["--kernel", str(KERNEL)])        # AC-kernel, exact
    run("RL", NBIN, 32768, KOFF + ["--kernel_off"])                      # AC-sample, closest
    print("done. Verify RL-kernel PV ~1.985 (T5 AC-kernel) and RL PV ~1.959 (T5 AC-sample).")


if __name__ == "__main__":
    main()
