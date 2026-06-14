#!/usr/bin/env python3
"""Export PyTorch reference fixtures for the C++ parity tests.

Dumps (into cpp_pricer/data/):
  actor.bin, critic.bin   — weights in the C++ canonical flat order (float64)
  ref_fwd.bin             — a fixed batch + actor/critic forward outputs
  ref_kernel.bin          — states/next_states + expected_critic_target output
  kernel_v64.bin          — the precomputed quadrature mesh (also used at training time)

Run from the repo root:  python cpp_pricer/tools/export_reference.py
"""
import os, sys, struct
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
torch.set_default_dtype(torch.float64)          # parity is checked in double precision

from src.agent import Agent
from src.swing_contract import SwingContract
from src.swing_env import SwingOptionEnv
from src.simulate_hhk_spot import simulate_hhk_spot, no_seasonal_function
from src.transition_kernel import KernelParams, precompute_kernel, expected_critic_target

DATA = os.path.join(REPO, "cpp_pricer", "data")
os.makedirs(DATA, exist_ok=True)

# ---- v64 focal config ----------------------------------------------------
contract = SwingContract(q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0,
                         maturity=0.0833, n_rights=22, r=0.05, c_cost=0.04, gamma_cost=2.0)
hhk = dict(S0=1.0, alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3)
SEED = 11

# N_max: 0 = H-K1 canonical (M=2, jump folded into its unconditional mean — matches config.hpp
# default and the speedup/NN research); 1 = legacy M=4 mesh.  `export_reference.py [N_max]`.
_NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
kp = KernelParams(alpha=hhk["alpha"], sigma=hhk["sigma"], beta=hhk["beta"], lam=hhk["lam"],
                  mu_J=hhk["mu_J"], dt=contract.dt, M_x=2, N_max=_NMAX, M_per_k=1, f_id="no_seasonal")
kernel = precompute_kernel(kp, n_rights=contract.n_rights, f_callable=no_seasonal_function,
                           maturity=contract.maturity, force_rebuild=True)

agent = Agent(state_size=9, action_size=1, n_step=1, random_seed=SEED,
              hidden_size=64, actor_layers=3, critic_layers=3, optimizer="adamw",
              weight_decay_actor=5e-5, weight_decay_critic=1e-4, BUFFER_SIZE=100000,
              BATCH_SIZE=128, GAMMA=1.0, t=0.0032, LR_ACTOR=3e-4, LR_CRITIC=6e-4,
              LEARN_EVERY=2, LEARN_NUMBER=2, min_replay_size=1000,
              noise_sigma0=1.30, noise_floor=0.26, noise_schedule="linear",
              activation="silu", norm_type="layernorm", init_method="orthogonal",
              critic_warmup_episodes=512, adaptive_noise_scale=0.6, warmup_noise_fraction=0.3,
              weight_averaging="ema", ema_decay=0.999, action_output="beta_sigmoid_1.5",
              use_robust_normalization=True, strike=1.0,
              expected_target_kernel=kernel, total_episodes=4096)

# wire the profitability gate (run.py does this via the stack walk)
for a in (agent.actor_local, agent.actor_target):
    a.set_profitability_params(q_min=0.0, q_max=2.0, c_cost=0.04, gamma_cost=2.0, s_minus_k_index=0)


def write_flat(path, arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    with open(path, "wb") as f:
        f.write(struct.pack("<i", arr.size))
        f.write(arr.tobytes())


def actor_flat(net):
    out = []
    for blk in net.hidden_layers:          # [Linear, LayerNorm, act]
        out += [blk[0].weight, blk[0].bias, blk[1].weight, blk[1].bias]
    out += [net.fc4.weight, net.fc4.bias]
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in out])


def critic_flat(net):
    out = [net.state_encoder[0].weight, net.state_encoder[0].bias,
           net.state_encoder[1].weight, net.state_encoder[1].bias,
           net.action_layer[0].weight, net.action_layer[0].bias,
           net.action_layer[1].weight, net.action_layer[1].bias,
           net.post_layers[0][0].weight, net.post_layers[0][0].bias,
           net.post_layers[0][1].weight, net.post_layers[0][1].bias,
           net.fc4.weight, net.fc4.bias]
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in out])


write_flat(os.path.join(DATA, "actor.bin"), actor_flat(agent.actor_local))
write_flat(os.path.join(DATA, "critic.bin"), critic_flat(agent.critic_local))

# ---- collect a realistic batch of transitions via a random rollout -------
t, S, X, Y = simulate_hhk_spot(n_paths=256, n_steps=contract.n_rights - 1, T=contract.maturity,
                               f=no_seasonal_function, seed=SEED, dtype=np.float64,
                               stratify=False, **hhk)
env = SwingOptionEnv(contract=contract, hhk_params=hhk, dataset=(t, S, X, Y), obs_dtype=np.float64)
rng = np.random.default_rng(0)
states, actions, next_states = [], [], []
for ep in range(8):
    s, _ = env.reset()
    while True:
        a = rng.random(1).astype(np.float64)
        ns, r, term, trunc, _ = env.step(a)
        states.append(s.copy()); actions.append(float(a[0])); next_states.append(ns.copy())
        s = ns
        if term or trunc:
            break
        if len(states) >= 128:
            break
    if len(states) >= 128:
        break
states = np.asarray(states[:128], dtype=np.float64)
actions = np.asarray(actions[:128], dtype=np.float64)
next_states = np.asarray(next_states[:128], dtype=np.float64)
B = states.shape[0]

st = torch.from_numpy(states); ac = torch.from_numpy(actions).reshape(-1, 1)
with torch.no_grad():
    a_out = agent.actor_local(st).reshape(-1).cpu().numpy()
    c_out = agent.critic_local(st, ac).reshape(-1).cpu().numpy()

with open(os.path.join(DATA, "ref_fwd.bin"), "wb") as f:
    f.write(struct.pack("<i", B))
    f.write(states.tobytes()); f.write(actions.tobytes())
    f.write(a_out.astype(np.float64).tobytes()); f.write(c_out.astype(np.float64).tobytes())

# ---- kernel target reference --------------------------------------------
ns_t = torch.from_numpy(next_states)
with torch.no_grad():
    qn = expected_critic_target(agent.critic_target, agent.actor_target,
                                states=st, next_states=ns_t, kernel=kernel,
                                strike=1.0, chunk_size=None).reshape(-1).cpu().numpy()
with open(os.path.join(DATA, "ref_kernel.bin"), "wb") as f:
    f.write(struct.pack("<i", B))
    f.write(states.tobytes()); f.write(next_states.tobytes())
    f.write(qn.astype(np.float64).tobytes())

# ---- kernel mesh (C++ binary format) ------------------------------------
with open(os.path.join(DATA, "kernel_v64.bin"), "wb") as f:
    f.write(struct.pack("<i", 0x4B524E4C))                       # magic 'KRNL'
    f.write(struct.pack("<ddd", kernel.decay_X, kernel.sigma_X, kernel.decay_Y))
    f.write(struct.pack("<iii", kernel.n_rights, kernel.M_x, kernel.M_y))
    for arr in (kernel.z_X, kernel.w_X, kernel.delta_Y, kernel.w_Y, kernel.exp_f_grid):
        f.write(np.asarray(arr, dtype=np.float64).tobytes())

print(f"Exported reference fixtures to {DATA} (B={B}, M={kernel.M})")
