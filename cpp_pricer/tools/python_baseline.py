#!/usr/bin/env python3
"""Faithful PyTorch baseline for the two timed operations (apples-to-apples vs C++).

  0 -> n_train : simulate train paths + warm-start + training loop
  n_train -> n_eval : simulate OOS paths + greedy EMA rollout (canonical evaluator)

Emits the same JSON shape as the C++ CLI. No LSM / TensorBoard / parquet / periodic eval.
Usage:
  python cpp_pricer/tools/python_baseline.py --seed 11 --n_train 4096 --n_eval 65536 [--threads N]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from src.agent import Agent
from src.swing_contract import SwingContract
from src.swing_env import SwingOptionEnv
from src.simulate_hhk_spot import simulate_hhk_spot, no_seasonal_function
from src.transition_kernel import KernelParams, precompute_kernel
from src.agent_evaluation import evaluate_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_eval", type=int, default=65536)
    ap.add_argument("--eval_batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    contract = SwingContract(q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0,
                             maturity=0.0833, n_rights=22, r=0.05, c_cost=0.04, gamma_cost=2.0)
    hhk = dict(S0=1.0, alpha=12.0, sigma=1.2, beta=150.0, lam=6.0, mu_J=0.3)
    kp = KernelParams(alpha=hhk["alpha"], sigma=hhk["sigma"], beta=hhk["beta"], lam=hhk["lam"],
                      mu_J=hhk["mu_J"], dt=contract.dt, M_x=2, N_max=1, M_per_k=1, f_id="no_seasonal")
    kernel = precompute_kernel(kp, n_rights=contract.n_rights, f_callable=no_seasonal_function,
                               maturity=contract.maturity)

    def make_agent():
        return Agent(state_size=9, action_size=1, n_step=1, random_seed=args.seed,
                     hidden_size=64, actor_layers=3, critic_layers=3, optimizer="adamw",
                     weight_decay_actor=5e-5, weight_decay_critic=1e-4, BUFFER_SIZE=100000,
                     BATCH_SIZE=128, GAMMA=1.0, t=0.0032, LR_ACTOR=3e-4, LR_CRITIC=6e-4,
                     LEARN_EVERY=2, LEARN_NUMBER=2, min_replay_size=1000,
                     noise_sigma0=1.30, noise_floor=0.26, noise_schedule="linear",
                     noise_decay_episodes=args.n_train, activation="silu", norm_type="layernorm",
                     init_method="orthogonal", critic_warmup_episodes=512, adaptive_noise_scale=0.6,
                     warmup_noise_fraction=0.3, weight_averaging="ema", ema_decay=0.999,
                     action_output="beta_sigmoid_1.5", use_robust_normalization=True, strike=1.0,
                     expected_target_kernel=kernel, total_episodes=args.n_train)

    # ============ 0 -> n_train ============
    t0 = time.perf_counter()
    t, S, X, Y = simulate_hhk_spot(n_paths=args.n_train, n_steps=contract.n_rights - 1,
                                   T=contract.maturity, f=no_seasonal_function, seed=args.seed,
                                   dtype=np.float32, stratify=True, batch_size=128, **hhk)
    train_env = SwingOptionEnv(contract=contract, hhk_params=hhk, dataset=(t, S, X, Y),
                               obs_dtype=np.float32)
    t_sim_train = time.perf_counter()

    agent = make_agent()
    agent.calibrate_bias(env=train_env, n_episodes=1024, target_std=0.005, mode="closed_form")
    t_calib = time.perf_counter()

    low = np.array([0.0], dtype=np.float32); high = np.array([1.0], dtype=np.float32)
    total_steps = 0
    for ep in range(1, args.n_train + 1):
        agent.update_episode_count(ep)
        state, _ = train_env.reset()
        while True:
            action = agent.act(np.expand_dims(state, 0))
            action_v = np.clip(action, low, high)
            next_state, reward, term, trunc, _ = train_env.step(action_v[0])
            done = term or trunc
            total_steps += 1
            agent.step(state, action_v[0], reward, next_state, done, total_steps, None)
            state = next_state
            if done:
                break
        agent.step_lr_schedulers(ep)
    t1 = time.perf_counter()

    # ============ n_train -> n_eval ============
    te, Se, Xe, Ye = simulate_hhk_spot(n_paths=args.n_eval, n_steps=contract.n_rights - 1,
                                       T=contract.maturity, f=no_seasonal_function,
                                       seed=args.seed + 777, dtype=np.float32, stratify=False, **hhk)
    eval_env = SwingOptionEnv(contract=contract, hhk_params=hhk, dataset=(te, Se, Xe, Ye),
                              obs_dtype=np.float32)
    t_sim_eval = time.perf_counter()

    summary = evaluate_agent(agent, eval_env, writer=None, path=None, evaluations_dir=None,
                             lsm_price=None, eval_batch_size=args.eval_batch, n_episodes=args.n_eval)
    t2 = time.perf_counter()

    out = dict(price=summary.option_price, ci95=1.96*summary.price_std/np.sqrt(args.n_eval),
               std=summary.price_std, seed=args.seed, n_train=args.n_train, n_eval=args.n_eval,
               threads=torch.get_num_threads(), precision="fp32",
               t_sim_train=t_sim_train-t0, t_calibrate=t_calib-t_sim_train, t_train=t1-t_calib,
               t_sim_eval=t_sim_eval-t1, t_eval=t2-t_sim_eval,
               t_zero_to_4k=t1-t0, t_4k_to_65k=t2-t1, t_total=t2-t0)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
