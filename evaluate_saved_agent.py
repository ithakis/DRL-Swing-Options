import argparse
import json
import os

import gymnasium as gym
import torch

from src.agent import Agent
from src.agent_evaluation import evaluate_agent


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent using the shared evaluation module.")
    parser.add_argument(
        "--run_name",
        type=str,
        default="test",
        help="Name of the run to load hyperparameters and (optionally) model weights, default: test",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of evaluation episodes, default: 10")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation inference (default: 1, legacy behavior).",
    )
    parser.add_argument(
        "--profile_eval", action="store_true", help="Profile evaluation with cProfile (stdout + profilelogs/)."
    )
    args = parser.parse_args()

    if args.eval_batch_size < 1:
        raise ValueError("eval_batch_size must be >= 1")

    with open(f"runs/{args.run_name}.json", "r") as f:
        parameters = json.load(f)

    print("Parameters: \n", parameters)
    parameters = dotdict(parameters)

    eval_env = gym.make(parameters.env)
    eval_env.reset(seed=parameters.seed)
    setattr(eval_env, "n_paths_eval", args.runs)

    action_high = eval_env.action_space.high[0]
    action_low = eval_env.action_space.low[0]
    state_size = eval_env.observation_space.shape[0]
    action_size = eval_env.action_space.shape[0]

    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        n_step=parameters.nstep,
        random_seed=parameters.seed,
        hidden_size=parameters.layer_size,
        BATCH_SIZE=parameters.batch_size,
        BUFFER_SIZE=parameters.replay_memory,
        GAMMA=parameters.gamma,
        LR_ACTOR=parameters.lr_a,
        LR_CRITIC=parameters.lr_c,
        t=getattr(parameters, "t", getattr(parameters, "tau", 0.002)),
        LEARN_EVERY=parameters.learn_every,
        LEARN_NUMBER=parameters.learn_number,
        noise_sigma0=getattr(parameters, "noise_sigma0", getattr(parameters, "pre_noise_sigma0", 1.0)),
        noise_floor=getattr(parameters, "noise_floor", getattr(parameters, "pre_noise_floor", 0.05)),
        noise_plateau=getattr(parameters, "noise_plateau", getattr(parameters, "pre_noise_plateau", 0)),
        device="cpu",
        paths=0,
        min_replay_size=getattr(parameters, "min_replay_size", parameters.batch_size * 10),
        activation=getattr(parameters, "activation", "silu"),
    )

    weights_path = f"runs/{args.run_name}.pth"
    if os.path.exists(weights_path):
        agent.actor_local.load_state_dict(torch.load(weights_path, map_location="cpu"))  # type: ignore
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"⚠️ Weights file not found at {weights_path}, evaluating current agent parameters.")

    summary = evaluate_agent(
        agent=agent,
        eval_env=eval_env,
        writer=None,
        path=None,
        evaluations_dir=None,
        lsm_price=None,
        csv_writer=None,
        eval_batch_size=args.eval_batch_size,
        profile=args.profile_eval,
        n_episodes=args.runs,
    )

    print(
        f"\nEvaluated {summary.n_paths} episodes | "
        f"mean return {summary.option_price:.3f} | "
        f"std {summary.price_std:.3f} | "
        f"95% CI ±{summary.confidence_95:.3f}"
    )


if __name__ == "__main__":
    main()
