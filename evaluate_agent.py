# import pybullet_envs # to run e.g. HalfCheetahBullet-v0 different reward function bullet-v0 starts ~ -1500. pybullet-v0 starts at 0
import argparse
import json

import gymnasium as gym
import numpy as np

#from  files import MultiPro
from src.agent import Agent


def evaluate(n_paths_eval=512):
    """
    Makes an evaluation run 
    """

    for _ in range(n_paths_eval):
        state, _ = eval_env.reset()

        rewards = 0
        while True:
            eval_env.render()
            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)

            state, reward, terminated, truncated, _ = eval_env.step(action_v[0])
            done = terminated or truncated
            rewards += reward
            if done:
                print("Episode Rewards: {}".format(rewards))
                break

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--run_name", type=str, default="test", help="Name of the run to load the hyperparameter and the model weights, default: test")
    parser.add_argument("--runs", type=int, default=10, help="Number of evaluation runs with the policy, default: 10")
    
    args = parser.parse_args()
    
    with open('runs/'+args.run_name+".json", 'r') as f:
        parameters = json.load(f)
    
    
    print("Parameters: \n", parameters)
    parameters = dotdict(parameters)
    # create eval environement
    eval_env = gym.make(parameters.env)
    eval_env.reset(seed=parameters.seed)
    action_high = eval_env.action_space.high[0]
    action_low = eval_env.action_space.low[0]
    state_size = eval_env.observation_space.shape[0]
    action_size = eval_env.action_space.shape[0]
    
    # create agent
    agent = Agent(state_size=state_size, action_size=action_size, n_step=parameters.nstep, per=parameters.per, munchausen=parameters.munchausen,distributional=parameters.iqn,
                 noise_type=parameters.noise, noise_sigma=getattr(parameters, 'noise_sigma', 1.0), noise_anneal_power=getattr(parameters, 'noise_anneal_power', 1.0),
                 random_seed=parameters.seed,
                 hidden_size=parameters.layer_size,
                 BATCH_SIZE=parameters.batch_size, BUFFER_SIZE=parameters.replay_memory, GAMMA=parameters.gamma,
                 LR_ACTOR=parameters.lr_a, LR_CRITIC=parameters.lr_c, t=getattr(parameters, 't', getattr(parameters, 'tau', 0.002)),
                 tau_final=getattr(parameters, 'tau_final', None) if getattr(parameters, 'tau_final', -1) > 0 else None,
                 tau_schedule_frac=getattr(parameters, 'tau_schedule_frac', 0.0),
                 LEARN_EVERY=parameters.learn_every,
                 LEARN_NUMBER=parameters.learn_number, device="cpu", paths=0,
                 per_priority_floor=getattr(parameters, 'per_priority_floor', 1e-6),
                 per_priority_clip_pct=getattr(parameters, 'per_priority_clip_pct', 99.5),
                 critic_ema_decay=getattr(parameters, 'critic_ema_decay', 0.0)) 
    evaluate(args.runs)
    
