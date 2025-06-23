import argparse
import json
import multiprocessing as mp
import os
import time
import warnings
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from scripts.agent import Agent

# Suppress the macOS PyTorch profiling warning
warnings.filterwarnings("ignore", message=".*record_context_cpp.*")

# import pybullet_envs # to run e.g. HalfCheetahBullet-v0 different reward function bullet-v0 starts ~ -1500. pybullet-v0 starts at 0


def timer(start,end):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def evaluate(frame, eval_runs=5, capture=False, render=False):
    """
    Makes an evaluation run 
    """

    reward_batch = []
    for i in range(eval_runs):
        state, _ = eval_env.reset()
        if render: 
            eval_env.render()
        rewards = 0.0
        while True:
            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)

            state, reward, terminated, truncated, _ = eval_env.step(action_v[0])
            done = terminated or truncated
            rewards += float(reward)
            if done:
                break
        reward_batch.append(rewards)
    if not capture and frame is not None:   
        writer.add_scalar("Reward", np.mean(reward_batch), frame)
    
    return reward_batch



def run(frames=1000, eval_every=1000, eval_runs=5):
    """Deep Q-Learning.
    
    Params
    ======
        frames (int): total number of environment steps to run
        eval_every (int): evaluate every N environment steps  
        eval_runs (int): number of evaluation runs
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    i_episode = 1
    state, _ = train_env.reset()
    score = 0.0
    curiosity_logs = []
    
    # Performance monitoring variables
    start_time = time.time()
    
    for current_frame in range(1, frames + 1):
        # evaluation runs
        if current_frame % eval_every == 0 or current_frame == 1:
            evaluate(current_frame, eval_runs)

        action = agent.act(np.expand_dims(state, axis=0))
        action_v = np.clip(action, action_low, action_high)
        next_state, reward, terminated, truncated, _ = train_env.step(action_v[0])
        done = terminated or truncated

        agent.step(state, action_v[0], reward, next_state, done, current_frame, writer)
            
        if args.icm:
            reward_i = agent.icm.get_intrinsic_reward(state, next_state, action_v[0])
            curiosity_logs.append((current_frame, reward_i))
        
        state = next_state
        score += float(reward)
        
        if done:
            # Calculate performance metrics
            current_time = time.time()
            total_elapsed = current_time - start_time
            
            # Calculate frames per second (based on total training time)
            if total_elapsed > 0:
                frames_per_second = current_frame / total_elapsed
            else:
                frames_per_second = 0.0
            
            # Calculate episode return (accumulated reward for this episode)
            episode_return = float(score)
            
            scores_window.append(episode_return)       # save most recent score
            scores.append(episode_return)              # save most recent score
            writer.add_scalar("Average100", np.mean(scores_window), current_frame)
            writer.add_scalar("Episode_Return", episode_return, current_frame)
            writer.add_scalar("Frames_Per_Second", frames_per_second, current_frame)
            
            for v in curiosity_logs:
                i, r = v[0], v[1]
                writer.add_scalar("Intrinsic Reward", r, i)
            
            # Enhanced performance monitoring output - matches your requested format
            print(f'\rEpisode Return = {episode_return:.3f} | Frames = {current_frame}/{frames} | Frames Per Second = {frames_per_second:.3f}', end="")
            
            i_episode += 1 
            state, _ = train_env.reset()
            score = 0.0
            curiosity_logs = []
            




parser = argparse.ArgumentParser(description="")
parser.add_argument("-env", type=str,default="HalfCheetahBulletEnv-v0", help="Environment name, default = HalfCheetahBulletEnv-v0")
parser.add_argument("--device", type=str, default="gpu", help="Select trainig device [gpu/cpu], default = gpu")
parser.add_argument("-nstep", type=int, default=1, help ="Nstep bootstrapping, default 1")
parser.add_argument("-per", type=int, default=0, choices=[0,1], help="Adding Priorizied Experience Replay to the agent if set to 1, default = 0")
parser.add_argument("-munchausen", type=int, default=0, choices=[0,1], help="Adding Munchausen RL to the agent if set to 1, default = 0")
parser.add_argument("-iqn", type=int, choices=[0,1], default=0, help="Use distributional IQN Critic if set to 1, default = 1")
parser.add_argument("-noise", type=str, choices=["ou", "gauss"], default="OU", help="Choose noise type: ou = OU-Noise, gauss = Gaussian noise, default ou")
parser.add_argument("-info", type=str, help="Information or name of the run")

parser.add_argument("-frames", type=int, default=1_000_000, help="The amount of training interactions with the environment, default is 1mio")
parser.add_argument("-eval_every", type=int, default=10000, help="Number of interactions after which the evaluation runs are performed, default = 10000")
parser.add_argument("-eval_runs", type=int, default=1, help="Number of evaluation runs performed, default = 1")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr_a", type=float, default=3e-4, help="Actor learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-lr_c", type=float, default=3e-4, help="Critic learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-learn_every", type=int, default=1, help="Learn every x interactions, default = 1")
parser.add_argument("-learn_number", type=int, default=1, help="Learn x times per interaction, default = 1")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("--max_replay_size", type=int, default=int(1e6), help="Maximum size of the replay buffer, default is 1e6")
parser.add_argument("--min_replay_size", type=int, default=1000, help="Minimum replay buffer size before learning starts (default: 1000)")
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size, default is 128")
parser.add_argument("-t", "--tau", type=float, default=1e-3, help="Softupdate factor tau, default is 1e-3") #for per 1e-2 for regular 1e-3 -> Pendulum!
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("-n_cores", type=int, default=None, help="Maximum number of CPU cores to use (default: use all available cores)")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("--icm", type=int, default=0, choices=[0,1], help="Using Intrinsic Curiosity Module, default=0 (NO!)")
parser.add_argument("--add_ir", type=int, default=0, choices=[0,1], help="Add intrisic reward to the extrinsic reward, default = 0 (NO!) ")
parser.add_argument("--compile", type=int, default=0, choices=[0,1], help="Use torch.compile for model optimization, default=0 (NO!)")
parser.add_argument("--fp32", type=int, default=1, choices=[0,1], help="Use float32 precision for better performance, default=1 (YES!)")

args = parser.parse_args()

# Apply float32 optimization if enabled
if args.fp32:
    torch.set_default_dtype(torch.float32)
    print("Using float32 precision for improved performance")
else:
    print("Using default precision (float64)")

if __name__ == "__main__":
    env_name = args.env
    seed = args.seed
    frames = args.frames
    GAMMA = args.gamma
    TAU = args.tau
    HIDDEN_SIZE = args.layer_size
    BUFFER_SIZE = int(args.max_replay_size)
    BATCH_SIZE = args.batch_size
    LR_ACTOR = args.lr_a         # learning rate of the actor 
    LR_CRITIC = args.lr_c        # learning rate of the critic
    saved_model = args.saved_model

    writer = SummaryWriter("runs/"+args.info)
    train_env = gym.make(args.env)
    eval_env = gym.make(args.env)
    
    # Seed environments
    train_env.reset(seed=seed)
    eval_env.reset(seed=seed+1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Device selection (CPU only)
    if args.device == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_str = "cuda:0"
            print("Using CUDA GPU")
        else:
            device = torch.device("cpu")
            device_str = "cpu"
            print("GPU requested but not available, using CPU")
    else:
        device = torch.device("cpu")
        device_str = "cpu"
        print("Using CPU (as requested)")
    
    # Configure CPU cores
    if args.n_cores is not None:
        n_cores = min(args.n_cores, mp.cpu_count())
        print(f"Using {n_cores} CPU cores (requested: {args.n_cores}, available: {mp.cpu_count()})")
    else:
        n_cores = mp.cpu_count()
        print(f"Using all available CPU cores: {n_cores}")
    
    # Set threading environment variables
    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    os.environ['MKL_NUM_THREADS'] = str(n_cores)
    
    # Configure PyTorch threading
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(min(4, n_cores // 2))
    
    # Enable optimized CPU kernels if available
    try:
        if hasattr(torch.backends, 'mkl') and hasattr(torch.backends.mkl, 'enabled'):
            torch.backends.mkl.enabled = True
    except Exception:
        pass
    try:
        if hasattr(torch.backends, 'mkldnn') and hasattr(torch.backends.mkldnn, 'enabled'):
            torch.backends.mkldnn.enabled = True
    except Exception:
        pass
    
    print("PyTorch optimized for {} CPU cores (intra: {}, inter: {})".format(
        n_cores, torch.get_num_threads(), torch.get_num_interop_threads()))
    
    # Initialize environments and get action/state space info
    temp_env = gym.make(args.env)
    temp_env.reset(seed=seed)
    action_high = temp_env.action_space.high[0]
    action_low = temp_env.action_space.low[0]
    state_size = temp_env.observation_space.shape[0]
    action_size = temp_env.action_space.shape[0]
    temp_env.close()  # Close temporary environment
    agent = Agent(state_size=state_size, action_size=action_size, n_step=args.nstep, per=args.per, munchausen=args.munchausen,distributional=args.iqn,
                 curiosity=(args.icm, args.add_ir), noise_type=args.noise, random_seed=seed, hidden_size=HIDDEN_SIZE, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE, GAMMA=GAMMA,
                 LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, TAU=TAU, LEARN_EVERY=args.learn_every, LEARN_NUMBER=args.learn_number, device=device_str, frames=args.frames, 
                 min_replay_size=args.min_replay_size, use_compile=bool(args.compile)) 
    
    t0 = time.time()
    if saved_model is not None:
        agent.actor_local.load_state_dict(torch.load(saved_model))
        evaluate(frame=None, capture=False)
    else:    
        run(frames=args.frames,
            eval_every=args.eval_every,
            eval_runs=args.eval_runs)

    # Final evaluation at the end of training
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    
    final_eval_runs = max(10, args.eval_runs * 2)  # Use more runs for final evaluation
    final_rewards = evaluate(frame=args.frames, eval_runs=final_eval_runs, capture=True)
    
    # Calculate statistics
    avg_return = np.mean(final_rewards)
    std_return = np.std(final_rewards)
    min_return = np.min(final_rewards)
    max_return = np.max(final_rewards)
    
    print(f"Average Episode Return: {avg_return:.3f} ± {std_return:.3f}")
    print(f"Min Episode Return: {min_return:.3f}")
    print(f"Max Episode Return: {max_return:.3f}")
    print(f"Number of Evaluation Episodes: {final_eval_runs}")
    
    # Log final metrics to tensorboard
    writer.add_scalar("Final_Evaluation/Average_Return", avg_return, args.frames)
    writer.add_scalar("Final_Evaluation/Std_Return", std_return, args.frames)
    writer.add_scalar("Final_Evaluation/Min_Return", min_return, args.frames)
    writer.add_scalar("Final_Evaluation/Max_Return", max_return, args.frames)
    
    print("="*60)

    t1 = time.time()
    eval_env.close()
    timer(t0, t1)
    # save trained model 
    torch.save(agent.actor_local.state_dict(), 'runs/'+args.info+".pth")
    # save parameter
    with open('runs/'+args.info+".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)