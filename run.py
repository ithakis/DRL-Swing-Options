import argparse
import json
import multiprocessing as mp

# Configure PyTorch threading BEFORE importing torch to avoid runtime errors
import os
import time
import warnings

# Suppress the macOS PyTorch profiling warning
warnings.filterwarnings("ignore", message=".*record_context_cpp.*")

# import pybullet_envs # to run e.g. HalfCheetahBullet-v0 different reward function bullet-v0 starts ~ -1500. pybullet-v0 starts at 0
from collections import deque

import gymnasium as gym
import numpy as np

# Set threading environment variables if not already set
if not os.environ.get('OMP_NUM_THREADS'):
    cpu_count = mp.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)

import torch
from torch.utils.tensorboard import SummaryWriter

# Set PyTorch threading after import but before any operations
if torch.get_num_threads() != mp.cpu_count():
    try:
        torch.set_num_threads(mp.cpu_count())
        torch.set_num_interop_threads(min(4, mp.cpu_count() // 2))
        print(f"🔧 PyTorch configured for {mp.cpu_count()} CPU cores")
    except RuntimeError:
        print("⚠️  Threading already configured, skipping manual setup")

from scripts import MultiPro
from scripts.agent import Agent


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
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    if not capture and frame is not None:   
        writer.add_scalar("Reward", np.mean(reward_batch), frame)
    
    return reward_batch



def run(frames=1000, eval_every=1000, eval_runs=5, worker=1):
    """Deep Q-Learning.
    
    Params
    ======
        frames (int): total number of environment steps to run
        eval_every (int): evaluate every N environment steps  
        eval_runs (int): number of evaluation runs
        worker (int): number of parallel environments
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    i_episode = 1
    state = envs.reset()
    score = 0.0
    curiosity_logs = []
    
    # Performance monitoring variables
    start_time = time.time()
    
    # Calculate iterations needed: total frames divided by number of workers
    # Each iteration collects 'worker' number of environment steps
    total_iterations = frames // worker
    
    for iteration in range(1, total_iterations + 1):
        # Current total environment steps
        current_frame = iteration * worker
        
        # evaluation runs
        if current_frame % eval_every == 0 or iteration == 1:
            evaluate(current_frame, eval_runs)

        action = agent.act(state)
        action_v = np.clip(action, action_low, action_high)
        next_state, reward, done, _ = envs.step(action_v)
        # done is already the combination of terminated and truncated from the vectorized env

        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            agent.step(s, a, r, ns, d, current_frame, writer)
            
        if args.icm:
            reward_i = agent.icm.get_intrinsic_reward(state[0], next_state[0], action[0])
            curiosity_logs.append((current_frame, reward_i))
        state = next_state
        score += reward
        
        if done.any():
            # Calculate performance metrics
            current_time = time.time()
            total_elapsed = current_time - start_time
            
            # Calculate frames per second (based on total training time)
            if total_elapsed > 0:
                frames_per_second = current_frame / total_elapsed
            else:
                frames_per_second = 0.0
            
            # Calculate episode return (accumulated reward for this episode)
            # score is accumulated from vectorized environments, so we take the sum
            if hasattr(score, '__len__'):
                episode_return = np.sum(score)
            else:
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
            state = envs.reset()
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
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-3, help="Softupdate factor tau, default is 1e-3") #for per 1e-2 for regular 1e-3 -> Pendulum!
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("-w", "--worker", type=int, default=1, help="Number of parallel environments, default = 1")
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
    worker = args.worker
    GAMMA = args.gamma
    TAU = args.tau
    HIDDEN_SIZE = args.layer_size
    BUFFER_SIZE = int(args.replay_memory)
    BATCH_SIZE = args.batch_size  # Keep batch size constant regardless of worker count
    LR_ACTOR = args.lr_a         # learning rate of the actor 
    LR_CRITIC = args.lr_c        # learning rate of the critic
    saved_model = args.saved_model

    writer = SummaryWriter("runs/"+args.info)
    envs = MultiPro.SubprocVecEnv([lambda: gym.make(args.env) for i in range(args.worker)])
    eval_env = gym.make(args.env)
    envs.seed(seed)
    # Update seeding for gymnasium
    eval_env.reset(seed=seed+1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Device selection (CPU only)
    if args.device == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Using CUDA GPU")
        else:
            device = torch.device("cpu")
            print("GPU requested but not available, using CPU")
    else:
        device = torch.device("cpu")
        print("Using CPU (as requested)")
    
    # Optimize PyTorch threading for better CPU utilization
    import multiprocessing as mp
    cpu_count = mp.cpu_count()
    
    # For training, use all available cores for intra-op parallelism
    # but limit inter-op parallelism to reduce contention  
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(min(4, cpu_count // 2))
    
    # Enable optimized CPU kernels if available
    if hasattr(torch.backends, 'mkl'):
        torch.backends.mkl.enabled = True
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
    
    print("PyTorch optimized for {} CPU cores (intra: {}, inter: {})".format(
        cpu_count, torch.get_num_threads(), torch.get_num_interop_threads()))
    
    action_high = eval_env.action_space.high[0]
    action_low = eval_env.action_space.low[0]
    state_size = eval_env.observation_space.shape[0]
    action_size = eval_env.action_space.shape[0]
    agent = Agent(state_size=state_size, action_size=action_size, n_step=args.nstep, per=args.per, munchausen=args.munchausen,distributional=args.iqn,
                 curiosity=(args.icm, args.add_ir), noise_type=args.noise, random_seed=seed, hidden_size=HIDDEN_SIZE, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE, GAMMA=GAMMA,
                 LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, TAU=TAU, LEARN_EVERY=args.learn_every, LEARN_NUMBER=args.learn_number, device=device, frames=args.frames, worker=args.worker, use_compile=bool(args.compile)) 
    
    t0 = time.time()
    if saved_model is not None:
        agent.actor_local.load_state_dict(torch.load(saved_model))
        evaluate(frame=None, capture=False)
    else:    
        run(frames = args.frames,  # Keep total frames constant, not divided by workers
            eval_every=args.eval_every,
            eval_runs=args.eval_runs,
            worker=args.worker)

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