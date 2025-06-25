import argparse
import csv
import json
import multiprocessing as mp
import os
import time
import warnings
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agent import Agent
from src.swing_env import SwingOptionEnv

# Suppress the macOS PyTorch profiling warning
warnings.filterwarnings("ignore", message=".*record_context_cpp.*")

# import pybullet_envs # to run e.g. HalfCheetahBullet-v0 different reward function bullet-v0 starts ~ -1500. pybullet-v0 starts at 0


def timer(start,end):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def init_csv_logging(info_string):
    """Initialize CSV logging files for training, evaluation, and raw evaluation episodes"""
    # Create main logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create experiment-specific directory
    experiment_dir = f"logs/{info_string}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create validation_runs subdirectory for step-by-step data
    validation_runs_dir = f"{experiment_dir}/validation_runs"
    os.makedirs(validation_runs_dir, exist_ok=True)
    
    # Define CSV file paths - main experiment CSVs stay in experiment folder
    training_csv = f"{experiment_dir}/{info_string}_training.csv"
    evaluation_csv = f"{experiment_dir}/{info_string}_evaluation.csv"
    raw_episodes_csv = f"{experiment_dir}/{info_string}_raw_episodes.csv"
    
    # Initialize training CSV with headers
    with open(training_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'return', 'steps', 'total_steps', 'avg_100', 
            'paths_per_sec', 'steps_per_sec', 'time_elapsed'
        ])
    
    # Initialize evaluation CSV with headers  
    with open(evaluation_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'training_episode', 'eval_run', 'option_price', 'price_std', 
            'confidence_95', 'avg_total_exercised', 'avg_exercise_count',
            'min_return', 'max_return', 'n_runs'
        ])
    
    # Initialize raw evaluation episodes CSV with headers
    with open(raw_episodes_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'training_episode', 'eval_run', 'episode_idx', 'seed', 'return', 
            'steps', 'total_exercised', 'exercise_count', 'final_inventory', 
            'avg_spot_price', 'max_spot_price', 'min_spot_price'
        ])
    
    return training_csv, evaluation_csv, raw_episodes_csv, validation_runs_dir


def log_training_episode(csv_file, episode, episode_return, path_steps, total_steps, 
                        avg_100, paths_per_sec, steps_per_sec, time_elapsed):
    """Log training episode data to CSV"""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            episode, round(episode_return, 6), path_steps, total_steps, 
            round(avg_100, 6), round(paths_per_sec, 3), round(steps_per_sec, 1), 
            round(time_elapsed, 2)
        ])


def log_evaluation_run(csv_file, training_episode, eval_run, pricing_stats):
    """Log evaluation run data to CSV"""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            training_episode, eval_run, round(pricing_stats['option_price'], 6),
            round(pricing_stats['price_std'], 6), round(pricing_stats['confidence_95'], 6),
            round(pricing_stats['avg_total_exercised'], 6), 
            round(pricing_stats['avg_exercise_count'], 3),
            round(min(pricing_stats['all_returns']), 6),
            round(max(pricing_stats['all_returns']), 6),
            pricing_stats['n_runs']
        ])


def log_raw_evaluation_episode(csv_file, training_episode, eval_run, episode_idx, 
                              seed, episode_return, steps, total_exercised, exercise_count,
                              final_inventory, avg_spot_price, max_spot_price, min_spot_price):
    """Log individual raw evaluation episode data to CSV"""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            training_episode, eval_run, episode_idx, seed, round(episode_return, 6),
            steps, round(total_exercised, 6), exercise_count, round(final_inventory, 6),
            round(avg_spot_price, 6), round(max_spot_price, 6), round(min_spot_price, 6)
        ])


def log_detailed_step_data(validation_runs_dir, training_episode, all_episodes_data):
    """Log detailed step-by-step data for all episodes in an evaluation run to one CSV file"""
    # Create filename for this evaluation run
    filename = f"eval_run_{training_episode}.csv"
    filepath = os.path.join(validation_runs_dir, filename)
    
    # Write all episodes data to the file
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write compact headers including episode_idx
        writer.writerow(['episode_idx', 'step', 'spot', 'q_remain', 'q_exerc', 'time_left', 'action', 'q_actual', 'reward'])
        
        # Write step data for all episodes with minimal precision to save space
        for episode_idx, step_data in enumerate(all_episodes_data):
            for step_info in step_data:
                writer.writerow([
                    episode_idx,
                    step_info['step'],
                    round(step_info['spot_price'], 4),
                    round(step_info['q_remaining'], 4), 
                    round(step_info['q_exercised'], 4),
                    round(step_info['time_left'], 4),
                    round(step_info['action'], 6),
                    round(step_info['q_actual'], 6),
                    round(step_info['reward'], 6)
                ])


def evaluate_swing_option_price(agent, eval_env, runs=100, base_seed=0, raw_episodes_csv=None, training_episode=None, validation_runs_dir=None):
    """
    Evaluate swing option price using Monte Carlo simulation
    
    Args:
        agent: Trained D4PG agent
        eval_env: Swing option environment for evaluation
        runs: Number of Monte Carlo runs
        base_seed: Base seed for reproducible evaluation
        raw_episodes_csv: Path to raw episodes CSV file for detailed logging
        training_episode: Current training episode number for logging
        validation_runs_dir: Directory for detailed step-by-step CSV files
        
    Returns:
        dict: Dictionary with pricing statistics
    """
    discounted_returns = []
    exercise_stats = []
    all_episodes_step_data = []  # Collect step data for all episodes
    
    for i in range(runs):
        path_seed = base_seed + i + 1
        state, _ = eval_env.reset(seed=path_seed)
        
        disc_return = 0.0
        total_exercised = 0.0
        exercise_count = 0
        step = 0
        spot_prices = []
        
        # Store detailed step data for this episode
        episode_step_data = []
        
        # Track initial inventory for final calculation
        initial_inventory = eval_env.contract.Q_max if hasattr(eval_env, 'contract') else 0.0
        
        while True:
            # Get action from agent
            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action, 0.0, 1.0)  # Ensure valid action range
            
            # Track spot price if available in state
            if len(state) > 0:
                spot_prices.append(float(state[0]))  # Assuming first element is spot price
            
            # Store step data before taking action
            step_info = {
                'step': step,
                'spot_price': float(state[0]) if len(state) > 0 else 0.0,
                'q_remaining': float(state[2]) if len(state) > 2 else 0.0,  # Q_remaining
                'q_exercised': float(state[1]) if len(state) > 1 else 0.0,   # Q_exercised
                'time_left': float(state[3]) if len(state) > 3 else 0.0,     # time_to_maturity
                'action': float(action_v.item()) if hasattr(action_v, 'item') else float(action_v)
            }
            
            state, reward, terminated, truncated, info = eval_env.step(action_v)
            
            # Complete step info with results
            step_info['q_actual'] = info.get('q_actual', 0.0)
            step_info['reward'] = float(reward)
            episode_step_data.append(step_info)
            
            disc_return += reward  # Reward already includes discounting
            if info.get('q_actual', 0) > 1e-6:
                exercise_count += 1
                total_exercised += info['q_actual']
            
            step += 1
            
            if terminated or truncated:
                break
        
        # Add this episode's step data to the collection
        all_episodes_step_data.append(episode_step_data)
        
        # Calculate episode statistics
        final_inventory = initial_inventory - total_exercised if initial_inventory > 0 else 0.0
        avg_spot_price = np.mean(spot_prices) if spot_prices else 0.0
        max_spot_price = np.max(spot_prices) if spot_prices else 0.0
        min_spot_price = np.min(spot_prices) if spot_prices else 0.0
        
        # Log raw episode data if CSV file provided
        if raw_episodes_csv is not None and training_episode is not None:
            log_raw_evaluation_episode(
                raw_episodes_csv, training_episode, 1, i, path_seed,
                disc_return, step, total_exercised, exercise_count,
                final_inventory, avg_spot_price, max_spot_price, min_spot_price
            )
        
        discounted_returns.append(disc_return)
        exercise_stats.append({
            'total_exercised': total_exercised,
            'exercise_count': exercise_count,
            'steps': step
        })
    
    # Log detailed step data for ALL episodes in this evaluation run
    if validation_runs_dir is not None and training_episode is not None:
        log_detailed_step_data(validation_runs_dir, training_episode, all_episodes_step_data)
    
    # Calculate statistics
    option_price = np.mean(discounted_returns)
    price_std = np.std(discounted_returns)
    avg_exercised = np.mean([s['total_exercised'] for s in exercise_stats])
    avg_exercises = np.mean([s['exercise_count'] for s in exercise_stats])
    
    return {
        'option_price': option_price,
        'price_std': price_std,
        'confidence_95': 1.96 * price_std / np.sqrt(runs),
        'avg_total_exercised': avg_exercised,
        'avg_exercise_count': avg_exercises,
        'all_returns': discounted_returns,
        'n_runs': runs
    }


def evaluate(path, eval_runs=5, capture=False, render=False, evaluation_csv=None, raw_episodes_csv=None, validation_runs_dir=None):
    """
    Standard evaluation function adapted for swing options
    """
    if capture:
        # Return swing option pricing evaluation for final assessment
        pricing_stats = evaluate_swing_option_price(agent, eval_env, runs=eval_runs, base_seed=args.seed)
        return pricing_stats['all_returns']
    else:
        # Standard evaluation for monitoring training progress
        pricing_stats = evaluate_swing_option_price(
            agent, eval_env, runs=eval_runs, base_seed=args.seed,
            raw_episodes_csv=raw_episodes_csv, training_episode=path, validation_runs_dir=validation_runs_dir
        )
        avg_price = pricing_stats['option_price']
        
        # Log to CSV if file path provided
        if evaluation_csv is not None and path is not None:
            log_evaluation_run(evaluation_csv, path, 1, pricing_stats)
        
        if path is not None:
            writer.add_scalar("Swing_Option_Price", avg_price, path)
            writer.add_scalar("Price_Std", pricing_stats['price_std'], path)
            writer.add_scalar("Avg_Total_Exercised", pricing_stats['avg_total_exercised'], path)
            writer.add_scalar("Avg_Exercise_Count", pricing_stats['avg_exercise_count'], path)
            
            # Print detailed evaluation results
            print(f"\n{'='*50}")
            print(f"EVALUATION RESULTS (Path {path})")
            print(f"{'='*50}")
            print(f"Option Price: {avg_price:.3f} ± {pricing_stats['confidence_95']:.3f}")
            print(f"Price Std Dev: {pricing_stats['price_std']:.3f}")
            print(f"Avg Total Exercised: {pricing_stats['avg_total_exercised']:.3f}")
            print(f"Avg Exercise Count: {pricing_stats['avg_exercise_count']:.1f}")
            print(f"Evaluation Runs: {pricing_stats['n_runs']}")
            print(f"Min Return: {min(pricing_stats['all_returns']):.3f}")
            print(f"Max Return: {max(pricing_stats['all_returns']):.3f}")
            print(f"{'='*50}")
        
        return pricing_stats['all_returns']



def run(n_paths=10000, eval_every=1000, eval_runs=5, training_csv=None, evaluation_csv=None, raw_episodes_csv=None, validation_runs_dir=None):
    """Deep Q-Learning for Swing Option Pricing.
    
    Params
    ======
        n_paths (int): total number of Monte Carlo paths/episodes to simulate
        eval_every (int): evaluate every N paths
        eval_runs (int): number of evaluation runs
        training_csv (str): path to training CSV file for logging
        evaluation_csv (str): path to evaluation CSV file for logging
        raw_episodes_csv (str): path to raw episodes CSV file for logging
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    total_steps = 0                    # total environment interactions across all paths
    
    # Performance monitoring variables
    start_time = time.time()
    
    # Rolling window for performance metrics
    episode_times = deque(maxlen=50)  # timestamps for last 50 episodes
    episode_steps = deque(maxlen=50)  # steps for last 50 episodes
    
    for current_path in range(1, n_paths + 1):
        # evaluation runs
        if current_path % eval_every == 0 or current_path == 1:
            print(f"\n🔍 Starting evaluation at path {current_path} (eval_runs={eval_runs})...")
            evaluate(current_path, eval_runs, evaluation_csv=evaluation_csv, raw_episodes_csv=raw_episodes_csv, validation_runs_dir=validation_runs_dir)

        # Reset environment for new path
        state, _ = train_env.reset()
        score = 0.0
        path_steps = 0
        
        # Complete one full path/episode
        while True:
            total_steps += 1
            path_steps += 1
            
            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)
            next_state, reward, terminated, truncated, _ = train_env.step(action_v[0])
            done = terminated or truncated
            agent.step(state, action_v[0], reward, next_state, done, total_steps, writer)
            
            state = next_state
            score += float(reward)
            
            if done:
                break
        
        # Calculate performance metrics after each completed path
        current_time = time.time()
        total_elapsed = current_time - start_time
        
        # Add current episode data to rolling windows
        episode_times.append(current_time)
        episode_steps.append(path_steps)
        
        # Calculate performance metrics based on rolling window
        if len(episode_times) >= 2:
            # Time span for the episodes in the window
            window_time_span = episode_times[-1] - episode_times[0]
            # Number of episodes in the window
            window_episodes = len(episode_times)
            # Total steps in the window
            window_total_steps = sum(episode_steps)
            
            if window_time_span > 0:
                paths_per_second = (window_episodes - 1) / window_time_span
                steps_per_second = window_total_steps / window_time_span
            else:
                paths_per_second = 0.0
                steps_per_second = 0.0
        else:
            # For the first episode, use current episode metrics
            if total_elapsed > 0:
                paths_per_second = 1.0 / total_elapsed
                steps_per_second = path_steps / total_elapsed
            else:
                paths_per_second = 0.0
                steps_per_second = 0.0
        
        # Performance degradation detection
        if current_path > 100 and current_path % 1000 == 0:
            # Check if performance has degraded significantly
            if hasattr(agent, 'monitor_performance'):
                perf_info = agent.monitor_performance()
                if perf_info:
                    print(f"\n📊 Performance Check (Episode {current_path}):")
                    print(f"  Steps/sec: {steps_per_second:.1f}")
                    print(f"  Memory: {perf_info['memory_mb']:.1f} MB")
                    print(f"  Buffer: {perf_info['buffer_size']:,} ({perf_info['buffer_fill_ratio']*100:.1f}%)")
                    if 'buffer_memory_mb' in perf_info and perf_info['buffer_memory_mb'] > 0:
                        print(f"  Buffer memory: {perf_info['buffer_memory_mb']:.1f} MB")
                    
                    # Force cleanup if performance is poor
                    if steps_per_second < 100 and perf_info['memory_mb'] > 6000:
                        print("⚠️ Low performance detected, forcing memory cleanup...")
                        agent.cleanup_memory(force=True)
        
        # Calculate episode return (accumulated reward for this path)
        episode_return = float(score)
        
        scores_window.append(episode_return)       # save most recent score
        scores.append(episode_return)              # save most recent score
        avg_100 = np.mean(scores_window)
        
        # Log to CSV if file path provided
        if training_csv is not None:
            log_training_episode(training_csv, current_path, episode_return, path_steps, 
                               total_steps, avg_100, paths_per_second, steps_per_second, total_elapsed)
        
        writer.add_scalar("Average100", avg_100, current_path)
        writer.add_scalar("Episode_Return", episode_return, current_path)
        writer.add_scalar("Paths_Per_Second", paths_per_second, current_path)
        writer.add_scalar("Steps_Per_Second", steps_per_second, current_path)
        writer.add_scalar("Total_Steps", total_steps, current_path)
        writer.add_scalar("Path_Length", path_steps, current_path)
        
        print(f'Path {current_path}/{n_paths} | Return = {episode_return:.3f} | Steps = {path_steps} | Paths/sec = {paths_per_second:.1f} | Steps/sec = {steps_per_second:.0f}')
            




parser = argparse.ArgumentParser(description="Swing Option Pricing with D4PG")

# Training Parameters
parser.add_argument("-n_paths", type=int, default=10000, help="The total number of Monte Carlo paths/episodes to simulate, default is 10k")
parser.add_argument("-eval_every", type=int, default=1000, help="Number of paths after which evaluation runs are performed, default = 1000")
parser.add_argument("-eval_runs", type=int, default=1, help="Number of evaluation runs performed, default = 1")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-info", type=str, help="Information or name of the run")

# Swing Option Contract Parameters
parser.add_argument("--q_min", type=float, default=0.0, help="Minimum exercise quantity per period, default = 0.0")
parser.add_argument("--q_max", type=float, default=1.0, help="Maximum exercise quantity per period, default = 1.0")
parser.add_argument("--Q_min", type=float, default=0.0, help="Global minimum total volume, default = 0.0")
parser.add_argument("--Q_max", type=float, default=10.0, help="Global maximum total volume, default = 10.0")
parser.add_argument("--strike", type=float, default=100.0, help="Strike price K, default = 100.0")
parser.add_argument("--maturity", type=float, default=1.0, help="Time to maturity in years, default = 1.0")
parser.add_argument("--n_rights", type=int, default=250, help="Number of decision dates, default = 250")
parser.add_argument("--risk_free_rate", type=float, default=0.05, help="Risk-free rate for discounting, default = 0.05")
parser.add_argument("--min_refraction_days", type=int, default=0, help="Minimum days between exercises, default = 0")

# HHK Stochastic Process Parameters
parser.add_argument("--S0", type=float, default=100.0, help="Initial spot price, default = 100.0")
parser.add_argument("--alpha", type=float, default=7.0, help="Mean reversion speed, default = 7.0")
parser.add_argument("--sigma", type=float, default=1.4, help="Volatility of OU process, default = 1.4")
parser.add_argument("--beta", type=float, default=200.0, help="Jump decay rate, default = 200.0")
parser.add_argument("--lam", type=float, default=4.0, help="Jump intensity (jumps per year), default = 4.0")
parser.add_argument("--mu_J", type=float, default=0.4, help="Mean jump size, default = 0.4")

# D4PG Algorithm Parameters
parser.add_argument("--device", type=str, default="cpu", help="Select trainig device [gpu/cpu], default = cpu")
parser.add_argument("-nstep", type=int, default=1, help ="Nstep bootstrapping, default 1")
parser.add_argument("-per", type=int, default=1, choices=[0,1], help="Adding Priorizied Experience Replay to the agent if set to 1, default = 1")
parser.add_argument("-munchausen", type=int, default=1, choices=[0,1], help="Adding Munchausen RL to the agent if set to 1, default = 1")
parser.add_argument("-iqn", type=int, choices=[0,1], default=0, help="Use distributional IQN Critic if set to 1, default = 0 (no IQN)")
parser.add_argument("-noise", type=str, choices=["ou", "gauss"], default="OU", help="Choose noise type: ou = OU-Noise, gauss = Gaussian noise, default ou")

# Network and Learning Parameters  
parser.add_argument("-lr_a", type=float, default=3e-4, help="Actor learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-lr_c", type=float, default=3e-4, help="Critic learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-learn_every", type=int, default=1, help="Learn every x interactions, default = 1")
parser.add_argument("-learn_number", type=int, default=1, help="Learn x times per interaction, default = 1")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("--max_replay_size", type=int, default=int(1e6), help="Maximum size of the replay buffer, default is 1e6")
parser.add_argument("--min_replay_size", type=int, default=1000, help="Minimum replay buffer size before learning starts (default: 1000)")
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size, default is 128")
parser.add_argument("-t", "--tau", type=float, default=1e-3, help="Softupdate factor tau, default is 1e-3") #for per 1e-2 for regular 1e-3 -> Pendulum!
parser.add_argument("-g", "--gamma", type=float, default=1, help="discount factor gamma, default is 1")

# System and Optimization Parameters
parser.add_argument("-n_cores", type=int, default=None, help="Maximum number of CPU cores to use (default: use all available cores)")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("--compile", type=int, default=0, choices=[0,1], help="Use torch.compile for model optimization, default=0 (NO!)")
parser.add_argument("--fp32", type=int, default=1, choices=[0,1], help="Use float32 precision for better performance, default=1 (YES!)")
parser.add_argument("--use_circular_buffer", type=int, default=1, choices=[0,1], help="Use optimized circular array buffer instead of deque, default=1 (YES!)")

args = parser.parse_args()

# Apply float32 optimization if enabled
if args.fp32:
    torch.set_default_dtype(torch.float32)
    print("Using float32 precision for improved performance")
else:
    print("Using default precision (float64)")

if __name__ == "__main__":
    seed = args.seed
    n_paths = args.n_paths
    GAMMA = args.gamma
    TAU = args.tau
    HIDDEN_SIZE = args.layer_size
    BUFFER_SIZE = int(args.max_replay_size)
    BATCH_SIZE = args.batch_size
    LR_ACTOR = args.lr_a         # learning rate of the actor 
    LR_CRITIC = args.lr_c        # learning rate of the critic
    saved_model = args.saved_model

    # ===============================================
    # SWING OPTION ENVIRONMENT SETUP
    # ===============================================
    
    # Create swing option contract with command line parameters
    from src.swing_contract import SwingContract
    contract = SwingContract(
        q_min=args.q_min,                           # Local minimum exercise per period
        q_max=args.q_max,                           # Local maximum exercise per period
        Q_min=args.Q_min,                           # Global minimum total volume
        Q_max=args.Q_max,                           # Global maximum total volume
        strike=args.strike,                         # Strike price K
        maturity=args.maturity,                     # Time to maturity (years)
        n_rights=args.n_rights,                     # Number of decision dates
        r=args.risk_free_rate,                      # Risk-free rate for discounting
        min_refraction_days=args.min_refraction_days # Minimum days between exercises
    )
    
    # Create HHK stochastic process parameters with command line parameters
    from src.simulate_hhk_spot import default_seasonal_function
    hhk_params = {
        'S0': args.S0,                              # Initial spot price
        'alpha': args.alpha,                        # Mean reversion speed
        'sigma': args.sigma,                        # Volatility of OU process
        'beta': args.beta,                          # Jump decay rate
        'lam': args.lam,                            # Jump intensity (jumps per year)
        'mu_J': args.mu_J,                          # Mean jump size
        'f': default_seasonal_function              # Seasonal function
    }

    writer = SummaryWriter("runs/"+args.info)
    
    # Initialize CSV logging
    training_csv, evaluation_csv, raw_episodes_csv, validation_runs_dir = init_csv_logging(args.info)
    print("CSV logging initialized:")
    print(f"  Training data: {training_csv}")
    print(f"  Evaluation data: {evaluation_csv}")
    print(f"  Raw episodes data: {raw_episodes_csv}")
    print(f"  Validation runs data: {validation_runs_dir}/eval_run_*.csv")
    
    # Create swing option environments with custom parameters
    train_env = SwingOptionEnv(contract=contract, hhk_params=hhk_params)
    eval_env = SwingOptionEnv(contract=contract, hhk_params=hhk_params)
    
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
    torch.set_num_interop_threads(max(1, min(4, n_cores // 2)))
    
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
    temp_env = SwingOptionEnv(contract=contract, hhk_params=hhk_params)
    temp_env.reset(seed=seed)
    action_high = temp_env.action_space.high[0]
    action_low = temp_env.action_space.low[0]
    state_size = temp_env.observation_space.shape[0]
    action_size = temp_env.action_space.shape[0]
    temp_env.close()  # Close temporary environment
    agent = Agent(state_size=state_size, action_size=action_size, n_step=args.nstep, per=args.per, munchausen=args.munchausen,distributional=args.iqn,
                 noise_type=args.noise, random_seed=seed, hidden_size=HIDDEN_SIZE, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE, GAMMA=GAMMA,
                 LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, TAU=TAU, LEARN_EVERY=args.learn_every, LEARN_NUMBER=args.learn_number, device=device_str, paths=args.n_paths, 
                 min_replay_size=args.min_replay_size, use_compile=bool(args.compile), use_circular_buffer=bool(args.use_circular_buffer)) 
    
    t0 = time.time()
    if saved_model is not None:
        agent.actor_local.load_state_dict(torch.load(saved_model))
        evaluate(path=None, capture=False, evaluation_csv=evaluation_csv, raw_episodes_csv=raw_episodes_csv, validation_runs_dir=validation_runs_dir)
    else:    
        run(n_paths=args.n_paths,
            eval_every=args.eval_every,
            eval_runs=args.eval_runs,
            training_csv=training_csv,
            evaluation_csv=evaluation_csv,
            raw_episodes_csv=raw_episodes_csv,
            validation_runs_dir=validation_runs_dir)

    # Final evaluation at the end of training
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    
    final_eval_runs = max(10, args.eval_runs * 2)  # Use more runs for final evaluation
    final_rewards = evaluate(path=args.n_paths, eval_runs=final_eval_runs, capture=True)
    
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
    writer.add_scalar("Final_Evaluation/Average_Return", avg_return, args.n_paths)
    writer.add_scalar("Final_Evaluation/Std_Return", std_return, args.n_paths)
    writer.add_scalar("Final_Evaluation/Min_Return", min_return, args.n_paths)
    writer.add_scalar("Final_Evaluation/Max_Return", max_return, args.n_paths)
    
    print("="*60)

    t1 = time.time()
    eval_env.close()
    timer(t0, t1)
    # save trained model 
    torch.save(agent.actor_local.state_dict(), 'runs/'+args.info+".pth")
    # save parameter
    with open('runs/'+args.info+".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)