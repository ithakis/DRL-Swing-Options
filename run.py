#!/usr/bin/env python3
"""
Swing Option Pricing with D4PG - Deep Deterministic Policy Gradient

This script implements reinforcement learning for swing option pricing using the D4PG algorithm.
It supports configurable swing option contracts, HHK stochastic processes, and comprehensive
evaluation with pre-generated Monte Carlo paths.

Author: Senior AI RL Developer
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
import warnings
from collections import deque
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agent import Agent
from src.swing_env import SwingOptionEnv

# Import LSM pricer for benchmarking
from src.lsm_swing_pricer import LSMSwingPricer

# Suppress the macOS PyTorch profiling warning
warnings.filterwarnings("ignore", message=".*record_context_cpp.*")


class ConfigManager:
    """Handles configuration and argument parsing"""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create and configure argument parser"""
        parser = argparse.ArgumentParser(description="Swing Option Pricing with D4PG")

        # Training Parameters
        parser.add_argument("-n_paths", type=int, default=10000, 
                          help="The total number of Monte Carlo paths/episodes to simulate, default is 10k")
        parser.add_argument("-eval_every", type=int, default=1000, 
                          help="Number of paths after which evaluation runs are performed, default = 1000")
        parser.add_argument("-n_paths_eval", type=int, default=1, 
                          help="Number of evaluation runs performed, default = 1")
        parser.add_argument("-seed", type=int, default=0, 
                          help="Seed for the env and torch network weights, default is 0")
        parser.add_argument("-name", type=str, 
                          help="Name of the run (default: SwingOption_{timestamp})")

        # Swing Option Contract Parameters
        parser.add_argument("--q_min", type=float, default=0.0, 
                          help="Minimum exercise quantity per period, default = 0.0")
        parser.add_argument("--q_max", type=float, default=1.0, 
                          help="Maximum exercise quantity per period, default = 1.0")
        parser.add_argument("--Q_min", type=float, default=0.0, 
                          help="Global minimum total volume, default = 0.0")
        parser.add_argument("--Q_max", type=float, default=10.0, 
                          help="Global maximum total volume, default = 10.0")
        parser.add_argument("--strike", type=float, default=100.0, 
                          help="Strike price K, default = 100.0")
        parser.add_argument("--maturity", type=float, default=1.0, 
                          help="Time to maturity in years, default = 1.0")
        parser.add_argument("--n_rights", type=int, default=250, 
                          help="Number of decision dates, default = 250")
        parser.add_argument("--risk_free_rate", type=float, default=0.05, 
                          help="Risk-free rate for discounting, default = 0.05")
        parser.add_argument("--min_refraction_days", type=int, default=0, 
                          help="Minimum days between exercises, default = 0")

        # HHK Stochastic Process Parameters
        parser.add_argument("--S0", type=float, default=100.0, 
                          help="Initial spot price, default = 100.0")
        parser.add_argument("--alpha", type=float, default=7.0, 
                          help="Mean reversion speed, default = 7.0")
        parser.add_argument("--sigma", type=float, default=1.4, 
                          help="Volatility of OU process, default = 1.4")
        parser.add_argument("--beta", type=float, default=200.0, 
                          help="Jump decay rate, default = 200.0")
        parser.add_argument("--lam", type=float, default=4.0, 
                          help="Jump intensity (jumps per year), default = 4.0")
        parser.add_argument("--mu_J", type=float, default=0.4, 
                          help="Mean jump size, default = 0.4")

        # D4PG Algorithm Parameters
        parser.add_argument("--device", type=str, default="cpu", 
                          help="Select training device [gpu/cpu], default = cpu")
        parser.add_argument("-nstep", type=int, default=1, 
                          help="Nstep bootstrapping, default 1")
        parser.add_argument("-per", type=int, default=1, choices=[0,1], 
                          help="Adding Prioritized Experience Replay to the agent if set to 1, default = 1")
        parser.add_argument("-munchausen", type=int, default=1, choices=[0,1], 
                          help="Adding Munchausen RL to the agent if set to 1, default = 1")
        parser.add_argument("-iqn", type=int, choices=[0,1], default=0, 
                          help="Use distributional IQN Critic if set to 1, default = 0 (no IQN)")
        parser.add_argument("-noise", type=str, choices=["ou", "gauss"], default="OU", 
                          help="Choose noise type: ou = OU-Noise, gauss = Gaussian noise, default ou")

        # Network and Learning Parameters  
        parser.add_argument("-lr_a", type=float, default=3e-4, 
                          help="Actor learning rate of adapting the network weights, default is 3e-4")
        parser.add_argument("-lr_c", type=float, default=3e-4, 
                          help="Critic learning rate of adapting the network weights, default is 3e-4")
        parser.add_argument("-learn_every", type=int, default=1, 
                          help="Learn every x interactions, default = 1")
        parser.add_argument("-learn_number", type=int, default=1, 
                          help="Learn x times per interaction, default = 1")
        parser.add_argument("-layer_size", type=int, default=256, 
                          help="Number of nodes per neural network layer, default is 256")
        parser.add_argument("--max_replay_size", type=int, default=int(1e5), 
                          help="Maximum size of the replay buffer, default is 1e5")
        parser.add_argument("--min_replay_size", type=int, default=1000, 
                          help="Minimum replay buffer size before learning starts (default: 1000)")
        parser.add_argument("-bs", "--batch_size", type=int, default=128, 
                          help="Batch size, default is 128")
        parser.add_argument("-t", "--tau", type=float, default=1e-3, 
                          help="Softupdate factor tau, default is 1e-3")
        parser.add_argument("-g", "--gamma", type=float, default=1, 
                          help="discount factor gamma, default is 1")

        # System and Optimization Parameters
        parser.add_argument("-n_cores", type=int, default=None, 
                          help="Maximum number of CPU cores to use (default: use all available cores)")
        parser.add_argument("--saved_model", type=str, default=None, 
                          help="Load a saved model to perform a test run!")
        parser.add_argument("--compile", type=int, default=0, choices=[0,1], 
                          help="Use torch.compile for model optimization, default=0 (NO!)")
        parser.add_argument("--fp32", type=int, default=1, choices=[0,1], 
                          help="Use float32 precision for better performance, default=1 (YES!)")
        parser.add_argument("--use_circular_buffer", type=int, default=1, choices=[0,1], 
                          help="Use optimized circular array buffer instead of deque, default=1 (YES!)")

        return parser
    
    @staticmethod
    def generate_run_name(name: Optional[str]) -> str:
        """Generate run name with timestamp if not provided"""
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"SwingOption_{timestamp}"
        return name
    
    @staticmethod
    def setup_pytorch_optimizations(args: argparse.Namespace) -> None:
        """Setup PyTorch optimizations based on arguments"""
        # Apply float32 optimization if enabled
        if args.fp32:
            torch.set_default_dtype(torch.float32)
            print("Using float32 precision for improved performance")
        else:
            print("Using default precision (float64)")


class EnvironmentManager:
    """Manages environment setup and configuration"""
    
    @staticmethod
    def create_contract(args: argparse.Namespace):
        """Create swing option contract from arguments"""
        from src.swing_contract import SwingContract
        return SwingContract(
            q_min=args.q_min,
            q_max=args.q_max,
            Q_min=args.Q_min,
            Q_max=args.Q_max,
            strike=args.strike,
            maturity=args.maturity,
            n_rights=args.n_rights,
            r=args.risk_free_rate,
            min_refraction_days=args.min_refraction_days
        )
    
    @staticmethod
    def create_hhk_params(args: argparse.Namespace) -> Dict[str, Any]:
        """Create HHK stochastic process parameters from arguments"""
        from src.simulate_hhk_spot import default_seasonal_function
        return {
            'S0': args.S0,
            'alpha': args.alpha,
            'sigma': args.sigma,
            'beta': args.beta,
            'lam': args.lam,
            'mu_J': args.mu_J,
            'f': default_seasonal_function
        }
    
    @staticmethod
    def setup_device_and_cores(args: argparse.Namespace) -> Tuple[torch.device, str]:
        """Setup device and CPU cores configuration"""
        # Device selection
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
        
        return device, device_str


class LoggingManager:
    """Manages all logging functionality including CSV and TensorBoard"""
    
    @staticmethod
    def save_run_parameters(run_name: str, args_dict: Dict[str, Any], logs_dir: str) -> str:
        """Save run parameters to logs directory"""
        run_logs_dir = os.path.join(logs_dir, run_name)
        os.makedirs(run_logs_dir, exist_ok=True)
        
        params_file = os.path.join(run_logs_dir, f"{run_name}_parameters.json")
        with open(params_file, 'w') as f:
            json.dump(args_dict, f, indent=2)
        
        print(f"Parameters saved to: {params_file}")
        return params_file
    
    @staticmethod
    def init_csv_logging(info_string: str) -> Tuple[str, str, str, str]:
        """Initialize CSV logging files for training, evaluation, and raw evaluation episodes"""
        # Create main logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create experiment-specific directory
        experiment_dir = f"logs/{info_string}"
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create evaluation_runs subdirectory for step-by-step data
        evaluation_runs_dir = f"{experiment_dir}/evaluation_runs"
        os.makedirs(evaluation_runs_dir, exist_ok=True)
        
        # Define CSV file paths
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
        
        return training_csv, evaluation_csv, raw_episodes_csv, evaluation_runs_dir
    
    @staticmethod
    def log_training_episode(csv_file: str, episode: int, episode_return: float, 
                           path_steps: int, total_steps: int, avg_100: float, 
                           paths_per_sec: float, steps_per_sec: float, time_elapsed: float) -> None:
        """Log training episode data to CSV"""
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, round(episode_return, 6), path_steps, total_steps, 
                round(avg_100, 6), round(paths_per_sec, 3), round(steps_per_sec, 1), 
                round(time_elapsed, 2)
            ])
    
    @staticmethod
    def log_evaluation_run(csv_file: str, training_episode: int, eval_run: int, 
                          pricing_stats: Dict[str, Any]) -> None:
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
    
    @staticmethod
    def log_raw_evaluation_episode(csv_file: str, training_episode: int, eval_run: int, 
                                 episode_idx: int, seed: int, episode_return: float, 
                                 steps: int, total_exercised: float, exercise_count: int,
                                 final_inventory: float, avg_spot_price: float, 
                                 max_spot_price: float, min_spot_price: float) -> None:
        """Log individual raw evaluation episode data to CSV"""
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                training_episode, eval_run, episode_idx, seed, round(episode_return, 6),
                steps, round(total_exercised, 6), exercise_count, round(final_inventory, 6),
                round(avg_spot_price, 6), round(max_spot_price, 6), round(min_spot_price, 6)
            ])
    
    @staticmethod
    def log_detailed_step_data(evaluation_runs_dir: str, training_episode: int, 
                             all_episodes_data: List[List[Dict[str, Any]]]) -> None:
        """Log detailed step-by-step data for all episodes in an evaluation run to one CSV file"""
        filename = f"eval_run_{training_episode}.csv"
        filepath = os.path.join(evaluation_runs_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode_idx', 'step', 'spot', 'q_remain', 'q_exerc', 
                           'time_left', 'action', 'q_actual', 'reward'])
            
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
                        round(step_info['q_actual'], 4),
                        round(step_info['reward'], 6)
                    ])
    
    @staticmethod
    def save_lsm_benchmark_results(run_name: str, lsm_results: Dict[str, Any], logs_dir: str) -> str:
        """Save LSM benchmark results to logs directory"""
        run_logs_dir = os.path.join(logs_dir, run_name)
        os.makedirs(run_logs_dir, exist_ok=True)
        
        lsm_file = os.path.join(run_logs_dir, f"{run_name}_lsm_benchmark.json")
        with open(lsm_file, 'w') as f:
            # Remove numpy arrays for JSON serialization
            serializable_results = {
                'lsm_option_price': float(lsm_results['lsm_option_price']),
                'lsm_std_error': float(lsm_results['lsm_std_error']),
                'lsm_confidence_95': float(lsm_results['lsm_confidence_95']),
                'lsm_pricing_time': float(lsm_results['lsm_pricing_time']),
                'lsm_paths_used': int(lsm_results['lsm_paths_used'])
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"LSM benchmark results saved to: {lsm_file}")
        return lsm_file


class EvaluationManager:
    """Manages evaluation functionality for swing option pricing"""
    
    @staticmethod
    def evaluate_swing_option_price_pregenerated(agent, eval_env, runs: int = 100, 
                                               raw_episodes_csv: Optional[str] = None,
                                               training_episode: Optional[int] = None,
                                               evaluation_runs_dir: Optional[str] = None,
                                               eval_t=None, eval_S=None, eval_X=None, eval_Y=None) -> Dict[str, Any]:
        """
        Evaluate swing option price using pre-generated Monte Carlo paths
        
        Args:
            agent: Trained D4PG agent
            eval_env: Swing option environment for evaluation
            runs: Number of Monte Carlo runs (should match pre-generated paths)
            raw_episodes_csv: Path to raw episodes CSV file for detailed logging
            training_episode: Current training episode number for logging
            evaluation_runs_dir: Directory for detailed step-by-step CSV files
            eval_t: Pre-generated time grid
            eval_S: Pre-generated spot price paths
            eval_X: Pre-generated X process paths
            eval_Y: Pre-generated Y process paths
            
        Returns:
            dict: Dictionary with pricing statistics
        """
        discounted_returns = []
        exercise_stats = []
        all_episodes_step_data = []
        
        # Use min to handle case where we have fewer pre-generated paths than requested runs
        actual_runs = min(runs, eval_S.shape[0]) if eval_S is not None else runs
        
        for i in range(actual_runs):
            # Use pre-generated path for this evaluation run
            if eval_S is not None:
                eval_env.set_pregenerated_path(eval_t, eval_S[i], eval_X[i], eval_Y[i])
            
            state, _ = eval_env.reset()
            
            disc_return = 0.0
            total_exercised = 0.0
            exercise_count = 0
            step = 0
            spot_prices = []
            episode_step_data = []
            
            # Track initial inventory for final calculation
            initial_inventory = eval_env.contract.Q_max if hasattr(eval_env, 'contract') else 0.0
            
            while True:
                # Get action from agent
                action = agent.act(np.expand_dims(state, axis=0))
                action_v = np.clip(action, 0.0, 1.0)  # Ensure valid action range
                
                # Track spot price if available in state
                if len(state) > 0:
                    spot_prices.append(float(state[0] * eval_env.contract.strike))
                
                # Store step data before taking action
                step_info = {
                    'step': step,
                    'spot_price': float(state[0] * eval_env.contract.strike) if len(state) > 0 else 0.0,
                    'q_remaining': float(state[2] * eval_env.contract.Q_max) if len(state) > 2 else 0.0,
                    'q_exercised': float(state[1] * eval_env.contract.Q_max) if len(state) > 1 else 0.0,
                    'time_left': float(state[3] * eval_env.contract.maturity) if len(state) > 3 else 0.0,
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
                path_seed = training_episode * 10000 + i + 1
                LoggingManager.log_raw_evaluation_episode(
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
        if evaluation_runs_dir is not None and training_episode is not None:
            LoggingManager.log_detailed_step_data(evaluation_runs_dir, training_episode, all_episodes_step_data)
        
        # Calculate statistics
        option_price = np.mean(discounted_returns)
        price_std = np.std(discounted_returns)
        avg_exercised = np.mean([s['total_exercised'] for s in exercise_stats])
        avg_exercises = np.mean([s['exercise_count'] for s in exercise_stats])
        
        return {
            'option_price': option_price,
            'price_std': price_std,
            'confidence_95': 1.96 * price_std / np.sqrt(actual_runs),
            'avg_total_exercised': avg_exercised,
            'avg_exercise_count': avg_exercises,
            'all_returns': discounted_returns,
            'n_runs': actual_runs
        }
    
    @staticmethod
    def evaluate_with_pregenerated_paths(agent, eval_env, writer, path: int, n_paths_eval: int = 5, 
                                       evaluation_csv: Optional[str] = None, 
                                       raw_episodes_csv: Optional[str] = None, 
                                       evaluation_runs_dir: Optional[str] = None, 
                                       eval_t=None, eval_S=None, eval_X=None, eval_Y=None) -> List[float]:
        """Evaluation function using pre-generated paths"""
        pricing_stats = EvaluationManager.evaluate_swing_option_price_pregenerated(
            agent, eval_env, runs=n_paths_eval, 
            raw_episodes_csv=raw_episodes_csv, training_episode=path, 
            evaluation_runs_dir=evaluation_runs_dir,
            eval_t=eval_t, eval_S=eval_S, eval_X=eval_X, eval_Y=eval_Y
        )
        avg_price = pricing_stats['option_price']
        
        # Log to CSV if file path provided
        if evaluation_csv is not None and path is not None:
            LoggingManager.log_evaluation_run(evaluation_csv, path, 1, pricing_stats)
        
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


class DatasetManager:
    """Manages dataset generation and pre-generated paths"""
    
    @staticmethod
    def generate_datasets(train_env, eval_env, n_paths: int, n_paths_eval: int, 
                         seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                                          np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate training and evaluation datasets"""
        print(f"\n{'='*60}")
        print(f"DATASET GENERATION")
        print(f"{'='*60}")
        print(f"📊 FIXED DATASET APPROACH:")
        print(f"   - Creating TRAIN dataset with seed={seed}")
        print(f"   - Creating VALIDATION dataset with seed={seed + 1}")
        print(f"   - No paths generated during training/evaluation")
        print(f"   - Same validation set used for all evaluations")
        print(f"\n🎲 Generating Training Dataset:")
        print(f"   - Seed: {seed}")
        print(f"   - Paths: {n_paths}")
        print(f"   - Purpose: RL Agent Training")
        print(f"\n🎲 Generating Validation Dataset:")
        print(f"   - Seed: {seed + 1}")
        print(f"   - Paths: {n_paths_eval}")  
        print(f"   - Purpose: RL Evaluation & LSM Benchmark")
        print(f"{'='*60}")
        
        pre_generation_start = time.time()
        
        # Generate training paths with base seed
        from src.simulate_hhk_spot import simulate_hhk_spot
        print(f"Generating training paths...")
        training_t, training_S, training_X, training_Y = simulate_hhk_spot(
            T=train_env.contract.maturity,
            n_steps=train_env.contract.n_rights,
            n_paths=n_paths,
            seed=seed,
            **train_env.hhk_params
        )
        
        # Generate evaluation paths with seed+1 for different quasi-random sequence
        print(f"Generating evaluation paths...")
        eval_t, eval_S, eval_X, eval_Y = simulate_hhk_spot(
            T=eval_env.contract.maturity,
            n_steps=eval_env.contract.n_rights,
            n_paths=n_paths_eval,
            seed=seed + 1,
            **eval_env.hhk_params
        )
        
        pre_generation_time = time.time() - pre_generation_start
        print(f"✅ Dataset generation completed in {pre_generation_time:.2f}s")
        print(f"   Training paths: {training_S.shape[0]} x {training_S.shape[1]} steps")
        print(f"   Evaluation paths: {eval_S.shape[0]} x {eval_S.shape[1]} steps")
        
        return training_t, training_S, training_X, training_Y, eval_t, eval_S, eval_X, eval_Y


class TrainingManager:
    """Manages the main training loop and performance monitoring"""
    
    @staticmethod
    def timer(start: float, end: float) -> None:
        """Helper to print training time"""
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    
    @staticmethod
    def should_evaluate(current_path: int, n_paths: int, eval_every: int, 
                       initial_eval_done: bool) -> Tuple[bool, bool]:
        """Determine if evaluation should be performed at current path"""
        should_evaluate = False
        new_initial_eval_done = initial_eval_done
        
        if eval_every == 0:
            # Error case: eval_every cannot be 0
            raise ValueError("eval_every cannot be 0. Use -1 for end-only evaluation or positive value for periodic evaluation.")
        elif eval_every == -1:
            # Only evaluate at the end (after the last episode has completed)
            should_evaluate = (current_path == n_paths)
        elif eval_every > 0:
            # Run initial evaluation at path 1 (with 0 episodes completed)
            if current_path == 1 and not initial_eval_done:
                should_evaluate = True
                new_initial_eval_done = True
            # Run evaluation every eval_every episodes after training
            elif current_path > 1 and (current_path - 1) % eval_every == 0:
                should_evaluate = True
            # Run final evaluation if n_paths is not aligned with eval_every schedule
            elif current_path == n_paths and (n_paths - 1) % eval_every != 0:
                should_evaluate = True
        
        return should_evaluate, new_initial_eval_done
    
    @staticmethod
    def calculate_performance_metrics(episode_times: deque, episode_steps: deque, 
                                    start_time: float, path_steps: int) -> Tuple[float, float]:
        """Calculate performance metrics"""
        current_time = time.time()
        total_elapsed = current_time - start_time
        
        # Add current episode data to rolling windows
        episode_times.append(current_time)
        episode_steps.append(path_steps)
        
        # Calculate performance metrics based on rolling window
        if len(episode_times) >= 2:
            window_time_span = episode_times[-1] - episode_times[0]
            window_episodes = len(episode_times)
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
        
        return paths_per_second, steps_per_second
    
    @staticmethod
    def monitor_performance(agent, current_path: int, steps_per_second: float) -> None:
        """Monitor and handle performance degradation"""
        if current_path > 100 and current_path % 1000 == 0:
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


def run_training(agent, train_env, eval_env, writer, args: argparse.Namespace,
                action_low: float, action_high: float,
                training_t, training_S, training_X, training_Y,
                eval_t, eval_S, eval_X, eval_Y,
                training_csv: str, evaluation_csv: str, raw_episodes_csv: str, 
                evaluation_runs_dir: str) -> None:
    """
    Main training function for Deep Q-Learning for Swing Option Pricing
    
    Args:
        agent: D4PG agent
        train_env: Training environment
        eval_env: Evaluation environment
        writer: TensorBoard writer
        args: Parsed command line arguments
        action_low: Minimum action value
        action_high: Maximum action value
        training_t, training_S, training_X, training_Y: Training dataset
        eval_t, eval_S, eval_X, eval_Y: Evaluation dataset
        training_csv: Path to training CSV file
        evaluation_csv: Path to evaluation CSV file
        raw_episodes_csv: Path to raw episodes CSV file
        evaluation_runs_dir: Directory for evaluation runs
    """
    scores = []
    scores_window = deque(maxlen=100)
    total_steps = 0
    
    print(f"\n{'='*60}")
    print(f"STARTING RL AGENT TRAINING")
    print(f"{'='*60}")
    print(f"Training Episodes: {args.n_paths}")
    print(f"Evaluation Every: {args.eval_every if args.eval_every > 0 else 'End Only'}")
    print(f"Evaluation Paths: {args.n_paths_eval}")
    print(f"{'='*60}")
    
    # Performance monitoring variables
    start_time = time.time()
    episode_times = deque(maxlen=50)
    episode_steps = deque(maxlen=50)
    initial_eval_done = False
    
    # Validate eval_every parameter
    if args.eval_every == 0:
        raise ValueError("eval_every cannot be 0. Use -1 for end-only evaluation or positive value for periodic evaluation.")
    
    for current_path in range(1, args.n_paths + 1):
        # Initial evaluation before any training (only for eval_every > 0)
        if current_path == 1 and args.eval_every > 0 and not initial_eval_done:
            print(f"\n🔍 Starting INITIAL evaluation before training (0 episodes completed)...")
            print(f"   Using VALIDATION dataset (same paths used for all evaluations)")
            EvaluationManager.evaluate_with_pregenerated_paths(
                agent, eval_env, writer, 0, args.n_paths_eval, 
                evaluation_csv=evaluation_csv, raw_episodes_csv=raw_episodes_csv, 
                evaluation_runs_dir=evaluation_runs_dir, eval_t=eval_t, 
                eval_S=eval_S, eval_X=eval_X, eval_Y=eval_Y
            )
            initial_eval_done = True

        # Use pre-generated training path for this episode
        path_idx = (current_path - 1) % training_S.shape[0]
        train_env.set_pregenerated_path(training_t, training_S[path_idx], 
                                       training_X[path_idx], training_Y[path_idx])
        
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
        
        # Calculate performance metrics
        paths_per_sec, steps_per_sec = TrainingManager.calculate_performance_metrics(
            episode_times, episode_steps, start_time, path_steps
        )
        
        # Performance monitoring
        TrainingManager.monitor_performance(agent, current_path, steps_per_sec)
        
        # Calculate episode return and update tracking
        episode_return = float(score)
        scores_window.append(episode_return)
        scores.append(episode_return)
        avg_100 = np.mean(scores_window)
        
        # Logging
        total_elapsed = time.time() - start_time
        LoggingManager.log_training_episode(
            training_csv, current_path, episode_return, path_steps, 
            total_steps, avg_100, paths_per_sec, steps_per_sec, total_elapsed
        )
        
        # TensorBoard logging
        writer.add_scalar("Average100", avg_100, current_path)
        writer.add_scalar("Episode_Return", episode_return, current_path)
        writer.add_scalar("Paths_Per_Second", paths_per_sec, current_path)
        writer.add_scalar("Steps_Per_Second", steps_per_sec, current_path)
        writer.add_scalar("Total_Steps", total_steps, current_path)
        writer.add_scalar("Path_Length", path_steps, current_path)
        
        print(f'Path {current_path}/{args.n_paths} | Return = {episode_return:.3f} | '
              f'Steps = {path_steps} | Paths/sec = {paths_per_sec:.1f} | '
              f'Steps/sec = {steps_per_sec:.0f}')

        # Post-training evaluation logic
        should_eval_after = False
        
        if args.eval_every == -1:
            # Only evaluate after the last episode has completed
            should_eval_after = (current_path == args.n_paths)
        elif args.eval_every > 0:
            # Evaluate after every eval_every episodes have completed
            if current_path % args.eval_every == 0:
                should_eval_after = True
            # Also evaluate at the end if not aligned with eval_every schedule
            elif current_path == args.n_paths and current_path % args.eval_every != 0:
                should_eval_after = True
        
        if should_eval_after:
            print(f"\n🔍 Starting evaluation after {current_path} episodes completed...")
            print(f"   Using VALIDATION dataset (same paths used for all evaluations)")
            EvaluationManager.evaluate_with_pregenerated_paths(
                agent, eval_env, writer, current_path, args.n_paths_eval, 
                evaluation_csv=evaluation_csv, raw_episodes_csv=raw_episodes_csv, 
                evaluation_runs_dir=evaluation_runs_dir, eval_t=eval_t, 
                eval_S=eval_S, eval_X=eval_X, eval_Y=eval_Y
            )


class LSMBenchmarkManager:
    """Manages LSM benchmarking functionality"""
    
    @staticmethod
    def convert_swing_contract_to_lsm(contract):
        """Convert RL swing contract to LSM contract format - no conversion needed"""
        return contract  # Already using SwingContract class
    
    @staticmethod
    def convert_hhk_params_to_lsm(hhk_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert RL HHK parameters to LSM format - no conversion needed"""
        return hhk_params  # Already using Dict format
    
    @staticmethod
    def run_lsm_benchmark_with_pregenerated_paths(contract, hhk_params: Dict[str, Any], 
                                                 eval_t, eval_S, eval_X, eval_Y, 
                                                 n_paths_eval: int, seed: int,
                                                 evaluation_runs_dir: Optional[str] = None,
                                                 run_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run LSM pricing using the same pre-generated paths as RL evaluation
        
        Args:
            contract: RL swing contract
            hhk_params: RL HHK parameters dictionary
            eval_t, eval_S, eval_X, eval_Y: Pre-generated evaluation paths
            n_paths_eval: Number of evaluation paths
            seed: Random seed for LSM
            evaluation_runs_dir: Directory to save evaluation run CSV data
            run_name: Name of the run for file naming
            
        Returns:
            Dict with LSM pricing results
        """
        print(f"\n{'='*60}")
        print(f"LSM BENCHMARK PRICING")
        print(f"{'='*60}")
        print(f"Using same evaluation dataset as RL agent:")
        print(f"   - Paths: {n_paths_eval}")
        print(f"   - Steps: {eval_S.shape[1]} decision dates")
        print(f"   - Same seed sequence as RL evaluation")
        print(f"{'='*60}")
        
        # Convert parameters to LSM format
        lsm_contract = LSMBenchmarkManager.convert_swing_contract_to_lsm(contract)
        lsm_hhk_params = LSMBenchmarkManager.convert_hhk_params_to_lsm(hhk_params)
        
        # Create LSM pricer - we'll override its path generation
        pricer = LSMSwingPricer(
            contract=lsm_contract,
            hhk_params=lsm_hhk_params,
            n_paths=n_paths_eval,
            poly_degree=3,
            seed=seed
        )
        
        # Price option using pre-generated paths
        print("🔍 Running LSM pricing algorithm...")
        lsm_start_time = time.time()
        
        # Use the new method to price with pre-generated paths
        results = pricer.price_option_with_pregenerated_paths(
            eval_t=eval_t,
            eval_S=eval_S,
            eval_X=eval_X,
            eval_Y=eval_Y,
            verbose=True
        )
        
        lsm_time = time.time() - lsm_start_time
        
        # Print results
        print(f"✅ LSM pricing completed in {lsm_time:.2f}s")
        print(f"{'='*50}")
        print(f"LSM BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"Option Price: {results['option_price']:.6f}")
        print(f"Standard Error: {results['option_std_error']:.6f}")
        print(f"95% Confidence Interval: [{results['option_price'] - 1.96*results['option_std_error']:.6f}, {results['option_price'] + 1.96*results['option_std_error']:.6f}]")
        print(f"Paths Used: {n_paths_eval}")
        print(f"Decision Dates: {lsm_contract.n_rights}")
        print(f"Contract Terms: Strike={lsm_contract.strike}, Q_max={lsm_contract.Q_max}")
        print(f"{'='*50}")
        
        # Simulate optimal strategy and save step-by-step data if requested
        if evaluation_runs_dir and run_name:
            print(f"\nSaving LSM step-by-step evaluation data...")
            strategy_results = pricer.simulate_optimal_strategy(
                results=results,
                n_scenarios=n_paths_eval,  # Use full evaluation dataset
                plot=False,
                save_step_data=True,
                evaluation_runs_dir=evaluation_runs_dir,
                run_name=f"LSM_{run_name}"
            )
        
        return {
            'lsm_option_price': results['option_price'],
            'lsm_std_error': results['option_std_error'],
            'lsm_confidence_95': 1.96 * results['option_std_error'],
            'lsm_pricing_time': lsm_time,
            'lsm_paths_used': n_paths_eval,
            'lsm_results': results
        }


def main():
    """Main execution function"""
    # Parse arguments and setup
    parser = ConfigManager.create_parser()
    args = parser.parse_args()
    
    # Generate run name
    run_name = ConfigManager.generate_run_name(args.name)
    print(f"Run name: {run_name}")
    
    # Setup PyTorch optimizations
    ConfigManager.setup_pytorch_optimizations(args)
    
    # Seed setup
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environments and contracts
    contract = EnvironmentManager.create_contract(args)
    hhk_params = EnvironmentManager.create_hhk_params(args)
    
    # Initialize logging
    LoggingManager.save_run_parameters(run_name, args.__dict__, "logs")
    training_csv, evaluation_csv, raw_episodes_csv, evaluation_runs_dir = LoggingManager.init_csv_logging(run_name)
    
    print("CSV logging initialized:")
    print(f"  Training data: {training_csv}")
    print(f"  Evaluation data: {evaluation_csv}")
    print(f"  Raw episodes data: {raw_episodes_csv}")
    print(f"  Evaluation runs data: {evaluation_runs_dir}/eval_run_*.csv")
    
    # Create environments
    train_env = SwingOptionEnv(contract=contract, hhk_params=hhk_params)
    eval_env = SwingOptionEnv(contract=contract, hhk_params=hhk_params)
    
    # Seed environments
    train_env.reset(seed=seed)
    eval_env.reset(seed=seed+1)
    
    # Setup device and cores
    device, device_str = EnvironmentManager.setup_device_and_cores(args)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter("runs/"+run_name)
    
    # Get environment specs
    temp_env = SwingOptionEnv(contract=contract, hhk_params=hhk_params)
    temp_env.reset(seed=seed)
    action_high = temp_env.action_space.high[0]
    action_low = temp_env.action_space.low[0]
    state_size = temp_env.observation_space.shape[0]
    action_size = temp_env.action_space.shape[0]
    temp_env.close()
    
    # Create agent
    agent = Agent(
        state_size=state_size, action_size=action_size, n_step=args.nstep, 
        per=args.per, munchausen=args.munchausen, distributional=args.iqn,
        noise_type=args.noise, random_seed=seed, hidden_size=args.layer_size, 
        BATCH_SIZE=args.batch_size, BUFFER_SIZE=int(args.max_replay_size), 
        GAMMA=args.gamma, LR_ACTOR=args.lr_a, LR_CRITIC=args.lr_c, TAU=args.tau, 
        LEARN_EVERY=args.learn_every, LEARN_NUMBER=args.learn_number, device=device_str, 
        paths=args.n_paths, min_replay_size=args.min_replay_size, 
        use_compile=bool(args.compile), use_circular_buffer=bool(args.use_circular_buffer)
    )
    
    # Training execution
    t0 = time.time()
    
    if args.saved_model is not None:
        agent.actor_local.load_state_dict(torch.load(args.saved_model))
        print("WARNING: Pre-generated paths not available for saved model evaluation.")
    else:    
        # Generate datasets
        training_t, training_S, training_X, training_Y, eval_t, eval_S, eval_X, eval_Y = DatasetManager.generate_datasets(
            train_env, eval_env, args.n_paths, args.n_paths_eval, seed
        )
        
        # Run LSM benchmark pricing using the same evaluation paths as RL agent
        lsm_benchmark_results = LSMBenchmarkManager.run_lsm_benchmark_with_pregenerated_paths(
            contract=contract, 
            hhk_params=hhk_params,
            eval_t=eval_t, 
            eval_S=eval_S, 
            eval_X=eval_X, 
            eval_Y=eval_Y,
            n_paths_eval=args.n_paths_eval,
            seed=seed + 1,  # Same seed as evaluation dataset
            evaluation_runs_dir=evaluation_runs_dir,
            run_name=run_name
        )
        
        # Run training
        run_training(
            agent, train_env, eval_env, writer, args, action_low, action_high,
            training_t, training_S, training_X, training_Y,
            eval_t, eval_S, eval_X, eval_Y,
            training_csv, evaluation_csv, raw_episodes_csv, evaluation_runs_dir
        )

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)

    t1 = time.time()
    eval_env.close()
    TrainingManager.timer(t0, t1)
    
    # Save trained model 
    torch.save(agent.actor_local.state_dict(), 'runs/'+run_name+".pth")
    
    # Save parameters
    with open('runs/'+run_name+".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    main()