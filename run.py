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
import shutil
import time
import warnings
from collections import deque
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agent import Agent
from src.fdm_swing_pricer import price_swing_option_fdm
from src.lsm_swing_pricer import price_swing_option_lsm

# Import LSM pricer for benchmarking
from src.simulate_hhk_spot import no_seasonal_function, simulate_hhk_spot
from src.swing_contract import SwingContract
from src.swing_env import SwingOptionEnv

# Suppress the macOS PyTorch profiling warning
warnings.filterwarnings("ignore", message=".*record_context_cpp.*")


class AsyncCSVWriter:
    """Asynchronous CSV writer to avoid blocking main execution"""

    def __init__(self):
        self.write_queue = Queue()
        self.writer_thread = None
        self.is_running = False

    def start(self):
        """Start the background writer thread"""
        if not self.is_running:
            self.is_running = True
            self.writer_thread = Thread(target=self._write_worker, daemon=True)
            self.writer_thread.start()

    def stop(self):
        """Stop the background writer thread"""
        if self.is_running:
            self.write_queue.put(None)  # Sentinel to stop thread
            if self.writer_thread:
                self.writer_thread.join(timeout=5.0)
            self.is_running = False

    def _write_worker(self):
        """Background worker that processes the write queue"""
        while True:
            item = self.write_queue.get()
            if item is None:  # Sentinel to stop
                break

            try:
                filepath, data, headers = item

                # Check if file exists to determine if we need headers
                file_exists = os.path.exists(filepath)

                with open(filepath, "a", newline="") as f:
                    writer = csv.writer(f)

                    # Write headers if file is new
                    if not file_exists and headers:
                        writer.writerow(headers)

                    # Write data rows
                    if isinstance(data[0], (list, tuple)):
                        # Multiple rows
                        writer.writerows(data)
                    else:
                        # Single row
                        writer.writerow(data)

            except Exception as e:
                print(f"Error writing to CSV: {e}")
            finally:
                self.write_queue.task_done()

    def write_csv(self, filepath: str, data, headers=None):
        """Queue data to be written to CSV file"""
        if self.is_running:
            self.write_queue.put((filepath, data, headers))


class ConfigManager:
    """Handles configuration and argument parsing"""

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create and configure argument parser"""
        parser = argparse.ArgumentParser(description="Swing Option Pricing with D4PG")

        # Training Parameters
        parser.add_argument(
            "-n_paths",
            type=int,
            default=10000,
            help="The total number of Monte Carlo paths/episodes to simulate, default is 10k",
        )
        parser.add_argument(
            "-eval_every",
            type=int,
            default=1000,
            help="Number of paths after which evaluation runs are performed, default = 1000",
        )
        parser.add_argument(
            "-n_paths_eval", type=int, default=1, help="Number of evaluation runs performed, default = 1"
        )
        parser.add_argument(
            "-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0"
        )
        parser.add_argument("-name", type=str, help="Name of the run (default: SwingOption_{timestamp})")

        # Swing Option Contract Parameters
        parser.add_argument(
            "--q_min", type=float, default=0.0, help="Minimum exercise quantity per period, default = 0.0"
        )
        parser.add_argument(
            "--q_max", type=float, default=1.0, help="Maximum exercise quantity per period, default = 1.0"
        )
        parser.add_argument(
            "--Q_min", type=float, default=0.0, help="Global minimum total volume, default = 0.0"
        )
        parser.add_argument(
            "--Q_max", type=float, default=10.0, help="Global maximum total volume, default = 10.0"
        )
        parser.add_argument("--strike", type=float, default=100.0, help="Strike price K, default = 100.0")
        parser.add_argument(
            "--maturity", type=float, default=1.0, help="Time to maturity in years, default = 1.0"
        )
        parser.add_argument("--n_rights", type=int, default=250, help="Number of decision dates, default = 250")
        parser.add_argument(
            "--risk_free_rate", type=float, default=0.05, help="Risk-free rate for discounting, default = 0.05"
        )
        parser.add_argument(
            "--min_refraction_days", type=int, default=0, help="Minimum days between exercises, default = 0"
        )

        # HHK Stochastic Process Parameters
        parser.add_argument("--S0", type=float, default=100.0, help="Initial spot price, default = 100.0")
        parser.add_argument("--alpha", type=float, default=7.0, help="Mean reversion speed, default = 7.0")
        parser.add_argument("--sigma", type=float, default=1.4, help="Volatility of OU process, default = 1.4")
        parser.add_argument("--beta", type=float, default=200.0, help="Jump decay rate, default = 200.0")
        parser.add_argument(
            "--lam", type=float, default=4.0, help="Jump intensity (jumps per year), default = 4.0"
        )
        parser.add_argument("--mu_J", type=float, default=0.4, help="Mean jump size, default = 0.4")

        # D4PG Algorithm Parameters
        parser.add_argument(
            "--device", type=str, default="cpu", help="Select training device [gpu/cpu], default = cpu"
        )
        parser.add_argument("-nstep", type=int, default=1, help="Nstep bootstrapping, default 1")
        # PER hyperparameters
        parser.add_argument("--per_alpha", type=float, default=0.6, help="PER: priority exponent alpha (default: 0.6)")
        parser.add_argument("--per_beta_start", type=float, default=0.4, help="PER: initial importance sampling weight beta_start (default: 0.4)")
        parser.add_argument("--per_beta_frames", type=int, default=100000, help="PER: frames to anneal beta to 1.0 (default: 100000)")
        parser.add_argument(
            "-per",
            type=int,
            default=1,
            choices=[0, 1],
            help="Adding Prioritized Experience Replay to the agent if set to 1, default = 1",
        )
        parser.add_argument(
            "-munchausen",
            type=int,
            default=1,
            choices=[0, 1],
            help="Adding Munchausen RL to the agent if set to 1, default = 1",
        )
        parser.add_argument(
            "-iqn",
            type=int,
            choices=[0, 1],
            default=0,
            help="Use distributional IQN Critic if set to 1, default = 0 (no IQN)",
        )
        parser.add_argument(
            "-noise",
            type=str,
            choices=["ou", "gauss"],
            default="OU",
            help="Choose noise type: ou = OU-Noise, gauss = Gaussian noise, default ou",
        )

        # Network and Learning Parameters
        parser.add_argument(
            "-lr_a",
            type=float,
            default=3e-4,
            help="Actor learning rate of adapting the network weights, default is 3e-4",
        )
        parser.add_argument(
            "-lr_c",
            type=float,
            default=3e-4,
            help="Critic learning rate of adapting the network weights, default is 3e-4",
        )
        parser.add_argument("-learn_every", type=int, default=1, help="Learn every x interactions, default = 1")
        parser.add_argument(
            "-learn_number", type=int, default=1, help="Learn x times per interaction, default = 1"
        )
        parser.add_argument(
            "-layer_size",
            type=int,
            default=256,
            help="Number of nodes per neural network layer, default is 256",
        )
        parser.add_argument(
            "-epsilon",
            type=float,
            default=0.3,
            help="Initial epsilon for exploration noise (default: 0.3)",
        )
        parser.add_argument(
            "-epsilon_decay",
            type=float,
            default=1.0,
            help="Epsilon decay rate per episode (default: 1.0)",
        )
        parser.add_argument(
            "--final_lr_fraction",
            type=float,
            default=1.0,
            help="Final learning rate as fraction of initial LR (1.0=no decay, 0.1=decay to 10%)",
        )
        parser.add_argument(
            "--warmup_frac",
            type=float,
            default=0.05,
            help="Fraction of total episodes for LR warm-up (default: 0.05 = 5%)",
        )
        parser.add_argument(
            "--min_lr",
            type=float,
            default=1e-7,
            help="Minimum learning rate floor (default: 1e-7)",
        )
        parser.add_argument(
            "--max_replay_size",
            type=int,
            default=int(1e5),
            help="Maximum size of the replay buffer, default is 1e5",
        )
        parser.add_argument(
            "--min_replay_size",
            type=int,
            default=1000,
            help="Minimum replay buffer size before learning starts (default: 1000)",
        )
        parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size, default is 128")
        parser.add_argument(
            "-t", "--t", type=float, default=1e-3, help="Softupdate factor t, default is 1e-3"
        )
        parser.add_argument("-g", "--gamma", type=float, default=1, help="discount factor gamma, default is 1")

        # System and Optimization Parameters
        parser.add_argument(
            "-n_cores",
            type=int,
            default=None,
            help="Maximum number of CPU cores to use (default: use all available cores)",
        )
        parser.add_argument(
            "--saved_model", type=str, default=None, help="Load a saved model to perform a test run!"
        )
        parser.add_argument(
            "--compile",
            type=int,
            default=0,
            choices=[0, 1],
            help="Use torch.compile for model optimization, default=0 (NO!)",
        )
        parser.add_argument(
            "--fp32",
            type=int,
            default=1,
            choices=[0, 1],
            help="Use float32 precision for better performance, default=1 (YES!)",
        )

        return parser

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
        os.environ["OMP_NUM_THREADS"] = str(n_cores)
        os.environ["MKL_NUM_THREADS"] = str(n_cores)

        # Configure PyTorch threading
        torch.set_num_threads(n_cores)
        torch.set_num_interop_threads(max(1, min(4, n_cores // 2)))

        # Enable optimized CPU kernels if available
        try:
            if hasattr(torch.backends, "mkl") and hasattr(torch.backends.mkl, "enabled"):
                torch.backends.mkl.enabled = True
        except Exception:
            pass
        try:
            if hasattr(torch.backends, "mkldnn") and hasattr(torch.backends.mkldnn, "enabled"):
                torch.backends.mkldnn.enabled = True
        except Exception:
            pass

        print(
            "PyTorch optimized for {} CPU cores (intra: {}, inter: {})".format(
                n_cores, torch.get_num_threads(), torch.get_num_interop_threads()
            )
        )

        return device, device_str


class LoggingManager:
    """Manages all logging functionality including CSV and TensorBoard"""

    @staticmethod
    def save_run_parameters(run_name: str, args_dict: Dict[str, Any], logs_dir: str) -> str:
        """Save run parameters to logs directory"""
        run_logs_dir = os.path.join(logs_dir, run_name)
        os.makedirs(run_logs_dir, exist_ok=True)

        params_file = os.path.join(run_logs_dir, f"{run_name}_parameters.json")
        with open(params_file, "w") as f:
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
        with open(training_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "return",
                    "steps",
                    "total_steps",
                    "avg_100",
                    "paths_per_sec",
                    "steps_per_sec",
                    "time_elapsed",
                ]
            )

        # Initialize evaluation CSV with headers
        with open(evaluation_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "training_episode",
                    "eval_run",
                    "option_price",
                    "price_std",
                    "confidence_95",
                    "avg_total_exercised",
                    "avg_exercise_count",
                    "min_return",
                    "max_return",
                    "n_runs",
                ]
            )

        # Initialize raw evaluation episodes CSV with headers
        with open(raw_episodes_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "training_episode",
                    "eval_run",
                    "episode_idx",
                    "seed",
                    "return",
                    "steps",
                    "total_exercised",
                    "exercise_count",
                    "final_inventory",
                    "avg_spot_price",
                    "max_spot_price",
                    "min_spot_price",
                ]
            )

        return training_csv, evaluation_csv, raw_episodes_csv, evaluation_runs_dir

    @staticmethod
    def log_training_episode(
        csv_file: str,
        episode: int,
        episode_return: float,
        path_steps: int,
        total_steps: int,
        avg_100: float,
        paths_per_sec: float,
        steps_per_sec: float,
        time_elapsed: float,
    ) -> None:
        """Log training episode data to CSV"""
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    episode,
                    round(episode_return, 6),
                    path_steps,
                    total_steps,
                    round(avg_100, 6),
                    round(paths_per_sec, 3),
                    round(steps_per_sec, 1),
                    round(time_elapsed, 2),
                ]
            )

    @staticmethod
    def log_evaluation_run(
        csv_file: str, training_episode: int, eval_run: int, pricing_stats: Dict[str, Any]
    ) -> None:
        """Log evaluation run data to CSV"""
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    training_episode,
                    eval_run,
                    round(pricing_stats["option_price"], 6),
                    round(pricing_stats["price_std"], 6),
                    round(pricing_stats["confidence_95"], 6),
                    round(pricing_stats["avg_total_exercised"], 6),
                    round(pricing_stats["avg_exercise_count"], 3),
                    round(min(pricing_stats["all_returns"]), 6),
                    round(max(pricing_stats["all_returns"]), 6),
                    pricing_stats["n_runs"],
                ]
            )

    @staticmethod
    def log_raw_evaluation_episode(
        csv_file: str,
        training_episode: int,
        eval_run: int,
        episode_idx: int,
        seed: int,
        episode_return: float,
        steps: int,
        total_exercised: float,
        exercise_count: int,
        final_inventory: float,
        avg_spot_price: float,
        max_spot_price: float,
        min_spot_price: float,
    ) -> None:
        """Log individual raw evaluation episode data to CSV"""
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    training_episode,
                    eval_run,
                    episode_idx,
                    seed,
                    round(episode_return, 6),
                    steps,
                    round(total_exercised, 6),
                    exercise_count,
                    round(final_inventory, 6),
                    round(avg_spot_price, 6),
                    round(max_spot_price, 6),
                    round(min_spot_price, 6),
                ]
            )

    @staticmethod
    def log_detailed_step_data(
        evaluation_runs_dir: str, training_episode: int, all_episodes_data: List[List[Dict[str, Any]]]
    ) -> None:
        """Log detailed step-by-step data for all episodes in an evaluation run to one CSV file"""
        filename = f"eval_run_{training_episode}.csv"
        filepath = os.path.join(evaluation_runs_dir, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode_idx",
                    "step",
                    "spot",
                    "q_remain",
                    "q_exerc",
                    "time_left",
                    "action",
                    "q_actual",
                    "reward",
                ]
            )

            for episode_idx, step_data in enumerate(all_episodes_data):
                for step_info in step_data:
                    writer.writerow(
                        [
                            episode_idx,
                            step_info["step"],
                            round(step_info["spot_price"], 4),
                            round(step_info["q_remaining"], 4),
                            round(step_info["q_exercised"], 4),
                            round(step_info["time_left"], 4),
                            round(step_info["action"], 6),
                            round(step_info["q_actual"], 4),
                            round(step_info["reward"], 6),
                        ]
                    )

    @staticmethod
    def save_lsm_benchmark_results(run_name: str, lsm_results: Dict[str, Any], logs_dir: str) -> str:
        """Save LSM benchmark results to logs directory"""
        run_logs_dir = os.path.join(logs_dir, run_name)
        os.makedirs(run_logs_dir, exist_ok=True)

        lsm_file = os.path.join(run_logs_dir, f"{run_name}_lsm_benchmark.json")
        with open(lsm_file, "w") as f:
            # Remove numpy arrays for JSON serialization
            serializable_results = {
                "lsm_option_price": float(lsm_results["lsm_option_price"]),
                "lsm_std_error": float(lsm_results["lsm_std_error"]),
                "lsm_confidence_95": float(lsm_results["lsm_confidence_95"]),
                "lsm_pricing_time": float(lsm_results["lsm_pricing_time"]),
                "lsm_paths_used": int(lsm_results["lsm_paths_used"]),
            }
            json.dump(serializable_results, f, indent=2)

        print(f"LSM benchmark results saved to: {lsm_file}")
        return lsm_file


def evaluate_swing_option(
    agent: Agent,
    eval_env: SwingOptionEnv,
    writer: SummaryWriter,
    path: int,
    evaluations_dir: str,
    csv_writer: Optional[AsyncCSVWriter] = None,
) -> List[float]:
    """
    Evaluate swing option price using the evaluation environment dataset
    Now includes detailed CSV logging for each path

    Args:
        agent: Trained D4PG agent
        eval_env: Swing option environment for evaluation (contains dataset)
        writer: TensorBoard writer for logging
        path: Current training path/episode number
        evaluations_dir: Directory to save evaluation CSV files
        csv_writer: AsyncCSVWriter instance for non-blocking CSV writes

    Returns:
        List[float]: List of all episode returns
    """
    discounted_returns = []
    exercise_stats = []

    # Get number of paths from eval_env dataset
    n_paths = eval_env.S.shape[0]
    
    # Define state column names based on SwingOptionEnv._get_observation()
            #     spot_price - self.contract.strike,  # Payoff
            # self.q_exercised / self.contract.Q_max,  # Normalized cumulative exercise
            # q_remaining / self.contract.Q_max,  # Normalized remaining capacity
            # time_to_maturity / self.contract.maturity,  # Normalized time to maturity
            # normalized_time,  # Progress through contract
            # spot_price, # Spot Price
            # X_t,  # Mean-reverting component
            # Y_t,  # Jump component  
            # # self.recent_volatility,  # Recent realized volatility
            # days_since_exercise / self.contract.n_rights  # Normalized refraction time
    state_columns = [
        'Payoff',      # spot_price - strike
        'q_exercised_norm',     # q_exercised / Q_max  
        'q_remaining_norm',     # q_remaining / Q_max
        'time_to_maturity_norm', # time_to_maturity / maturity
        'normalized_time',      # current_step / n_rights
        'spot',                 # S_t
        'X_t',                  # Mean-reverting component
        'Y_t',                  # Jump component
        # 'recent_volatility',    # Recent realized volatility
        'days_since_exercise_norm'  # days_since_exercise / n_rights
    ]
    
    # CSV headers
    csv_headers = ['path', 'time_step'] + state_columns + ['q_t', 'reward']
    
    # Prepare CSV file path
    csv_filename = f"rl_episode_{path}.csv"
    csv_filepath = os.path.join(evaluations_dir, csv_filename)

    # Collect all path data for batch writing
    all_path_data = []

    for i in range(n_paths):
        state, _ = eval_env.reset()
        
        disc_return = 0.0
        total_exercised = 0.0
        exercise_count = 0
        step = 0
        
        # Store path data for CSV
        path_data = []

        while True:
            # Get action from agent
            action = agent.act(np.expand_dims(state, axis=0), add_noise=False)
            action_v = np.clip(action, 0.0, 1.0)  # Ensure valid action range
            
            # Step environment
            next_state, reward, terminated, truncated, info = eval_env.step(action_v)
            
            # Store step data for CSV (with optimizations)
            # Round values to reduce file size while preserving precision
            step_row = [
                i,  # path_n (int)
                step,  # step (int)
                round(state[0], 6),  # spot_price_norm
                round(state[1], 6),  # q_exercised_norm  
                round(state[2], 6),  # q_remaining_norm
                round(state[3], 6),  # time_to_maturity_norm
                round(state[4], 6),  # normalized_time
                round(state[5], 6),  # X_t
                round(state[6], 6),  # Y_t
                round(state[7], 6),  # recent_volatility
                round(state[8], 6),  # days_since_exercise_norm
                round(info.get("q_actual", 0), 6),  # action (extract scalar from array and round)
                round(reward, 6)   # reward
            ]
            path_data.append(step_row)
            
            # Update tracking variables
            disc_return += reward  # Reward already includes discounting
            if info.get("q_actual", 0) > 1e-6:
                exercise_count += 1
                total_exercised += info["q_actual"]
            step += 1
            
            state = next_state

            if terminated or truncated:
                break
        
        # Add this path's data to the collection
        all_path_data.extend(path_data)

        discounted_returns.append(disc_return)
        exercise_stats.append(
            {"total_exercised": total_exercised, "exercise_count": exercise_count, "steps": step}
        )

    # Write all data to CSV asynchronously (non-blocking)
    if csv_writer and all_path_data:
        csv_writer.write_csv(csv_filepath, all_path_data, csv_headers)
    elif all_path_data:
        # Fallback: write synchronously if no async writer provided
        try:
            with open(csv_filepath, 'w', newline='') as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow(csv_headers)
                writer_csv.writerows(all_path_data)
        except Exception as e:
            print(f"Warning: Failed to write CSV file {csv_filepath}: {e}")

    # reset the eval_env._episode_counter to -1 so future resets work correctly
    eval_env._episode_counter = -1

    # Calculate statistics
    option_price = np.mean(discounted_returns)
    price_std = np.std(discounted_returns)
    avg_exercised = np.mean([s["total_exercised"] for s in exercise_stats])
    avg_exercises = np.mean([s["exercise_count"] for s in exercise_stats])
    confidence_95 = 1.96 * price_std / np.sqrt(n_paths)

    if path is not None:
        writer.add_scalar("Swing_Option_Price", option_price, path)
        writer.add_scalar("Price_Std", price_std, path)
        writer.add_scalar("Avg_Total_Exercised", avg_exercised, path)
        writer.add_scalar("Avg_Exercise_Count", avg_exercises, path)

        # Print detailed evaluation results
        print(f"\n{'=' * 50}")
        print(f"EVALUATION RESULTS (Episode {path})")
        print(f"{'=' * 50}")
        print(f"Option Price: {option_price:.3f} ± {confidence_95:.3f}")
        print(f"Price Std Dev: {price_std:.3f}")
        print(f"Avg Total Exercised: {avg_exercised:.3f}")
        print(f"Avg Exercise Count: {avg_exercises:.1f}")
        print(f"Evaluation Runs: {n_paths}")
        print(f"Min Return: {min(discounted_returns):.3f}")
        print(f"Max Return: {max(discounted_returns):.3f}")
        if csv_writer:
            print(f"CSV saved: {csv_filename} (async)")
        else:
            print(f"CSV saved: {csv_filename}")
        print(f"{'=' * 50}")

    return discounted_returns


def generate_datasets(
    stochastic_process_params: Dict, n_paths: int, n_paths_eval: int, seed: int
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Generate training and evaluation datasets using antithetic variance reduction and QMC.
    """

    print(f"🎲 Generating datasets: {n_paths} training + {n_paths_eval} eval paths...")
    pre_generation_start = time.time()

    train_t, train_S, train_X, train_Y = simulate_hhk_spot(
        **stochastic_process_params, n_paths=n_paths, seed=seed
    )
    # print(f'>>>> shape of train_S: {train_S.shape}')
    # # Plot 200 sample paths and the mean
    # plt.figure(figsize=(12, 6))
    # n_plot = min(200, train_S.shape[0])
    # for i in range(n_plot):
    #     plt.plot(train_t, train_S[i], color='lightgray', linewidth=0.5, alpha=0.7)
    # plt.plot(train_t, np.mean(train_S, axis=0), label='Mean Training Spot Price', color='blue', linewidth=2)
    # plt.title(
    #     f"Training Spot Price Simulation\n"
    #     f"S0={stochastic_process_params['S0']}, alpha={stochastic_process_params['alpha']}, "
    #     f"sigma={stochastic_process_params['sigma']}, beta={stochastic_process_params['beta']}, "
    #     f"lam={stochastic_process_params['lam']}, mu_J={stochastic_process_params['mu_J']}"
    # )
    # plt.xlabel('Time')
    # plt.ylabel('Spot Price')
    # plt.legend()
    # plt.grid()
    # plt.show()

    eval_t, eval_S, eval_X, eval_Y = simulate_hhk_spot(
        **stochastic_process_params, n_paths=n_paths_eval, seed=seed + 1
    )

    train_ds = train_t, train_S, train_X, train_Y
    eval_ds = eval_t, eval_S, eval_X, eval_Y
    pre_generation_time = time.time() - pre_generation_start

    # Calculate total storage for both datasets
    total_storage_mb = (sum(arr.nbytes for arr in train_ds) + sum(arr.nbytes for arr in eval_ds)) / 1024**2
    print(f"✅ Datasets generated in {pre_generation_time:.2f}s ({total_storage_mb:.1f} MB)")

    return train_ds, eval_ds


class TrainingManager:
    """Manages the main training loop and performance monitoring"""

    @staticmethod
    def timer(start: float, end: float) -> None:
        """Helper to print training time"""
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    @staticmethod
    def should_evaluate(
        current_path: int, n_paths: int, eval_every: int, initial_eval_done: bool
    ) -> Tuple[bool, bool]:
        """Determine if evaluation should be performed at current path"""
        should_evaluate = False
        new_initial_eval_done = initial_eval_done

        if eval_every == 0:
            # Error case: eval_every cannot be 0
            raise ValueError(
                "eval_every cannot be 0. Use -1 for end-only evaluation or positive value for periodic evaluation."
            )
        elif eval_every == -1:
            # Only evaluate at the end (after the last episode has completed)
            should_evaluate = current_path == n_paths
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
    def calculate_performance_metrics(
        episode_times: deque, episode_steps: deque, start_time: float, path_steps: int
    ) -> Tuple[float, float]:
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
            if hasattr(agent, "monitor_performance"):
                perf_info = agent.monitor_performance()
                if perf_info:
                    print(f"\n📊 Performance Check (Episode {current_path}):")
                    print(f"  Steps/sec: {steps_per_second:.1f}")
                    print(f"  Memory: {perf_info['memory_mb']:.1f} MB")
                    print(
                        f"  Buffer: {perf_info['buffer_size']:,} ({perf_info['buffer_fill_ratio'] * 100:.1f}%)"
                    )
                    if "buffer_memory_mb" in perf_info and perf_info["buffer_memory_mb"] > 0:
                        print(f"  Buffer memory: {perf_info['buffer_memory_mb']:.1f} MB")

                    # Force cleanup if performance is poor
                    if steps_per_second < 100 and perf_info["memory_mb"] > 6000:
                        print("⚠️ Low performance detected, forcing memory cleanup...")
                        agent.cleanup_memory(force=True)

def run_training(
    agent: Agent,
    train_env: SwingOptionEnv,
    eval_env: SwingOptionEnv,
    tensorboard_writer: SummaryWriter,
    args: argparse.Namespace,
    action_low: float,
    action_high: float,
    evaluations_dir: str,
    csv_writer: Optional[AsyncCSVWriter] = None,
) -> None:
    """
    Main training function for Deep Q-Learning for Swing Option Pricing

    Args:
        agent: D4PG agent
        train_env: Training environment
        eval_env: Evaluation environment
        tensorboard_writer: TensorBoard writer
        args: Parsed command line arguments
        action_low: Minimum action value
        action_high: Maximum action value
    """
    scores = []
    scores_window = deque(maxlen=100)
    total_steps = 0

    print(f"\n{'=' * 60}")
    print("STARTING RL AGENT TRAINING")
    print(f"{'=' * 60}")
    print(f"Training Episodes: {args.n_paths}")
    print(f"Evaluation Every: {args.eval_every if args.eval_every > 0 else 'End Only'}")
    print(f"Evaluation Paths: {args.n_paths_eval}")
    print(f"{'=' * 60}")

    # Performance monitoring variables
    start_time = time.time()
    episode_times = deque(maxlen=50)
    episode_steps = deque(maxlen=50)

    # eval_every should not be 0, either -1 or >0
    if args.eval_every == 0:
        raise ValueError("eval_every cannot be 0. Use -1 for end-only evaluation or positive value for periodic evaluation.")

    # Initial evaluation of DRL agent without any training - if eval_every > 0
    if not args.eval_every == -1:
        evaluate_swing_option(
            agent=agent,
            eval_env=eval_env,
            writer=tensorboard_writer,
            path=0,
            evaluations_dir=evaluations_dir,
            csv_writer=csv_writer,
        )
    for current_path in range(1, args.n_paths + 1):
        # Fix 3: Update episode count in PER for proper beta annealing
        agent.update_episode_count(current_path)
        
        # Use pre-generated training path for this episode (1:1 mapping, no cycling)
        path_idx = current_path - 1  # Direct mapping: episode i uses training path i

        # Validate we have enough training paths (should always be true with updated dataset generation)
        if path_idx >= train_env.S.shape[0]:
            raise ValueError(
                f"Episode {current_path} requires training path {path_idx}, but only {train_env.S.shape[0]} paths generated. "
                f"Ensure n_paths ({args.n_paths}) matches number of training paths generated."
            )

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
            agent.step(state, action_v[0], reward, next_state, done, total_steps, tensorboard_writer)

            state = next_state
            score += float(reward)

            if done:
                break

        # Step learning rate schedulers at the end of each episode
        agent.step_lr_schedulers(current_path)

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
        avg_100 = float(np.mean(scores_window))

        # TensorBoard logging
        tensorboard_writer.add_scalar("Average100", avg_100, current_path)
        tensorboard_writer.add_scalar("Episode_Return", episode_return, current_path)
        tensorboard_writer.add_scalar("Paths_Per_Second", paths_per_sec, current_path)
        tensorboard_writer.add_scalar("Steps_Per_Second", steps_per_sec, current_path)
        tensorboard_writer.add_scalar("Total_Steps", total_steps, current_path)
        tensorboard_writer.add_scalar("Path_Length", path_steps, current_path)
        
        # Log learning rates for monitoring decay
        if agent.actor_scheduler is not None:
            tensorboard_writer.add_scalar("Learning_Rate/Actor", agent.actor_optimizer.param_groups[0]['lr'], current_path)
            tensorboard_writer.add_scalar("Learning_Rate/Critic", agent.critic_optimizer.param_groups[0]['lr'], current_path)

        print(
            f"Path {current_path}/{args.n_paths} | Return = {episode_return:.3f} | "
            f"Steps = {path_steps} | Paths/sec = {paths_per_sec:.1f} | "
            f"Steps/sec = {steps_per_sec:.0f}"
        )

        # Post-training evaluation logic
        should_eval_after = False

        if args.eval_every == -1:
            # Only evaluate after the last episode has completed
            should_eval_after = current_path == args.n_paths
        elif args.eval_every > 0:
            # Evaluate after every eval_every episodes have completed
            if current_path % args.eval_every == 0:
                should_eval_after = True
            # Also evaluate at the end if not aligned with eval_every schedule
            elif current_path == args.n_paths and current_path % args.eval_every != 0:
                should_eval_after = True

        if should_eval_after:
            print(f"\n🔍 Starting evaluation after {current_path} episodes completed...")
            print("   Using VALIDATION dataset (same paths used for all evaluations)")
            evaluate_swing_option(
                agent=agent,
                eval_env=eval_env,
                writer=tensorboard_writer,
                path=current_path,
                evaluations_dir=evaluations_dir,
                csv_writer=csv_writer,
            )


def main():
    """Main execution function"""
    # Parse arguments and setup
    parser = ConfigManager.create_parser()
    args = parser.parse_args()

    # Setup PyTorch optimizations
    ConfigManager.setup_pytorch_optimizations(args)

    # Seed setup
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environments and contracts
    swing_contract = SwingContract(
        q_min=args.q_min,
        q_max=args.q_max,
        Q_min=args.Q_min,
        Q_max=args.Q_max,
        strike=args.strike,
        maturity=args.maturity,
        n_rights=args.n_rights,
        r=args.risk_free_rate,
        min_refraction_days=args.min_refraction_days,
    )

    stochastic_process_params = {
        "S0": args.S0,
        "T": swing_contract.maturity,
        "n_steps": swing_contract.n_rights - 1,  # Generate n_rights time points (0 to T)
        "alpha": args.alpha,
        "sigma": args.sigma,
        "beta": args.beta,
        "lam": args.lam,
        "mu_J": args.mu_J,
        "f": no_seasonal_function,
    }

    # Generate training and evaluation datasets:
    train_ds, eval_ds = generate_datasets(
        stochastic_process_params=stochastic_process_params,
        n_paths=args.n_paths,
        n_paths_eval=args.n_paths_eval,
        seed=seed,
    )

    # Create environments - Stores the pre-generated paths in the environment
    train_env = SwingOptionEnv(
        contract=swing_contract, hhk_params=stochastic_process_params, dataset=train_ds)
    eval_env = SwingOptionEnv(
        contract=swing_contract, hhk_params=stochastic_process_params, dataset=eval_ds)

    ## Create Experiment Directory
    exp_dir = 'logs/' + args.name # experiment dir
    shutil.rmtree(exp_dir, ignore_errors=True) # Remove old experiment directory if it exists
    os.makedirs(exp_dir) # create new experiment directory
    # create "evaluations" subdirectory
    os.makedirs(exp_dir + '/evaluations', exist_ok=True) # create new /evaluations subdirectory

    # Price with Monte Carlo - LSM Benchmark
    evaluations_dir = exp_dir + '/evaluations'
    mean_lsm_price, (th5q_price,th95q_price) = price_swing_option_lsm(
        contract=swing_contract,
        dataset=eval_ds,
        poly_degree=3, seed=seed+1,
        csv_path=evaluations_dir + '/lsm.csv'
    )
    print(f"LSM Benchmark Price: {mean_lsm_price:.4f} (95% CI: [{th5q_price:.4f}, {th95q_price:.4f}])")
    # Price with Quantlib - Finite Differences Method
    fdm_price = price_swing_option_fdm(
        contract=swing_contract,
        stochastic_process_params=stochastic_process_params,
        tGrid=25, xGrid=25, yGrid=50
    )
    print(f"Quantlib's Finite Differences Method Price: {fdm_price:.4f}")
    print('\n\n\n\n' + '=' * 60)

    ############################################################################################
    ############################################################################################
    ############################################################################################
    # Run training

    # Setup device and cores
    device, device_str = EnvironmentManager.setup_device_and_cores(args)

    # Initialize TensorBoard writer
    tensorboard_writer = SummaryWriter("runs/" + args.name)
    
    # Initialize AsyncCSVWriter for non-blocking CSV logging
    csv_writer = AsyncCSVWriter()
    csv_writer.start()
    print("✅ AsyncCSVWriter initialized for evaluation logging")

    # Get environment specs
    action_high = train_env.action_space.high[0]        # pyright: ignore[reportAttributeAccessIssue]
    action_low = train_env.action_space.low[0]          # pyright: ignore[reportAttributeAccessIssue]
    state_size = train_env.observation_space.shape[0]   # pyright: ignore[reportOptionalSubscript]
    action_size = train_env.action_space.shape[0]       # pyright: ignore[reportOptionalSubscript]

    agent = Agent(
        state_size=train_env.observation_space.shape[0],
        action_size=train_env.action_space.shape[0],
        n_step=args.nstep,
        per=args.per,
        munchausen=args.munchausen,
        distributional=args.iqn,
        noise_type=args.noise,
        random_seed=seed,
        hidden_size=args.layer_size,
        BUFFER_SIZE=args.max_replay_size,
        BATCH_SIZE=args.batch_size,
        GAMMA=args.gamma,
        t=args.t,
        LR_ACTOR=args.lr_a,
        LR_CRITIC=args.lr_c,
        WEIGHT_DECAY=0,
        LEARN_EVERY=args.learn_every,
        LEARN_NUMBER=args.learn_number,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        device=device,
        paths=args.n_paths,
        min_replay_size=args.min_replay_size,
        speed_mode=True,
        use_compile=args.compile,
        use_amp=False,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_frames=args.per_beta_frames,
        final_lr_fraction=args.final_lr_fraction,
        total_episodes=args.n_paths,
        warmup_frac=args.warmup_frac,
        min_lr=args.min_lr
    )
    t0 = time.time()

    if args.saved_model is not None:
        agent.actor_local.load_state_dict(torch.load(args.saved_model)) # type: ignore
        print("WARNING: Pre-generated paths not available for saved model evaluation.")
    else:
        # Run training
        run_training(
            agent=agent,
            train_env=train_env,
            eval_env=eval_env,
            tensorboard_writer=tensorboard_writer,
            args=args,
            action_low=action_low,
            action_high=action_high,
            evaluations_dir=evaluations_dir,
            csv_writer=csv_writer,
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    t1 = time.time()
    eval_env.close()
    
    # Stop the CSV writer and wait for any pending writes
    csv_writer.stop()
    print("✅ AsyncCSVWriter stopped - all CSV files written")
    
    TrainingManager.timer(t0, t1)

    # Save trained model (handle compiled models)
    try:
        if hasattr(agent, '_actor_local_orig'):
            # Use original uncompiled model if torch.compile was used
            torch.save(agent._actor_local_orig.state_dict(), "runs/" + args.name + ".pth")
            print(f"✅ Model saved to: runs/{args.name}.pth (original uncompiled version)")
        elif hasattr(agent.actor_local, 'state_dict'):
            # No compilation was used, save the regular actor_local
            torch.save(agent.actor_local.state_dict(), "runs/" + args.name + ".pth")
            print(f"✅ Model saved to: runs/{args.name}.pth")
        else:
            print("⚠️ Could not save model: actor_local appears to be compiled and no original model found")
            print("Consider disabling torch.compile (--compile 0) for model saving")
    except Exception as e:
        print(f"⚠️ Could not save model: {e}")

    # Save parameters
    with open("runs/" + args.name + ".json", "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
