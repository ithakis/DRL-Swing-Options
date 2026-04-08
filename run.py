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
import math
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
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.tensorboard import SummaryWriter

# Optional SciPy for sigma root finding; fall back to pure Python if unavailable
try:
    from scipy.optimize import brentq  # type: ignore
    from scipy.stats import norm  # type: ignore
except Exception:  # pragma: no cover - SciPy is optional
    brentq = None
    norm = None

from src.agent import Agent
from src.agent_evaluation import benchmark_evaluation, evaluate_agent
from src.lsm_swing_pricer import fit_lsm_estimators, price_swing_option_lsm_oos

# Import LSM pricer for benchmarking
from src.simulate_hhk_spot import no_seasonal_function, simulate_hhk_spot
from src.swing_contract import SwingContract
from src.swing_env import SwingOptionEnv, approximate_Q_T

# Suppress the macOS PyTorch profiling warning
warnings.filterwarnings("ignore", message=".*record_context_cpp.*")



class AsyncParquetWriter:
    """Asynchronous Parquet writer to avoid blocking main execution"""

    def __init__(self):
        self.write_queue = Queue()
        self.writer_thread = None
        self.is_running = False
        self._writers = {}

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
                # Wait indefinitely for the writer thread to finish pending writes
                # This prevents closing file handles while the thread is still using them
                self.writer_thread.join()
            for writer in self._writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            self._writers.clear()
            self.is_running = False

    def _write_worker(self):
        """Background worker that processes the write queue"""
        while True:
            item = self.write_queue.get()
            if item is None:  # Sentinel to stop
                break

            try:
                filepath, data, columns = item
                if not data:
                    continue
                arr = np.asarray(data)
                table = pa.Table.from_arrays(
                    [pa.array(arr[:, idx]) for idx in range(arr.shape[1])],
                    names=columns,
                )
                writer = self._writers.get(filepath)
                if writer is None:
                    writer = pq.ParquetWriter(
                        filepath,
                        table.schema,
                        compression="zstd",
                        compression_level=22,
                    )
                    self._writers[filepath] = writer
                writer.write_table(table)
            except Exception as e:
                print(f"Error writing to Parquet: {e}")
            finally:
                self.write_queue.task_done()

    def write_parquet(self, filepath: str, data, columns=None):
        """Queue data to be written to Parquet file"""
        if self.is_running:
            self.write_queue.put((filepath, data, columns))


class ConfigManager:
    """Handles configuration and argument parsing"""

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create and configure argument parser"""
        parser = argparse.ArgumentParser(description="Swing Option Pricing with D4PG")

        parser.add_argument(
            "--actor_type",
            type=str,
            default="standard",
            choices=["standard", "finance_informed"],
            help="Type of actor architecture. 'finance_informed' uses a greedy baseline.",
        )
        # Replay Buffer Parameters
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
            "--eval_batch_size",
            type=int,
            default=512,
            help="Batch size for batched evaluation inference (default: 512, auto-clamped to n_paths_eval).",
        )
        parser.add_argument(
            "--profile_eval",
            action="store_true",
            help="Profile the evaluation path with cProfile and emit results to profilelogs/.",
        )
        parser.add_argument(
            "--eval_benchmark",
            action="store_true",
            help="Run a standalone evaluation benchmark (no training) using current args.",
        )
        parser.add_argument(
            "-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0"
        )
        parser.add_argument("-name", type=str, help="Name of the run (default: SwingOption_{timestamp})")
        parser.add_argument(
            "--disable_csv_logging",
            type=int,
            choices=[0, 1],
            default=0,
            help="Disable evaluation Parquet logging outputs (evaluation/LSM). Set 1 to disable, 0 to allow.",
        )
        parser.add_argument(
            "--limit_logging_frequency",
            type=int,
            choices=[0, 1],
            default=0,
            help="Reduce per-step TensorBoard logging frequency by 10x to shrink event files. Set 1 to enable, 0 to disable.",
        )

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
            "--min_refraction_periods", type=int, default=0, help="Minimum periods between exercises, default = 0"
        )
        parser.add_argument(
            "--c_cost",
            type=float,
            default=0.0,
            help="Convex cost coefficient c for per-unit exercise cost (0 disables the cost).",
        )
        parser.add_argument(
            "--gamma_cost",
            type=float,
            default=1.0,
            help="Convex cost exponent gamma for per-unit exercise cost.",
        )

        # LSM Benchmark Configuration
        parser.add_argument(
            "--lsm_basis",
            type=str,
            default="power",
            choices=["power", "laguerre", "hermite", "chebyshev"],
            help="Basis family for the LSM regression (default: power).",
        )
        parser.add_argument(
            "--lsm_degree",
            type=int,
            default=3,
            help="Polynomial degree for LSM basis functions (default: 3).",
        )
        parser.add_argument(
            "--lsm_reg",
            type=str,
            default="none",
            choices=["none", "ridge", "lasso"],
            help="Regularization to use for LSM regression (default: none).",
        )
        parser.add_argument(
            "--lsm_reg_alpha",
            type=float,
            default=1e-6,
            help="Regularization strength alpha for ridge/lasso LSM (default: 1e-6).",
        )
        parser.add_argument(
            "--lsm_n_actions",
            type=int,
            default=5,
            help="Number of discretized exercise levels for LSM baseline (default: 5).",
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
        # CPU-only project: no device selector (CUDA/MPS intentionally unsupported).
        parser.add_argument("-nstep", type=int, default=1, help="Nstep bootstrapping, default 1")
        # PER hyperparameters
        parser.add_argument("--per_alpha", type=float, default=0.6, help="PER: priority exponent alpha (default: 0.6)")
        parser.add_argument("--per_beta_start", type=float, default=0.4, help="PER: initial importance sampling weight beta_start (default: 0.4)")
        parser.add_argument("--per_beta_frames", type=int, default=100000, help="PER: frames to anneal beta to 1.0 (default: 100000)")
        parser.add_argument("--per_priority_floor", type=float, default=1e-6, help="Minimum PER priority to avoid zeros (default: 1e-6)")
        parser.add_argument("--per_priority_clip_pct", type=float, default=0.0, help="Clip PER priorities to this percentile (0 disables, default: disabled)")
        parser.add_argument(
            "--per_priority_scheme",
            type=str,
            default="standard",
            choices=["standard", "lap"],
            help="PER: base priority scheme {standard=|TD|, lap=Huber loss} (default: standard)",
        )
        parser.add_argument(
            "--per_huber_kappa",
            type=float,
            default=1.0,
            help="PER: Huber threshold kappa for lap scheme (default: 1.0)",
        )
        parser.add_argument("--per_alpha_final", type=float, default=None, help="Optional PER alpha target for scheduling (None disables)")
        parser.add_argument("--per_alpha_ramp_start", type=int, default=0, help="Episode to start PER alpha ramp (0 disables if end<=start)")
        parser.add_argument("--per_alpha_ramp_end", type=int, default=0, help="Episode to end PER alpha ramp (0 disables if end<=start)")
        parser.add_argument("--per_beta_final", type=float, default=None, help="Optional PER beta target for scheduling (None keeps default beta behavior)")
        parser.add_argument("--per_alpha_sigmoid", type=int, default=0, choices=[0, 1], help="Use sigmoid ramp for PER alpha/beta (default: 0 / linear)")
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
            default=0,
            choices=[0, 1],
            help="Adding Munchausen RL to the agent if set to 1 (default: 0 / off)",
        )
        parser.add_argument(
            "-iqn",
            type=int,
            choices=[0, 1],
            default=0,
            help="Use distributional IQN Critic if set to 1, default = 0 (no IQN)",
        )
        parser.add_argument(
            "-noise_sigma0",
            type=float,
            default=1.0,
            help="Initial pre-squash noise std (applied before action squashing)",
        )
        parser.add_argument(
            "-noise_floor",
            type=float,
            default=0.05,
            help="Minimum pre-squash noise std after decay",
        )
        parser.add_argument(
            "-noise_plateau",
            type=int,
            default=0,
            help="Episodes to hold the initial pre-squash noise level before decay",
        )
        parser.add_argument(
            "--critic_warmup_episodes",
            type=int,
            default=0,
            help="Number of episodes to freeze actor updates (default: 0). Recommended: 1024 (matches min_replay_size approx)."
        )
        parser.add_argument(
            "--adaptive_noise_scale",
            type=float,
            default=0.0,
            help="Scale factor for pre-activation noise relative to |u|. Default: 0.0 (off). Recommended: 0.5."
        )
        # v60 parameters
        parser.add_argument(
            "--warmup_noise_fraction",
            type=float,
            default=1.0,
            help="v60: Fraction of normal noise to use during critic warmup (0.0-1.0). Default: 1.0 (no reduction). Recommended: 0.2"
        )
        parser.add_argument(
            "--target_noise_decay_start",
            type=int,
            default=0,
            help="v60: Episode to start decaying target policy noise (0 disables). Recommended: 15000"
        )
        parser.add_argument(
            "--target_noise_floor",
            type=float,
            default=0.02,
            help="v60: Minimum target policy noise after decay. Default: 0.02"
        )
        parser.add_argument(
            "--actor_output_activation",
            type=str,
            default="tanh01",
            help="v60: Actor output activation {tanh01, sigmoid, beta_sigmoid, beta_sigmoid_X}. Default: tanh01"
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
            default=64,
            help="Legacy hidden size flag applied when specific actor/critic sizes are not provided (default: 64 for 2×64 MLPs)",
        )
        parser.add_argument(
            "--actor_hidden_size",
            type=int,
            default=None,
            help="Hidden width for the actor network (overrides -layer_size for the policy)",
        )
        parser.add_argument(
            "--critic_hidden_size",
            type=int,
            default=None,
            help="Hidden width for the critic network (overrides -layer_size for the value function)",
        )
        parser.add_argument(
            "--actor_layers",
            type=int,
            default=2,
            help="Number of hidden layers in the actor network (default: 2)",
        )
        parser.add_argument(
            "--critic_layers",
            type=int,
            default=2,
            help="Number of hidden layers in the critic network (default: 2; must be >=2)",
        )
        parser.add_argument(
            "--activation",
            type=str,
            default="silu",
            choices=["relu", "leaky_relu", "silu"],
            help="Activation function for actor/critic MLPs (default: silu)",
        )
        parser.add_argument(
            "--norm",
            type=str,
            default="layernorm",
            choices=["layernorm", "rmsnorm", "none"],
            help="Normalization layer in actor/critic/IQN MLPs (default: layernorm)",
        )
        parser.add_argument(
            "--init_method",
            type=str,
            default="orthogonal",
            choices=["orthogonal", "he", "kaiming"],
            help="Weight initialization for actor/critic/IQN MLPs (default: orthogonal)",
        )
        parser.add_argument(
            "--final_lr_fraction",
            type=float,
            default=1.0,
            help="Final learning rate as fraction of initial LR (1.0=no decay, 0.1=decay to 10 percent)",
        )
        parser.add_argument(
            "--warmup_episodes",
            type=int,
            default=1024,
            help="Number of episodes for LR warm-up (ramps to full LR by this episode; default: 1024).",
        )
        parser.add_argument(
            "--lr_schedule_episodes",
            type=int,
            default=65000,
            help="Episode horizon for LR warmup+cosine schedule (can exceed training length; default: 65000).",
        )
        parser.add_argument(
            "--min_lr",
            type=float,
            default=1e-7,
            help="Minimum learning rate floor (default: 1e-7)",
        )
        parser.add_argument(
            "--actor_grad_clip",
            type=float,
            default=0.0,
            help="Clip threshold for actor gradients (<=0 disables clipping, default: 0.0 / disabled)",
        )
        parser.add_argument(
            "--critic_grad_clip",
            type=float,
            default=0.0,
            help="Clip threshold for critic gradients (<=0 disables clipping, default: 0.0 / disabled)",
        )
        parser.add_argument(
            "--actor_grad_clip_type",
            type=str,
            choices=["norm", "value", "none"],
            default="none",
            help="Clipping strategy for actor gradients (default: none/disabled)",
        )
        parser.add_argument(
            "--critic_grad_clip_type",
            type=str,
            choices=["norm", "value", "none"],
            default="none",
            help="Clipping strategy for critic gradients (default: none/disabled)",
        )
        parser.add_argument(
            "--grad_clip_norm_type",
            type=float,
            default=2.0,
            help="Norm type used when clipping by norm (default: 2.0 for L2 norm)",
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
        parser.add_argument(
            "--replay_memmap",
            type=int,
            choices=[0, 1],
            default=0,
            help="Use disk-backed memmap arrays for replay buffer storage to reduce RAM usage (may slow training).",
        )
        parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size, default is 128")
        parser.add_argument(
            "-t", "--t", type=float, default=2e-3, help="Softupdate factor t (Polyak tau), default is 2e-3"
        )
        parser.add_argument("-g", "--gamma", type=float, default=1, help="discount factor gamma, default is 1")
        parser.add_argument(
            "--optimizer",
            type=str,
            choices=["adam", "adamw"],
            default="adamw",
            help="Optimizer for actor/critic (default: adamw)",
        )
        parser.add_argument(
            "--weight_decay_actor",
            type=float,
            default=5e-5,
            help="Decoupled weight decay applied to the actor optimizer (default: 5e-5)",
        )
        parser.add_argument(
            "--weight_decay_critic",
            type=float,
            default=1e-4,
            help="Decoupled weight decay applied to the critic optimizer (default: 1e-4)",
        )
        parser.add_argument(
            "--critic_ema_decay",
            type=float,
            default=0.0,
            help="EMA decay for critic eval smoothing (0 disables, default: 0.0)",
        )
        parser.add_argument(
            "--target_policy_noise",
            type=float,
            default=0.1,
            help="Std of noise added to target actions for smoothing (default: 0.1; 0 disables)",
        )
        parser.add_argument(
            "--target_policy_clip",
            type=float,
            default=0.25,
            help="Clipping range for target policy smoothing noise (default: 0.25)",
        )

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
            help="Use float32 precision for simulation, tensors, and buffers (default: 1). Set 0 for float64.",
        )
        parser.add_argument(
            "--use_robust_normalization",
            type=int,
            choices=[0, 1],
            default=0,
            help="Enable Robust HHK Normalization (Log-moneyness + Median/IQR scaling). Default: 0 (disabled).",
        )

        return parser

    @staticmethod
    def setup_pytorch_optimizations(args: argparse.Namespace) -> None:
        """Setup PyTorch optimizations based on arguments"""
        dtype_choice = "float32" if args.fp32 else "float64"
        torch.set_default_dtype(torch.float32 if dtype_choice == "float32" else torch.float64)
        # Apply float32 optimization if enabled
        if dtype_choice == "float32":
            torch.set_default_dtype(torch.float32)
            print("Using float32 precision for improved performance")
        else:
            print("Using default precision (float64)")


class EnvironmentManager:
    """Manages environment setup and configuration"""

    @staticmethod
    def setup_device_and_cores(args: argparse.Namespace) -> Tuple[torch.device, str]:
        """Setup device and CPU cores configuration"""
        device = torch.device("cpu")
        device_str = "cpu"
        print("Selected device: cpu")

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



def generate_datasets(
    stochastic_process_params: Dict, n_paths: int, n_paths_eval: int, seed: int, batch_size: int = 128
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Generate training and evaluation datasets using antithetic variance reduction and QMC.
    """

    print(f"🎲 Generating datasets: {n_paths} training + {n_paths_eval} eval paths...")
    pre_generation_start = time.time()

    train_t, train_S, train_X, train_Y = simulate_hhk_spot(
        **stochastic_process_params, n_paths=n_paths, seed=seed, stratify=True, batch_size=batch_size
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
        **stochastic_process_params,
        n_paths=n_paths_eval,
        seed=seed + 1,
        stratify=True,
        batch_size=batch_size,
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
    lsm_price: float,
    parquet_writer: Optional[AsyncParquetWriter] = None,
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
    last_update_time = time.time()
    last_update_count = agent.updates_done
    last_eval_price: Optional[float] = None
    last_avg_exercised: Optional[float] = None

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
    episode_durations = deque(maxlen=100)
    # Legacy placeholders removed: no safe training or diagnostics logging

    # eval_every should not be 0, either -1 or >0
    if args.eval_every == 0:
        raise ValueError("eval_every cannot be 0. Use -1 for end-only evaluation or positive value for periodic evaluation.")

    # Initial evaluation of DRL agent without any training - if eval_every > 0
    if not args.eval_every == -1:
        eval_summary = evaluate_agent(
            agent=agent,
            eval_env=eval_env,
            writer=tensorboard_writer,
            path=0,
            evaluations_dir=evaluations_dir,
            lsm_price=lsm_price,
            parquet_writer=parquet_writer,
            eval_batch_size=args.eval_batch_size,
            profile=args.profile_eval,
        )
        last_eval_price = eval_summary.option_price
        last_avg_exercised = eval_summary.avg_total_exercised
    for current_path in range(1, args.n_paths + 1):
        # Fix 3: Update episode count in PER for proper beta annealing
        agent.update_episode_count(current_path)
        path_start_time = time.time()
        
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
        actions_episode = []

        # Complete one full path/episode
        while True:
            total_steps += 1
            path_steps += 1

            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)
            next_state, reward, terminated, truncated, _ = train_env.step(action_v[0])
            done = terminated or truncated
            agent.step(state, action_v[0], reward, next_state, done, total_steps, tensorboard_writer)
            actions_episode.append(action_v[0])

            state = next_state
            score += float(reward)

            if done:
                break

        # Step learning rate schedulers at the end of each episode
        agent.step_lr_schedulers(current_path)
        episode_durations.append(time.time() - path_start_time)

        # Calculate performance metrics
        paths_per_sec, steps_per_sec = TrainingManager.calculate_performance_metrics(
            episode_times, episode_steps, start_time, path_steps
        )

        # Performance monitoring
        TrainingManager.monitor_performance(agent, current_path, steps_per_sec)

        current_update_time = time.time()
        updates_completed = agent.updates_done
        update_interval = current_update_time - last_update_time
        if update_interval > 0:
            updates_per_second = (updates_completed - last_update_count) / update_interval
        else:
            updates_per_second = 0.0
        last_update_time = current_update_time
        last_update_count = updates_completed

        # Calculate episode return and update tracking
        episode_return = float(score)
        scores_window.append(episode_return)
        scores.append(episode_return)
        avg_100 = float(np.mean(scores_window))
        # No safe-training diagnostics; minimal logging only

        # TensorBoard logging
        tensorboard_writer.add_scalar("Average100", avg_100, current_path)
        tensorboard_writer.add_scalar("Episode_Return", episode_return, current_path)
        tensorboard_writer.add_scalar("Paths_Per_Second", paths_per_sec, current_path)
        tensorboard_writer.add_scalar("Steps_Per_Second", steps_per_sec, current_path)
        tensorboard_writer.add_scalar("Updates_Per_Second", updates_per_second, current_path)
        tensorboard_writer.add_scalar("Total_Steps", total_steps, current_path)
        tensorboard_writer.add_scalar("Path_Length", path_steps, current_path)
        try:
            pre_noise_sigma = agent._pre_noise_sigma()
            tensorboard_writer.add_scalar("Exploration/Pre_Action_Noise", pre_noise_sigma, current_path)
        except Exception:
            pass
        
        # Log learning rates for monitoring decay
        if agent.actor_scheduler is not None:
            tensorboard_writer.add_scalar("Learning_Rate/Actor", agent.actor_optimizer.param_groups[0]['lr'], current_path)
            tensorboard_writer.add_scalar("Learning_Rate/Critic", agent.critic_optimizer.param_groups[0]['lr'], current_path)

        last_price_display = f"{last_eval_price:.3f}" if last_eval_price is not None else "No Data"
        last_q_display = f"{last_avg_exercised:.3f}" if last_avg_exercised is not None else "No Data"

        eta_str = ""
        if current_path % 100 == 0 and len(episode_durations) > 0:
            avg_dur = sum(episode_durations) / len(episode_durations)
            remaining = max(0, args.n_paths - current_path)
            eta_seconds = remaining * avg_dur
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            eta_str = f" ({eta_hours:02d}:{eta_minutes:02d})"

        print(
            f"Path {current_path}/{args.n_paths}{eta_str} | Last Price = {last_price_display} | "
            f"E[Q_t] = {last_q_display} | Average100 = {avg_100:.3f} | Paths/sec = {paths_per_sec:.1f}"
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
            eval_summary = evaluate_agent(
                agent=agent,
                eval_env=eval_env,
                writer=tensorboard_writer,
                path=current_path,
                evaluations_dir=evaluations_dir,
                lsm_price=lsm_price,
                parquet_writer=parquet_writer,
                eval_batch_size=args.eval_batch_size,
                profile=args.profile_eval,
            )
            last_eval_price = eval_summary.option_price
            last_avg_exercised = eval_summary.avg_total_exercised


def main():
    """Main execution function"""
    # Parse arguments and setup
    parser = ConfigManager.create_parser()
    args = parser.parse_args()

    if args.eval_batch_size < 1:
        raise ValueError("eval_batch_size must be >= 1")

    disable_csv_logging = bool(args.disable_csv_logging)
    limit_logging_frequency = bool(args.limit_logging_frequency)
    csv_logging_enabled = not disable_csv_logging
    log_interval_scale = 10 if limit_logging_frequency else 1

    # Setup PyTorch optimizations
    ConfigManager.setup_pytorch_optimizations(args)
    dtype_choice = "float32" if args.fp32 else "float64"
    np_dtype = np.float32 if args.fp32 else np.float64

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
        min_refraction_periods=args.min_refraction_periods,
        c_cost=args.c_cost,
        gamma_cost=args.gamma_cost,
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
        "dtype": np_dtype,
    }

    # Generate training and evaluation datasets:
    train_ds, eval_ds = generate_datasets(
        stochastic_process_params=stochastic_process_params,
        n_paths=args.n_paths,
        n_paths_eval=args.n_paths_eval,
        seed=seed,
        batch_size=args.batch_size,
    )

    # Create environments - Stores the pre-generated paths in the environment
    train_env = SwingOptionEnv(
        contract=swing_contract, hhk_params=stochastic_process_params, dataset=train_ds, obs_dtype=np_dtype)
    eval_env = SwingOptionEnv(
        contract=swing_contract, hhk_params=stochastic_process_params, dataset=eval_ds, obs_dtype=np_dtype)

    ## Create Experiment Directory
    evaluations_dir: Optional[str] = None
    if csv_logging_enabled:
        exp_dir = os.path.join("logs", args.name) # experiment dir
        shutil.rmtree(exp_dir, ignore_errors=True) # Remove old experiment directory if it exists
        os.makedirs(exp_dir, exist_ok=True) # create new experiment directory
        # create "evaluations" subdirectory
        evaluations_dir = os.path.join(exp_dir, "evaluations")
        os.makedirs(evaluations_dir, exist_ok=True) # create new /evaluations subdirectory

    # Price with Monte Carlo - LSM Benchmark
    # LSM benchmark uses float64 for stability; fit on in-sample, evaluate out-of-sample
    lsm_train_ds = tuple(np.asarray(arr, dtype=np.float64) for arr in train_ds)
    lsm_eval_ds = tuple(np.asarray(arr, dtype=np.float64) for arr in eval_ds)
    lsm_parquet_path = os.path.join(evaluations_dir, "lsm.parquet") if evaluations_dir else None
    lsm_estimators = fit_lsm_estimators(
        contract=swing_contract,
        dataset=lsm_train_ds,
        poly_degree=args.lsm_degree,
        basis_type=args.lsm_basis,
        reg_type=args.lsm_reg,
        reg_alpha=args.lsm_reg_alpha,
        n_actions=args.lsm_n_actions,
    )
    mean_lsm_price, (th5q_price, th95q_price) = price_swing_option_lsm_oos(
        contract=swing_contract,
        dataset=lsm_eval_ds,
        estimators=lsm_estimators,
        seed=seed + 1,
        csv_path=lsm_parquet_path,
    )
    print(f"LSM Benchmark Price: {mean_lsm_price:.4f} (95% CI: [{th5q_price:.4f}, {th95q_price:.4f}])")
    print('\n\n\n\n' + '=' * 60)

    ############################################################################################
    ############################################################################################
    ############################################################################################
    # Run training

    # Setup device and cores
    device, device_str = EnvironmentManager.setup_device_and_cores(args)

    # Initialize TensorBoard writer
    tensorboard_writer = SummaryWriter("runs/" + args.name)
    
    # Initialize AsyncParquetWriter for non-blocking evaluation logging
    parquet_writer: Optional[AsyncParquetWriter] = None
    if csv_logging_enabled:
        parquet_writer = AsyncParquetWriter()
        parquet_writer.start()
        print("✅ AsyncParquetWriter initialized for evaluation logging")
    else:
        print("Parquet logging disabled; skipping writer initialization")

    try:
        # Get environment specs
        action_high = train_env.action_space.high[0]        # pyright: ignore[reportAttributeAccessIssue]
        action_low = train_env.action_space.low[0]          # pyright: ignore[reportAttributeAccessIssue]
        state_size = train_env.observation_space.shape[0]   # pyright: ignore[reportOptionalSubscript]
        action_size = train_env.action_space.shape[0]       # pyright: ignore[reportOptionalSubscript]

        actor_hidden_size = args.actor_hidden_size or args.layer_size
        critic_hidden_size = args.critic_hidden_size or args.layer_size

        agent = Agent(
            state_size=train_env.observation_space.shape[0],
            action_size=train_env.action_space.shape[0],
            n_step=args.nstep,
            per=args.per,
            munchausen=args.munchausen,
            distributional=args.iqn,
            random_seed=seed,
            hidden_size=args.layer_size,
            actor_hidden_size=actor_hidden_size,
            critic_hidden_size=critic_hidden_size,
            actor_layers=args.actor_layers,
            critic_layers=args.critic_layers,
            optimizer=args.optimizer,
            weight_decay_actor=args.weight_decay_actor,
            weight_decay_critic=args.weight_decay_critic,
            critic_ema_decay=args.critic_ema_decay,
            target_policy_noise=args.target_policy_noise,
            target_policy_clip=args.target_policy_clip,
            BUFFER_SIZE=args.max_replay_size,
            BATCH_SIZE=args.batch_size,
            GAMMA=args.gamma,
            t=args.t,
            LR_ACTOR=args.lr_a,
            LR_CRITIC=args.lr_c,
            LEARN_EVERY=args.learn_every,
            LEARN_NUMBER=args.learn_number,
            noise_sigma0=args.noise_sigma0,
            noise_floor=args.noise_floor,
            noise_plateau=args.noise_plateau,
            device=device,
            paths=args.n_paths,
            min_replay_size=args.min_replay_size,
            speed_mode=True,
            use_compile=args.compile,
            use_amp=False,
            per_alpha=args.per_alpha,
            per_beta_start=args.per_beta_start,
            per_beta_frames=args.per_beta_frames,
            per_priority_floor=args.per_priority_floor,
            per_priority_clip_pct=args.per_priority_clip_pct,
            per_priority_scheme=args.per_priority_scheme,
            per_huber_kappa=args.per_huber_kappa,
            per_alpha_final=args.per_alpha_final,
            per_alpha_ramp_start=args.per_alpha_ramp_start,
            per_alpha_ramp_end=args.per_alpha_ramp_end,
            per_beta_final=args.per_beta_final,
            per_alpha_sigmoid=bool(args.per_alpha_sigmoid),
            final_lr_fraction=args.final_lr_fraction,
            lr_schedule_episodes=args.lr_schedule_episodes,
            warmup_episodes=args.warmup_episodes,
            min_lr=args.min_lr,
            actor_grad_clip=args.actor_grad_clip,
            critic_grad_clip=args.critic_grad_clip,
            actor_grad_clip_type=args.actor_grad_clip_type,
            critic_grad_clip_type=args.critic_grad_clip_type,
            grad_clip_norm_type=args.grad_clip_norm_type,
            activation=args.activation,
            norm_type=args.norm,
            init_method=args.init_method,
            log_interval_scale=log_interval_scale,
            replay_memmap=bool(args.replay_memmap),
            actor_type=args.actor_type,
            critic_warmup_episodes=args.critic_warmup_episodes,
            adaptive_noise_scale=args.adaptive_noise_scale,
            # v60 parameters
            warmup_noise_fraction=args.warmup_noise_fraction,
            target_noise_decay_start=args.target_noise_decay_start,
            target_noise_floor=args.target_noise_floor,
            action_output=args.actor_output_activation,
            # v62 parameters
            use_robust_normalization=bool(args.use_robust_normalization),
            strike=args.strike,
        )
        try:
            param_device = next(agent.actor_local.parameters()).device
        except StopIteration:
            param_device = torch.device("cpu")
        print(f"Sanity: model param device = {param_device}")
        t0 = time.time()

        if args.saved_model is not None:
            agent.actor_local.load_state_dict(torch.load(args.saved_model)) # type: ignore
            print("WARNING: Pre-generated paths not available for saved model evaluation.")
        else:
            # Warm up the actor using the first 2048 episodes (no training) 
            # Iteratively optimize bias to maximize swing option price
            # Warm up the actor using the first warmup_episodes (no training) 
            # Iteratively optimize bias to maximize swing option price (Rprop)
            agent.calibrate_bias(
                env=train_env, 
                n_episodes=args.warmup_episodes, 
                max_iterations=20, 
                target_std=0.005
            )
        
        if args.eval_benchmark:
            print("\nRunning standalone evaluation benchmark (evaluation only)...")
            benchmark_evaluation(
                agent=agent,
                eval_env=eval_env,
                writer=tensorboard_writer,
                evaluations_dir=evaluations_dir,
                lsm_price=mean_lsm_price,
                parquet_writer=parquet_writer,
                eval_batch_size=args.eval_batch_size,
                profile=args.profile_eval,
            )
            eval_env.close()
            if parquet_writer:
                parquet_writer.stop()
                print("✅ AsyncParquetWriter stopped - all Parquet files written")
            TrainingManager.timer(t0, time.time())
            return

        if args.saved_model is None:
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
                lsm_price=mean_lsm_price,
                parquet_writer=parquet_writer,
            )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        t1 = time.time()
        TrainingManager.timer(t0, t1)

    finally:
        # Clean up environments
        if 'eval_env' in locals():
            eval_env.close()
        if 'train_env' in locals():
            train_env.close()
        
        # Stop the Parquet writer and wait for any pending writes
        if parquet_writer:
            parquet_writer.stop()
            print("✅ AsyncParquetWriter stopped - all Parquet files written")

        # Close TensorBoard writer
        if 'tensorboard_writer' in locals():
            tensorboard_writer.close()

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
    mp.set_start_method("spawn", force=True)
    main()
