"""
Swing Option Environment for Reinforcement Learning

PRICING FORMULA IMPLEMENTATION:
===============================

This environment implements the swing option pricing formula:

1. Per-step Payoff:
   Payoff at time t = q_t * (S_t - K)^+

2. Path-wise Total Discounted Payoff:
   P_path = sum_{t=1}^T e^{-r*t} * q_t * (S_t - K)^+

3. Option Value (Monte Carlo Estimate):
   V_0 = (1/N) * sum_{i=1}^N P_path,i

Where:
- q_t: Exercise quantity at time t (q_actual in the code)
- S_t: Spot price at time t
- K: Strike price
- r: Risk-free rate
- t: Time = (step + 1) * dt
- N: Number of Monte Carlo paths

The calculate_standardized_reward() function implements the per-step
discounted payoff calculation, and the evaluation functions in run.py
compute the Monte Carlo average.
"""
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from .swing_contract import SwingContract


def calculate_standardized_reward(
    spot_price: float,
    q_actual: float,
    strike: float,
    current_step: int,
    discount_factor: float,
    cost_coefficient: float,
    cost_exponent: float,
) -> Tuple[float, float, float, float]:
    """
    Standardized reward calculation for reinforcement learning
    
    Updated to match the swing option pricing formula:
    Per-step Payoff: q_t * (S_t - K)^+
    Path-wise Total: sum_{j=0}^{T-1} (df**j) * q_j * (S_j - K)^+
    
    Args:
        spot_price: Current spot price
        q_actual: Actual exercise quantity
        strike: Strike price
        current_step: 0-based time step index j
        discount_factor: Discount factor per step
        cost_coefficient: Convex exercise cost coefficient
        cost_exponent: Convex exercise cost exponent
        
    Returns:
        Tuple (discounted reward, gross payoff, exercise cost, net payoff)
    """
    # Calculate immediate payoff: q_t * (S_t - K)^+
    payoff_per_unit = max(spot_price - strike, 0.0)
    
    gross_payoff = q_actual * payoff_per_unit
    exercise_cost = cost_coefficient * (q_actual ** cost_exponent)
    net_payoff = gross_payoff - exercise_cost
    
    # Apply discrete discounting with 0-based exponent (aligns with t_j = j * dt)
    discounted_reward = (discount_factor ** current_step) * net_payoff
    
    return discounted_reward, gross_payoff, exercise_cost, net_payoff


class SwingOptionEnv(gym.Env):
    """
    Gymnasium environment for swing option pricing using D4PG
    
    State: [S_t, Q_exercised, Q_remaining, time_to_maturity, normalized_time, 
            X_t, Y_t, recent_volatility, days_since_last_exercise]
    
    Action: Normalized exercise quantity in [0, 1] 
            (gets mapped to [q_min, q_max] range)
    
    Reward: Immediate payoff from exercise: q_t * max(S_t - K, 0)
    """
    
    def __init__(self, 
                 contract: SwingContract,
                 hhk_params: Dict,
                 dataset:Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 obs_dtype: Optional[np.dtype] = None):
        """
        Initialize swing option environment
        
        Args:
            contract: Swing option contract specifications
            hhk_params: HHK model parameters for underlying simulation
            dataset: Tuple of (t_paths, S_paths, X_paths, Y_paths) pre-generated data
            max_episode_steps: Maximum steps per episode (defaults to contract n_rights)
        """
        super().__init__()

        self.contract = contract
        self.hhk_params = hhk_params
        
        # Unpack dataset into individual components for easier access
        self.t, self.S, self.X, self.Y = dataset
        self.max_episode_steps = self.contract.n_rights
        self.obs_dtype = np.dtype(obs_dtype) if obs_dtype is not None else np.float32
        
        # Action space: normalized exercise quantity [0, 1]
        self.action_space = Box(
            low=0.0, 
            high=1.0, 
            shape=(1,), 
            dtype=self.obs_dtype
        )
        
        # State space dimensions
        # [S_t, Q_exercised, Q_remaining, time_to_maturity, normalized_time,
        #  X_t, Y_t, recent_volatility, days_since_last_exercise]
        state_dim = 9
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=self.obs_dtype
        )
        
        # Episode tracking - Will be incremented to 0 on first reset()
        self._episode_counter = -1  # Episode counter starts at 0 (which equals path index 0)
        
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one step in the environment"""
        # Safely extract scalar action to avoid NumPy deprecation warnings
        action_value = float(np.asarray(action).reshape(-1)[0])

        # Denormalize action to contract quantity
        q_proposed = self.contract.denormalize_action(action_value)

        # Check feasibility and clip if necessary
        q_actual = self._get_feasible_action(q_proposed)

        # Current spot at this decision time
        current_step = self.current_step
        spot_price = self.spot_path[current_step]

        # Evaluate payoff before committing to an exercise
        total_reward, gross_payoff, exercise_cost, net_payoff = calculate_standardized_reward(
            spot_price=spot_price,
            q_actual=q_actual,
            strike=self.contract.strike,
            current_step=current_step,
            discount_factor=self.contract.discount_factor,
            cost_coefficient=self.contract.c_cost,
            cost_exponent=self.contract.gamma_cost,
        )

        # Enforce q_t = 0 whenever the realized reward would be non-positive
        action_masked = False
        if net_payoff <= 0.0:
            action_masked = q_actual > 0.0
            q_actual = 0.0
            total_reward = 0.0
            gross_payoff = 0.0
            exercise_cost = 0.0
            net_payoff = 0.0

        # Track last exercise time if any amount exercised
        if q_actual > 1e-6:
            self.last_exercise_step = current_step

        # Compute new cumulative exercised and advance time
        new_q_exercised = self.q_exercised + q_actual
        self.current_step += 1

        # Termination conditions
        terminated = (
            self.current_step >= self.contract.n_rights
            or new_q_exercised >= self.contract.Q_max - 1e-6
        )
        truncated = False

        # Update episode bookkeeping
        self.q_exercised = new_q_exercised
        self.episode_return += total_reward

        # Info for analysis/logging
        info = {
            "spot_price": spot_price,
            "q_proposed": q_proposed,
            "q_actual": q_actual,
            "gross_payoff": gross_payoff,
            "exercise_cost": exercise_cost,
            "immediate_payoff": net_payoff,
            "discounted_reward": total_reward,
            "terminal_penalty": 0.0,
            "cumulative_exercised": self.q_exercised,
            "episode_return": self.episode_return,
            "action_masked": action_masked,
        }

        next_obs = self._get_observation()
        return next_obs, total_reward, terminated, truncated, info
    
    def _get_feasible_action(self, q_proposed: float) -> float:
        """
        Ensure action satisfies all constraints
        """
        # Local constraints
        q_feasible = np.clip(q_proposed, self.contract.q_min, self.contract.q_max)
        
        # Global maximum constraint
        max_allowed = self.contract.Q_max - self.q_exercised
        q_feasible = min(q_feasible, max_allowed)
        
        # Refraction constraint
        if (self.contract.min_refraction_periods > 0 and 
            self.last_exercise_step >= 0 and
            self.current_step - self.last_exercise_step <= self.contract.min_refraction_periods):
            q_feasible = 0.0
            
        # Ensure we can still meet global minimum in remaining steps
        remaining_steps = self.contract.n_rights - self.current_step - 1
        if remaining_steps > 0:
            min_needed_later = max(0, self.contract.Q_min - self.q_exercised - q_feasible)
            max_possible_later = self.contract.q_max * remaining_steps
            if min_needed_later > max_possible_later:
                # Must exercise more now to meet minimum
                required_now = min_needed_later - max_possible_later
                q_feasible = max(q_feasible, required_now)
                q_feasible = min(q_feasible, self.contract.q_max)  # Respect local max
        
        return max(0.0, q_feasible)
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct state observation vector
        """
        if self.current_step >= len(self.spot_path):
            self.current_step = len(self.spot_path) - 1
            
        # Current state variables
        spot_price = self.spot_path[self.current_step]
        q_remaining = self.contract.Q_max - self.q_exercised
        time_to_maturity = (self.contract.n_rights - self.current_step) * self.contract.dt
        normalized_time = self.current_step / self.contract.n_rights
        
        # Underlying process states
        X_t = self.X_path[self.current_step]
        Y_t = self.Y_path[self.current_step]
        
        # Recent volatility
        # self.recent_volatility = self._calculate_recent_volatility(self.current_step)
        
        # Days since last exercise
        days_since_exercise = (self.current_step - self.last_exercise_step 
                              if self.last_exercise_step >= 0 else self.current_step)
        
        # State:
        # - spot_minus_strike (S_t - K)

        state = np.array([
            spot_price - self.contract.strike,  # spot_minus_strike (S_t - K)
            self.q_exercised / self.contract.Q_max,  # Normalized cumulative exercise
            q_remaining / self.contract.Q_max,  # Normalized remaining capacity
            time_to_maturity / self.contract.maturity,  # Normalized time to maturity
            normalized_time,  # Progress through contract
            spot_price, # Spot Price
            X_t,  # Mean-reverting component
            Y_t,  # Jump component  
            # self.recent_volatility,  # Recent realized volatility
            days_since_exercise / self.contract.n_rights  # Normalized refraction time
        ], dtype=self.obs_dtype)
        return state
    
    def _calculate_recent_volatility(self, current_idx: int, lookback: int = 10) -> float:
        """Calculate recent realized volatility"""
        if current_idx < lookback:
            lookback = current_idx
            
        if lookback <= 1:
            return 0.0
            
        # Calculate log returns over lookback period
        prices = self.spot_path[max(0, current_idx - lookback):current_idx + 1]
        if len(prices) <= 1:
            return 0.0
            
        log_returns = np.diff(np.log(prices))
        return float(np.std(log_returns) * 16)  # Annualized volatility - 16 ~ sqrt(252)
    
    def render(self, mode: str = 'human') -> None:
        """Render environment (not implemented)"""
        pass
    
    def close(self) -> None:
        """Clean up environment"""
        pass
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment"""
        return self

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Increment episode counter
        self._episode_counter += 1
        
        # Use direct mapping: episode counter directly corresponds to path index
        # Episode 0 -> path 0, Episode 1 -> path 1, etc.
        path_idx = self._episode_counter
        self.time_path = self.t[path_idx] if self.t.ndim > 1 else self.t
        self.spot_path = self.S[path_idx]
        self.X_path = self.X[path_idx] 
        self.Y_path = self.Y[path_idx]
        
        # Initialize episode state
        self.current_step = 0
        self.q_exercised = 0.0
        self.last_exercise_step = -1
        self.episode_return = 0.0
        
        # Calculate initial volatility
        # self.recent_volatility = self._calculate_recent_volatility(current_idx=0)
        
        return self._get_observation(), {}
