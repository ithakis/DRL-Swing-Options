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


def calculate_standardized_reward(spot_price: float, q_actual: float, strike: float, 
                                current_step: int, discount_factor: float) -> float:
    """
    Standardized reward calculation for reinforcement learning
    
    Updated to match the swing option pricing formula:
    Per-step Payoff: q_t * (S_t - K)^+
    Path-wise Total: sum_{t=1}^T e^{-r*t} * q_t * (S_t - K)^+
    
    Args:
        spot_price: Current spot price
        q_actual: Actual exercise quantity
        strike: Strike price
        current_step: Current time step
        discount_factor: Discount factor per step
        
    Returns:
        Discounted reward
    """
    # Calculate immediate payoff: q_t * (S_t - K)^+
    payoff_per_unit = max(spot_price - strike, 0.0)
    immediate_payoff = q_actual * payoff_per_unit
    
    # Apply discrete discounting
    discounted_reward = (discount_factor ** (current_step + 1)) * immediate_payoff
    
    return discounted_reward


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
                 dataset:Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
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
        
        # Action space: normalized exercise quantity [0, 1]
        self.action_space = Box(
            low=0.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # State space dimensions
        # [S_t, Q_exercised, Q_remaining, time_to_maturity, normalized_time,
        #  X_t, Y_t, recent_volatility, days_since_last_exercise]
        state_dim = 9
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float32
        )
        
        # Episode tracking - Will be incremented to 0 on first reset()
        self._episode_counter = -1  # Episode counter starts at 0 (which equals path index 0)
        
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one step in the environment"""
        action_value = float(action[0])
        
        # Denormalize action to contract quantity
        q_proposed = self.contract.denormalize_action(action_value)
        
        # Check feasibility and clip if necessary
        q_actual = self._get_feasible_action(q_proposed)
        
        # Calculate reward using standardized function
        spot_price = self.spot_path[self.current_step]
        
        # Update state first to get current totals
        if q_actual > 1e-6:  # Threshold for "exercise occurred"
            self.last_exercise_step = self.current_step
            
        new_q_exercised = self.q_exercised + q_actual
        self.current_step += 1
        
        # Check termination conditions
        terminated = (self.current_step >= self.contract.n_rights or 
                     new_q_exercised >= self.contract.Q_max - 1e-6)
        truncated = False
        
        # Calculate total reward including terminal penalty if needed
        # Use continuous discounting as per the swing option pricing formula
        total_reward = calculate_standardized_reward(
            spot_price, q_actual, self.contract.strike, 
            self.current_step - 1, self.contract.discount_factor
        )
        
        # Update episode state
        self.q_exercised = new_q_exercised
        self.episode_return += total_reward
        
        # Calculate components for info (for backward compatibility)
        immediate_reward = q_actual * max(spot_price - self.contract.strike, 0.0)
        discounted_reward = (self.contract.discount_factor ** (self.current_step - 1)) * immediate_reward
        terminal_penalty = total_reward - discounted_reward
        
        info = {
            'spot_price': spot_price,
            'q_proposed': q_proposed,
            'q_actual': q_actual,
            'immediate_payoff': immediate_reward,
            'discounted_reward': discounted_reward,
            'terminal_penalty': terminal_penalty,
            'cumulative_exercised': self.q_exercised,
            'episode_return': self.episode_return
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
        if (self.contract.min_refraction_days > 0 and 
            self.last_exercise_step >= 0 and
            self.current_step - self.last_exercise_step < self.contract.min_refraction_days):
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
        self.recent_volatility = self._calculate_recent_volatility(self.current_step)
        
        # Days since last exercise
        days_since_exercise = (self.current_step - self.last_exercise_step 
                              if self.last_exercise_step >= 0 else self.current_step)
        
        state = np.array([
            spot_price / self.contract.strike,  # Normalized by strike
            self.q_exercised / self.contract.Q_max,  # Normalized cumulative exercise
            q_remaining / self.contract.Q_max,  # Normalized remaining capacity
            time_to_maturity / self.contract.maturity,  # Normalized time to maturity
            normalized_time,  # Progress through contract
            X_t,  # Mean-reverting component
            Y_t,  # Jump component  
            self.recent_volatility,  # Recent realized volatility
            days_since_exercise / self.contract.n_rights  # Normalized refraction time
        ], dtype=np.float32)
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
        self.recent_volatility = self._calculate_recent_volatility(current_idx=0)
        
        return self._get_observation(), {}