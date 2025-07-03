"""
Swing Option Environment for Reinforcement Learning
"""
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from .simulate_hhk_spot import DEFAULT_HHK_PARAMS, simulate_single_path
from .swing_contract import DEFAULT_CONTRACT, SwingContract


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
                 contract: Optional[SwingContract] = None,
                 hhk_params: Optional[Dict] = None,
                 max_episode_steps: Optional[int] = None):
        """
        Initialize swing option environment
        
        Args:
            contract: Swing option contract specifications
            hhk_params: HHK model parameters for underlying simulation
            max_episode_steps: Maximum steps per episode (defaults to contract n_rights)
        """
        super().__init__()
        
        self.contract = contract or DEFAULT_CONTRACT
        self.hhk_params = hhk_params or DEFAULT_HHK_PARAMS.copy()
        self.max_episode_steps = max_episode_steps or self.contract.n_rights
        
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
        
        # Episode state
        self.reset()
        
    def set_pregenerated_path(self, t_grid: np.ndarray, spot_path: np.ndarray, 
                             X_path: np.ndarray, Y_path: np.ndarray):
        """
        Set pre-generated paths for this environment instance.
        This allows for variance reduction by using Sobol-generated paths.
        
        Args:
            t_grid: Time grid for the paths
            spot_path: Pre-generated spot price path
            X_path: Pre-generated X process path  
            Y_path: Pre-generated Y process path
        """
        self.t_grid = t_grid.copy()
        self.spot_path = spot_path.copy()
        self.X_path = X_path.copy()
        self.Y_path = Y_path.copy()
        self._using_pregenerated = True
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one step in the environment"""
        action_value = float(action[0])
        
        # Denormalize action to contract quantity
        q_proposed = self.contract.denormalize_action(action_value)
        
        # Check feasibility and clip if necessary
        q_actual = self._get_feasible_action(q_proposed)
        
        # Calculate reward (immediate payoff)
        spot_price = self.spot_path[self.current_step]
        payoff_per_unit = max(spot_price - self.contract.strike, 0.0)
        immediate_reward = q_actual * payoff_per_unit
        
        # Apply discounting
        discount_factor = self.contract.discount_factor ** self.current_step
        discounted_reward = discount_factor * immediate_reward
        
        # Update state
        if q_actual > 1e-6:  # Threshold for "exercise occurred"
            self.last_exercise_step = self.current_step
            
        self.q_exercised += q_actual
        self.episode_return += discounted_reward
        self.current_step += 1
        
        # Check termination conditions
        terminated = (self.current_step >= self.contract.n_rights or 
                     self.q_exercised >= self.contract.Q_max - 1e-6)
        truncated = False
        
        # Penalty for not meeting global minimum (applied at termination)
        terminal_penalty = 0.0
        if terminated and self.q_exercised < self.contract.Q_min:
            # Penalty proportional to unmet obligation
            shortage = self.contract.Q_min - self.q_exercised
            terminal_penalty = -shortage * self.contract.strike  # Negative of strike price
            
        total_reward = discounted_reward + terminal_penalty
        
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
        
        next_obs = self._get_observation() if not terminated else self._get_observation()
        
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
        X_t = self.X_path[self.current_step] if hasattr(self, 'X_path') else 0.0
        Y_t = self.Y_path[self.current_step] if hasattr(self, 'Y_path') else 0.0
        
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
        return float(np.std(log_returns) * np.sqrt(252))  # Annualized volatility
    
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

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Only generate new paths if we're not using pre-generated ones
        if not hasattr(self, '_using_pregenerated') or not self._using_pregenerated:
            if seed is not None:
                episode_seed = seed
            else:
                episode_seed = None
                
            # Generate new price path
            self.t_grid, self.spot_path = simulate_single_path(
                T=self.contract.maturity,
                n_steps=self.contract.n_rights,
                seed=episode_seed,
                **self.hhk_params
            )
            
            # Also get full simulation for X, Y processes
            from .simulate_hhk_spot import simulate_hhk_spot
            _, S_full, X_full, Y_full = simulate_hhk_spot(
                T=self.contract.maturity,
                n_steps=self.contract.n_rights,
                n_paths=1,
                seed=episode_seed,
                **self.hhk_params
            )
            self.spot_path = S_full[0]
            self.X_path = X_full[0] 
            self.Y_path = Y_full[0]
        # If using pre-generated paths, they're already set via set_pregenerated_path()
        
        # Initialize episode state
        self.current_step = 0
        self.q_exercised = 0.0
        self.last_exercise_step = -1
        self.episode_return = 0.0
        
        # Calculate initial volatility
        self.recent_volatility = self._calculate_recent_volatility(0)
        
        return self._get_observation(), {}


def create_swing_env(contract_type: str = 'default') -> SwingOptionEnv:
    """
    Factory function to create swing option environments
    
    Args:
        contract_type: Type of contract ('default', 'positive_min', 'custom')
        
    Returns:
        SwingOptionEnv instance
    """
    if contract_type == 'default':
        from .swing_contract import DEFAULT_CONTRACT
        return SwingOptionEnv(contract=DEFAULT_CONTRACT)
    elif contract_type == 'positive_min':
        from .swing_contract import POSITIVE_MIN_CONTRACT
        return SwingOptionEnv(contract=POSITIVE_MIN_CONTRACT)
    else:
        return SwingOptionEnv()


if __name__ == "__main__":
    # Test environment
    env = create_swing_env('default')
    
    print("=== Swing Option Environment Test ===")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Contract: q_range=[{env.contract.q_min}, {env.contract.q_max}], "
          f"Q_range=[{env.contract.Q_min}, {env.contract.Q_max}], "
          f"K={env.contract.strike}, T={env.contract.maturity}")
    
    # Run test episode
    obs, _ = env.reset(seed=42)
    total_reward = 0.0
    exercised_amounts = []
    
    print(f"\nInitial observation: {obs}")
    
    for step in range(min(10, env.contract.n_rights)):
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        exercised_amounts.append(info['q_actual'])
        
        print(f"Step {step}: S={info['spot_price']:.2f}, "
              f"action={action[0]:.3f}, q_actual={info['q_actual']:.3f}, "
              f"reward={reward:.3f}, cum_exercised={info['cumulative_exercised']:.3f}")
        
        if terminated or truncated:
            break
    
    print(f"\nTest completed: Total reward = {total_reward:.3f}")
    print(f"Total exercised: {sum(exercised_amounts):.3f}")
    env.close()
