"""
Swing Option Contract Specifications
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SwingContract:
    """
    Swing option contract parameters following Hambly et al. (2009)
    """
    # Local constraints (per decision time)
    q_min: float = 0.0              # Minimum lift per decision time
    q_max: float = 1.0              # Maximum lift per decision time
    
    # Global constraints (over contract life)
    Q_min: float = 0.0              # Global minimum volume
    Q_max: float = 10.0             # Global maximum volume (cap)
    
    # Contract terms
    strike: float = 100.0           # Strike price K
    maturity: float = 1.0           # Time to maturity in years
    n_rights: int = 250             # Number of decision dates
    
    # Refraction period (minimum periods between exercises)
    min_refraction_periods: int = 0    # If >0, after exercising you must wait this many periods before you can exercise again
    
    # Discount rate
    r: float = 0.05                 # Risk-free rate for discounting
    
    def __post_init__(self):
        """Validate contract parameters"""
        assert self.q_min >= 0, "q_min must be non-negative"
        assert self.q_max >= self.q_min, "q_max must be >= q_min"
        assert self.Q_max >= self.Q_min, "Q_max must be >= Q_min"
        assert self.n_rights > 0, "n_rights must be positive"
        assert self.maturity > 0, "maturity must be positive"
        assert self.r >= 0, "discount rate must be non-negative"
        
        # Ensure global constraints are feasible
        max_possible = self.q_max * self.n_rights
        assert self.Q_max <= max_possible, f"Q_max ({self.Q_max}) exceeds maximum possible volume ({max_possible})"
        
        min_possible = self.q_min * max(1, self.Q_min // self.q_max if self.q_max > 0 else 0)
        assert self.Q_min <= min_possible or self.q_min == 0, "Global minimum constraint may be infeasible"
    
    @property
    def dt(self) -> float:
        """Time step between decision dates"""
        return self.maturity / (self.n_rights - 1)
    
    @property
    def discount_factor(self) -> float:
        """Per-step discount factor"""
        return np.exp(-self.r * self.dt)
    
    def is_feasible_action(self, action: float, q_exercised: float, 
                          remaining_steps: int, last_exercise_step: int, 
                          current_step: int) -> bool:
        """
        Check if an action is feasible given current state
        
        Args:
            action: Proposed exercise quantity
            q_exercised: Total quantity exercised so far
            remaining_steps: Number of decision steps remaining
            last_exercise_step: Step when last exercise occurred (-1 if none)
            current_step: Current decision step
        """
        # Local constraints
        if action < self.q_min or action > self.q_max:
            return False
        
        # Global maximum constraint
        if q_exercised + action > self.Q_max:
            return False
        
        # Refraction constraint
        if (self.min_refraction_periods > 0 and 
            last_exercise_step >= 0 and 
            current_step - last_exercise_step <= self.min_refraction_periods):
            if action > 0:  # Only applies if we're trying to exercise
                return False
        
        # Check if remaining rights can satisfy global minimum
        max_remaining = self.q_max * remaining_steps
        if q_exercised + action + max_remaining < self.Q_min:
            return False
            
        return True
    
    def normalize_action(self, action: float) -> float:
        """Normalize action to [0, 1] range for RL agent"""
        if self.q_max == self.q_min:
            return 0.0
        return (action - self.q_min) / (self.q_max - self.q_min)
    
    def denormalize_action(self, normalized_action: float) -> float:
        """Convert normalized action [0, 1] back to contract quantities"""
        return self.q_min + normalized_action * (self.q_max - self.q_min)