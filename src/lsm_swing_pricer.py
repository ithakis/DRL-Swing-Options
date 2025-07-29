"""
Least Squares Monte Carlo (LSM) Swing Option Pricer
Based on Hambly-Howison-Kluge spot model and LSM methodology from:
- "Modelling spikes and pricing swing options in electricity markets" Hambly et al. (2009)
- "Operating a Swing Option on Today's Gas Markets" Hanfeld & Schlüter

Implementation follows the 4-step LSM algorithm:
1. Discretize time and cumulative offtake levels
2. Formulate the decision problem  
3. Apply LSMC algorithm with backward recursion
4. Use forward induction to find optimal strategy
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import warnings
import argparse
import time
from datetime import datetime
warnings.filterwarnings('ignore')

try:
    # When imported as a module from outside src/
    from .swing_contract import SwingContract
    from .simulate_hhk_spot import simulate_hhk_spot
except ImportError:
    # When run directly from src/ directory or as script
    from swing_contract import SwingContract
    from simulate_hhk_spot import simulate_hhk_spot
# # Import existing classes from the codebase
# try:
#     # When imported as a module from outside src/
#     from .swing_contract import SwingContract
#     from .simulate_hhk_spot import simulate_hhk_spot
# except ImportError:
#     # When run directly from src/ directory
#     from swing_contract import SwingContract
#     from simulate_hhk_spot import simulate_hhk_spot


class LSMSwingPricer:
    """Least Squares Monte Carlo swing option pricer"""
    
    def __init__(
        self,
        contract: SwingContract,
        hhk_params: Dict,
        n_paths: int = 16384,
        poly_degree: int = 3,
        seed: Optional[int] = None
    ):
        self.contract = contract
        self.hhk_params = hhk_params  
        self.n_paths = n_paths
        self.poly_degree = poly_degree
        self.seed = seed
        
        # State space discretization
        self.create_state_grid()
        
    def create_state_grid(self):
        """Create discretized state space for cumulative exercise"""
        max_exercises = int(self.contract.Q_max / self.contract.q_max) + 1
        self.Q_grid = np.linspace(0, self.contract.Q_max, max_exercises)
        self.n_Q_states = len(self.Q_grid)
        
    def payoff(self, S: float, action: float) -> float:
        """Immediate payoff from exercising action at spot price S"""
        return action * max(S - self.contract.strike, 0)
        
    def is_feasible_action(
        self, 
        action: float, 
        q_cumulative: float, 
        remaining_steps: int
    ) -> bool:
        """Check if action is feasible given current state"""
        # Local constraints
        if action < self.contract.q_min or action > self.contract.q_max:
            return False
            
        # Global maximum constraint
        if q_cumulative + action > self.contract.Q_max:
            return False
            
        # Check if we can still meet global minimum
        max_future = self.contract.q_max * remaining_steps
        if q_cumulative + action + max_future < self.contract.Q_min:
            return False
            
        return True
        
    def get_feasible_actions(self, q_cumulative: float, remaining_steps: int) -> np.ndarray:
        """Get all feasible actions for current state"""
        actions = [0.0, self.contract.q_max]  # No exercise or max exercise (bang-bang)
        
        feasible_actions = []
        for action in actions:
            if self.is_feasible_action(action, q_cumulative, remaining_steps):
                feasible_actions.append(action)
                
        return np.array(feasible_actions)
        
    def fit_continuation_value(
        self,
        spot_prices: np.ndarray,
        continuation_values: np.ndarray
    ) -> LinearRegression:
        """Fit continuation value using polynomial regression"""
        if len(spot_prices) == 0 or np.all(continuation_values == 0):
            # Return dummy regressor for edge cases
            reg = LinearRegression()
            reg.coef_ = np.zeros(self.poly_degree + 1)
            reg.intercept_ = 0.0
            return reg
        
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=self.poly_degree, include_bias=True)
        X = poly_features.fit_transform(spot_prices.reshape(-1, 1))
        
        # Fit regression
        reg = LinearRegression()
        reg.fit(X, continuation_values)
        
        return reg
        
    def evaluate_continuation_value(
        self,
        regressor: LinearRegression,
        spot_price: float
    ) -> float:
        """Evaluate continuation value at given spot price"""
        if not hasattr(regressor, 'coef_'):
            return 0.0
            
        poly_features = PolynomialFeatures(degree=self.poly_degree, include_bias=True)
        X = poly_features.fit_transform(np.array([[spot_price]]))
        
        return regressor.predict(X)[0]
        
    def price_option(self, verbose: bool = True) -> Dict:
        """
        Price swing option using LSM algorithm
        
        Returns:
            Dictionary with pricing results and diagnostics
        """
        if verbose:
            print("Pricing swing option using LSM...")
            print(f"Contract: Strike={self.contract.strike}, Maturity={self.contract.maturity:.4f}")
            print(f"Rights: {self.contract.n_rights}, Q_max={self.contract.Q_max}")
            print(f"Paths: {self.n_paths}, Polynomial degree: {self.poly_degree}")
        
        # Step 1: Discretize time and cumulative offtake levels
        t, S, X, Y = simulate_hhk_spot(
            T=self.contract.maturity,
            n_steps=self.contract.n_rights,
            n_paths=self.n_paths,
            seed=self.seed,
            **self.hhk_params
        )
        
        # Step 2: Initialize value arrays
        # V[path, time, q_state] = option value
        V = np.zeros((self.n_paths, self.contract.n_rights + 1, self.n_Q_states))
        
        # Store regression models for each (time, q_state)
        regressors = {}
        
        # Step 3: Backward recursion (LSM algorithm)
        if verbose:
            print("Running backward recursion...")
            
        # Terminal conditions: V[path, T, q] = 0 (no value at expiry)
        V[:, -1, :] = 0.0
        
        # Work backwards through time
        for t_idx in range(self.contract.n_rights - 1, -1, -1):
            remaining_steps = self.contract.n_rights - t_idx
            
            for q_idx, q_cumulative in enumerate(self.Q_grid):
                # Get feasible actions
                feasible_actions = self.get_feasible_actions(q_cumulative, remaining_steps)
                
                if len(feasible_actions) == 0:
                    V[:, t_idx, q_idx] = 0.0
                    continue
                
                # For each path, find optimal action
                path_values = np.zeros(self.n_paths)
                
                for path in range(self.n_paths):
                    S_current = S[path, t_idx]
                    
                    best_value = -np.inf
                    best_action = 0.0
                    
                    for action in feasible_actions:
                        # Immediate payoff
                        immediate_payoff = self.payoff(S_current, action)
                        
                        # Find next state
                        next_q_cumulative = q_cumulative + action
                        next_q_idx = np.argmin(np.abs(self.Q_grid - next_q_cumulative))
                        
                        # Continuation value
                        if t_idx < self.contract.n_rights - 1:
                            continuation_value = V[path, t_idx + 1, next_q_idx]
                        else:
                            continuation_value = 0.0
                        
                        total_value = immediate_payoff + self.contract.discount_factor * continuation_value
                        
                        if total_value > best_value:
                            best_value = total_value
                            best_action = action
                    
                    path_values[path] = best_value
                
                V[:, t_idx, q_idx] = path_values
                
                # Fit regression for continuation value (for analysis purposes)
                if t_idx > 0:
                    exercise_paths = []
                    continuation_values = []
                    for path in range(self.n_paths):
                        if V[path, t_idx, q_idx] > 0:  # Only consider paths with positive value
                            exercise_paths.append(S[path, t_idx])
                            continuation_values.append(V[path, t_idx, q_idx])
                    
                    if len(exercise_paths) > 0:
                        reg = self.fit_continuation_value(
                            np.array(exercise_paths), 
                            np.array(continuation_values)
                        )
                        regressors[(t_idx, q_idx)] = reg
        
        # Step 4: Calculate option value
        option_values = V[:, 0, 0]  # Value at t=0, q=0
        option_price = np.mean(option_values)
        option_std = np.std(option_values) / np.sqrt(self.n_paths)
        
        # Additional diagnostics
        final_spots = S[:, -1]
        
        results = {
            'option_price': option_price,
            'option_std_error': option_std,
            'confidence_interval_95': (
                option_price - 1.96 * option_std,
                option_price + 1.96 * option_std
            ),
            'paths_used': self.n_paths,
            'final_spot_mean': np.mean(final_spots),
            'final_spot_std': np.std(final_spots),
            'intrinsic_value': np.mean(np.maximum(final_spots - self.contract.strike, 0)),
            'time_grid': t,
            'spot_paths': S,
            'option_values': option_values,
            'regressors': regressors
        }
        
        if verbose:
            print(f"\nPricing Results:")
            print(f"Option Price: {option_price:.4f} ± {option_std:.4f}")
            print(f"95% CI: [{results['confidence_interval_95'][0]:.4f}, {results['confidence_interval_95'][1]:.4f}]")
            print(f"Final Spot Mean: {results['final_spot_mean']:.2f}")
            print(f"Intrinsic Value: {results['intrinsic_value']:.4f}")
        
        return results
    
    def price_option_with_pregenerated_paths(self, eval_t, eval_S, eval_X, eval_Y, verbose: bool = True) -> Dict:
        """
        Price swing option using pre-generated Monte Carlo paths
        
        Args:
            eval_t: Pre-generated time grid
            eval_S: Pre-generated spot price paths
            eval_X: Pre-generated X process paths  
            eval_Y: Pre-generated Y process paths
            verbose: Whether to print progress
            
        Returns:
            Dictionary with pricing results and diagnostics
        """
        if verbose:
            print("Pricing swing option using LSM with pre-generated paths...")
            print(f"Contract: Strike={self.contract.strike}, Maturity={self.contract.maturity:.4f}")
            print(f"Rights: {self.contract.n_rights}, Q_max={self.contract.Q_max}")
            print(f"Paths: {eval_S.shape[0]}, Polynomial degree: {self.poly_degree}")
        
        # Use pre-generated paths
        t, S, X, Y = eval_t, eval_S, eval_X, eval_Y
        actual_n_paths = S.shape[0]
        
        # Step 2: Initialize value arrays
        # V[path, time, q_state] = option value
        V = np.zeros((actual_n_paths, self.contract.n_rights + 1, self.n_Q_states))
        
        # Store regression models for each (time, q_state)
        regressors = {}
        
        # Step 3: Backward recursion (LSM algorithm)
        if verbose:
            print("Running backward recursion...")
            
        # Terminal conditions: V[path, T, q] = 0 (no value at expiry)
        V[:, -1, :] = 0.0
        
        # Work backwards through time
        for t_idx in range(self.contract.n_rights - 1, -1, -1):
            remaining_steps = self.contract.n_rights - t_idx
            
            for q_idx, q_cumulative in enumerate(self.Q_grid):
                # Get feasible actions
                feasible_actions = self.get_feasible_actions(q_cumulative, remaining_steps)
                
                if len(feasible_actions) == 0:
                    V[:, t_idx, q_idx] = 0.0
                    continue
                
                # For each path, find optimal action
                path_values = np.zeros(actual_n_paths)
                
                for path in range(actual_n_paths):
                    S_current = S[path, t_idx]
                    
                    best_value = -np.inf
                    
                    for action in feasible_actions:
                        # Immediate payoff
                        immediate = self.payoff(S_current, action)
                        
                        # Find next state
                        q_next = q_cumulative + action
                        q_next_idx = np.argmin(np.abs(self.Q_grid - q_next))
                        
                        # Continuation value (already discounted from previous step)
                        continuation = V[path, t_idx + 1, q_next_idx]
                        
                        total_value = immediate + self.contract.discount_factor * continuation
                        
                        if total_value > best_value:
                            best_value = total_value
                    
                    path_values[path] = best_value
                
                V[:, t_idx, q_idx] = path_values
                
                # Fit regression for continuation value (for next backward step)
                if t_idx > 0:  # No need to fit at t=0
                    continuation_values = path_values.copy()
                    spot_prices = S[:, t_idx]
                    
                    # Only fit on in-the-money paths to avoid bias
                    itm_mask = spot_prices > self.contract.strike * 0.8  # 80% moneyness threshold
                    
                    if np.sum(itm_mask) > self.poly_degree + 1:  # Need enough points
                        reg = self.fit_continuation_value(
                            spot_prices[itm_mask],
                            continuation_values[itm_mask]
                        )
                        regressors[(t_idx, q_idx)] = reg
        
        # Step 4: Calculate option value
        option_values = V[:, 0, 0]  # Value at t=0, q=0
        option_price = np.mean(option_values)
        option_std = np.std(option_values) / np.sqrt(actual_n_paths)
        
        # Additional diagnostics
        final_spots = S[:, -1]
        
        results = {
            'option_price': option_price,
            'option_std_error': option_std,
            'confidence_interval_95': (
                option_price - 1.96 * option_std,
                option_price + 1.96 * option_std
            ),
            'paths_used': actual_n_paths,
            'final_spot_mean': np.mean(final_spots),
            'final_spot_std': np.std(final_spots),
            'intrinsic_value': np.mean(np.maximum(final_spots - self.contract.strike, 0)),
            'time_grid': t,
            'spot_paths': S,
            'option_values': option_values,
            'regressors': regressors
        }
        
        if verbose:
            print(f"\nPricing Results:")
            print(f"Option Price: {option_price:.4f} ± {option_std:.4f}")
            print(f"95% CI: [{results['confidence_interval_95'][0]:.4f}, {results['confidence_interval_95'][1]:.4f}]")
            print(f"Final Spot Mean: {results['final_spot_mean']:.2f}")
            print(f"Intrinsic Value: {results['intrinsic_value']:.4f}")
        
        return results
    
    def simulate_optimal_strategy(
        self,
        results: Dict,
        n_scenarios: int = 10,
        plot: bool = True,
        save_step_data: bool = False,
        evaluation_runs_dir: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> Dict:
        """
        Simulate optimal exercise strategy using forward induction
        
        Args:
            results: Results from price_option()
            n_scenarios: Number of price scenarios to simulate
            plot: Whether to create plots
            save_step_data: Whether to save detailed step-by-step data
            evaluation_runs_dir: Directory to save evaluation runs data
            run_name: Name of the run for file naming
        """
        print(f"Simulating optimal strategy for {n_scenarios} scenarios...")
        
        # Simulate new price paths for strategy testing
        t, S_test, _, _ = simulate_hhk_spot(
            T=self.contract.maturity,
            n_steps=self.contract.n_rights,
            n_paths=n_scenarios,
            seed=self.seed + 1000 if self.seed else None,
            **self.hhk_params
        )
        
        strategies = []
        all_episodes_step_data = []  # Store step data for CSV saving
        
        for scenario in range(n_scenarios):
            strategy = {
                'times': [],
                'spot_prices': [],
                'actions': [],
                'cumulative_exercise': [],
                'payoffs': [],
                'total_payoff': 0.0
            }
            
            episode_step_data = []  # Store step data for this episode
            q_cumulative = 0.0
            
            # Forward induction
            for t_idx in range(self.contract.n_rights):
                remaining_steps = self.contract.n_rights - t_idx
                S_current = S_test[scenario, t_idx]
                time_left = self.contract.maturity * (remaining_steps / self.contract.n_rights)
                
                # Find current q_state
                q_idx = np.argmin(np.abs(self.Q_grid - q_cumulative))
                
                # Get feasible actions
                feasible_actions = self.get_feasible_actions(q_cumulative, remaining_steps)
                
                if len(feasible_actions) == 0:
                    action = 0.0
                else:
                    # Choose action using regressor if available
                    if (t_idx, q_idx) in results['regressors']:
                        reg = results['regressors'][(t_idx, q_idx)]
                        
                        best_value = -np.inf
                        best_action = 0.0
                        
                        for action_candidate in feasible_actions:
                            immediate = self.payoff(S_current, action_candidate)
                            
                            if t_idx < self.contract.n_rights - 1:
                                q_next = q_cumulative + action_candidate
                                q_next_idx = np.argmin(np.abs(self.Q_grid - q_next))
                                continuation = self.evaluate_continuation_value(reg, S_current)
                            else:
                                continuation = 0.0
                            
                            total_value = immediate + self.contract.discount_factor * continuation
                            
                            if total_value > best_value:
                                best_value = total_value
                                best_action = action_candidate
                        
                        action = best_action
                    else:
                        # Simple heuristic: exercise if in-the-money
                        if S_current > self.contract.strike and self.contract.q_max in feasible_actions:
                            action = self.contract.q_max
                        else:
                            action = 0.0
                            action = 0.0
                
                # Execute action
                payoff = self.payoff(S_current, action)
                q_actual = action  # Actual quantity exercised
                q_cumulative += action
                
                # Store step data for CSV export (matching RL agent format)
                if save_step_data:
                    step_info = {
                        'step': t_idx,
                        'spot_price': S_current,
                        'q_remaining': self.contract.Q_max - q_cumulative + action,  # Before action
                        'q_exercised': q_cumulative - action,  # Before action
                        'time_left': time_left,
                        'action': action,
                        'q_actual': q_actual,
                        'reward': payoff
                    }
                    episode_step_data.append(step_info)
                
                strategy['times'].append(t[t_idx])
                strategy['spot_prices'].append(S_current)
                strategy['actions'].append(action)
                strategy['cumulative_exercise'].append(q_cumulative)
                strategy['payoffs'].append(payoff)
                strategy['total_payoff'] += payoff * self.contract.discount_factor**t_idx
            
            strategies.append(strategy)
            if save_step_data:
                all_episodes_step_data.append(episode_step_data)
        
        # Create plots if requested
        if plot:
            self.plot_strategies(strategies, S_test, t)
        
        # Calculate statistics
        total_payoffs = [s['total_payoff'] for s in strategies]
        
        strategy_results = {
            'strategies': strategies,
            'mean_payoff': np.mean(total_payoffs),
            'std_payoff': np.std(total_payoffs),
            'exercise_rates': [
                np.mean([s['cumulative_exercise'][-1] for s in strategies])
            ],
            'test_paths': S_test,
            'time_grid': t
        }
        
        print(f"Strategy Results:")
        print(f"Mean Payoff: {strategy_results['mean_payoff']:.4f}")
        print(f"Payoff Std: {strategy_results['std_payoff']:.4f}")
        print(f"Mean Exercise Rate: {strategy_results['exercise_rates'][0]:.2f} / {self.contract.Q_max}")
        
        # Save step-by-step data to CSV if requested
        if save_step_data and evaluation_runs_dir and run_name:
            self.save_evaluation_run_csv(all_episodes_step_data, evaluation_runs_dir, run_name)
        
        return strategy_results
    
    def plot_strategies(self, strategies: List[Dict], S_paths: np.ndarray, t: np.ndarray):
        """Plot optimal exercise strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Spot price paths and exercise decisions
        ax = axes[0, 0]
        for i, strategy in enumerate(strategies[:5]):  # Show first 5 scenarios
            times = strategy['times']
            spots = strategy['spot_prices']
            actions = strategy['actions']
            
            ax.plot(times, spots, alpha=0.7, label=f'Scenario {i+1}')
            
            # Mark exercise times
            exercise_times = [times[j] for j, a in enumerate(actions) if a > 0]
            exercise_spots = [spots[j] for j, a in enumerate(actions) if a > 0]
            if exercise_times:
                ax.scatter(exercise_times, exercise_spots, color='red', s=50, alpha=0.8)
        
        ax.axhline(y=self.contract.strike, color='black', linestyle='--', label='Strike')
        ax.set_title('Spot Prices and Exercise Decisions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Spot Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative exercise
        ax = axes[0, 1]
        for i, strategy in enumerate(strategies[:5]):
            ax.plot(strategy['times'], strategy['cumulative_exercise'], 
                   alpha=0.7, label=f'Scenario {i+1}')
        
        ax.axhline(y=self.contract.Q_max, color='red', linestyle='--', label='Q_max')
        ax.set_title('Cumulative Exercise')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Volume')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Payoff distribution
        ax = axes[1, 0]
        total_payoffs = [s['total_payoff'] for s in strategies]
        ax.hist(total_payoffs, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(total_payoffs), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(total_payoffs):.3f}')
        ax.set_title('Payoff Distribution')
        ax.set_xlabel('Total Payoff')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Exercise timing
        ax = axes[1, 1]
        all_exercise_times = []
        for strategy in strategies:
            exercise_times = [t for t, a in zip(strategy['times'], strategy['actions']) if a > 0]
            all_exercise_times.extend(exercise_times)
        
        if all_exercise_times:
            ax.hist(all_exercise_times, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title('Exercise Timing Distribution')
            ax.set_xlabel('Time')
            ax.set_ylabel('Number of Exercises')
            ax.grid(True, alpha=0.3)
            ax.hist(all_exercise_times, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title('Exercise Timing Distribution')
            ax.set_xlabel('Time')
            ax.set_ylabel('Number of Exercises')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_evaluation_run_csv(self, all_episodes_step_data: List[List[Dict]], 
                              evaluation_runs_dir: str, run_name: str) -> None:
        """Save detailed step-by-step data for all episodes to CSV (matching RL agent format)"""
        import csv
        import os
        
        # Create evaluation_runs directory if it doesn't exist
        os.makedirs(evaluation_runs_dir, exist_ok=True)
        
        # Use run_name as filename (similar to how RL agent uses training episode number)
        filename = f"eval_run_{run_name}.csv"
        filepath = os.path.join(evaluation_runs_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode_idx', 'step', 'spot', 'q_remain', 'q_exerc', 
                           'time_left', 'action', 'q_actual', 'reward'])
            
            for episode_idx, step_data in enumerate(all_episodes_step_data):
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
        
        print(f"Evaluation run data saved to: {filepath}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for LSM pricer"""
    parser = argparse.ArgumentParser(description="Least Squares Monte Carlo Swing Option Pricer")

    # LSM Algorithm Parameters
    parser.add_argument("-n_paths", type=int, default=16384, 
                      help="Number of Monte Carlo paths for LSM simulation (default: 16384)")
    parser.add_argument("-poly_degree", type=int, default=3, 
                      help="Polynomial degree for continuation value regression (default: 3)")
    parser.add_argument("-seed", type=int, default=42, 
                      help="Random seed for reproducibility (default: 42)")
    parser.add_argument("-n_scenarios", type=int, default=10, 
                      help="Number of scenarios for optimal strategy simulation (default: 10)")
    parser.add_argument("--plot", type=int, default=1, choices=[0,1], 
                      help="Create plots for strategy analysis (default: 1)")
    parser.add_argument("--verbose", type=int, default=1, choices=[0,1], 
                      help="Print detailed output (default: 1)")
    parser.add_argument("-name", type=str, 
                      help="Name of the run (default: LSM_{timestamp})")

    # Swing Option Contract Parameters
    parser.add_argument("--q_min", type=float, default=0.0, 
                      help="Minimum exercise quantity per period (default: 0.0)")
    parser.add_argument("--q_max", type=float, default=1.0, 
                      help="Maximum exercise quantity per period (default: 1.0)")
    parser.add_argument("--Q_min", type=float, default=0.0, 
                      help="Global minimum total volume (default: 0.0)")
    parser.add_argument("--Q_max", type=float, default=10.0, 
                      help="Global maximum total volume (default: 10.0)")
    parser.add_argument("--strike", type=float, default=100.0, 
                      help="Strike price K (default: 100.0)")
    parser.add_argument("--maturity", type=float, default=1.0, 
                      help="Time to maturity in years (default: 1.0)")
    parser.add_argument("--n_rights", type=int, default=250, 
                      help="Number of decision dates (default: 250)")
    parser.add_argument("--risk_free_rate", type=float, default=0.05, 
                      help="Risk-free rate for discounting (default: 0.05)")
    parser.add_argument("--min_refraction_days", type=int, default=0, 
                      help="Minimum days between exercises (default: 0)")

    # HHK Stochastic Process Parameters
    parser.add_argument("--S0", type=float, default=100.0, 
                      help="Initial spot price (default: 100.0)")
    parser.add_argument("--alpha", type=float, default=7.0, 
                      help="Mean reversion speed (default: 7.0)")
    parser.add_argument("--sigma", type=float, default=1.4, 
                      help="Volatility of OU process (default: 1.4)")
    parser.add_argument("--beta", type=float, default=200.0, 
                      help="Jump decay rate (default: 200.0)")
    parser.add_argument("--lam", type=float, default=4.0, 
                      help="Jump intensity (jumps per year) (default: 4.0)")
    parser.add_argument("--mu_J", type=float, default=0.4, 
                      help="Mean jump size (default: 0.4)")

    return parser


def create_contract(args: argparse.Namespace) -> SwingContract:
    """Create swing option contract from command line arguments"""
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


def create_hhk_params(args: argparse.Namespace) -> Dict:
    """Create HHK parameters dictionary from command line arguments"""
    try:
        from .simulate_hhk_spot import default_seasonal_function
    except ImportError:
        from simulate_hhk_spot import default_seasonal_function
    
    return {
        'S0': args.S0,
        'alpha': args.alpha,
        'sigma': args.sigma,
        'beta': args.beta,
        'lam': args.lam,
        'mu_J': args.mu_J,
        'f': default_seasonal_function
    }


def generate_run_name(name: Optional[str]) -> str:
    """Generate run name with timestamp if not provided"""
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"LSM_{timestamp}"
    return name


def save_results(run_name: str, results: Dict, strategy_results: Dict, args: argparse.Namespace) -> str:
    """Save LSM results to files"""
    import json
    import os
    
    # Create logs directory structure (same as RL agent)
    logs_dir = "logs"
    run_logs_dir = os.path.join(logs_dir, run_name)
    evaluation_runs_dir = os.path.join(run_logs_dir, "evaluation_runs")
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(run_logs_dir, exist_ok=True)
    os.makedirs(evaluation_runs_dir, exist_ok=True)
    
    # Save parameters (same format as RL)
    params_file = os.path.join(run_logs_dir, f"{run_name}_parameters.json")
    with open(params_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Save results summary
    summary_file = f"logs/{run_name}_results.json"
    summary = {
        'option_price': results['option_price'],
        'option_std_error': results['option_std_error'],
        'confidence_interval_95': results['confidence_interval_95'],
        'paths_used': results['paths_used'],
        'final_spot_mean': results['final_spot_mean'],
        'final_spot_std': results['final_spot_std'],
        'intrinsic_value': results['intrinsic_value'],
        'mean_strategy_payoff': strategy_results['mean_payoff'],
        'strategy_payoff_std': strategy_results['std_payoff'],
        'mean_exercise_rate': strategy_results['exercise_rates'][0],
        'exercise_utilization_pct': 100 * strategy_results['exercise_rates'][0] / args.Q_max
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"  Parameters: {params_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Evaluation runs directory: {evaluation_runs_dir}")
    
    return evaluation_runs_dir


if __name__ == "__main__":
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Generate run name
    run_name = generate_run_name(args.name)
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("=== LSM Swing Option Pricer ===")
    print(f"Run name: {run_name}")
    print(f"Using {args.n_paths} Monte Carlo paths")
    print(f"Polynomial degree: {args.poly_degree}")
    print(f"Random seed: {args.seed}")
    
    # Create contract and HHK parameters from arguments
    contract = create_contract(args)
    hhk_params = create_hhk_params(args)
    
    if args.verbose:
        print(f"\nContract Parameters:")
        print(f"  Strike: {contract.strike}")
        print(f"  Maturity: {contract.maturity} years")
        print(f"  Rights: {contract.n_rights}")
        print(f"  q_min/q_max: {contract.q_min}/{contract.q_max}")
        print(f"  Q_min/Q_max: {contract.Q_min}/{contract.Q_max}")
        print(f"  Risk-free rate: {contract.r}")
        
        print(f"\nHHK Process Parameters:")
        print(f"  S0: {hhk_params['S0']}")
        print(f"  Alpha: {hhk_params['alpha']}")
        print(f"  Sigma: {hhk_params['sigma']}")
        print(f"  Beta: {hhk_params['beta']}")
        print(f"  Lambda: {hhk_params['lam']}")
        print(f"  Mu_J: {hhk_params['mu_J']}")
    
    # Initialize LSM pricer
    start_time = time.time()
    pricer = LSMSwingPricer(
        contract=contract,
        hhk_params=hhk_params,
        n_paths=args.n_paths,
        poly_degree=args.poly_degree,
        seed=args.seed
    )
    
    # Price the option
    print(f"\n{'='*60}")
    print("PRICING SWING OPTION")
    print(f"{'='*60}")
    
    results = pricer.price_option(verbose=bool(args.verbose))
    
    # Simulate optimal strategies
    print(f"\n{'='*60}")
    print("SIMULATING OPTIMAL STRATEGIES")
    print(f"{'='*60}")
    
    strategy_results = pricer.simulate_optimal_strategy(
        results=results,
        n_scenarios=args.n_scenarios,
        plot=bool(args.plot)
    )
    
    # Calculate total runtime
    total_time = time.time() - start_time
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Option Price: {results['option_price']:.4f} ± {results['option_std_error']:.4f}")
    print(f"95% CI: [{results['confidence_interval_95'][0]:.4f}, {results['confidence_interval_95'][1]:.4f}]")
    print(f"Mean Strategy Payoff: {strategy_results['mean_payoff']:.4f}")
    print(f"Strategy Payoff Std: {strategy_results['std_payoff']:.4f}")
    print(f"Contract Utilization: {strategy_results['exercise_rates'][0]:.1f}/{args.Q_max} " +
          f"({100*strategy_results['exercise_rates'][0]/args.Q_max:.1f}%)")
    print(f"Total Runtime: {total_time:.2f} seconds")
    
    # Save results to files and get evaluation_runs_dir
    evaluation_runs_dir = save_results(run_name, results, strategy_results, args)
    
    # Now simulate again with CSV saving enabled for detailed step data
    print(f"\nSaving detailed step-by-step evaluation data...")
    pricer.simulate_optimal_strategy(
        results=results,
        n_scenarios=args.n_scenarios,
        plot=False,  # Don't create plots again
        save_step_data=True,
        evaluation_runs_dir=evaluation_runs_dir,
        run_name=run_name
    )
