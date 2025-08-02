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


class LSMSwingPricer:
    """Least Squares Monte Carlo swing option pricer"""
    pricer = LSMSwingPricer(
            contract=swing_contract,
            dataset=eval_ds,
            n_paths=n_paths_eval, 
            poly_degree=3, seed=seed+1
        )
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
    
    def save_results_csv(self, run_name: str, results: Dict) -> None:
        """Save LSM pricing results to CSV file"""
        import csv
        import os
        
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save results to CSV
        filename = f"LSM_{run_name}_results.csv"
        filepath = os.path.join(logs_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['option_price', round(results['option_price'], 6)])
            writer.writerow(['option_std_error', round(results['option_std_error'], 6)])
            writer.writerow(['ci_95_lower', round(results['confidence_interval_95'][0], 6)])
            writer.writerow(['ci_95_upper', round(results['confidence_interval_95'][1], 6)])
            writer.writerow(['paths_used', results['paths_used']])
            writer.writerow(['final_spot_mean', round(results['final_spot_mean'], 4)])
            writer.writerow(['final_spot_std', round(results['final_spot_std'], 4)])
            writer.writerow(['intrinsic_value', round(results['intrinsic_value'], 6)])
        
        print(f"Results saved to: {filepath}")


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


def price_with_external_data(contract: SwingContract, eval_t, eval_S, eval_X, eval_Y, 
                           run_name: str, poly_degree: int = 3) -> Dict:
    """
    Price swing option using externally provided paths (called from run.py)
    
    Args:
        contract: Swing option contract specification
        eval_t: Pre-generated time grid
        eval_S: Pre-generated spot price paths
        eval_X: Pre-generated X process paths  
        eval_Y: Pre-generated Y process paths
        run_name: Name for the run
        poly_degree: Polynomial degree for regression
        
    Returns:
        Dictionary with pricing results
    """
    print("=== LSM Swing Option Pricer ===")
    print(f"Using pre-generated paths: {eval_S.shape[0]} scenarios")
    
    # Create dummy hhk_params (not used with pre-generated paths)
    hhk_params = {}
    
    # Initialize LSM pricer
    pricer = LSMSwingPricer(
        contract=contract,
        hhk_params=hhk_params,
        n_paths=eval_S.shape[0],
        poly_degree=poly_degree,
        seed=42
    )
    
    # Price the option using pre-generated paths
    results = pricer.price_option_with_pregenerated_paths(
        eval_t, eval_S, eval_X, eval_Y, verbose=False
    )
    
    # Print results
    print(f"Option Price: {results['option_price']:.6f}")
    print(f"95% CI: [{results['confidence_interval_95'][0]:.6f}, {results['confidence_interval_95'][1]:.6f}]")
    
    # Save to CSV
    pricer.save_results_csv(run_name, results)
    
    return results


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
    
    print(f"\nContract: Strike={contract.strike}, Maturity={contract.maturity}, Rights={contract.n_rights}")
    print(f"Volume limits: Q_min={contract.Q_min}, Q_max={contract.Q_max}")
    
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
    print(f"\nPricing swing option...")
    results = pricer.price_option(verbose=False)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    
    # Print results
    print(f"\nOption Price: {results['option_price']:.6f}")
    print(f"95% CI: [{results['confidence_interval_95'][0]:.6f}, {results['confidence_interval_95'][1]:.6f}]")
    print(f"Runtime: {total_time:.2f} seconds")
    
    # Save results to CSV
    pricer.save_results_csv(run_name, results)
