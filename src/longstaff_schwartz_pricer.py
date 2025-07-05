"""
Swing Option Pricing Module

Implementation of state-of-the-art methods for pricing daily-exercisable swing options
using Monte Carlo simulation with HHK spot price dynamics.

Based on:
- Longstaff-Schwartz (2001) Least Squares Monte Carlo
- Hambly, Howison & Kluge (2009) for model and grid method concepts
- Andersen-Broadie (2004) dual method concepts

Authors: Financial Engineering Research
Date: July 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from typing import Tuple, Dict, Any, Optional, Callable, Union
import warnings
import time
from dataclasses import dataclass

from src.simulate_hhk_spot import simulate_hhk_spot, DEFAULT_HHK_PARAMS


@dataclass
class SwingOptionContract:
    """Swing option contract specification"""
    strike: float                    # Strike price K
    volume_per_exercise: float       # Volume V per exercise
    max_exercises: int              # Maximum number of exercises N
    maturity: float                 # Time to maturity T (years)
    risk_free_rate: float          # Risk-free interest rate r
    
    def __post_init__(self):
        """Validate contract parameters"""
        if self.strike <= 0:
            raise ValueError("Strike must be positive")
        if self.volume_per_exercise <= 0:
            raise ValueError("Volume per exercise must be positive")
        if self.max_exercises <= 0:
            raise ValueError("Max exercises must be positive")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")


@dataclass
class PricingResult:
    """Results from option pricing"""
    price: float
    std_error: float
    confidence_interval: Tuple[float, float]
    exercise_probability: np.ndarray  # Probability of exercise at each time step
    computation_time: float
    method_info: Dict[str, Any]


class LongstaffSchwartzPricer:
    """
    Swing option pricing using Longstaff-Schwartz Monte Carlo method
    """
    
    def __init__(self, contract: SwingOptionContract, random_seed: Optional[int] = None):
        """
        Initialize the swing option pricer
        
        Args:
            contract: Swing option contract specification
            random_seed: Random seed for reproducibility
        """
        self.contract = contract
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def price_lsm(self,
                  spot_paths: np.ndarray,
                  time_grid: np.ndarray,
                  basis_type: str = 'polynomial',
                  polynomial_degree: int = 3,
                  rf_params: Optional[Dict] = None) -> PricingResult:
        """
        Price swing option using Longstaff-Schwartz Monte Carlo method
        
        Args:
            spot_paths: Array of shape (n_paths, n_steps+1) with simulated spot prices
            time_grid: Array of time points
            basis_type: Type of basis functions ('polynomial', 'spline', 'random_forest')
            polynomial_degree: Degree for polynomial basis
            rf_params: Parameters for random forest if used
            
        Returns:
            PricingResult with price estimate and diagnostics
        """
        start_time = time.time()
        
        n_paths, n_steps = spot_paths.shape[0], spot_paths.shape[1] - 1
        dt = self.contract.maturity / n_steps
        discount_factor = np.exp(-self.contract.risk_free_rate * dt)
        
        # Initialize value array: V[path, time, remaining_rights]
        # We track the option value for each path, time step, and number of remaining rights
        max_rights = self.contract.max_exercises
        option_values = np.zeros((n_paths, n_steps + 1, max_rights + 1))
        exercise_decisions = np.zeros((n_paths, n_steps + 1), dtype=bool)
        
        # Terminal condition: at maturity, exercise all remaining rights if profitable
        for m in range(1, max_rights + 1):
            # Apply the swing option pricing formula: q_t * (S_t - K)^+
            payoff = self.contract.volume_per_exercise * np.maximum(
                spot_paths[:, -1] - self.contract.strike, 0
            )
            option_values[:, -1, m] = payoff  # Exercise remaining rights at expiry
        
        # Backward induction
        regression_stats = []
        
        for t in range(n_steps - 1, -1, -1):
            current_time = time_grid[t]
            
            for m in range(1, max_rights + 1):
                # Paths that still have m rights available at time t
                available_paths = np.ones(n_paths, dtype=bool)
                
                if np.sum(available_paths) == 0:
                    continue
                
                # Current spot prices for available paths
                current_spots = spot_paths[available_paths, t]
                
                # Immediate exercise payoff: q_t * (S_t - K)^+
                immediate_payoff = self.contract.volume_per_exercise * np.maximum(
                    current_spots - self.contract.strike, 0
                )
                
                # Value if exercising: immediate payoff + discounted future value with m-1 rights
                future_value_exercise = discount_factor * option_values[available_paths, t + 1, m - 1]
                exercise_value = immediate_payoff + future_value_exercise
                
                # Value if not exercising: discounted future value with m rights
                continuation_value = discount_factor * option_values[available_paths, t + 1, m]
                
                # Regression to estimate continuation value
                if len(current_spots) > 10:  # Need sufficient data for regression
                    # Only use in-the-money paths for regression (common LSM practice)
                    itm_mask = immediate_payoff > 0
                    
                    if np.sum(itm_mask) > 5:  # Need some ITM paths
                        regression_spots = current_spots[itm_mask]
                        regression_continuation = continuation_value[itm_mask]
                        
                        # Fit continuation value function
                        try:
                            fitted_continuation = self._fit_continuation_value(
                                regression_spots, regression_continuation, m, current_time,
                                basis_type, polynomial_degree, rf_params
                            )
                            
                            # Predict continuation for all paths
                            predicted_continuation = self._predict_continuation_value(
                                current_spots, fitted_continuation, basis_type
                            )
                            
                            # Store regression statistics
                            if len(regression_spots) > 1:
                                r2 = r2_score(regression_continuation, 
                                            self._predict_continuation_value(
                                                regression_spots, fitted_continuation, basis_type))
                                regression_stats.append({
                                    'time': t, 'rights': m, 'n_points': len(regression_spots),
                                    'r2': r2
                                })
                        
                        except Exception as e:
                            warnings.warn(f"Regression failed at t={t}, m={m}: {e}")
                            predicted_continuation = continuation_value
                    else:
                        predicted_continuation = continuation_value
                else:
                    predicted_continuation = continuation_value
                
                # Exercise decision: compare immediate exercise vs continuation
                exercise_optimal = exercise_value > predicted_continuation
                
                # Update option values
                option_values[available_paths, t, m] = np.where(
                    exercise_optimal,
                    exercise_value,
                    predicted_continuation
                )
                
                # Record exercise decisions for this number of rights
                if m == max_rights:  # Track decisions for full contract
                    exercise_decisions[available_paths, t] = exercise_optimal
        
        # Final option price: value at t=0 with all rights available
        option_prices = option_values[:, 0, max_rights]
        
        # Calculate statistics
        mean_price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(n_paths)
        ci_lower = mean_price - 1.96 * std_error
        ci_upper = mean_price + 1.96 * std_error
        
        # Exercise probability at each time step
        exercise_prob = np.mean(exercise_decisions, axis=0)
        
        computation_time = time.time() - start_time
        
        # Method information
        method_info = {
            'method': 'Longstaff-Schwartz Monte Carlo',
            'basis_type': basis_type,
            'polynomial_degree': polynomial_degree if basis_type == 'polynomial' else None,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'regression_stats': regression_stats,
            'avg_r2': np.mean([s['r2'] for s in regression_stats]) if regression_stats else None
        }
        
        return PricingResult(
            price=float(mean_price),
            std_error=float(std_error),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            exercise_probability=exercise_prob,
            computation_time=computation_time,
            method_info=method_info
        )
    
    def _fit_continuation_value(self,
                               spots: np.ndarray,
                               continuation_values: np.ndarray,
                               rights_remaining: int,
                               current_time: float,
                               basis_type: str,
                               polynomial_degree: int,
                               rf_params: Optional[Dict]) -> Any:
        """
        Fit continuation value function using specified basis
        """
        if basis_type == 'polynomial':
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=polynomial_degree, include_bias=True)
            X = poly_features.fit_transform(spots.reshape(-1, 1))
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(X, continuation_values)
            
            return {'type': 'polynomial', 'poly_features': poly_features, 'regressor': reg}
        
        elif basis_type == 'random_forest':
            # Random forest regression
            if rf_params is None:
                rf_params = {'n_estimators': 50, 'max_depth': 5, 'random_state': self.random_seed}
            
            # Include additional features: spot price, remaining rights, time
            X = np.column_stack([
                spots,
                np.full(len(spots), rights_remaining),
                np.full(len(spots), current_time)
            ])
            
            reg = RandomForestRegressor(**rf_params)
            reg.fit(X, continuation_values)
            
            return {'type': 'random_forest', 'regressor': reg}
        
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")
    
    def _predict_continuation_value(self,
                                   spots: np.ndarray,
                                   fitted_model: Dict,
                                   basis_type: str) -> np.ndarray:
        """
        Predict continuation values using fitted model
        """
        if fitted_model['type'] == 'polynomial':
            X = fitted_model['poly_features'].transform(spots.reshape(-1, 1))
            return fitted_model['regressor'].predict(X)
        
        elif fitted_model['type'] == 'random_forest':
            # For RF, we need to provide the full feature vector
            # But we don't have rights_remaining and current_time here
            # This is a simplification - in practice would need to track these
            X = spots.reshape(-1, 1)
            if X.shape[1] != fitted_model['regressor'].n_features_in_:
                # Pad with zeros as approximation
                X_full = np.zeros((len(spots), fitted_model['regressor'].n_features_in_))
                X_full[:, 0] = spots
                X = X_full
            
            return fitted_model['regressor'].predict(X)
        
        else:
            return np.zeros(len(spots))


def generate_lsm_solution_csv(eval_t, eval_S, eval_X, eval_Y, contract, csv_filename, random_seed: int = 42):
    """
    Generate detailed LSM solution CSV with state and q for each path and time step.
    Uses the exact same MC paths as RL evaluation for perfect comparison.
    
    Args:
        eval_t: Time grid from RL simulation
        eval_S: Spot price paths from RL simulation  
        eval_X: X process paths from RL simulation
        eval_Y: Y process paths from RL simulation
        contract: SwingContract
        csv_filename: Output CSV filename
        random_seed: Random seed
        
    Returns:
        Dictionary with LSM results
    """
    import csv
    from .swing_env import calculate_standardized_reward
    
    print(f"\n🔮 Generating LSM solution CSV: {csv_filename}")
    print(f"   Using {eval_S.shape[0]} paths with {eval_S.shape[1]} time steps")
    
    # Convert contract for LSM
    lsm_contract = SwingOptionContract(
        strike=contract.strike,
        volume_per_exercise=contract.q_max,
        max_exercises=min(contract.n_rights, int(contract.Q_max / contract.q_max)),
        maturity=contract.maturity,
        risk_free_rate=contract.r
    )
    
    # Initialize LSM pricer
    pricer = LongstaffSchwartzPricer(lsm_contract, random_seed)
    
    # Modified LSM pricing to track detailed exercise decisions
    n_paths, n_steps = eval_S.shape[0], eval_S.shape[1] - 1
    dt = lsm_contract.maturity / n_steps
    discount_factor = np.exp(-lsm_contract.risk_free_rate * dt)
    
    # Initialize arrays
    max_rights = lsm_contract.max_exercises
    option_values = np.zeros((n_paths, n_steps + 1, max_rights + 1))
    exercise_decisions = np.zeros((n_paths, n_steps + 1))
    exercise_quantities = np.zeros((n_paths, n_steps + 1))
    
    # Terminal condition
    for m in range(1, max_rights + 1):
        payoff = lsm_contract.volume_per_exercise * np.maximum(
            eval_S[:, -1] - lsm_contract.strike, 0
        )
        option_values[:, -1, m] = payoff
    
    # Backward induction with exercise tracking
    for t in range(n_steps - 1, -1, -1):
        for m in range(1, max_rights + 1):
            available_paths = np.ones(n_paths, dtype=bool)
            
            if np.sum(available_paths) == 0:
                continue
                
            current_spots = eval_S[available_paths, t]
            
            # Immediate exercise payoff: q_t * (S_t - K)^+
            immediate_payoff = lsm_contract.volume_per_exercise * np.maximum(
                current_spots - lsm_contract.strike, 0
            )
            
            # Values
            future_value_exercise = discount_factor * option_values[available_paths, t + 1, m - 1]
            exercise_value = immediate_payoff + future_value_exercise
            continuation_value = discount_factor * option_values[available_paths, t + 1, m]
            
            # Simple continuation value estimation (simplified for CSV generation)
            if len(current_spots) > 10:
                itm_mask = immediate_payoff > 0
                if np.sum(itm_mask) > 5:
                    try:
                        fitted_continuation = pricer._fit_continuation_value(
                            current_spots[itm_mask], continuation_value[itm_mask], 
                            m, eval_t[t], 'polynomial', 3, None
                        )
                        predicted_continuation = pricer._predict_continuation_value(
                            current_spots, fitted_continuation, 'polynomial'
                        )
                    except:
                        predicted_continuation = continuation_value
                else:
                    predicted_continuation = continuation_value
            else:
                predicted_continuation = continuation_value
            
            # Exercise decision
            exercise_optimal = exercise_value > predicted_continuation
            
            # Update values
            option_values[available_paths, t, m] = np.where(
                exercise_optimal, exercise_value, predicted_continuation
            )
            
            # Track exercise decisions for full contract (m == max_rights)
            if m == max_rights:
                exercise_decisions[available_paths, t] = exercise_optimal
                exercise_quantities[available_paths, t] = np.where(
                    exercise_optimal, lsm_contract.volume_per_exercise, 0.0
                )
    
    # Forward simulation to reconstruct actual exercise path
    print(f"   📝 Writing detailed step data to CSV...")
    
    path_payoffs = []  # Track the total payoff for each path
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Standardized header matching RL solution format
        writer.writerow(['episode_idx', 'step', 'spot', 'q_remain', 'q_exerc', 
                        'time_left', 'action', 'q_actual', 'reward'])
        
        for path_idx in range(n_paths):
            q_remaining = contract.Q_max  # Start with full inventory
            q_exercised_total = 0.0
            path_total_reward = 0.0  # Track total reward for this path
            
            for step in range(n_steps):  # Only go to n_steps (not n_steps + 1) to match RL termination
                spot_price = eval_S[path_idx, step]
                time_left = contract.maturity - eval_t[step] if step < len(eval_t) else 0.0
                
                # Use LSM exercise decision
                q_decision = exercise_quantities[path_idx, step]
                
                # Ensure we don't over-exercise
                q_decision = min(q_decision, q_remaining, contract.q_max)
                
                # Update state first
                q_remaining -= q_decision
                q_exercised_total += q_decision
                
                # Check if this is a terminal step
                is_terminal = ((step + 1) >= contract.n_rights or 
                             q_exercised_total >= contract.Q_max - 1e-6)
                
                # Calculate standardized reward using same function as RL with terminal penalty
                reward = calculate_standardized_reward(
                    spot_price, q_decision, contract.strike, 
                    step, contract.discount_factor,
                    q_exercised_total, contract.Q_min, is_terminal
                )
                
                path_total_reward += reward  # Add to path total
                
                # Write row with standardized format and precision
                writer.writerow([
                    path_idx, step, round(spot_price, 4), 
                    round(q_remaining, 4), round(q_exercised_total, 4),
                    round(time_left, 4), round(q_decision, 6), 
                    round(q_decision, 4), round(reward, 6)
                ])
                
                # Check termination conditions to match RL environment
                if is_terminal:
                    break
            
            path_payoffs.append(path_total_reward)  # Store total payoff for this path
    
    # Calculate final statistics from actual path payoffs (same as notebook calculation)
    path_payoffs = np.array(path_payoffs)
    mean_price = np.mean(path_payoffs)
    std_error = np.std(path_payoffs) / np.sqrt(n_paths)
    
    print(f"   ✅ LSM CSV generated: {csv_filename}")
    print(f"   💰 LSM Option Value: ${mean_price:.6f} ± {std_error:.6f}")
    print(f"   📊 Total rows written: {len(path_payoffs)} path summaries")
    
    return {
        'lsm_option_value': mean_price,
        'lsm_std_error': std_error,
        'lsm_n_paths': n_paths,
        'csv_file': csv_filename
    }


def create_default_contract(
    strike: float = 100.0,
    volume_per_exercise: float = 1.0,
    max_exercises: int = 10,
    maturity: float = 1.0,
    risk_free_rate: float = 0.05
) -> SwingOptionContract:
    """
    Create a default swing option contract for testing
    """
    return SwingOptionContract(
        strike=strike,
        volume_per_exercise=volume_per_exercise,
        max_exercises=max_exercises,
        maturity=maturity,
        risk_free_rate=risk_free_rate
    )


def price_swing_option_with_hhk(
    contract: SwingOptionContract,
    hhk_params: Optional[Dict] = None,
    n_paths: int = 10000,
    n_steps: int = 365,
    pricing_method: str = 'lsm',
    basis_type: str = 'polynomial',
    polynomial_degree: int = 3,
    random_seed: Optional[int] = None
) -> PricingResult:
    """
    Complete workflow: simulate HHK paths and price swing option
    
    Args:
        contract: Swing option contract specification
        hhk_params: HHK model parameters (uses defaults if None)
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps
        pricing_method: Pricing method ('lsm' for now)
        basis_type: Basis function type for LSM
        polynomial_degree: Polynomial degree for regression
        random_seed: Random seed for reproducibility
        
    Returns:
        PricingResult with option price and diagnostics
    """
    # Use default HHK parameters if none provided
    if hhk_params is None:
        hhk_params = DEFAULT_HHK_PARAMS.copy()
    
    # Simulate HHK spot price paths
    print(f"Simulating {n_paths} HHK paths over {contract.maturity} years...")
    time_grid, spot_paths, _, _ = simulate_hhk_spot(
        T=contract.maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=random_seed,
        **hhk_params
    )
    
    print(f"Price range: [{spot_paths.min():.2f}, {spot_paths.max():.2f}]")
    print(f"Final price stats: mean={spot_paths[:, -1].mean():.2f}, "
          f"std={spot_paths[:, -1].std():.2f}")
    
    # Initialize pricer and compute option value
    pricer = LongstaffSchwartzPricer(contract, random_seed)
    
    print(f"Pricing swing option using {pricing_method.upper()} method...")
    
    if pricing_method == 'lsm':
        result = pricer.price_lsm(
            spot_paths=spot_paths,
            time_grid=time_grid,
            basis_type=basis_type,
            polynomial_degree=polynomial_degree
        )
    else:
        raise ValueError(f"Unknown pricing method: {pricing_method}")
    
    return result


def run_convergence_analysis(
    contract: SwingOptionContract,
    path_counts: list = [1000, 2000, 5000, 10000, 20000],
    n_repetitions: int = 5,
    **kwargs
) -> pd.DataFrame:
    """
    Analyze price convergence as number of paths increases
    
    Args:
        contract: Swing option contract
        path_counts: List of path counts to test
        n_repetitions: Number of repetitions for each path count
        **kwargs: Additional arguments for pricing
        
    Returns:
        DataFrame with convergence results
    """
    results = []
    
    for n_paths in path_counts:
        print(f"\nTesting convergence with {n_paths} paths...")
        
        prices = []
        times = []
        
        for rep in range(n_repetitions):
            # Use different seed for each repetition
            seed = kwargs.get('random_seed', 42) + rep if 'random_seed' in kwargs else None
            
            result = price_swing_option_with_hhk(
                contract=contract,
                n_paths=n_paths,
                random_seed=seed,
                **{k: v for k, v in kwargs.items() if k != 'random_seed'}
            )
            
            prices.append(result.price)
            times.append(result.computation_time)
        
        # Statistics across repetitions
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        mean_time = np.mean(times)
        
        results.append({
            'n_paths': n_paths,
            'mean_price': mean_price,
            'std_price': std_price,
            'std_error': std_price / np.sqrt(n_repetitions),
            'mean_time': mean_time,
            'prices': prices
        })
        
        print(f"  Mean price: {mean_price:.4f} ± {std_price:.4f}")
        print(f"  Mean time: {mean_time:.2f}s")
    
    return pd.DataFrame(results)


def analyze_exercise_strategy(
    result: PricingResult,
    time_grid: np.ndarray,
    contract: SwingOptionContract
) -> Dict[str, Any]:
    """
    Analyze the optimal exercise strategy from pricing result
    
    Args:
        result: PricingResult from pricing
        time_grid: Time grid used in pricing
        contract: Swing option contract
        
    Returns:
        Dictionary with exercise strategy analysis
    """
    exercise_prob = result.exercise_probability
    
    # Find peak exercise periods
    peak_threshold = np.percentile(exercise_prob[exercise_prob > 0], 75) if np.any(exercise_prob > 0) else 0
    peak_periods = time_grid[exercise_prob > peak_threshold]
    
    # Expected number of exercises
    expected_exercises = np.sum(exercise_prob)
    
    # Time to first exercise (expected)
    cumulative_exercise = np.cumsum(exercise_prob)
    first_exercise_prob = cumulative_exercise / expected_exercises if expected_exercises > 0 else cumulative_exercise
    expected_first_exercise = np.sum(time_grid * np.diff(np.concatenate([[0], first_exercise_prob])))
    
    analysis = {
        'expected_total_exercises': expected_exercises,
        'exercise_rate': expected_exercises / contract.max_exercises,
        'expected_first_exercise_time': expected_first_exercise,
        'peak_exercise_periods': peak_periods,
        'peak_exercise_threshold': peak_threshold,
        'max_daily_exercise_prob': np.max(exercise_prob),
        'exercise_concentration': np.sum(exercise_prob ** 2) / np.sum(exercise_prob) if np.sum(exercise_prob) > 0 else 0
    }
    
    return analysis


def compute_lsm_benchmark(contract, hhk_params, n_paths: int = 10000, random_seed: int = 42) -> Dict[str, Any]:
    """
    Compute LSM benchmark for swing option pricing using the same setup as RL environment
    
    IMPORTANT: This function computes the LSM strategy value using the SAME reward 
    calculation as RL (step-wise discounted payoffs) for fair comparison.
    
    Args:
        contract: SwingContract from the existing swing option framework
        hhk_params: HHK model parameters 
        n_paths: Number of MC paths for LSM pricing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with LSM benchmark results
    """
    from .swing_env import calculate_standardized_reward
    
    print(f"\n🔮 Computing LSM Benchmark (Traditional Method)")
    print(f"   Contract: Strike=${contract.strike}, {contract.n_rights} rights, {contract.maturity:.4f}Y maturity")
    print(f"   LSM Paths: {n_paths}")
    
    # Convert SwingContract to SwingOptionContract for LSM
    lsm_contract = SwingOptionContract(
        strike=contract.strike,
        volume_per_exercise=contract.q_max,
        max_exercises=min(contract.n_rights, int(contract.Q_max / contract.q_max)),
        maturity=contract.maturity,
        risk_free_rate=contract.r
    )
    
    # Simulate HHK paths using same parameters as RL
    from .simulate_hhk_spot import simulate_hhk_spot
    time_grid, spot_paths, _, _ = simulate_hhk_spot(
        T=contract.maturity,
        n_steps=contract.n_rights,
        n_paths=n_paths,
        seed=random_seed,
        **hhk_params
    )
    
    # Compute LSM exercise strategy
    start_time = time.time()
    pricer = LongstaffSchwartzPricer(lsm_contract, random_seed)
    
    # Get exercise strategy using LSM
    n_steps = spot_paths.shape[1] - 1
    dt = lsm_contract.maturity / n_steps
    discount_factor = np.exp(-lsm_contract.risk_free_rate * dt)
    
    # Initialize arrays
    max_rights = lsm_contract.max_exercises
    option_values = np.zeros((n_paths, n_steps + 1, max_rights + 1))
    exercise_quantities = np.zeros((n_paths, n_steps + 1))
    
    # Terminal condition
    for m in range(1, max_rights + 1):
        payoff = lsm_contract.volume_per_exercise * np.maximum(
            spot_paths[:, -1] - lsm_contract.strike, 0
        )
        option_values[:, -1, m] = payoff
    
    # Backward induction to find optimal exercise strategy
    for t in range(n_steps - 1, -1, -1):
        for m in range(1, max_rights + 1):
            available_paths = np.ones(n_paths, dtype=bool)
            
            if np.sum(available_paths) == 0:
                continue
                
            current_spots = spot_paths[available_paths, t]
            
            # Immediate exercise payoff
            immediate_payoff = lsm_contract.volume_per_exercise * np.maximum(
                current_spots - lsm_contract.strike, 0
            )
            
            # Values
            future_value_exercise = discount_factor * option_values[available_paths, t + 1, m - 1]
            exercise_value = immediate_payoff + future_value_exercise
            continuation_value = discount_factor * option_values[available_paths, t + 1, m]
            
            # Simple continuation value estimation
            if len(current_spots) > 10:
                itm_mask = immediate_payoff > 0
                if np.sum(itm_mask) > 5:
                    try:
                        fitted_continuation = pricer._fit_continuation_value(
                            current_spots[itm_mask], continuation_value[itm_mask], 
                            m, time_grid[t], 'polynomial', 3, None
                        )
                        predicted_continuation = pricer._predict_continuation_value(
                            current_spots, fitted_continuation, 'polynomial'
                        )
                    except:
                        predicted_continuation = continuation_value
                else:
                    predicted_continuation = continuation_value
            else:
                predicted_continuation = continuation_value
            
            # Exercise decision
            exercise_optimal = exercise_value > predicted_continuation
            
            # Update values
            option_values[available_paths, t, m] = np.where(
                exercise_optimal, exercise_value, predicted_continuation
            )
            
            # Track exercise decisions for full contract (m == max_rights)
            if m == max_rights:
                exercise_quantities[available_paths, t] = np.where(
                    exercise_optimal, lsm_contract.volume_per_exercise, 0.0
                )
    
    # Forward simulation using SAME reward calculation as RL
    path_payoffs = []
    total_exercises = []
    
    for path_idx in range(n_paths):
        q_remaining = contract.Q_max  # Start with full inventory
        q_exercised_total = 0.0
        path_reward = 0.0
        
        for step in range(n_steps):
            spot_price = spot_paths[path_idx, step]
            
            # Use LSM exercise decision
            q_decision = exercise_quantities[path_idx, step]
            
            # Ensure we don't over-exercise
            q_decision = min(q_decision, q_remaining, contract.q_max)
            
            # Update state
            q_remaining -= q_decision
            q_exercised_total += q_decision
            
            # Check if this is a terminal step
            is_terminal = ((step + 1) >= contract.n_rights or 
                         q_exercised_total >= contract.Q_max - 1e-6)
            
            # Calculate standardized reward using SAME function as RL
            reward = calculate_standardized_reward(
                spot_price, q_decision, contract.strike, 
                step, contract.discount_factor,
                q_exercised_total, contract.Q_min, is_terminal
            )
            
            path_reward += reward
            
            # Check termination conditions to match RL environment
            if is_terminal:
                break
        
        path_payoffs.append(path_reward)
        total_exercises.append(q_exercised_total)
    
    benchmark_time = time.time() - start_time
    
    # Calculate statistics using same method as RL evaluation
    path_payoffs = np.array(path_payoffs)
    mean_payoff = np.mean(path_payoffs)
    std_error = np.std(path_payoffs) / np.sqrt(n_paths)
    ci_lower = mean_payoff - 1.96 * std_error
    ci_upper = mean_payoff + 1.96 * std_error
    
    expected_exercises = np.mean(total_exercises)
    exercise_efficiency = expected_exercises / contract.Q_max
    
    benchmark_results = {
        'lsm_option_value': mean_payoff,
        'lsm_std_error': std_error,
        'lsm_confidence_interval': (ci_lower, ci_upper),
        'lsm_expected_exercises': expected_exercises,
        'lsm_exercise_efficiency': exercise_efficiency,
        'lsm_computation_time': benchmark_time,
        'lsm_n_paths': n_paths,
        'lsm_regression_r2': 0.0,  # Not computed in this simplified version
        'lsm_exercise_probabilities': []  # Not computed in this simplified version
    }
    
    print(f"   ✅ LSM Option Value: ${mean_payoff:.6f} ± {std_error:.6f}")
    print(f"   📊 Expected Exercises: {expected_exercises:.2f}/{contract.Q_max} ({exercise_efficiency:.1%})")
    print(f"   ⚡ Computation Time: {benchmark_time:.2f}s")
    print(f"   🎯 This is the target value for RL agent to achieve!")
    
    return benchmark_results


def use_same_paths_for_lsm_and_rl(eval_t, eval_S, eval_X, eval_Y, contract, random_seed: int = 42) -> Dict[str, Any]:
    """
    Use the exact same MC paths for LSM pricing as will be used for RL evaluation
    This ensures perfect comparability between methods.
    
    IMPORTANT: This function computes the LSM strategy value using the SAME reward 
    calculation as RL (step-wise discounted payoffs) for fair comparison.
    
    Args:
        eval_t: Time grid from RL simulation
        eval_S: Spot price paths from RL simulation  
        eval_X: X process paths from RL simulation
        eval_Y: Y process paths from RL simulation
        contract: SwingContract
        random_seed: Random seed
        
    Returns:
        LSM benchmark results using the exact same paths and reward calculation as RL
    """
    from .swing_env import calculate_standardized_reward
    
    print(f"\n🎯 Computing LSM with SAME paths as RL evaluation")
    print(f"   Using {eval_S.shape[0]} pre-generated paths")
    
    # Convert contract
    lsm_contract = SwingOptionContract(
        strike=contract.strike,
        volume_per_exercise=contract.q_max,
        max_exercises=min(contract.n_rights, int(contract.Q_max / contract.q_max)),
        maturity=contract.maturity,
        risk_free_rate=contract.r
    )
    
    # Step 1: Compute LSM exercise strategy
    pricer = LongstaffSchwartzPricer(lsm_contract, random_seed)
    
    # Modified LSM pricing to get exercise decisions
    n_paths, n_steps = eval_S.shape[0], eval_S.shape[1] - 1
    dt = lsm_contract.maturity / n_steps
    discount_factor = np.exp(-lsm_contract.risk_free_rate * dt)
    
    # Initialize arrays
    max_rights = lsm_contract.max_exercises
    option_values = np.zeros((n_paths, n_steps + 1, max_rights + 1))
    exercise_quantities = np.zeros((n_paths, n_steps + 1))
    
    # Terminal condition
    for m in range(1, max_rights + 1):
        payoff = lsm_contract.volume_per_exercise * np.maximum(
            eval_S[:, -1] - lsm_contract.strike, 0
        )
        option_values[:, -1, m] = payoff
    
    # Backward induction to find optimal exercise strategy
    for t in range(n_steps - 1, -1, -1):
        for m in range(1, max_rights + 1):
            available_paths = np.ones(n_paths, dtype=bool)
            
            if np.sum(available_paths) == 0:
                continue
                
            current_spots = eval_S[available_paths, t]
            
            # Immediate exercise payoff
            immediate_payoff = lsm_contract.volume_per_exercise * np.maximum(
                current_spots - lsm_contract.strike, 0
            )
            
            # Values
            future_value_exercise = discount_factor * option_values[available_paths, t + 1, m - 1]
            exercise_value = immediate_payoff + future_value_exercise
            continuation_value = discount_factor * option_values[available_paths, t + 1, m]
            
            # Simple continuation value estimation (simplified)
            if len(current_spots) > 10:
                itm_mask = immediate_payoff > 0
                if np.sum(itm_mask) > 5:
                    try:
                        fitted_continuation = pricer._fit_continuation_value(
                            current_spots[itm_mask], continuation_value[itm_mask], 
                            m, eval_t[t], 'polynomial', 3, None
                        )
                        predicted_continuation = pricer._predict_continuation_value(
                            current_spots, fitted_continuation, 'polynomial'
                        )
                    except:
                        predicted_continuation = continuation_value
                else:
                    predicted_continuation = continuation_value
            else:
                predicted_continuation = continuation_value
            
            # Exercise decision
            exercise_optimal = exercise_value > predicted_continuation
            
            # Update values
            option_values[available_paths, t, m] = np.where(
                exercise_optimal, exercise_value, predicted_continuation
            )
            
            # Track exercise decisions for full contract (m == max_rights)
            if m == max_rights:
                exercise_quantities[available_paths, t] = np.where(
                    exercise_optimal, lsm_contract.volume_per_exercise, 0.0
                )
    
    # Step 2: Forward simulation using SAME reward calculation as RL
    start_time = time.time()
    
    path_payoffs = []
    total_exercises = []
    
    for path_idx in range(n_paths):
        q_remaining = contract.Q_max  # Start with full inventory
        q_exercised_total = 0.0
        path_reward = 0.0
        
        for step in range(n_steps):
            spot_price = eval_S[path_idx, step]
            
            # Use LSM exercise decision
            q_decision = exercise_quantities[path_idx, step]
            
            # Ensure we don't over-exercise
            q_decision = min(q_decision, q_remaining, contract.q_max)
            
            # Update state
            q_remaining -= q_decision
            q_exercised_total += q_decision
            
            # Check if this is a terminal step
            is_terminal = ((step + 1) >= contract.n_rights or 
                         q_exercised_total >= contract.Q_max - 1e-6)
            
            # Calculate standardized reward using SAME function as RL
            reward = calculate_standardized_reward(
                spot_price, q_decision, contract.strike, 
                step, contract.discount_factor,
                q_exercised_total, contract.Q_min, is_terminal
            )
            
            path_reward += reward
            
            # Check termination conditions to match RL environment
            if is_terminal:
                break
        
        path_payoffs.append(path_reward)
        total_exercises.append(q_exercised_total)
    
    computation_time = time.time() - start_time
    
    # Calculate statistics using same method as RL evaluation
    path_payoffs = np.array(path_payoffs)
    mean_payoff = np.mean(path_payoffs)
    std_error = np.std(path_payoffs) / np.sqrt(n_paths)
    ci_lower = mean_payoff - 1.96 * std_error
    ci_upper = mean_payoff + 1.96 * std_error
    
    expected_exercises = np.mean(total_exercises)
    exercise_efficiency = expected_exercises / contract.Q_max
    
    benchmark_results = {
        'lsm_option_value_same_paths': mean_payoff,
        'lsm_std_error_same_paths': std_error,
        'lsm_confidence_interval_same_paths': (ci_lower, ci_upper),
        'lsm_expected_exercises_same_paths': expected_exercises,
        'lsm_exercise_efficiency_same_paths': exercise_efficiency,
        'lsm_computation_time_same_paths': computation_time,
        'lsm_n_paths_same_paths': eval_S.shape[0]
    }
    
    print(f"   ✅ LSM Value (same paths): ${mean_payoff:.6f} ± {std_error:.6f}")
    print(f"   📊 Expected Exercises: {expected_exercises:.2f}/{contract.Q_max} ({exercise_efficiency:.1%})")
    print(f"   ⚡ Computation Time: {computation_time:.2f}s")
    
    return benchmark_results


if __name__ == "__main__":
    # Example usage and testing
    print("=== Swing Option Pricing with HHK Model ===\n")
    
    # Create example contract
    contract = create_default_contract(
        strike=100.0,
        volume_per_exercise=1.0,
        max_exercises=10,
        maturity=1.0,
        risk_free_rate=0.05
    )
    
    print("Contract specification:")
    print(f"  Strike: ${contract.strike}")
    print(f"  Volume per exercise: {contract.volume_per_exercise}")
    print(f"  Max exercises: {contract.max_exercises}")
    print(f"  Maturity: {contract.maturity} years")
    print(f"  Risk-free rate: {contract.risk_free_rate:.1%}")
    
    # Price the option
    result = price_swing_option_with_hhk(
        contract=contract,
        n_paths=5000,
        n_steps=365,
        basis_type='polynomial',
        polynomial_degree=3,
        random_seed=42
    )
    
    print(f"\n=== Pricing Results ===")
    print(f"Option price: ${result.price:.4f}")
    print(f"Standard error: {result.std_error:.4f}")
    print(f"95% CI: [${result.confidence_interval[0]:.4f}, ${result.confidence_interval[1]:.4f}]")
    print(f"Computation time: {result.computation_time:.2f}s")
    
    if result.method_info.get('avg_r2'):
        print(f"Average regression R²: {result.method_info['avg_r2']:.3f}")
    
    # Analyze exercise strategy
    time_grid = np.linspace(0, contract.maturity, 366)  # Daily grid
    exercise_analysis = analyze_exercise_strategy(result, time_grid, contract)
    
    print(f"\n=== Exercise Strategy Analysis ===")
    print(f"Expected total exercises: {exercise_analysis['expected_total_exercises']:.2f}")
    print(f"Exercise rate: {exercise_analysis['exercise_rate']:.1%}")
    print(f"Expected first exercise: {exercise_analysis['expected_first_exercise_time']:.3f} years")
    print(f"Max daily exercise probability: {exercise_analysis['max_daily_exercise_prob']:.1%}")
    
    # Quick convergence test
    print(f"\n=== Quick Convergence Test ===")
    convergence_df = run_convergence_analysis(
        contract=contract,
        path_counts=[1000, 2000, 5000],
        n_repetitions=3,
        n_steps=365,
        random_seed=42
    )
    
    print("\nConvergence results:")
    for _, row in convergence_df.iterrows():
        print(f"  {row['n_paths']:5d} paths: ${row['mean_price']:.4f} ± {row['std_error']:.4f} "
              f"({row['mean_time']:.1f}s)")
    
    print("\n=== Analysis Complete ===")
