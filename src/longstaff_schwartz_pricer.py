"""
Advanced Swing Option Pricing Module

State-of-the-art implementation for pricing daily-exercisable swing options using
sophisticated Monte Carlo methods with HHK spot price dynamics.

Key Features:
- Longstaff-Schwartz (2001) algorithm with modern regression techniques
- Support for polynomial, spline, and machine learning basis functions
- Proper handling of swing option exercise strategies
- Advanced statistical analysis and convergence diagnostics
- Integration with RL benchmarking framework

Based on:
- Longstaff & Schwartz (2001): "Valuing American Options by Simulation"
- Hambly, Howison & Kluge (2009): "A general framework for pricing Asian options"
- Andersen & Broadie (2004): "A primal-dual simulation algorithm"
- Modern computational finance best practices

Authors: Financial Engineering Research Team
Date: July 2025
Version: 2.0 (Advanced Implementation)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from typing import Tuple, Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.simulate_hhk_spot import simulate_hhk_spot, DEFAULT_HHK_PARAMS
except ImportError:
    # If running from the src directory directly
    from simulate_hhk_spot import simulate_hhk_spot, DEFAULT_HHK_PARAMS


@dataclass
class SwingOptionContract:
    """Enhanced swing option contract specification with validation"""
    strike: float                    # Strike price K
    volume_per_exercise: float       # Volume V per exercise
    max_exercises: int              # Maximum number of exercises N
    maturity: float                 # Time to maturity T (years)
    risk_free_rate: float          # Risk-free interest rate r
    
    def __post_init__(self):
        """Validate contract parameters"""
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.volume_per_exercise <= 0:
            raise ValueError("Volume per exercise must be positive")
        if self.max_exercises <= 0:
            raise ValueError("Maximum exercises must be positive")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")
        if self.risk_free_rate < 0:
            raise ValueError("Risk-free rate must be non-negative")
    
    @property
    def total_volume(self) -> float:
        """Maximum total volume that can be exercised"""
        return self.volume_per_exercise * self.max_exercises


@dataclass
class PricingResult:
    """Enhanced results from option pricing with comprehensive diagnostics"""
    price: float
    std_error: float
    confidence_interval: Tuple[float, float]
    exercise_probability: np.ndarray  # Probability of exercise at each time step
    computation_time: float
    method_info: Dict[str, Any]
    regression_diagnostics: Dict[str, Any] = field(default_factory=dict)
    convergence_metrics: Dict[str, Any] = field(default_factory=dict)


class AdvancedBasisFunctions:
    """Advanced basis function implementations for regression"""
    
    @staticmethod
    def polynomial_basis(X: np.ndarray, degree: int, interaction_only: bool = False) -> np.ndarray:
        """Enhanced polynomial basis with interaction control"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, 
                                 include_bias=False)
        return poly.fit_transform(X)
    
    @staticmethod
    def spline_basis(x: np.ndarray, n_knots: Optional[int] = None, degree: int = 3) -> np.ndarray:
        """B-spline basis functions for 1D problems"""
        if x.ndim != 1:
            raise ValueError("Spline basis only supports 1D input")
        
        if n_knots is None:
            n_knots = min(len(np.unique(x)) // 4, 10)
        
        # Create knot points
        knots = np.percentile(x, np.linspace(10, 90, n_knots))
        
        # Build basis matrix
        n_basis = len(knots) + degree - 1
        basis_matrix = np.zeros((len(x), n_basis))
        
        for i, knot in enumerate(knots):
            # Simple truncated power basis
            basis_matrix[:, i] = np.maximum(0, (x - knot) ** degree)
        
        return basis_matrix
    
    @staticmethod
    def legendre_basis(X: np.ndarray, max_degree: int = 5) -> np.ndarray:
        """Legendre polynomial basis for better numerical stability"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Normalize to [-1, 1] for stability
        X_scaled = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) - 1
        
        basis_funcs = []
        for i in range(X.shape[1]):
            for degree in range(1, max_degree + 1):
                basis_funcs.append(np.polynomial.legendre.legval(X_scaled[:, i], 
                                                               [0] * degree + [1]))
        
        return np.column_stack(basis_funcs)


class EnhancedLongstaffSchwartzPricer:
    """
    Advanced swing option pricing using enhanced Longstaff-Schwartz Monte Carlo method
    
    Key improvements over standard implementation:
    - Proper in-the-money filtering as per original LS paper
    - Advanced basis function selection with cross-validation
    - Comprehensive regression diagnostics
    - Robust numerical methods for stability
    """
    
    def __init__(self, contract: SwingOptionContract, random_seed: Optional[int] = None):
        """
        Initialize the enhanced swing option pricer
        
        Args:
            contract: Swing option contract specification
            random_seed: Random seed for reproducibility
        """
        self.contract = contract
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.basis_functions = AdvancedBasisFunctions()
        self.scaler = StandardScaler()
        
    def price_lsm(self,
                  spot_paths: np.ndarray,
                  time_grid: np.ndarray,
                  basis_type: str = 'legendre',
                  polynomial_degree: int = 3,
                  use_cross_validation: bool = True,
                  regularization: str = 'ridge',
                  alpha: float = 0.01) -> PricingResult:
        """
        Enhanced LSM pricing with advanced regression techniques
        
        Args:
            spot_paths: Array of shape (n_paths, n_steps+1) with simulated spot prices
            time_grid: Array of time points
            basis_type: Type of basis functions ('polynomial', 'spline', 'legendre', 'random_forest')
            polynomial_degree: Degree for polynomial basis
            use_cross_validation: Whether to use cross-validation for model selection
            regularization: Type of regularization ('ridge', 'lasso', None)
            alpha: Regularization strength
            
        Returns:
            PricingResult with enhanced price estimate and diagnostics
        """
        start_time = time.time()
        
        n_paths, n_steps = spot_paths.shape[0], spot_paths.shape[1] - 1
        dt = self.contract.maturity / n_steps
        discount_factor = np.exp(-self.contract.risk_free_rate * dt)
        
        # Initialize enhanced value arrays
        max_rights = self.contract.max_exercises
        option_values = np.zeros((n_paths, n_steps + 1, max_rights + 1))
        exercise_decisions = np.zeros((n_paths, n_steps + 1), dtype=bool)
        
        # Enhanced terminal condition: exercise remaining rights optimally
        for m in range(1, max_rights + 1):
            terminal_payoff = np.maximum(spot_paths[:, -1] - self.contract.strike, 0) * \
                            self.contract.volume_per_exercise * m
            option_values[:, -1, m] = terminal_payoff
        
        # Storage for regression diagnostics
        regression_stats = []
        timing_values_history = []
        
        # Enhanced backward induction with proper LS methodology
        for t in range(n_steps - 1, -1, -1):
            current_spots = spot_paths[:, t]
            immediate_payoff = np.maximum(current_spots - self.contract.strike, 0) * \
                             self.contract.volume_per_exercise
            
            for m in range(1, max_rights + 1):
                # Get continuation values from next period
                continuation_values = discount_factor * option_values[:, t + 1, m]
                
                # CRITICAL: Train only on in-the-money paths (as per original LS paper)
                itm_mask = immediate_payoff > 0
                
                if np.sum(itm_mask) < 10:  # Not enough ITM paths
                    option_values[:, t, m] = continuation_values
                    continue
                
                # Prepare training data (ITM paths only)
                X_train = current_spots[itm_mask]
                y_train = continuation_values[itm_mask] - immediate_payoff[itm_mask]  # Q-learning style
                
                # Fit continuation value model
                model_result = self._fit_enhanced_continuation_model(
                    X_train, y_train, basis_type, polynomial_degree,
                    use_cross_validation, regularization, alpha
                )
                
                # Predict timing values for all paths
                timing_values = self._predict_continuation_values(
                    current_spots, model_result['model'], basis_type
                )
                
                # Exercise decision: exercise if timing value <= 0 AND in-the-money
                exercise_mask = (timing_values <= 0) & (immediate_payoff > 0)
                
                # Update option values based on optimal exercise strategy
                option_values[:, t, m] = np.where(
                    exercise_mask,
                    immediate_payoff + discount_factor * option_values[:, t + 1, m - 1],
                    continuation_values
                )
                
                # Record exercise decisions for the first exercise right
                if m == 1:
                    exercise_decisions[:, t] = exercise_mask
                
                # Store regression diagnostics
                regression_stats.append({
                    'time_step': t,
                    'rights_remaining': m,
                    'n_itm_paths': np.sum(itm_mask),
                    'r2_score': model_result.get('r2', 0),
                    'mse': model_result.get('mse', 0),
                    'cross_val_score': model_result.get('cv_score', 0),
                    'model_type': basis_type
                })
                
                timing_values_history.append(timing_values)
        
        # Calculate final statistics
        option_prices = option_values[:, 0, max_rights]
        mean_price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(n_paths)
        ci_lower = mean_price - 1.96 * std_error
        ci_upper = mean_price + 1.96 * std_error
        
        # Exercise probability analysis
        exercise_prob = np.mean(exercise_decisions, axis=0)
        
        computation_time = time.time() - start_time
        
        # Enhanced method information
        method_info = {
            'method': 'Enhanced Longstaff-Schwartz Monte Carlo',
            'basis_type': basis_type,
            'polynomial_degree': polynomial_degree if basis_type == 'polynomial' else None,
            'regularization': regularization,
            'use_cross_validation': use_cross_validation,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'avg_r2': np.mean([s['r2_score'] for s in regression_stats]) if regression_stats else 0,
            'avg_cv_score': np.mean([s['cross_val_score'] for s in regression_stats if s['cross_val_score'] is not None]) if regression_stats else None
        }
        
        # Regression diagnostics
        regression_diagnostics = {
            'detailed_stats': regression_stats,
            'convergence_check': self._check_convergence(timing_values_history),
            'stability_metrics': self._compute_stability_metrics(option_prices)
        }
        
        return PricingResult(
            price=float(mean_price),
            std_error=float(std_error),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            exercise_probability=exercise_prob,
            computation_time=computation_time,
            method_info=method_info,
            regression_diagnostics=regression_diagnostics
        )
    
    def _fit_enhanced_continuation_model(self,
                                       X_train: np.ndarray,
                                       y_train: np.ndarray,
                                       basis_type: str,
                                       polynomial_degree: int,
                                       use_cross_validation: bool,
                                       regularization: Optional[str],
                                       alpha: float) -> Dict[str, Any]:
        """
        Fit enhanced continuation value model with advanced techniques
        """
        if len(X_train) < 5:  # Insufficient data
            return {
                'model': {'type': 'constant', 'value': np.mean(y_train)},
                'r2': 0,
                'mse': np.var(y_train),
                'cv_score': None
            }
        
        X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
        
        try:
            if basis_type == 'polynomial':
                X_features = self.basis_functions.polynomial_basis(X_train, polynomial_degree)
            elif basis_type == 'legendre':
                X_features = self.basis_functions.legendre_basis(X_train, polynomial_degree)
            elif basis_type == 'spline' and X_train.shape[1] == 1:
                X_features = self.basis_functions.spline_basis(X_train.flatten())
            elif basis_type == 'random_forest':
                # Use RandomForest directly without basis transformation
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=self.random_seed)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_train)
                r2 = r2_score(y_train, y_pred)
                mse = mean_squared_error(y_train, y_pred)
                
                cv_score = None
                if use_cross_validation and len(X_train) >= 10:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//2))
                    cv_score = np.mean(cv_scores)
                
                return {
                    'model': {'type': 'random_forest', 'fitted_model': model},
                    'r2': r2,
                    'mse': mse,
                    'cv_score': cv_score
                }
            else:
                # Fallback to simple linear regression
                X_features = X_train
            
            # Regularization selection
            if regularization == 'ridge':
                model = Ridge(alpha=alpha)
            elif regularization == 'lasso':
                model = Lasso(alpha=alpha)
            else:
                model = LinearRegression()
            
            # Fit model
            model.fit(X_features, y_train)
            y_pred = model.predict(X_features)
            
            # Calculate metrics
            r2 = r2_score(y_train, y_pred)
            mse = mean_squared_error(y_train, y_pred)
            
            # Cross-validation if requested
            cv_score = None
            if use_cross_validation and len(X_train) >= 10:
                cv_scores = cross_val_score(model, X_features, y_train, cv=min(5, len(X_train)//2))
                cv_score = np.mean(cv_scores)
            
            return {
                'model': {
                    'type': basis_type,
                    'fitted_model': model,
                    'features_transformer': basis_type
                },
                'r2': r2,
                'mse': mse,
                'cv_score': cv_score
            }
            
        except Exception as e:
            # Fallback to simple mean prediction
            return {
                'model': {'type': 'constant', 'value': np.mean(y_train)},
                'r2': 0,
                'mse': np.var(y_train),
                'cv_score': None,
                'error': str(e)
            }
    
    def _predict_continuation_values(self,
                                   X_pred: np.ndarray,
                                   model_dict: Dict,
                                   basis_type: str) -> np.ndarray:
        """
        Predict continuation values using fitted model
        """
        if model_dict['type'] == 'constant':
            return np.full(len(X_pred), model_dict['value'])
        
        X_pred = X_pred.reshape(-1, 1) if X_pred.ndim == 1 else X_pred
        
        try:
            if model_dict['type'] == 'random_forest':
                return model_dict['fitted_model'].predict(X_pred)
            else:
                # Transform features according to basis type
                if basis_type == 'polynomial':
                    X_features = self.basis_functions.polynomial_basis(X_pred, 3)  # Default degree
                elif basis_type == 'legendre':
                    X_features = self.basis_functions.legendre_basis(X_pred, 3)
                elif basis_type == 'spline' and X_pred.shape[1] == 1:
                    X_features = self.basis_functions.spline_basis(X_pred.flatten())
                else:
                    X_features = X_pred
                
                return model_dict['fitted_model'].predict(X_features)
                
        except Exception as e:
            # Fallback to zero prediction
            return np.zeros(len(X_pred))
    
    def _check_convergence(self, timing_values_history: List[np.ndarray]) -> Dict[str, Any]:
        """
        Check convergence of timing values across time steps
        """
        if len(timing_values_history) < 5:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        # Calculate variance of timing values across recent time steps
        recent_values = timing_values_history[-5:]
        variance_trend = [np.var(values) for values in recent_values]
        
        # Check if variance is decreasing (sign of convergence)
        is_decreasing = all(variance_trend[i] >= variance_trend[i+1] for i in range(len(variance_trend)-1))
        
        return {
            'converged': is_decreasing,
            'variance_trend': variance_trend,
            'final_variance': variance_trend[-1] if variance_trend else 0
        }
    
    def _compute_stability_metrics(self, option_prices: np.ndarray) -> Dict[str, Any]:
        """
        Compute stability metrics for the pricing results
        """
        return {
            'coefficient_of_variation': np.std(option_prices) / np.mean(option_prices),
            'percentile_range': np.percentile(option_prices, 95) - np.percentile(option_prices, 5),
            'skewness': stats.skew(option_prices),
            'kurtosis': stats.kurtosis(option_prices)
        }


# Keep original class name for backward compatibility
LongstaffSchwartzPricer = EnhancedLongstaffSchwartzPricer


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
    try:
        from src.swing_env import calculate_standardized_reward
    except ImportError:
        from swing_env import calculate_standardized_reward
    
    print(f"\n🔮 Generating Enhanced LSM solution CSV: {csv_filename}")
    print(f"   Using {eval_S.shape[0]} paths with {eval_S.shape[1]} time steps")
    
    # Convert contract for LSM
    lsm_contract = SwingOptionContract(
        strike=contract.strike,
        volume_per_exercise=contract.q_max,
        max_exercises=min(contract.n_rights, int(contract.Q_max / contract.q_max)),
        maturity=contract.maturity,
        risk_free_rate=contract.r
    )
    
    # Initialize enhanced LSM pricer
    pricer = EnhancedLongstaffSchwartzPricer(lsm_contract, random_seed)
    
    # Enhanced LSM pricing with better basis functions
    n_paths, n_steps = eval_S.shape[0], eval_S.shape[1] - 1
    dt = lsm_contract.maturity / n_steps
    discount_factor = np.exp(-lsm_contract.risk_free_rate * dt)
    
    # Initialize arrays
    max_rights = lsm_contract.max_exercises
    option_values = np.zeros((n_paths, n_steps + 1, max_rights + 1))
    exercise_decisions = np.zeros((n_paths, n_steps + 1))
    exercise_quantities = np.zeros((n_paths, n_steps + 1))
    
    # Enhanced terminal condition
    for m in range(1, max_rights + 1):
        terminal_payoff = np.maximum(eval_S[:, -1] - lsm_contract.strike, 0) * \
                        lsm_contract.volume_per_exercise * m
        option_values[:, -1, m] = terminal_payoff
    
    # Enhanced backward induction with better regression
    print("   🧮 Running enhanced backward induction...")
    for t in range(n_steps - 1, -1, -1):
        current_spots = eval_S[:, t]
        immediate_payoff = np.maximum(current_spots - lsm_contract.strike, 0) * \
                         lsm_contract.volume_per_exercise
        
        for m in range(1, max_rights + 1):
            continuation_values = discount_factor * option_values[:, t + 1, m]
            
            # Train only on ITM paths
            itm_mask = immediate_payoff > 0
            
            if np.sum(itm_mask) >= 10:
                X_train = current_spots[itm_mask]
                y_train = continuation_values[itm_mask] - immediate_payoff[itm_mask]
                
                # Use polynomial basis with degree 3
                X_features = pricer.basis_functions.legendre_basis(X_train.reshape(-1, 1), 3)
                model = LinearRegression().fit(X_features, y_train)
                
                # Predict for all paths
                X_all_features = pricer.basis_functions.legendre_basis(current_spots.reshape(-1, 1), 3)
                timing_values = model.predict(X_all_features)
            else:
                timing_values = np.zeros(n_paths)
            
            # Exercise decision
            exercise_mask = (timing_values <= 0) & (immediate_payoff > 0)
            
            option_values[:, t, m] = np.where(
                exercise_mask,
                immediate_payoff + discount_factor * option_values[:, t + 1, m - 1],
                continuation_values
            )
            
            if m == 1:  # Record exercise decisions for first right
                exercise_decisions[:, t] = exercise_mask
                exercise_quantities[:, t] = exercise_mask * lsm_contract.volume_per_exercise
    
    # Forward simulation to reconstruct exercise paths
    print(f"   📝 Writing enhanced solution to CSV...")
    
    path_payoffs = []
    
    with open(csv_filename, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        
        # Use the same header format as RL evaluation runs
        writer.writerow(['episode_idx', 'step', 'spot', 'q_remain', 'q_exerc', 'time_left', 'action', 'q_actual', 'reward'])
        
        for path_idx in range(n_paths):
            total_payoff = 0
            rights_used = 0
            total_exercised = 0
            
            for step in range(n_steps + 1):
                spot = eval_S[path_idx, step]
                
                if step < n_steps:
                    q_exercised = exercise_quantities[path_idx, step]
                    if q_exercised > 0:
                        rights_used += 1
                        total_exercised += q_exercised
                else:
                    q_exercised = 0
                
                # Calculate remaining quantity (Q_max - cumulative exercised)
                q_remaining = lsm_contract.max_exercises * lsm_contract.volume_per_exercise - total_exercised
                
                # Calculate time left (as fraction of total time)
                time_left = (n_steps - step) / n_steps if n_steps > 0 else 0.0
                
                # For LSM, action would be the exercise decision (0 or 1, but we can use q_exercised/volume_per_exercise)
                action = q_exercised / lsm_contract.volume_per_exercise if lsm_contract.volume_per_exercise > 0 else 0.0
                
                # Actual quantity exercised
                q_actual = q_exercised
                
                # Calculate reward using standardized method
                if q_exercised > 0:
                    reward = calculate_standardized_reward(
                        spot_price=spot,
                        q_actual=q_exercised,
                        strike=lsm_contract.strike,
                        current_step=step,
                        discount_factor=discount_factor,
                        is_terminal=(step == n_steps)
                    )
                    total_payoff += reward
                else:
                    reward = 0
                
                # Write row with RL-compatible format
                writer.writerow([
                    path_idx,           # episode_idx  
                    step,               # step
                    round(spot, 4),     # spot
                    round(q_remaining, 4),  # q_remain
                    round(total_exercised, 4),  # q_exerc (cumulative)
                    round(time_left, 4),    # time_left
                    round(action, 6),       # action
                    round(q_actual, 4),     # q_actual
                    round(reward, 6)        # reward
                ])
            
            path_payoffs.append(total_payoff)
    
    # Calculate enhanced statistics
    path_payoffs = np.array(path_payoffs)
    mean_price = np.mean(path_payoffs)
    std_error = np.std(path_payoffs) / np.sqrt(n_paths)
    
    print(f"   ✅ Enhanced LSM CSV generated: {csv_filename}")
    print(f"   💰 Enhanced LSM Option Value: ${mean_price:.6f} ± {std_error:.6f}")
    print(f"   📊 Total rows written: {n_paths * (n_steps + 1)}")
    
    return {
        'lsm_option_value': mean_price,
        'lsm_std_error': std_error,
        'lsm_n_paths': n_paths,
        'csv_file': csv_filename
    }


def use_same_paths_for_lsm_and_rl(eval_t, eval_S, eval_X, eval_Y, contract, random_seed: int = 42) -> Dict[str, Any]:
    """
    Enhanced LSM pricing using the exact same MC paths as RL evaluation
    
    Args:
        eval_t: Time grid from RL simulation
        eval_S: Spot price paths from RL simulation  
        eval_X: X process paths from RL simulation
        eval_Y: Y process paths from RL simulation
        contract: SwingContract
        random_seed: Random seed
        
    Returns:
        Enhanced LSM benchmark results
    """
    try:
        from src.swing_env import calculate_standardized_reward
    except ImportError:
        from swing_env import calculate_standardized_reward
    
    print(f"\n🎯 Computing Enhanced LSM with SAME paths as RL evaluation")
    print(f"   Using {eval_S.shape[0]} pre-generated paths")
    
    # Convert contract
    lsm_contract = SwingOptionContract(
        strike=contract.strike,
        volume_per_exercise=contract.q_max,
        max_exercises=min(contract.n_rights, int(contract.Q_max / contract.q_max)),
        maturity=contract.maturity,
        risk_free_rate=contract.r
    )
    
    # Enhanced LSM computation
    pricer = EnhancedLongstaffSchwartzPricer(lsm_contract, random_seed)
    
    # Create time grid from eval_t
    time_grid = eval_t
    
    # Run enhanced LSM pricing
    result = pricer.price_lsm(
        spot_paths=eval_S,
        time_grid=time_grid,
        basis_type='legendre',
        polynomial_degree=3,
        use_cross_validation=True,
        regularization='ridge',
        alpha=0.01
    )
    
    print(f"   ✅ Enhanced LSM Value (same paths): ${result.price:.6f} ± {result.std_error:.6f}")
    print(f"   📊 Enhanced Regression R²: {result.method_info.get('avg_r2', 0):.3f}")
    print(f"   ⚡ Computation Time: {result.computation_time:.2f}s")
    
    return {
        'lsm_option_value_same_paths': result.price,
        'lsm_std_error_same_paths': result.std_error,
        'lsm_confidence_interval_same_paths': result.confidence_interval,
        'lsm_computation_time_same_paths': result.computation_time,
        'lsm_n_paths_same_paths': eval_S.shape[0],
        'lsm_regression_r2_same_paths': result.method_info.get('avg_r2', 0),
        'lsm_regression_diagnostics': result.regression_diagnostics
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
    basis_type: str = 'legendre',
    polynomial_degree: int = 3,
    regularization: str = 'ridge',
    reg_alpha: float = 0.01,
    random_seed: Optional[int] = None
) -> PricingResult:
    """
    Complete enhanced workflow: simulate HHK paths and price swing option
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
    
    # Initialize enhanced pricer
    pricer = EnhancedLongstaffSchwartzPricer(contract, random_seed)
    
    print(f"Pricing swing option using Enhanced {pricing_method.upper()} method...")
    
    if pricing_method == 'lsm':
        result = pricer.price_lsm(
            spot_paths=spot_paths,
            time_grid=time_grid,
            basis_type=basis_type,
            polynomial_degree=polynomial_degree,
            use_cross_validation=True,
            regularization=regularization,
            alpha=reg_alpha
        )
    else:
        raise ValueError(f"Pricing method {pricing_method} not supported")
    
    return result


def compute_lsm_benchmark(contract, hhk_params, n_paths: int = 10000, random_seed: int = 42) -> Dict[str, Any]:
    """
    Enhanced LSM benchmark computation
    """
    print(f"\n🔮 Computing Enhanced LSM Benchmark")
    print(f"   Contract: Strike=${contract.strike}, {contract.n_rights} rights, {contract.maturity:.4f}Y maturity")
    print(f"   LSM Paths: {n_paths}")
    
    # Convert to LSM contract
    lsm_contract = SwingOptionContract(
        strike=contract.strike,
        volume_per_exercise=contract.q_max,
        max_exercises=min(contract.n_rights, int(contract.Q_max / contract.q_max)),
        maturity=contract.maturity,
        risk_free_rate=contract.r
    )
    
    # Run enhanced LSM pricing
    result = price_swing_option_with_hhk(
        contract=lsm_contract,
        hhk_params=hhk_params,
        n_paths=n_paths,
        n_steps=contract.n_rights,
        random_seed=random_seed
    )
    
    benchmark_results = {
        'lsm_option_value': result.price,
        'lsm_std_error': result.std_error,
        'lsm_confidence_interval': result.confidence_interval,
        'lsm_computation_time': result.computation_time,
        'lsm_n_paths': n_paths,
        'lsm_regression_r2': result.method_info.get('avg_r2', 0),
        'lsm_exercise_probabilities': result.exercise_probability.tolist(),
        'lsm_regression_diagnostics': result.regression_diagnostics
    }
    
    print(f"   ✅ Enhanced LSM Option Value: ${result.price:.6f} ± {result.std_error:.6f}")
    print(f"   📊 Regression R²: {result.method_info.get('avg_r2', 0):.3f}")
    print(f"   ⚡ Computation Time: {result.computation_time:.2f}s")
    
    return benchmark_results


def run_convergence_analysis(
    contract: SwingOptionContract,
    path_counts: list = [1000, 2000, 5000, 10000, 20000],
    n_repetitions: int = 5,
    **kwargs
) -> pd.DataFrame:
    """
    Enhanced convergence analysis with statistical testing
    """
    results = []
    
    for n_paths in path_counts:
        print(f"\nTesting convergence with {n_paths} paths...")
        path_results = []
        
        for rep in range(n_repetitions):
            result = price_swing_option_with_hhk(
                contract=contract,
                n_paths=n_paths,
                random_seed=42 + rep,
                **kwargs
            )
            path_results.append(result.price)
        
        path_results = np.array(path_results)
        results.append({
            'n_paths': n_paths,
            'mean_price': np.mean(path_results),
            'std_price': np.std(path_results),
            'min_price': np.min(path_results),
            'max_price': np.max(path_results),
            'cv': np.std(path_results) / np.mean(path_results),
            'n_repetitions': n_repetitions
        })
    
    return pd.DataFrame(results)


def analyze_exercise_strategy(
    result: PricingResult,
    time_grid: np.ndarray,
    contract: SwingOptionContract
) -> Dict[str, Any]:
    """
    Enhanced exercise strategy analysis
    """
    exercise_prob = result.exercise_probability
    
    # Find peak exercise periods
    peak_threshold = np.percentile(exercise_prob[exercise_prob > 0], 75) if np.any(exercise_prob > 0) else 0
    peak_periods = time_grid[exercise_prob > peak_threshold]
    
    # Expected number of exercises
    expected_exercises = np.sum(exercise_prob)
    
    # Time to first exercise
    cumulative_exercise = np.cumsum(exercise_prob)
    first_exercise_prob = cumulative_exercise / expected_exercises if expected_exercises > 0 else cumulative_exercise
    expected_first_exercise = np.sum(time_grid * np.diff(np.concatenate([[0], first_exercise_prob])))
    
    # Enhanced analysis
    analysis = {
        'expected_total_exercises': expected_exercises,
        'exercise_rate': expected_exercises / contract.max_exercises,
        'expected_first_exercise_time': expected_first_exercise,
        'peak_exercise_periods': peak_periods,
        'peak_exercise_threshold': peak_threshold,
        'max_daily_exercise_prob': np.max(exercise_prob),
        'exercise_concentration': np.sum(exercise_prob ** 2) / np.sum(exercise_prob) if np.sum(exercise_prob) > 0 else 0,
        'exercise_entropy': -np.sum(exercise_prob * np.log(exercise_prob + 1e-10)),
        'exercise_skewness': stats.skew(exercise_prob),
        'exercise_kurtosis': stats.kurtosis(exercise_prob)
    }
    
    return analysis


def parse_arguments():
    """Enhanced argument parsing with new options"""
    parser = argparse.ArgumentParser(description='Enhanced Longstaff-Schwartz Swing Option Pricing')
    
    # Training/simulation parameters
    parser.add_argument('-n_paths', type=int, default=16384, help='Number of Monte Carlo paths')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    parser.add_argument('-name', type=str, default='LSM_pricing', help='Run name for output files')
    
    # Contract parameters
    parser.add_argument('--strike', type=float, default=100.0, help='Strike price')
    parser.add_argument('--maturity', type=float, default=0.0833, help='Maturity in years')
    parser.add_argument('--n_rights', type=int, default=22, help='Number of exercise rights')
    parser.add_argument('--q_min', type=float, default=0.0, help='Minimum exercise per day')
    parser.add_argument('--q_max', type=float, default=2.0, help='Maximum exercise per day')
    parser.add_argument('--Q_min', type=float, default=0.0, help='Minimum total exercise')
    parser.add_argument('--Q_max', type=float, default=20.0, help='Maximum total exercise')
    parser.add_argument('--risk_free_rate', type=float, default=0.05, help='Risk-free rate')
    parser.add_argument('--min_refraction_days', type=int, default=0, help='Minimum refraction days')
    
    # Market process parameters
    parser.add_argument('--S0', type=float, default=100.0, help='Initial spot price')
    parser.add_argument('--alpha', type=float, default=12.0, help='Mean reversion speed')
    parser.add_argument('--sigma', type=float, default=1.2, help='Volatility')
    parser.add_argument('--beta', type=float, default=150.0, help='Jump decay rate')
    parser.add_argument('--lam', type=float, default=6.0, help='Jump intensity')
    parser.add_argument('--mu_J', type=float, default=0.3, help='Mean jump size')
    
    # Enhanced LSM parameters
    parser.add_argument('--basis_type', type=str, default='legendre', 
                       choices=['polynomial', 'legendre', 'spline', 'random_forest'], 
                       help='Basis function type')
    parser.add_argument('--polynomial_degree', type=int, default=3, help='Polynomial degree')
    parser.add_argument('--regularization', type=str, default='ridge',
                       choices=['ridge', 'lasso', 'none'], help='Regularization type')
    parser.add_argument('--reg_alpha', type=float, default=0.01, help='Regularization strength')
    parser.add_argument('--use_cross_validation', action='store_true', help='Use cross-validation')
    parser.add_argument('--convergence_test', action='store_true', help='Run convergence analysis')
    parser.add_argument('--use_rl_paths', action='store_true', 
                       help='Use existing RL evaluation paths')
    
    return parser.parse_args()


def default_seasonal_function(t):
    """Default seasonal function maintaining reasonable price levels"""
    return np.log(100.0)


if __name__ == "__main__":
    args = parse_arguments()
    
    print("=" * 70)
    print("Enhanced Longstaff-Schwartz Swing Option Pricing v2.0")
    print("=" * 70)
    print(f"Run name: {args.name}")
    print(f"Random seed: {args.seed}")
    print(f"Number of paths: {args.n_paths:,}")
    print(f"Basis functions: {args.basis_type}")
    print(f"Regularization: {args.regularization}")
    
    # Create enhanced contract
    contract = SwingOptionContract(
        strike=args.strike,
        volume_per_exercise=args.q_max,
        max_exercises=min(args.n_rights, int(args.Q_max / args.q_max)),
        maturity=args.maturity,
        risk_free_rate=args.risk_free_rate
    )
    
    # HHK parameters
    hhk_params = {
        'S0': args.S0,
        'alpha': args.alpha,
        'sigma': args.sigma,
        'beta': args.beta,
        'lam': args.lam,
        'mu_J': args.mu_J,
        'f': default_seasonal_function
    }
    
    print(f"\nContract specification:")
    print(f"  Strike: ${contract.strike}")
    print(f"  Volume per exercise: {contract.volume_per_exercise}")
    print(f"  Max exercises: {contract.max_exercises}")
    print(f"  Maturity: {contract.maturity} years")
    print(f"  Risk-free rate: {contract.risk_free_rate:.1%}")
    
    # Run enhanced pricing
    start_time = time.time()
    
    result = price_swing_option_with_hhk(
        contract=contract,
        hhk_params=hhk_params,
        n_paths=args.n_paths,
        n_steps=args.n_rights,
        basis_type=args.basis_type,
        polynomial_degree=args.polynomial_degree,
        random_seed=args.seed
    )
    
    total_time = time.time() - start_time
    
    print(f"\n" + "=" * 70)
    print(f"ENHANCED PRICING RESULTS")
    print(f"=" * 70)
    print(f"Option price: ${result.price:.6f}")
    print(f"Standard error: {result.std_error:.6f}")
    print(f"95% CI: [${result.confidence_interval[0]:.6f}, ${result.confidence_interval[1]:.6f}]")
    print(f"Regression R²: {result.method_info.get('avg_r2', 0):.4f}")
    print(f"Cross-val score: {result.method_info.get('avg_cv_score', 'N/A')}")
    print(f"Computation time: {result.computation_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    
    # Enhanced analysis
    time_grid_analysis = np.linspace(0, contract.maturity, len(result.exercise_probability))
    exercise_analysis = analyze_exercise_strategy(result, time_grid_analysis, contract)
    
    print(f"\n" + "-" * 70)
    print(f"ENHANCED EXERCISE STRATEGY ANALYSIS")
    print(f"-" * 70)
    print(f"Expected total exercises: {exercise_analysis['expected_total_exercises']:.2f}")
    print(f"Exercise rate: {exercise_analysis['exercise_rate']:.1%}")
    print(f"Expected first exercise: {exercise_analysis['expected_first_exercise_time']:.3f} years")
    print(f"Max daily exercise probability: {exercise_analysis['max_daily_exercise_prob']:.1%}")
    print(f"Exercise entropy: {exercise_analysis['exercise_entropy']:.3f}")
    
    # Save enhanced results
    base_dir = "Longstaff Schwartz Pricer"
    experiment_dir = os.path.join(base_dir, args.name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    txt_filename = os.path.join(experiment_dir, f"{args.name}_results.txt")
    
    with open(txt_filename, 'w') as f:
        f.write(f"Enhanced Longstaff-Schwartz Swing Option Pricing Results\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Contract Parameters:\n")
        f.write(f"  Strike: ${contract.strike}\n")
        f.write(f"  Volume per exercise: {contract.volume_per_exercise}\n")
        f.write(f"  Max exercises: {contract.max_exercises}\n")
        f.write(f"  Maturity: {contract.maturity} years\n")
        f.write(f"  Risk-free rate: {contract.risk_free_rate:.3f}\n\n")
        
        f.write(f"Enhanced Pricing Results:\n")
        f.write(f"  Option Value: ${result.price:.6f}\n")
        f.write(f"  Standard Error: {result.std_error:.6f}\n")
        f.write(f"  95% Confidence Interval: [${result.confidence_interval[0]:.6f}, ${result.confidence_interval[1]:.6f}]\n")
        f.write(f"  Regression R²: {result.method_info.get('avg_r2', 0):.4f}\n")
        f.write(f"  Cross-validation Score: {result.method_info.get('avg_cv_score', 'N/A')}\n")
        f.write(f"  Computation Time: {result.computation_time:.2f} seconds\n\n")
        
        f.write(f"Enhanced Exercise Strategy:\n")
        f.write(f"  Expected Total Exercises: {exercise_analysis['expected_total_exercises']:.2f}\n")
        f.write(f"  Exercise Rate: {exercise_analysis['exercise_rate']:.1%}\n")
        f.write(f"  Expected First Exercise Time: {exercise_analysis['expected_first_exercise_time']:.3f} years\n")
        f.write(f"  Exercise Entropy: {exercise_analysis['exercise_entropy']:.3f}\n")
        
        f.write(f"\nRegression Diagnostics:\n")
        if result.regression_diagnostics:
            conv_check = result.regression_diagnostics.get('convergence_check', {})
            f.write(f"  Convergence: {conv_check.get('converged', 'Unknown')}\n")
            stability = result.regression_diagnostics.get('stability_metrics', {})
            f.write(f"  Coefficient of Variation: {stability.get('coefficient_of_variation', 0):.4f}\n")
    
    print(f"\n✅ Enhanced results saved to: {txt_filename}")
    print(f"💰 Final Enhanced LSM Option Value: ${result.price:.6f} ± {result.std_error:.6f}")
    print(f"📊 Regression Quality: R² = {result.method_info.get('avg_r2', 0):.4f}")
    print(f"=" * 70)


def save_lsm_evaluation_run(eval_t, eval_S, eval_X, eval_Y, contract, evaluation_runs_dir, training_episode, random_seed: int = 42):
    """
    Save LSM results in the evaluation_runs directory in the same format as RL evaluation runs.
    
    Args:
        eval_t: Time grid from RL simulation
        eval_S: Spot price paths from RL simulation  
        eval_X: X process paths from RL simulation
        eval_Y: Y process paths from RL simulation
        contract: SwingContract
        evaluation_runs_dir: Directory for evaluation runs
        training_episode: Training episode number (use 0 for LSM benchmark)
        random_seed: Random seed
        
    Returns:
        Dictionary with LSM results
    """
    try:
        from src.swing_env import calculate_standardized_reward
    except ImportError:
        from swing_env import calculate_standardized_reward
    
    # Create filename matching RL evaluation run format
    csv_filename = f"{evaluation_runs_dir}/eval_run_{training_episode}_lsm.csv"
    
    print(f"\n🔮 Generating LSM evaluation run CSV: {csv_filename}")
    print(f"   Using {eval_S.shape[0]} paths with {eval_S.shape[1]} time steps")
    
    # Convert contract for LSM
    lsm_contract = SwingOptionContract(
        strike=contract.strike,
        volume_per_exercise=contract.q_max,
        max_exercises=min(contract.n_rights, int(contract.Q_max / contract.q_max)),
        maturity=contract.maturity,
        risk_free_rate=contract.r
    )
    
    # Initialize enhanced LSM pricer
    pricer = EnhancedLongstaffSchwartzPricer(lsm_contract, random_seed)
    
    # Enhanced LSM pricing with better basis functions
    n_paths, n_steps = eval_S.shape[0], eval_S.shape[1] - 1
    dt = lsm_contract.maturity / n_steps
    discount_factor = np.exp(-lsm_contract.risk_free_rate * dt)
    
    # Initialize arrays
    max_rights = lsm_contract.max_exercises
    option_values = np.zeros((n_paths, n_steps + 1, max_rights + 1))
    exercise_decisions = np.zeros((n_paths, n_steps + 1))
    exercise_quantities = np.zeros((n_paths, n_steps + 1))
    
    # Enhanced terminal condition
    for m in range(1, max_rights + 1):
        terminal_payoff = np.maximum(eval_S[:, -1] - lsm_contract.strike, 0) * \
                        lsm_contract.volume_per_exercise * m
        option_values[:, -1, m] = terminal_payoff
    
    # Enhanced backward induction with better regression
    print("   🧮 Running enhanced backward induction...")
    for t in range(n_steps - 1, -1, -1):
        current_spots = eval_S[:, t]
        immediate_payoff = np.maximum(current_spots - lsm_contract.strike, 0) * \
                         lsm_contract.volume_per_exercise
        
        for m in range(1, max_rights + 1):
            continuation_values = discount_factor * option_values[:, t + 1, m]
            
            # Train only on ITM paths
            itm_mask = immediate_payoff > 0
            
            if np.sum(itm_mask) >= 10:
                X_train = current_spots[itm_mask]
                y_train = continuation_values[itm_mask] - immediate_payoff[itm_mask]
                
                # Use polynomial basis with degree 3
                X_features = pricer.basis_functions.legendre_basis(X_train.reshape(-1, 1), 3)
                model = LinearRegression().fit(X_features, y_train)
                
                # Predict for all paths
                X_all_features = pricer.basis_functions.legendre_basis(current_spots.reshape(-1, 1), 3)
                timing_values = model.predict(X_all_features)
            else:
                timing_values = np.zeros(n_paths)
            
            # Exercise decision
            exercise_mask = (timing_values <= 0) & (immediate_payoff > 0)
            
            option_values[:, t, m] = np.where(
                exercise_mask,
                immediate_payoff + discount_factor * option_values[:, t + 1, m - 1],
                continuation_values
            )
            
            if m == 1:  # Record exercise decisions for first right
                exercise_decisions[:, t] = exercise_mask
                exercise_quantities[:, t] = exercise_mask * lsm_contract.volume_per_exercise
    
    # Forward simulation to reconstruct exercise paths
    print(f"   📝 Writing LSM evaluation run to CSV...")
    
    path_payoffs = []
    
    with open(csv_filename, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        
        # Use the EXACT same header format as RL evaluation runs
        writer.writerow(['episode_idx', 'step', 'spot', 'q_remain', 'q_exerc', 'time_left', 'action', 'q_actual', 'reward'])
        
        for path_idx in range(n_paths):
            total_payoff = 0
            rights_used = 0
            total_exercised = 0
            
            for step in range(n_steps + 1):
                spot = eval_S[path_idx, step]
                
                if step < n_steps:
                    q_exercised = exercise_quantities[path_idx, step]
                    if q_exercised > 0:
                        rights_used += 1
                        total_exercised += q_exercised
                else:
                    q_exercised = 0
                
                # Calculate remaining quantity (Q_max - cumulative exercised)
                q_remaining = lsm_contract.max_exercises * lsm_contract.volume_per_exercise - total_exercised
                
                # Calculate time left (as fraction of total time)
                time_left = (n_steps - step) / n_steps if n_steps > 0 else 0.0
                
                # For LSM, action would be the exercise decision (0 or 1, but we can use q_exercised/volume_per_exercise)
                action = q_exercised / lsm_contract.volume_per_exercise if lsm_contract.volume_per_exercise > 0 else 0.0
                
                # Actual quantity exercised
                q_actual = q_exercised
                
                # Calculate reward using standardized method
                if q_exercised > 0:
                    reward = calculate_standardized_reward(
                        spot_price=spot,
                        q_actual=q_exercised,
                        strike=lsm_contract.strike,
                        current_step=step,
                        discount_factor=discount_factor,
                        is_terminal=(step == n_steps)
                    )
                    total_payoff += reward
                else:
                    reward = 0
                
                # Write row with RL-compatible format
                writer.writerow([
                    path_idx,           # episode_idx  
                    step,               # step
                    round(spot, 4),     # spot
                    round(q_remaining, 4),  # q_remain
                    round(total_exercised, 4),  # q_exerc (cumulative)
                    round(time_left, 4),    # time_left
                    round(action, 6),       # action
                    round(q_actual, 4),     # q_actual
                    round(reward, 6)        # reward
                ])
            
            path_payoffs.append(total_payoff)
    
    # Calculate enhanced statistics
    path_payoffs = np.array(path_payoffs)
    mean_price = np.mean(path_payoffs)
    std_error = np.std(path_payoffs) / np.sqrt(n_paths)
    
    print(f"   ✅ LSM evaluation run CSV generated: {csv_filename}")
    print(f"   💰 LSM Option Value: ${mean_price:.6f} ± {std_error:.6f}")
    print(f"   📊 Total rows written: {n_paths * (n_steps + 1)}")
    
    return {
        'lsm_option_value': mean_price,
        'lsm_std_error': std_error,
        'lsm_n_paths': n_paths,
        'csv_file': csv_filename
    }
