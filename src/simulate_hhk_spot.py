"""
Hambly-Howison-Kluge (HHK) Spot Price Simulation
Based on "Modelling spikes and pricing swing options in electricity markets" 
Hambly et al. (2009)
"""
import numpy as np
from scipy.stats import qmc, norm, poisson, gamma
from typing import Tuple, Callable, Optional


def simulate_hhk_spot(
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    alpha: float,
    sigma: float,
    beta: float,
    lam: float,
    mu_J: float,
    f: Callable[[float], float],
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate spot prices using the Hambly–Howison–Kluge model with exponential jump sizes.
    
    The model follows:
    dX_t = -α X_t dt + σ dW_t
    dY_t = -β Y_t dt + J_t dN_t  
    S_t = exp(f(t) + X_t + Y_t)
    
    Where:
    - X_t is mean-reverting Ornstein-Uhlenbeck process
    - Y_t is mean-reverting jump process  
    - N_t is Poisson process with intensity λ
    - J_t are i.i.d. exponential(1/μ_J) jump sizes
    - f(t) is deterministic seasonal function
    
    Args:
        S0: Initial spot price
        T: Time horizon in years
        n_steps: Number of time steps
        n_paths: Number of Monte Carlo paths
        alpha: Mean reversion speed for X process
        sigma: Volatility of X process
        beta: Mean reversion speed for Y process (jump decay)
        lam: Jump intensity (jumps per unit time)
        mu_J: Mean jump size (scale parameter for exponential distribution)
        f: Seasonal function f(t)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (time_grid, spot_paths, X_paths, Y_paths)
        - time_grid: (n_steps+1,) array of time points
        - spot_paths: (n_paths, n_steps+1) array of spot prices
        - X_paths: (n_paths, n_steps+1) array of X process values
        - Y_paths: (n_paths, n_steps+1) array of Y process values
    """
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    # Exact OU step statistics for X process
    e_m = np.exp(-alpha * dt)
    var_m = sigma**2 * (1.0 - e_m**2) / (2.0 * alpha)
    
    # Y process decay factor
    e_y = np.exp(-beta * dt)

    # Generate quasi-random numbers using Sobol sequence
    # For each time step we need: Z_X (Gaussian), U_Poisson, U_Gamma
    dim_per_step = 3
    sampler = qmc.Sobol(d=dim_per_step * n_steps, scramble=True)
    if seed is not None:
        np.random.seed(seed)
    u = sampler.random(n_paths).reshape(n_paths, n_steps, dim_per_step).transpose(1, 0, 2)

    # Ensure no exactly 0 or 1 values for inverse transforms
    eps = 1e-12
    u = np.clip(u, eps, 1.0 - eps)

    # Transform uniform to required distributions
    z_x = norm.ppf(u[:, :, 0])  # Gaussian innovations for X
    n_jmp = poisson.ppf(u[:, :, 1], lam * dt).astype(int)  # Number of jumps per step
    u_gam = u[:, :, 2]  # Uniform for jump size generation

    # Initialize state arrays
    X = np.empty((n_paths, n_steps + 1))
    Y = np.empty((n_paths, n_steps + 1))
    S = np.empty((n_paths, n_steps + 1))
    
    # Initial conditions
    X[:, 0] = np.log(S0) - f(0.0)  # X_0 such that S_0 = S0
    Y[:, 0] = 0.0
    S[:, 0] = S0

    # Simulate forward in time
    for k in range(1, n_steps + 1):
        # OU process for X: exact discretization
        X[:, k] = e_m * X[:, k-1] + np.sqrt(var_m) * z_x[k-1]

        # Y process with mean-reverting jumps
        n_k = n_jmp[k-1]  # Number of jumps in this step
        
        # Sum of exponential jumps using gamma distribution property:
        # Sum of n i.i.d. Exp(1/μ) ~ Gamma(n, μ)
        jump_sum = np.where(
            n_k > 0,
            gamma.ppf(u_gam[k-1], a=n_k, scale=mu_J),
            0.0
        )
        Y[:, k] = e_y * Y[:, k-1] + jump_sum

        # Spot price
        S[:, k] = np.exp(f(t[k]) + X[:, k] + Y[:, k])

    return t, S, X, Y


def default_seasonal_function(t: float) -> float:
    """
    Default seasonal function with annual cycle
    
    Args:
        t: Time in years
        
    Returns:
        Seasonal component value
    """
    return 0 #np.log(100.0) + 0.5 * np.cos(2 * np.pi * t)


# Default HHK parameters based on Hambly et al. (2009)
DEFAULT_HHK_PARAMS = {
    'S0': 100.0,
    'alpha': 7.0,        # Fast mean reversion for normal variations  
    'sigma': 1.4,        # Volatility of normal variations
    'beta': 200.0,       # Very fast decay of spike component
    'lam': 4.0,          # 4 spikes per year on average
    'mu_J': 0.4,         # Average spike size (use 0.8 for bigger spikes)
    'f': default_seasonal_function
}


def simulate_single_path(
    T: float,
    n_steps: int,
    seed: Optional[int] = None,
    **hhk_params
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to simulate a single HHK path
    
    Args:
        T: Time horizon in years
        n_steps: Number of time steps
        seed: Random seed
        **hhk_params: HHK model parameters (uses defaults if not provided)
        
    Returns:
        Tuple of (time_grid, spot_path)
    """
    params = DEFAULT_HHK_PARAMS.copy()
    params.update(hhk_params)
    
    t, S, _, _ = simulate_hhk_spot(
        T=T,
        n_steps=n_steps, 
        n_paths=1,
        seed=seed,
        **params
    )
    
    return t, S[0]  # Return single path


if __name__ == "__main__":
    # Test simulation
    import matplotlib.pyplot as plt
    
    # Simulate test paths
    t, S, X, Y = simulate_hhk_spot(
        S0=100.0,
        T=2.0,
        n_steps=730,
        n_paths=1000,
        seed=42,
        **DEFAULT_HHK_PARAMS
    )
    
    print(f"Simulated {S.shape[0]} paths over {t[-1]:.1f} years")
    print(f"Final price statistics: mean={S[:,-1].mean():.2f}, std={S[:,-1].std():.2f}")
    print(f"Price range: [{S.min():.2f}, {S.max():.2f}]")
    
    # Plot sample paths
    plt.figure(figsize=(12, 4))
    plt.plot(t, S[:50].T, alpha=0.3, color='blue')
    plt.plot(t, S.mean(axis=0), 'r-', linewidth=2, label='Mean')
    plt.xlabel('Time (years)')
    plt.ylabel('Spot Price')
    plt.title('HHK Spot Price Simulation')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
