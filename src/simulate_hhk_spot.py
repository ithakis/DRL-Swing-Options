from typing import Callable, Optional, Tuple

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import numpy as np
from scipy.stats import norm, qmc
from tqdm import tqdm


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
    dtype: np.dtype | str = np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hambly–Howison–Kluge spot model with antithetic variance reduction.
    
    Note: Assumes n_paths is always even and always applies antithetic variance reduction.

    Returns
    -------
    t : (n_steps+1,) ndarray     time grid
    S : (n_paths, n_steps+1)     spot price paths
    X : (n_paths, n_steps+1)     diffusive OU paths
    Y : (n_paths, n_steps+1)     spike (jump‑OU) paths
    """
    dtype = np.dtype(dtype)
    dtype_scalar = dtype.type

    rng = np.random.default_rng(seed)
    dt  = dtype_scalar(T / n_steps)
    t   = np.linspace(0.0, dtype_scalar(T), n_steps + 1, dtype=dtype)

    # ── Diffusive OU driver (Sobol) ───────────────────────────────────────────
    if seed is not None: np.random.seed(seed)
    sampler = qmc.Sobol(d=n_steps, scramble=True)
    z_x     = norm.ppf(np.clip(sampler.random(n_paths), 1e-12, 1-1e-12)).astype(dtype, copy=False)

    e_m     = dtype_scalar(np.exp(-alpha * dt))
    var_m   = dtype_scalar(sigma**2 * (1.0 - e_m**2) / (2.0 * alpha))
    sqrt_vm = np.sqrt(var_m, dtype=dtype)

    e_dt = dtype_scalar(np.exp(-beta * dt))                 # common factor in jump weight

    # ── Allocate output arrays ────────────────────────────────────────────────
    X = np.empty((n_paths, n_steps + 1), dtype=dtype)
    Y = np.zeros_like(X)
    S = np.empty_like(X)

    X[:, 0] = dtype_scalar(np.log(S0) - f(0.0))   # ensure S starts at S0
    S[:, 0] = dtype_scalar(S0)

    # ── Pre‑draw Poisson counts ───────────────────────────────────────────────
    counts = rng.poisson(lam * dt, size=(n_steps, n_paths // 2))

    # ── MAIN TIME LOOP – VECTORISED INSIDE ────────────────────────────────────        
    for k in tqdm(range(1, n_steps + 1), leave=False, desc="Simulating"):

        # --- diffusive OU -----------------------------------------------------
        X[:, k] = e_m * X[:, k - 1] + sqrt_vm * z_x[:, k - 1]

        # --- spike OU with antithetic variance reduction ----------------------
        jump_inc = np.zeros(n_paths, dtype=dtype)
        
        # Antithetic variance reduction for multiple paths
        c_k = counts[k - 1]                       # (n_paths//2,)
        pos_idx = np.nonzero(c_k)[0]              # indices with ≥1 jump

        if pos_idx.size:                          # skip if no jumps
            n_tot = c_k[pos_idx].sum()            # total #jumps this step
            U = rng.uniform(0.0, dt, size=n_tot).astype(dtype, copy=False)  # arrival times
            V = rng.random(n_tot).astype(dtype, copy=False)                 # uniforms for Exp marks
            V_bar = dtype_scalar(1.0) - V                                   # antithetic uniforms

            # exponential jump sizes (antithetic)
            J1 = (-mu_J * np.log(V)).astype(dtype, copy=False)
            J2 = (-mu_J * np.log(V_bar)).astype(dtype, copy=False)

            decay = (e_dt * np.exp(beta * U)).astype(dtype, copy=False)       # e^{-β(dt-U)}

            # map each draw to its pair id
            pair_ids = np.repeat(pos_idx, c_k[pos_idx])

            # accumulate contributions to each half‑pair
            contrib1 = np.bincount(pair_ids, weights=J1 * decay,
                                   minlength=n_paths // 2)
            contrib2 = np.bincount(pair_ids, weights=J2 * decay,
                                   minlength=n_paths // 2)

            jump_inc[0::2] = contrib1             # original path
            jump_inc[1::2] = contrib2             # antithetic path

        Y[:, k] = np.exp(-beta * dt) * Y[:, k - 1] + jump_inc

        # --- spot price -------------------------------------------------------
        S[:, k] = np.exp(f(t[k]) + X[:, k] + Y[:, k])

    return t, S, X, Y

######################################################################################
######################## Validation of the Stochastic Process ########################
def bootstrap_moments(data: np.ndarray):
    """
    Bootstrap the empirical moments of a given dataset.

    Args:
        data: 1D numpy array of data points.
       
    Returns:
        Tuple of (mean_results, std_results) where each result is a list
        containing the value, lower bound, and upper bound of the statistic.
    """
    # First Moment: Mean - E[X]
    mean_results = bs.bootstrap(data, stat_func=bs_stats.mean, is_pivotal=False,
                                iteration_batch_size=128, num_iterations=128*4*16, num_threads=-1)

    # Second Moment: Std - Sqrt(Var[X])
    std_results = bs.bootstrap(data, stat_func=bs_stats.std, is_pivotal=False,
                               iteration_batch_size=128, num_iterations=128*4*16, num_threads=-1)

    mean_results = [float(mean_results.value), float(mean_results.lower_bound), float(mean_results.upper_bound)]
    std_results = [float(std_results.value), float(std_results.lower_bound), float(std_results.upper_bound)]
    return mean_results, std_results


def theoretical_moments(
    S0: float,
    T: float,
    alpha: float,
    sigma: float,
    beta: float,
    lam: float,
    mu_J: float,
    f: Callable[[float], float],
) -> Tuple[float, float, float, float, float, float]:
    """
    Analytic moments at time T for the Hambly–Howison–Kluge spot model
        S_t = exp( f(t) + X_t + Y_t ).

    Parameters
    ----------
    S0      : initial spot  S_0
    T       : horizon (years)
    alpha   : mean-reversion speed of diffusive OU factor X
    sigma   : volatility of X
    beta    : mean-reversion speed of spike factor Y
    lam     : Poisson intensity of jumps in Y
    mu_J    : mean of exponential jump sizes  ( J ~ Exp(1/μ_J) )
    f       : deterministic seasonal function  f(t)

    Returns
    -------
    ( E[S_T] ,  Std[S_T] ,
      E[X_T] ,  Std[X_T] ,
      E[Y_T] ,  Std[Y_T] )
    """
    # --- initial factor values consistent with S0 ----------------------------
    Y0 = 0.0                                   # same convention as simulator
    X0 = np.log(S0) - f(0.0) - Y0

    # --- moments of X --------------------------------------------------------
    mX = X0 * np.exp(-alpha * T)
    vX = sigma**2 * (1.0 - np.exp(-2.0 * alpha * T)) / (2.0 * alpha)

    # --- moments of Y --------------------------------------------------------
    mY = Y0 * np.exp(-beta * T) + lam * mu_J / beta * (1.0 - np.exp(-beta * T))
    vY = lam * mu_J**2 / beta * (1.0 - np.exp(-2.0 * beta * T))

    sX = np.sqrt(vX)
    sY = np.sqrt(vY)

    # --- log-spot Z = f(T)+X_T+Y_T ------------------------------------------
    fT = f(T)

    #  E[exp(Z_T)]  &  E[exp(2 Z_T)]  via mgf (indep. X and Y):
    #  M_X(θ) = exp(θ mX + ½ θ² vX)
    #  M_Y(θ) = ((1 - θ μ_J e^{-βT}) / (1 - θ μ_J))^{λ/β} ,  θ < 1/μ_J
    def _M(θ: float) -> float:
        # guard domain for exponential jumps
        if θ * mu_J >= 1.0:
            raise ValueError(f"θ={θ} violates θ μ_J < 1 (μ_J={mu_J}).")
        mx_part = np.exp(θ * mX + 0.5 * θ**2 * vX)
        y_part  = ((1.0 - θ * mu_J * np.exp(-beta * T)) /
                   (1.0 - θ * mu_J)) ** (lam / beta)
        return np.exp(θ * fT) * mx_part * y_part

    ES = _M(1.0)                      # 𝔼[S_T]
    E2 = _M(2.0)                      # 𝔼[S_T²]
    varS = E2 - ES**2
    sS = np.sqrt(varS)

    return ES, sS, mX, sX, mY, sY


def no_seasonal_function(t: float) -> float:
    return 0.0  # zero seasonality as in the paper

def simple_seasonal_function(t: float) -> float:
    # example seasonal function used by HHK in other paper.
    return np.log(100.0) + 0.5*np.sin(2.0 * np.pi * t)  
