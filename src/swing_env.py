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

try:  # SciPy is a project dependency, but keep a small fallback path.
    from scipy.optimize import brentq  # type: ignore
    from scipy.stats import multivariate_normal, norm  # type: ignore
except Exception:  # pragma: no cover
    brentq = None
    multivariate_normal = None
    norm = None


def approximate_Q_T(
    contract: SwingContract,
    hhk_params: Dict,
    *,
    utilization_factor: float = 1.0,
    max_pairs: int = 4096,
    seed: int = 0,
) -> float:
    """
    HHK-based approximation of the expected total exercised quantity E[Q_T].

    Model/contract regime targeted:
      - linear swing cashflows with Q_min=0, no refraction, bang–bang optimality,
      - HHK spot: S_t = exp(f(t) + X_t + Y_t), with Z_t := X_t + Y_t.

    Method:
      1) Approximate marginal ITM probabilities p(t_k)=P(S_{t_k}>K) via a Lugannani–Rice
         saddlepoint formula using the HHK CGF of Z_t.
      2) Approximate the correlated ITM count N = sum 1{S_{t_k}>K} by a Beta–Binomial
         with moment-matched mean p̄ and an exchangeable indicator correlation ρ_I
         (Gaussian-copula mapping from Corr(Z_{t_i},Z_{t_j})).
      3) E[Q_T] ≈ q_max * E[min(m, N)] (plus a last partial fill if Q_max/q_max ∉ N).
    """
    if contract.q_max <= 0.0 or contract.Q_max <= 0.0 or contract.n_rights <= 0:
        return 0.0
    if contract.strike <= 0.0:
        raise ValueError("contract.strike must be > 0 for log-threshold computations")
    if utilization_factor <= 0.0:
        return 0.0

    S0 = float(hhk_params["S0"])
    alpha = float(hhk_params["alpha"])
    sigma = float(hhk_params["sigma"])
    beta = float(hhk_params["beta"])
    lam = float(hhk_params["lam"])
    mu_J = float(hhk_params["mu_J"])
    f = hhk_params.get("f", lambda _t: 0.0)

    n = int(contract.n_rights)
    T = float(contract.maturity)
    if n <= 0 or T <= 0.0:
        return 0.0

    # Use the "k=1..n" convention (no decision at t=0) to avoid the t=0 degeneracy.
    dt = T / n
    t_grid = np.linspace(dt, T, n, dtype=np.float64)

    X0 = float(np.log(S0) - f(0.0))
    logK = float(np.log(contract.strike))

    def _phi(x: float) -> float:
        if norm is not None:
            return float(norm.pdf(x))
        return float(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))

    def _Phi(x: float) -> float:
        if norm is not None:
            return float(norm.cdf(x))
        import math

        return float(0.5 * (1.0 + math.erf(x / np.sqrt(2.0))))

    def _betaln(a: float, b: float) -> float:
        # log Beta(a,b) via lgamma; avoids requiring scipy.special.
        import math

        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def _log_choose(n_: int, k_: int) -> float:
        import math

        return math.lgamma(n_ + 1) - math.lgamma(k_ + 1) - math.lgamma(n_ - k_ + 1)

    def _brent_or_bisect(fn, lo: float, hi: float) -> float:
        import math

        if brentq is not None:
            return float(brentq(fn, lo, hi, maxiter=200))
        flo = fn(lo)
        fhi = fn(hi)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fmid = fn(mid)
            if abs(fmid) < 1e-10 or abs(hi - lo) < 1e-10:
                return float(mid)
            if flo * fmid <= 0.0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
        return float(0.5 * (lo + hi))

    def _p_itm(t: float) -> float:
        # Threshold in Z-space: Z_t > log(K) - f(t).
        x = logK - float(f(t))

        mX = X0 * np.exp(-alpha * t)
        vX = (sigma * sigma) * (1.0 - np.exp(-2.0 * alpha * t)) / (2.0 * alpha)

        if lam <= 0.0 or mu_J <= 0.0:
            # Pure Gaussian OU (exact normal tail).
            mean = float(mX)
            var = float(vX)
            std = float(np.sqrt(max(var, 1e-16)))
            return float(1.0 - _Phi((x - mean) / std))

        eb = float(np.exp(-beta * t))
        upper = (1.0 / mu_J) - 1e-10

        def K(th: float) -> float:
            if th >= upper:
                return np.inf
            mx_part = th * mX + 0.5 * th * th * vX
            a = th * mu_J * eb
            b = th * mu_J
            if a >= 1.0 or b >= 1.0:
                return np.inf
            jump_part = (lam / beta) * (np.log1p(-a) - np.log1p(-b))
            return float(mx_part + jump_part)

        def K1(th: float) -> float:
            a = th * mu_J * eb
            b = th * mu_J
            jump1 = (lam / beta) * (
                -(mu_J * eb) / (1.0 - a) + (mu_J) / (1.0 - b)
            )
            return float(mX + th * vX + jump1)

        def K2(th: float) -> float:
            a = th * mu_J * eb
            b = th * mu_J
            jump2 = (lam / beta) * (
                -((mu_J * eb) ** 2) / ((1.0 - a) ** 2) + (mu_J**2) / ((1.0 - b) ** 2)
            )
            return float(vX + jump2)

        # Saddlepoint: solve K'(th)=x. CGF is convex; K' is increasing.
        def g(th: float) -> float:
            return K1(th) - x

        g0 = g(0.0)
        if g0 == 0.0:
            th_hat = 0.0
        else:
            if g0 > 0.0:
                lo, hi = -1.0, 0.0
                while g(lo) > 0.0 and lo > -100.0:
                    lo *= 2.0
            else:
                lo, hi = 0.0, min(upper * 0.999, 50.0)
                while g(hi) < 0.0 and hi < upper * 0.999:
                    hi = min(hi * 2.0, upper * 0.999)
            th_hat = _brent_or_bisect(g, lo, hi)

        if th_hat == 0.0:
            # LR has a removable singularity at th=0; fall back to normal approx.
            mean = float(K1(0.0))
            var = float(max(K2(0.0), 1e-16))
            return float(1.0 - _Phi((x - mean) / np.sqrt(var)))

        K_hat = K(th_hat)
        K2_hat = max(K2(th_hat), 1e-16)

        # Lugannani–Rice: P(Z<=x) ≈ Φ(w) + φ(w)(1/w - 1/u)
        import math

        w2 = 2.0 * (th_hat * x - K_hat)
        w2 = max(w2, 0.0)
        w = math.copysign(math.sqrt(w2), th_hat)
        u = th_hat * math.sqrt(K2_hat)
        if abs(w) < 1e-8 or abs(u) < 1e-12:
            mean = float(K1(0.0))
            var = float(max(K2(0.0), 1e-16))
            return float(1.0 - _Phi((x - mean) / np.sqrt(var)))

        cdf = _Phi(w) + _phi(w) * (1.0 / w - 1.0 / u)
        return float(np.clip(1.0 - cdf, 0.0, 1.0))

    p = np.array([_p_itm(float(t)) for t in t_grid], dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    p_bar = float(np.mean(p))

    # HHK correlation structure for Z_t = X_t + Y_t.
    varX = (sigma * sigma) * (1.0 - np.exp(-2.0 * alpha * t_grid)) / (2.0 * alpha)
    varY = (lam * (mu_J**2) / beta) * (1.0 - np.exp(-2.0 * beta * t_grid)) if lam > 0 else 0.0
    varZ = np.maximum(varX + varY, 1e-16)

    # Map Corr(Z_i,Z_j) -> Corr(1{S_i>K},1{S_j>K}) via Gaussian copula at thresholds.
    if multivariate_normal is None or norm is None:
        rho_I = 0.0
    else:
        thr = norm.ppf(1.0 - p)  # latent N threshold: I=1{N>thr}
        pairs_total = n * (n - 1) // 2
        rng = np.random.default_rng(seed)
        if pairs_total <= max_pairs:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            flat = rng.choice(pairs_total, size=max_pairs, replace=False)
            pairs = []
            # Map flat index -> (i,j) over upper triangle.
            # For n<=250 this is fine; keep it simple.
            for idx in flat:
                i = 0
                remaining = int(idx)
                while remaining >= (n - i - 1):
                    remaining -= (n - i - 1)
                    i += 1
                j = i + 1 + remaining
                pairs.append((i, j))

        corr_sum = 0.0
        corr_cnt = 0
        for i, j in pairs:
            ti = float(t_grid[i])
            tj = float(t_grid[j])
            if ti > tj:
                ti, tj = tj, ti
                i, j = j, i
            covX = (sigma * sigma) * (np.exp(-alpha * (tj - ti)) - np.exp(-alpha * (tj + ti))) / (2.0 * alpha)
            covY = 0.0
            if lam > 0.0:
                varY_i = (lam * (mu_J**2) / beta) * (1.0 - np.exp(-2.0 * beta * ti))
                covY = np.exp(-beta * (tj - ti)) * varY_i
            covZ = float(covX + covY)
            rhoZ = float(np.clip(covZ / np.sqrt(varZ[i] * varZ[j]), -0.999, 0.999))

            cov = np.array([[1.0, rhoZ], [rhoZ, 1.0]], dtype=np.float64)
            Phi2 = float(multivariate_normal.cdf([thr[i], thr[j]], mean=[0.0, 0.0], cov=cov))
            joint = 1.0 - float(norm.cdf(thr[i])) - float(norm.cdf(thr[j])) + Phi2
            denom = float(np.sqrt(p[i] * (1.0 - p[i]) * p[j] * (1.0 - p[j])))
            if denom <= 0.0:
                continue
            corr_sum += (joint - float(p[i] * p[j])) / denom
            corr_cnt += 1
        rho_I = float(np.clip(corr_sum / max(corr_cnt, 1), 1e-6, 0.999))

    s = (1.0 - rho_I) / rho_I
    a = p_bar * s
    b = (1.0 - p_bar) * s

    m_full = int(contract.Q_max // contract.q_max)
    q_last = float(contract.Q_max - m_full * contract.q_max)

    # E[min(m_full,N)] under Beta–Binomial(n,a,b).
    e_min = 0.0
    p_gt = 0.0
    base = _betaln(a, b)
    for k in range(n + 1):
        log_pmf = _log_choose(n, k) + _betaln(k + a, (n - k) + b) - base
        pmf = float(np.exp(log_pmf))
        e_min += min(m_full, k) * pmf
        if k > m_full:
            p_gt += pmf

    q_T = contract.q_max * e_min + (q_last * p_gt if q_last > 1e-12 else 0.0)
    q_T = float(np.clip(q_T * utilization_factor, 0.0, contract.Q_max))
    return q_T


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
                 obs_dtype: Optional[np.dtype] = None,
                 enable_antithetic: bool = False,
                 partner_idx_arr: Optional[np.ndarray] = None):
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
        # H8: antithetic pairing.
        # If partner_idx_arr is provided, use it (it correctly maps stratified
        # indices to their antithetic partner's stratified index).
        # Otherwise fall back to XOR pairing (only valid when stratify=False).
        self.enable_antithetic = bool(enable_antithetic)
        self._partner_idx_arr = (
            np.asarray(partner_idx_arr, dtype=np.int64) if partner_idx_arr is not None else None
        )
        self._partner_path_idx = -1
        
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
        # H8: also expose the antithetic partner's next-state if enabled
        if self.enable_antithetic and self._partner_path_idx >= 0:
            info["next_state_antithetic"] = self._get_partner_observation(next_obs)
        return next_obs, total_reward, terminated, truncated, info

    def _get_partner_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """Build the antithetic partner's observation at self.current_step.

        Everything except (S - K, S, X, Y) is identical to the base obs because
        the contract-level state (Q exercised, time, days since exercise) is
        deterministic given (action, history) and unchanged by the jump-component
        antithetic pairing. We just swap in the partner's S, X, Y values.
        """
        partner = base_obs.copy()
        idx = int(self._partner_path_idx)
        step = int(self.current_step)
        if step >= self.S.shape[1]:
            step = self.S.shape[1] - 1
        S_partner = float(self.S[idx, step])
        X_partner = float(self.X[idx, step])
        Y_partner = float(self.Y[idx, step])
        partner[0] = S_partner - self.contract.strike
        partner[5] = S_partner
        partner[6] = X_partner
        partner[7] = Y_partner
        return partner
    
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
        # H8: identify antithetic partner path
        if self.enable_antithetic:
            n_paths = self.S.shape[0]
            if self._partner_idx_arr is not None and path_idx < len(self._partner_idx_arr):
                # Stratify-preserved partner mapping
                cand = int(self._partner_idx_arr[path_idx])
                self._partner_path_idx = cand if 0 <= cand < n_paths else path_idx
            else:
                # Legacy fallback: XOR pairing (valid only if stratify was off)
                self._partner_path_idx = (path_idx ^ 1) if (path_idx ^ 1) < n_paths else path_idx
        else:
            self._partner_path_idx = -1
        
        # Initialize episode state
        self.current_step = 0
        self.q_exercised = 0.0
        self.last_exercise_step = -1
        self.episode_return = 0.0
        
        # Calculate initial volatility
        # self.recent_volatility = self._calculate_recent_volatility(current_idx=0)
        
        return self._get_observation(), {}
