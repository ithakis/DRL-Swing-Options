"""
Modernized neural networks for D4PG-QR-FRM algorithm.

Updated for PyTorch 2.8+ and Python 3.11 with modern best practices:
- Type hints for better code clarity and IDE support
- Improved documentation with Google-style docstrings
- Modern PyTorch initialization methods with torch.compile support
- Improved error handling and validation
- Code organization following PEP 8 and modern Python standards
- Enhanced numerical stability and performance optimizations
"""

import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import calibration utility
try:
    from .utils_calibration import calibrate_actor_output
except ImportError:
    # Fallback if running as script vs module
    try:
        from utils_calibration import calibrate_actor_output
    except ImportError:
        calibrate_actor_output = None


# Modern PyTorch 2.x optimizations
def make_compilable(model: nn.Module) -> nn.Module:
    """Make a model compilable with torch.compile for PyTorch 2.x performance.
    
    Args:
        model: PyTorch model to make compilable
        
    Returns:
        Potentially compiled model for better performance
        
    Note:
        torch.compile may not be available in all environments,
        so we return the original model if compilation fails.
    """
    try:
        if hasattr(torch, 'compile'):
            # Type ignore for torch.compile return type compatibility
            return torch.compile(model)  # type: ignore
    except Exception:
        pass
    return model


def hidden_init(layer: nn.Linear) -> Tuple[float, float]:
    """Calculate uniform initialization bounds for a linear layer.
    
    Args:
        layer: PyTorch Linear layer
        
    Returns:
        Tuple of (lower_bound, upper_bound) for uniform initialization
        
    Note:
        This maintains the original behavior which uses fan_out (output dimension)
        rather than fan_in. Modern practice would typically use fan_in.
    """
    fan_out = layer.weight.data.size(0)  # Output dimension
    lim = 1.0 / np.sqrt(fan_out)
    return (-lim, lim)


def weight_init_kaiming(layers: List[nn.Module]) -> None:
    """Initialize weights using Kaiming (He) normal initialization.
    
    Args:
        layers: List of PyTorch layers to initialize
    """
    for layer in layers:
        if hasattr(layer, 'weight') and isinstance(layer.weight, torch.Tensor):
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


def weight_init_xavier(layers: List[nn.Module]) -> None:
    """Initialize weights using Xavier uniform initialization.
    
    Args:
        layers: List of PyTorch layers to initialize
    """
    for layer in layers:
        if hasattr(layer, 'weight') and isinstance(layer.weight, torch.Tensor):
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)


def weight_init_orthogonal(layers: List[nn.Module], gain: float = 1.0) -> None:
    """Initialize weights using orthogonal initialization.
    
    Orthogonal initialization preserves variance in both forward and backward passes,
    which is particularly beneficial for RL applications as it keeps gradients stable
    and improves exploration by maintaining isotropic output covariance.
    
    Args:
        layers: List of PyTorch layers to initialize
        gain: Scaling factor for the orthogonal matrix (√2 for ReLU, 1.0 for tanh)
    """
    for layer in layers:
        if hasattr(layer, 'weight') and isinstance(layer.weight, torch.Tensor):
            torch.nn.init.orthogonal_(layer.weight, gain=gain)
        if hasattr(layer, 'bias') and isinstance(layer.bias, torch.Tensor):
            torch.nn.init.zeros_(layer.bias)


def _build_activation(activation: str) -> Tuple[Callable[[], nn.Module], float]:
    """Return an activation factory and recommended gain for initialization."""
    act = activation.lower()
    if act == "relu":
        return lambda: nn.ReLU(inplace=True), math.sqrt(2.0)
    if act == "leaky_relu":
        negative_slope = 0.01
        gain = nn.init.calculate_gain("leaky_relu", negative_slope)
        return lambda: nn.LeakyReLU(negative_slope=negative_slope, inplace=True), gain
    if act == "silu":
        # torch init does not expose a SiLU-specific gain; use ReLU-equivalent
        return lambda: nn.SiLU(inplace=True), math.sqrt(2.0)
    if act == "tanh":
        return lambda: nn.Tanh(), nn.init.calculate_gain("tanh")
    raise ValueError(f"Unsupported activation '{activation}'. Choose from 'relu', 'leaky_relu', 'silu', or 'tanh'.")


def _normalize_init_method(init_method: Optional[str]) -> str:
    method = (init_method or "orthogonal").lower()
    if method == "kaiming":
        method = "he"
    if method not in {"orthogonal", "he"}:
        raise ValueError(f"Unsupported init_method '{init_method}'. Choose from 'orthogonal' or 'he'.")
    return method


def _init_linear(layer: nn.Linear, *, method: str, gain: float) -> None:
    if method == "orthogonal":
        torch.nn.init.orthogonal_(layer.weight, gain=gain)
    elif method == "he":
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    else:
        raise ValueError(f"Unsupported init_method '{method}'. Choose from 'orthogonal' or 'he'.")
    torch.nn.init.zeros_(layer.bias)

    torch.nn.init.zeros_(layer.bias)


class RobustNormalizer(nn.Module):
    """
    Robust Normalization using Median and IQR (Inter-Quartile Range).
    
    Transformation:
        x_hat = (x - median) / (IQR + epsilon)
        
    Why:
        Standard Z-score (mean/std) is sensitive to outliers (jumps).
        Robust statistics scale signals based on their 'typical' range,
        preventing rare jump events from squashing the distribution of normal events.
        
    Args:
        size: Number of features
        median: Initial estimate of median (can be updated)
        iqr: Initial estimate of IQR (can be updated)
        clip: Optional clipping range (e.g., 10.0) to bound extreme outliers
    """
    def __init__(self, size: int, median: float = 0.0, iqr: float = 1.0, clip: float = 10.0):
        super().__init__()
        self.register_buffer("median", torch.full((size,), median, dtype=torch.float32))
        self.register_buffer("iqr", torch.full((size,), iqr, dtype=torch.float32))
        self.clip = clip
        self.eps = 1e-6

    def forward(self, x: Tensor) -> Tensor:
        # Robust centering and scaling
        x_robust = (x - self.median) / (self.iqr + self.eps)
        if self.clip > 0:
            x_robust = torch.clamp(x_robust, -self.clip, self.clip)
        return x_robust


class HHKInputLayer(nn.Module):
    """
    Specialized Input Normalization Layer for HHK Swing Process.
    
    Applies domain-specific transformations to the 9-dim state vector:
    1. Spot Price (S_t): Log-Moneyness -> log(S_t / K)
    2. Factors (X_t, Y_t): Robust Scaling (Median/IQR)
    3. Time/Inventory: Pass-through (already in [0,1])
    
    State mapping (from swing_env.py):
    [0]: S - K (ignored/replaced by better features)
    [1]: Q_exercised (normalized)
    [2]: Q_remaining (normalized)
    [3]: T_maturity (normalized)
    [4]: t_current (normalized)
    [5]: S_t (Raw Spot) -> Converted to Log-Moneyness
    [6]: X_t (OU Factor) -> Robust Scaled
    [7]: Y_t (Jump Factor) -> Robust Scaled
    [8]: Days since exercise (normalized)
    """
    def __init__(self, state_dim: int = 9, strike: float = 100.0, 
                 x_median: float = 0.0, x_iqr: float = 1.0,
                 y_median: float = 0.0, y_iqr: float = 1.0):
        super().__init__()
        self.register_buffer("strike", torch.tensor(strike, dtype=torch.float32))
        
        # Robust scalers for X and Y factors
        self.x_norm = RobustNormalizer(1, median=x_median, iqr=x_iqr)
        self.y_norm = RobustNormalizer(1, median=y_median, iqr=y_iqr)

    def forward(self, state: Tensor) -> Tensor:
        """
        Transform state features IN-PLACE to preserve semantic order.
        
        Input/Output order (both 9-dim):
        [0]: S - K          -> UNCHANGED (kept for profitability gate)
        [1]: Q_exercised    -> UNCHANGED
        [2]: Q_remaining    -> UNCHANGED
        [3]: T_maturity     -> UNCHANGED
        [4]: t_current      -> UNCHANGED
        [5]: S_t (raw)      -> log(S_t / K) (Log-Moneyness)
        [6]: X_t (OU)       -> Robust Scaled X
        [7]: Y_t (Jump)     -> Robust Scaled Y
        [8]: Days since     -> UNCHANGED
        """
        # Clone to avoid modifying original tensor
        out = state.clone()
        
        # 1. Log-Moneyness: replace S_t at index 5
        s_t = state[..., 5:6]
        s_safe = torch.clamp(s_t, min=1e-4)
        out[..., 5:6] = torch.log(s_safe / self.strike)
        
        # 2. Robust-scale X_t at index 6
        out[..., 6:7] = self.x_norm(state[..., 6:7])
        
        # 3. Robust-scale Y_t at index 7
        out[..., 7:8] = self.y_norm(state[..., 7:8])
        
        # Indices 0-4 and 8 are already normalized, pass through unchanged
        return out


class _RMSNorm(nn.Module):
    """Minimal RMSNorm fallback for older PyTorch builds."""

    def __init__(self, normalized_shape: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def _build_norm(norm_type: str, hidden_size: int) -> nn.Module:
    norm = (norm_type or "layernorm").lower()
    if norm in {"layernorm", "ln", "layer_norm"}:
        return nn.LayerNorm(hidden_size)
    if norm in {"rmsnorm", "rms", "rms_norm"}:
        if hasattr(nn, "RMSNorm"):
            return nn.RMSNorm(hidden_size)  # type: ignore[attr-defined]
        return _RMSNorm(hidden_size)
    if norm in {"none", "identity", "no"}:
        return nn.Identity()
    raise ValueError("Unsupported norm_type. Choose from {'layernorm', 'rmsnorm', 'none'}.")


class Actor(nn.Module):
    """Actor (Policy) network for continuous control.
    
    Maps states to actions using a deep neural network with tanh activation
    to ensure actions are in the range [-1, 1].
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        action_output: str = "tanh01",
        target_action_mean: Optional[float] = 0.5,
        target_action_std: Optional[float] = math.sqrt(1.0 / 12.0),
        activation: str = "silu",
        norm_type: str = "layernorm",
        init_method: str = "orthogonal",
        input_preprocessor: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the Actor network.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space  
            seed: Random seed for reproducibility
            hidden_size: Number of units in hidden layers (default: 64 for a lightweight 2×64 policy)
            n_layers: Number of hidden layers (default: 2)
            device: Device to place the network on (cpu)
            action_output: Output activation. One of {"tanh", "tanh01", "sigmoid"}.
                "tanh01" maps tanh output from [-1, 1] to [0, 1] (default).
            target_action_mean: Optional target mean for initialization (used only if provided).
            target_action_std: Optional target std for initialization (used only if provided).
            activation: Hidden-layer activation ("silu" default; supports "relu" and "leaky_relu").
        """
        super().__init__()

        # CPU-only
        self.device = torch.device("cpu")
            
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        if n_layers < 1:
            raise ValueError("Actor requires at least one hidden layer")

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.action_output = action_output.lower()
        # Parse β for beta_sigmoid activation (default β=2.0)
        # Format: "beta_sigmoid" or "beta_sigmoid_X" where X is the temperature
        if self.action_output.startswith("beta_sigmoid"):
            parts = self.action_output.split("_")
            self._output_beta = float(parts[2]) if len(parts) > 2 else 2.0
        else:
            self._output_beta = 1.0  # unused for other activations
        self.target_action_mean = target_action_mean
        self.target_action_std = target_action_std
        self.activation_name = activation.lower()
        self._activation_factory, self._activation_gain = _build_activation(self.activation_name)
        self.norm_type = norm_type.lower()
        self.init_method = _normalize_init_method(init_method)
        self.input_preprocessor = input_preprocessor

        layers: List[nn.Sequential] = []
        input_dim = state_size
        for _ in range(n_layers):
            block = nn.Sequential(
                nn.Linear(input_dim, hidden_size, bias=True),
                _build_norm(self.norm_type, hidden_size),
                self._activation_factory()
            )
            layers.append(block)
            input_dim = hidden_size

        self.hidden_layers = nn.ModuleList(layers)
        self.fc4 = nn.Linear(input_dim, action_size)
        # Initialize weights
        self.reset_parameters()

        # --- Profitability constraint (hard gate + STE) ---
        # These are configured by the Agent/env wiring via set_profitability_params().
        # Defaults mirror SwingContract defaults (q in [0,1], no convex cost).
        self._profit_s_minus_k_index: int = 0
        self._profit_q_min: float = 0.0
        self._profit_q_max: float = 1.0
        self._profit_q_range: float = 1.0
        self._profit_c_cost: float = 0.0
        self._profit_gamma_cost: float = 1.0
        self._profitability_params_set: bool = False

        # Move to device
        self.to(self.device)
        
        # Store compilation state for PyTorch 2.x optimization
        self._compiled = False

    def reset_parameters(self) -> None:
        """Reset network parameters using D4PG-recommended initialization.
        
        Uses orthogonal initialization for hidden layers with an activation-appropriate
        gain and small uniform initialization for the final layer to start with
        small, centered actions near the data manifold.
        """
        # Orthogonal initialization for hidden layers with activation-appropriate gain
        # Only initialize Linear layers, not LayerNorm
        linear_layers = [block[0] for block in self.hidden_layers]  # index 0 = Linear
        for layer in linear_layers:
            if isinstance(layer, nn.Linear):
                _init_linear(layer, method=self.init_method, gain=self._activation_gain)

        # Small uniform initialization for final layer (actor output)
        torch.nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        torch.nn.init.zeros_(self.fc4.bias)
    
    def compile_for_performance(self) -> None:
        """Compile the model for better performance in PyTorch 2.x.
        
        Note:
            This is optional and may not work in all environments.
            Call this after model creation for potential speedups.
        """
        if not self._compiled:
            try:
                compiled_model = make_compilable(self)
                if compiled_model is not self:
                    # If compilation succeeded, we would need to replace self
                    # For now, just mark as compiled
                    self._compiled = True
                    print("✓ Actor model compiled for performance")
                else:
                    print("⚠ torch.compile not available, using standard model")
            except Exception as e:
                print(f"⚠ Compilation failed: {e}")

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the actor network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_size)
            
        Returns:
            Action tensor of shape (batch_size, action_size); range depends on
            action_output ("tanh": [-1, 1], "tanh01"/"sigmoid": [0, 1])
        """
        _, q_gated = self.forward_raw_and_gated(state)
        return q_gated

    def forward_preact(self, state: Tensor) -> Tensor:
        """Forward pass up to the final linear layer (pre-activation outputs)."""
        x = state
        if self.input_preprocessor is not None:
            x = self.input_preprocessor(x)
            
        for block in self.hidden_layers:
            x = block(x)
        return self.fc4(x)

    def forward_raw(self, state: Tensor) -> Tensor:
        """Forward pass returning the unconstrained action q_raw (pre profitability-gate)."""
        return self._apply_output_activation(self.forward_preact(state))

    def set_profitability_params(
        self,
        *,
        q_min: float,
        q_max: float,
        c_cost: float,
        gamma_cost: float,
        s_minus_k_index: int = 0,
    ) -> None:
        """Configure the hard profitability gate parameters.

        The environment uses normalized actions in [0,1] which are denormalized via:
            q_actual = q_min + q_norm * (q_max - q_min)

        The immediate (undiscounted) profit used for gating is:
            Pi(q_actual) = q_actual * relu(S - K) - c_cost * q_actual**gamma_cost
        """
        self._profit_s_minus_k_index = int(s_minus_k_index)
        self._profit_q_min = float(q_min)
        self._profit_q_max = float(q_max)
        self._profit_q_range = float(q_max) - float(q_min)
        self._profit_c_cost = float(c_cost)
        self._profit_gamma_cost = float(gamma_cost)
        self._profitability_params_set = True

    def forward_raw_and_gated(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Return both q_raw and the profitability-gated action q_gated (STE)."""
        q_raw = self.forward_raw(state)
        q_gated = self.apply_profitability_gate(q_raw=q_raw, state=state)
        return q_raw, q_gated

    def apply_profitability_gate(self, *, q_raw: Tensor, state: Tensor) -> Tensor:
        """Apply a projected profitability gate with a straight-through estimator (STE).

        Calculates the break-even quantity q_limit where Revenue = Cost.
        Projects q_raw to min(q_raw, q_limit) to ensure non-negative immediate profit.
        
        Backward: gradients flow as-if q=q_raw (STE via detach trick).
        """
        idx = self._profit_s_minus_k_index
        s_minus_k = state[..., idx: idx + 1]
        
        # Payoff per unit = max(S-K, 0)
        payoff_per_unit = torch.relu(s_minus_k)

        q_min = self._profit_q_min
        q_max = self._profit_q_max
        q_range = getattr(self, "_profit_q_range", q_max - q_min)
        c_cost = self._profit_c_cost
        gamma_cost = self._profit_gamma_cost

        # Fast paths for common no-cost case
        if c_cost == 0.0:
            if q_min >= 0.0 and self.action_output in ("tanh01", "sigmoid"):
                # If q >= 0 and no cost, profit >= 0 is guaranteed if (S-K) > 0.
                # If (S-K) == 0, profit is 0 regardless of q.
                # So we only need to mask if (S-K) < 0 (OTM), but payoff_per_unit handles that implicitly 
                # (profit would be 0).
                # However, strictly speaking, if S < K, profit is 0. 
                # To be consistent with "gate", we might want to force 0 if OTM?
                # Actually, standard practice for simple calls is just let it be.
                # But here we enforce "no negative profit".
                # If S < K, payoff is 0. Cost is 0. Profit is 0. So q is allowed.
                return q_raw
            
            # If q can be negative (not common here), or linear cost...
            # For c=0, break-even is effectively infinite if P > 0, or defined by P >= 0.
            return q_raw

        # Calculate Break-Even Quantity (q_limit) where P*q = c*q^gamma
        # Gamma = 1: P = c. If P > c, limit = q_max. Else limit = 0.
        # Gamma > 1: P/c = q^(gamma-1) => q_limit = (P/c)^(1/(gamma-1))
        
        if gamma_cost == 1.0:
            # Linear cost: Bang-bang feasibility
            limit_mask = (payoff_per_unit > c_cost).to(dtype=q_raw.dtype)
            # If P > c, limit is q_max. If P <= c, limit is 0 (assuming q_min=0)
            # We want to project q_raw. So if P > c, q_projected = q_raw (since q_raw <= 1.0 <= q_max_norm)
            # If P <= c, q_projected = 0.
            q_projected = q_raw * limit_mask
        else:
            # Convex cost: q_limit = (P/c)^(1/(gamma-1))
            # Avoid div by zero if c is epsilon
            safe_c = max(c_cost, 1e-9)
            inv_power = 1.0 / (gamma_cost - 1.0)
            
            # q_limit in contract units
            q_limit_abs = torch.pow(payoff_per_unit / safe_c, inv_power)
            
            # Convert to normalized space [0, 1]
            # q_norm = (q_abs - q_min) / q_range
            q_limit_norm = (q_limit_abs - q_min) / q_range
            
            # Clamp limit to valid normalized range is handled by min(q_raw, limit) 
            # effectively since q_raw <= 1. But limit could be negative if q_min > q_limit_abs.
            # So clamp lower bound.
            q_limit_norm = torch.clamp(q_limit_norm, 0.0, 1e9) # Upper bound open, effectively 1.0+
            
            # Project: q_new = min(q_raw, q_limit_norm)
            q_projected = torch.min(q_raw, q_limit_norm)

        # Straight-Through Estimator:
        # Forward pass uses q_projected.
        # Backward pass uses gradients from q_projected but passed through to q_raw 
        # (effectively identity if not clamped, or 1 if clamped? No, simple identity usually).
        # Actually standard STE for projection is q_raw + (q_proj - q_raw).detach()
        return q_raw + (q_projected - q_raw).detach()

    def _apply_output_activation(self, out: Tensor) -> Tensor:
        """Apply the configured output activation."""
        if self.action_output == "tanh":
            return torch.tanh(out)
        if self.action_output == "tanh01":
            return 0.5 * (torch.tanh(out) + 1.0)
        if self.action_output == "sigmoid":
            return torch.sigmoid(out)
        if self.action_output.startswith("beta_sigmoid"):
            # β-sigmoid: sigmoid(β * u) with tunable temperature β
            # Format: "beta_sigmoid" uses default β=2.0, or "beta_sigmoid_X" uses β=X
            # Provides softer gradient saturation than tanh01, improving late-stage learning
            beta = self._output_beta
            return torch.sigmoid(beta * out)
        raise ValueError(f"Unsupported action_output '{self.action_output}'. Expected one of 'tanh', 'tanh01', 'sigmoid', 'beta_sigmoid'.")


class FinanceInformedActor(Actor):
    """
    Finance-informed Actor that uses an analytic greedy baseline.
    The network outputs a residual correction to the immediate-profit-maximizing action.
    """

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass merging greedy baseline with NN correction."""
        # Get raw NN output (0.5 = neutral, >0.5 = more than greedy, <0.5 = less than greedy)
        q_nn = super().forward(state)

        if not self._profitability_params_set:
            return q_nn

        # Extract spot-strike (payoff) from state
        s_minus_k = state[..., self._profit_s_minus_k_index: self._profit_s_minus_k_index + 1]
        payoff = torch.relu(s_minus_k)

        c = self._profit_c_cost
        g = self._profit_gamma_cost
        q_min = self._profit_q_min
        q_max = self._profit_q_max
        q_range = self._profit_q_range

        # Calculate Greedy Action q* that maximizes: q(S-K) - c*q^g
        if c == 0.0:
            # Linear payoff, no cost: boundary solution
            q_greedy = torch.where(payoff > 0.0, q_max, q_min)
        elif g == 1.0:
            # Linear payoff, linear cost: boundary solution based on cost margin
            q_greedy = torch.where(payoff > c, q_max, q_min)
        else:
            # Convex cost: interior solution via FOC: (S-K) = c*g*q^(g-1)
            # q* = [(S-K)/(c*g)]^(1/(g-1))
            # Handle potential div-by-zero or negative payoff (though relu handles negative)
            q_star = (payoff / max(1e-9, c * g)).pow(1.0 / (g - 1.0))
            q_greedy = torch.clamp(q_star, q_min, q_max)

        # Normalize greedy action to [0, 1] scale
        q_greedy_norm = (q_greedy - q_min) / q_range

        # Combine: Final action = clamp(q_greedy_norm + (q_nn - 0.5), 0, 1)
        # This allows the NN to deviate from the greedy baseline by up to +/- 50%
        # of the total capacity to account for continuation value / opportunity cost.
        return torch.clamp(q_greedy_norm + (q_nn - 0.5), 0.0, 1.0)


class Critic(nn.Module):
    """Critic (Q-value) network for continuous control.
    
    Estimates Q-values for state-action pairs using a deep neural network.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        activation: str = "silu",
        norm_type: str = "layernorm",
        init_method: str = "orthogonal",
        input_preprocessor: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the Critic network.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            seed: Random seed for reproducibility
            hidden_size: Number of units in hidden layers (default: 64 for a lightweight 2×64 critic)
            n_layers: Number of hidden layers (default: 2)
            device: Device to place the network on (cpu)
            activation: Hidden-layer activation ("silu" default; supports "relu" and "leaky_relu").
        """
        super().__init__()

        # CPU-only
        self.device = torch.device("cpu")
            
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        if n_layers < 2:
            raise ValueError("Critic requires at least two hidden layers to integrate actions")

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.activation_name = activation.lower()
        self._activation_factory, self._activation_gain = _build_activation(self.activation_name)
        self.norm_type = norm_type.lower()
        self.init_method = _normalize_init_method(init_method)
        self.input_preprocessor = input_preprocessor
        self.input_preprocessor = input_preprocessor

        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size, bias=True),
            _build_norm(self.norm_type, hidden_size),
            self._activation_factory()
        )

        self.action_layer = nn.Sequential(
            nn.Linear(hidden_size + action_size, hidden_size, bias=True),
            _build_norm(self.norm_type, hidden_size),
            self._activation_factory()
        )

        self.post_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                _build_norm(self.norm_type, hidden_size),
                self._activation_factory()
            )
            for _ in range(n_layers - 2)
        )

        self.fc4 = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.reset_parameters()
        
        # Move to device
        self.to(self.device)
        
        # Store compilation state for PyTorch 2.x optimization
        self._compiled = False

    def reset_parameters(self) -> None:
        """Reset network parameters using D4PG-recommended initialization.
        
        Uses orthogonal initialization for hidden layers with activation-aware gain
        and specific initialization for the final layer to produce neutral
        Q-values initially, improving critic stability.
        """
        # Orthogonal initialization for hidden layers with activation-aware gain
        # Only initialize Linear layers, not LayerNorm
        linear_layers = [self.state_encoder[0], self.action_layer[0]]
        linear_layers.extend(block[0] for block in self.post_layers)
        for layer in linear_layers:
            if isinstance(layer, nn.Linear):
                _init_linear(layer, method=self.init_method, gain=self._activation_gain)

        # Initialize final layer to produce neutral Q-values
        # Small uniform initialization for the final critic layer
        torch.nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        torch.nn.init.zeros_(self.fc4.bias)

    def compile_for_performance(self) -> None:
        """Compile the model for better performance in PyTorch 2.x.
        
        Note:
            This is optional and may not work in all environments.
            Call this after model creation for potential speedups.
        """
        if not self._compiled:
            try:
                compiled_model = make_compilable(self)
                if compiled_model is not self:
                    # If compilation succeeded, we would need to replace self
                    # For now, just mark as compiled
                    self._compiled = True
                    print("✓ Critic model compiled for performance")
                else:
                    print("⚠ torch.compile not available, using standard model")
            except Exception as e:
                print(f"⚠ Compilation failed: {e}")

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """Forward pass through the critic network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_size)
            action: Input action tensor of shape (batch_size, action_size)
            
        Returns:
            Q-value tensor of shape (batch_size, 1)
        """
        if self.input_preprocessor is not None:
            state = self.input_preprocessor(state)
            
        xs = self.state_encoder(state)

        x = torch.cat((xs, action), dim=1)
        x = self.action_layer(x)

        for block in self.post_layers:
            x = block(x)

        return self.fc4(x)


# =====================================================================
# Fast non-NN function approximators (approximator study)
# ---------------------------------------------------------------------
# Drop-in replacements for the NN Actor/Critic that match the same forward
# signatures and reuse the profitability-gate STE + output activation.
# Each is a differentiable  CuratedFeatures -> feature-map -> linear-head
# model, so the D4PG/DDPG gradients (dQ/da for the actor update, dQ/dw and
# da/dtheta for SGD) flow through autograd exactly as before.  They are
# cheaper than the 2x64 SiLU+LayerNorm net and port trivially to C++
# (a fixed feature transform followed by a single BLAS gemv).
#
# The four contenders share one front-end (CuratedFeatures) and differ only
# in the feature map:
#   poly    -> Chebyshev tensor-product basis (interpretable, LSM-family)
#   rff     -> Random Fourier Features (frozen random projection + cosine)
#   rbf     -> Radial Basis Function network (localized Gaussian kernels)
#   mlp     -> tiny 1-hidden-layer MLP on the curated features (the "tiny_nn"
#              safety net: keeps universal approximation, still ~6-8x cheaper)
# =====================================================================

# State-vector indices (mirror src/transition_kernel.py OBS_IDX_* and
# src/agent_evaluation._build_state_batch).
_OBS_S_MINUS_K = 0
_OBS_Q_EXERCISED = 1
_OBS_Q_REMAINING = 2
_OBS_TTM = 3
_OBS_T_NORM = 4
_OBS_S = 5
_OBS_X = 6
_OBS_Y = 7
_OBS_DAYS = 8


class CuratedFeatures(nn.Module):
    """Domain-informed, fixed-scale feature builder shared by all approximators.

    Maps the raw 9-dim HHK swing state to a compact, roughly-standardized
    feature vector.  Scales are fixed: the HHK process is identical across the
    convex-cost regimes we study (only c_cost/gamma_cost vary), so fixed scales
    are valid everywhere and -- unlike running BatchNorm-style stats -- need no
    buffer synchronisation between the local and target networks.

    Output (use_cross=True) is 10-dim:
        [ log-moneyness, ttm, q_remaining, q_exercised, X, Y, days_since,
          logm*q_remaining, ttm*q_remaining, X*Y ]
    """

    def __init__(
        self,
        strike: float = 1.0,
        *,
        use_cross: bool = True,
        x_scale: float = 4.0,
        y_scale: float = 2.0,
        logm_scale: float = 2.857,
        clip: float = 5.0,
    ) -> None:
        super().__init__()
        self.strike = float(strike)
        self.use_cross = bool(use_cross)
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)
        self.logm_scale = float(logm_scale)
        self.clip = float(clip)
        self.out_dim = 7 + (3 if self.use_cross else 0)

    def forward(self, state: Tensor) -> Tensor:
        s = state[..., _OBS_S:_OBS_S + 1].clamp_min(1e-6)
        logm = torch.log(s / self.strike) * self.logm_scale
        ttm = (state[..., _OBS_TTM:_OBS_TTM + 1] - 0.5) * 2.0
        q_rem = (state[..., _OBS_Q_REMAINING:_OBS_Q_REMAINING + 1] - 0.5) * 2.0
        q_ex = (state[..., _OBS_Q_EXERCISED:_OBS_Q_EXERCISED + 1] - 0.5) * 2.0
        x = state[..., _OBS_X:_OBS_X + 1] * self.x_scale
        y = state[..., _OBS_Y:_OBS_Y + 1] * self.y_scale
        days = (state[..., _OBS_DAYS:_OBS_DAYS + 1] - 0.5) * 2.0
        feats = [logm, ttm, q_rem, q_ex, x, y, days]
        if self.use_cross:
            feats.append(logm * q_rem)
            feats.append(ttm * q_rem)
            feats.append(x * y)
        z = torch.cat(feats, dim=-1)
        return torch.clamp(z, -self.clip, self.clip)


def _chebyshev_stack(x: Tensor, degree: int) -> Tensor:
    """Stack Chebyshev polynomials T_1..T_degree per input feature.

    Args:
        x: input tensor with values in (-1, 1), shape (..., K).
        degree: highest Chebyshev order.
    Returns:
        Tensor of shape (..., K * degree) -- T_1, T_2, ..., T_degree for each feature.
    """
    cols = [x]
    t_prev = torch.ones_like(x)
    t_cur = x
    for _ in range(2, degree + 1):
        t_next = 2.0 * x * t_cur - t_prev
        cols.append(t_next)
        t_prev, t_cur = t_cur, t_next
    return torch.cat(cols, dim=-1)


class PolyFeatureMap(nn.Module):
    """Chebyshev (per-feature) polynomial basis on curated features.

    Inputs are squashed through tanh into (-1, 1) for Chebyshev stability, then
    expanded to degree d.  A leading constant gives the bias term.
    """

    def __init__(self, in_dim: int, degree: int = 3) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.degree = int(degree)
        self.out_dim = 1 + self.in_dim * self.degree

    def forward(self, z: Tensor) -> Tensor:
        xb = torch.tanh(z)
        cheb = _chebyshev_stack(xb, self.degree)
        bias = torch.ones(z.shape[:-1] + (1,), dtype=z.dtype, device=z.device)
        return torch.cat([bias, cheb], dim=-1)


class RFFFeatureMap(nn.Module):
    """Random Fourier Features: phi(z) = sqrt(2/D) * cos(Omega z + b).

    Omega ~ N(0, 1/lengthscale^2), b ~ U[0, 2pi).  Frozen by default (only the
    linear head trains); optionally learnable (arXiv 2112.03257).
    """

    def __init__(
        self,
        in_dim: int,
        n_features: int = 256,
        lengthscale: float = 1.0,
        learnable: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(n_features)
        g = torch.Generator().manual_seed(int(seed))
        omega = torch.randn(self.in_dim, self.out_dim, generator=g) / float(lengthscale)
        b = torch.rand(self.out_dim, generator=g) * (2.0 * math.pi)
        if learnable:
            self.omega = nn.Parameter(omega)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("omega", omega)
            self.register_buffer("b", b)
        self._scale = math.sqrt(2.0 / self.out_dim)

    def forward(self, z: Tensor) -> Tensor:
        return self._scale * torch.cos(z @ self.omega + self.b)


class RBFFeatureMap(nn.Module):
    """Radial Basis Function network: phi_i(z) = exp(-||z - c_i||^2 / 2 sigma^2).

    Centers are sampled once in the standardized curated-feature space (which is
    ~unit scale by construction); bandwidth is shared and optionally learnable.
    """

    def __init__(
        self,
        in_dim: int,
        n_centers: int = 128,
        bandwidth: float = 1.0,
        learnable_bandwidth: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(n_centers)
        g = torch.Generator().manual_seed(int(seed) + 1234)
        centers = torch.randn(self.out_dim, self.in_dim, generator=g)
        self.register_buffer("centers", centers)
        # ``bandwidth`` is a relative multiplier: the effective sigma scales with
        # sqrt(in_dim) so unit-variance inputs give O(1) activations at any dim
        # (expected squared distance between two unit-Gaussian points is 2*dim).
        eff_bw = max(float(bandwidth), 1e-6) * math.sqrt(max(self.in_dim, 1))
        log_bw = math.log(eff_bw)
        if learnable_bandwidth:
            self.log_bw = nn.Parameter(torch.tensor(log_bw))
        else:
            self.register_buffer("log_bw", torch.tensor(log_bw))

    def forward(self, z: Tensor) -> Tensor:
        bw = torch.exp(self.log_bw)
        d2 = torch.cdist(z, self.centers).pow(2)
        return torch.exp(-d2 / (2.0 * bw * bw))


class MLPFeatureMap(nn.Module):
    """Tiny 1-hidden-layer MLP feature map (the 'tiny_nn' contender front-end)."""

    def __init__(self, in_dim: int, width: int = 32, activation: str = "silu", seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        act_factory, gain = _build_activation(activation)
        linear = nn.Linear(int(in_dim), int(width))
        torch.nn.init.orthogonal_(linear.weight, gain=gain)
        torch.nn.init.zeros_(linear.bias)
        self.net = nn.Sequential(linear, act_factory())
        self.out_dim = int(width)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


def _build_feature_map(kind: str, in_dim: int, cfg: dict, seed: int) -> nn.Module:
    kind = (kind or "poly").lower()
    if kind == "poly":
        return PolyFeatureMap(in_dim, degree=int(cfg.get("degree", 3)))
    if kind == "rff":
        return RFFFeatureMap(
            in_dim,
            n_features=int(cfg.get("rff_dim", 256)),
            lengthscale=float(cfg.get("lengthscale", 1.0)),
            learnable=bool(cfg.get("learnable", False)),
            seed=seed,
        )
    if kind == "rbf":
        return RBFFeatureMap(
            in_dim,
            n_centers=int(cfg.get("rbf_centers", 128)),
            bandwidth=float(cfg.get("bandwidth", 1.0)),
            learnable_bandwidth=bool(cfg.get("learnable_bandwidth", False)),
            seed=seed,
        )
    if kind in ("mlp", "tiny_nn"):
        return MLPFeatureMap(
            in_dim,
            width=int(cfg.get("width", 32)),
            activation=str(cfg.get("activation", "silu")),
            seed=seed,
        )
    raise ValueError(f"Unknown feature_kind '{kind}'. Choose from poly, rff, rbf, mlp.")


class BasisActor(Actor):
    """Actor with a feature-map + linear head instead of an MLP stack.

    Subclasses Actor so the profitability-gate STE, output activation, noise
    interface, and ``set_profitability_params`` all work unchanged; only the
    pre-activation computation (``forward_preact``) is replaced.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        action_output: str = "tanh01",
        activation: str = "silu",
        norm_type: str = "layernorm",
        init_method: str = "orthogonal",
        input_preprocessor: Optional[nn.Module] = None,
        *,
        feature_kind: str = "poly",
        feature_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(
            state_size,
            action_size,
            seed,
            hidden_size=hidden_size,
            n_layers=n_layers,
            action_output=action_output,
            activation=activation,
            norm_type=norm_type,
            init_method=init_method,
            input_preprocessor=input_preprocessor,
        )
        cfg = dict(feature_cfg or {})
        self.feature_kind = (feature_kind or "poly").lower()
        self.curated = CuratedFeatures(
            strike=float(cfg.get("strike", 1.0)),
            use_cross=bool(cfg.get("use_cross", True)),
        )
        self.feature_map = _build_feature_map(self.feature_kind, self.curated.out_dim, cfg, seed)
        # Output layer keeps the name ``fc4`` so Agent.calibrate_bias (which scales
        # the actor's output weight/bias) works unchanged; ``head`` aliases it.
        del self.hidden_layers
        del self.fc4
        self.fc4 = nn.Linear(self.feature_map.out_dim, action_size)
        torch.nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        torch.nn.init.zeros_(self.fc4.bias)
        self.to(self.device)

    @property
    def head(self) -> nn.Module:
        return self.fc4

    def forward_preact(self, state: Tensor) -> Tensor:
        # CuratedFeatures consumes the RAW env state (fixed-scale normalization
        # built in), so any HHKInputLayer preprocessor is intentionally bypassed
        # to avoid double-transforming the spot/factor slots.
        return self.fc4(self.feature_map(self.curated(state)))


class BasisCritic(Critic):
    """Critic with a feature-map + linear head instead of an MLP stack.

    For poly we couple the (scaled) action explicitly via [phi_s, a*phi_s, a, a^2]
    so dQ/da is state-dependent (a DPG-compatible quadratic-in-action critic,
    cf. Silver et al. 2014 Thm 3).  For rff/rbf/mlp the scaled action is simply
    appended to the curated vector and the map handles the coupling.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        activation: str = "silu",
        norm_type: str = "layernorm",
        init_method: str = "orthogonal",
        input_preprocessor: Optional[nn.Module] = None,
        *,
        feature_kind: str = "poly",
        feature_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(
            state_size,
            action_size,
            seed,
            hidden_size=hidden_size,
            n_layers=n_layers,
            activation=activation,
            norm_type=norm_type,
            init_method=init_method,
            input_preprocessor=input_preprocessor,
        )
        cfg = dict(feature_cfg or {})
        self.feature_kind = (feature_kind or "poly").lower()
        self.curated = CuratedFeatures(
            strike=float(cfg.get("strike", 1.0)),
            use_cross=bool(cfg.get("use_cross", True)),
        )
        kc = self.curated.out_dim
        if self.feature_kind == "poly":
            self.feature_map = _build_feature_map("poly", kc, cfg, seed + 7)
            head_dim = 2 * self.feature_map.out_dim + 2
        else:
            self.feature_map = _build_feature_map(self.feature_kind, kc + 1, cfg, seed + 7)
            head_dim = self.feature_map.out_dim
        self.head = nn.Linear(head_dim, 1)
        torch.nn.init.uniform_(self.head.weight, -3e-3, 3e-3)
        torch.nn.init.zeros_(self.head.bias)
        del self.state_encoder
        del self.action_layer
        del self.post_layers
        del self.fc4
        self.to(self.device)

    def _features(self, state: Tensor, action: Tensor) -> Tensor:
        # See BasisActor.forward_preact: curated features use the RAW state, so
        # the input_preprocessor is bypassed by design.
        z = self.curated(state)
        a_scaled = (action - 0.5) * 2.0
        if self.feature_kind == "poly":
            phi_s = self.feature_map(z)
            return torch.cat([phi_s, a_scaled * phi_s, a_scaled, a_scaled * a_scaled], dim=-1)
        return self.feature_map(torch.cat([z, a_scaled], dim=-1))

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.head(self._features(state, action))



# Backward compatibility aliases
weight_init = weight_init_orthogonal  # Use orthogonal as default for D4PG
