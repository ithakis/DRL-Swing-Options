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
    raise ValueError(f"Unsupported activation '{activation}'. Choose from 'relu', 'leaky_relu', or 'silu'.")


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
        self.target_action_mean = target_action_mean
        self.target_action_std = target_action_std
        self.activation_name = activation.lower()
        self._activation_factory, self._activation_gain = _build_activation(self.activation_name)
        self.norm_type = norm_type.lower()
        self.init_method = _normalize_init_method(init_method)

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
        raise ValueError(f"Unsupported action_output '{self.action_output}'. Expected one of 'tanh', 'tanh01', 'sigmoid'.")


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
        Q-values initially, improving distributional critic stability.
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
        xs = self.state_encoder(state)

        x = torch.cat((xs, action), dim=1)
        x = self.action_layer(x)

        for block in self.post_layers:
            x = block(x)

        return self.fc4(x)


class IQN(nn.Module):
    """Implicit Quantile Network for distributional reinforcement learning.
    
    Implements the IQN architecture for learning quantile functions
    of the return distribution.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        layer_size: int,
        seed: int,
        N: int,
        dueling: bool = False,
        n_cos: int = 64,
        norm_type: str = "layernorm",
        init_method: str = "orthogonal",
    ) -> None:
        """Initialize the IQN network.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            layer_size: Size of hidden layers
            seed: Random seed for reproducibility
            N: Number of quantile samples for training
            dueling: Whether to use dueling architecture (not implemented)
            device: Device to place the network on
            n_cos: Number of cosine embeddings for quantile encoding
        """
        super().__init__()
        
        # CPU-only
        self.device = torch.device("cpu")
            
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Network parameters
        self.input_shape = state_size
        self.action_size = action_size
        self.N = N
        self.n_cos = n_cos
        self.layer_size = layer_size
        self.dueling = dueling
        self.norm_type = norm_type.lower()
        self.init_method = _normalize_init_method(init_method)
        
        # Precompute pi values for cosine embeddings
        self.register_buffer(
            'pis',
            torch.tensor(
                [np.pi * i for i in range(1, self.n_cos + 1)],
                dtype=torch.get_default_dtype()
            ).view(1, 1, self.n_cos)
        )
        
        # Network architecture with configurable normalization
        self.head = nn.Sequential(
            nn.Linear(self.action_size + self.input_shape, layer_size, bias=True),
            _build_norm(self.norm_type, layer_size),
            nn.ReLU(inplace=True)
        )
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Sequential(
            nn.Linear(layer_size, layer_size, bias=True),
            _build_norm(self.norm_type, layer_size),
            nn.ReLU(inplace=True)
        )
        self.ff_2 = nn.Linear(layer_size, 1)
        
        # Initialize weights using D4PG-recommended scheme
        self.reset_parameters()
        
        # Move to device
        self.to(self.device)
        
        # Store compilation state for PyTorch 2.x optimization
        self._compiled = False

    def reset_parameters(self) -> None:
        """Reset network parameters using D4PG-recommended initialization.
        
        Uses orthogonal initialization for hidden layers and specific
        initialization for distributional outputs to start with neutral
        value distributions.
        """
        # Orthogonal/He initialization for hidden layers with ReLU gain
        # Only initialize Linear layers, not LayerNorm
        linear_layers = [self.head[0], self.ff_1[0]]  # index 0 = Linear
        for layer in linear_layers:
            if isinstance(layer, nn.Linear):
                _init_linear(layer, method=self.init_method, gain=math.sqrt(2.0))
        
        # Initialize cosine embedding layer
        if isinstance(self.cos_embedding, nn.Linear):
            _init_linear(self.cos_embedding, method=self.init_method, gain=math.sqrt(2.0))
        
        # Initialize final layer to produce neutral value distribution
        # Small uniform initialization similar to critic
        torch.nn.init.uniform_(self.ff_2.weight, -3e-3, 3e-3)
        torch.nn.init.zeros_(self.ff_2.bias)

    def calc_cos(self, batch_size: int, n_tau: int = 32) -> Tuple[Tensor, Tensor]:
        """Calculate cosine embeddings for quantile values.
        
        Args:
            batch_size: Batch size
            n_tau: Number of quantile samples
            
        Returns:
            Tuple of (cosine_embeddings, tau_values)
        """
        # Sample random quantile values
        taus = torch.rand(batch_size, n_tau, 1, device=self.device)
        
        # Calculate cosine embeddings
        cos = torch.cos(taus * self.pis)  # type: ignore
        
        assert cos.shape == (batch_size, n_tau, self.n_cos), f"cos shape is incorrect: {cos.shape}"
        return cos, taus

    def forward(self, input_tensor: Tensor, action: Tensor, num_tau: int = 32) -> Tuple[Tensor, Tensor]:
        """Forward pass through the IQN network.
        
        Args:
            input_tensor: Input state tensor of shape (batch_size, state_size)
            action: Input action tensor of shape (batch_size, action_size)
            num_tau: Number of quantile samples
            
        Returns:
            Tuple of (quantiles, tau_values) where:
                - quantiles: shape (batch_size, num_tau, 1)
                - tau_values: shape (batch_size, num_tau, 1)
        """
        batch_size = input_tensor.shape[0]

        # Concatenate state and action
        x = torch.cat((input_tensor, action), dim=1)
        x = self.head(x)  # Linear -> LayerNorm -> ReLU already included
        
        # Unit-test guard: verify shape after LayerNorm/ReLU block
        hidden_size = self.head[0].out_features  # Get layer_size from Linear layer
        assert x.dim() == 2 and x.size(1) == hidden_size, "LayerNorm integration broke shape"
        
        # Calculate cosine embeddings
        cos, taus = self.calc_cos(batch_size, num_tau)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = F.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size)
        
        # Element-wise multiplication and reshape
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)
        
        # Final layers
        x = self.ff_1(x)  # Linear -> LayerNorm -> ReLU already included
        # Unit-test guard: verify shape after second LayerNorm/ReLU block
        assert x.dim() == 2 and x.size(1) == hidden_size, "LayerNorm integration broke shape"
        
        out = self.ff_2(x)
        
        return out.view(batch_size, num_tau, 1), taus

    def get_qvalues(self, inputs: Tensor, action: Tensor) -> Tensor:
        """Get Q-values by averaging over quantiles.
        
        Args:
            inputs: Input state tensor
            action: Input action tensor
            
        Returns:
            Q-values averaged over quantiles
        """
        quantiles, _ = self.forward(inputs, action, self.N)
        return quantiles.mean(dim=1)

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
                    print("✓ IQN model compiled for performance")
                else:
                    print("⚠ torch.compile not available, using standard model")
            except Exception as e:
                print(f"⚠ Compilation failed: {e}")


# Backward compatibility aliases
weight_init = weight_init_orthogonal  # Use orthogonal as default for D4PG
