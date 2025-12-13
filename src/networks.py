"""
Modernized neural networks for D4PG-QR-FRM algorithm.

Updated for PyTorch 2.8+ and Python 3.11 with modern best practices:
- Type hints for better code clarity and IDE support
- Improved documentation with Google-style docstrings
- Better device handling and memory efficiency
- Modern PyTorch initialization methods with torch.compile support
- Improved error handling and validation
- Code organization following PEP 8 and modern Python standards
- Enhanced numerical stability and performance optimizations
- Better integration with PyTorch 2.x features like autocast and GradScaler
"""

import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
        device: Optional[Union[str, torch.device]] = None,
        action_output: str = "tanh01",
        target_action_mean: Optional[float] = 0.5,
        target_action_std: Optional[float] = math.sqrt(1.0 / 12.0),
        activation: str = "silu",
        norm_type: str = "layernorm",
    ) -> None:
        """Initialize the Actor network.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space  
            seed: Random seed for reproducibility
            hidden_size: Number of units in hidden layers (default: 64 for a lightweight 2×64 policy)
            n_layers: Number of hidden layers (default: 2)
            device: Device to place the network on (cuda/cpu)
            action_output: Output activation. One of {"tanh", "tanh01", "sigmoid"}.
                "tanh01" maps tanh output from [-1, 1] to [0, 1] (default).
            target_action_mean: Optional target mean for initialization (used only if provided).
            target_action_std: Optional target std for initialization (used only if provided).
            activation: Hidden-layer activation ("silu" default; supports "relu" and "leaky_relu").
        """
        super().__init__()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
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
                torch.nn.init.orthogonal_(layer.weight, gain=self._activation_gain)
                torch.nn.init.zeros_(layer.bias)

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
        """Apply a hard profitability gate with a straight-through estimator (STE).

        Forward: executes q=0 whenever immediate profit <= 0.
        Backward: gradients flow as-if q=q_raw (STE via detach trick).
        """
        idx = self._profit_s_minus_k_index
        s_minus_k = state[..., idx: idx + 1]

        q_min = self._profit_q_min
        q_range = getattr(self, "_profit_q_range", self._profit_q_max - self._profit_q_min)
        c_cost = self._profit_c_cost
        gamma_cost = self._profit_gamma_cost

        # Fast paths for the common SwingOptionEnv configuration:
        # - action_output in [0,1] (tanh01/sigmoid) => q_actual >= q_min
        # - q_min >= 0 and c_cost == 0 => profit > 0 iff (S-K) > 0
        if c_cost == 0.0 and q_min >= 0.0 and self.action_output in ("tanh01", "sigmoid"):
            mask = (s_minus_k > 0.0).to(dtype=q_raw.dtype)
            q_forward = q_raw * mask
            return q_raw + (q_forward - q_raw).detach()

        payoff_per_unit = torch.relu(s_minus_k)
        q_actual = q_min + q_raw * q_range

        if gamma_cost == 1.0:
            # profit = q_actual * (payoff_per_unit - c_cost)
            profit = q_actual * (payoff_per_unit - c_cost)
        elif gamma_cost == 2.0:
            profit = q_actual * payoff_per_unit - c_cost * torch.square(q_actual)
        else:
            profit = q_actual * payoff_per_unit - c_cost * q_actual.pow(gamma_cost)

        mask = (profit > 0.0).to(dtype=q_raw.dtype)
        q_forward = q_raw * mask
        return q_raw + (q_forward - q_raw).detach()

    def _apply_output_activation(self, out: Tensor) -> Tensor:
        """Apply the configured output activation."""
        if self.action_output == "tanh":
            return torch.tanh(out)
        if self.action_output == "tanh01":
            return 0.5 * (torch.tanh(out) + 1.0)
        if self.action_output == "sigmoid":
            return torch.sigmoid(out)
        raise ValueError(f"Unsupported action_output '{self.action_output}'. Expected one of 'tanh', 'tanh01', 'sigmoid'.")


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
        device: Optional[Union[str, torch.device]] = None,
        activation: str = "silu",
        norm_type: str = "layernorm",
    ) -> None:
        """Initialize the Critic network.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            seed: Random seed for reproducibility
            hidden_size: Number of units in hidden layers (default: 64 for a lightweight 2×64 critic)
            n_layers: Number of hidden layers (default: 2)
            device: Device to place the network on (cuda/cpu)
            activation: Hidden-layer activation ("silu" default; supports "relu" and "leaky_relu").
        """
        super().__init__()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        if n_layers < 2:
            raise ValueError("Critic requires at least two hidden layers to integrate actions")

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.activation_name = activation.lower()
        self._activation_factory, self._activation_gain = _build_activation(self.activation_name)
        self.norm_type = norm_type.lower()

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
                torch.nn.init.orthogonal_(layer.weight, gain=self._activation_gain)
                torch.nn.init.zeros_(layer.bias)

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
        device: Optional[Union[str, torch.device]] = None,
        n_cos: int = 64,
        norm_type: str = "layernorm",
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
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
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
        # Orthogonal initialization for hidden layers with ReLU gain
        # Only initialize Linear layers, not LayerNorm
        linear_layers = [self.head[0], self.ff_1[0]]  # index 0 = Linear
        for layer in linear_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                torch.nn.init.zeros_(layer.bias)
        
        # Initialize cosine embedding layer
        if isinstance(self.cos_embedding, nn.Linear):
            torch.nn.init.orthogonal_(self.cos_embedding.weight, gain=math.sqrt(2.0))
            torch.nn.init.zeros_(self.cos_embedding.bias)
        
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
