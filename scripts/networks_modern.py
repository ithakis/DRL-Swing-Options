"""
Modernized neural networks for D4PG-QR-FRM algorithm.

Updated for PyTorch 2.7.1 and Python 3.11 with modern best practices:
- Type hints for better code clarity and IDE support
- Improved documentation with Google-style docstrings
- Better device handling and memory efficiency
- Modern PyTorch initialization methods
- Improved error handling and validation
- Code organization following PEP 8 and modern Python standards
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
        if hasattr(layer, 'weight'):
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


def weight_init_xavier(layers: List[nn.Module]) -> None:
    """Initialize weights using Xavier uniform initialization.
    
    Args:
        layers: List of PyTorch layers to initialize
    """
    for layer in layers:
        if hasattr(layer, 'weight'):
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)


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
        hidden_size: int = 256,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """Initialize the Actor network.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space  
            seed: Random seed for reproducibility
            hidden_size: Number of units in hidden layers
            device: Device to place the network on (cuda/cpu)
        """
        super().__init__()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Network architecture
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self.reset_parameters()
        
        # Move to device
        self.to(self.device)

    def reset_parameters(self) -> None:
        """Reset network parameters using custom initialization."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the actor network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_size)
            
        Returns:
            Action tensor of shape (batch_size, action_size) in range [-1, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Q-value) network for continuous control.
    
    Estimates Q-values for state-action pairs using a deep neural network.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_size: int = 256,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """Initialize the Critic network.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            seed: Random seed for reproducibility
            hidden_size: Number of units in hidden layers
            device: Device to place the network on (cuda/cpu)
        """
        super().__init__()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Network architecture
        self.fcs1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.reset_parameters()
        
        # Move to device
        self.to(self.device)

    def reset_parameters(self) -> None:
        """Reset network parameters using custom initialization."""
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """Forward pass through the critic network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_size)
            action: Input action tensor of shape (batch_size, action_size)
            
        Returns:
            Q-value tensor of shape (batch_size, 1)
        """
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
        n_cos: int = 64
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
        
        # Precompute pi values for cosine embeddings
        self.register_buffer(
            'pis',
            torch.tensor(
                [np.pi * i for i in range(1, self.n_cos + 1)],
                dtype=torch.float32
            ).view(1, 1, self.n_cos)
        )
        
        # Network architecture
        self.head = nn.Linear(self.action_size + self.input_shape, layer_size)
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, 1)
        
        # Move to device
        self.to(self.device)

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
        cos = torch.cos(taus * self.pis)
        
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
        x = F.relu(self.head(x))
        
        # Calculate cosine embeddings
        cos, taus = self.calc_cos(batch_size, num_tau)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = F.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size)
        
        # Element-wise multiplication and reshape
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)
        
        # Final layers
        x = F.relu(self.ff_1(x))
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


# Backward compatibility aliases
weight_init = weight_init_kaiming  # Maintain original function name
