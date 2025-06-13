"""
JIT-optimized versions of neural networks for D4PG
These networks are designed to be JIT-compiled for faster inference
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class JITActor(nn.Module):
    """JIT-optimized Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=256):
        """Initialize parameters and build model."""
        super(JITActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.fc1.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        self.fc1.weight.data.uniform_(-lim, lim)
        
        fan_in = self.fc2.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        self.fc2.weight.data.uniform_(-lim, lim)
        
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    @torch.jit.script_method
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class JITCritic(nn.Module):
    """JIT-optimized Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=256):
        """Initialize parameters and build model."""
        super(JITCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.fc1.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        self.fc1.weight.data.uniform_(-lim, lim)
        
        fan_in = self.fc2.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        self.fc2.weight.data.uniform_(-lim, lim)
        
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    @torch.jit.script_method
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = torch.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def create_jit_networks(actor_network, critic_network, state_size, action_size, device):
    """
    Create JIT-compiled versions of the networks for faster inference
    
    Args:
        actor_network: Original actor network
        critic_network: Original critic network  
        state_size: State dimension
        action_size: Action dimension
        device: Device to run on
        
    Returns:
        tuple: (jit_actor, jit_critic) or (None, None) if compilation fails
    """
    try:
        # Copy state dict from original networks
        jit_actor = JITActor(state_size, action_size, 0).to(device)
        jit_actor.load_state_dict(actor_network.state_dict())
        
        # Only create JIT critic for non-distributional networks
        jit_critic = None
        if hasattr(critic_network, '__class__') and 'IQN' not in critic_network.__class__.__name__:
            jit_critic = JITCritic(state_size, action_size, 0).to(device)
            jit_critic.load_state_dict(critic_network.state_dict())
        
        # Compile the networks
        jit_actor.eval()
        if jit_critic is not None:
            jit_critic.eval()
            
        sample_state = torch.randn(1, state_size).to(device)
        sample_action = torch.randn(1, action_size).to(device)
        
        # Trace the networks
        jit_actor = torch.jit.trace(jit_actor, sample_state)
        if jit_critic is not None:
            jit_critic = torch.jit.trace(jit_critic, (sample_state, sample_action))
        
        print("JIT compilation successful")
        return jit_actor, jit_critic
        
    except Exception as e:
        print(f"JIT compilation failed: {e}")
        return None, None
