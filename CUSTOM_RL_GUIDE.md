# D4PG-QR-FRM Custom RL Problem Adaptation Guide

## Overview

This guide explains how to adapt the D4PG-QR-FRM codebase to solve your own reinforcement learning problems. The codebase is designed with a modular architecture that makes it easy to integrate custom environments while leveraging the advanced D4PG algorithm with distributional learning, prioritized experience replay, and other state-of-the-a# Enable all advanced features
python run.py -env "YourEnv-v0" \
  -per 1 \              # Prioritized Experience Replay
  -munchausen 1 \       # Munchausen RL
  -iqn 1 \              # Distributional learning with IQN
  -nstep 5 \            # 5-step bootstrapping
  -info "advanced_experiment"sions.

## Table of Contents

1. [Understanding the Codebase Architecture](Here's how Munchausen RL applies to this specific problem setting.understanding-the-codebase-architecture)
2. [Environment Requirements](#environment-requirements)
3. [Creating Your Custom Environment](#creating-your-custom-environment)
4. [Integration Approaches](#integration-approaches)
5. [Required Code Changes](#required-code-changes)
6. [Configuration and Hyperparameters](#configuration-and-hyperparameters)
7. [Training and Evaluation](#training-and-evaluation)
8. [Advanced Customizations](#advanced-customizations)
9. [Troubleshooting](#troubleshooting)

---

## Understanding the Codebase Architecture

### Core Components

The D4PG-QR-FRM codebase follows a clean modular architecture:

```
D4PG-QR-FRM/
├── run.py              # Main training script
├── enjoy.py            # Policy evaluation script
├── scripts/
│   ├── agent.py        # D4PG Agent implementation
│   ├── networks.py     # Actor, Critic, and IQN networks
│   ├── replay_buffer.py # Experience replay (standard & prioritized)
│   └── MultiPro.py    # Vectorized environments
└── runs/              # Model storage and logs
```

### Key Features Available

- **Distributional Learning**: Uses IQN (Implicit Quantile Networks) for value distribution
- **Prioritized Experience Replay (PER)**: Importance sampling for efficient learning
- **Munchausen RL**: Entropy-regularized policy improvement
- **N-Step Bootstrapping**: Multi-step returns for faster value propagation
- **Vectorized Environments**: Parallel training support
- **Performance Optimizations**: torch.compile, mixed precision, CPU/GPU optimization

---

## Environment Requirements

### Gymnasium Interface Compatibility

Your custom environment must implement the standard Gymnasium interface:

```python
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

class CustomEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        # Reset environment to initial state
        # Return: (observation, info)
        return observation, info
    
    def step(self, action):
        # Execute action and return results
        # Return: (observation, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        # Optional: visualization
        pass
    
    def close(self):
        # Cleanup resources
        pass
```

### Critical Requirements

1. **Continuous Action Space**: D4PG is designed for continuous control
   - Use `gym.spaces.Box` for action space
   - Actions should typically be bounded (e.g., [-1, 1])

2. **State Representation**: 
   - Flat observation vector (1D numpy array)
   - Use `gym.spaces.Box` with appropriate bounds

3. **Reward Structure**:
   - Dense rewards work better than sparse rewards
   - Consider reward shaping for complex tasks

4. **Episode Termination**:
   - Implement proper `terminated` (task completion) vs `truncated` (time limit) logic

---

## Creating Your Custom Environment

### Option 1: Standalone Environment File

Create a new file `my_custom_env.py`:

```python
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class MyCustomEnvironment(gym.Env):
    """
    Custom environment for your specific RL problem.
    
    Example: A simple 2D navigation task
    """
    
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.max_steps = 200
        self.target_threshold = 0.1
        
        # Define spaces
        # State: [x, y, target_x, target_y, velocity_x, velocity_y]
        self.observation_space = Box(
            low=-10.0, 
            high=10.0, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # Action: [acceleration_x, acceleration_y]
        self.action_space = Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize state
        self.position = np.random.uniform(-5, 5, 2)
        self.velocity = np.zeros(2)
        self.target = np.random.uniform(-5, 5, 2)
        self.step_count = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        # Apply action (acceleration)
        acceleration = np.clip(action, -1, 1)
        self.velocity += acceleration * 0.1
        self.velocity = np.clip(self.velocity, -2, 2)  # Velocity limits
        
        # Update position
        self.position += self.velocity * 0.1
        self.position = np.clip(self.position, -10, 10)  # Boundary limits
        
        # Calculate reward
        distance = np.linalg.norm(self.position - self.target)
        reward = -distance  # Negative distance as reward
        
        # Check termination conditions
        terminated = distance < self.target_threshold
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        # Add bonus for reaching target
        if terminated:
            reward += 100.0
        
        observation = self._get_observation()
        info = {'distance': distance, 'steps': self.step_count}
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        return np.concatenate([
            self.position,
            self.target,
            self.velocity
        ]).astype(np.float32)
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Position: {self.position}, Target: {self.target}, Distance: {np.linalg.norm(self.position - self.target):.3f}")

# Register the environment
gym.register(
    id='MyCustomEnv-v0',
    entry_point='my_custom_env:MyCustomEnvironment',
)
```

### Option 2: Wrapper for Existing Environment

If you want to modify an existing environment:

```python
import gymnasium as gym
import numpy as np

class CustomEnvironmentWrapper(gym.Wrapper):
    """
    Wrapper to modify existing environments.
    Example: Adding noise to observations or modifying rewards.
    """
    
    def __init__(self, env_name):
        base_env = gym.make(env_name)
        super().__init__(base_env)
        
        # Modify spaces if needed
        # self.observation_space = Box(...)
        # self.action_space = Box(...)
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # Modify observation if needed
        return self._modify_observation(observation), info
    
    def step(self, action):
        # Modify action if needed
        modified_action = self._modify_action(action)
        
        observation, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # Modify observation and reward
        modified_observation = self._modify_observation(observation)
        modified_reward = self._modify_reward(reward, observation, action)
        
        return modified_observation, modified_reward, terminated, truncated, info
    
    def _modify_observation(self, observation):
        # Example: Add noise to observations
        noise = np.random.normal(0, 0.01, observation.shape)
        return observation + noise
    
    def _modify_action(self, action):
        # Example: Scale actions
        return action
    
    def _modify_reward(self, reward, observation, action):
        # Example: Add action penalty
        action_penalty = -0.001 * np.sum(np.square(action))
        return reward + action_penalty
```

---

## Integration Approaches

### Approach 1: Direct Integration (Recommended)

Modify `run.py` to use your custom environment directly:

```python
# In run.py, replace the environment creation:

# Original code:
# temp_env = gym.make(args.env)

# Your custom code:
if args.env == "MyCustomEnv-v0":
    from my_custom_env import MyCustomEnvironment
    temp_env = MyCustomEnvironment()
    train_env = MyCustomEnvironment()
    eval_env = MyCustomEnvironment()
else:
    temp_env = gym.make(args.env)
    train_env = gym.make(args.env)
    eval_env = gym.make(args.env)
```

### Approach 2: Registration Method

Register your environment with Gymnasium:

```python
# In your environment file or at the top of run.py
import gymnasium as gym
from my_custom_env import MyCustomEnvironment

gym.register(
    id='MyCustomEnv-v0',
    entry_point='my_custom_env:MyCustomEnvironment',
    max_episode_steps=200,
)

# Then use normally:
# python run.py -env "MyCustomEnv-v0"
```

### Approach 3: Factory Pattern

Create an environment factory:

```python
# Create env_factory.py
import gymnasium as gym

def create_environment(env_name, **kwargs):
    if env_name == "MyCustomEnv-v0":
        from my_custom_env import MyCustomEnvironment
        return MyCustomEnvironment(**kwargs)
    elif env_name == "MyWrappedEnv-v0":
        from my_custom_env import CustomEnvironmentWrapper
        return CustomEnvironmentWrapper("Pendulum-v1")
    else:
        return gym.make(env_name)
```

---

## Required Code Changes

### 1. Modify `run.py` (Minimal Changes)

The main script requires minimal modifications:

```python
# Add at the top of run.py (after imports):
try:
    from my_custom_env import MyCustomEnvironment
    import gymnasium as gym
    gym.register(
        id='MyCustomEnv-v0',
        entry_point='my_custom_env:MyCustomEnvironment',
    )
except ImportError:
    print("Custom environment not found, using standard environments only")

# The rest of the code remains unchanged!
# The agent automatically adapts to your environment's state/action dimensions
```

### 2. Optional: Modify `enjoy.py` for Evaluation

```python
# Add the same import and registration code to enjoy.py
# if you want to evaluate your trained models
```

### 3. No Changes Needed in Core Files

The beauty of this codebase is that **NO changes are required** in:
- `scripts/agent.py`
- `scripts/networks.py`
- `scripts/replay_buffer.py`

The agent automatically adapts to your environment's dimensions!

---

## Configuration and Hyperparameters

### Default Configuration

The codebase works well with default hyperparameters for most environments:

```bash
python run.py -env "MyCustomEnv-v0" -info "my_experiment"
```

### Common Hyperparameter Adjustments

For your custom environment, you might need to adjust:

```bash
# Learning rates (if learning is unstable)
python run.py -env "MyCustomEnv-v0" -lr_a 1e-4 -lr_c 1e-3

# Network size (for complex environments)
python run.py -env "MyCustomEnv-v0" -layer_size 256

# Training duration
python run.py -env "MyCustomEnv-v0" -frames 100000

# Batch size and replay buffer
python run.py -env "MyCustomEnv-v0" -batch_size 128 -max_replay_size 1000000

# Evaluation frequency
python run.py -env "MyCustomEnv-v0" -eval_every 5000 -eval_runs 10
```

### Algorithm Extensions

Enable advanced features for better performance:

```bash
# Enable all advanced features
python run.py -env "MyCustomEnv-v0" \
  -per 1 \              # Prioritized Experience Replay
  -munchausen 1 \       # Munchausen RL
  -iqn 1 \              # Distributional learning with IQN
  -nstep 5 \            # 5-step bootstrapping
  -icm 1 \              # Intrinsic Curiosity Module
  -info "advanced_experiment"
```

### Environment-Specific Tuning

Create a configuration file for your environment:

```python
# config_my_env.py
MY_ENV_CONFIG = {
    'frames': 50000,
    'lr_a': 1e-4,
    'lr_c': 1e-3,
    'layer_size': 128,
    'batch_size': 64,
    'tau': 1e-3,
    'gamma': 0.99,
    'per': True,
    'munchausen': True,
    'iqn': True,
    'nstep': 3,
    'eval_every': 2000,
    'eval_runs': 5
}
```

---

## Training and Evaluation

### Training Your Custom Environment

```bash
# Basic training
python run.py -env "MyCustomEnv-v0" -info "baseline"

# Advanced training with all features
python run.py -env "MyCustomEnv-v0" \
  -frames 100000 \
  -per 1 -munchausen 1 -iqn 1 -nstep 5 \
  -layer_size 256 -batch_size 128 \
  -eval_every 5000 -eval_runs 10 \
  -info "advanced_training"

# Multi-environment parallel training (if your env supports it)
python run.py -env "MyCustomEnv-v0" -w 4 -info "parallel"
```

### Monitoring Training Progress

```bash
# Start TensorBoard
tensorboard --logdir=runs

# View metrics at: http://localhost:6006
```

Key metrics to monitor:
- **Reward**: Episode return over time
- **Critic_loss**: Value function learning progress
- **Actor_loss**: Policy learning progress
- **Collection_Progress**: Initial experience collection (if using min_replay_size)

### Evaluating Trained Models

```bash
# Evaluate a trained model
python enjoy.py --run_name "advanced_training" --runs 20
```

### Saving and Loading Models

Models are automatically saved in the `runs/` directory:
- `{info_name}.pth`: Model weights
- `{info_name}.json`: Hyperparameters and configuration
- `{info_name}/`: TensorBoard logs

---

## Advanced Customizations

### 1. Custom Reward Functions

If your environment has complex reward requirements:

```python
class RewardShapedEnvironment(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add custom reward shaping
        shaped_reward = reward
        
        # Example: Add action smoothness penalty
        if hasattr(self, 'prev_action'):
            action_diff = np.linalg.norm(action - self.prev_action)
            shaped_reward -= 0.01 * action_diff
        
        self.prev_action = action.copy()
        
        return obs, shaped_reward, terminated, truncated, info
```

### 2. Custom Observation Processing

```python
class ObservationProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Modify observation space if needed
        self.observation_space = Box(
            low=-1, high=1, 
            shape=(processed_obs_dim,), 
            dtype=np.float32
        )
    
    def observation(self, obs):
        # Normalize observations
        processed_obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return processed_obs.astype(np.float32)
```

### 3. Custom Action Processing

```python
class ActionProcessingWrapper(gym.ActionWrapper):
    def action(self, action):
        # Transform actions from [-1, 1] to your environment's range
        scaled_action = action * self.action_scale + self.action_offset
        return scaled_action
    
    def reverse_action(self, action):
        # Transform back for logging/analysis
        return (action - self.action_offset) / self.action_scale
```

### 4. Domain-Specific Networks

If you need custom network architectures, modify `scripts/networks.py`:

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Dimension Mismatch Errors

**Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Solution**: Check your observation and action space dimensions:
```python
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Sample observation: {env.reset()[0].shape}")
```

#### 2. Unstable Training

**Symptoms**: Reward oscillates wildly, doesn't improve

**Solutions**:
- Reduce learning rates: `-lr_a 5e-5 -lr_c 1e-4`
- Increase TAU for more stable target updates: `-tau 1e-2`
- Use smaller batch sizes: `-batch_size 64`
- Enable Munchausen RL for stability: `-munchausen 1`

#### 3. Slow Learning

**Symptoms**: Reward improves very slowly

**Solutions**:
- Enable PER: `-per 1`
- Use n-step returns: `-nstep 5`
- Increase learning frequency: `-learn_every 1 -learn_number 2`

#### 4. Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
- Reduce buffer size: `-max_replay_size 100000`
- Reduce batch size: `-batch_size 32`
- Use smaller networks: `-layer_size 128`

#### 5. Environment-Specific Issues

**Reward Scale**: If rewards are too large/small:
```python
# Scale rewards in your environment
def step(self, action):
    obs, reward, term, trunc, info = super().step(action)
    scaled_reward = reward / 100.0  # Scale down large rewards
    return obs, scaled_reward, term, trunc, info
```

**Action Bounds**: Ensure actions are properly bounded:
```python
def step(self, action):
    # Clip actions to valid range
    clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
    return super().step(clipped_action)
```

### Debugging Tips

1. **Start Simple**: Test with minimal features first
2. **Monitor Metrics**: Use TensorBoard to track all metrics
3. **Sanity Checks**: Verify environment returns expected shapes
4. **Baseline Comparison**: Test on known environments first
5. **Gradual Complexity**: Add features one at a time

---

## Complete Example: Putting It All Together

Here's a complete example of adapting the codebase for a custom task:

### Step 1: Create Custom Environment

```python
# my_trading_env.py
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class TradingEnvironment(gym.Env):
    """Simple trading environment example"""
    
    def __init__(self, data_file="market_data.csv"):
        super().__init__()
        
        # Load your data
        self.data = self._load_data(data_file)
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # State: [price_features, portfolio_state, technical_indicators]
        self.observation_space = Box(
            low=-np.inf, high=np.inf, 
            shape=(20,), dtype=np.float32
        )
        
        # Action: [position_change] (-1 to 1)
        self.action_space = Box(
            low=-1.0, high=1.0, 
            shape=(1,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.portfolio_value = 10000
        self.position = 0
        return self._get_observation(), {}
    
    def step(self, action):
        # Trading logic here
        obs = self._get_observation()
        reward = self._calculate_reward(action)
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        self.current_step += 1
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        # Return market features + portfolio state
        return np.random.randn(20).astype(np.float32)  # Placeholder
    
    def _calculate_reward(self, action):
        # Implement your reward function
        return np.random.randn()  # Placeholder

# Register environment
gym.register(
    id='Trading-v0',
    entry_point='my_trading_env:TradingEnvironment',
)
```

### Step 2: Minimal Integration

```python
# Add to top of run.py (after imports)
try:
    from my_trading_env import TradingEnvironment
    gym.register(
        id='Trading-v0',
        entry_point='my_trading_env:TradingEnvironment',
    )
    print("✅ Custom trading environment registered")
except ImportError:
    print("⚠️ Custom environment not found")
```

### Step 3: Train

```bash
# Train with advanced features
python run.py -env "Trading-v0" \
  -frames 100000 \
  -per 1 -munchausen 1 -iqn 1 \
  -layer_size 256 \
  -lr_a 1e-4 -lr_c 3e-4 \
  -eval_every 5000 \
  -info "trading_v1"
```

### Step 4: Monitor and Evaluate

```bash
# Monitor training
tensorboard --logdir=runs

# Evaluate trained model
python enjoy.py --run_name "trading_v1" --runs 50
```

---

## What is Munchausen RL?

Before diving into the integration details, let's understand the key advanced feature available in this codebase:

### Munchausen RL (M-RL)

**Munchausen Reinforcement Learning** is an entropy-regularized RL technique that enhances training stability and exploration by modifying the reward function.

#### How it Works:
1. **Entropy Regularization**: Adds a scaled log-policy term to the reward
2. **Modified Reward**: `r_munchausen = r_original + α * clamp(τ * log π(a|s), min_val, 0)`
   - `α` (alpha): Scaling factor for the entropy term (default: 0.9)
   - `τ` (tau): Temperature parameter controlling entropy strength (default: 0.03)  
   - `clamp`: Prevents extreme negative values that could destabilize training

#### Benefits:
- **Improved Stability**: Entropy regularization prevents policy collapse and reduces variance
- **Better Exploration**: The log-policy term encourages diverse action selection
- **Faster Convergence**: Often reaches better policies with fewer samples
- **Robustness**: More resistant to hyperparameter sensitivity

#### When to Use:
- Environments with sparse or delayed rewards
- When training is unstable or converges to suboptimal policies
- Complex continuous control tasks
- When you want more robust exploration

```bash
# Enable Munchausen RL
python run.py -env "YourEnv-v0" -munchausen 1 -info "with_munchausen"
```

---

## Use Case: Financial Policy Optimization with Stochastic Environments

A common advanced use case for RL is optimizing financial policies, such as dynamic hedging strategies. These environments have a key characteristic: the state transitions are **stochastic** (e.g., simulated with Geometric Brownian Motion), meaning the next state is drawn from a distribution and is not deterministic.

Here’s how Munchausen RL and ICM apply to this specific problem setting.

### Munchausen RL: ✅ Highly Recommended

M-RL is exceptionally well-suited for financial applications.

**Why it Works:**
1.  **Promotes Robust Policies**: Financial markets are inherently random. A purely deterministic policy can be brittle and overfit to specific historical paths. M-RL adds entropy to the policy, encouraging it to be slightly stochastic. This results in a more robust strategy that is less sensitive to small variations in market conditions and can generalize better to new, unseen market data.
2.  **Better Strategic Exploration**: In a financial context, "exploration" means trying different hedging strategies (e.g., varying hedge ratios or rebalancing triggers). By adding the log-policy term to the reward, M-RL encourages the agent to avoid committing too early to a single strategy. It maintains a "softer" policy, which helps it discover more nuanced and effective hedging approaches.
3.  **Stabilizes Learning with Noisy Rewards**: Financial rewards (e.g., portfolio P&L) can be noisy and delayed. M-RL's entropy regularization provides a more stable learning signal, preventing the agent from being misled by short-term market noise.

**Recommendation**: For any financial policy optimization task, enabling Munchausen RL is a strong starting point. It directly addresses the need for robust, non-brittle policies required to handle market stochasticity.

```bash
# Recommended command for a financial hedging task
python run.py -env "YourHedgingEnv-v0" \
  -munchausen 1 \
  -per 1 \
  -iqn 1 \
  -info "robust_hedging_strategy"
```

This demonstrates the power of Munchausen RL for creating robust, entropy-regularized policies that handle market stochasticity effectively. For financial applications, focus on this proven technique rather than curiosity-driven methods that are unsuitable for inherently noisy environments.
