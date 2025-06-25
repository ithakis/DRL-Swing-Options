from collections import deque
from typing import Any, Optional, Tuple

import numpy as np
import torch


class CircularReplayBuffer:
    """
    High-performance circular array-based replay buffer.
    
    Key optimizations:
    - Uses numpy circular arrays instead of deque for O(1) operations
    - Pre-allocated memory for zero-copy operations
    - Vectorized sampling with efficient indexing
    - Memory-mapped storage option for very large buffers
    - SIMD-optimized operations where possible
    """

    def __init__(self, buffer_size: int, batch_size: int, n_step: int, parallel_env: int, 
                 device: torch.device, seed: int, gamma: float, 
                 state_shape: Optional[Tuple[int, ...]] = None,
                 action_shape: Optional[Tuple[int, ...]] = None,
                 use_memmap: bool = False):
        """
        Initialize CircularReplayBuffer with pre-allocated arrays.
        
        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Size of sampling batch
            n_step: Number of steps for n-step returns
            parallel_env: Number of parallel environments (usually 1)
            device: PyTorch device for tensor operations
            seed: Random seed for reproducibility
            gamma: Discount factor for n-step returns
            state_shape: Shape of state observations (auto-detected if None)
            action_shape: Shape of actions (auto-detected if None)
            use_memmap: Use memory mapping for very large buffers (>1GB)
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.device = device
        self.gamma = gamma
        self.use_memmap = use_memmap
        
        # Initialize random state
        self.rng = np.random.RandomState(seed)
        
        # Circular buffer state
        self.position = 0
        self.size = 0  # Current number of stored experiences
        self.full = False  # Whether buffer has wrapped around
        
        # Pre-allocated arrays (will be initialized on first add)
        self.states: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.rewards: Optional[np.ndarray] = None
        self.next_states: Optional[np.ndarray] = None
        self.dones: Optional[np.ndarray] = None
        
        # Store initial shapes for lazy initialization
        self.state_shape = state_shape
        self.action_shape = action_shape
        
        # N-step circular buffers for each parallel environment
        self.n_step_buffers = [CircularNStepBuffer(n_step, gamma) for _ in range(parallel_env)]
        self.env_iter = 0
        
        print("🚀 CircularReplayBuffer initialized:")
        print(f"  - Buffer size: {buffer_size:,}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - N-step: {n_step}")
        print(f"  - Memory mapping: {use_memmap}")

    def _initialize_arrays(self, state: np.ndarray, action: np.ndarray) -> None:
        """Lazy initialization of storage arrays based on first experience."""
        if self.states is not None:
            return
            
        # Determine shapes from first experience if not provided
        if self.state_shape is None:
            self.state_shape = state.shape
        if self.action_shape is None:
            self.action_shape = action.shape
            
        # Calculate memory requirements
        state_bytes = np.prod(self.state_shape) * 4 * self.buffer_size  # float32
        action_bytes = np.prod(self.action_shape) * 4 * self.buffer_size
        total_mb = (state_bytes * 2 + action_bytes + self.buffer_size * 8) / (1024 * 1024)
        
        print(f"📊 Allocating {total_mb:.1f} MB for replay buffer arrays...")
        
        # Choose storage type based on size and user preference
        if self.use_memmap and total_mb > 1000:  # Use memmap for buffers >1GB
            print("💾 Using memory-mapped storage for large buffer")
            self.states = np.memmap('replay_states.dat', dtype=np.float32, mode='w+', 
                                  shape=(self.buffer_size,) + self.state_shape)
            self.next_states = np.memmap('replay_next_states.dat', dtype=np.float32, mode='w+',
                                       shape=(self.buffer_size,) + self.state_shape)
            self.actions = np.memmap('replay_actions.dat', dtype=np.float32, mode='w+',
                                   shape=(self.buffer_size,) + self.action_shape)
        else:
            # Standard numpy arrays
            self.states = np.empty((self.buffer_size,) + self.state_shape, dtype=np.float32)
            self.next_states = np.empty((self.buffer_size,) + self.state_shape, dtype=np.float32)
            self.actions = np.empty((self.buffer_size,) + self.action_shape, dtype=np.float32)
        
        # These are always small enough for standard arrays
        self.rewards = np.empty(self.buffer_size, dtype=np.float32)
        self.dones = np.empty(self.buffer_size, dtype=np.bool_)
        
        print("✅ Buffer arrays initialized successfully")

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """Add experience to the appropriate n-step buffer."""
        # Cycle through parallel environments
        if self.env_iter >= self.parallel_env:
            self.env_iter = 0
            
        # Add to n-step buffer
        self.n_step_buffers[self.env_iter].add(state, action, reward, next_state, done)
        
        # Check if n-step buffer is ready
        if self.n_step_buffers[self.env_iter].is_ready():
            n_step_experience = self.n_step_buffers[self.env_iter].get_experience()
            self._add_to_buffer(*n_step_experience)
            
        self.env_iter += 1

    def _add_to_buffer(self, state: np.ndarray, action: np.ndarray, reward: float,
                      next_state: np.ndarray, done: bool) -> None:
        """Add processed n-step experience directly to circular buffer."""
        # Initialize arrays on first call
        self._initialize_arrays(state, action)
        
        # Store experience at current position
        assert self.states is not None, "States array not initialized"
        assert self.actions is not None, "Actions array not initialized"
        assert self.rewards is not None, "Rewards array not initialized"
        assert self.next_states is not None, "Next states array not initialized"
        assert self.dones is not None, "Dones array not initialized"
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Update circular buffer state
        self.position = (self.position + 1) % self.buffer_size
        if self.size < self.buffer_size:
            self.size += 1
        else:
            self.full = True

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, None]:
        """
        Efficiently sample a batch of experiences.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size < self.batch_size:
            raise ValueError(f"Not enough samples in buffer: {self.size} < {self.batch_size}")
        
        if self.states is None:
            raise RuntimeError("Buffer not initialized - no experiences added yet")
        
        assert self.actions is not None, "Actions array not initialized"
        assert self.rewards is not None, "Rewards array not initialized" 
        assert self.next_states is not None, "Next states array not initialized"
        assert self.dones is not None, "Dones array not initialized"
        
        # Vectorized random sampling - much faster than random.sample()
        indices = self.rng.choice(self.size, size=self.batch_size, replace=False)
        
        # Vectorized array indexing - O(1) operation
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        
        # Convert to tensors with optimized memory transfer
        states = torch.from_numpy(batch_states).to(self.device, non_blocking=True)
        actions = torch.from_numpy(batch_actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(batch_rewards).unsqueeze(1).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(batch_next_states).to(self.device, non_blocking=True)
        dones = torch.from_numpy(batch_dones).unsqueeze(1).to(self.device, non_blocking=True)
        
        return (states, actions, rewards, next_states, dones, indices, None)

    def __len__(self) -> int:
        """Return current number of experiences in buffer."""
        return self.size

    def is_ready(self) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= self.batch_size

    def get_memory_usage(self) -> float:
        """Return approximate memory usage in MB."""
        if self.states is None:
            return 0.0
        
        assert self.actions is not None, "Actions array not initialized"
        assert self.rewards is not None, "Rewards array not initialized"
        assert self.next_states is not None, "Next states array not initialized"
        assert self.dones is not None, "Dones array not initialized"
        
        total_bytes = (
            self.states.nbytes + 
            self.next_states.nbytes + 
            self.actions.nbytes + 
            self.rewards.nbytes + 
            self.dones.nbytes
        )
        return total_bytes / (1024 * 1024)


class CircularNStepBuffer:
    """Efficient circular buffer for n-step return calculation."""
    
    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: list = []
        self.position = 0
        self.full = False
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Add experience to n-step buffer."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.n_step:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.n_step
            self.full = True
    
    def is_ready(self) -> bool:
        """Check if buffer has enough experiences for n-step return."""
        return len(self.buffer) == self.n_step
    
    def get_experience(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Calculate and return n-step experience."""
        if not self.is_ready():
            raise ValueError("Buffer not ready for n-step calculation")
        
        # Calculate n-step return efficiently
        n_step_return = 0.0
        for i in range(self.n_step):
            n_step_return += (self.gamma ** i) * self.buffer[i][2]  # reward
        
        # Return (initial_state, initial_action, n_step_return, final_next_state, final_done)
        return (
            self.buffer[0][0],  # initial state
            self.buffer[0][1],  # initial action  
            n_step_return,      # n-step return
            self.buffer[-1][3], # final next_state
            self.buffer[-1][4]  # final done
        )


# Legacy ReplayBuffer class for backward compatibility
class ReplayBuffer(CircularReplayBuffer):
    """Legacy wrapper for backward compatibility."""
    
    def __init__(self, buffer_size: int, batch_size: int, n_step: int, parallel_env: int, 
                 device: torch.device, seed: int, gamma: float):
        super().__init__(
            buffer_size=buffer_size,
            batch_size=batch_size,
            n_step=n_step,
            parallel_env=parallel_env,
            device=device,
            seed=seed,
            gamma=gamma,
            use_memmap=False  # Default to standard arrays for compatibility
        )
        print("⚠️  Using legacy ReplayBuffer interface - consider upgrading to CircularReplayBuffer")

    def calc_multistep_return(self, n_step_buffer: Any) -> Tuple[Any, Any, float, Any, Any]:
        """Legacy method for backward compatibility."""
        Return = 0.0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]
    

class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, batch_size, device, seed, gamma=0.99, n_step=1, parallel_env=1, alpha=0.6, beta_start = 0.4, beta_paths=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_paths = beta_paths
        self.device = device
        self.path = 1 #for beta calculation
        self.batch_size = batch_size
        self.capacity   = capacity
        self.buffer     = deque(maxlen=capacity)
        self.pos        = 0
        self.priorities = deque(maxlen=capacity)
        self.seed = np.random.seed(seed)
        self.parallel_env = parallel_env
        self.n_step = n_step
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0
        
        self.gamma = gamma

    def calc_multistep_return(self,n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def beta_by_path(self, path_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_paths.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + path_idx * (1.0 - self.beta_start) / self.beta_paths)
    
    def add(self, state, action, reward, next_state, done):
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        action = torch.from_numpy(action).unsqueeze(0)

        # n_step calc
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[self.iter_])

        # Keep track of max priority efficiently
        if not hasattr(self, '_max_priority'):
            self._max_priority = 1.0
        max_prio = self._max_priority
        

        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)
        self.iter_ += 1
        
        # Invalidate probability cache when buffer changes
        if hasattr(self, '_prob_cache'):
            delattr(self, '_prob_cache')

        
    def sample(self):
        N = len(self.buffer)
        if N == 0:
            return None
            
        # More efficient priority sampling - avoid creating full arrays
        if not hasattr(self, '_prob_cache') or len(self._prob_cache) != N:
            prios = np.array(list(self.priorities), dtype=np.float32)
            probs = prios ** self.alpha
            self._prob_cache = probs / probs.sum()
        
        P = self._prob_cache
        
        # Get indices based on probability  
        indices = np.random.choice(N, self.batch_size, p=P, replace=True)
        indices = np.asarray(indices, dtype=int)  # Ensure it's a numpy array
        
        beta = self.beta_by_path(self.path)
        self.path += 1
                
        # Compute importance-sampling weights more efficiently
        weights = (N * P[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)
        
        # Extract samples efficiently
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # More efficient tensor creation with concatenation
        states = torch.from_numpy(np.concatenate(states).astype(np.float32)).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(np.concatenate(next_states).astype(np.float32)).to(self.device, non_blocking=True)
        actions = torch.cat(actions).to(self.device, non_blocking=True)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.from_numpy(weights).unsqueeze(1).to(self.device, non_blocking=True)
        
        return states, actions, rewards, next_states, dones, indices, weights
    

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            # Ensure priority is a scalar value
            if hasattr(prio, 'item'):
                prio = prio.item()
            elif isinstance(prio, np.ndarray):
                prio = float(prio.flatten()[0])
            prio = float(prio)
            self.priorities[idx] = prio
            # Update max priority tracking
            if prio > self._max_priority:
                self._max_priority = prio
        
        # Invalidate probability cache when priorities change
        if hasattr(self, '_prob_cache'):
            delattr(self, '_prob_cache') 

    def __len__(self):
        return len(self.buffer)
