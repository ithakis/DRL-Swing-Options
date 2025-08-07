from collections import deque
from typing import Optional, Tuple

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

    def set_episode_count(self, episode_count):
        """Set episode count (no-op for non-prioritized replay)."""
        pass
    
    def set_frame_count(self, frame_count):
        """Set frame count (no-op for non-prioritized replay)."""
        pass


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



    

class PrioritizedReplay(object):
    """
    Prioritized Experience Replay (PER) following DeepMind's implementation principles.
    
    This implementation follows the proportional prioritization approach from:
    "PRIORITIZED EXPERIENCE REPLAY" by Schaul et al.
    
    Key features:
    - Proportional priority sampling based on TD-error magnitude
    - Importance sampling with annealing β parameter
    - Efficient sum-tree-like sampling using numpy cumsum
    - Compatible with D4PG swing option pricing framework
    """
    def __init__(self, capacity, batch_size, device, seed, gamma=0.99, n_step=1, parallel_env=1, alpha=0.6, beta_start=0.4, beta_frames=100000):
        # PER hyperparameters following the paper
        self.alpha = alpha  # Priority exponent (0=uniform, 1=proportional)
        self.beta_start = beta_start  # Initial importance sampling weight
        self.beta_frames = beta_frames  # Frames to anneal β to 1.0
        self.device = device
        self.frame_count = 0  # Track frames (not episodes) for proper β annealing
        self.batch_size = batch_size
        self.capacity = capacity
        # Circular buffer implementation with numpy arrays
        self.pos = 0
        self.size = 0  # Current number of stored experiences
        self.full = False
        
        # Pre-allocated numpy arrays for maximum performance  
        self.states: Optional[np.ndarray] = None  # Lazy initialization
        self.actions: Optional[np.ndarray] = None
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states: Optional[np.ndarray] = None  
        self.dones = np.empty(capacity, dtype=np.bool_)
        
        # Priority storage - following DeepMind's approach
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.min_priority = 1e-6  # Minimum priority to avoid zero probabilities
        
        # Random state for reproducibility and efficiency
        self.rng = np.random.RandomState(seed)
        
        # N-step calculation
        self.parallel_env = parallel_env
        self.n_step = n_step
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0
        self.gamma = gamma
        
        # Performance optimizations
        self._prob_alpha_cache: Optional[np.ndarray] = None
        self._cumsum_cache: Optional[np.ndarray] = None  # Cache cumulative sum
        self._prob_sum_cache = None
        self._cache_valid = False
        self._cumsum_valid = False  # Separate flag for cumsum cache
        
        print("🚀 High-Performance PrioritizedReplay initialized:")
        print(f"  - Capacity: {capacity:,}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Alpha (priority exponent): {alpha}")
        print(f"  - Beta start (IS weight): {beta_start}")
        print(f"  - Beta frames: {beta_frames}")
        print(f"  - N-step: {n_step}")

    def _initialize_arrays(self, state: np.ndarray, action: np.ndarray) -> None:
        """Lazy initialization of storage arrays based on first experience."""
        if self.states is not None:
            return
            
        # Determine shapes from first experience
        state_shape = state.shape
        action_shape = action.shape if hasattr(action, 'shape') else (1,)
        
        print(f"📊 Initializing PER arrays with shapes: state{state_shape}, action{action_shape}")
        
        # Pre-allocate all arrays for maximum performance
        self.states = np.empty((self.capacity,) + state_shape, dtype=np.float32)
        self.next_states = np.empty((self.capacity,) + state_shape, dtype=np.float32)
        self.actions = np.empty((self.capacity,) + action_shape, dtype=np.float32)
        
        # Initialize probability cache
        self._prob_alpha_cache = np.empty(self.capacity, dtype=np.float32)
        self._cumsum_cache = np.empty(self.capacity, dtype=np.float32)  # Pre-allocate cumsum cache
        
        print("✅ PER arrays initialized successfully")

    def calc_multistep_return(self,n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def beta_by_frame(self, frame_idx):
        """
        Linearly anneals β from beta_start to 1.0 over frames (following Acme implementation).
        
        This follows the annealing schedule from the PER paper:
        "We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent β that reaches 1 only at 
        the end of learning."
        
        Args:
            frame_idx: Current frame/step count
            
        Returns:
            Current β value for importance sampling weight calculation
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def set_frame_count(self, frame_count):
        """Set the current frame count for proper β annealing."""
        self.frame_count = frame_count
    
    def add(self, state, action, reward, next_state, done):
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        assert state.ndim == next_state.ndim
        # Remove unnecessary expand_dims - states should maintain their original shape
        # state      = np.expand_dims(state, 0)
        # next_state = np.expand_dims(next_state, 0)
        action = torch.from_numpy(action).unsqueeze(0) if not torch.is_tensor(action) else action.unsqueeze(0)

        # n_step calc
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[self.iter_])
            self._add_to_buffer(state, action, reward, next_state, done)

        self.iter_ += 1
        
    def _add_to_buffer(self, state, action, reward, next_state, done):
        """Add experience to circular buffer with maximum efficiency."""
        # Initialize arrays on first call
        self._initialize_arrays(state, action.cpu().numpy() if torch.is_tensor(action) else action)
        
        # Convert action to numpy if it's a tensor
        if torch.is_tensor(action):
            action_np = action.cpu().numpy()
        else:
            action_np = action
        
        # Store experience at current position (vectorized operation)
        assert self.states is not None, "States array not initialized"
        assert self.actions is not None, "Actions array not initialized" 
        assert self.next_states is not None, "Next states array not initialized"
        
        self.states[self.pos] = state
        self.actions[self.pos] = action_np
        self.rewards[self.pos] = reward  
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        # Set priority to maximum for new experiences
        self.priorities[self.pos] = self.max_priority
        
        # Update circular buffer state
        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.full = True
            
        # Invalidate probability cache
        self._cache_valid = False

        
    def sample(self):
        """
        Sample experiences with proportional prioritization following DeepMind's approach.
        
        This implements the proportional prioritization from the PER paper where
        the probability of sampling transition i is: P(i) = p_i^α / Σ_k p_k^α
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
            where weights are importance sampling weights for bias correction
        """
        if self.size == 0:
            return None
            
        if self.size < self.batch_size:
            raise ValueError(f"Not enough samples: {self.size} < {self.batch_size}")
            
        # Ensure arrays are initialized
        assert self.states is not None, "States array not initialized"
        assert self.actions is not None, "Actions array not initialized"
        assert self.next_states is not None, "Next states array not initialized"
        assert self._prob_alpha_cache is not None, "Probability cache not initialized"
        assert self._cumsum_cache is not None, "Cumsum cache not initialized"
        
        # Increment frame count for β annealing
        self.frame_count += 1
            
        # Performance optimization: Update caches lazily and in batches
        update_threshold = max(10, self.batch_size // 4)  # Update every few steps
        
        # Update probability cache if needed (expensive operation)
        if not self._cache_valid or (hasattr(self, '_update_counter') and self._update_counter > update_threshold):
            self._update_probability_cache()
            self._update_counter = 0  # Reset counter
        
        # Update cumsum cache if needed (less expensive, but still cached)
        if not self._cumsum_valid:
            self._update_cumsum_cache()
        
        # Fast sampling using pre-computed cumsum
        total = self._cumsum_cache[self.size - 1]
        
        # Generate random values for sampling
        random_vals = self.rng.uniform(0, total, self.batch_size)
        
        # Use searchsorted for O(log n) sampling instead of O(n)
        indices = np.searchsorted(self._cumsum_cache[:self.size], random_vals)
        indices = np.clip(indices, 0, self.size - 1)  # Safety clamp
        
        # Calculate importance sampling weights vectorized (optimized)
        beta = self.beta_by_frame(self.frame_count)
        
        # Fast weight calculation using pre-computed probabilities
        if total > 0:  # Avoid division by zero
            probs = self._prob_alpha_cache[indices] / total
            # Optimized weight calculation - avoid double exponentiation when possible
            if beta == 1.0:
                # When beta=1, weights are just inverse probabilities normalized
                weights = 1.0 / (self.size * probs)
                weights = weights / weights.max()
            else:
                weights = (self.size * probs) ** (-beta)
                weights = weights / weights.max()  # Normalize for stability
        else:
            # Fallback for edge case
            weights = np.ones(self.batch_size, dtype=np.float32)
        
        weights = weights.astype(np.float32)
        
        # Vectorized batch extraction
        batch_states = self.states[indices]
        batch_actions = self.actions[indices] 
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        
        # Efficient tensor conversion with minimal copies
        states = torch.from_numpy(batch_states).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(batch_next_states).to(self.device, non_blocking=True)
        
        # Handle actions - convert to tensor format consistently
        actions = torch.from_numpy(batch_actions).to(self.device, non_blocking=True)
        if actions.dim() == 3 and actions.size(1) == 1:  # Remove singleton dimension if present
            actions = actions.squeeze(1)
        
        rewards = torch.from_numpy(batch_rewards).unsqueeze(1).to(self.device, non_blocking=True)
        dones = torch.from_numpy(batch_dones).unsqueeze(1).to(self.device, non_blocking=True)
        weights = torch.from_numpy(weights).unsqueeze(1).to(self.device, non_blocking=True)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def _update_probability_cache(self):
        """Update cached probability calculations for efficient sampling."""
        if self.size == 0:
            return
        
        # Ensure cache is initialized
        if self._prob_alpha_cache is None:
            self._prob_alpha_cache = np.empty(self.capacity, dtype=np.float32)
        if self._cumsum_cache is None:
            self._cumsum_cache = np.empty(self.capacity, dtype=np.float32)
            
        # Vectorized priority to probability conversion
        priorities_slice = self.priorities[:self.size]
        self._prob_alpha_cache[:self.size] = priorities_slice ** self.alpha
        self._cache_valid = True
        self._cumsum_valid = False  # Cumsum needs to be recalculated
    
    def _update_cumsum_cache(self):
        """Update cached cumulative sum for ultra-fast sampling."""
        if self.size == 0 or not self._cache_valid:
            return
        
        # Ensure cache is initialized
        if self._prob_alpha_cache is None or self._cumsum_cache is None:
            return
            
        # Use in-place cumsum for better performance
        np.cumsum(self._prob_alpha_cache[:self.size], out=self._cumsum_cache[:self.size])
        self._cumsum_valid = True
    
    def update_priorities(self, batch_indices, batch_priorities):
        """
        Update priorities based on TD-error magnitudes following DeepMind's approach.
        
        This follows the priority update rule from the PER paper:
        p_i = (|δ_i| + ε)^α
        
        Where δ_i is the TD-error and ε is a small positive constant to ensure
        that transitions with zero TD-error still have a non-zero probability.
        
        Args:
            batch_indices: Indices of experiences to update
            batch_priorities: New priority values (should be |TD-error| magnitudes)
        """
        # Convert to numpy arrays if needed
        if not isinstance(batch_indices, np.ndarray):
            batch_indices = np.array(batch_indices, dtype=np.int32)
        if not isinstance(batch_priorities, np.ndarray):
            batch_priorities = np.array(batch_priorities, dtype=np.float32)
        
        # Ensure priorities are positive and add small epsilon (following DeepMind)
        batch_priorities = np.maximum(batch_priorities.flatten(), self.min_priority)
        
        # Clip indices to valid range
        batch_indices = np.clip(batch_indices, 0, self.size - 1)
        
        # Vectorized priority update (much faster than loop)
        self.priorities[batch_indices] = batch_priorities
        
        # Update max priority efficiently
        new_max = np.max(batch_priorities)
        if new_max > self.max_priority:
            self.max_priority = new_max
        
        # Invalidate cache
        self._cache_valid = False
        self._cumsum_valid = False  # Also invalidate cumsum cache
        
        # Performance optimization: Only recalculate cache every N updates
        self._update_counter = getattr(self, '_update_counter', 0) + 1

    def __len__(self):
        """Return current number of experiences in buffer."""
        return self.size

    def get_memory_usage(self) -> float:
        """Return approximate memory usage in MB."""
        if self.size == 0 or self.states is None or self.actions is None or self.next_states is None:
            return 0.0
            
        # Calculate actual memory usage from numpy arrays
        total_bytes = (
            self.states.nbytes + 
            self.next_states.nbytes + 
            self.actions.nbytes + 
            self.rewards.nbytes + 
            self.dones.nbytes +
            self.priorities.nbytes
        )
        
        return total_bytes / (1024 * 1024)
