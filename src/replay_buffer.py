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
        """Add experience to the appropriate n-step buffer and flush ready transitions.

        Emits zero or more processed n-step transitions per call (e.g., when a terminal
        appears inside the window). Each emitted transition respects early terminals and
        does not bootstrap across episodes.
        """
        # Cycle through parallel environments
        if self.env_iter >= self.parallel_env:
            self.env_iter = 0
            
        # Add to n-step buffer and flush any ready transitions
        ready_exps = self.n_step_buffers[self.env_iter].add(state, action, reward, next_state, done)
        for exp in ready_exps:
            self._add_to_buffer(*exp)
            
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

    # Compatibility stubs so agent diagnostic code can call PER-like helpers safely
    def get_priority_stats(self):
        return {
            'priority_entropy': 0.0,
            'priority_max': 0.0,
            'priority_min': 0.0,
            'priority_mean': 0.0,
            'priority_std': 0.0
        }

    def sample_priority_values(self, k: int = 0):  # returns empty array
        return np.array([], dtype=np.float32)

    def update_priorities(self, *args, **kwargs):  # no-op
        return None


class CircularNStepBuffer:
    """Efficient n-step accumulator that respects terminals within the window.

    Contract:
    - add(s,a,r,s',done) may emit 0..K processed n-step transitions, each of the form
      (s_t, a_t, R_t^{(n)}, s_{t+n_or_term}, done_any)
    - If any done occurs within the first n steps of the window, we stop summation at the first terminal
      and do NOT bootstrap beyond it (done_any=True). No cross-episode leakage.
    """

    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        from collections import deque as _dq
        self.buffer = _dq()  # stores tuples (s, a, r, s_next, done)

    def _ready_for_front(self) -> bool:
        if len(self.buffer) == 0:
            return False
        L = min(self.n_step, len(self.buffer))
        # If we have full n steps, ready; else if any done in first L, also ready
        if L == self.n_step:
            return True
        # check any terminal in available window
        for k in range(L):
            if self.buffer[k][4]:
                return True
        return False

    def _pop_front_transition(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Build one processed transition starting at current front and pop left once."""
        assert len(self.buffer) > 0
        L = min(self.n_step, len(self.buffer))
        s0, a0, _, _, _ = self.buffer[0]
        ret = 0.0
        done_any = False
        next_s = self.buffer[0][3]
        for k in range(L):
            s, a, r, s_next, d = self.buffer[k]
            ret += (self.gamma ** k) * float(r)
            if d and not done_any:
                done_any = True
                next_s = s_next
                break
        if not done_any:
            # full n-step available
            _, _, _, s_next, _ = self.buffer[self.n_step - 1]
            next_s = s_next
        # pop the front and return
        self.buffer.popleft()
        return s0, a0, ret, next_s, done_any

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """Add a transition and emit all ready n-step experiences (list)."""
        self.buffer.append((state, action, reward, next_state, done))
        out = []
        while self._ready_for_front():
            out.append(self._pop_front_transition())
        return out



    

class PrioritizedReplay(object):
    """
    Prioritized Experience Replay (PER) with robust n-step handling.
    """

    def __init__(self, capacity, batch_size, device, seed, gamma=0.99, n_step=1, parallel_env=1, alpha=0.6, beta_start=0.4, beta_frames=100000):
        # Hyperparameters
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.device = device
        self.frame_count = 0
        self.batch_size = batch_size
        self.capacity = capacity

        # Circular storage
        self.pos = 0
        self.size = 0
        self.full = False

        # Lazy arrays
        self.states: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states: Optional[np.ndarray] = None
        self.dones = np.empty(capacity, dtype=np.bool_)

        # Priorities
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.min_priority = 1e-6

        # RNG
        self.rng = np.random.RandomState(seed)

        # N-step accumulators
        self.parallel_env = parallel_env
        self.n_step = n_step
        self.gamma = gamma
        self._nstep_accums = [CircularNStepBuffer(n_step, gamma) for _ in range(parallel_env)]
        self.iter_ = 0

        # Caches
        self._prob_alpha_cache = None  # type: Optional[np.ndarray]
        self._cumsum_cache = None  # type: Optional[np.ndarray]
        self._prob_sum_cache = None
        self._cache_valid = False
        self._cumsum_valid = False

        print("🚀 High-Performance PrioritizedReplay initialized:")
        print(f"  - Capacity: {capacity:,}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Alpha (priority exponent): {alpha}")
        print(f"  - Beta start (IS weight): {beta_start}")
        print(f"  - Beta frames: {beta_frames}")
        print(f"  - N-step: {n_step}")

    def _initialize_arrays(self, state: np.ndarray, action: np.ndarray) -> None:
        if self.states is not None:
            return
        state_shape = state.shape
        action_shape = action.shape if hasattr(action, 'shape') else (1,)
        print(f"📊 Initializing PER arrays with shapes: state{state_shape}, action{action_shape}")
        self.states = np.empty((self.capacity,) + state_shape, dtype=np.float32)
        self.next_states = np.empty((self.capacity,) + state_shape, dtype=np.float32)
        self.actions = np.empty((self.capacity,) + action_shape, dtype=np.float32)
        self._prob_alpha_cache = np.empty(self.capacity, dtype=np.float32)
        self._cumsum_cache = np.empty(self.capacity, dtype=np.float32)
        print("✅ PER arrays initialized successfully")

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def set_frame_count(self, frame_count):
        self.frame_count = frame_count

    def add(self, state, action, reward, next_state, done):
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        assert state.ndim == next_state.ndim
        ready = self._nstep_accums[self.iter_].add(state, action, reward, next_state, done)
        for (s0, a0, Rn, sN, done_any) in ready:
            self._add_to_buffer(s0, a0, Rn, sN, done_any)
        self.iter_ += 1

    def _add_to_buffer(self, state, action, reward, next_state, done):
        self._initialize_arrays(state, action.cpu().numpy() if torch.is_tensor(action) else action)
        action_np = action.cpu().numpy() if torch.is_tensor(action) else action
        assert self.states is not None and self.actions is not None and self.next_states is not None
        self.states[self.pos] = state
        self.actions[self.pos] = action_np
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.full = True
        self._cache_valid = False

    def sample(self):
        if self.size == 0:
            return None
        if self.size < self.batch_size:
            raise ValueError(f"Not enough samples: {self.size} < {self.batch_size}")
        assert self.states is not None and self.actions is not None and self.next_states is not None
        assert self._prob_alpha_cache is not None and self._cumsum_cache is not None
        self.frame_count += 1
        update_threshold = max(10, self.batch_size // 4)
        if not self._cache_valid or (hasattr(self, '_update_counter') and self._update_counter > update_threshold):
            self._update_probability_cache()
            self._update_counter = 0
        if not self._cumsum_valid:
            self._update_cumsum_cache()
        total = self._cumsum_cache[self.size - 1]
        random_vals = self.rng.uniform(0, total, self.batch_size)
        indices = np.searchsorted(self._cumsum_cache[:self.size], random_vals)
        indices = np.clip(indices, 0, self.size - 1)
        beta = self.beta_by_frame(self.frame_count)
        if total > 0:
            probs = self._prob_alpha_cache[indices] / total
            if beta == 1.0:
                weights = 1.0 / (self.size * probs)
                weights = weights / weights.max()
            else:
                weights = (self.size * probs) ** (-beta)
                weights = weights / weights.max()
        else:
            weights = np.ones(self.batch_size, dtype=np.float32)
        weights = weights.astype(np.float32)
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        states = torch.from_numpy(batch_states).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(batch_next_states).to(self.device, non_blocking=True)
        actions = torch.from_numpy(batch_actions).to(self.device, non_blocking=True)
        if actions.dim() == 3 and actions.size(1) == 1:
            actions = actions.squeeze(1)
        rewards = torch.from_numpy(batch_rewards).unsqueeze(1).to(self.device, non_blocking=True)
        dones = torch.from_numpy(batch_dones).unsqueeze(1).to(self.device, non_blocking=True)
        weights = torch.from_numpy(weights).unsqueeze(1).to(self.device, non_blocking=True)
        return states, actions, rewards, next_states, dones, indices, weights

    def _update_probability_cache(self):
        if self.size == 0:
            return
        if self._prob_alpha_cache is None:
            self._prob_alpha_cache = np.empty(self.capacity, dtype=np.float32)
        if self._cumsum_cache is None:
            self._cumsum_cache = np.empty(self.capacity, dtype=np.float32)
        priorities_slice = self.priorities[:self.size]
        self._prob_alpha_cache[:self.size] = priorities_slice ** self.alpha
        self._cache_valid = True
        self._cumsum_valid = False

    def _update_cumsum_cache(self):
        if self.size == 0 or not self._cache_valid:
            return
        if self._prob_alpha_cache is None or self._cumsum_cache is None:
            return
        np.cumsum(self._prob_alpha_cache[:self.size], out=self._cumsum_cache[:self.size])
        self._cumsum_valid = True

    def update_priorities(self, batch_indices, batch_priorities):
        if not isinstance(batch_indices, np.ndarray):
            batch_indices = np.array(batch_indices, dtype=np.int32)
        if not isinstance(batch_priorities, np.ndarray):
            batch_priorities = np.array(batch_priorities, dtype=np.float32)
        batch_priorities = np.maximum(batch_priorities.flatten(), self.min_priority)
        batch_indices = np.clip(batch_indices, 0, self.size - 1)
        self.priorities[batch_indices] = batch_priorities
        new_max = np.max(batch_priorities)
        if new_max > self.max_priority:
            self.max_priority = new_max
        self._cache_valid = False
        self._cumsum_valid = False
        self._update_counter = getattr(self, '_update_counter', 0) + 1

    def __len__(self):
        return self.size

    def get_memory_usage(self) -> float:
        if self.size == 0 or self.states is None or self.actions is None or self.next_states is None:
            return 0.0
        total_bytes = (
            self.states.nbytes + self.next_states.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes + self.priorities.nbytes
        )
        return total_bytes / (1024 * 1024)

    def get_priority_stats(self):
        if self.size == 0:
            return {'priority_entropy': 0.0, 'priority_max': 0.0, 'priority_min': 0.0, 'priority_mean': 0.0, 'priority_std': 0.0}
        pr = self.priorities[:self.size]
        pa = pr ** self.alpha
        s = pa.sum()
        if s <= 0:
            entropy = 0.0
            pa_mean = 0.0
            pa_std = 0.0
        else:
            p = pa / s
            entropy = float(-(p * (np.log(p + 1e-12))).sum())
            pa_mean = float(pr.mean())
            pa_std = float(pr.std())
        return {'priority_entropy': entropy, 'priority_max': float(pr.max()), 'priority_min': float(pr.min()), 'priority_mean': pa_mean, 'priority_std': pa_std}

    def sample_priority_values(self, k: int = 512):
        if self.size == 0:
            return np.array([], dtype=np.float32)
        k = min(k, self.size)
        idx = self.rng.choice(self.size, size=k, replace=False)
        return self.priorities[idx].astype(np.float32)
