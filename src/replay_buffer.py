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

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        n_step: int,
        parallel_env: int,
        device: torch.device,
        seed: int,
        gamma: float,
        state_shape: Optional[Tuple[int, ...]] = None,
        action_shape: Optional[Tuple[int, ...]] = None,
        use_memmap: bool = False,
    ):
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
            self.states = np.memmap(
                "replay_states.dat", dtype=np.float32, mode="w+", shape=(self.buffer_size,) + self.state_shape
            )
            self.next_states = np.memmap(
                "replay_next_states.dat",
                dtype=np.float32,
                mode="w+",
                shape=(self.buffer_size,) + self.state_shape,
            )
            self.actions = np.memmap(
                "replay_actions.dat", dtype=np.float32, mode="w+", shape=(self.buffer_size,) + self.action_shape
            )
        else:
            # Standard numpy arrays
            self.states = np.empty((self.buffer_size,) + self.state_shape, dtype=np.float32)
            self.next_states = np.empty((self.buffer_size,) + self.state_shape, dtype=np.float32)
            self.actions = np.empty((self.buffer_size,) + self.action_shape, dtype=np.float32)

        # These are always small enough for standard arrays
        self.rewards = np.empty(self.buffer_size, dtype=np.float32)
        self.dones = np.empty(self.buffer_size, dtype=np.bool_)

        print("✅ Buffer arrays initialized successfully")

    def add(
        self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
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

    def _add_to_buffer(
        self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
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

    def sample(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, None]:
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
            self.states.nbytes
            + self.next_states.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.dones.nbytes
        )
        return total_bytes / (1024 * 1024)


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
            ret += (self.gamma**k) * float(r)
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

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """Add a transition and emit all ready n-step experiences (list)."""
        self.buffer.append((state, action, reward, next_state, done))
        out = []
        while self._ready_for_front():
            out.append(self._pop_front_transition())
        return out
