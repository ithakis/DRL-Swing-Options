import copy

# Performance monitoring imports
import gc
import random

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

try:
    # Try relative imports first (when called from run.py)
    from .networks import IQN, Actor, Critic
    from .replay_buffer import CircularReplayBuffer, PrioritizedReplay
except ImportError:
    # Fall back to absolute imports (when called from notebook or directly)
    from networks import IQN, Actor, Critic
    from replay_buffer import CircularReplayBuffer, PrioritizedReplay

# TODO: Check for batch norm comparison! batch norm seems to have a big impact on final performance
#       Also check if normal gaussian noise is enough. -> D4PG paper says there is no difference maybe chooseable parameter for the implementation

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size,
                      action_size,
                      n_step,
                      per, 
                      munchausen,
                      distributional,
                      noise_type,
                      random_seed,
                      hidden_size,
                      BUFFER_SIZE = int(1e6),  # replay buffer size
                      BATCH_SIZE = 128,        # minibatch size
                      GAMMA = 0.99,            # discount factor
                      TAU = 1e-3,              # for soft update of target parameters
                      LR_ACTOR = 1e-4,         # learning rate of the actor 
                      LR_CRITIC = 1e-4,        # learning rate of the critic
                      WEIGHT_DECAY = 0,#1e-2        # L2 weight decay
                      LEARN_EVERY = 1,
                      LEARN_NUMBER = 1,
                      EPSILON = 1.0,
                      EPSILON_DECAY = 1,
                      device = "cpu",
                      paths = 100000,
                      min_replay_size=None,     # NEW: Minimum replay buffer size before learning starts
                      speed_mode=True,        # NEW: Enable speed optimizations
                      use_compile=False,       # NEW: Enable torch.compile optimization (disabled by default)
                      use_amp=False           # NEW: Enable automatic mixed precision
                      ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        # Convert device string to torch device object if needed
        if isinstance(device, str):
            if device.lower() == "cuda" or device.lower() == "gpu":
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
            else:
                device = torch.device("cpu")
        
        # CPU-specific optimizations
        if device.type == 'cpu':
            # Enable optimized CPU kernels if available
            if hasattr(torch.backends, 'mkl'):
                torch.backends.mkl.enabled = True
            if hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = True
            
            
        self.state_size = state_size
        self.action_size = action_size
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.per = per
        self.munchausen = munchausen
        self.n_step = n_step
        self.distributional = distributional
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LEARN_EVERY = LEARN_EVERY
        self.LEARN_NUMBER = LEARN_NUMBER
        self.EPSILON_DECAY = EPSILON_DECAY
        self.device = device
        self.seed = random.seed(random_seed)
        self.use_amp = use_amp  # Store AMP setting
        
        # Initialize minimum replay size - use buffer size as fallback if None
        if min_replay_size is None:
            self.min_replay_size = BUFFER_SIZE
        else:
            self.min_replay_size = min_replay_size
        
        print(f"🔄 Minimum replay buffer size set to {self.min_replay_size:,} samples before learning starts")
        
        # Initialize AMP scaler if requested
        if self.use_amp and device.type == 'cuda':
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("✅ Automatic Mixed Precision (AMP) enabled")
        else:
            self.scaler = None
            if self.use_amp:
                print("⚠️ AMP requested but not available (GPU required)")
        # distributional Values
        self.N = 32
        self.entropy_coeff = 0.001
        # munchausen values
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9
        
        self.eta = torch.tensor([.1], dtype=torch.float32).to(device)
        
        print("Using: ", device)
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        if self.distributional:
            self.critic_local = IQN(state_size, action_size, layer_size=hidden_size, device=device, seed=random_seed, dueling=False, N=self.N).to(device)
            self.critic_target = IQN(state_size, action_size, layer_size=hidden_size, device=device, seed=random_seed, dueling=False, N=self.N).to(device)
        else:
            self.critic_local = Critic(state_size, action_size, random_seed).to(device)
            self.critic_target = Critic(state_size, action_size, random_seed).to(device)

        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        print("Actor: \n", self.actor_local)
        print("\nCritic: \n", self.critic_local)

        # Apply torch.compile with optimized settings
        if use_compile and hasattr(torch, 'compile'):
            print("Compiling models with torch.compile...")
            try:
                # Store original networks before compilation
                self._actor_local_orig = self.actor_local
                self._actor_target_orig = self.actor_target
                self._critic_local_orig = self.critic_local
                self._critic_target_orig = self.critic_target
                
                # Use appropriate compile mode based on device and network size
                if device.type == 'cpu':
                    # For CPU and small networks, use 'default' mode for faster compilation
                    # with minimal overhead while still getting some optimization benefits
                    compile_mode = 'default'
                    dynamic_setting = True  # Allow dynamic shapes to reduce recompilation
                    print("🚀 Using CPU-optimized torch.compile settings (mode: default, dynamic: True)")
                else:
                    # For GPU, use reduce-overhead mode for better balance
                    compile_mode = 'reduce-overhead'
                    dynamic_setting = True
                    print("🚀 Using GPU-optimized torch.compile settings (mode: reduce-overhead, dynamic: True)")
                    
                # torch.compile mode explanation:
                # - 'default': Balanced compilation time vs runtime performance (best for small networks)
                # - 'reduce-overhead': Optimizes for frequent model calls (good for RL)  
                # - 'max-autotune': Maximum optimization but high compilation overhead (GPU + large models)
                # - dynamic=True: Reduces recompilation when input shapes vary

                self.actor_local = torch.compile(self.actor_local, mode=compile_mode, dynamic=dynamic_setting)
                self.actor_target = torch.compile(self.actor_target, mode=compile_mode, dynamic=dynamic_setting)
                self.critic_local = torch.compile(self.critic_local, mode=compile_mode, dynamic=dynamic_setting)
                self.critic_target = torch.compile(self.critic_target, mode=compile_mode, dynamic=dynamic_setting)
                print(f"✅ All models successfully compiled with torch.compile (mode: {compile_mode}, dynamic: {dynamic_setting})")
                
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}")
                print("Falling back to non-compiled models")
        elif use_compile:
            print("⚠️ torch.compile not available in this PyTorch version")
        # Noise process
        self.noise_type = noise_type
        if noise_type == "ou":
            self.noise = OUNoise(action_size, random_seed)
            self.epsilon = EPSILON
        else:
            self.epsilon = 0.3
        print("Use Noise: ", noise_type)
        # Replay memory
        if per:
            self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE, device=device, seed=random_seed, gamma=GAMMA, n_step=n_step, parallel_env=1, beta_paths=paths)
        else:
            self.memory = CircularReplayBuffer(
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE, 
                n_step=n_step,
                parallel_env=1,
                device=device,
                seed=random_seed,
                gamma=GAMMA,
                use_memmap=BUFFER_SIZE > 500000  # Use memory mapping for large buffers
            )
            print("✅ Using optimized CircularReplayBuffer")
        
        if distributional:
            self.learn = self.learn_distribution
        else:
            self.learn = self.learn_

        print("Using PER: ", per)    
        print("Using Munchausen RL: ", munchausen)
        
        # Performance optimization settings
        self.performance_monitor = True
        self.memory_cleanup_frequency = 1000
        self.memory_threshold_mb = 8000  # 8GB threshold
        self.step_counter = 0
        
        print("🚀 Performance optimizations enabled:")
        print(f"  - Memory monitoring: {self.performance_monitor}")
        print(f"  - Memory cleanup frequency: {self.memory_cleanup_frequency}")
        print(f"  - Memory threshold: {self.memory_threshold_mb} MB")
        print("  - Always using optimized CircularReplayBuffer (when PER=False)")

        
    def step(self, state, action, reward, next_state, done, timestamp, writer):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.step_counter += 1
        
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Performance monitoring
        perf_info = self.monitor_performance()
        
        # Memory cleanup and optimization
        self.cleanup_memory()
        self.optimize_memory_usage()

        # Learn only after minimum replay size is reached and we have enough samples
        buffer_has_min_samples = len(self.memory) >= self.min_replay_size
        buffer_has_batch_samples = len(self.memory) > self.BATCH_SIZE
        
        if buffer_has_min_samples and buffer_has_batch_samples and timestamp % self.LEARN_EVERY == 0:
            losses = None
            for _ in range(self.LEARN_NUMBER):
                experiences = self.memory.sample()
                losses = self.learn(experiences, self.GAMMA)
            
            # Log losses if computed
            if losses is not None:
                writer.add_scalar("Critic_loss", losses[0], timestamp)
                writer.add_scalar("Actor_loss", losses[1], timestamp)
            
            # Log performance metrics
            if perf_info:
                writer.add_scalar("Performance/Memory_MB", perf_info['memory_mb'], timestamp)
                writer.add_scalar("Performance/Buffer_Fill_Ratio", perf_info['buffer_fill_ratio'], timestamp)
                
        elif not buffer_has_min_samples:
            # Log initial collection progress
            collection_progress = len(self.memory) / self.min_replay_size * 100
            if timestamp % 1000 == 0:  # Log every 1000 steps during initial collection
                writer.add_scalar("Collection_Progress", collection_progress, timestamp)
                print(f"\r🔄 Collecting samples: {collection_progress:.1f}% ({len(self.memory):,}/{self.min_replay_size:,} samples)", end="")

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device, non_blocking=True)

        assert state.shape == (state.shape[0], self.state_size), "shape: {}".format(state.shape)
        
        # Handle both compiled and non-compiled models
        if hasattr(self, '_actor_local_orig'):
            # Use original network for eval/train mode switching
            actor_network = self._actor_local_orig
        else:
            actor_network = self.actor_local
            
        # More efficient inference with torch.no_grad()
        actor_network.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy()
        actor_network.train()
        
        if add_noise:
            if self.noise_type == "ou":
                action += self.noise.sample() * self.epsilon
            else:
                # More efficient noise generation
                noise = np.random.normal(0, self.epsilon, size=action.shape)
                action += noise
                
        return action

    def reset(self):
        self.noise.reset()

    def learn_(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idx, weights = experiences

        # Optimized tensor operations - avoid redundant device transfers
        if not states.is_cuda and self.device.type == 'cuda':
            states = states.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            rewards = rewards.to(self.device, non_blocking=True)
            next_states = next_states.to(self.device, non_blocking=True)
            dones = dones.to(self.device, non_blocking=True)
            if weights is not None:
                weights = weights.to(self.device, non_blocking=True)
        
        # ---------------------------- update critic ---------------------------- #
        # Use context manager for better performance and memory management
        with torch.no_grad():
            if not self.munchausen:
                # Get predicted next-state actions and Q values from target models
                actions_next = self.actor_target(next_states)
                Q_targets_next = self.critic_target(next_states, actions_next)
                # Compute Q targets for current states (y_i)
                Q_targets = rewards + (gamma**self.n_step * Q_targets_next * (1 - dones))
            else:
                actions_next = self.actor_target(next_states)
                q_t_n = self.critic_target(next_states, actions_next)
                # calculate log-pi - more efficient computation
                logsum = torch.logsumexp(q_t_n / self.entropy_tau, 1, keepdim=True)
                tau_log_pi_next = q_t_n - self.entropy_tau * logsum
                
                pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1)
                Q_target = (self.GAMMA**self.n_step * (pi_target * (q_t_n - tau_log_pi_next) * (1 - dones)))

                if self.distributional:
                    q_k_target, _ = self.critic_target(states, actions)
                    q_k_target = q_k_target.mean(dim=1, keepdim=True)
                else:
                    q_k_target = self.critic_target(states, actions)
                    
                tau_log_pik = q_k_target - self.entropy_tau * torch.logsumexp(q_k_target / self.entropy_tau, 1, keepdim=True)
                # calc munchausen reward with more efficient clamping
                munchausen_reward = rewards + self.alpha * torch.clamp(tau_log_pik, min=self.lo, max=0)
                Q_targets = munchausen_reward + Q_target

        # Compute critic loss - more efficient forward pass
        Q_expected = self.critic_local(states, actions)
        
        if self.per:
            td_error = Q_targets - Q_expected
            critic_loss = (td_error.pow(2) * weights).mean()
            # Pre-compute priorities for efficiency
            priorities = torch.clamp(torch.abs(td_error), -1, 1).detach()
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            priorities = None
            
        # Optimize critic with improved gradient handling
        self.critic_optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Use AMP if available for faster training
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                critic_loss_scaled = self.scaler.scale(critic_loss)
            critic_loss_scaled.backward()
            self.scaler.unscale_(self.critic_optimizer)
            # Use original network for gradient clipping if compiled
            critic_params = self._critic_local_orig.parameters() if hasattr(self, '_critic_local_orig') else self.critic_local.parameters()
            clip_grad_norm_(critic_params, 1.0)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            critic_loss.backward()
            critic_params = self._critic_local_orig.parameters() if hasattr(self, '_critic_local_orig') else self.critic_local.parameters()
            clip_grad_norm_(critic_params, 1.0)
            self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss with shared forward pass
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                actor_loss_scaled = self.scaler.scale(actor_loss)
            actor_loss_scaled.backward()
            self.scaler.unscale_(self.actor_optimizer)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            actor_loss.backward()
            self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
        # Update priorities efficiently
        if self.per and priorities is not None and hasattr(self.memory, 'update_priorities'):
            priorities_np = priorities.cpu().numpy().flatten()
            self.memory.update_priorities(idx, priorities_np)
            
        # ----------------------- update epsilon and noise ----------------------- #
        self.epsilon *= self.EPSILON_DECAY
        
        if self.noise_type == "ou":
            self.noise.reset()
            
        # Return detached losses for logging
        return critic_loss.detach().cpu().item(), actor_loss.detach().cpu().item()

    
    
    
    def monitor_performance(self):
        """Monitor memory usage and performance metrics."""
        if not self.performance_monitor:
            return {}
            
        # Get memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Get buffer-specific metrics
        buffer_memory_mb = 0.0
        if hasattr(self.memory, 'get_memory_usage'):
            try:
                buffer_memory_mb = self.memory.get_memory_usage()
            except (AttributeError, TypeError):
                pass  # Older buffer implementations don't have this method
        
        return {
            'memory_mb': memory_mb,
            'buffer_size': len(self.memory),
            'buffer_fill_ratio': len(self.memory) / self.BUFFER_SIZE,
            'buffer_memory_mb': buffer_memory_mb
        }
    
    def cleanup_memory(self, force=False):
        """Perform memory cleanup operations."""
        if not force and self.step_counter % self.memory_cleanup_frequency != 0:
            return
            
        # Get current memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Perform cleanup if memory usage is high
        if memory_mb > self.memory_threshold_mb or force:
            print(f"🧹 Memory cleanup: {memory_mb:.1f} MB -> ", end="")
            
            # Garbage collection
            gc.collect()
            
            # CUDA cache cleanup if available
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Check memory after cleanup
            new_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"{new_memory_mb:.1f} MB (saved {memory_mb - new_memory_mb:.1f} MB)")
            
            # Reset compiled models if memory is still high
            if new_memory_mb > self.memory_threshold_mb * 1.2 and hasattr(self, '_actor_local_orig'):
                print("🔄 Resetting compiled models to prevent memory bloat...")
                self._reset_compiled_models()
    
    def _reset_compiled_models(self):
        """Reset compiled models to prevent memory accumulation."""
        try:
            # Restore original models
            self.actor_local = self._actor_local_orig
            self.actor_target = self._actor_target_orig
            self.critic_local = self._critic_local_orig
            self.critic_target = self._critic_target_orig
            
            # Recompile with fresh state
            if hasattr(torch, 'compile'):
                compile_mode = 'default' if self.device.type == 'cpu' else 'reduce-overhead'
                self.actor_local = torch.compile(self.actor_local, mode=compile_mode, dynamic=True)
                self.actor_target = torch.compile(self.actor_target, mode=compile_mode, dynamic=True)
                self.critic_local = torch.compile(self.critic_local, mode=compile_mode, dynamic=True)
                self.critic_target = torch.compile(self.critic_target, mode=compile_mode, dynamic=True)
                print("✅ Models recompiled successfully")
        except Exception as e:
            print(f"⚠️ Model reset failed: {e}")

    def soft_update(self, local_model, target_model):
        """Soft update model parameters with improved efficiency.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        # More efficient soft update using torch operations
        with torch.no_grad():
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                # Use in-place operations for better memory efficiency
                target_param.data.mul_(1.0 - self.TAU)
                target_param.data.add_(local_param.data, alpha=self.TAU)


    def learn_distribution(self, experiences, gamma):
            """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value

            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            states, actions, rewards, next_states, dones, idx, weights = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models

            # Get max predicted Q values (for next states) from target model
            if not self.munchausen:
                with torch.no_grad():
                    next_actions = self.actor_local(next_states)
                    Q_targets_next, _ = self.critic_target(next_states, next_actions, self.N)
                    Q_targets_next = Q_targets_next.transpose(1,2)
                # Compute Q targets for current states 
                Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
            else:
                with torch.no_grad():
                    #### CHECK FOR THE SHAPES!!
                    actions_next = self.actor_target(next_states.to(self.device))
                    Q_targets_next, _ = self.critic_target(next_states.to(self.device), actions_next.to(self.device), self.N)

                    q_t_n = Q_targets_next.mean(1)
                    # calculate log-pi - in the paper they subtracted the max_Q value from the Q to ensure stability since we only predict the max value we dont do that
                    # this might cause some instability (?) needs to be tested
                    logsum = torch.logsumexp(\
                        q_t_n /self.entropy_tau, 1).unsqueeze(-1) #logsum trick
                    assert logsum.shape == (self.BATCH_SIZE, 1), "log pi next has wrong shape: {}".format(logsum.shape)
                    tau_log_pi_next = (q_t_n  - self.entropy_tau*logsum).unsqueeze(1)
                    
                    pi_target = F.softmax(q_t_n/self.entropy_tau, dim=1).unsqueeze(1)
                    # in the original paper for munchausen RL they summed over all actions - we only predict the best Qvalue so we will not sum over all actions
                    Q_target = (self.GAMMA**self.n_step * (pi_target * (Q_targets_next-tau_log_pi_next)*(1 - dones.unsqueeze(-1)))).transpose(1,2)
                    assert Q_target.shape == (self.BATCH_SIZE, self.action_size, self.N), "has shape: {}".format(Q_target.shape)

                    q_k_target, _ = self.critic_target(states, actions)
                    q_k_target = q_k_target.mean(dim=1)  # shape: [batch_size, 1]
                    tau_log_pik = q_k_target - self.entropy_tau * torch.logsumexp(q_k_target/self.entropy_tau, 1, keepdim=True)
                    assert tau_log_pik.shape == (self.BATCH_SIZE, 1), f"shape instead is {tau_log_pik.shape}"
                    # calc munchausen reward:
                    munchausen_reward = (rewards + self.alpha*torch.clamp(tau_log_pik, min=self.lo, max=0)).unsqueeze(-1)
                    assert munchausen_reward.shape == (self.BATCH_SIZE, self.action_size, 1)
                    # Compute Q targets for current states 
                    Q_targets = munchausen_reward + Q_target
            # Get expected Q values from local model
            Q_expected, taus = self.critic_local(states, actions, self.N)
            assert Q_targets.shape == (self.BATCH_SIZE, 1, self.N)
            assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)
    
            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
            
            if self.per:
                critic_loss = (quantil_l.sum(dim=1).mean(dim=1, keepdim=True)*weights.to(self.device)).mean()
            else:
                critic_loss = quantil_l.sum(dim=1).mean(dim=1).mean()
            # Minimize the loss
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            clip_grad_norm_(self.critic_local.parameters(), 1.0)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local.get_qvalues(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target)
            self.soft_update(self.actor_local, self.actor_target)                     
            if self.per and hasattr(self.memory, 'update_priorities'):
                priorities = np.clip(abs(td_error.sum(dim=1).mean(dim=1,keepdim=True).data.cpu().numpy().flatten()), -1, 1)
                self.memory.update_priorities(idx, priorities)
            # ----------------------- update epsilon and noise ----------------------- #
            
            self.epsilon *= self.EPSILON_DECAY
            
            if self.noise_type == "ou":
                self.noise.reset()
            return critic_loss.detach().cpu().item(), actor_loss.detach().cpu().item()

        

    def optimize_memory_usage(self):
        """Optimize memory usage inspired by the other D4PG implementation."""
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            # Force garbage collection periodically
            if self.step_counter % self.memory_cleanup_frequency == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
    
    def get_network_info(self):
        """Get network parameter information for debugging."""
        total_params = 0
        trainable_params = 0
        
        for name, network in [('actor', self.actor_local), ('critic', self.critic_local)]:
            if hasattr(network, 'parameters'):
                params = sum(p.numel() for p in network.parameters())
                trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
                total_params += params
                trainable_params += trainable
                print(f"{name.capitalize()}: {params:,} parameters ({trainable:,} trainable)")
        
        print(f"Total: {total_params:,} parameters ({trainable_params:,} trainable)")
        return total_params, trainable_params
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

def calc_fraction_loss(FZ_,FZ, taus, weights=None):
    """calculate the loss for the fraction proposal network """
    
    gradients1 = FZ - FZ_[:, :-1]
    gradients2 = FZ - FZ_[:, 1:] 
    flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
    flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
    gradients = (torch.where(flag_1, gradients1, - gradients1) + torch.where(flag_2, gradients2, -gradients2)).view(taus.shape[0], 31)
    assert not gradients.requires_grad
    if weights is not None:
        loss = ((gradients * taus[:, 1:-1]).sum(dim=1)*weights).mean()
    else:
        loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
    return loss 
    
def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss