import random
from collections import deque, namedtuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, n_step, parallel_env, device, seed, gamma):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.parallel_env = parallel_env
        self.n_step = n_step
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[self.iter_])
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        self.iter_ += 1




    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, None, None)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    

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

        max_prio = max(self.priorities) if self.buffer else 1.0  # Use Python max instead of numpy
        

        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)
        self.iter_ += 1

        
    def sample(self):
        N = len(self.buffer)
        prios = np.array(self.priorities, dtype=float)
        assert N == len(prios)
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_path(self.path)
        #print(beta)
        self.path+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices])**(-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 

        states      = torch.tensor(np.float32(np.concatenate(states)), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.float32(np.concatenate(next_states)), dtype=torch.float32).to(self.device)
        actions     = torch.cat(actions).to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1) 
        dones       = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)
        weights    = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        #print("s",states.shape)
        #print("ns", next_states.shape)
        #print("a", actions.shape)
        #print("r", rewards.shape)
        #print("d", dones.shape)
        #print("w", weights.shape)
        
        return states, actions, rewards, next_states, dones, indices, weights
    

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            # Ensure priority is a scalar value
            if hasattr(prio, 'item'):
                prio = prio.item()
            elif isinstance(prio, np.ndarray):
                prio = float(prio.flatten()[0])
            self.priorities[idx] = float(prio) 

    def __len__(self):
        return len(self.buffer)
