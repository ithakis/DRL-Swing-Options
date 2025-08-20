import copy
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

try:
    from .networks import IQN, Actor, Critic
    from .replay_buffer import CircularReplayBuffer, PrioritizedReplay
except ImportError:
    from networks import IQN, Actor, Critic
    from replay_buffer import CircularReplayBuffer, PrioritizedReplay


class Agent:
    """Stable Agent implementation with diagnostics."""
    def __init__(self, state_size, action_size, n_step, per, munchausen, distributional, noise_type, random_seed, hidden_size,
                 BUFFER_SIZE=int(1e6), BATCH_SIZE=128, GAMMA=0.99, t=1e-3, LR_ACTOR=1e-4, LR_CRITIC=1e-4,
                 WEIGHT_DECAY=0, LEARN_EVERY=1, LEARN_NUMBER=1, epsilon=.3, epsilon_decay=1.0,
                 device="cpu", min_replay_size=None, per_alpha=0.6, per_beta_start=0.4, per_beta_frames=100000,
                 final_lr_fraction=1.0, total_episodes=None, warmup_frac=0.05, min_lr=1e-7, **kwargs):
        # kwargs absorbs unexpected legacy params (e.g., 'paths') without breaking
        if isinstance(device, str):
            device = torch.device('cuda' if device.lower() in ('cuda', 'gpu') and torch.cuda.is_available() else 'cpu')
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.per = per
        self.munchausen = munchausen
        self.distributional = distributional
        self.GAMMA = GAMMA
        self.t = t
        self.n_step = n_step
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARN_EVERY = LEARN_EVERY
        self.LEARN_NUMBER = LEARN_NUMBER
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.min_replay_size = min_replay_size or BATCH_SIZE * 10

        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        if distributional:
            self.N = 32
            self.critic_local = IQN(state_size, action_size, layer_size=hidden_size, device=device, seed=random_seed, dueling=False, N=self.N).to(device)
            self.critic_target = IQN(state_size, action_size, layer_size=hidden_size, device=device, seed=random_seed, dueling=False, N=self.N).to(device)
        else:
            self.critic_local = Critic(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
            self.critic_target = Critic(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.entropy_tau = 0.03
        self.lo = -1.0
        self.alpha = 0.9

        if per:
            self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE, device=device, seed=random_seed, gamma=GAMMA, n_step=n_step,
                                            parallel_env=1, alpha=per_alpha, beta_start=per_beta_start, beta_frames=per_beta_frames)
        else:
            self.memory = CircularReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, n_step=n_step, parallel_env=1,
                                                device=device, seed=random_seed, gamma=GAMMA, use_memmap=BUFFER_SIZE > 500000)

        self.final_lr_fraction = final_lr_fraction
        self.total_episodes = total_episodes or 10000
        self.warmup_frac = warmup_frac
        self.min_lr = min_lr
        warmup_episodes = int(self.total_episodes * warmup_frac)

        def lr_lambda(step: int, init_lr: float):
            if final_lr_fraction >= 1.0:
                return 1.0
            if step < warmup_episodes:
                return (step + 1) / max(1, warmup_episodes)
            decay_steps = step - warmup_episodes
            total_decay = max(1, self.total_episodes - warmup_episodes)
            frac = final_lr_fraction ** (decay_steps / total_decay)
            return max(min_lr / init_lr, frac)

        if final_lr_fraction < 1.0:
            self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lambda s: lr_lambda(s, LR_ACTOR))
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lambda s: lr_lambda(s, LR_CRITIC))
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None

        self.learn = self.learn_distribution if distributional else self.learn_
        self.step_counter = 0
        self._last_td_percentiles = None
        self._last_target_drift = None
        self._last_iqn_spread = None
        self.noise = OUNoise(action_size, random_seed) if noise_type == 'ou' else None
        self._episode_count = 0

    def update_episode_count(self, episode: int):
        """Update internal episode counter (used for PER beta annealing in caller)."""
        self._episode_count = episode

    def step_lr_schedulers(self, episode: int):
        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()

    def act(self, state: np.ndarray, add_noise: bool = True):
        state_t = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_t).cpu().numpy()
        self.actor_local.train()
        if add_noise and self.noise is not None:
            action += self.noise.sample() * self.epsilon
        elif add_noise and self.noise is None:
            action += np.random.normal(0, self.epsilon, size=action.shape)
        return action

    def step(self, state, action, reward, next_state, done, timestamp, writer):
        self.step_counter += 1
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) < self.min_replay_size or len(self.memory) <= self.BATCH_SIZE:
            if timestamp % 1000 == 0:
                writer.add_scalar("Collection_Progress", len(self.memory) / self.min_replay_size * 100, timestamp)
            return
        if timestamp % self.LEARN_EVERY != 0:
            return
        last_batch = None
        losses = None
        for _ in range(self.LEARN_NUMBER):
            last_batch = self.memory.sample()
            losses = self.learn(last_batch, self.GAMMA)
        if losses:
            writer.add_scalar("Critic_loss", losses[0], timestamp)
            writer.add_scalar("Actor_loss", losses[1], timestamp)
        if last_batch:
            self._log_batch_diagnostics(last_batch, timestamp, writer)
        if self.per and hasattr(self.memory, 'get_priority_stats') and timestamp % (self.LEARN_EVERY * 20) == 0:
            for k, v in self.memory.get_priority_stats().items():
                writer.add_scalar(f"PER/{k}", v, timestamp)

    def learn_(self, experiences, gamma) -> Tuple[float, float]:
        states, actions, rewards, next_states, dones, idx, weights = experiences
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            if not self.munchausen:
                q_target = rewards + (gamma ** self.n_step) * q_next * (1 - dones.float())
            else:
                logsum = torch.logsumexp(q_next / self.entropy_tau, dim=1, keepdim=True)
                tau_log_pi_next = q_next - self.entropy_tau * logsum
                pi = F.softmax(q_next / self.entropy_tau, dim=1)
                q_target = rewards + (self.GAMMA ** self.n_step) * (pi * (q_next - tau_log_pi_next) * (1 - dones.float()))
        q_expected = self.critic_local(states, actions)
        if self.per:
            td = q_target - q_expected
            critic_loss = (td.pow(2) * weights).mean()
            with torch.no_grad():
                abs_td = td.abs().flatten()
                if abs_td.numel() > 10:
                    self._last_td_percentiles = (
                        torch.quantile(abs_td, 0.5).item(),
                        torch.quantile(abs_td, 0.9).item(),
                        torch.quantile(abs_td, 0.99).item()
                    )
            priorities = td.abs().detach().clamp_min(1e-6)
        else:
            critic_loss = F.mse_loss(q_expected, q_target)
            priorities = None
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor_local.parameters(), 1.0)
        self.actor_optimizer.step()
        if self.step_counter % 200 == 0:
            with torch.no_grad():
                tgt_q = self.critic_target(states, self.actor_target(states))
                self._last_target_drift = (q_expected - tgt_q).abs().mean().item()
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        if self.per and priorities is not None and hasattr(self.memory, 'update_priorities'):
            self.memory.update_priorities(idx, priorities.cpu().numpy().flatten())
        self.epsilon *= self.epsilon_decay
        if self.noise is not None:
            self.noise.reset()
        return critic_loss.item(), actor_loss.item()

    def learn_distribution(self, experiences, gamma) -> Tuple[float, float]:
        states, actions, rewards, next_states, dones, idx, weights = experiences
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            qt_next, _ = self.critic_target(next_states, next_actions, self.N)
            qt_next = qt_next.transpose(1, 2)
            if not self.munchausen:
                q_targets = rewards.unsqueeze(-1) + (self.GAMMA ** self.n_step) * qt_next * (1 - dones.float().unsqueeze(-1))
            else:
                q_mean = qt_next.mean(-1)
                logsum = torch.logsumexp(q_mean / self.entropy_tau, dim=1, keepdim=True)
                tau_log_pi_next = (q_mean - self.entropy_tau * logsum).unsqueeze(1)
                pi_target = F.softmax(q_mean / self.entropy_tau, dim=1).unsqueeze(1)
                q_targets = rewards.unsqueeze(-1) + (self.GAMMA ** self.n_step) * (pi_target * (qt_next - tau_log_pi_next) * (1 - dones.float().unsqueeze(-1)))
        q_expected, taus = self.critic_local(states, actions, self.N)
        td_error = q_targets - q_expected
        huber = calculate_huber_loss(td_error, 1.0)
        quantile_loss = (torch.abs(taus - (td_error.detach() < 0).float()) * huber).sum(dim=1).mean(dim=1)
        if self.per:
            critic_loss = (quantile_loss.unsqueeze(1) * weights).mean()
        else:
            critic_loss = quantile_loss.mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        q_pred, _ = self.critic_local(states, actions_pred, self.N)
        actor_loss = -q_pred.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor_local.parameters(), 1.0)
        self.actor_optimizer.step()
        if self.per and hasattr(self.memory, 'update_priorities'):
            pr = td_error.mean(dim=(1, 2)).abs().clamp_min(1e-6).detach().cpu().numpy()
            self.memory.update_priorities(idx, pr)
        with torch.no_grad():
            flat = q_targets.view(q_targets.size(0), -1)
            q10 = torch.quantile(flat, 0.1, dim=1).mean().item()
            q50 = torch.quantile(flat, 0.5, dim=1).mean().item()
            q90 = torch.quantile(flat, 0.9, dim=1).mean().item()
            self._last_iqn_spread = (q10, q50, q90, (q90 - q10))
        if self.step_counter % 200 == 0:
            with torch.no_grad():
                tgt_q, _ = self.critic_target(states, self.actor_target(states), self.N)
                self._last_target_drift = (q_expected - tgt_q).abs().mean().item()
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        self.epsilon *= self.epsilon_decay
        if self.noise is not None:
            self.noise.reset()
        return critic_loss.item(), actor_loss.item()

    def soft_update(self, local, target):
        with torch.no_grad():
            for tp, lp in zip(target.parameters(), local.parameters()):
                tp.data.mul_(1 - self.t)
                tp.data.add_(lp.data, alpha=self.t)

    def _log_batch_diagnostics(self, batch, ts, writer):
        states, actions, rewards, next_states, dones, idx, weights = batch
        if torch.is_tensor(actions):
            with torch.no_grad():
                at_low = (actions <= -0.99).float().mean().item()
                at_high = (actions >= 0.99).float().mean().item()
                var_mean = actions.var(dim=0).mean().item() if actions.numel() > 1 else 0.0
            writer.add_scalar("Policy/Actions_at_lower_pct", at_low, ts)
            writer.add_scalar("Policy/Actions_at_upper_pct", at_high, ts)
            writer.add_scalar("Policy/Action_variance_mean", var_mean, ts)
        if self._last_td_percentiles and self.step_counter % 50 == 0:
            p50, p90, p99 = self._last_td_percentiles
            writer.add_scalar("TD_Error/p50", p50, ts)
            writer.add_scalar("TD_Error/p90", p90, ts)
            writer.add_scalar("TD_Error/p99", p99, ts)
        if self._last_target_drift and self.step_counter % 200 == 0:
            writer.add_scalar("Stability/Target_drift", self._last_target_drift, ts)
        if self.distributional and self._last_iqn_spread and self.step_counter % 200 == 0:
            q10, q50, q90, spread = self._last_iqn_spread
            writer.add_scalar("IQN/q10", q10, ts)
            writer.add_scalar("IQN/q50", q50, ts)
            writer.add_scalar("IQN/q90", q90, ts)
            writer.add_scalar("IQN/q90_minus_q10", spread, ts)

    def reset(self):
        if self.noise is not None:
            self.noise.reset()


class OUNoise:
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def calculate_huber_loss(td_errors, k=1.0):
    return torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))


def calc_fraction_loss(FZ_, FZ, taus, weights=None):
    gradients1 = FZ - FZ_[:, :-1]
    gradients2 = FZ - FZ_[:, 1:]
    flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
    flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
    gradients = (torch.where(flag_1, gradients1, -gradients1) + torch.where(flag_2, gradients2, -gradients2)).view(taus.shape[0], 31)
    if weights is not None:
        loss = ((gradients * taus[:, 1:-1]).sum(dim=1) * weights).mean()
    else:
        loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
    return loss