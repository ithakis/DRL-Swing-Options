import copy
import random
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

try:
    from .networks import IQN, Actor, Critic
    from .replay_buffer import CircularReplayBuffer, PrioritizedReplay
except ImportError:
    from networks import IQN, Actor, Critic
    from replay_buffer import CircularReplayBuffer, PrioritizedReplay


class Agent:
    """
    Stable Agent implementation with diagnostics and lightweight defaults (2×64 MLPs).
    Defaults: single-step TD targets, IQN and Munchausen are opt-in (off by default),
    gradient clipping is disabled unless thresholds are provided, actor/critic weight
    decay defaults to 5e-5/1e-4, and target updates use a smoother tau=0.002.
    """
    def __init__(self, state_size, action_size, n_step, per, munchausen, distributional, noise_type, random_seed, hidden_size: int = 64,
                 actor_hidden_size: Optional[int] = None, critic_hidden_size: Optional[int] = None,
                 actor_layers: int = 2, critic_layers: int = 2,
                 optimizer: str = "adamw", weight_decay_actor: float = 5e-5, weight_decay_critic: float = 1e-4,
                 BUFFER_SIZE=int(1e6), BATCH_SIZE=128, GAMMA=0.99, t=2e-3, LR_ACTOR=1e-4, LR_CRITIC=1e-4,
                 WEIGHT_DECAY=0, LEARN_EVERY=1, LEARN_NUMBER=1, epsilon=.3, epsilon_decay=1.0,
                 device="cpu", min_replay_size=None, per_alpha=0.6, per_beta_start=0.4, per_beta_frames=100000,
                 per_priority_floor: float = 1e-6, per_priority_clip_pct: float = 99.5,
                 final_lr_fraction=1.0, total_episodes=None, warmup_frac=0.05, min_lr=1e-7,
                 actor_grad_clip: float = 0.0, critic_grad_clip: float = 0.0,
                 actor_grad_clip_type: str = "none", critic_grad_clip_type: str = "none",
                 grad_clip_norm_type: float = 2.0,
                 noise_sigma: float = 1.0, noise_anneal_power: float = 1.0,
                 tau_final: Optional[float] = None, tau_schedule_frac: float = 0.0,
                 critic_ema_decay: float = 0.0,
                 **kwargs):
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
        self.tau_final = tau_final if tau_final is not None and tau_final > 0 else None
        self.tau_schedule_frac = max(0.0, min(1.0, tau_schedule_frac))
        self._current_tau = t
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

        actor_hidden_size = actor_hidden_size or hidden_size
        critic_hidden_size = critic_hidden_size or hidden_size
        if actor_layers < 1:
            raise ValueError(f"actor_layers must be >= 1, got {actor_layers}")
        if critic_layers < 2 and not distributional:
            raise ValueError("critic_layers must be >= 2 when using the standard critic")

        optimizer_name = optimizer.lower()
        if optimizer_name not in {"adam", "adamw"}:
            raise ValueError(f"Unsupported optimizer '{optimizer}'. Choose from 'adam', 'adamw'.")
        optim_cls = optim.AdamW if optimizer_name == "adamw" else optim.Adam

        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.optimizer_name = optimizer_name
        self._updates_done = 0

        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size=actor_hidden_size, n_layers=actor_layers).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, hidden_size=actor_hidden_size, n_layers=actor_layers).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = self._build_optimizer(
            model=self.actor_local,
            optim_cls=optim_cls,
            lr=LR_ACTOR,
            weight_decay=weight_decay_actor,
        )

        if distributional:
            self.N = 32
            self.critic_local = IQN(state_size, action_size, layer_size=critic_hidden_size, device=device, seed=random_seed, dueling=False, N=self.N).to(device)
            self.critic_target = IQN(state_size, action_size, layer_size=critic_hidden_size, device=device, seed=random_seed, dueling=False, N=self.N).to(device)
            self.critic_target.load_state_dict(self.critic_local.state_dict())
        else:
            self.critic_local = Critic(state_size, action_size, random_seed, hidden_size=critic_hidden_size, n_layers=critic_layers).to(device)
            self.critic_target = Critic(state_size, action_size, random_seed, hidden_size=critic_hidden_size, n_layers=critic_layers).to(device)
            self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = self._build_optimizer(
            model=self.critic_local,
            optim_cls=optim_cls,
            lr=LR_CRITIC,
            weight_decay=weight_decay_critic,
        )
        self.critic_ema_decay = critic_ema_decay
        self.critic_ema_state = None if critic_ema_decay <= 0 else copy.deepcopy(self.critic_local.state_dict())

        actor_params = sum(p.numel() for p in self.actor_local.parameters())
        critic_params = sum(p.numel() for p in self.critic_local.parameters())
        optimizer_label = optim_cls.__name__
        print(
            f"Optimizer setup: optimizer={optimizer_label} | "
            f"actor_lr={LR_ACTOR:.2e} (weight_decay={weight_decay_actor:.1e}) | "
            f"critic_lr={LR_CRITIC:.2e} (weight_decay={weight_decay_critic:.1e})"
        )
        print(
            f"Actor params: {actor_params:,} | hidden_size={actor_hidden_size} | layers={actor_layers}"
        )
        print(
            f"Critic params: {critic_params:,} | hidden_size={critic_hidden_size} | layers={'IQN' if distributional else critic_layers}"
        )

        self.entropy_tau = 0.03
        self.lo = -1.0
        self.alpha = 0.9

        if per:
            self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE, device=device, seed=random_seed, gamma=GAMMA, n_step=n_step,
                                            parallel_env=1, alpha=per_alpha, beta_start=per_beta_start, beta_frames=per_beta_frames)
            # Configure PER priority stability
            try:
                self.memory.min_priority = per_priority_floor
                self.memory.priority_clip_pct = per_priority_clip_pct
            except Exception:
                pass
        else:
            self.memory = CircularReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, n_step=n_step, parallel_env=1,
                                                device=device, seed=random_seed, gamma=GAMMA, use_memmap=BUFFER_SIZE > 500000)

        self.final_lr_fraction = final_lr_fraction
        self.total_episodes = total_episodes or 10000
        self.warmup_frac = warmup_frac
        self.min_lr = min_lr
        warmup_episodes = int(self.total_episodes * warmup_frac)

        self.grad_clip_norm_type = grad_clip_norm_type
        if actor_grad_clip_type == "none" or not actor_grad_clip or actor_grad_clip <= 0:
            self.actor_grad_clip = None
            self.actor_grad_clip_type = "none"
        else:
            self.actor_grad_clip = actor_grad_clip
            self.actor_grad_clip_type = actor_grad_clip_type if actor_grad_clip_type in {"norm", "value"} else "norm"
        if critic_grad_clip_type == "none" or not critic_grad_clip or critic_grad_clip <= 0:
            self.critic_grad_clip = None
            self.critic_grad_clip_type = "none"
        else:
            self.critic_grad_clip = critic_grad_clip
            self.critic_grad_clip_type = critic_grad_clip_type if critic_grad_clip_type in {"norm", "value"} else "norm"
        # Default behavior keeps clipping disabled; users can enable it via positive thresholds.

        self.noise_sigma = noise_sigma
        self.noise_anneal_power = noise_anneal_power

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

    def _build_optimizer(self, model: torch.nn.Module, optim_cls, lr: float, weight_decay: float):
        """Create optimizer with sensible weight decay exclusions."""
        params_decay = []
        params_no_decay = []
        if weight_decay > 0.0:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.dim() == 1 or name.endswith("bias") or "norm" in name.lower():
                    params_no_decay.append(param)
                else:
                    params_decay.append(param)

        param_groups = []
        if params_decay:
            param_groups.append({"params": params_decay, "weight_decay": weight_decay})
        if params_no_decay:
            param_groups.append({"params": params_no_decay, "weight_decay": 0.0})

        if not param_groups:
            return optim_cls(model.parameters(), lr=lr, weight_decay=0.0)
        return optim_cls(param_groups, lr=lr)

    def _clip_gradients(self, parameters: Iterable[torch.nn.Parameter], clip_value: Optional[float], clip_type: str):
        """Apply gradient clipping according to the configured strategy. No-op when clip_value<=0 (default)."""
        if clip_value is None or clip_value <= 0:
            return
        if clip_type == "norm":
            clip_grad_norm_(parameters, clip_value, norm_type=self.grad_clip_norm_type)
        elif clip_type == "value":
            clip_grad_value_(parameters, clip_value)

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
            scale = (self.epsilon ** self.noise_anneal_power) * self.noise_sigma
            action += self.noise.sample() * scale
        elif add_noise and self.noise is None:
            scale = (self.epsilon ** self.noise_anneal_power) * self.noise_sigma
            action += np.random.normal(0, scale, size=action.shape)
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
        self._clip_gradients(self.critic_local.parameters(), self.critic_grad_clip, self.critic_grad_clip_type)
        self.critic_optimizer.step()
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self._clip_gradients(self.actor_local.parameters(), self.actor_grad_clip, self.actor_grad_clip_type)
        self.actor_optimizer.step()
        self._updates_done += 1
        if self.step_counter % 200 == 0:
            with torch.no_grad():
                tgt_q = self.critic_target(states, self.actor_target(states))
                self._last_target_drift = (q_expected - tgt_q).abs().mean().item()
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        self._update_ema_buffers()
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
        self._clip_gradients(self.critic_local.parameters(), self.critic_grad_clip, self.critic_grad_clip_type)
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        q_pred, _ = self.critic_local(states, actions_pred, self.N)
        actor_loss = -q_pred.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self._clip_gradients(self.actor_local.parameters(), self.actor_grad_clip, self.actor_grad_clip_type)
        self.actor_optimizer.step()
        self._updates_done += 1
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
        self._update_ema_buffers()
        self.epsilon *= self.epsilon_decay
        if self.noise is not None:
            self.noise.reset()
        return critic_loss.item(), actor_loss.item()

    @property
    def updates_done(self) -> int:
        return self._updates_done

    def soft_update(self, local, target):
        tau = self._compute_tau()
        with torch.no_grad():
            for tp, lp in zip(target.parameters(), local.parameters()):
                tp.data.mul_(1 - tau)
                tp.data.add_(lp.data, alpha=tau)

    def _compute_tau(self) -> float:
        if self.tau_final is None or self.tau_schedule_frac <= 0.0:
            return self.t
        frac = min(1.0, max(0.0, (self._episode_count) / max(1, int(self.total_episodes * self.tau_schedule_frac))))
        self._current_tau = self.t * (1.0 - frac) + self.tau_final * frac
        return self._current_tau

    def _update_ema_buffers(self):
        if self.critic_ema_decay <= 0 or self.critic_ema_state is None:
            return
        with torch.no_grad():
            for k, v in self.critic_local.state_dict().items():
                self.critic_ema_state[k].mul_(self.critic_ema_decay).add_(v, alpha=1.0 - self.critic_ema_decay)

    def get_critic_eval_state(self):
        """Return EMA-smoothed critic parameters for evaluation if available."""
        return self.critic_ema_state if self.critic_ema_state is not None else self.critic_local.state_dict()

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
        # N-step debug metrics: ensure bootstrap is masked when any terminal occurs
        try:
            with torch.no_grad():
                dmean = dones.float().mean().item()
                bmask_mean = (1 - dones.float()).mean().item()
            writer.add_scalar("nstep/done_mean", dmean, ts)
            writer.add_scalar("nstep/bootstrap_mask_mean", bmask_mean, ts)
        except Exception:
            pass
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
