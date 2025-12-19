import copy
import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

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

    def __init__(
        self,
        state_size,
        action_size,
        n_step,
        per,
        munchausen,
        distributional,
        random_seed,
        hidden_size: int = 64,
        actor_hidden_size: Optional[int] = None,
        critic_hidden_size: Optional[int] = None,
        actor_layers: int = 2,
        critic_layers: int = 2,
        optimizer: str = "adamw",
        weight_decay_actor: float = 5e-5,
        weight_decay_critic: float = 1e-4,
        BUFFER_SIZE=int(1e6),
        BATCH_SIZE=128,
        GAMMA=0.99,
        t=2e-3,
        LR_ACTOR=1e-4,
        LR_CRITIC=1e-4,
        WEIGHT_DECAY=0,
        LEARN_EVERY=1,
        LEARN_NUMBER=1,
        device="cpu",
        min_replay_size=None,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100000,
        per_priority_floor: float = 1e-6,
        per_priority_clip_pct: float = 0.0,
        per_priority_scheme: str = "standard",
        per_huber_kappa: float = 1.0,
        per_alpha_final: Optional[float] = None,
        per_alpha_ramp_start: int = 0,
        per_alpha_ramp_end: int = 0,
        per_beta_final: Optional[float] = None,
        per_alpha_sigmoid: bool = False,
        final_lr_fraction=1.0,
        total_episodes=None,
        warmup_frac: float = 0.05,
        warmup_episodes: Optional[int] = None,
        min_lr=1e-7,
        lr_schedule_episodes: Optional[int] = None,
        actor_grad_clip: float = 0.0,
        critic_grad_clip: float = 0.0,
        actor_grad_clip_type: str = "none",
        critic_grad_clip_type: str = "none",
        grad_clip_norm_type: float = 2.0,
        noise_sigma0: float = 1.0,
        noise_floor: float = 0.05,
        noise_plateau: int = 0,
        critic_ema_decay: float = 0.0,
        target_policy_noise: float = 0.1,
        target_policy_clip: float = 0.25,
        activation: str = "silu",
        norm_type: str = "layernorm",
        use_compile: bool = False,
        log_interval_scale: float = 1.0,
        replay_memmap: bool = False,
        **kwargs,
    ):
        # kwargs absorbs unexpected legacy params (e.g., 'paths') without breaking
        if isinstance(device, str):
            device_arg = device.lower()
            if device_arg != "cpu":
                raise ValueError(f"CPU-only Agent: unsupported device '{device}'. Use device='cpu'.")
            device = torch.device("cpu")
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
        self.per_alpha_final = per_alpha_final
        self.per_alpha_ramp_start = max(0, int(per_alpha_ramp_start))
        self.per_alpha_ramp_end = max(0, int(per_alpha_ramp_end))
        self.per_beta_final = per_beta_final
        self.per_alpha_sigmoid = per_alpha_sigmoid
        self.per_priority_scheme = (per_priority_scheme or "standard").lower()
        self.per_huber_kappa = float(per_huber_kappa)
        self.per_priority_clip_pct = float(per_priority_clip_pct)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.min_replay_size = min_replay_size or BATCH_SIZE * 10

        actor_hidden_size = actor_hidden_size or hidden_size
        critic_hidden_size = critic_hidden_size or hidden_size
        if actor_layers < 1:
            raise ValueError(f"actor_layers must be >= 1, got {actor_layers}")
        if critic_layers < 2 and not distributional:
            raise ValueError("critic_layers must be >= 2 when using the standard critic")

        self.activation = (activation or "silu").lower()
        if self.activation not in {"relu", "leaky_relu", "silu"}:
            raise ValueError(
                f"Unsupported activation '{activation}'. Choose from 'relu', 'leaky_relu', or 'silu'."
            )

        self.norm_type = (norm_type or "layernorm").lower()
        if self.norm_type not in {"layernorm", "rmsnorm", "none"}:
            raise ValueError(
                f"Unsupported norm_type '{norm_type}'. Choose from 'layernorm', 'rmsnorm', or 'none'."
            )

        self.use_compile = bool(use_compile)
        self._actor_forward_preact = None
        self._compile_kwargs = {"mode": "reduce-overhead"}

        def _compile(obj):
            if not (self.use_compile and hasattr(torch, "compile")):
                return obj
            try:
                return torch.compile(obj, **self._compile_kwargs)  # type: ignore[misc]
            except TypeError:
                return torch.compile(obj)  # type: ignore[misc]

        optimizer_name = optimizer.lower()
        if optimizer_name not in {"adam", "adamw"}:
            raise ValueError(f"Unsupported optimizer '{optimizer}'. Choose from 'adam', 'adamw'.")
        optim_cls = optim.AdamW if optimizer_name == "adamw" else optim.Adam
        # AdamW betas tuned for actor/critic; fall back to library defaults for Adam.
        self.actor_betas = (0.9, 0.99) if optimizer_name == "adamw" else None
        self.critic_betas = (0.85, 0.99) if optimizer_name == "adamw" else None

        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.optimizer_name = optimizer_name
        self._updates_done = 0
        self.initial_actor_lr = LR_ACTOR
        self.initial_critic_lr = LR_CRITIC
        self.noise_sigma0 = float(noise_sigma0)
        self.noise_floor = float(noise_floor)
        self.noise_plateau = max(0, int(noise_plateau))
        self.target_policy_noise = target_policy_noise
        # Clipping functionality is intentionally disabled; keep attribute for backward-compat.
        self.target_policy_clip = None
        self.log_interval_scale = (
            max(1, int(round(log_interval_scale))) if log_interval_scale and log_interval_scale > 0 else 1
        )
        # Diagnostics/logging cadence (skip most steps to cut overhead; does not affect training)
        base_diag_interval = max(100, int(self.LEARN_EVERY * 50))
        base_td_interval = max(100, int(self.LEARN_EVERY * 50))
        base_per_stats_interval = max(self.LEARN_EVERY * 200, base_diag_interval)
        self.diagnostics_interval = int(base_diag_interval * self.log_interval_scale)
        self.per_stats_interval = int(base_per_stats_interval * self.log_interval_scale)
        self.td_quantile_interval = int(base_td_interval * self.log_interval_scale)
        self.per_step_log_interval = int(max(1, self.LEARN_EVERY * self.log_interval_scale))
        self.collection_progress_interval = int(1000 * self.log_interval_scale)

        self.actor_local = Actor(
            state_size,
            action_size,
            random_seed,
            hidden_size=actor_hidden_size,
            n_layers=actor_layers,
            activation=self.activation,
            norm_type=self.norm_type,
        )
        self.actor_target = Actor(
            state_size,
            action_size,
            random_seed,
            hidden_size=actor_hidden_size,
            n_layers=actor_layers,
            activation=self.activation,
            norm_type=self.norm_type,
        )
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self._actor_action_output = getattr(self.actor_local, "action_output", "tanh")
        self._actor_apply_profitability_gate = getattr(self.actor_local, "apply_profitability_gate", None)

        if self.use_compile and hasattr(torch, "compile"):
            try:
                self._actor_local_orig = self.actor_local
                self.actor_local = _compile(self.actor_local)
                self.actor_target = _compile(self.actor_target)
                # Compile the hot-path used by act(), which calls forward_preact directly.
                self._actor_forward_preact = _compile(self._actor_local_orig.forward_preact)
            except Exception:
                self.use_compile = False
                self._actor_forward_preact = None
        if self._actor_forward_preact is None:
            self._actor_forward_preact = self.actor_local.forward_preact

        self.actor_optimizer = self._build_optimizer(
            model=self.actor_local,
            optim_cls=optim_cls,
            lr=LR_ACTOR,
            weight_decay=weight_decay_actor,
            betas=self.actor_betas,
        )

        if distributional:
            self.N = 32
            self.critic_local = IQN(
                state_size,
                action_size,
                layer_size=critic_hidden_size,
                seed=random_seed,
                dueling=False,
                N=self.N,
                norm_type=self.norm_type,
            ).to(self.device)
            self.critic_target = IQN(
                state_size,
                action_size,
                layer_size=critic_hidden_size,
                seed=random_seed,
                dueling=False,
                N=self.N,
                norm_type=self.norm_type,
            ).to(self.device)
            self.critic_target.load_state_dict(self.critic_local.state_dict())
        else:
            self.critic_local = Critic(
                state_size,
                action_size,
                random_seed,
                hidden_size=critic_hidden_size,
                n_layers=critic_layers,
                activation=self.activation,
                norm_type=self.norm_type,
            )
            self.critic_target = Critic(
                state_size,
                action_size,
                random_seed,
                hidden_size=critic_hidden_size,
                n_layers=critic_layers,
                activation=self.activation,
                norm_type=self.norm_type,
            )
            self.critic_target.load_state_dict(self.critic_local.state_dict())

        if self.use_compile and hasattr(torch, "compile"):
            try:
                self._critic_local_orig = self.critic_local
                self.critic_local = _compile(self.critic_local)
                self.critic_target = _compile(self.critic_target)
            except Exception:
                self.use_compile = False

        self.critic_optimizer = self._build_optimizer(
            model=self.critic_local,
            optim_cls=optim_cls,
            lr=LR_CRITIC,
            weight_decay=weight_decay_critic,
            betas=self.critic_betas,
        )
        self.critic_ema_decay = critic_ema_decay
        self.critic_ema_state = None if critic_ema_decay <= 0 else copy.deepcopy(self.critic_local.state_dict())

        # Cache parameter lists and data views to reduce per-step iteration overhead
        self._actor_params = list(self.actor_local.parameters())
        self._critic_params = list(self.critic_local.parameters())
        self._actor_target_params = list(self.actor_target.parameters())
        self._critic_target_params = list(self.critic_target.parameters())
        self._actor_target_data = [p.data for p in self._actor_target_params]
        self._critic_target_data = [p.data for p in self._critic_target_params]
        self._actor_local_data = [p.data for p in self._actor_params]
        self._critic_local_data = [p.data for p in self._critic_params]

        actor_params = sum(p.numel() for p in self.actor_local.parameters())
        critic_params = sum(p.numel() for p in self.critic_local.parameters())
        optimizer_label = optim_cls.__name__
        print(
            f"Optimizer setup: optimizer={optimizer_label} | "
            f"actor_lr={LR_ACTOR:.2e} (weight_decay={weight_decay_actor:.1e}) | "
            f"critic_lr={LR_CRITIC:.2e} (weight_decay={weight_decay_critic:.1e})"
        )
        print(
            f"Actor params: {actor_params:,} | hidden_size={actor_hidden_size} | layers={actor_layers} | activation={self.activation}"
        )
        print(
            f"Critic params: {critic_params:,} | hidden_size={critic_hidden_size} | layers={'IQN' if distributional else critic_layers} | activation={self.activation if not distributional else 'relu'}"
        )

        self.entropy_tau = 0.03
        self.lo = -1.0
        self.alpha = 0.9

        if per:
            self.memory = PrioritizedReplay(
                BUFFER_SIZE,
                BATCH_SIZE,
                device=device,
                seed=random_seed,
                gamma=GAMMA,
                n_step=n_step,
                parallel_env=1,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
                use_memmap=bool(replay_memmap),
            )
            # Configure PER priority stability
            try:
                self.memory.min_priority = per_priority_floor
            except Exception:
                pass
        else:
            self.memory = CircularReplayBuffer(
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                n_step=n_step,
                parallel_env=1,
                device=device,
                seed=random_seed,
                gamma=GAMMA,
                use_memmap=bool(replay_memmap) or BUFFER_SIZE > 500000,
            )
        self._per_update_priorities = bool(per) and hasattr(self.memory, "update_priorities")
        self._per_has_priority_stats = bool(per) and hasattr(self.memory, "get_priority_stats")

        self.final_lr_fraction = final_lr_fraction
        schedule_horizon = lr_schedule_episodes if lr_schedule_episodes is not None else total_episodes
        self.total_episodes = max(1, int(schedule_horizon)) if schedule_horizon is not None else 65000
        self.warmup_frac = warmup_frac
        self.warmup_episodes = (
            max(1, int(warmup_episodes))
            if warmup_episodes is not None
            else max(1, int(self.total_episodes * warmup_frac))
        )
        self.min_lr = min_lr

        # Clipping functionality (grad clipping) is intentionally disabled.
        self.grad_clip_norm_type = grad_clip_norm_type
        self.actor_grad_clip = None
        self.actor_grad_clip_type = "none"
        self.critic_grad_clip = None
        self.critic_grad_clip_type = "none"

        def lr_lambda(step: int, init_lr: float):
            if final_lr_fraction >= 1.0:
                return 1.0
            episode = step + 1  # align scheduler step with 1-based episode indexing
            if episode <= self.warmup_episodes:
                scale = episode / self.warmup_episodes
            elif episode < self.total_episodes:
                t = (episode - self.warmup_episodes) / max(1, self.total_episodes - self.warmup_episodes)
                cosine = 0.5 * (1.0 + math.cos(math.pi * t))
                scale = final_lr_fraction + (1.0 - final_lr_fraction) * cosine
            else:
                scale = final_lr_fraction
            return max(min_lr / init_lr, scale)

        if final_lr_fraction < 1.0:
            self.actor_scheduler = optim.lr_scheduler.LambdaLR(
                self.actor_optimizer, lr_lambda=lambda s: lr_lambda(s, LR_ACTOR)
            )
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(
                self.critic_optimizer, lr_lambda=lambda s: lr_lambda(s, LR_CRITIC)
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None

        self.learn = self.learn_distribution if distributional else self.learn_
        self.step_counter = 0
        self._last_td_percentiles = None
        self._last_target_drift = None
        self._last_iqn_spread = None
        self._episode_count = 0

    def _build_optimizer(
        self,
        model: torch.nn.Module,
        optim_cls,
        lr: float,
        weight_decay: float,
        betas: Optional[Tuple[float, float]] = None,
    ):
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

        optim_kwargs = {"lr": lr}
        if betas is not None:
            optim_kwargs["betas"] = betas
        optim_kwargs_foreach = dict(optim_kwargs)
        optim_kwargs_foreach["foreach"] = True

        if not param_groups:
            try:
                return optim_cls(model.parameters(), weight_decay=0.0, **optim_kwargs_foreach)
            except TypeError:
                return optim_cls(model.parameters(), weight_decay=0.0, **optim_kwargs)
        try:
            return optim_cls(param_groups, **optim_kwargs_foreach)
        except TypeError:
            return optim_cls(param_groups, **optim_kwargs)

    def update_episode_count(self, episode: int):
        """Update internal episode counter (used for PER beta annealing in caller)."""
        self._episode_count = episode
        self._maybe_update_per_schedule()

    def step_lr_schedulers(self, episode: int):
        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()

    def boost_learning_rates(self, boost: float):
        """Boost learning rates up to their initial values (safety cap)."""
        if boost <= 1.0:
            return
        max_actor = self.initial_actor_lr
        max_critic = self.initial_critic_lr
        for g in self.actor_optimizer.param_groups:
            g["lr"] = min(max_actor, g["lr"] * boost)
        for g in self.critic_optimizer.param_groups:
            g["lr"] = min(max_critic, g["lr"] * boost)

    def _pre_noise_sigma(self) -> float:
        """Compute pre-squash noise std based on episode schedule."""
        e = max(1, self._episode_count)
        plateau = max(0, self.noise_plateau)
        if plateau > 0 and e < plateau:
            return self.noise_sigma0
        t = e - plateau if plateau > 0 else e
        denom = 1.0 + (t / max(1, plateau if plateau > 0 else 1))
        return self.noise_floor + (self.noise_sigma0 - self.noise_floor) / denom

    def _squash(self, preact: torch.Tensor) -> torch.Tensor:
        """Map pre-activation values to action space matching actor output activation."""
        action_output = self._actor_action_output
        if action_output == "tanh01":
            return 0.5 * (torch.tanh(preact) + 1.0)
        if action_output == "sigmoid":
            return torch.sigmoid(preact)
        return torch.tanh(preact)

    def _should_log(self, timestamp: int, interval: int) -> bool:
        interval = max(1, int(interval))
        return timestamp % interval == 0

    def _compute_base_priorities(
        self,
        td_or_loss: torch.Tensor,
        *,
        floor: float,
        clip_pct: float,
        scheme: str,
        huber_kappa: float,
        is_loss: bool = False,
    ) -> torch.Tensor:
        if is_loss:
            priorities = td_or_loss.detach()
        elif scheme == "lap":
            priorities = calculate_huber_loss(td_or_loss, k=huber_kappa).detach()
        else:
            priorities = td_or_loss.detach().abs()

        floor_f = float(floor)
        priorities = torch.nan_to_num(priorities, nan=floor_f, posinf=1e6, neginf=floor_f)
        priorities = priorities.clamp_min(floor_f)
        if clip_pct and clip_pct > 0:
            q = float(clip_pct) / 100.0
            thr = torch.quantile(priorities.flatten(), q)
            priorities = priorities.clamp_max(thr)
            priorities = priorities.clamp_min(floor_f)
        return priorities

    def act(self, state: np.ndarray, add_noise: bool = True):
        self._maybe_init_profitability_params()
        # Avoid train()/eval() toggling (expensive) while keeping inference_mode for no-grad
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.get_default_dtype())
        with torch.inference_mode():
            preact = self._actor_forward_preact(state_t)  # type: ignore[misc]
            if add_noise:
                sigma = self._pre_noise_sigma()
                if sigma > 0:
                    preact = preact + torch.randn_like(preact) * sigma
            # Exploration noise is applied pre-gate; the executed action is always profitability-gated.
            q_raw = self._squash(preact)
            apply_gate = self._actor_apply_profitability_gate
            if apply_gate is not None:
                action_t = apply_gate(q_raw=q_raw, state=state_t)
            else:
                action_t = q_raw
        action = action_t.cpu().numpy()
        np.clip(action, 0.0, 1.0, out=action)
        return action

    def step(self, state, action, reward, next_state, done, timestamp, writer):
        self.step_counter += 1
        self.memory.add(state, action, reward, next_state, done)
        writer_available = writer is not None
        log_step = timestamp  # keep TB index aligned with true environment step count
        if len(self.memory) < self.min_replay_size or len(self.memory) <= self.BATCH_SIZE:
            if writer_available and self._should_log(timestamp, self.collection_progress_interval):
                writer.add_scalar(
                    "Collection_Progress", len(self.memory) / self.min_replay_size * 100, log_step
                )
            return
        if timestamp % self.LEARN_EVERY != 0:
            return
        last_batch = None
        losses = None
        for _ in range(self.LEARN_NUMBER):
            last_batch = self.memory.sample()
            losses = self.learn(last_batch, self.GAMMA)
        if losses and writer_available and self._should_log(timestamp, self.per_step_log_interval):
            writer.add_scalar("Critic_loss", losses[0], log_step)
            writer.add_scalar("Actor_loss", losses[1], log_step)
        if last_batch and writer_available and self._should_log(timestamp, self.per_step_log_interval):
            self._log_batch_diagnostics(last_batch, log_step, writer)
        if (
            self._per_has_priority_stats
            and writer_available
            and self._should_log(timestamp, self.per_stats_interval)
        ):
            for k, v in self.memory.get_priority_stats().items():
                writer.add_scalar(f"PER/{k}", v, log_step)

    def learn_(self, experiences, gamma) -> Tuple[float, float]:
        self._maybe_init_profitability_params()
        states, actions, rewards, next_states, dones, idx, weights = experiences
        if states.device.type != self.device.type or actions.device.type != self.device.type:
            raise RuntimeError(
                f"Device mismatch: batch on {states.device}/{actions.device}, agent on {self.device}"
            )
        if (
            rewards.device.type != self.device.type
            or next_states.device.type != self.device.type
            or dones.device.type != self.device.type
        ):
            raise RuntimeError("Device mismatch: replay buffer returned CPU tensors for a non-CPU agent")
        if weights is not None and weights.device.type != self.device.type:
            raise RuntimeError("Device mismatch: PER weights not on agent device")

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            if self.target_policy_noise and self.target_policy_noise > 0:
                # No clipping of target policy noise (intentional).
                noise = torch.randn_like(next_actions) * self.target_policy_noise
                next_actions = next_actions + noise
            q_next = self.critic_target(next_states, next_actions)
            if not self.munchausen:
                q_target = rewards + (gamma**self.n_step) * q_next * (1 - dones.float())
            else:
                logsum = torch.logsumexp(q_next / self.entropy_tau, dim=1, keepdim=True)
                tau_log_pi_next = q_next - self.entropy_tau * logsum
                pi = F.softmax(q_next / self.entropy_tau, dim=1)
                q_target = rewards + (self.GAMMA**self.n_step) * (
                    pi * (q_next - tau_log_pi_next) * (1 - dones.float())
                )
        q_expected = self.critic_local(states, actions)
        if self.per:
            td = q_target - q_expected
            critic_loss = (td.pow(2) * weights).mean()
            if self.step_counter % self.td_quantile_interval == 0:
                with torch.no_grad():
                    abs_td = td.detach().abs().flatten()
                    if abs_td.numel() > 10:
                        self._last_td_percentiles = (
                            torch.quantile(abs_td, 0.5).item(),
                            torch.quantile(abs_td, 0.9).item(),
                            torch.quantile(abs_td, 0.99).item(),
                        )
            floor = float(getattr(self.memory, "min_priority", 1e-6))
            priorities = self._compute_base_priorities(
                td,
                floor=floor,
                clip_pct=self.per_priority_clip_pct,
                scheme=self.per_priority_scheme,
                huber_kappa=self.per_huber_kappa,
            )
        else:
            critic_loss = F.mse_loss(q_expected, q_target)
            priorities = None
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        q_val = self.critic_local(states, actions_pred)
        actor_loss = -q_val.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self._updates_done += 1
        if self.step_counter % 200 == 0:
            with torch.no_grad():
                tgt_q = self.critic_target(states, self.actor_target(states))
                self._last_target_drift = (q_expected - tgt_q).abs().mean().item()
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        self._update_ema_buffers()
        if self._per_update_priorities and priorities is not None:
            self.memory.update_priorities(idx, priorities.cpu().numpy().flatten())
        return critic_loss.item(), actor_loss.item()

    def learn_distribution(self, experiences, gamma) -> Tuple[float, float]:
        self._maybe_init_profitability_params()
        states, actions, rewards, next_states, dones, idx, weights = experiences
        if states.device.type != self.device.type or actions.device.type != self.device.type:
            raise RuntimeError(
                f"Device mismatch: batch on {states.device}/{actions.device}, agent on {self.device}"
            )
        if (
            rewards.device.type != self.device.type
            or next_states.device.type != self.device.type
            or dones.device.type != self.device.type
        ):
            raise RuntimeError("Device mismatch: replay buffer returned CPU tensors for a non-CPU agent")
        if weights is not None and weights.device.type != self.device.type:
            raise RuntimeError("Device mismatch: PER weights not on agent device")

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            if self.target_policy_noise and self.target_policy_noise > 0:
                # No clipping of target policy noise (intentional).
                noise = torch.randn_like(next_actions) * self.target_policy_noise
                next_actions = next_actions + noise
            qt_next, _ = self.critic_target(next_states, next_actions, self.N)
            qt_next = qt_next.transpose(1, 2)
            if not self.munchausen:
                q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step) * qt_next * (
                    1 - dones.float().unsqueeze(-1)
                )
            else:
                q_mean = qt_next.mean(-1)
                logsum = torch.logsumexp(q_mean / self.entropy_tau, dim=1, keepdim=True)
                tau_log_pi_next = (q_mean - self.entropy_tau * logsum).unsqueeze(1)
                pi_target = F.softmax(q_mean / self.entropy_tau, dim=1).unsqueeze(1)
                q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step) * (
                    pi_target * (qt_next - tau_log_pi_next) * (1 - dones.float().unsqueeze(-1))
                )
        q_expected, taus = self.critic_local(states, actions, self.N)
        td_error = q_targets - q_expected
        huber = calculate_huber_loss(td_error, 1.0)
        quantile_loss = (torch.abs(taus - (td_error.detach() < 0).float()) * huber).sum(dim=1).mean(dim=1)
        if self.per:
            critic_loss = (quantile_loss.unsqueeze(1) * weights).mean()
        else:
            critic_loss = quantile_loss.mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        q_pred, _ = self.critic_local(states, actions_pred, self.N)
        actor_loss = -q_pred.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self._updates_done += 1
        if self._per_update_priorities:
            floor = float(getattr(self.memory, "min_priority", 1e-6))
            if self.per_priority_scheme == "lap":
                huber_pr = calculate_huber_loss(td_error, k=self.per_huber_kappa)
                quantile_pr = (
                    torch.abs(taus - (td_error.detach() < 0).float()) * huber_pr
                ).sum(dim=1).mean(dim=1)
                priorities = self._compute_base_priorities(
                    quantile_pr,
                    floor=floor,
                    clip_pct=self.per_priority_clip_pct,
                    scheme=self.per_priority_scheme,
                    huber_kappa=self.per_huber_kappa,
                    is_loss=True,
                )
            else:
                td_proxy = td_error.mean(dim=(1, 2))
                priorities = self._compute_base_priorities(
                    td_proxy,
                    floor=floor,
                    clip_pct=self.per_priority_clip_pct,
                    scheme=self.per_priority_scheme,
                    huber_kappa=self.per_huber_kappa,
                )
            self.memory.update_priorities(idx, priorities.cpu().numpy().flatten())
        if self.step_counter % self.td_quantile_interval == 0:
            with torch.no_grad():
                flat = q_targets.view(q_targets.size(0), -1)
                q10 = torch.quantile(flat, 0.1, dim=1).mean().item()
                q50 = torch.quantile(flat, 0.5, dim=1).mean().item()
                q90 = torch.quantile(flat, 0.9, dim=1).mean().item()
                self._last_iqn_spread = (q10, q50, q90, (q90 - q10))
        if self.step_counter % self.diagnostics_interval == 0:
            with torch.no_grad():
                tgt_q, _ = self.critic_target(states, self.actor_target(states), self.N)
                self._last_target_drift = (q_expected - tgt_q).abs().mean().item()
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        self._update_ema_buffers()
        return critic_loss.item(), actor_loss.item()

    @property
    def updates_done(self) -> int:
        return self._updates_done

    def soft_update(self, local, target):
        tau = self._compute_tau()
        with torch.no_grad():
            if local is self.critic_local and target is self.critic_target:
                tgt_params = self._critic_target_data
                src_params = self._critic_local_data
            elif local is self.actor_local and target is self.actor_target:
                tgt_params = self._actor_target_data
                src_params = self._actor_local_data
            else:
                tgt_params = [p.data for p in target.parameters()]
                src_params = [p.data for p in local.parameters()]
            try:
                torch._foreach_mul_(tgt_params, 1 - tau)
                torch._foreach_add_(tgt_params, src_params, alpha=tau)
            except Exception:
                for tp, lp in zip(tgt_params, src_params):
                    tp.mul_(1 - tau)
                    tp.add_(lp, alpha=tau)

    def _compute_tau(self) -> float:
        return self.t

    def _update_ema_buffers(self):
        if self.critic_ema_decay <= 0 or self.critic_ema_state is None:
            return
        with torch.no_grad():
            for k, v in self.critic_local.state_dict().items():
                self.critic_ema_state[k].mul_(self.critic_ema_decay).add_(v, alpha=1.0 - self.critic_ema_decay)

    def _maybe_update_per_schedule(self):
        """Dynamically adjust PER alpha/beta to mimic uniform early and ramp later."""
        if not self.per or not hasattr(self, "memory") or not isinstance(self.memory, PrioritizedReplay):
            return
        # If no schedule configured, keep defaults
        if self.per_alpha_final is None or self.per_alpha_ramp_end <= self.per_alpha_ramp_start:
            return
        start = self.per_alpha_ramp_start
        end = self.per_alpha_ramp_end
        ep = self._episode_count
        if ep <= start:
            alpha = 0.0
        elif ep >= end:
            alpha = self.per_alpha_final
        else:
            frac = (ep - start) / max(1, end - start)
            if self.per_alpha_sigmoid:
                k = 8.0 / max(1.0, end - start)
                mid = 0.5 * (start + end)
                alpha = self.per_alpha_final / (1.0 + math.exp(-k * (ep - mid)))
            else:
                alpha = frac * self.per_alpha_final
        beta_final = self.per_beta_final if self.per_beta_final is not None else 1.0
        if ep <= start:
            beta = 1.0
        elif ep >= end:
            beta = beta_final
        else:
            frac = (ep - start) / max(1, end - start)
            if self.per_alpha_sigmoid:
                k = 8.0 / max(1.0, end - start)
                mid = 0.5 * (start + end)
                frac_sig = 1.0 / (1.0 + math.exp(-k * (ep - mid)))
                beta = 1.0 + (beta_final - 1.0) * frac_sig
            else:
                beta = 1.0 + (beta_final - 1.0) * frac
        # Apply to memory and invalidate caches so sampling uses updated alpha
        self.memory.alpha = alpha
        self.memory.beta_start = beta
        self.memory.beta_frames = int(1e9)  # effectively constant beta
        self.memory.set_frame_count(0)
        self.memory._cache_valid = False
        self.memory._cumsum_valid = False

    def _maybe_init_profitability_params(self) -> None:
        """Best-effort init of Actor profitability gate params from an env/contract in the call stack.

        Keeps changes localized (no extra wiring): if an object with a `.contract` is present in
        the caller stack (e.g., run_training(train_env=...)), we pull q/cost params from it.
        """
        if getattr(self, "_profitability_params_initialized", False):
            return

        # Allow explicit wiring by setting self.contract or self.env externally.
        contract = getattr(self, "contract", None)
        if contract is None:
            env = getattr(self, "env", None)
            contract = getattr(env, "contract", None) if env is not None else None

        if contract is None:
            import inspect

            frame = inspect.currentframe()
            try:
                while frame is not None:
                    for obj in frame.f_locals.values():
                        cand = getattr(obj, "contract", None)
                        if cand is None and hasattr(obj, "unwrapped"):
                            cand = getattr(getattr(obj, "unwrapped", None), "contract", None)
                        if cand is None:
                            cand = obj
                        if all(hasattr(cand, k) for k in ("q_min", "q_max", "c_cost", "gamma_cost")):
                            contract = cand
                            break
                    if contract is not None:
                        break
                    frame = frame.f_back
            finally:
                del frame

        if contract is None:
            return

        try:
            q_min = float(getattr(contract, "q_min"))
            q_max = float(getattr(contract, "q_max"))
            c_cost = float(getattr(contract, "c_cost"))
            gamma_cost = float(getattr(contract, "gamma_cost"))
        except Exception:
            return

        for actor in (self.actor_local, self.actor_target):
            if hasattr(actor, "set_profitability_params"):
                actor.set_profitability_params(
                    q_min=q_min,
                    q_max=q_max,
                    c_cost=c_cost,
                    gamma_cost=gamma_cost,
                    s_minus_k_index=0,  # SwingOptionEnv puts (S-K) at state[0]
                )

        self._profitability_params_initialized = True

    def get_critic_eval_state(self):
        """Return EMA-smoothed critic parameters for evaluation if available."""
        return self.critic_ema_state if self.critic_ema_state is not None else self.critic_local.state_dict()

    def _log_batch_diagnostics(self, batch, ts, writer):
        if ts % self.diagnostics_interval != 0:
            return
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
        """Reset agent state between episodes (placeholder for future use)."""
        return


def calculate_huber_loss(td_errors, k=1.0):
    return torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))


def calc_fraction_loss(FZ_, FZ, taus, weights=None):
    gradients1 = FZ - FZ_[:, :-1]
    gradients2 = FZ - FZ_[:, 1:]
    flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
    flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
    gradients = (
        torch.where(flag_1, gradients1, -gradients1) + torch.where(flag_2, gradients2, -gradients2)
    ).view(taus.shape[0], 31)
    if weights is not None:
        loss = ((gradients * taus[:, 1:-1]).sum(dim=1) * weights).mean()
    else:
        loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
    return loss
