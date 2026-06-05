import contextlib
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    from .networks import Actor, Critic, HHKInputLayer
    from .replay_buffer import CircularReplayBuffer
except ImportError:
    from networks import Actor, Critic, HHKInputLayer
    from replay_buffer import CircularReplayBuffer


class Agent:
    """
    Stable Agent implementation with diagnostics and lightweight defaults (2×64 MLPs).
    Defaults: single-step TD targets, gradient clipping is disabled unless thresholds
    are provided, actor/critic weight decay defaults to 5e-5/1e-4, and target updates
    use a smoother tau=0.002.
    """

    def __init__(
        self,
        state_size,
        action_size,
        n_step,
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
        final_lr_fraction=1.0,
        total_episodes=None,
        warmup_frac: float = 0.05,
        warmup_episodes: Optional[int] = None,
        min_lr=1e-7,
        lr_schedule_episodes: Optional[int] = None,
        noise_sigma0: float = 1.0,
        noise_floor: float = 0.05,
        noise_plateau: int = 0,
        activation: str = "silu",
        norm_type: str = "layernorm",
        init_method: str = "orthogonal",
        use_compile: bool = False,
        log_interval_scale: float = 1.0,
        replay_memmap: bool = False,
        single_critic_step: bool = True,    # v63 audit E4: single critic step is now canonical
        critic_warmup_episodes: int = 0,
        adaptive_noise_scale: float = 0.0,
        # Task 2 — scheduler simplification (all default to current behavior).
        noise_schedule: str = "hyperbolic",   # {hyperbolic, const_floor, linear}
        noise_decay_episodes: Optional[int] = None,  # horizon for the 'linear' noise decay
        weight_averaging: str = "off",         # {off, ema}; eval-only EMA of actor weights
        ema_decay: float = 0.999,
        warmup_noise_fraction: float = 1.0,  # v60: reduce noise during critic warmup
        action_output: str = "tanh01",        # v60: actor output activation
        use_robust_normalization: bool = False, # v62: Robust HHK Normalization
        strike: float = 100.0,
        expected_target_kernel=None,           # feat/semi-analytical-bootstrap
        expected_target_chunk_size: Optional[int] = None,
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
        self.GAMMA = GAMMA
        self.t = t
        self.n_step = n_step
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARN_EVERY = LEARN_EVERY
        self.LEARN_NUMBER = LEARN_NUMBER
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.min_replay_size = min_replay_size or BATCH_SIZE * 10

        actor_hidden_size = actor_hidden_size or hidden_size
        critic_hidden_size = critic_hidden_size or hidden_size
        if actor_layers < 1:
            raise ValueError(f"actor_layers must be >= 1, got {actor_layers}")
        if critic_layers < 2:
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
        self.init_method = (init_method or "orthogonal").lower()
        if self.init_method == "kaiming":
            self.init_method = "he"
        if self.init_method not in {"orthogonal", "he"}:
            raise ValueError(f"Unsupported init_method '{init_method}'. Choose from 'orthogonal' or 'he'.")

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
        # Task 2 — noise/LR/weight-averaging schedule knobs (defaults = current behavior).
        self.noise_schedule = (noise_schedule or "hyperbolic").lower()
        if self.noise_schedule not in {"hyperbolic", "const_floor", "linear"}:
            raise ValueError(f"Unsupported noise_schedule '{noise_schedule}'.")
        self._noise_decay_episodes_arg = noise_decay_episodes
        self.weight_averaging = (weight_averaging or "off").lower()
        if self.weight_averaging not in {"off", "ema"}:
            raise ValueError(f"Unsupported weight_averaging '{weight_averaging}'.")
        self.ema_decay = float(ema_decay)
        self.log_interval_scale = (
            max(1, int(round(log_interval_scale))) if log_interval_scale and log_interval_scale > 0 else 1
        )
        # Diagnostics/logging cadence (skip most steps to cut overhead; does not affect training)
        base_diag_interval = max(100, int(self.LEARN_EVERY * 50))
        self.diagnostics_interval = int(base_diag_interval * self.log_interval_scale)
        self.per_step_log_interval = int(max(1, self.LEARN_EVERY * self.log_interval_scale))
        self.collection_progress_interval = int(1000 * self.log_interval_scale)

        self.single_critic_step = bool(single_critic_step)
        self.action_output = (action_output or "tanh01").lower()  # v60: store action output activation
        actor_cls = Actor
        critic_cls = Critic

        # ── Semi-analytical TD target (feat/semi-analytical-bootstrap, H1) ──
        self._strike = float(strike)
        self.expected_target_kernel = expected_target_kernel
        self.expected_target_chunk_size = expected_target_chunk_size
        if self.expected_target_kernel is not None and n_step != 1:
            print(f"Agent: expected_target_kernel set; forcing n_step=1 (was {n_step}).")
            self.n_step = 1

        # Configure Input Preprocessor if enabled
        input_preprocessor = None
        if use_robust_normalization:
            input_preprocessor = HHKInputLayer(
                state_dim=state_size,
                strike=strike,
                x_median=0.0, x_iqr=1.0,  # Defaults, could be tuned via args/stats
                y_median=0.0, y_iqr=1.0,
            )
            print(f"Agent: Enabled Robust HHK Normalization (Log-Moneyness, Strike={strike})")

        self.actor_local = actor_cls(
            state_size,
            action_size,
            random_seed,
            hidden_size=actor_hidden_size,
            n_layers=actor_layers,
            action_output=self.action_output,  # v60: pass action output activation
            activation=self.activation,
            norm_type=self.norm_type,
            init_method=self.init_method,
            input_preprocessor=input_preprocessor,
        )
        self.actor_target = actor_cls(
            state_size,
            action_size,
            random_seed,
            hidden_size=actor_hidden_size,
            n_layers=actor_layers,
            action_output=self.action_output,  # v60: pass action output activation
            activation=self.activation,
            norm_type=self.norm_type,
            init_method=self.init_method,
            input_preprocessor=input_preprocessor,
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

        self.critic_local = critic_cls(
            state_size,
            action_size,
            random_seed,
            hidden_size=critic_hidden_size,
            n_layers=critic_layers,
            activation=self.activation,
            norm_type=self.norm_type,
            init_method=self.init_method,
            input_preprocessor=input_preprocessor,
        )
        self.critic_target = critic_cls(
            state_size,
            action_size,
            random_seed,
            hidden_size=critic_hidden_size,
            n_layers=critic_layers,
            activation=self.activation,
            norm_type=self.norm_type,
            init_method=self.init_method,
            input_preprocessor=input_preprocessor,
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

        # Cache parameter lists and data views to reduce per-step iteration overhead
        self._actor_params = list(self.actor_local.parameters())
        self._critic_params = list(self.critic_local.parameters())
        self._actor_target_params = list(self.actor_target.parameters())
        self._critic_target_params = list(self.critic_target.parameters())
        self._actor_target_data = [p.data for p in self._actor_target_params]
        self._critic_target_data = [p.data for p in self._critic_target_params]
        self._actor_local_data = [p.data for p in self._actor_params]
        self._critic_local_data = [p.data for p in self._critic_params]
        # Eval-only EMA shadow of actor weights (Task 2 L_ema). A plain tensor list
        # (not a module) avoids torch.compile aliasing; swapped into actor_local at
        # eval time via averaged_eval_actor(). None unless weight_averaging=='ema'.
        self._actor_ema_data = (
            [p.detach().clone() for p in self._actor_local_data]
            if self.weight_averaging == "ema" else None
        )

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
            f"Critic params: {critic_params:,} | hidden_size={critic_hidden_size} | layers={critic_layers} | activation={self.activation}"
        )

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

        self.final_lr_fraction = final_lr_fraction
        schedule_horizon = lr_schedule_episodes if lr_schedule_episodes is not None else total_episodes
        self.total_episodes = max(1, int(schedule_horizon)) if schedule_horizon is not None else 65000
        # Horizon for the 'linear' noise decay (Task 2 N_fulldecay). Defaults to the
        # LR schedule horizon, but the real training horizon (n_paths) should be passed
        # explicitly so the decay actually completes within the run.
        self.noise_decay_episodes = (
            max(1, int(self._noise_decay_episodes_arg))
            if self._noise_decay_episodes_arg
            else self.total_episodes
        )
        self.warmup_frac = warmup_frac
        self.warmup_episodes = (
            max(1, int(warmup_episodes))
            if warmup_episodes is not None
            else max(1, int(self.total_episodes * warmup_frac))
        )
        self.min_lr = min_lr

        # Gradient clipping was removed in v63: with the deterministic kernel TD
        # target (--use_expected_target=1), TD errors are bounded and clipping
        # never triggered. The clip args were already overwritten to None and
        # learn_() never called clip_grad_norm_, so every v63 result is clip-free.

        # v59/v60 parameters
        self.critic_warmup_episodes = int(critic_warmup_episodes)
        self.adaptive_noise_scale = float(adaptive_noise_scale)
        # v60: warmup noise reduction - reduce exploration noise during critic warmup to preserve calibration
        self.warmup_noise_fraction = float(warmup_noise_fraction)

        # LR is CONSTANT. The cosine/linear LR-decay schedule was removed (Task 2 Screen B,
        # HPT "Task 2 — scheduler simplification"): at the 4096-ep horizon decay was
        # neutral-to-harmful (cosine drifted negative; linear decay-to-0 FAILED nocost),
        # so constant LR is canonical and the most C++-port-friendly choice. The
        # final_lr_fraction / lr_schedule_episodes / min_lr args are still accepted for
        # script/config compatibility but are now INERT (they no longer affect the LR).
        self.actor_scheduler = None
        self.critic_scheduler = None
        if final_lr_fraction < 1.0:
            print("Agent: LR decay was removed (Task 2); final_lr_fraction<1 is now inert — "
                  "running CONSTANT LR.")

        self.learn = self.learn_
        self.step_counter = 0
        self._last_td_percentiles = None
        self._last_target_drift = None
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
        """Update internal episode counter (drives noise/warmup schedules)."""
        self._episode_count = episode

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
        """Compute pre-squash noise std based on episode schedule.
        
        v60: During critic warmup phase, noise is reduced by warmup_noise_fraction
        to preserve the calibrated actor policy while still allowing some exploration.
        
        v61: Gradual warmup noise ramp. Instead of a flat fraction, noise ramps
        linearly from warmup_noise_fraction at start to 1.0 at end of warmup.
        This provides smooth transition from calibration preservation to full exploration.
        """
        e = max(1, self._episode_count)
        plateau = max(0, self.noise_plateau)
        sched = self.noise_schedule
        if plateau > 0 and e < plateau:
            # All schedules hold sigma0 during the plateau.
            sigma = self.noise_sigma0
        elif sched == "const_floor":
            # Task 2 N_const: hard step to the floor after the plateau (no tail).
            sigma = self.noise_floor
        elif sched == "linear":
            # Task 2 N_fulldecay: linear sigma0->floor over [plateau, horizon].
            horizon = max(plateau + 1, self.noise_decay_episodes)
            frac = min(1.0, max(0.0, (e - plateau) / max(1, horizon - plateau)))
            sigma = self.noise_sigma0 + (self.noise_floor - self.noise_sigma0) * frac
        else:
            # hyperbolic (default, unchanged)
            t = e - plateau if plateau > 0 else e
            denom = 1.0 + (t / max(1, plateau if plateau > 0 else 1))
            sigma = self.noise_floor + (self.noise_sigma0 - self.noise_floor) / denom

        # v61: Gradual warmup noise ramp (was flat fraction in v60)
        # Ramp from warmup_noise_fraction to 1.0 over the critic warmup period
        if e <= self.critic_warmup_episodes and self.warmup_noise_fraction < 1.0:
            warmup_progress = e / max(1, self.critic_warmup_episodes)
            # Linear ramp: start at warmup_noise_fraction, end at 1.0
            current_fraction = self.warmup_noise_fraction + (1.0 - self.warmup_noise_fraction) * warmup_progress
            sigma *= current_fraction
        
        return sigma

    def _squash(self, preact: torch.Tensor) -> torch.Tensor:
        """Map pre-activation values to action space matching actor output activation."""
        action_output = self._actor_action_output
        if action_output == "tanh01":
            return 0.5 * (torch.tanh(preact) + 1.0)
        if action_output == "sigmoid":
            return torch.sigmoid(preact)
        if action_output.startswith("beta_sigmoid"):
            # Match the β-sigmoid from Actor (default β=2.0)
            beta = getattr(self.actor_local, "_output_beta", 2.0)
            return torch.sigmoid(beta * preact)
        return torch.tanh(preact)

    def _should_log(self, timestamp: int, interval: int) -> bool:
        interval = max(1, int(interval))
        return timestamp % interval == 0

    def act(self, state: np.ndarray, add_noise: bool = True):
        self._maybe_init_profitability_params()
        # Avoid train()/eval() toggling (expensive) while keeping inference_mode for no-grad
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.get_default_dtype())
        with torch.inference_mode():
            preact = self._actor_forward_preact(state_t)  # type: ignore[misc]
            if add_noise:
                sigma = self._pre_noise_sigma()
                if sigma > 0:
                    # Fix 3: Adaptive Pre-Squash Noise
                    # Scale noise by (1 + scale * |u|) to prevent saturation lock-in
                    if self.adaptive_noise_scale > 0:
                        scale = 1.0 + self.adaptive_noise_scale * preact.abs()
                        noise = torch.randn_like(preact) * sigma * scale
                    else:
                        noise = torch.randn_like(preact) * sigma
                    preact = preact + noise
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

    def learn_(self, experiences, gamma) -> Tuple[float, float]:
        self._maybe_init_profitability_params()
        states, actions, rewards, next_states, dones, _idx, _weights = experiences
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

        with torch.no_grad():
            if self.expected_target_kernel is not None:
                # Variance-reduced bootstrap via the precomputed quadrature kernel.
                from .transition_kernel import expected_critic_target
                q_next = expected_critic_target(
                    self.critic_target, self.actor_target,
                    states=states, next_states=next_states,
                    kernel=self.expected_target_kernel,
                    strike=self._strike,
                    chunk_size=self.expected_target_chunk_size,
                )
                q_target = rewards + (gamma ** self.n_step) * q_next * (1 - dones.float())
            else:
                next_actions = self.actor_target(next_states)
                q_next = self.critic_target(next_states, next_actions)
                q_target = rewards + (gamma**self.n_step) * q_next * (1 - dones.float())
        q_expected = self.critic_local(states, actions)
        critic_loss = (q_expected - q_target).pow(2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # v63 audit E4: a legacy duplicate self.critic_optimizer.step() lived here
        # (un-zeroed grad => ~2 effective critic steps/learn). A 12-seed x 3-regime
        # screen (lr_c=3e-4) showed single-step strictly beats the double-step on
        # mean Δ%, seed std, AND worst-case in every regime — so single-step is now
        # the default. single_critic_step=False restores the legacy double-step only
        # to reproduce pre-fix v63 numbers. See HPT.md "v63 Runtime Feature Audit".
        if not self.single_critic_step:
            self.critic_optimizer.step()

        actor_loss_val = 0.0
        # Critic warmup: skip actor updates until the critic is stable.
        if self._episode_count > self.critic_warmup_episodes:
             actions_pred = self.actor_local(states)
             q_val = self.critic_local(states, actions_pred)
             actor_loss = -q_val.mean()

             self.actor_optimizer.zero_grad(set_to_none=True)
             actor_loss.backward()
             self.actor_optimizer.step()
             actor_loss_val = actor_loss.item()
             self.soft_update(self.actor_local, self.actor_target)
             if self._actor_ema_data is not None:
                 self._update_ema()

        self._updates_done += 1
        if self.step_counter % 200 == 0:
            with torch.no_grad():
                tgt_q = self.critic_target(states, self.actor_target(states))
                self._last_target_drift = (q_expected - tgt_q).abs().mean().item()
        self.soft_update(self.critic_local, self.critic_target)
        return critic_loss.item(), actor_loss_val

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

    def _update_ema(self) -> None:
        """EMA-blend the actor-weight shadow toward the current local actor (Task 2 L_ema).
        ema <- decay*ema + (1-decay)*local. Eval-only: never feeds training/target nets."""
        d = self.ema_decay
        with torch.no_grad():
            try:
                torch._foreach_mul_(self._actor_ema_data, d)
                torch._foreach_add_(self._actor_ema_data, self._actor_local_data, alpha=1.0 - d)
            except Exception:
                for ep, lp in zip(self._actor_ema_data, self._actor_local_data):
                    ep.mul_(d)
                    ep.add_(lp, alpha=1.0 - d)

    @contextlib.contextmanager
    def averaged_eval_actor(self):
        """Context manager: temporarily load the EMA actor weights into actor_local for
        evaluation, restoring the training weights afterward. No-op unless EMA is on."""
        if self.weight_averaging != "ema" or self._actor_ema_data is None:
            yield
            return
        backup = [p.detach().clone() for p in self._actor_local_data]
        try:
            with torch.no_grad():
                for dst, src in zip(self._actor_local_data, self._actor_ema_data):
                    dst.copy_(src)
            yield
        finally:
            with torch.no_grad():
                for dst, b in zip(self._actor_local_data, backup):
                    dst.copy_(b)

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
        """Return critic parameters for evaluation."""
        return self.critic_local.state_dict()

    def _log_batch_diagnostics(self, batch, ts, writer):
        if ts % self.diagnostics_interval != 0:
            return
        states, actions, rewards, next_states, dones, idx, weights = batch
        if torch.is_tensor(actions):
            with torch.no_grad():
                action_output = getattr(self, "_actor_action_output", "tanh")
                if action_output in {"tanh01", "sigmoid"}:
                    low_thresh, high_thresh = 0.01, 0.99
                else:
                    low_thresh, high_thresh = -0.99, 0.99
                at_low = (actions <= low_thresh).float().mean().item()
                at_high = (actions >= high_thresh).float().mean().item()
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

    def calibrate_bias(
        self,
        env,
        n_episodes: int = 4096,
        max_iterations: int = 20,
        min_iterations: int = 10,
        target_std: float = 0.005,
        tolerance_percent: float = 0.001,
        epsilon_scale: float = 0.01,
        mode: str = "closed_form",
    ) -> None:
        """
        Warm-start the Actor output bias on the warmup dataset.

        mode="closed_form" (default): O(1)-pass myopic warm-start from the convex-cost
            FOC, averaged over the warmup spots (samples of the HHK/kernel density).
        mode="rprop": legacy 20-iteration Rprop finite-difference loop (bit-identical to
            the pre-change path; ~60+ batch price rollouts).
        """
        mode = (mode or "closed_form").lower()
        if mode not in {"closed_form", "rprop"}:
            raise ValueError(f"Unsupported calibrate_bias mode '{mode}'. Choose 'closed_form' or 'rprop'.")
        if mode == "closed_form":
            return self._calibrate_bias_closed_form(env, n_episodes, target_std=target_std)
        print(f"\n🧪 Calibrating Actor (Batch={n_episodes}, MaxIter={max_iterations}, Eps={epsilon_scale})...")
        
        # --- Step 1: Scaling ---
        print("   [Scaling Variance] ...")
        current_std, current_mean_q, _, _ = self._evaluate_batch_price(env, n_episodes)
        
        if current_std > 1e-9:
            scale_factor = target_std / current_std
            with torch.no_grad():
                self.actor_local.fc4.weight.mul_(scale_factor)
                self.actor_target.fc4.weight.mul_(scale_factor)
                
        # --- Step 2: Bias Initialization ---
        # Heuristic: shift bias so initial mean Q matches time-averaged constraint
        target_q = env.contract.Q_max / env.contract.n_rights
        denom = (env.contract.q_max - env.contract.q_min)
        target_q_norm = (target_q - env.contract.q_min) / denom if denom > 0 else 0.5
        target_q_norm = np.clip(target_q_norm, 0.05, 0.95)
        
        _, current_q_mean, _, _ = self._evaluate_batch_price(env, n_episodes // 4)
        bias_shift = (target_q_norm - current_q_mean) * 2.0 
        
        with torch.no_grad():
            self.actor_local.fc4.bias.add_(bias_shift)
            self.actor_target.fc4.bias.add_(bias_shift)

        # --- Step 3: Rprop Optimization Loop ---
        step_size = 0.1  # Initial step size
        min_step = 1e-4
        max_step = 0.5
        eta_plus = 1.2
        eta_minus = 0.5
        
        prev_grad = None
        best_price = -np.inf
        prev_price = None
        
        for i in range(max_iterations):
            # 1. Base Evaluation
            std_val, mean_q, price, std_price = self._evaluate_batch_price(env, n_episodes)
            se_price = std_price / np.sqrt(n_episodes)
            
            # Formatting
            diff_str = ""
            if prev_price is not None:
                 delta = price - prev_price
                 pct = (delta / abs(prev_price)) * 100 if abs(prev_price) > 1e-6 else 0.0
                 symbol = "+" if delta >= 0 else "-"
                 diff_str = f"| {symbol}{abs(delta):.4f} ({pct:+.2f}%)"
            
            # 2. Gradient (Finite Differences)
            epsilon = max(epsilon_scale * mean_q, 1e-3) 
            original_bias = self.actor_local.fc4.bias.data.clone()
            
            # P(b+eps)
            with torch.no_grad(): self.actor_local.fc4.bias.add_(epsilon)
            _, _, price_plus, _ = self._evaluate_batch_price(env, n_episodes)
            
            # P(b-eps)
            with torch.no_grad(): 
                 self.actor_local.fc4.bias.copy_(original_bias)
                 self.actor_local.fc4.bias.sub_(epsilon)
            _, _, price_minus, _ = self._evaluate_batch_price(env, n_episodes)
            
            # Restore
            with torch.no_grad(): self.actor_local.fc4.bias.copy_(original_bias)
            
            grad = (price_plus - price_minus) / (2 * epsilon)
            
            # 3. Rprop Update Rule
            update = 0.0
            step_type = "Init"
            
            if prev_grad is None:
                step_type = "Rprop(Init)"
            else:
                if grad * prev_grad > 0:
                    step_size = min(step_size * eta_plus, max_step)
                    step_type = "Rprop(Acc)"
                elif grad * prev_grad < 0:
                    step_size = max(step_size * eta_minus, min_step)
                    step_type = "Rprop(Dec)"
                else:
                    step_type = "Rprop(Keep)"
                    
            if abs(grad) < 1e-6:
                 update = 0.0
                 step_type = "Converged"
            else:
                 update = np.sign(grad) * step_size
            
            prev_grad = grad
            
            print(f"Iter {i+1:02d}: Price {price:.4f} (±{2*se_price:.4f}) {diff_str} | Q={mean_q:.3f} | Grad={grad:+.4f} | {step_type} Step={step_size:.4f}")

            # Convergence Check
            if price > best_price:
                best_price = price
                
            if i >= min_iterations:
                 if abs(update) < 1e-4 or (prev_price is not None and abs(price - prev_price) < tolerance_percent * abs(best_price)):
                      print(f"✅ Converged (Step too small or Price stable).")
                      break
            
            prev_price = price
            
            # Apply Update
            with torch.no_grad():
                self.actor_local.fc4.bias.add_(update)
                self.actor_target.fc4.bias.add_(update)

        _, final_q, final_price, _ = self._evaluate_batch_price(env, n_episodes)
        print(f"DONE: Final Price={final_price:.4f} | Mean Q={final_q:.3f}")

    def _output_slope(self, out_mean: float) -> float:
        """Local derivative d(output)/d(preact) of the actor output activation at a
        given mean output level. Used as the one-Newton-step gain for the closed-form
        bias shift (invert the squash locally instead of a magic constant)."""
        a = self._actor_action_output
        m = float(min(max(out_mean, 1e-3), 1.0 - 1e-3))
        if a == "tanh01":
            # out = 0.5*(tanh(z)+1) -> d/dz = 0.5*(1-tanh^2) = 2*m*(1-m)
            return 2.0 * m * (1.0 - m)
        if a == "sigmoid":
            return m * (1.0 - m)
        if a.startswith("beta_sigmoid"):
            beta = getattr(self.actor_local, "_output_beta", 2.0)
            return beta * m * (1.0 - m)
        # plain tanh maps to [-1,1]; treat m as a (0,1) proxy
        return max(1e-3, 1.0 - (2.0 * m - 1.0) ** 2)

    def _calibrate_bias_closed_form(self, env, n_episodes: int, target_std: float = 0.005) -> None:
        """Closed-form myopic warm-start (replaces the Rprop loop).

        Sets the actor output bias so the mean normalized exercise matches the myopic
        convex-cost optimum averaged over the warmup dataset spots (which ARE samples of
        the HHK/kernel density), budget-capped by Q_max/n_rights. Two batch passes total
        (variance scale + bias inversion) instead of ~20 Rprop iterations.

        Myopic FOC:  q*(S) = clip( ((S-K)+/(c*gamma))^(1/(gamma-1)), q_min, q_max )
          gamma=2 -> (S-K)+/(2c);  gamma=1 (linear cost) -> bang-bang on (S-K)>c;
          c=0      -> bang-bang on S>K.
        """
        print(f"\n🧪 Calibrating Actor (closed-form myopic, Batch={n_episodes})...")

        # Unwrap to access the vectorized warmup paths / contract.
        base = env
        if not hasattr(base, "S") and hasattr(base, "unwrapped"):
            base = base.unwrapped
        c = getattr(env, "contract", None) or getattr(base, "contract")

        q_min = float(c.q_min)
        q_max = float(c.q_max)
        c_cost = float(c.c_cost)
        gamma_cost = float(c.gamma_cost)
        strike = float(c.strike)
        denom = (q_max - q_min) if (q_max > q_min) else 1.0

        # --- Step 1: variance scaling (unchanged one-shot) ---
        current_std, _, _, _ = self._evaluate_batch_price(env, n_episodes)
        if current_std > 1e-9:
            scale_factor = target_std / current_std
            with torch.no_grad():
                self.actor_local.fc4.weight.mul_(scale_factor)
                self.actor_target.fc4.weight.mul_(scale_factor)

        # --- Step 2: closed-form myopic target mean action ā ---
        if hasattr(base, "S") and len(getattr(base, "S")) >= 1:
            n = min(n_episodes, len(base.S))
            n_rights = int(c.n_rights)
            S = np.asarray(base.S, dtype=np.float64)[:n]
            S_dec = S[:, :n_rights] if S.ndim >= 2 else S
            itm = np.maximum(S_dec - strike, 0.0)
            if c_cost <= 0.0:
                q_star = np.where(S_dec > strike, q_max, q_min)
            elif abs(gamma_cost - 1.0) < 1e-9:
                # Linear cost: marginal profit (S-K)-c is constant in q -> bang-bang.
                q_star = np.where(itm > c_cost, q_max, q_min)
            else:
                cg = max(c_cost * gamma_cost, 1e-12)
                base_q = np.power(itm / cg, 1.0 / (gamma_cost - 1.0))
                q_star = np.clip(base_q, q_min, q_max)
            a_myopic = float(np.mean((q_star - q_min) / denom))
        else:
            a_myopic = 0.5

        # Budget cap (same form as the legacy heuristic at the old Step 2).
        target_q = c.Q_max / c.n_rights
        target_q_budget = (target_q - q_min) / denom if denom > 0 else 0.5
        a_target = min(a_myopic, target_q_budget)
        a_target = float(np.clip(a_target, 0.05, 0.95))

        # --- Step 3: single bias shift via local squash inversion ---
        _, m0, _, _ = self._evaluate_batch_price(env, n_episodes)
        slope = max(self._output_slope(m0), 0.05)
        delta = float(np.clip((a_target - m0) / slope, -4.0, 4.0))
        with torch.no_grad():
            self.actor_local.fc4.bias.add_(delta)
            self.actor_target.fc4.bias.add_(delta)

        _, final_q, final_price, _ = self._evaluate_batch_price(env, n_episodes)
        print(
            f"DONE (closed-form): ā_myopic={a_myopic:.3f} budget={target_q_budget:.3f} "
            f"-> target ā={a_target:.3f} | m0={m0:.3f} | Δbias={delta:+.3f} | "
            f"Final Q={final_q:.3f} Price={final_price:.4f}"
        )

    def _evaluate_batch_price(self, env, n_episodes):
        """
        Vectorized evaluation of n_episodes to return (std_dev_output, mean_output, mean_price, std_price).
        Uses the agent's current local actor policy (noise disabled) for the entire batch.
        """
        # Unwrap if needed to access paths
        if not hasattr(env, 'S') and hasattr(env, 'unwrapped'):
            if hasattr(env.unwrapped, 'S'):
                env = env.unwrapped
                
        if not hasattr(env, 'S') or len(env.S) < n_episodes:
             # Fallback to slow loop if environment doesn't expose vectorized paths
             return self._evaluate_batch_slow(env, n_episodes)

        device = self.device
        
        # Helper: Convert to tensor (N, Steps, 1)
        def to_t(arr):
            t = torch.as_tensor(arr[:n_episodes], device=device, dtype=torch.float32)
            if t.ndim == 2: t = t.unsqueeze(-1)
            if t.ndim == 1: t = t.unsqueeze(-1).unsqueeze(-1)
            return t

        S_paths = to_t(env.S)
        X_paths = to_t(env.X)
        Y_paths = to_t(env.Y)
        
        N = n_episodes
        n_rights = env.contract.n_rights
        strike = env.contract.strike
        dt = env.contract.dt
        min_refraction = env.contract.min_refraction_periods
        
        Q_max = env.contract.Q_max
        q_min = env.contract.q_min
        q_max = env.contract.q_max
        c_cost = env.contract.c_cost
        gamma_cost = env.contract.gamma_cost
        r = env.contract.r

        # Tracking
        q_exercised = torch.zeros(N, device=device)
        last_exercise_step = torch.full((N,), -1.0, device=device)
        total_returns = torch.zeros(N, device=device)
        all_qs = [] # Collect for stats
        
        # Loop over steps
        for step in range(n_rights):
            # --- Construct State ---
            S_t = S_paths[:, step, 0]
            X_t = X_paths[:, step, 0]
            Y_t = Y_paths[:, step, 0]
            
            q_remaining = Q_max - q_exercised
            time_rem = (n_rights - step) * dt
            norm_time = step / n_rights
            
            days_since = step - last_exercise_step
            days_since = torch.where(last_exercise_step < 0, float(step), days_since)
            
            f1 = S_t - strike
            f2 = q_exercised / Q_max
            f3 = q_remaining / Q_max
            f4 = torch.full((N,), float(time_rem / env.contract.maturity), device=device)
            f5 = torch.full((N,), norm_time, device=device)
            f6 = S_t
            f7 = X_t
            f8 = Y_t
            f9 = days_since / n_rights
            
            state_batch = torch.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9], dim=1).float()
            
            # --- Agent Act ---
            with torch.no_grad():
                 # Use forward_raw_and_gated to bypass noise and get gated output
                 # If method missing, use standard act logic manually
                 if hasattr(self.actor_local, "forward_raw_and_gated"):
                     _, q_gated_t = self.actor_local.forward_raw_and_gated(state_batch)
                     action_vals = q_gated_t.squeeze(-1)
                 else:
                     preact = self.actor_local.forward_preact(state_batch)
                     action_vals = self._squash(preact).squeeze(-1)

            # Denormalize
            q_proposed = q_min + action_vals * (q_max - q_min)
            
            # --- Constraints ---
            q_feasible = torch.clamp(q_proposed, q_min, q_max)
            max_allowed = Q_max - q_exercised
            q_feasible = torch.min(q_feasible, max_allowed)
            
            if min_refraction > 0:
                 violation_mask = (step - last_exercise_step) <= min_refraction
                 active_exercise = last_exercise_step >= 0
                 mask = active_exercise & violation_mask
                 q_feasible = torch.where(mask, torch.tensor(0.0, device=device), q_feasible)
                 
            q_actual = torch.max(torch.tensor(0.0, device=device), q_feasible)
            
            # --- Reward ---
            payoff = q_actual * torch.relu(S_t - strike)
            cost = c_cost * torch.pow(q_actual, gamma_cost)
            net_reward = payoff - cost
            
            # Zero gain mask: if reward <= 0, don't exercise (optional, but standard for options)
            zero_gain_mask = net_reward <= 0.0
            q_actual = torch.where(zero_gain_mask, torch.tensor(0.0, device=device), q_actual)
            net_reward = torch.where(zero_gain_mask, torch.tensor(0.0, device=device), net_reward)
            
            # --- Updates ---
            q_exercised += q_actual
            exercised_mask = q_actual > 1e-6
            last_exercise_step = torch.where(exercised_mask, torch.tensor(float(step), device=device), last_exercise_step)
            
            # Stats input (normalized)
            denom = (q_max - q_min) if (q_max > q_min) else 1.0
            q_norm = (q_actual - q_min) / denom
            all_qs.append(q_norm)
            
            # Discounted Return
            df = float(np.exp(-r * (step + 1) * dt))
            total_returns += net_reward * df
        
        # Compile Stats
        all_qs_t = torch.cat(all_qs)
        mean_q = all_qs_t.mean().item()
        std_q = all_qs_t.std().item()
        mean_price = total_returns.mean().item()
        std_price = total_returns.std().item()
        
        return std_q, mean_q, mean_price, std_price

    def _evaluate_batch_slow(self, env, n_episodes):
        """Fallback for non-vectorized environments."""
        outputs = []
        returns = []
        if hasattr(env, "_episode_counter"):
            env._episode_counter = -1 
            
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_qs = []
            episode_return = 0.0
            
            while not done:
                action = self.act(np.expand_dims(state, axis=0), add_noise=False)
                action_val = action[0]
                episode_qs.append(action_val)
                next_state, reward, terminated, truncated, _ = env.step(action_val)
                done = terminated or truncated
                state = next_state
                episode_return += reward
                
            outputs.extend(episode_qs)
            returns.append(episode_return)
            
        outputs = np.array(outputs)
        returns = np.array(returns)
        return np.std(outputs), np.mean(outputs), np.mean(returns), np.std(returns)

    def reset(self):
        """Reset agent state between episodes (placeholder for future use)."""
        return
