"""
H5: Dyna-style synthetic experience augmentation.

At each critic update, in addition to the real batch sampled from the replay
buffer, we generate K *fresh* synthetic (state, action) pairs uniformly over
a sensible (X, Y, Q_remaining, t) domain. For each synthetic pair we:

  1. Compute reward r(s, a) using the env's gated reward function.
  2. Construct the deterministic part of next_state (Q_remaining update,
     t advancement, days_since, etc.) - the kernel handles the stochastic
     (X', Y', S') part.
  3. Use the H1 expected-target machinery (transition_kernel.expected_critic_target)
     to compute Q*(s, a) = r + gamma * E[Q_target(s', pi(s'))] * (1 - done).
  4. Train critic_local on (s, a, Q*) via MSE, mixed into the real-batch loss
     with weight `lambda`.

Distinct from:
  * H1: changes the *target* used for replay-buffer transitions only. H5
    adds *new* (s, a) training samples never seen in real rollouts.
  * H4: one-shot supervised pre-training before training starts. H5 supplies
    a continuous flow of supervised samples *during* training, and uses the
    current critic_target / actor_target rather than a frozen V grid - so
    bias correction happens implicitly via the soft-updated target network.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# Default sensible ranges over which to sample synthetic (X, Y) for the
# focal HHK regime (alpha=12, sigma=1.2, beta=150, lam=6, mu_J=0.3).  X has
# steady-state std ~0.24, Y has small mean but heavy-tailed jumps.
DEFAULT_X_RANGE = (-0.8, 0.8)
DEFAULT_Y_RANGE = (0.0, 1.5)


def sample_dyna_synthetic_states(
    n_samples: int,
    contract,
    *,
    X_range: Tuple[float, float] = DEFAULT_X_RANGE,
    Y_range: Tuple[float, float] = DEFAULT_Y_RANGE,
    rng: Optional[np.random.Generator] = None,
    obs_dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform-random (state, action) sampling over the synthetic domain.

    Returns
    -------
    states : (n_samples, 9) - env-observation-shaped
    actions : (n_samples, 1) - normalised in [0, 1]
    t_idx : (n_samples,) int64 - the synthetic current-step index
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(n_samples)
    Q_max = float(contract.Q_max)
    K = float(contract.strike)
    n_rights = int(contract.n_rights)

    X = rng.uniform(*X_range, size=n)
    Y = rng.uniform(*Y_range, size=n)
    Q_rem = rng.uniform(0.0, Q_max, size=n)
    t_idx = rng.integers(0, n_rights, size=n).astype(np.int64)
    a = rng.uniform(0.0, 1.0, size=n)

    # Spot under no-seasonality f(t)=0
    S = np.exp(X + Y)

    states = np.zeros((n, 9), dtype=obs_dtype)
    states[:, 0] = (S - K).astype(obs_dtype)
    states[:, 1] = ((Q_max - Q_rem) / Q_max).astype(obs_dtype)
    states[:, 2] = (Q_rem / Q_max).astype(obs_dtype)
    states[:, 3] = ((n_rights - t_idx) / n_rights).astype(obs_dtype)
    states[:, 4] = (t_idx / n_rights).astype(obs_dtype)
    states[:, 5] = S.astype(obs_dtype)
    states[:, 6] = X.astype(obs_dtype)
    states[:, 7] = Y.astype(obs_dtype)
    # days_since_exercise/n_rights - assume no prior exercise -> equals t_idx/n_rights
    states[:, 8] = (t_idx / n_rights).astype(obs_dtype)
    actions = a.reshape(-1, 1).astype(obs_dtype)
    return states, actions, t_idx


def compute_dyna_rewards_and_eff_actions(
    states: np.ndarray,
    actions: np.ndarray,
    t_idx: np.ndarray,
    contract,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute env-gated discounted reward + effective action.

    Mirrors swing_env.calculate_standardized_reward + the env's profitability
    gate: if net_payoff <= 0, a_eff = 0, reward = 0.
    """
    Q_max = float(contract.Q_max)
    K = float(contract.strike)
    c_cost = float(contract.c_cost)
    gamma_cost = float(contract.gamma_cost)
    q_min = float(contract.q_min)
    q_max = float(contract.q_max)
    disc = float(contract.discount_factor)

    S = states[:, 5].astype(np.float64)
    Q_rem = states[:, 2].astype(np.float64) * Q_max
    a_norm = actions[:, 0].astype(np.float64).clip(0.0, 1.0)
    a = q_min + a_norm * (q_max - q_min)
    a = np.minimum(a, Q_rem)

    payoff_per_unit = np.maximum(S - K, 0.0)
    gross = a * payoff_per_unit
    cost = c_cost * np.power(a, gamma_cost)
    net = gross - cost
    mask = net > 0.0
    a_eff = np.where(mask, a, 0.0)
    net = np.where(mask, net, 0.0)
    rewards = (disc ** t_idx) * net
    return rewards.astype(np.float32), a_eff.astype(np.float32)


def build_dyna_next_state_template(
    states: np.ndarray,
    a_eff: np.ndarray,
    t_idx: np.ndarray,
    contract,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the deterministic part of the next state for each synthetic
    transition. The (S-K, S, X, Y) slots are zero placeholders that
    `expected_critic_target` will overwrite with kernel samples.

    Returns (next_states, dones).
    """
    Q_max = float(contract.Q_max)
    n_rights = int(contract.n_rights)

    n = states.shape[0]
    Q_rem = states[:, 2].astype(np.float64) * Q_max
    a_eff_f = a_eff.astype(np.float64)
    Q_ex_new = (Q_max - Q_rem) + a_eff_f
    Q_rem_new = Q_rem - a_eff_f
    t_idx_new = t_idx + 1

    next_states = np.zeros_like(states)
    next_states[:, 0] = 0.0
    next_states[:, 1] = (Q_ex_new / Q_max).astype(states.dtype)
    next_states[:, 2] = (Q_rem_new / Q_max).astype(states.dtype)
    next_states[:, 3] = (np.maximum(n_rights - t_idx_new, 0) / n_rights).astype(states.dtype)
    # Clip t_idx_new for the t/n_rights field; the kernel uses next_states[:, 4]
    # to extract t_idx, so it must equal (t_idx + 1) / n_rights (clipped to <= 1).
    next_states[:, 4] = (np.minimum(t_idx_new, n_rights) / n_rights).astype(states.dtype)
    next_states[:, 5] = 0.0
    next_states[:, 6] = 0.0
    next_states[:, 7] = 0.0
    # days_since_next: if we exercised this step, reset to 1; else t_idx_new (assuming
    # no prior exercise before t_idx).
    exercised = a_eff_f > 1e-6
    days_since_new = np.where(exercised, 1.0, t_idx_new + 1.0)
    next_states[:, 8] = (days_since_new / n_rights).astype(states.dtype)

    dones = ((t_idx_new >= n_rights) | (Q_rem_new < 1e-6)).astype(np.float32)
    return next_states, dones


def generate_dyna_batch(
    n_samples: int,
    contract,
    *,
    rng: Optional[np.random.Generator] = None,
    obs_dtype: np.dtype = np.float32,
    X_range: Tuple[float, float] = DEFAULT_X_RANGE,
    Y_range: Tuple[float, float] = DEFAULT_Y_RANGE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end: states, actions, rewards, next_state_template, dones."""
    states, actions, t_idx = sample_dyna_synthetic_states(
        n_samples, contract, X_range=X_range, Y_range=Y_range,
        rng=rng, obs_dtype=obs_dtype,
    )
    rewards, a_eff = compute_dyna_rewards_and_eff_actions(states, actions, t_idx, contract)
    next_states, dones = build_dyna_next_state_template(states, a_eff, t_idx, contract)
    return states, actions, rewards, next_states, dones


__all__ = [
    "sample_dyna_synthetic_states",
    "compute_dyna_rewards_and_eff_actions",
    "build_dyna_next_state_template",
    "generate_dyna_batch",
    "DEFAULT_X_RANGE",
    "DEFAULT_Y_RANGE",
]
