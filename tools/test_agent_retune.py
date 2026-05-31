"""Unit tests for the deterministic-target retune (Tasks 1-3).

Covers the new agent code paths that were previously only validated end-to-end via
tools/sweep_v63_audit.py:
  * Task 1 — closed-form calibrate_bias (FOC warm-start, _output_slope).
  * Task 2 — noise schedules (linear / const_floor) and eval-only EMA weight averaging.

These are intentionally lightweight (tiny nets, CPU, small fake env) and do NOT touch
the kernel or training loop. Run with:  pytest tools/test_agent_retune.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import Agent
from src.swing_contract import SwingContract


def _make_agent(**overrides):
    """Minimal CPU Agent (9-dim state, 1-dim action) for unit testing."""
    kw = dict(
        state_size=9, action_size=1, n_step=1, random_seed=0,
        hidden_size=16, actor_layers=2, critic_layers=2,
        action_output="beta_sigmoid_3.0", total_episodes=4096,
        warmup_episodes=64,
    )
    kw.update(overrides)
    return Agent(**kw)


# --------------------------------------------------------------------------- #
# Task 1 — _output_slope (local squash derivative used for the bias inversion)
# --------------------------------------------------------------------------- #
def test_output_slope_beta_sigmoid():
    a = _make_agent(action_output="beta_sigmoid_3.0")
    beta = getattr(a.actor_local, "_output_beta", 2.0)
    m = 0.4
    assert a._output_slope(m) == pytest.approx(beta * m * (1 - m), rel=1e-6)


def test_output_slope_clamps_extremes():
    a = _make_agent(action_output="sigmoid")
    # At m->0 or m->1 the slope must stay strictly positive (clamped), never 0/negative.
    assert a._output_slope(0.0) > 0.0
    assert a._output_slope(1.0) > 0.0


# --------------------------------------------------------------------------- #
# Task 2 — noise schedules
# --------------------------------------------------------------------------- #
def test_noise_linear_monotone_decay():
    a = _make_agent(noise_schedule="linear", noise_sigma0=1.30, noise_floor=0.26,
                    noise_plateau=0, noise_decay_episodes=4096,
                    critic_warmup_episodes=0, warmup_noise_fraction=1.0)
    sigmas = []
    for e in [1, 1000, 2000, 3000, 4096, 5000]:
        a.update_episode_count(e)
        sigmas.append(a._pre_noise_sigma())
    # Strictly decreasing to the floor, then clamped at the floor.
    assert sigmas[0] > sigmas[1] > sigmas[2] > sigmas[3]
    assert sigmas[4] == pytest.approx(0.26, abs=1e-6)   # reaches floor at horizon
    assert sigmas[5] == pytest.approx(0.26, abs=1e-6)   # clamped beyond horizon


def test_noise_const_floor_steps_to_floor():
    a = _make_agent(noise_schedule="const_floor", noise_sigma0=1.30, noise_floor=0.26,
                    noise_plateau=1000, critic_warmup_episodes=0, warmup_noise_fraction=1.0)
    a.update_episode_count(500)
    assert a._pre_noise_sigma() == pytest.approx(1.30, abs=1e-6)   # sigma0 during plateau
    a.update_episode_count(1500)
    assert a._pre_noise_sigma() == pytest.approx(0.26, abs=1e-6)   # hard step to floor after


def test_noise_hyperbolic_default_unchanged():
    # Default schedule must remain hyperbolic (bit-identical default behavior).
    a = _make_agent(noise_sigma0=1.0, noise_floor=0.05, noise_plateau=0,
                    critic_warmup_episodes=0, warmup_noise_fraction=1.0)
    assert a.noise_schedule == "hyperbolic"
    a.update_episode_count(1)
    # hyperbolic at e=1, plateau=0: floor + (sigma0-floor)/(1+1)
    assert a._pre_noise_sigma() == pytest.approx(0.05 + (1.0 - 0.05) / 2.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Task 2 — eval-only EMA weight averaging
# --------------------------------------------------------------------------- #
def test_ema_disabled_by_default():
    a = _make_agent()
    assert a.weight_averaging == "off"
    assert a._actor_ema_data is None
    # Context manager is a no-op when off (weights untouched).
    before = [p.detach().clone() for p in a._actor_local_data]
    with a.averaged_eval_actor():
        pass
    for p, b in zip(a._actor_local_data, before):
        assert torch.equal(p, b)


def test_ema_update_moves_shadow_toward_local():
    a = _make_agent(weight_averaging="ema", ema_decay=0.9)
    assert a._actor_ema_data is not None
    # Perturb the live actor away from the (cloned) shadow.
    with torch.no_grad():
        for p in a._actor_local_data:
            p.add_(1.0)
    shadow_before = [s.detach().clone() for s in a._actor_ema_data]
    a._update_ema()
    # ema <- 0.9*ema + 0.1*local ; with local = shadow_before + 1 => ema moves up by 0.1.
    for s_new, s_old in zip(a._actor_ema_data, shadow_before):
        assert torch.allclose(s_new, s_old + 0.1, atol=1e-6)


def test_averaged_eval_actor_swaps_and_restores():
    a = _make_agent(weight_averaging="ema", ema_decay=0.999)
    # Make the shadow distinct from the live weights.
    with torch.no_grad():
        for s in a._actor_ema_data:
            s.add_(2.0)
    train_w = [p.detach().clone() for p in a._actor_local_data]
    ema_w = [s.detach().clone() for s in a._actor_ema_data]
    with a.averaged_eval_actor():
        for p, e in zip(a._actor_local_data, ema_w):
            assert torch.allclose(p, e, atol=1e-7)   # EMA weights are live during eval
    for p, t in zip(a._actor_local_data, train_w):
        assert torch.allclose(p, t, atol=1e-7)        # training weights restored after


# --------------------------------------------------------------------------- #
# Task 1 — closed-form calibrate_bias integration on a lightweight fake env
# --------------------------------------------------------------------------- #
class _FakeEnv:
    """Minimal env exposing vectorized HHK-like paths + a SwingContract."""
    def __init__(self, contract, n=256, seed=0):
        rng = np.random.default_rng(seed)
        nr = contract.n_rights
        # Spot around the strike so a non-trivial fraction is ITM.
        self.S = contract.strike * np.exp(0.3 * rng.standard_normal((n, nr + 1)))
        self.X = 0.3 * rng.standard_normal((n, nr + 1))
        self.Y = np.zeros((n, nr + 1))
        self.contract = contract


@pytest.mark.parametrize("c_cost,gamma_cost", [(0.0, 1.0), (0.04, 1.0), (0.04, 2.0)])
def test_closed_form_calibrate_runs_and_shifts_bias(c_cost, gamma_cost):
    contract = SwingContract(
        q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0, maturity=0.0833,
        n_rights=22, r=0.05, c_cost=c_cost, gamma_cost=gamma_cost,
    )
    env = _FakeEnv(contract, n=256)
    a = _make_agent(action_output="beta_sigmoid_3.0", strike=1.0, warmup_episodes=256)
    bias_before = a.actor_local.fc4.bias.detach().clone()
    a.calibrate_bias(env=env, n_episodes=256, mode="closed_form")
    # Bias must have been updated and mirrored to actor_target (no NaNs).
    assert not torch.equal(a.actor_local.fc4.bias, bias_before)
    assert torch.allclose(a.actor_local.fc4.bias, a.actor_target.fc4.bias, atol=1e-6)
    assert torch.isfinite(a.actor_local.fc4.bias).all()


def test_calibrate_bias_rejects_bad_mode():
    contract = SwingContract(q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0, strike=1.0,
                             maturity=0.0833, n_rights=22, c_cost=0.04, gamma_cost=2.0)
    env = _FakeEnv(contract, n=64)
    a = _make_agent(strike=1.0)
    with pytest.raises(ValueError):
        a.calibrate_bias(env=env, n_episodes=64, mode="nonsense")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
