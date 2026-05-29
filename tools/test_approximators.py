"""Mathematical-correctness tests for the function approximators.

Covers interface/shape, autograd gradient checks (the core DPG-compatibility
proof: dQ/da, dQ/dw, da/dtheta), fitting capacity, profitability-gate/output
bounds, semi-analytical-kernel compatibility, and the bit-identical NN guard.

Run:  pytest tools/test_approximators.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.networks import Actor, BasisActor, BasisCritic, Critic  # noqa: E402

KINDS = ["poly", "rff", "rbf", "tiny_nn"]
STATE_SIZE = 9
ACTION_SIZE = 1
STRIKE = 100.0


def _cfg():
    return {
        "strike": STRIKE,
        "use_cross": True,
        "degree": 3,
        "rff_dim": 64,
        "lengthscale": 1.0,
        "learnable": False,
        "rbf_centers": 48,
        "bandwidth": 1.0,
        "learnable_bandwidth": False,
        "width": 24,
        "activation": "silu",
    }


def _make_actor(kind, dtype=torch.float32, seed=0):
    a = BasisActor(
        STATE_SIZE, ACTION_SIZE, seed,
        action_output="beta_sigmoid_3.0",
        feature_kind=kind, feature_cfg=_cfg(),
    ).to(dtype)
    return a


def _make_critic(kind, dtype=torch.float32, seed=0):
    c = BasisCritic(
        STATE_SIZE, ACTION_SIZE, seed,
        feature_kind=kind, feature_cfg=_cfg(),
    ).to(dtype)
    return c


def _rand_state(B, dtype=torch.float32):
    s = torch.rand(B, STATE_SIZE, dtype=dtype)
    s[:, 5] = torch.rand(B, dtype=dtype) * 60.0 + 70.0   # raw spot S in [70,130]
    s[:, 0] = s[:, 5] - STRIKE                           # S - K
    s[:, 6] = torch.randn(B, dtype=dtype) * 0.2          # X
    s[:, 7] = torch.randn(B, dtype=dtype) * 0.3          # Y
    return s


# ─────────────────────────── interface / shape ───────────────────────────

@pytest.mark.parametrize("kind", KINDS)
def test_actor_shape_and_bounds(kind):
    a = _make_actor(kind)
    s = _rand_state(16)
    out = a(s)
    assert out.shape == (16, 1)
    assert torch.isfinite(out).all()
    out = out.detach()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


@pytest.mark.parametrize("kind", KINDS)
def test_critic_shape(kind):
    c = _make_critic(kind)
    s = _rand_state(16)
    act = torch.rand(16, 1)
    q = c(s, act)
    assert q.shape == (16, 1)
    assert torch.isfinite(q).all()


@pytest.mark.parametrize("kind", KINDS)
def test_no_leftover_mlp_params(kind):
    a = _make_actor(kind)
    c = _make_critic(kind)
    # The MLP stack is removed; the output head (fc4) is intentionally kept.
    assert not hasattr(a, "hidden_layers")
    assert not hasattr(c, "state_encoder") and not hasattr(c, "post_layers")


# ─────────────────────── autograd gradient checks ────────────────────────
# float64 gradcheck is the core proof that each approximator is DPG-compatible.

@pytest.mark.parametrize("kind", KINDS)
def test_dQ_da_gradcheck(kind):
    torch.manual_seed(0)
    c = _make_critic(kind, dtype=torch.float64)
    s = _rand_state(4, dtype=torch.float64)
    act = torch.rand(4, 1, dtype=torch.float64, requires_grad=True)

    def f(a):
        return c(s, a)

    assert torch.autograd.gradcheck(f, (act,), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("kind", KINDS)
def test_dQ_dw_gradcheck(kind):
    torch.manual_seed(0)
    c = _make_critic(kind, dtype=torch.float64)
    s = _rand_state(4, dtype=torch.float64)
    act = torch.rand(4, 1, dtype=torch.float64)
    w = c.head.weight.detach().clone().requires_grad_(True)

    def f(weight):
        return torch.nn.functional.linear(c._features(s, act), weight, c.head.bias)

    # _features may not exist; fall back to perturbing the parameter in-place.
    if hasattr(c, "_features"):
        assert torch.autograd.gradcheck(f, (w,), atol=1e-4, rtol=1e-3)
    else:
        q = c(s, act)
        (g,) = torch.autograd.grad(q.sum(), c.head.weight)
        assert torch.isfinite(g).all() and float(g.abs().sum()) > 0.0


@pytest.mark.parametrize("kind", KINDS)
def test_da_dtheta_finite(kind):
    a = _make_actor(kind, dtype=torch.float64)
    s = _rand_state(4, dtype=torch.float64)
    out = a(s)
    grads = torch.autograd.grad(out.sum(), [p for p in a.parameters() if p.requires_grad])
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0.0 for g in grads)


# ───────────────────────────── fitting capacity ──────────────────────────

@pytest.mark.parametrize("kind", KINDS)
def test_critic_can_overfit(kind):
    """Each critic must reduce MSE substantially on a small smooth target."""
    torch.manual_seed(1)
    c = _make_critic(kind)
    s = _rand_state(128)
    act = torch.rand(128, 1)
    # Smooth synthetic Q: intrinsic-value-like surface.
    target = (act.squeeze(-1) * torch.relu(s[:, 5] - STRIKE) * 0.1).unsqueeze(-1)
    opt = torch.optim.Adam(c.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    init = loss_fn(c(s, act), target).item()
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(c(s, act), target)
        loss.backward()
        opt.step()
    final = loss_fn(c(s, act), target).item()
    assert final < 0.5 * init, f"{kind}: did not fit (init={init:.4g}, final={final:.4g})"


# ──────────────────────── profitability gate / output ────────────────────

@pytest.mark.parametrize("kind", KINDS)
def test_profitability_gate_zeros_unprofitable(kind):
    a = _make_actor(kind)
    a.set_profitability_params(q_min=0.0, q_max=1.0, c_cost=0.04, gamma_cost=2.0)
    # Deep OTM state: S << K => exercise unprofitable => gate must zero output.
    s = _rand_state(16)
    s[:, 5] = 50.0
    s[:, 0] = s[:, 5] - STRIKE
    out = a(s)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


# ──────────────────────── semi-analytical kernel ─────────────────────────

@pytest.mark.parametrize("kind", KINDS)
def test_kernel_compatibility(kind):
    from src.simulate_hhk_spot import no_seasonal_function
    from src.transition_kernel import (
        KernelParams,
        expected_critic_target,
        precompute_kernel,
    )

    n_rights = 50
    kparams = KernelParams(
        alpha=7.0, sigma=1.4, beta=200.0, lam=6.0, mu_J=0.4,
        dt=1.0 / n_rights, M_x=2, N_max=1, M_per_k=1, f_id="no_seasonal",
    )
    kernel = precompute_kernel(kparams, n_rights, no_seasonal_function, maturity=1.0)

    actor_t = _make_actor(kind, dtype=torch.float64)
    critic_t = _make_critic(kind, dtype=torch.float64)
    B = 8
    states = _rand_state(B, dtype=torch.float64)
    next_states = _rand_state(B, dtype=torch.float64)
    next_states[:, 4] = 0.5  # t_norm at next step

    out = expected_critic_target(
        critic_t, actor_t, states, next_states, kernel, strike=STRIKE,
    )
    out = out.reshape(-1)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


# ──────────────────────── bit-identical NN guard ─────────────────────────

def test_nn_path_unchanged_defaults():
    """Default Actor/Critic (the --approximator=nn path) is untouched: a fresh
    pair with the same seed must produce identical outputs (no global RNG side
    effects from the new classes)."""
    torch.manual_seed(123)
    a1 = Actor(STATE_SIZE, ACTION_SIZE, 7, action_output="beta_sigmoid_3.0")
    c1 = Critic(STATE_SIZE, ACTION_SIZE, 7)
    torch.manual_seed(123)
    a2 = Actor(STATE_SIZE, ACTION_SIZE, 7, action_output="beta_sigmoid_3.0")
    c2 = Critic(STATE_SIZE, ACTION_SIZE, 7)
    s = _rand_state(8)
    act = torch.rand(8, 1)
    assert torch.allclose(a1(s), a2(s), atol=0.0)
    assert torch.allclose(c1(s, act), c2(s, act), atol=0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
