"""
H4: Semi-analytical critic warm-start via backward induction on a
(X, Y, Q_remaining, t) grid.

Pipeline:
  1.  build_value_grid(...) -> V[n_X, n_Y, n_Q, n_t] computed by
      backward induction.  At each (X, Y, Q, t) we enumerate a discrete
      set of feasible actions a, compute net payoff (with the convex
      profitability gate matching the env / LSM), use the Phase-0
      transition kernel to compute E[V(X', Y', Q-a, t+1)] via bilinear
      interpolation on the (X, Y) plane and nearest-integer in Q, then
      take the max over actions.
  2.  warm_start_critic(agent, ...) samples N synthetic (state, action)
      pairs uniformly over the grid, computes Q*(s, a) targets, and
      runs K supervised MSE epochs on agent.critic_local; then copies
      the warm-started weights to critic_target.

Plain numpy float64 + numba njit on the hot inner loops to stay
portable to C/C++ later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .transition_kernel import (
    TransitionKernel,
    build_next_state_grid_batched,
    build_quadrature_weights,
    OBS_IDX_S_MINUS_K,
    OBS_IDX_S,
    OBS_IDX_X,
    OBS_IDX_Y,
    OBS_IDX_T_NORM,
)

try:
    import numba as nb

    _NJIT = nb.njit(cache=True, fastmath=True, parallel=False)
except Exception:  # pragma: no cover
    nb = None

    def _NJIT(fn):
        return fn


@dataclass
class WarmStartGrid:
    """Container for the backward-induction V function + grid axes."""
    V: np.ndarray             # (n_X, n_Y, n_Q, n_t+1)
    X_axis: np.ndarray        # (n_X,)
    Y_axis: np.ndarray        # (n_Y,)
    n_Q: int                  # Q axis is integer 0..n_Q-1
    n_t: int                  # number of decision dates (= contract.n_rights)
    discount_factor: float    # per-step discount
    strike: float
    c_cost: float
    gamma_cost: float
    Q_max: float
    q_max: float
    q_min: float


# ---------------------------------------------------------------------------
# Hot kernel: bilinear interpolation of V on the (X, Y) plane
# ---------------------------------------------------------------------------


@_NJIT
def _bilinear_interp(
    V_plane: np.ndarray,         # (n_X, n_Y) for a specific Q, t
    X_axis: np.ndarray,          # (n_X,) ascending
    Y_axis: np.ndarray,          # (n_Y,) ascending
    X_pts: np.ndarray,           # (M,) - X values to interpolate at
    Y_pts: np.ndarray,           # (M,) - Y values
    out: np.ndarray,             # (M,) - output
) -> None:
    n_X = X_axis.shape[0]
    n_Y = Y_axis.shape[0]
    dx = (X_axis[-1] - X_axis[0]) / (n_X - 1) if n_X > 1 else 1.0
    dy = (Y_axis[-1] - Y_axis[0]) / (n_Y - 1) if n_Y > 1 else 1.0
    for m in range(X_pts.shape[0]):
        # Locate in X
        fx = (X_pts[m] - X_axis[0]) / dx
        if fx < 0.0:
            ix0 = 0; ix1 = 0; wx = 0.0
        elif fx >= n_X - 1:
            ix0 = n_X - 1; ix1 = n_X - 1; wx = 0.0
        else:
            ix0 = int(fx)
            ix1 = ix0 + 1
            wx = fx - ix0
        # Locate in Y
        fy = (Y_pts[m] - Y_axis[0]) / dy
        if fy < 0.0:
            iy0 = 0; iy1 = 0; wy = 0.0
        elif fy >= n_Y - 1:
            iy0 = n_Y - 1; iy1 = n_Y - 1; wy = 0.0
        else:
            iy0 = int(fy)
            iy1 = iy0 + 1
            wy = fy - iy0
        v00 = V_plane[ix0, iy0]
        v01 = V_plane[ix0, iy1]
        v10 = V_plane[ix1, iy0]
        v11 = V_plane[ix1, iy1]
        out[m] = (
            (1.0 - wx) * (1.0 - wy) * v00
            + (1.0 - wx) * wy * v01
            + wx * (1.0 - wy) * v10
            + wx * wy * v11
        )


@_NJIT
def _bilinear_interp_Q(
    V: np.ndarray,               # (n_X, n_Y, n_Q) at fixed t+1
    X_axis: np.ndarray,
    Y_axis: np.ndarray,
    X_pts: np.ndarray,           # (M,)
    Y_pts: np.ndarray,           # (M,)
    Q_target: float,             # non-integer Q (may be fractional)
    n_Q: int,
    out: np.ndarray,             # (M,)
) -> None:
    """Trilinear interpolation in (X, Y, Q). Q axis is integer 0..n_Q-1."""
    # Q linear interp
    if Q_target <= 0.0:
        iq0 = 0; iq1 = 0; wq = 0.0
    elif Q_target >= n_Q - 1:
        iq0 = n_Q - 1; iq1 = n_Q - 1; wq = 0.0
    else:
        iq0 = int(Q_target)
        iq1 = iq0 + 1
        wq = Q_target - iq0

    n_X = X_axis.shape[0]
    n_Y = Y_axis.shape[0]
    dx = (X_axis[-1] - X_axis[0]) / (n_X - 1) if n_X > 1 else 1.0
    dy = (Y_axis[-1] - Y_axis[0]) / (n_Y - 1) if n_Y > 1 else 1.0
    for m in range(X_pts.shape[0]):
        fx = (X_pts[m] - X_axis[0]) / dx
        if fx < 0.0:
            ix0 = 0; ix1 = 0; wx = 0.0
        elif fx >= n_X - 1:
            ix0 = n_X - 1; ix1 = n_X - 1; wx = 0.0
        else:
            ix0 = int(fx); ix1 = ix0 + 1; wx = fx - ix0
        fy = (Y_pts[m] - Y_axis[0]) / dy
        if fy < 0.0:
            iy0 = 0; iy1 = 0; wy = 0.0
        elif fy >= n_Y - 1:
            iy0 = n_Y - 1; iy1 = n_Y - 1; wy = 0.0
        else:
            iy0 = int(fy); iy1 = iy0 + 1; wy = fy - iy0
        # Q0 plane
        v000 = V[ix0, iy0, iq0]; v001 = V[ix0, iy1, iq0]
        v010 = V[ix1, iy0, iq0]; v011 = V[ix1, iy1, iq0]
        plane0 = (
            (1.0 - wx) * (1.0 - wy) * v000 + (1.0 - wx) * wy * v001
            + wx * (1.0 - wy) * v010 + wx * wy * v011
        )
        if wq == 0.0:
            out[m] = plane0
        else:
            v100 = V[ix0, iy0, iq1]; v101 = V[ix0, iy1, iq1]
            v110 = V[ix1, iy0, iq1]; v111 = V[ix1, iy1, iq1]
            plane1 = (
                (1.0 - wx) * (1.0 - wy) * v100 + (1.0 - wx) * wy * v101
                + wx * (1.0 - wy) * v110 + wx * wy * v111
            )
            out[m] = (1.0 - wq) * plane0 + wq * plane1


# ---------------------------------------------------------------------------
# Backward induction
# ---------------------------------------------------------------------------


def build_value_grid(
    kernel: TransitionKernel,
    *,
    strike: float,
    c_cost: float,
    gamma_cost: float,
    q_min: float,
    q_max: float,
    Q_max: float,
    n_rights: int,
    discount_factor: float,
    n_X: int = 25,
    n_Y: int = 20,
    n_actions: int = 11,
    X_range: Tuple[float, float] = (-0.8, 0.8),
    Y_range: Tuple[float, float] = (0.0, 1.5),
    seasonal_log_at: Optional[np.ndarray] = None,  # log f(t_k); default zeros
    verbose: bool = False,
) -> WarmStartGrid:
    """Build V(X, Y, Q, t) by backward induction.

    Q is integer-indexed 0..Q_max (inclusive). Actions are discretised at
    n_actions equally-spaced points in [q_min, q_max]. We assume Q_max is
    an integer (which it is for the focal contract); a generalisation to
    fractional Q is straightforward but not implemented here.
    """
    n_Q = int(round(Q_max)) + 1
    if abs(Q_max - (n_Q - 1)) > 1e-9:
        raise ValueError("build_value_grid currently requires integer Q_max")

    X_axis = np.linspace(X_range[0], X_range[1], n_X, dtype=np.float64)
    Y_axis = np.linspace(Y_range[0], Y_range[1], n_Y, dtype=np.float64)
    actions = np.linspace(q_min, q_max, n_actions, dtype=np.float64)

    if seasonal_log_at is None:
        # f(t) = 0 for the focal regime
        seasonal_log_at = np.zeros(n_rights + 1, dtype=np.float64)

    M = kernel.M
    w = build_quadrature_weights(kernel)  # (M,)

    # Pre-compute (X', Y') quadrature points for each (X_t, Y_t) grid cell.
    # We'll iterate t backwards. The kernel is time-homogeneous in dt so the
    # X' nodes depend only on X_t, and Y' nodes depend on Y_t (with the small
    # decay_Y * Y_t shift); but the S' value uses the next time index's
    # seasonal factor (which is constant = 1 here).
    # For each (ix, iy) we build a length-M (X', Y') array.

    BX = n_X * n_Y
    X_flat = np.repeat(X_axis, n_Y).astype(np.float64)   # (BX,)
    Y_flat = np.tile(Y_axis, n_X).astype(np.float64)     # (BX,)

    # Initialise V (terminal: V[..., n_rights] = 0)
    V = np.zeros((n_X, n_Y, n_Q, n_rights + 1), dtype=np.float64)

    # Buffers for per-action operations
    V_next_interp = np.empty(M, dtype=np.float64)
    # Temporary action axis: (n_actions,) candidate Q* values per state
    Q_star_buf = np.empty(n_actions, dtype=np.float64)

    if verbose:
        print(f"[H4] V grid: ({n_X}, {n_Y}, {n_Q}, {n_rights + 1}) "
              f"x {n_actions} actions x M={M} kernel nodes")

    for t in range(n_rights - 1, -1, -1):
        if verbose and t % 5 == 0:
            print(f"  backward t={t} ...")
        # Per-state kernel grid: we need (X', Y') for each (X_t, Y_t) cell.
        # build_next_state_grid_batched expects shapes (B,).
        t_idx_next = np.full(BX, t + 1, dtype=np.int64)
        _, X_grid, Y_grid = build_next_state_grid_batched(
            X_flat, Y_flat, t_idx_next, kernel
        )  # shapes (BX, M)

        for ib in range(BX):
            ix = ib // n_Y
            iy = ib % n_Y
            X_pts = X_grid[ib]    # (M,)
            Y_pts = Y_grid[ib]    # (M,)

            for iq in range(n_Q):
                Q_rem = float(iq)

                # Pre-compute spot at this (X, Y, t) cell (under f(t)=0)
                # Reward in the env uses the *current* spot S_t, not S_{t+1}.
                X_t = X_axis[ix]
                Y_t = Y_axis[iy]
                # The env at decision time t reads S_t = exp(f(t) + X_t + Y_t).
                # Note V is computed BEFORE the exercise decision, with S_t known.
                S_t = math.exp(seasonal_log_at[t] + X_t + Y_t)

                best = -1e30
                for ia in range(n_actions):
                    a = actions[ia]
                    # Respect remaining capacity
                    if a > Q_rem + 1e-12:
                        a = Q_rem

                    payoff_per_unit = max(S_t - strike, 0.0)
                    gross = a * payoff_per_unit
                    cost = c_cost * (a ** gamma_cost)
                    net = gross - cost
                    if net <= 0.0:
                        net = 0.0
                        a_eff = 0.0
                    else:
                        a_eff = a

                    # Discounted reward: env uses discount_factor**t (0-based step)
                    r = (discount_factor ** t) * net

                    # Continuation: Q_next = Q_rem - a_eff (fractional; linear interp in Q).
                    Q_next = Q_rem - a_eff
                    if Q_next < 0.0: Q_next = 0.0
                    if Q_next > n_Q - 1: Q_next = float(n_Q - 1)

                    # Trilinear interp of V[:, :, :, t+1] at (X', Y', Q_next)
                    _bilinear_interp_Q(
                        V[:, :, :, t + 1],
                        X_axis, Y_axis,
                        X_pts, Y_pts,
                        Q_next, n_Q,
                        V_next_interp,
                    )
                    EV = 0.0
                    for m in range(M):
                        EV += w[m] * V_next_interp[m]

                    Qstar = r + EV
                    Q_star_buf[ia] = Qstar
                    if Qstar > best:
                        best = Qstar

                V[ix, iy, iq, t] = best

    return WarmStartGrid(
        V=V, X_axis=X_axis, Y_axis=Y_axis,
        n_Q=n_Q, n_t=n_rights, discount_factor=discount_factor,
        strike=strike, c_cost=c_cost, gamma_cost=gamma_cost,
        Q_max=Q_max, q_max=q_max, q_min=q_min,
    )


def grid_price_at_origin(
    grid: WarmStartGrid,
    *,
    X_0: float = 0.0,
    Y_0: float = 0.0,
) -> float:
    """Return V(X_0, Y_0, Q_max, t=0) - the grid-implied initial swing-option
    price. Useful as a sanity check vs LSM benchmark."""
    # Bilinear interp at (X_0, Y_0)
    n_X = grid.X_axis.shape[0]
    n_Y = grid.Y_axis.shape[0]
    dx = (grid.X_axis[-1] - grid.X_axis[0]) / (n_X - 1) if n_X > 1 else 1.0
    dy = (grid.Y_axis[-1] - grid.Y_axis[0]) / (n_Y - 1) if n_Y > 1 else 1.0
    fx = (X_0 - grid.X_axis[0]) / dx
    fy = (Y_0 - grid.Y_axis[0]) / dy
    ix0 = max(0, min(n_X - 2, int(fx))); ix1 = ix0 + 1; wx = fx - ix0
    iy0 = max(0, min(n_Y - 2, int(fy))); iy1 = iy0 + 1; wy = fy - iy0
    iq = grid.n_Q - 1
    it = 0
    v00 = grid.V[ix0, iy0, iq, it]
    v01 = grid.V[ix0, iy1, iq, it]
    v10 = grid.V[ix1, iy0, iq, it]
    v11 = grid.V[ix1, iy1, iq, it]
    return float(
        (1 - wx) * (1 - wy) * v00 + (1 - wx) * wy * v01
        + wx * (1 - wy) * v10 + wx * wy * v11
    )


# ---------------------------------------------------------------------------
# Q* target generation for supervised pre-training
# ---------------------------------------------------------------------------


def compute_Q_targets(
    states: np.ndarray,          # (N, 9) - env-observation-shaped
    actions: np.ndarray,         # (N,)   - in [0, 1] (normalised)
    grid: WarmStartGrid,
    kernel: TransitionKernel,
) -> np.ndarray:
    """Compute Q*(s, a) for a batch of (state, action) pairs.

    Q*(s, a) = r(s, a) + E[V(s_{t+1}) | s_t, a] where:
      * r is the env's discounted reward with profitability gate;
      * the expectation is integrated over the kernel and bilinear-interpolated
        on V[:, :, Q-a, t+1].
    """
    N = states.shape[0]
    X = states[:, OBS_IDX_X].astype(np.float64)
    Y = states[:, OBS_IDX_Y].astype(np.float64)
    S = states[:, OBS_IDX_S].astype(np.float64)
    t_norm = states[:, OBS_IDX_T_NORM].astype(np.float64)
    t_idx = np.clip(np.rint(t_norm * grid.n_t).astype(np.int64), 0, grid.n_t - 1)
    # Q_remaining = states[:, 2] * Q_max
    Q_rem = states[:, 2].astype(np.float64) * grid.Q_max

    # Denormalise action to [q_min, q_max]
    a = grid.q_min + np.clip(actions.astype(np.float64), 0.0, 1.0) * (grid.q_max - grid.q_min)
    # Respect remaining capacity
    a = np.minimum(a, Q_rem)
    # Profitability gate
    payoff_per_unit = np.maximum(S - grid.strike, 0.0)
    gross = a * payoff_per_unit
    cost = grid.c_cost * np.power(a, grid.gamma_cost)
    net = gross - cost
    mask = net > 0.0
    a_eff = np.where(mask, a, 0.0)
    net = np.where(mask, net, 0.0)
    reward = (grid.discount_factor ** t_idx) * net

    # Continuation expectation: kernel + bilinear interp
    # Build per-sample next-state quadrature
    t_idx_next = np.clip(t_idx + 1, 0, grid.n_t)
    S_g, X_g, Y_g = build_next_state_grid_batched(
        X, Y, t_idx_next, kernel
    )  # (N, M)
    w = build_quadrature_weights(kernel)  # (M,)

    # Q_next as fractional (linear interp in Q)
    Q_next = np.clip(Q_rem - a_eff, 0.0, grid.n_Q - 1.0)

    Q_target = np.empty(N, dtype=np.float64)
    M = kernel.M
    buf = np.empty(M, dtype=np.float64)
    for i in range(N):
        # If next step is terminal (t_idx+1 >= n_t), V is 0 by definition
        if t_idx_next[i] >= grid.n_t:
            EV = 0.0
        else:
            _bilinear_interp_Q(
                grid.V[:, :, :, t_idx_next[i]],
                grid.X_axis, grid.Y_axis,
                X_g[i], Y_g[i],
                float(Q_next[i]), grid.n_Q,
                buf,
            )
            EV = float(np.dot(w, buf))
        Q_target[i] = reward[i] + EV
    return Q_target


def sample_synthetic_states(
    grid: WarmStartGrid,
    kernel: TransitionKernel,
    n_samples: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (state, action) pairs uniformly over the grid's domain.

    State features that depend only on (X, Y, Q, t) are computed exactly;
    'days_since_exercise' is set to t (i.e., no prior exercise).  Actions
    are uniform in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    n_X = grid.X_axis.shape[0]
    n_Y = grid.Y_axis.shape[0]
    X = rng.uniform(grid.X_axis[0], grid.X_axis[-1], size=n_samples)
    Y = rng.uniform(grid.Y_axis[0], grid.Y_axis[-1], size=n_samples)
    Q_rem = rng.integers(0, grid.n_Q, size=n_samples).astype(np.float64)
    t = rng.integers(0, grid.n_t, size=n_samples).astype(np.int64)
    a = rng.uniform(0.0, 1.0, size=n_samples)

    # Reconstruct the env's 9-d state vector
    S = np.exp(X + Y)  # f(t)=0
    states = np.zeros((n_samples, 9), dtype=np.float32)
    states[:, OBS_IDX_S_MINUS_K] = (S - grid.strike).astype(np.float32)
    # q_exercised / Q_max
    states[:, 1] = ((grid.Q_max - Q_rem) / grid.Q_max).astype(np.float32)
    # q_remaining / Q_max
    states[:, 2] = (Q_rem / grid.Q_max).astype(np.float32)
    # ttm / T
    states[:, 3] = ((grid.n_t - t) / grid.n_t).astype(np.float32)
    # normalised time
    states[:, OBS_IDX_T_NORM] = (t / grid.n_t).astype(np.float32)
    states[:, OBS_IDX_S] = S.astype(np.float32)
    states[:, OBS_IDX_X] = X.astype(np.float32)
    states[:, OBS_IDX_Y] = Y.astype(np.float32)
    # days_since_exercise / n_rights - assume no prior exercise => t
    states[:, 8] = (t / grid.n_t).astype(np.float32)
    actions = a.astype(np.float32)
    return states, actions


def warm_start_critic(
    agent,
    *,
    kernel: TransitionKernel,
    contract,
    n_samples: int = 16384,
    n_epochs: int = 50,
    batch_size: int = 256,
    n_X: int = 25,
    n_Y: int = 20,
    n_actions: int = 11,
    seed: int = 0,
    verbose: bool = True,
    copy_target: bool = True,
) -> WarmStartGrid:
    """End-to-end: build V grid, sample (s, a) pairs, supervise critic.

    Returns the warm-start grid (so the caller may run sanity checks).
    """
    import torch

    grid = build_value_grid(
        kernel=kernel,
        strike=contract.strike,
        c_cost=contract.c_cost,
        gamma_cost=contract.gamma_cost,
        q_min=contract.q_min, q_max=contract.q_max, Q_max=contract.Q_max,
        n_rights=contract.n_rights,
        discount_factor=contract.discount_factor,
        n_X=n_X, n_Y=n_Y, n_actions=n_actions,
        verbose=verbose,
    )
    if verbose:
        price0 = grid_price_at_origin(grid)
        print(f"[H4] Grid implied V(X=0, Y=0, Q_max, t=0) = {price0:.4f}")

    rng = np.random.default_rng(seed)
    states_np, actions_np = sample_synthetic_states(grid, kernel, n_samples, rng=rng)
    targets_np = compute_Q_targets(states_np, actions_np, grid, kernel)

    if verbose:
        print(f"[H4] Q* targets: mean={targets_np.mean():.3f}, "
              f"std={targets_np.std():.3f}, "
              f"min={targets_np.min():.3f}, max={targets_np.max():.3f}")

    device = next(agent.critic_local.parameters()).device
    states_t = torch.from_numpy(states_np).to(device=device)
    actions_t = torch.from_numpy(actions_np).to(device=device).unsqueeze(-1)
    targets_t = torch.from_numpy(targets_np.astype(np.float32)).to(device=device).unsqueeze(-1)

    # Supervised pre-training loop
    losses = []
    opt = agent.critic_optimizer
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        total = 0.0; n = 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            s = states_t[idx]; a = actions_t[idx]; y = targets_t[idx]
            q = agent.critic_local(s, a)
            loss = ((q - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * s.shape[0]; n += s.shape[0]
        avg = total / max(n, 1)
        losses.append(avg)
        if verbose and (epoch < 5 or epoch == n_epochs - 1 or epoch % 10 == 0):
            print(f"[H4] supervised epoch {epoch}/{n_epochs}  MSE={avg:.4e}")

    if copy_target:
        # Copy local -> target (so target network is also warm)
        agent.critic_target.load_state_dict(agent.critic_local.state_dict())
        target_msg = "target copied"
    else:
        target_msg = "target left untouched (TD must catch up)"
    if verbose:
        print(f"[H4] critic warm-start complete; final MSE={losses[-1]:.4e} ({target_msg})")
    return grid


__all__ = [
    "WarmStartGrid",
    "build_value_grid",
    "grid_price_at_origin",
    "compute_Q_targets",
    "sample_synthetic_states",
    "warm_start_critic",
]
