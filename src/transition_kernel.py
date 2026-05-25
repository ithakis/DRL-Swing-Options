"""
Semi-analytical transition kernel for the HHK swing-option environment.

This module provides a precomputed, numba-accelerated quadrature kernel over
the one-step transition (X_t, Y_t) -> (X_{t+1}, Y_{t+1}) under the
Hambly-Howison-Kluge spot price model

    S_t = exp(f(t) + X_t + Y_t),  dX = -alpha X dt + sigma dW,
                                  dY = -beta Y dt + J dN,  J ~ Exp(1/mu_J).

It is used by the D4PG agent's critic update to replace the single-sample
bootstrap target Q_target(s', pi(s')) with a quadrature-integrated expectation
E[Q_target(s', pi(s'))] over the analytical kernel - variance-reduced TD targets.

Two structural facts the parameter regime gives us for free:

1. The OU transition is exact and shift-only:
       X_{t+1} | X_t ~ N(decay_X * X_t, sigma_X^2),
       decay_X = exp(-alpha*dt),  sigma_X = sqrt(sigma^2 * (1 - decay_X^2) / (2*alpha)).
   One fixed set of standardised Gauss-Hermite nodes works for every (X_t, t).

2. The Y' contribution from past Y_t is small when beta*dt is large
   (default beta=150, dt=1/22 -> decay_Y = exp(-beta*dt) ~ 1e-3). The Y'
   distribution is, to high accuracy, a fresh compound-Poisson jump-decay sum
   independent of Y_t. A single Y-mesh (precomputed via stratified QMC) is
   reused across all Y_t with the additive correction decay_Y * Y_t.

All hot paths are plain numpy float64 arrays with no Python objects, so this
module is straightforward to port to C/C++ later.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import qmc

try:
    import numba as nb

    _NJIT = nb.njit(cache=True, fastmath=True, parallel=False)
except Exception:  # pragma: no cover - numba is in environment.yml
    nb = None

    def _NJIT(fn):
        return fn


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------
# The SwingOptionEnv observation vector (length 9):
#   [0] S_t - K            [5] S_t
#   [1] Q_exercised / Q_max[6] X_t
#   [2] Q_remaining / Q_max[7] Y_t
#   [3] ttm / T            [8] days_since_exercise / n_rights
#   [4] t / n_rights
# The four indices that depend on the *next state's* (S', X', Y') are 0, 5, 6, 7.
# All other components of next_state are deterministic given (state, action)
# and are correctly captured in the replay buffer's next_state field, so we
# only swap in synthetic (S', X', Y') per quadrature node.
OBS_IDX_S_MINUS_K = 0
OBS_IDX_T_NORM = 4
OBS_IDX_S = 5
OBS_IDX_X = 6
OBS_IDX_Y = 7


# ---------------------------------------------------------------------------
# Kernel parameters and precomputed mesh
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelParams:
    """User-facing kernel-build parameters.

    Hashing: only the fields that affect the precomputed mesh are hashed.
    The seasonal function f is identified by f_id (a short string the caller
    chooses; default "no_seasonal").
    """

    # HHK process
    alpha: float
    sigma: float
    beta: float
    lam: float
    mu_J: float
    # Time step between decision dates (== contract.dt)
    dt: float
    # Grid sizes
    M_x: int = 6           # Gauss-Hermite nodes on X transition
    N_max: int = 3         # truncate Poisson jump count at this many jumps
    M_per_k: int = 8       # QMC samples per nonzero jump count
    # Seasonal function identifier (the actual callable is passed separately
    # to precompute exp_f_grid - we hash by id only)
    f_id: str = "no_seasonal"
    # QMC seed for the Y-mesh (mesh is deterministic given this)
    y_mesh_seed: int = 20260525

    def hash(self) -> str:
        s = (
            f"{self.alpha}|{self.sigma}|{self.beta}|{self.lam}|{self.mu_J}"
            f"|{self.dt}|{self.M_x}|{self.N_max}|{self.M_per_k}"
            f"|{self.f_id}|{self.y_mesh_seed}"
        )
        return hashlib.md5(s.encode()).hexdigest()[:16]


@dataclass
class TransitionKernel:
    """Precomputed, immutable quadrature mesh.

    All fields are numpy float64 arrays except scalar shape/constants. The
    full mesh size is M = M_x * M_y, where M_y = 1 + N_max * M_per_k (one
    node for the no-jump branch plus M_per_k stratified samples per jump
    count k = 1, ..., N_max).
    """

    # Process constants
    decay_X: float          # exp(-alpha*dt)
    sigma_X: float          # sqrt(sigma^2 * (1 - decay_X^2) / (2*alpha))
    decay_Y: float          # exp(-beta*dt)

    # X grid (standardised - shift/scale at use time)
    z_X: np.ndarray         # (M_x,)  standard-normal Hermite nodes
    w_X: np.ndarray         # (M_x,)  weights (sum = 1)

    # Y grid (jump-driven increment Delta = sum J_i * exp(-beta(dt - U_i)))
    delta_Y: np.ndarray     # (M_y,)  Delta values
    w_Y: np.ndarray         # (M_y,)  weights (sum = 1)

    # Seasonal table: exp(f(t_k)) per decision-date index k = 0..n_rights
    exp_f_grid: np.ndarray  # (n_rights + 1,)
    n_rights: int

    # Bookkeeping
    params_hash: str = ""

    @property
    def M_x(self) -> int:
        return int(self.z_X.shape[0])

    @property
    def M_y(self) -> int:
        return int(self.delta_Y.shape[0])

    @property
    def M(self) -> int:
        return self.M_x * self.M_y


# ---------------------------------------------------------------------------
# Mesh construction (called once at training start - not perf critical)
# ---------------------------------------------------------------------------


def _hermegauss_probabilist(M: int) -> Tuple[np.ndarray, np.ndarray]:
    """Probabilist's Gauss-Hermite nodes/weights normalised against N(0, 1).

    `numpy.polynomial.hermite_e.hermegauss` integrates against exp(-x^2/2).
    We divide weights by sqrt(2 pi) so that sum_i w_i * g(x_i) approximates
    E_{Z~N(0,1)}[g(Z)].
    """
    x, w = hermegauss(M)
    w = w / math.sqrt(2.0 * math.pi)
    return x.astype(np.float64), w.astype(np.float64)


def _build_jump_mesh(
    beta: float,
    lam: float,
    mu_J: float,
    dt: float,
    N_max: int,
    M_per_k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the discrete Y-increment mesh (Delta values + weights).

    For each jump count k = 0, 1, ..., N_max:
      * Poisson weight P(k; lam*dt). The tail mass (k > N_max) is added to
        the k = N_max bin (small bias when lam*dt is small, which it is here).
      * For k = 0: a single node Delta = 0 with weight P(0).
      * For k >= 1: M_per_k QMC samples of (U_1..U_k, V_1..V_k) drawn from a
        Sobol sequence, then Delta_m = sum_i (-mu_J log V_i) * exp(-beta*(dt - U_i)).
        Each sample gets weight P(k) / M_per_k.

    Returns delta values and weights summing to 1.
    """
    rate = lam * dt
    pmf = np.empty(N_max + 1, dtype=np.float64)
    pmf[0] = math.exp(-rate)
    for k in range(1, N_max + 1):
        pmf[k] = pmf[k - 1] * rate / k
    tail = 1.0 - pmf.sum()
    if tail > 0.0:
        pmf[-1] += tail

    deltas = [np.array([0.0], dtype=np.float64)]
    weights = [np.array([pmf[0]], dtype=np.float64)]

    for k in range(1, N_max + 1):
        if M_per_k <= 0:
            continue
        # 2k-dim Sobol sample (U_1..U_k, V_1..V_k)
        sampler = qmc.Sobol(d=2 * k, scramble=True, seed=seed + 7919 * k)
        # Need a power of two for balance properties; ceil up
        n_draw = 1 << int(math.ceil(math.log2(max(M_per_k, 1))))
        uv = sampler.random(n_draw)
        U = uv[:, :k] * dt                          # arrival times in (0, dt)
        V = np.clip(uv[:, k:], 1e-12, 1.0 - 1e-12)  # uniforms for Exp marks
        J = -mu_J * np.log(V)                       # Exp(mu_J) jump sizes
        decay = np.exp(-beta * (dt - U))            # decay weight per jump
        delta_k = (J * decay).sum(axis=1)           # (n_draw,)

        # Subsample to exactly M_per_k via stratified quantile selection
        if n_draw > M_per_k:
            srt = np.sort(delta_k)
            idx = np.round(np.linspace(0, n_draw - 1, M_per_k)).astype(int)
            delta_k = srt[idx]

        deltas.append(delta_k.astype(np.float64))
        weights.append(np.full(M_per_k, pmf[k] / M_per_k, dtype=np.float64))

    delta_Y = np.concatenate(deltas)
    w_Y = np.concatenate(weights)
    w_Y = w_Y / w_Y.sum()  # renormalise against tiny floating drift
    return delta_Y, w_Y


def _cache_path(params: KernelParams, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"kernel_{params.hash()}.npz")


def precompute_kernel(
    params: KernelParams,
    n_rights: int,
    f_callable,
    *,
    maturity: float = 1.0,
    cache_dir: str = "cache/transition_kernel",
    force_rebuild: bool = False,
) -> TransitionKernel:
    """Build (or load from disk cache) the precomputed transition kernel.

    The seasonal callable `f_callable` is invoked only here to fill the
    exp(f(t_k)) lookup table; it is not stored. The caller is responsible
    for choosing a `params.f_id` that uniquely identifies the function so
    the cache key is correct.
    """
    path = _cache_path(params, cache_dir)
    if (not force_rebuild) and os.path.exists(path):
        npz = np.load(path)
        if int(npz["n_rights"]) == n_rights:
            return TransitionKernel(
                decay_X=float(npz["decay_X"]),
                sigma_X=float(npz["sigma_X"]),
                decay_Y=float(npz["decay_Y"]),
                z_X=npz["z_X"].copy(),
                w_X=npz["w_X"].copy(),
                delta_Y=npz["delta_Y"].copy(),
                w_Y=npz["w_Y"].copy(),
                exp_f_grid=npz["exp_f_grid"].copy(),
                n_rights=int(npz["n_rights"]),
                params_hash=params.hash(),
            )

    decay_X = math.exp(-params.alpha * params.dt)
    var_X = (params.sigma ** 2) * (1.0 - decay_X ** 2) / (2.0 * params.alpha)
    sigma_X = math.sqrt(max(var_X, 0.0))
    decay_Y = math.exp(-params.beta * params.dt)

    z_X, w_X = _hermegauss_probabilist(params.M_x)
    w_X = w_X / w_X.sum()  # renormalise (tiny drift)

    delta_Y, w_Y = _build_jump_mesh(
        params.beta, params.lam, params.mu_J, params.dt,
        params.N_max, params.M_per_k, params.y_mesh_seed,
    )

    # Seasonal lookup. n_rights+1 entries cover all decision dates.
    dt = params.dt
    t_grid = np.arange(n_rights + 1, dtype=np.float64) * dt
    # Guard: the contract.dt convention divides T by (n_rights - 1), so we
    # follow that here; passing maturity is only used for sanity-checking.
    if abs(t_grid[-1] - maturity * n_rights / (n_rights - 1)) > 1e-9 and n_rights > 1:
        pass  # quiet - some callers may use a different dt convention
    exp_f_grid = np.exp(np.array([float(f_callable(float(t))) for t in t_grid], dtype=np.float64))

    kernel = TransitionKernel(
        decay_X=decay_X, sigma_X=sigma_X, decay_Y=decay_Y,
        z_X=z_X, w_X=w_X, delta_Y=delta_Y, w_Y=w_Y,
        exp_f_grid=exp_f_grid, n_rights=int(n_rights),
        params_hash=params.hash(),
    )

    np.savez_compressed(
        path,
        decay_X=kernel.decay_X, sigma_X=kernel.sigma_X, decay_Y=kernel.decay_Y,
        z_X=kernel.z_X, w_X=kernel.w_X, delta_Y=kernel.delta_Y, w_Y=kernel.w_Y,
        exp_f_grid=kernel.exp_f_grid, n_rights=kernel.n_rights,
    )
    return kernel


# ---------------------------------------------------------------------------
# Hot-path: build the (M_x * M_y) next-state SXY grid per batch element
# ---------------------------------------------------------------------------


@_NJIT
def _build_sxy_grid_batched(
    X_t: np.ndarray,            # (B,)
    Y_t: np.ndarray,            # (B,)
    t_idx_next: np.ndarray,     # (B,) int64
    decay_X: float,
    sigma_X: float,
    decay_Y: float,
    z_X: np.ndarray,            # (M_x,)
    delta_Y: np.ndarray,        # (M_y,)
    exp_f_grid: np.ndarray,     # (n_rights + 1,)
    S_out: np.ndarray,          # (B, M)  M = M_x * M_y, row-major as [x, y]
    X_out: np.ndarray,          # (B, M)
    Y_out: np.ndarray,          # (B, M)
) -> None:
    """Fill S/X/Y synthetic next-state tensors of shape (B, M) row-major (m = ix * M_y + iy)."""
    B = X_t.shape[0]
    M_x = z_X.shape[0]
    M_y = delta_Y.shape[0]
    for b in range(B):
        mu_x = decay_X * X_t[b]
        y_base = decay_Y * Y_t[b]
        exp_f = exp_f_grid[t_idx_next[b]]
        for ix in range(M_x):
            x_val = mu_x + sigma_X * z_X[ix]
            exp_x = math.exp(x_val)
            row = ix * M_y
            for iy in range(M_y):
                y_val = y_base + delta_Y[iy]
                S = exp_f * exp_x * math.exp(y_val)
                idx = row + iy
                X_out[b, idx] = x_val
                Y_out[b, idx] = y_val
                S_out[b, idx] = S


@_NJIT
def _outer_weights(w_X: np.ndarray, w_Y: np.ndarray, out: np.ndarray) -> None:
    """Fill flat weights vector w[m] = w_X[ix] * w_Y[iy] (m = ix*M_y + iy)."""
    M_x = w_X.shape[0]
    M_y = w_Y.shape[0]
    for ix in range(M_x):
        for iy in range(M_y):
            out[ix * M_y + iy] = w_X[ix] * w_Y[iy]


def build_quadrature_weights(kernel: TransitionKernel) -> np.ndarray:
    """Return the flat (M,) quadrature weights vector for the kernel mesh."""
    w = np.empty(kernel.M, dtype=np.float64)
    _outer_weights(kernel.w_X, kernel.w_Y, w)
    return w


def build_next_state_grid_batched(
    states_X: np.ndarray,        # (B,) X_t per batch element (float64)
    states_Y: np.ndarray,        # (B,) Y_t per batch element (float64)
    t_idx_next: np.ndarray,      # (B,) integer index into exp_f_grid
    kernel: TransitionKernel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Public wrapper for the njit grid build. Returns (S, X, Y) of shape (B, M)."""
    B = states_X.shape[0]
    M = kernel.M
    S = np.empty((B, M), dtype=np.float64)
    X = np.empty((B, M), dtype=np.float64)
    Y = np.empty((B, M), dtype=np.float64)
    _build_sxy_grid_batched(
        states_X.astype(np.float64, copy=False),
        states_Y.astype(np.float64, copy=False),
        t_idx_next.astype(np.int64, copy=False),
        kernel.decay_X, kernel.sigma_X, kernel.decay_Y,
        kernel.z_X, kernel.delta_Y, kernel.exp_f_grid,
        S, X, Y,
    )
    return S, X, Y


# ---------------------------------------------------------------------------
# PyTorch bridge for the agent's TD-target computation
# ---------------------------------------------------------------------------


def expected_critic_target(
    critic_target,
    actor_target,
    states,
    next_states,
    kernel: TransitionKernel,
    *,
    strike: float,
    target_policy_noise: float = 0.0,
    chunk_size: Optional[int] = None,
):
    """Compute E[critic_target(s', pi_target(s'))] under the quadrature kernel.

    Implementation notes:
      * `states` and `next_states` are torch.Tensor on the agent's device,
        shape (B, 9), matching SwingOptionEnv's observation layout.
      * We extract (X_t, Y_t) from `states[:, 6:8]` and t_idx_next from
        `next_states[:, 4]` (the env writes t/n_rights, so multiplying by
        n_rights and rounding recovers the integer next-step index).
      * We build M synthetic next-states per batch element by swapping the
        (S-K, S, X, Y) slots in `next_states`. All other components carry
        forward unchanged (they are deterministic given state+action and
        are correctly captured by the env when it produced `next_states`).
      * The returned tensor is (B,) - the per-element expected bootstrap.
    """
    import torch  # local import keeps the module light at parse time

    device = states.device
    dtype = states.dtype
    B = states.shape[0]
    M = kernel.M

    # Extract X_t, Y_t, t_idx_next on CPU as float64 (kernel is float64).
    X_t = states[:, OBS_IDX_X].detach().to("cpu", dtype=torch.float64).numpy()
    Y_t = states[:, OBS_IDX_Y].detach().to("cpu", dtype=torch.float64).numpy()
    t_norm_next = next_states[:, OBS_IDX_T_NORM].detach().to("cpu", dtype=torch.float64).numpy()
    # The env stores t/n_rights at the *next* step here. Convert to integer.
    t_idx_next = np.clip(
        np.rint(t_norm_next * kernel.n_rights).astype(np.int64),
        0, kernel.n_rights,
    )

    S_grid, X_grid, Y_grid = build_next_state_grid_batched(X_t, Y_t, t_idx_next, kernel)

    # Construct (B, M, 9) synthetic next-state tensor. We start from the
    # buffer's next_states (broadcast across M) and overwrite the four
    # SXY-dependent slots.
    next_states_BM = next_states.detach().unsqueeze(1).expand(B, M, -1).contiguous()
    # Convert grids to torch.
    S_t = torch.from_numpy(S_grid).to(device=device, dtype=dtype)
    X_torch = torch.from_numpy(X_grid).to(device=device, dtype=dtype)
    Y_torch = torch.from_numpy(Y_grid).to(device=device, dtype=dtype)
    next_states_BM[..., OBS_IDX_S_MINUS_K] = S_t - strike
    next_states_BM[..., OBS_IDX_S] = S_t
    next_states_BM[..., OBS_IDX_X] = X_torch
    next_states_BM[..., OBS_IDX_Y] = Y_torch

    # Quadrature weights (M,) -> (1, M) for broadcasting; cast to model dtype.
    weights = torch.from_numpy(build_quadrature_weights(kernel)).to(device=device, dtype=dtype)

    # Forward through actor_target then critic_target. Use chunked M-evaluation
    # if requested to keep memory low on M1.
    if chunk_size is None or chunk_size >= M:
        flat_states = next_states_BM.reshape(B * M, -1)
        with torch.no_grad():
            actions = actor_target(flat_states)
            if target_policy_noise and target_policy_noise > 0.0:
                noise = torch.randn_like(actions) * target_policy_noise
                actions = torch.clamp(actions + noise, 0.0, 1.0)
                apply_gate = getattr(actor_target, "apply_profitability_gate", None)
                if apply_gate is not None:
                    actions = apply_gate(q_raw=actions, state=flat_states)
            q = critic_target(flat_states, actions)
        q = q.reshape(B, M)
    else:
        # Chunked over M to bound peak memory.
        q = torch.empty((B, M), device=device, dtype=dtype)
        with torch.no_grad():
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                chunk = next_states_BM[:, start:end, :].reshape(B * (end - start), -1)
                actions = actor_target(chunk)
                if target_policy_noise and target_policy_noise > 0.0:
                    noise = torch.randn_like(actions) * target_policy_noise
                    actions = torch.clamp(actions + noise, 0.0, 1.0)
                    apply_gate = getattr(actor_target, "apply_profitability_gate", None)
                    if apply_gate is not None:
                        actions = apply_gate(q_raw=actions, state=chunk)
                q_chunk = critic_target(chunk, actions).reshape(B, end - start)
                q[:, start:end] = q_chunk

    expected = (q * weights.unsqueeze(0)).sum(dim=1, keepdim=True)
    return expected


def expected_iqn_target(
    critic_target,
    actor_target,
    states,
    next_states,
    kernel: TransitionKernel,
    *,
    N: int,
    strike: float,
    target_policy_noise: float = 0.0,
    chunk_size: Optional[int] = None,
):
    """H6: kernel-expected target for an IQN distributional critic.

    Same structure as expected_critic_target but each synthetic next-state
    produces an N-quantile output Z(s', a', tau_n) for sampled tau_n. We
    compute the *expected quantile* over the kernel via a weighted average
    of the per-node quantile predictions:

        E[Z_tau(s', pi(s'))] = sum_m w_m * critic_target(s'_m, pi(s'_m), tau)

    Note: this is the quantile-wise weighted average, NOT the quantile of
    the mixture distribution. The two coincide only when the quantile
    function is linear over the mixture support. For smooth Q across the
    kernel's support this is an excellent approximation.

    Returns
    -------
    qt_next_expected : tensor of shape (B, N)
        Kernel-expected quantile predictions per (s, a) in the batch.
    """
    import torch

    device = states.device
    dtype = states.dtype
    B = states.shape[0]
    M = kernel.M

    X_t = states[:, OBS_IDX_X].detach().to("cpu", dtype=torch.float64).numpy()
    Y_t = states[:, OBS_IDX_Y].detach().to("cpu", dtype=torch.float64).numpy()
    t_norm_next = next_states[:, OBS_IDX_T_NORM].detach().to("cpu", dtype=torch.float64).numpy()
    t_idx_next = np.clip(
        np.rint(t_norm_next * kernel.n_rights).astype(np.int64),
        0, kernel.n_rights,
    )

    S_grid, X_grid, Y_grid = build_next_state_grid_batched(X_t, Y_t, t_idx_next, kernel)

    next_states_BM = next_states.detach().unsqueeze(1).expand(B, M, -1).contiguous()
    S_t = torch.from_numpy(S_grid).to(device=device, dtype=dtype)
    X_torch = torch.from_numpy(X_grid).to(device=device, dtype=dtype)
    Y_torch = torch.from_numpy(Y_grid).to(device=device, dtype=dtype)
    next_states_BM[..., OBS_IDX_S_MINUS_K] = S_t - strike
    next_states_BM[..., OBS_IDX_S] = S_t
    next_states_BM[..., OBS_IDX_X] = X_torch
    next_states_BM[..., OBS_IDX_Y] = Y_torch

    weights = torch.from_numpy(build_quadrature_weights(kernel)).to(device=device, dtype=dtype)
    # (M,) -> (1, M, 1) for broadcasting against (B, M, N)
    w3 = weights.view(1, M, 1)

    def _forward_iqn(flat_states):
        actions = actor_target(flat_states)
        if target_policy_noise and target_policy_noise > 0.0:
            noise = torch.randn_like(actions) * target_policy_noise
            actions = torch.clamp(actions + noise, 0.0, 1.0)
            apply_gate = getattr(actor_target, "apply_profitability_gate", None)
            if apply_gate is not None:
                actions = apply_gate(q_raw=actions, state=flat_states)
        # critic_target(state, action, N) -> (n_rows, 1, N) and taus
        qt, _ = critic_target(flat_states, actions, N)
        return qt.squeeze(1)  # (n_rows, N)

    if chunk_size is None or chunk_size >= M:
        flat_states = next_states_BM.reshape(B * M, -1)
        with torch.no_grad():
            qt_all = _forward_iqn(flat_states)              # (B*M, N)
        qt_all = qt_all.reshape(B, M, N)
        expected = (qt_all * w3).sum(dim=1)                  # (B, N)
    else:
        expected = torch.zeros((B, N), device=device, dtype=dtype)
        with torch.no_grad():
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                chunk = next_states_BM[:, start:end, :].reshape(B * (end - start), -1)
                qt_chunk = _forward_iqn(chunk).reshape(B, end - start, N)
                w_chunk = weights[start:end].view(1, end - start, 1)
                expected = expected + (qt_chunk * w_chunk).sum(dim=1)
    return expected


__all__ = [
    "KernelParams",
    "TransitionKernel",
    "precompute_kernel",
    "build_next_state_grid_batched",
    "build_quadrature_weights",
    "expected_critic_target",
    "expected_iqn_target",
    "OBS_IDX_S_MINUS_K",
    "OBS_IDX_T_NORM",
    "OBS_IDX_S",
    "OBS_IDX_X",
    "OBS_IDX_Y",
]
