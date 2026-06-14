"""Read a hedge_swing (v65 C++) export blob and reconstruct the hedging objects the Hedging
notebook consumes — reusing the EXACT closed-form / regression math from src/greeks.py so the
only difference vs the old Python pipeline is that the policy rolls now come from the v65 C++ net.

Blob layout ('HEDG'): int32 magic,n,T,n_grid; double h,S0; double grid[ng],grid_pv[ng];
float32 S,X,Y,cf,q_before,Vp,Vm [n*T]; float32 pv[n].
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from src.greeks import _condition_on_state
from src.hedging_utils import hhk_forward_price


@dataclass
class HedgeExport:
    grid: np.ndarray          # (n_grid,) S0 nodes
    grid_pv: np.ndarray       # (n_grid,) mean PV at each node
    pv: np.ndarray            # (n,)
    pnl_unhedged: np.ndarray
    pnl_hedged: np.ndarray    # continuation-delta hedge
    delta_t: np.ndarray       # (n,N)
    theta_t: np.ndarray       # (n,N)
    Htilde: np.ndarray        # (n,N)
    spot: np.ndarray          # (n,N) == S
    t_grid: np.ndarray
    cf: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    q_before: np.ndarray


def _read(path):
    with open(path, "rb") as f:
        magic, n, T, ng = struct.unpack("<iiii", f.read(16))
        assert magic == 0x48454447, f"bad HEDG magic {magic:#x}"
        h, S0 = struct.unpack("<dd", f.read(16))
        grid = np.frombuffer(f.read(ng * 8), dtype="<f8").copy()
        grid_pv = np.frombuffer(f.read(ng * 8), dtype="<f8").copy()
        sz = n * T
        def f32(k):
            return np.frombuffer(f.read(k * 4), dtype="<f4").astype(np.float64)
        S = f32(sz).reshape(n, T); X = f32(sz).reshape(n, T); Y = f32(sz).reshape(n, T)
        cf = f32(sz).reshape(n, T); q_before = f32(sz).reshape(n, T)
        Vp = f32(sz).reshape(n, T); Vm = f32(sz).reshape(n, T)
        pv = f32(n)
    return dict(n=n, T=T, h=h, S0=S0, grid=grid, grid_pv=grid_pv, S=S, X=X, Y=Y,
                cf=cf, q_before=q_before, Vp=Vp, Vm=Vm, pv=pv)


def load_hedge(path, contract, hhk, *, condition: bool = True) -> HedgeExport:
    """Reconstruct the continuation-delta hedge (mirrors src.greeks.rl_dynamic_delta_hedge tail)."""
    d = _read(path)
    n, N = d["n"], d["T"]
    S, X, Y, cf, q_before, Vp, Vm, pv = (d[k] for k in ("S", "X", "Y", "cf", "q_before", "Vp", "Vm", "pv"))
    alpha = float(hhk["alpha"]); Tmat = float(contract.maturity); Qmax = contract.Q_max; K = contract.strike
    dt = contract.maturity / (contract.n_rights - 1)
    t_grid = np.arange(N, dtype=np.float64) * dt
    h = d["h"]
    fwd_kw = dict(delivery_time=Tmat, alpha=alpha, sigma=float(hhk["sigma"]), beta=float(hhk["beta"]),
                  lam=float(hhk["lam"]), mu_J=float(hhk["mu_J"]))
    DF = np.power(contract.discount_factor, np.arange(N, dtype=np.float64))

    F_raw = np.zeros((n, N))
    for t in range(N):
        F_raw[:, t] = hhk_forward_price(current_time=np.full(n, t_grid[t]), X_t=X[:, t], Y_t=Y[:, t], **fwd_kw)
    Htilde = DF[None, :] * F_raw

    delta_t = np.zeros((n, N)); theta_t = np.zeros((n, N))
    for t in range(N - 1):
        dV = Vp[:, t] - Vm[:, t]
        delta_raw = dV / (S[:, t] * (np.exp(h) - np.exp(-h)))
        Fp = hhk_forward_price(current_time=np.full(n, t_grid[t]), X_t=X[:, t] + h, Y_t=Y[:, t], **fwd_kw)
        Fm = hhk_forward_price(current_time=np.full(n, t_grid[t]), X_t=X[:, t] - h, Y_t=Y[:, t], **fwd_kw)
        dF = Fp - Fm
        theta_raw = np.where(np.abs(dF) > 1e-12, dV / dF, 0.0)
        active = q_before[:, t] < Qmax - 1e-6
        if condition:
            q_rem = (Qmax - q_before[:, t]) / Qmax
            delta_t[:, t] = _condition_on_state(delta_raw, S[:, t] - K, q_rem, active)
            theta_t[:, t] = _condition_on_state(theta_raw, S[:, t] - K, q_rem, active)
        else:
            delta_t[:, t] = np.where(active, delta_raw, 0.0)
            theta_t[:, t] = np.where(active, theta_raw, 0.0)

    incr = DF[1:][None, :] * (F_raw[:, 1:] - F_raw[:, :-1])
    gains = (theta_t[:, :-1] * incr).sum(axis=1)
    EpV = float(pv.mean())
    return HedgeExport(
        grid=d["grid"], grid_pv=d["grid_pv"], pv=pv, pnl_unhedged=EpV - pv, pnl_hedged=EpV - pv + gains,
        delta_t=delta_t, theta_t=theta_t, Htilde=Htilde, spot=S, t_grid=t_grid, cf=cf, X=X, Y=Y,
        q_before=q_before)


def greeks_grid(path):
    """(grid, grid_pv) for the PV-vs-S0 panel (Delta/Gamma via np.gradient in the notebook)."""
    d = _read(path)
    return d["grid"], d["grid_pv"]
