import numpy as np

from src.swing_contract import SwingContract
from src.swing_env import SwingOptionEnv


def make_env_with_paths(strike: float, spot_first: float, n_rights: int = 5, q_min: float = 0.0, q_max: float = 1.0):
    contract = SwingContract(
        q_min=q_min,
        q_max=q_max,
        Q_min=0.0,
        Q_max=10.0,
        strike=strike,
        maturity=1.0,
        n_rights=n_rights,
        min_refraction_days=0,
        r=0.05,
    )
    # Time grid length n_rights (env uses dt = maturity/(n_rights-1))
    t = np.linspace(0, contract.maturity, n_rights, dtype=float)
    # One path with constant spot = spot_first (makes first step OTM/ITM as desired)
    S = np.full((1, n_rights), spot_first, dtype=float)
    X = np.zeros_like(S)
    Y = np.zeros_like(S)
    env = SwingOptionEnv(contract=contract, hhk_params={}, dataset=(t, S, X, Y))
    return env


def test_otm_exercise_is_ignored():
    strike = 100.0
    spot_first = 95.0  # OTM
    env = make_env_with_paths(strike, spot_first)

    obs, info = env.reset()
    # Propose max normalized action (attempt to exercise)
    action = np.array([1.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    # Should be ignored: q_actual forced to 0, reward 0, cumulative_exercised unchanged
    assert step_info["q_actual"] == 0.0
    assert step_info["cumulative_exercised"] == 0.0
    assert reward == 0.0
    assert not terminated
    assert not truncated


def test_itm_exercise_is_registered():
    strike = 100.0
    spot_first = 120.0  # ITM
    env = make_env_with_paths(strike, spot_first)

    obs, info = env.reset()
    action = np.array([1.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    # Should register: q_actual > 0, reward > 0, cumulative_exercised > 0
    assert step_info["q_actual"] > 0.0
    assert step_info["cumulative_exercised"] > 0.0
    assert reward > 0.0
    assert not terminated
    assert not truncated
