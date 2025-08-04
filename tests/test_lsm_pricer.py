import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath('.'))
from src.swing_contract import SwingContract
from src.lsm_swing_pricer import price_swing_option_lsm

def build_dataset(S0: float, future: np.ndarray, T: float) -> tuple:
    n_paths, n_steps = future.shape
    S = np.concatenate([np.full((n_paths,1), S0), future], axis=1)
    t = np.linspace(0.0, T, n_steps + 1)
    X = np.zeros_like(S)
    Y = np.zeros_like(S)
    return t, S, X, Y

def test_constant_in_the_money(tmp_path):
    contract = SwingContract(q_min=0.0, q_max=1.0, Q_min=0.0, Q_max=2.0,
                             strike=1.0, maturity=2.0, n_rights=2, r=0.0)
    future = np.full((1, contract.n_rights), 2.0)
    dataset = build_dataset(2.0, future, contract.maturity)
    price, _ = price_swing_option_lsm(contract, dataset, poly_degree=1,
                                     n_bootstrap=100, seed=0,
                                     csv_path=str(tmp_path/'res.csv'))
    assert np.isclose(price, 2.0, atol=1e-6)
    df = pd.read_csv(tmp_path/'res.csv')
    assert df['q_t'].sum() == pytest.approx(2.0)

def test_out_of_the_money(tmp_path):
    contract = SwingContract(q_min=0.0, q_max=1.0, Q_min=0.0, Q_max=2.0,
                             strike=1.0, maturity=2.0, n_rights=2, r=0.0)
    future = np.full((1, contract.n_rights), 0.5)
    dataset = build_dataset(0.5, future, contract.maturity)
    price, _ = price_swing_option_lsm(contract, dataset, poly_degree=1,
                                     n_bootstrap=100, seed=1,
                                     csv_path=str(tmp_path/'res.csv'))
    assert np.isclose(price, 0.0, atol=1e-6)
    df = pd.read_csv(tmp_path/'res.csv')
    assert df['q_t'].sum() == pytest.approx(0.0)

def test_large_monthly_dataset(tmp_path):
    contract = SwingContract(q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0,
                             strike=1.0, maturity=0.0833, n_rights=22, r=0.05)
    n_paths = 16000
    rng = np.random.default_rng(0)
    dt = contract.maturity / contract.n_rights
    increments = rng.normal(scale=0.2*np.sqrt(dt), size=(n_paths, contract.n_rights))
    future = np.exp(np.log(contract.strike) + np.cumsum(increments, axis=1))
    dataset = build_dataset(contract.strike, future, contract.maturity)
    price, ci = price_swing_option_lsm(contract, dataset, poly_degree=2,
                                       n_bootstrap=100, seed=2,
                                       csv_path=str(tmp_path/'res.csv'))
    assert np.isfinite(price)
    assert ci[1] >= ci[0]
    assert (tmp_path/'res.csv').exists()
