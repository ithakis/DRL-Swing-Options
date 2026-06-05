"""Regression guard: v64 argparse defaults must not drift.

If run.py's defaults change accidentally, this test fails immediately so the
"winning config is the default" invariant is always protected. Update V64_DEFAULTS
here deliberately when you intentionally promote a new canonical.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# v64 canonical — single source of truth for what the winning config is.
V64_DEFAULTS: dict = {
    # kernel
    "use_expected_target": 1,
    "kernel_M_x": 2,
    "kernel_M_per_k": 1,
    "kernel_N_max": 1,
    # network (Stage-C winners)
    "actor_output_activation": "beta_sigmoid_1.5",
    "actor_layers": 3,
    "critic_layers": 3,
    "learn_number": 2,
    "learn_every": 2,
    # optimiser
    "lr_c": 6e-4,
    "lr_a": 3e-4,
    "t": 0.0032,
    # noise + eval
    "noise_schedule": "linear",
    "noise_plateau": 0,
    "weight_averaging": "ema",
    "ema_decay": 0.999,
    # warmup + normalisation
    "critic_warmup_episodes": 512,
    "use_robust_normalization": 1,
    # focal contract
    "c_cost": 0.04,
    "gamma_cost": 2.0,
    "strike": 1.0,
    "maturity": 0.0833,
    "n_rights": 22,
    "q_max": 2.0,
    "Q_max": 20.0,
    # HHK process
    "S0": 1.0,
    "alpha": 12.0,
    "sigma": 1.2,
    "beta": 150.0,
    "lam": 6.0,
    "mu_J": 0.3,
    # LSM benchmark
    "lsm_basis": "chebyshev",
    "lsm_degree": 2,
}


def _run_defaults() -> dict:
    """Execute only run.py's create_parser() and return the default namespace."""
    src = (ROOT / "run.py").read_text()
    # Exec only lines before the main() guard so we don't launch training.
    cutoff = src.rfind("\ndef main(")
    safe_src = src[:cutoff] if cutoff != -1 else src
    ns: dict = {}
    orig_argv = sys.argv[:]
    sys.argv = ["run.py"]
    try:
        exec(compile(safe_src, str(ROOT / "run.py"), "exec"), ns)  # noqa: S102
    finally:
        sys.argv = orig_argv
    # run.py exposes the parser via ConfigManager.create_parser() (static method)
    ConfigManager = ns.get("ConfigManager")
    if ConfigManager is None or not hasattr(ConfigManager, "create_parser"):
        raise RuntimeError(
            "ConfigManager.create_parser() not found in run.py — update test_config.py"
        )
    return vars(ConfigManager.create_parser().parse_args([]))


def test_v64_defaults() -> None:
    """All v64 canonical defaults must match run.py's argparse defaults.

    If this test fails you either:
      (a) drifted a default accidentally — fix run.py, or
      (b) intentionally promoted a new canonical — update V64_DEFAULTS here.
    """
    defaults = _run_defaults()
    mismatches = []
    for key, expected in V64_DEFAULTS.items():
        actual = defaults.get(key)
        # Use approximate comparison for floats
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-9:
                mismatches.append(f"  {key}: expected {expected!r}, got {actual!r}")
        else:
            if actual != expected:
                mismatches.append(f"  {key}: expected {expected!r}, got {actual!r}")

    assert not mismatches, (
        "v64 default drift detected — run.py defaults no longer match the v64 canonical.\n"
        "Fix run.py, or update V64_DEFAULTS in tools/test_config.py if this is intentional.\n"
        + "\n".join(mismatches)
    )
