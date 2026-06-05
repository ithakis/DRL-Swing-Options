"""Import smoke tests — every src/ module must import without error.

Guards against:
  * Accidental top-level imports of optional/undeclared packages (e.g. the
    bootstrapped incident: a notebook-only dep was imported at module level).
  * Broken __init__ code triggered by import.
  * Basic circular-import regressions.

These tests run before anything trains, so they catch environment issues fast.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Every src/ module that must be importable in a clean environment.
# Add new modules here as they are created.
SRC_MODULES = [
    "src.swing_contract",
    "src.replay_buffer",
    "src.simulate_hhk_spot",
    "src.swing_env",
    "src.networks",
    "src.agent",
    "src.agent_evaluation",
    "src.transition_kernel",
    "src.lsm_swing_pricer",
    "src.greeks",
    "src.hedging_utils",
]


@pytest.mark.parametrize("module", SRC_MODULES)
def test_module_imports(module: str) -> None:
    """Each src/ module must import cleanly without side-effects or missing deps."""
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        pytest.fail(
            f"{module} failed to import: {exc}\n"
            "If this is an optional dependency (like 'bootstrapped'), "
            "make sure the import is lazy (inside the function that uses it)."
        )
    assert mod is not None
