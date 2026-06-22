"""gbagent — Game Boy RL agent."""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

_MODULES = ["action", "agent", "buffer", "config", "env", "ppo"]


def __getattr__(name):
    if name in _MODULES:
        return importlib.import_module(f"gbagent.{name}")
    raise AttributeError(f"module gbagent has no attribute {name!r}")


__all__ = list(_MODULES)
