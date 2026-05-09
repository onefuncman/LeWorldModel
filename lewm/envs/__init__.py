"""Environment registry and shared interface.

Each env exposes:
    reset() -> obs (H, W, 3) uint8
    step(action) -> (obs, terminated, info)
    obs_size: int (square)
    action_dim: int
    render(state=None) -> obs

Plus a `make_env(name)` factory.
"""
from __future__ import annotations

from typing import Callable, Dict


_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable):
        _REGISTRY[name] = fn
        return fn
    return deco


def make_env(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown env {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_envs():
    return sorted(_REGISTRY)


# Trigger registration
from . import tworoom    # noqa: F401
from . import reacher    # noqa: F401
from . import pusht      # noqa: F401
from . import cube       # noqa: F401
