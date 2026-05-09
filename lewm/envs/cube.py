"""OGBench-Cube wrapper.

Wraps `ogbench.make_env_and_datasets('cube-single-play-v0')`. The bundled
dataset is state-based (28-dim obs, 5-dim action, 1M transitions). To run a
pixel-based world model on it, we re-render frames by setting the env's
qpos/qvel from the saved states and calling `env.render()`.

Two modes:
- interactive `step(a)` — runs the underlying env and returns rendered RGB.
- offline iteration over the dataset (used by the dataset loader).

We keep this module light: the heavy dataset-rendering loop lives in
`lewm.datasets.cube_dataset`.
"""
from __future__ import annotations

import warnings
import numpy as np

from . import register


_DATASET_NAME = "cube-single-play-v0"


def _load_ogbench(env_only: bool = True):
    warnings.filterwarnings("ignore")
    import ogbench
    return ogbench.make_env_and_datasets(_DATASET_NAME, env_only=env_only)


class Cube:
    """Pixel-rendering wrapper over ogbench's cube-single-play-v0."""

    def __init__(self, obs_size: int = 64, seed: int | None = None):
        self.obs_size = obs_size
        self._env = _load_ogbench(env_only=True)
        self.action_dim = self._env.action_space.shape[0]
        if seed is not None:
            self._env.reset(seed=int(seed))

    def reset(self) -> np.ndarray:
        self._env.reset()
        return self.render()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, self._env.action_space.low, self._env.action_space.high)
        _obs, _r, terminated, truncated, info = self._env.step(a)
        done = bool(terminated or truncated)
        return self.render(), done, info

    def render(self) -> np.ndarray:
        img = self._env.render()
        if img is None:
            raise RuntimeError("ogbench env returned None from render()")
        if img.shape[0] != self.obs_size or img.shape[1] != self.obs_size:
            import cv2
            img = cv2.resize(img, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)
        return img

    @property
    def underlying(self):
        return self._env


@register("cube")
def _make(obs_size: int = 64, seed: int | None = None) -> Cube:
    return Cube(obs_size=obs_size, seed=seed)
