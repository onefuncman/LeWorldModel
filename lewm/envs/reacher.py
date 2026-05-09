"""Reacher: 2-link planar arm.

State: two joint angles theta1, theta2 (and angular velocities).
Action: 2D continuous joint torques in [-1, 1].
Dynamics: simple damped point-mass at each joint. Lengths l1, l2; gravity off.

Observations: RGB image of the arm, with a fixed target dot drawn for context.
Goal-conditioning is exposed via `set_target(xy)`.
"""
from __future__ import annotations

import math
import numpy as np

from . import register


L1 = 0.4
L2 = 0.4
DAMPING = 0.9
TORQUE_SCALE = 0.5
DT = 0.05


class Reacher:
    action_dim = 2

    def __init__(self, obs_size: int = 48, seed: int | None = None):
        self.obs_size = obs_size
        self.rng = np.random.default_rng(seed)
        self.theta = np.zeros(2, dtype=np.float32)
        self.omega = np.zeros(2, dtype=np.float32)
        self.target = np.array([0.5, 0.5], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.theta = self.rng.uniform(-math.pi, math.pi, size=2).astype(np.float32)
        self.omega = np.zeros(2, dtype=np.float32)
        # Sample target inside reachable annulus
        r = self.rng.uniform(0.1, L1 + L2 - 0.05)
        ang = self.rng.uniform(0, 2 * math.pi)
        self.target = (np.array([0.5, 0.5]) + r * np.array([math.cos(ang), math.sin(ang)])).astype(np.float32)
        return self.render()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0) * TORQUE_SCALE
        self.omega = DAMPING * self.omega + a
        self.theta = (self.theta + DT * self.omega).astype(np.float32)
        # Wrap angles to [-pi, pi]
        self.theta = ((self.theta + math.pi) % (2 * math.pi) - math.pi).astype(np.float32)
        return self.render(), False, {}

    def _endpoints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base = np.array([0.5, 0.5], dtype=np.float32)
        j1 = base + L1 * np.array([math.cos(self.theta[0]), math.sin(self.theta[0])], dtype=np.float32)
        # Second joint angle is relative to first
        j2 = j1 + L2 * np.array(
            [math.cos(self.theta[0] + self.theta[1]), math.sin(self.theta[0] + self.theta[1])],
            dtype=np.float32,
        )
        return base, j1, j2

    def render(self) -> np.ndarray:
        S = self.obs_size
        img = np.full((S, S, 3), 245, dtype=np.uint8)
        base, j1, j2 = self._endpoints()
        _draw_line(img, base, j1, (60, 60, 200), thickness=2)
        _draw_line(img, j1, j2, (60, 200, 60), thickness=2)
        _draw_disk(img, j2, 0.025, (200, 60, 60))    # tip
        _draw_disk(img, self.target, 0.020, (60, 60, 60), filled=False)  # target ring
        _draw_disk(img, base, 0.018, (40, 40, 40))   # base
        return img


def _to_px(p: np.ndarray, S: int) -> tuple[int, int]:
    return int(round(p[0] * S)), int(round((1.0 - p[1]) * S))  # flip y for image


def _draw_disk(img: np.ndarray, center: np.ndarray, radius: float, color, filled: bool = True):
    S = img.shape[0]
    cx, cy = _to_px(center, S)
    r = max(1, int(round(radius * S)))
    ys, xs = np.ogrid[:S, :S]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    if filled:
        mask = d2 <= r * r
    else:
        mask = (d2 <= r * r) & (d2 >= max(0, r - 2) ** 2)
    img[mask] = color


def _draw_line(img: np.ndarray, p0: np.ndarray, p1: np.ndarray, color, thickness: int = 1):
    S = img.shape[0]
    x0, y0 = _to_px(p0, S)
    x1, y1 = _to_px(p1, S)
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    ts = np.linspace(0, 1, n)
    xs = (x0 + ts * (x1 - x0)).round().astype(int)
    ys = (y0 + ts * (y1 - y0)).round().astype(int)
    for dy in range(-thickness, thickness + 1):
        for dx in range(-thickness, thickness + 1):
            xx = np.clip(xs + dx, 0, S - 1)
            yy = np.clip(ys + dy, 0, S - 1)
            img[yy, xx] = color


@register("reacher")
def _make(obs_size: int = 48, seed: int | None = None) -> Reacher:
    return Reacher(obs_size=obs_size, seed=seed)
