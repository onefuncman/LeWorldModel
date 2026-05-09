"""Two-Room navigation env.

A unit square divided in two by a vertical wall at x=0.5 with a doorway at
y in [0.45, 0.55]. The agent (a small filled circle) moves with continuous
2D velocity actions, clipped to a small magnitude. Walls are treated as
hard obstacles via axis-aligned clamping + doorway check.

Observations: RGB image of the current state at obs_size x obs_size.
Actions: 2D in [-1, 1].
"""
from __future__ import annotations

import numpy as np

from . import register


WALL_X = 0.5
DOOR_Y_LO = 0.45
DOOR_Y_HI = 0.55
AGENT_RADIUS = 0.03
ACTION_SCALE = 0.04  # max ~4% of canvas per step


class TwoRoom:
    action_dim = 2

    def __init__(self, obs_size: int = 48, seed: int | None = None):
        self.obs_size = obs_size
        self.rng = np.random.default_rng(seed)
        self.pos = np.array([0.25, 0.5], dtype=np.float32)

    def reset(self) -> np.ndarray:
        # Start uniformly in the left room.
        self.pos = np.array(
            [self.rng.uniform(0.05, WALL_X - AGENT_RADIUS - 0.01),
             self.rng.uniform(0.05, 0.95)],
            dtype=np.float32,
        )
        return self.render()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        new_pos = self.pos + a * ACTION_SCALE

        # Bounds
        new_pos = np.clip(new_pos, AGENT_RADIUS, 1.0 - AGENT_RADIUS)

        # Wall: only block if crossing x=WALL_X outside the doorway.
        x0, x1 = self.pos[0], new_pos[0]
        if (x0 < WALL_X - AGENT_RADIUS) != (x1 < WALL_X - AGENT_RADIUS):
            # Crossing — interpolate y at x=WALL_X
            t = (WALL_X - x0) / (x1 - x0 + 1e-8)
            y_at_wall = self.pos[1] + t * (new_pos[1] - self.pos[1])
            if not (DOOR_Y_LO < y_at_wall < DOOR_Y_HI):
                # Block at wall - epsilon
                if x1 > x0:
                    new_pos[0] = WALL_X - AGENT_RADIUS - 1e-3
                else:
                    new_pos[0] = WALL_X + AGENT_RADIUS + 1e-3

        self.pos = new_pos.astype(np.float32)
        return self.render(), False, {}

    def render(self) -> np.ndarray:
        S = self.obs_size
        img = np.full((S, S, 3), 240, dtype=np.uint8)  # light bg

        # Vertical wall
        wall_x_px = int(round(WALL_X * S))
        door_lo_px = int(round(DOOR_Y_LO * S))
        door_hi_px = int(round(DOOR_Y_HI * S))
        img[:door_lo_px, wall_x_px - 1:wall_x_px + 1] = (40, 40, 40)
        img[door_hi_px:, wall_x_px - 1:wall_x_px + 1] = (40, 40, 40)

        # Agent disk
        cx, cy = int(round(self.pos[0] * S)), int(round(self.pos[1] * S))
        r = max(1, int(round(AGENT_RADIUS * S)))
        ys, xs = np.ogrid[:S, :S]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        img[mask] = (200, 60, 60)
        return img

    def get_state(self) -> np.ndarray:
        """Underlying state for eval: agent (x, y) in [0, 1]."""
        return self.pos.copy()

    def state_distance(self, goal_state: np.ndarray) -> float:
        return float(np.linalg.norm(self.pos - np.asarray(goal_state)))


@register("tworoom")
def _make(obs_size: int = 48, seed: int | None = None) -> TwoRoom:
    return TwoRoom(obs_size=obs_size, seed=seed)
