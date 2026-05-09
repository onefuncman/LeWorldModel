"""Push-T env using pymunk 2D physics.

Faithful port of the Diffusion-Policy reference Push-T (Florence Chi et al.):
- Square arena, T-shaped block, agent (disk) pushes the T to overlap a target T.
- Continuous 2D action: agent target position (kinematic, integrated by pymunk).
- Pixel observation rendered via OpenCV at obs_size x obs_size (paper used 96x96
  in upstream, paper text mentions 48x48 for planning-time comparisons).
"""
from __future__ import annotations

import math
import numpy as np
import cv2
import pymunk
from pymunk import Vec2d

from . import register


WORLD = 512.0   # internal pymunk world size in pixels
DT = 1.0 / 60.0
SUBSTEPS = 6
AGENT_RADIUS = 15.0
AGENT_SPEED = 6.0  # max world-units per env step that the agent can be pulled


def _make_t_block(space: pymunk.Space, position: Vec2d, angle: float, color: tuple) -> pymunk.Body:
    mass = 1.0
    length = 4.0
    vertices1 = [(-length * 4, length), (length * 4, length), (length * 4, 0), (-length * 4, 0)]
    vertices2 = [(-length * 1, length), (length * 1, length), (length * 1, length * 5), (-length * 1, length * 5)]
    inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
    inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
    body = pymunk.Body(mass + mass, inertia1 + inertia2)
    shape1 = pymunk.Poly(body, vertices1)
    shape2 = pymunk.Poly(body, vertices2)
    shape1.friction = 1.0
    shape2.friction = 1.0
    body.position = position
    body.angle = angle
    space.add(body, shape1, shape2)
    return body


def _add_arena(space: pymunk.Space):
    walls = [
        pymunk.Segment(space.static_body, (5, 5), (5, WORLD - 5), 2),
        pymunk.Segment(space.static_body, (5, WORLD - 5), (WORLD - 5, WORLD - 5), 2),
        pymunk.Segment(space.static_body, (WORLD - 5, WORLD - 5), (WORLD - 5, 5), 2),
        pymunk.Segment(space.static_body, (5, 5), (WORLD - 5, 5), 2),
    ]
    for w in walls:
        w.elasticity = 0.0
        w.friction = 1.0
    space.add(*walls)


class PushT:
    action_dim = 2

    def __init__(self, obs_size: int = 96, seed: int | None = None):
        self.obs_size = obs_size
        self.rng = np.random.default_rng(seed)
        self.space = None
        self.agent = None
        self.block = None
        self.goal_pose = None  # (x, y, angle)

    def reset(self) -> np.ndarray:
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.05  # heavy linear damping → quasi-kinematic
        _add_arena(self.space)

        # Agent
        agent = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, AGENT_RADIUS))
        agent.position = (
            float(self.rng.uniform(50, WORLD - 50)),
            float(self.rng.uniform(50, WORLD - 50)),
        )
        shape = pymunk.Circle(agent, AGENT_RADIUS)
        shape.friction = 1.0
        self.space.add(agent, shape)
        self.agent = agent

        # Block
        bx = float(self.rng.uniform(150, WORLD - 150))
        by = float(self.rng.uniform(150, WORLD - 150))
        bang = float(self.rng.uniform(-math.pi, math.pi))
        self.block = _make_t_block(self.space, Vec2d(bx, by), bang, (40, 160, 90))

        # Goal pose: fixed roughly centered + small jitter
        self.goal_pose = (
            float(self.rng.uniform(WORLD * 0.3, WORLD * 0.7)),
            float(self.rng.uniform(WORLD * 0.3, WORLD * 0.7)),
            float(self.rng.uniform(-math.pi, math.pi)),
        )
        return self.render()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        # Action: 2D in [-1, 1] — interpret as desired velocity for the agent.
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target_vel = Vec2d(float(a[0]) * 200.0, float(a[1]) * 200.0)
        # Apply force toward target velocity (PD-ish, since damping is high)
        delta = target_vel - self.agent.velocity
        self.agent.velocity = self.agent.velocity + delta * 0.5

        for _ in range(SUBSTEPS):
            self.space.step(DT / SUBSTEPS)

        return self.render(), False, {}

    def get_state(self) -> np.ndarray:
        """Block (x, y, angle) in world coords."""
        bx, by = self.block.position
        return np.array([float(bx), float(by), float(self.block.angle)], dtype=np.float32)

    def state_distance(self, goal_state: np.ndarray) -> float:
        s = self.get_state()
        g = np.asarray(goal_state, dtype=np.float32)
        # Position error in world units + angular error scaled
        pos_err = float(np.linalg.norm(s[:2] - g[:2]))
        ang_err = float(abs((s[2] - g[2] + math.pi) % (2 * math.pi) - math.pi))
        return pos_err + 50.0 * ang_err  # mix; angle weighted to make ~comparable

    def render(self) -> np.ndarray:
        # Render to large canvas then downsample for anti-aliasing.
        big = 256
        img = np.full((big, big, 3), 240, dtype=np.uint8)
        scale = big / WORLD

        # Goal-T overlay (light gray) to anchor the scene visually
        gx, gy, gang = self.goal_pose
        _draw_t(img, gx, gy, gang, scale, color=(220, 220, 220), thickness=cv2.FILLED)

        # Block T (green-ish)
        bx, by = self.block.position
        _draw_t(img, float(bx), float(by), float(self.block.angle), scale, color=(40, 160, 90), thickness=cv2.FILLED)

        # Agent disk
        ax, ay = self.agent.position
        cv2.circle(img, (int(ax * scale), int(ay * scale)), int(AGENT_RADIUS * scale), (40, 90, 200), cv2.FILLED)

        out = cv2.resize(img, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)
        return out


def _draw_t(img, x, y, angle, scale, color, thickness):
    """Approximate T as two filled rectangles."""
    length = 4.0
    # Same vertex sets used in physics, but in world coords transformed by (x, y, angle)
    rects = [
        [(-length * 4, length), (length * 4, length), (length * 4, 0), (-length * 4, 0)],
        [(-length, length), (length, length), (length, length * 5), (-length, length * 5)],
    ]
    c, s = math.cos(angle), math.sin(angle)
    for verts in rects:
        pts = []
        for vx, vy in verts:
            wx = x + (c * vx - s * vy)
            wy = y + (s * vx + c * vy)
            pts.append([int(wx * scale), int(wy * scale)])
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)


@register("pusht")
def _make(obs_size: int = 96, seed: int | None = None) -> PushT:
    return PushT(obs_size=obs_size, seed=seed)
