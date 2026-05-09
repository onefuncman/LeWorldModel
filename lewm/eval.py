"""Control evaluation: MPC over a trained LeWorldModel.

`MPCController` wraps `cem_plan`: at each control step it plans a horizon-H
action sequence in latent space, executes the first `replan_every` actions in
the real env, observes the resulting frame, then replans. This is the standard
receding-horizon MPC strategy the paper uses to mitigate autoregressive drift.

`evaluate(model, env_name, ...)` runs N episodes per env. For each:
  1. Roll out a random "demo" trajectory to pick a goal frame `goal_steps`
     steps ahead of a fresh reset.
  2. Reset the env to a fresh start.
  3. Run MPC for `max_steps` steps, recording the trajectory.
  4. Score with both terminal latent distance and (where the env exposes one)
     env-grounded `state_distance`.

Returns aggregated metrics so a CLI can print them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import torch

from .model import LeWorldModel
from .planner import cem_plan
from .envs import make_env
from .probing import recalibrate_bn_from_env


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    # uint8 (H, W, 3) -> float32 (3, H, W) in [0, 1] on device
    t = torch.from_numpy(obs).float().permute(2, 0, 1) / 255.0
    return t.to(device)


@dataclass
class MPCConfig:
    horizon: int = 10           # actions planned per CEM call
    replan_every: int = 1       # MPC: execute this many planned actions before replanning
    n_samples: int = 300
    n_elites: int = 30
    n_iters: int = 30
    init_std: float = 1.0
    min_std: float = 0.05


class MPCController:
    """Receding-horizon MPC using the trained LeWM as the world model."""

    def __init__(self, model: LeWorldModel, cfg: MPCConfig | None = None):
        self.model = model
        self.cfg = cfg or MPCConfig()
        self.model.eval()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal_obs: np.ndarray) -> np.ndarray:
        """Plan H actions from current obs toward goal; return all H actions.

        The caller decides how many of those to execute before calling `act`
        again (controlled by MPCConfig.replan_every in `evaluate`).
        """
        plan = cem_plan(
            self.model,
            start_obs=_obs_to_tensor(obs, self.device),
            goal_obs=_obs_to_tensor(goal_obs, self.device),
            horizon=self.cfg.horizon,
            n_samples=self.cfg.n_samples,
            n_elites=self.cfg.n_elites,
            n_iters=self.cfg.n_iters,
            init_std=self.cfg.init_std,
            min_std=self.cfg.min_std,
            device=self.device,
        )
        return plan.cpu().numpy()  # (H, A)


@dataclass
class EpisodeResult:
    success: bool
    terminal_latent_dist: float
    terminal_state_dist: Optional[float]
    initial_state_dist: Optional[float]
    steps: int
    states_visited: int  # for logging
    pixel_mse_terminal: float


def _latent_distance(model: LeWorldModel, a: np.ndarray, b: np.ndarray) -> float:
    device = next(model.parameters()).device
    with torch.no_grad():
        za = model.encoder(_obs_to_tensor(a, device).unsqueeze(0))
        zb = model.encoder(_obs_to_tensor(b, device).unsqueeze(0))
    return float((za - zb).pow(2).sum().item())


def _has_state_distance(env) -> bool:
    return hasattr(env, "get_state") and hasattr(env, "state_distance")


def evaluate(
    model: LeWorldModel,
    env_name: str,
    *,
    num_episodes: int = 8,
    max_steps: int = 30,
    goal_steps: int = 12,
    obs_size: Optional[int] = None,
    mpc_cfg: MPCConfig | None = None,
    success_threshold: Optional[float] = None,
    seed: int = 0,
    verbose: bool = True,
    bn_recal_frames: int = 1024,
) -> dict:
    """Run goal-reaching MPC episodes and aggregate metrics.

    Args:
        goal_steps: how many steps of random rollout to advance from the
            fresh reset to pick the goal frame. Ensures goal is reachable.
        success_threshold: if env exposes state_distance, threshold to call
            an episode a "success". Defaults to env-specific values.
    """
    mpc_cfg = mpc_cfg or MPCConfig()
    rng = np.random.default_rng(seed)

    # Per-env defaults for image size and success threshold
    default_size = {"tworoom": 48, "reacher": 48, "pusht": 96, "cube": 64}
    obs_size = obs_size or default_size.get(env_name, 48)
    default_thresh = {
        "tworoom": 0.08,   # ~8% of canvas
        "reacher": 0.05,   # ~5% of canvas
        "pusht":   30.0,   # combined pos+angle distance (world units)
        "cube":    1.0,    # 28-dim state distance
    }
    success_threshold = success_threshold or default_thresh.get(env_name, float("inf"))

    # Refresh the encoder's BN running stats so eval-mode embeddings are
    # correctly scaled (the trained running stats lag the actual feature
    # distribution; see lewm/probing.py for the full story). Skip if the
    # caller passes 0.
    if bn_recal_frames > 0:
        if verbose:
            print(f"recalibrating BN running stats over {bn_recal_frames} fresh {env_name} frames...")
        recalibrate_bn_from_env(
            model, env_name, num_frames=bn_recal_frames, obs_size=obs_size, seed=seed,
        )

    controller = MPCController(model, mpc_cfg)

    episodes: list[EpisodeResult] = []
    for ep in range(num_episodes):
        # 1) Build goal: roll out a random trajectory `goal_steps` ahead from
        #    a fresh reset and grab the resulting frame + state.
        env = make_env(env_name, obs_size=obs_size, seed=int(rng.integers(0, 2**31 - 1)))
        env.reset()
        for _ in range(goal_steps):
            a = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            env.step(a)
        goal_obs = env.render()
        goal_state = env.get_state() if _has_state_distance(env) else None

        # 2) Fresh start (different seed -> different start state).
        env = make_env(env_name, obs_size=obs_size, seed=int(rng.integers(0, 2**31 - 1)))
        obs = env.reset()
        init_dist = env.state_distance(goal_state) if goal_state is not None else None

        # 3) MPC loop with receding horizon.
        steps = 0
        traj_obs = [obs]
        terminated = False
        while steps < max_steps and not terminated:
            plan = controller.act(obs, goal_obs)  # (H, A)
            for k in range(min(mpc_cfg.replan_every, plan.shape[0])):
                if steps >= max_steps:
                    break
                obs, done, _ = env.step(plan[k])
                traj_obs.append(obs)
                steps += 1
                if done:
                    terminated = True
                    break

        # 4) Score
        latent_d = _latent_distance(model, obs, goal_obs)
        if _has_state_distance(env):
            state_d = float(env.state_distance(goal_state))
        else:
            state_d = None
        pixel_mse = float(((obs.astype(np.float32) - goal_obs.astype(np.float32)) ** 2).mean()) / (255.0 ** 2)
        success = (state_d is not None and state_d < success_threshold)
        episodes.append(EpisodeResult(
            success=success,
            terminal_latent_dist=latent_d,
            terminal_state_dist=state_d,
            initial_state_dist=init_dist,
            steps=steps,
            states_visited=len(traj_obs),
            pixel_mse_terminal=pixel_mse,
        ))
        if verbose:
            init_str = f"{init_dist:.3f}" if init_dist is not None else "n/a"
            term_str = f"{state_d:.3f}" if state_d is not None else "n/a"
            print(
                f"  ep {ep:02d}: state_dist {init_str} -> {term_str}  "
                f"latent_d {latent_d:.4g}  pix_mse {pixel_mse:.4f}  "
                f"steps {steps}  success={success}"
            )

    # Aggregate
    out = {
        "env": env_name,
        "num_episodes": num_episodes,
        "success_rate": float(np.mean([e.success for e in episodes])) if episodes else 0.0,
        "mean_terminal_latent_dist": float(np.mean([e.terminal_latent_dist for e in episodes])),
        "mean_pixel_mse_terminal": float(np.mean([e.pixel_mse_terminal for e in episodes])),
        "per_episode_terminal_latent_dist": [e.terminal_latent_dist for e in episodes],
        "per_episode_pixel_mse_terminal": [e.pixel_mse_terminal for e in episodes],
    }
    if all(e.terminal_state_dist is not None for e in episodes):
        out["mean_terminal_state_dist"] = float(np.mean([e.terminal_state_dist for e in episodes]))
        out["mean_initial_state_dist"] = float(np.mean([e.initial_state_dist for e in episodes]))
        out["mean_state_dist_reduction"] = out["mean_initial_state_dist"] - out["mean_terminal_state_dist"]
        out["per_episode_terminal_state_dist"] = [e.terminal_state_dist for e in episodes]
        out["per_episode_initial_state_dist"] = [e.initial_state_dist for e in episodes]
    return out
