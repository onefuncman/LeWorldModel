"""Cross-Entropy Method planner over latent rollouts."""
from __future__ import annotations

import torch

from .model import LeWorldModel


@torch.no_grad()
def cem_plan(
    model: LeWorldModel,
    start_obs: torch.Tensor,
    goal_obs: torch.Tensor,
    horizon: int = 10,
    n_samples: int = 300,
    n_elites: int = 30,
    n_iters: int = 30,
    init_std: float = 1.0,
    min_std: float = 0.05,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Plan `horizon` actions to drive `start_obs` toward `goal_obs`.

    Args:
        start_obs, goal_obs: (C, H, W) tensors (single observations).
    Returns:
        actions: (horizon, A) — the mean action sequence after the last CEM iteration.
    """
    device = device or next(model.parameters()).device
    model.eval()
    cfg = model.cfg

    start = start_obs.to(device).unsqueeze(0)  # (1, C, H, W)
    goal = goal_obs.to(device).unsqueeze(0)

    z_start = model.encoder(start)  # (1, D)
    z_goal = model.encoder(goal)    # (1, D)

    A = cfg.action_dim
    mean = torch.zeros(horizon, A, device=device)
    std = torch.full((horizon, A), init_std, device=device)

    for _ in range(n_iters):
        # Sample N action sequences ~ N(mean, std^2), shape (N, H, A).
        eps = torch.randn(n_samples, horizon, A, device=device)
        actions = mean + std * eps

        # Rollout each candidate from z_start in latent space.
        z_hist = z_start.unsqueeze(1).expand(n_samples, 1, -1).contiguous()  # (N, 1, D)
        z_pred = model.rollout_latent(z_hist, actions)  # (N, H, D)
        z_terminal = z_pred[:, -1]                      # (N, D)

        cost = (z_terminal - z_goal).pow(2).sum(dim=-1)  # (N,)
        elite_idx = cost.topk(n_elites, largest=False).indices
        elites = actions[elite_idx]                      # (n_elites, H, A)

        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp_min(min_std)

    return mean
