"""Synthetic trajectory dataset for smoke tests.

Generates short rollouts of a small "moving blob" environment: a Gaussian blob
on a 2D canvas, where the action shifts the blob's position (with a little
nonlinearity). This gives an easy-to-learn, action-controllable image
trajectory — enough to confirm the pipeline trains without divergence.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticConfig:
    img_size: int = 56
    in_chans: int = 3
    seq_len: int = 8           # number of frames per trajectory
    action_dim: int = 2
    action_scale: float = 4.0  # pixel shift magnitude per action unit
    blob_sigma: float = 4.0
    num_trajectories: int = 1024


class MovingBlobDataset(Dataset):
    """Each item: (obs, actions) with shapes (T, C, H, W) and (T-1, A)."""

    def __init__(self, cfg: SyntheticConfig, seed: int = 0):
        self.cfg = cfg
        self.gen = torch.Generator().manual_seed(seed)
        # Pre-sample initial states + action sequences for reproducibility.
        H = cfg.img_size
        self.x0 = torch.empty(cfg.num_trajectories, 2).uniform_(0.25 * H, 0.75 * H, generator=self.gen)
        self.actions = torch.randn(cfg.num_trajectories, cfg.seq_len - 1, cfg.action_dim, generator=self.gen) * 0.5

    def __len__(self) -> int:
        return self.cfg.num_trajectories

    def _render(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (T, 2). Returns (T, C, H, W) RGB images of a Gaussian blob.
        H = self.cfg.img_size
        ys = torch.arange(H).float()
        xs = torch.arange(H).float()
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, H)
        sigma = self.cfg.blob_sigma
        T = positions.shape[0]
        imgs = torch.empty(T, self.cfg.in_chans, H, H)
        for t in range(T):
            cx, cy = positions[t]
            d2 = (gx - cx).pow(2) + (gy - cy).pow(2)
            blob = torch.exp(-0.5 * d2 / (sigma * sigma))
            # Color-shift the channels slightly so the encoder has something
            # non-trivial to look at across channels.
            for c in range(self.cfg.in_chans):
                imgs[t, c] = blob * (0.6 + 0.2 * c)
        return imgs

    def __getitem__(self, idx: int):
        cfg = self.cfg
        a = self.actions[idx]  # (T-1, A)
        # Roll out positions deterministically from x0 + cumulative action shifts
        # with a mild nonlinearity so the predictor can't be a pure linear map.
        H = cfg.img_size
        pos = torch.empty(cfg.seq_len, 2)
        pos[0] = self.x0[idx]
        for t in range(cfg.seq_len - 1):
            shift = cfg.action_scale * (a[t] + 0.1 * torch.tanh(a[t] * pos[t][:cfg.action_dim] / H))
            pos[t + 1] = (pos[t] + shift).clamp(2.0, H - 2.0)
        obs = self._render(pos)
        return obs, a
