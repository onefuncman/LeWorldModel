"""LeWorldModel: encoder + predictor wired together with the two-term loss."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ViTEncoder
from .predictor import ActionConditionedPredictor
from .sigreg import sigreg_loss


@dataclass
class LeWMConfig:
    # Image / encoder
    img_size: int = 56
    patch_size: int = 14
    in_chans: int = 3
    enc_dim: int = 192
    enc_depth: int = 12
    enc_heads: int = 3

    # Predictor (paper: ~10M params -> dim ~320 with 6 layers, 16 heads)
    pred_dim: int = 320
    pred_depth: int = 6
    pred_heads: int = 16
    pred_dropout: float = 0.1
    action_dim: int = 4
    action_emb_dim: int = 320
    max_seq_len: int = 64

    # Loss
    sigreg_lambda: float = 0.1
    sigreg_projections: int = 1024
    sigreg_quadrature: int = 32
    sigreg_t_min: float = 0.2
    sigreg_t_max: float = 4.0


class LeWorldModel(nn.Module):
    def __init__(self, cfg: LeWMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            dim=cfg.enc_dim,
            depth=cfg.enc_depth,
            num_heads=cfg.enc_heads,
            out_dim=cfg.pred_dim,
        )
        self.predictor = ActionConditionedPredictor(
            dim=cfg.pred_dim,
            depth=cfg.pred_depth,
            num_heads=cfg.pred_heads,
            action_dim=cfg.action_dim,
            action_emb_dim=cfg.action_emb_dim,
            dropout=cfg.pred_dropout,
            max_seq_len=cfg.max_seq_len,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, C, H, W) -> z: (B, T, D)."""
        b, t = obs.shape[:2]
        z = self.encoder(obs.reshape(b * t, *obs.shape[2:]))
        return z.reshape(b, t, -1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """Compute predictor outputs and the two-term loss.

        obs:     (B, T, C, H, W)
        actions: (B, T-1, A)  -- one action per transition
        """
        b, t = obs.shape[:2]
        if actions.shape[1] != t - 1:
            raise ValueError(
                f"expected actions of length T-1={t-1}, got {actions.shape[1]}"
            )

        z = self.encode(obs)                       # (B, T, D)
        z_in = z[:, :-1]                           # (B, T-1, D)
        z_target = z[:, 1:]                        # (B, T-1, D)
        z_pred = self.predictor(z_in, actions)     # (B, T-1, D)

        # Next-embedding prediction loss (no stop-gradient on target — paper
        # explicitly avoids EMA / stop-grad and relies on SIGReg for stability).
        pred_loss = F.mse_loss(z_pred, z_target)

        # SIGReg over the encoded embeddings, applied stepwise: shape (T, B, D).
        # We compute the statistic for each timestep across the batch and average.
        z_for_reg = z.transpose(0, 1).contiguous()  # (T, B, D)
        reg = sigreg_loss(
            z_for_reg,
            num_projections=self.cfg.sigreg_projections,
            t_min=self.cfg.sigreg_t_min,
            t_max=self.cfg.sigreg_t_max,
            num_quadrature=self.cfg.sigreg_quadrature,
        )

        loss = pred_loss + self.cfg.sigreg_lambda * reg
        return {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "sigreg": reg.detach(),
            "z": z,
            "z_pred": z_pred,
        }

    @torch.no_grad()
    def rollout_latent(
        self,
        z_history: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive rollout in latent space.

        z_history: (B, T0, D)  -- known prefix (typically T0=1 from goal-conditioning)
        actions:   (B, H, A)
        Returns next-step predictions stacked: (B, H, D).
        """
        b, t0, d = z_history.shape
        h = actions.shape[1]
        if t0 + h > self.cfg.max_seq_len:
            raise ValueError(
                f"prefix+horizon {t0+h} exceeds max_seq_len {self.cfg.max_seq_len}"
            )

        z_seq = z_history.clone()
        # We will roll forward, appending one predicted z per step.
        # The predictor expects (z_{1..t}, a_{1..t}) and outputs ẑ_{2..t+1};
        # we take its last token as the next-step prediction.
        out = []
        for k in range(h):
            t_now = z_seq.shape[1]
            a_in = actions[:, :t_now]  # one action per current token
            # Pad a_in to t_now with the latest action when prefix is shorter than k+1.
            # In our simple usage T0=1 and we feed [a_0, a_1, ..., a_{t-1}] for tokens [z_0..z_{t-1}].
            z_pred = self.predictor(z_seq, a_in)
            next_z = z_pred[:, -1:]
            out.append(next_z)
            z_seq = torch.cat([z_seq, next_z], dim=1)
        return torch.cat(out, dim=1)
