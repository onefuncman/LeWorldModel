"""ViT-tiny encoder + 1-layer MLP/BatchNorm projector."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, dim: int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """ViT-tiny: dim=192, depth=12, heads=3, patch=14 by default.

    Output is the [CLS] token passed through a 1-layer MLP + BatchNorm projector.
    The projector follows the paper: ViT's final LayerNorm interferes with the
    Gaussian regularizer, so a BN-normalized linear projection is added.
    """

    def __init__(
        self,
        img_size: int = 56,
        patch_size: int = 14,
        in_chans: int = 3,
        dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        out_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        out_dim = out_dim or dim
        self.out_dim = out_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

        # Projector: 1-layer MLP + BatchNorm. We use BN1d on the output features.
        self.proj = nn.Linear(dim, out_dim)
        self.proj_bn = nn.BatchNorm1d(out_dim, affine=False)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, out_dim)."""
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        z = self.proj(cls_out)
        # BN expects (B, D); pass-through if batch=1 in eval
        z = self.proj_bn(z)
        return z


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
