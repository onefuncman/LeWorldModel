"""Action-conditioned Transformer predictor with per-token AdaLN-Zero.

Operates on a sequence of latent embeddings z_{1:T} produced by the encoder.
At each layer, the action a_t modulates the LayerNorms of token t (per-token
AdaLN — distinct from DiT's class-conditional, sequence-shared modulation).
The modulation MLPs are zero-initialized so the predictor starts as identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: (B, T, D); shift/scale: (B, T, D)
    return x * (1.0 + scale) + shift


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, action_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        # Per-token AdaLN modulation: 6 vectors of size dim (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_dim, 6 * dim, bias=True),
        )
        # Zero init so the block is initially identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        # action_emb: (B, T, action_dim) -> (B, T, 6*dim)
        mod = self.adaLN_modulation(action_emb)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        h = _modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False, is_causal=False)
        x = x + gate1 * attn_out

        h = _modulate(self.norm2(x), shift2, scale2)
        x = x + gate2 * self.mlp(h)
        return x


class ActionConditionedPredictor(nn.Module):
    """Predicts ẑ_{t+1} from (z_{1:t}, a_{1:t}) autoregressively via causal mask.

    Forward expects sequences of length T:
        z:       (B, T, D)
        actions: (B, T, A_raw)
    Returns:
        z_pred:  (B, T, D)  — z_pred[:, t] is the predicted next-step embedding at position t
                              (i.e. the predictor's prediction of z_{t+1} given z_{1..t}).
    A 1-layer MLP + BN projector is applied to the output, mirroring the encoder side.
    """

    def __init__(
        self,
        dim: int = 192,
        depth: int = 6,
        num_heads: int = 16,
        action_dim: int = 4,
        action_emb_dim: int = 192,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.action_emb = nn.Linear(action_dim, action_emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [AdaLNBlock(dim, num_heads, action_emb_dim, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)

        # Projector: 1-layer MLP + BN, matching the encoder side
        self.proj = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim, affine=False)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.action_emb.weight, std=0.02)
        nn.init.zeros_(self.action_emb.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def _causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        # Float mask: 0 where attend, -inf where masked. Shape (T, T).
        mask = torch.full((t, t), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        b, t, d = z.shape
        if t > self.pos_embed.shape[1]:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.pos_embed.shape[1]}")
        x = z + self.pos_embed[:, :t]
        x = self.input_dropout(x)
        a = self.action_emb(actions)
        mask = self._causal_mask(t, z.device)
        for blk in self.blocks:
            x = blk(x, a, mask)
        x = self.norm_out(x)
        # Apply projector token-wise; flatten (B, T, D) -> (B*T, D) for BN1d
        x = self.proj(x)
        x = self.proj_bn(x.reshape(b * t, d)).reshape(b, t, d)
        return x
