"""Diagnostic: per-token AdaLN-Zero vs sequence-shared AdaLN.

The paper says "AdaLN at each layer" without specifying per-token vs shared.
The current implementation is per-token (each token modulated by its own
action), which is the natural fit for autoregressive next-step prediction
where action a_t directly drives the transition from z_t to z_{t+1}.

The sequence-shared variant (DiT-style) pools all actions in a sequence to a
single (B, A) context and broadcasts the same modulation across every token
in every layer. This is wrong for per-step action conditioning -- it strips
the predictor's ability to differentiate "which action goes with which step".

Running both for 50 steps on synthetic data should show the per-token variant
crashing pred_loss faster. If they look similar, action conditioning isn't
doing meaningful work and we should look at the action embedding pipeline.

Run:
  .venv/Scripts/python.exe -m diagnostics.adaln_variant
"""
from __future__ import annotations

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm import LeWMConfig, LeWorldModel
from lewm.predictor import AdaLNBlock, ActionConditionedPredictor
from lewm.data import MovingBlobDataset, SyntheticConfig


class SharedAdaLNBlock(nn.Module):
    """Like AdaLNBlock but the action context is mean-pooled over time first.

    Result: every token in the sequence gets the same modulation per-layer
    (DiT-style "class-conditional" AdaLN). Per-token information is lost.
    """

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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        # action_emb: (B, T, A) -> pool to (B, A) -> (B, 1, 6*D)
        pooled = action_emb.mean(dim=1, keepdim=True)
        mod = self.adaLN_modulation(pooled)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        h = self.norm1(x) * (1.0 + scale1) + shift1
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False, is_causal=False)
        x = x + gate1 * attn_out

        h = self.norm2(x) * (1.0 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


class SharedActionPredictor(ActionConditionedPredictor):
    """Identical to ActionConditionedPredictor but uses SharedAdaLNBlock."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace per-token blocks with sequence-shared ones, matching shapes.
        depth = len(self.blocks)
        first = self.blocks[0]
        dim = first.norm1.normalized_shape[0]
        num_heads = first.attn.num_heads
        action_dim = first.adaLN_modulation[-1].in_features
        mlp_ratio = first.mlp[0].out_features / dim
        # Best-effort: pull dropout from the dropout module in mlp
        dropout_p = first.mlp[2].p if isinstance(first.mlp[2], nn.Dropout) else 0.1
        self.blocks = nn.ModuleList(
            [
                SharedAdaLNBlock(dim, num_heads, action_dim, mlp_ratio=mlp_ratio, dropout=dropout_p)
                for _ in range(depth)
            ]
        )


def build_per_token_model(cfg: LeWMConfig, device: torch.device) -> LeWorldModel:
    return LeWorldModel(cfg).to(device)


def build_shared_model(cfg: LeWMConfig, device: torch.device) -> LeWorldModel:
    """Same LeWorldModel but with SharedActionPredictor swapped in."""
    model = LeWorldModel(cfg).to(device)
    shared = SharedActionPredictor(
        dim=cfg.pred_dim,
        depth=cfg.pred_depth,
        num_heads=cfg.pred_heads,
        action_dim=cfg.action_dim,
        action_emb_dim=cfg.action_emb_dim,
        dropout=cfg.pred_dropout,
        max_seq_len=cfg.max_seq_len,
    ).to(device)
    model.predictor = shared
    return model


def run_one(model: LeWorldModel, loader, steps: int, lr: float, label: str) -> list[tuple[int, float, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    history = []
    it = iter(loader)
    t0 = time.time()
    for step in range(steps):
        try:
            obs, actions = next(it)
        except StopIteration:
            it = iter(loader)
            obs, actions = next(it)
        obs = obs.to(next(model.parameters()).device, non_blocking=True)
        actions = actions.to(next(model.parameters()).device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out = model(obs, actions)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append((step, float(out["pred_loss"].item()), float(out["sigreg"].item())))
    print(f"  {label}: {steps} steps in {time.time()-t0:.1f}s")
    return history


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--img-size", type=int, default=56)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    data_cfg = SyntheticConfig(
        img_size=args.img_size,
        seq_len=args.seq_len,
        num_trajectories=1024,
        action_dim=2,
    )
    ds = MovingBlobDataset(data_cfg, seed=args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    cfg = LeWMConfig(
        img_size=args.img_size,
        action_dim=2,
        max_seq_len=max(64, args.seq_len + 8),
    )

    print(f"# AdaLN variant comparison ({args.steps} steps, batch {args.batch_size}, T={args.seq_len})")
    print()

    # Train per-token (current implementation)
    torch.manual_seed(args.seed)
    print("[per-token] building...")
    m_pt = build_per_token_model(cfg, device)
    pt_hist = run_one(m_pt, loader, args.steps, args.lr, "per-token")

    # Train sequence-shared
    torch.manual_seed(args.seed)
    print("[shared]    building...")
    m_sh = build_shared_model(cfg, device)
    sh_hist = run_one(m_sh, loader, args.steps, args.lr, "shared   ")

    # Print comparison table
    print()
    print(f"  {'step':>4s} {'per-token pred':>18s} {'shared pred':>14s} {'pt sigreg':>11s} {'sh sigreg':>11s}")
    every = max(1, args.steps // 10)
    rows = sorted(set([0, args.steps - 1] + list(range(every - 1, args.steps, every))))
    for r in rows:
        s_pt, p_pt, sg_pt = pt_hist[r]
        s_sh, p_sh, sg_sh = sh_hist[r]
        print(f"  {r:>4d} {p_pt:>18.4f} {p_sh:>14.4f} {sg_pt:>11.4f} {sg_sh:>11.4f}")

    pt_final = pt_hist[-1][1]
    sh_final = sh_hist[-1][1]
    print()
    print(f"final pred_loss: per-token={pt_final:.4f}  shared={sh_final:.4f}  ratio={pt_final/sh_final:.3f}")
    if pt_final < 0.7 * sh_final:
        print("VERDICT: per-token is meaningfully better. Per-step action conditioning matters.")
    elif pt_final > 1.3 * sh_final:
        print("VERDICT: shared is unexpectedly better. Investigate whether action conditioning is hurting.")
    else:
        print("VERDICT: per-token and shared converge to the same loss in this budget.")
        print("         This is suspicious -- per-step actions should matter for autoregressive prediction.")
        print("         Likely the action embedding / modulation is not being used. Check action gradients.")


if __name__ == "__main__":
    main()
