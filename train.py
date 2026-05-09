"""LeWorldModel training entry point.

Usage:
    .venv/Scripts/python.exe train.py --epochs 5 --batch-size 32
"""
from __future__ import annotations

import argparse
import time
import torch
from torch.utils.data import DataLoader

from lewm import LeWMConfig, LeWorldModel
from lewm.data import MovingBlobDataset, SyntheticConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--img-size", type=int, default=56)
    p.add_argument("--num-trajs", type=int, default=1024)
    p.add_argument("--sigreg-lambda", type=float, default=0.1)
    p.add_argument("--sigreg-projections", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default="checkpoints/lewm.pt")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data_cfg = SyntheticConfig(
        img_size=args.img_size,
        seq_len=args.seq_len,
        num_trajectories=args.num_trajs,
    )
    train_set = MovingBlobDataset(data_cfg, seed=args.seed)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    cfg = LeWMConfig(
        img_size=args.img_size,
        action_dim=data_cfg.action_dim,
        sigreg_lambda=args.sigreg_lambda,
        sigreg_projections=args.sigreg_projections,
        max_seq_len=max(64, args.seq_len + 8),
    )
    model = LeWorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.2f}M  (encoder + predictor)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for obs, actions in loader:
            obs = obs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            out = model(obs, actions)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 20 == 0:
                dt = time.time() - t0
                print(
                    f"epoch {epoch:02d} step {step:05d}  "
                    f"loss {out['loss'].item():.4f}  "
                    f"pred {out['pred_loss'].item():.4f}  "
                    f"sigreg {out['sigreg'].item():.4f}  "
                    f"({dt:.1f}s)"
                )
            step += 1

    import os
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.ckpt)
    print(f"saved checkpoint -> {args.ckpt}")


if __name__ == "__main__":
    main()
