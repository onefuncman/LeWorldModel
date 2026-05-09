"""LeWorldModel training entry point.

Usage:
    .venv/Scripts/python.exe train.py --env tworoom --epochs 5 --batch-size 32
    .venv/Scripts/python.exe train.py --env synthetic --epochs 5
"""
from __future__ import annotations

import argparse
import time
import torch
from torch.utils.data import DataLoader

from lewm import LeWMConfig, LeWorldModel, list_envs, make_dataset
from lewm.data import MovingBlobDataset, SyntheticConfig


# Sensible per-env defaults if user doesn't override.
ENV_DEFAULTS = {
    "tworoom":   {"obs_size": 48, "seq_len": 16, "num_trajs": 512},
    "reacher":   {"obs_size": 48, "seq_len": 16, "num_trajs": 512},
    "pusht":     {"obs_size": 96, "seq_len": 16, "num_trajs": 256},
    "cube":      {"obs_size": 64, "seq_len": 16, "num_trajs": 128},
    "synthetic": {"obs_size": 56, "seq_len": 8,  "num_trajs": 1024},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="synthetic",
                   choices=list_envs() + ["synthetic"],
                   help="env to train on (default: synthetic moving blob)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None, help="override env obs_size")
    p.add_argument("--num-trajs", type=int, default=None)
    p.add_argument("--sigreg-lambda", type=float, default=0.1)
    p.add_argument("--sigreg-projections", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default=None)
    return p.parse_args()


def build_dataset(args):
    """Either the synthetic moving-blob dataset or one of the four real envs."""
    defaults = ENV_DEFAULTS[args.env]
    obs_size = args.img_size or defaults["obs_size"]
    seq_len = args.seq_len or defaults["seq_len"]
    num_trajs = args.num_trajs or defaults["num_trajs"]

    if args.env == "synthetic":
        cfg = SyntheticConfig(
            img_size=obs_size,
            seq_len=seq_len,
            num_trajectories=num_trajs,
        )
        return MovingBlobDataset(cfg, seed=args.seed), obs_size, seq_len, cfg.action_dim

    ds = make_dataset(
        env_name=args.env,
        num_trajectories=num_trajs,
        seq_len=seq_len,
        obs_size=obs_size,
        seed=args.seed,
    )
    action_dim = ds.actions.shape[-1]
    return ds, obs_size, seq_len, action_dim


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_set, obs_size, seq_len, action_dim = build_dataset(args)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    print(f"env={args.env}  obs={obs_size}x{obs_size}  T={seq_len}  A={action_dim}  N={len(train_set)}")

    cfg = LeWMConfig(
        img_size=obs_size,
        action_dim=action_dim,
        sigreg_lambda=args.sigreg_lambda,
        sigreg_projections=args.sigreg_projections,
        max_seq_len=max(64, seq_len + 8),
    )
    # If obs_size isn't divisible by patch_size, fall back to a smaller patch.
    if cfg.img_size % cfg.patch_size != 0:
        for p in (16, 12, 8, 4):
            if cfg.img_size % p == 0:
                cfg.patch_size = p
                break
        else:
            raise ValueError(f"cannot find patch_size dividing img_size={cfg.img_size}")
        print(f"[adjusted] patch_size -> {cfg.patch_size} (img_size={cfg.img_size})")

    model = LeWorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.2f}M")

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
    ckpt = args.ckpt or f"checkpoints/lewm_{args.env}.pt"
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "env": args.env}, ckpt)
    print(f"saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
