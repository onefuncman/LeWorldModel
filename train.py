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

from lewm import LeWMConfig, LeWorldModel, list_envs, make_dataset, recalibrate_bn
from lewm.data import MovingBlobDataset, SyntheticConfig


# Sensible per-env defaults if user doesn't override.
ENV_DEFAULTS = {
    "tworoom":      {"obs_size": 48, "seq_len": 16, "num_trajs": 512},
    "reacher":      {"obs_size": 48, "seq_len": 16, "num_trajs": 512},
    "pusht":        {"obs_size": 96, "seq_len": 16, "num_trajs": 256},
    "pusht_expert": {"obs_size": 96, "seq_len": 16, "num_trajs": 1024},
    "cube":         {"obs_size": 64, "seq_len": 16, "num_trajs": 128},
    "synthetic":    {"obs_size": 56, "seq_len": 8,  "num_trajs": 1024},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="synthetic",
                   choices=list_envs() + ["synthetic", "pusht_expert"],
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
    p.add_argument("--pred-dim", type=int, default=None,
                   help="override predictor/encoder out dim (default: LeWMConfig default = 320)")
    p.add_argument("--pred-heads", type=int, default=None,
                   help="override predictor heads (default: LeWMConfig default = 16)")
    p.add_argument("--log-rank-every", type=int, default=100,
                   help="log effective rank of encoder embeddings every N steps (0 to disable)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--amp", action="store_true",
                   help="enable mixed-precision (bf16 autocast) training")
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

    cfg_kwargs = dict(
        img_size=obs_size,
        action_dim=action_dim,
        sigreg_lambda=args.sigreg_lambda,
        sigreg_projections=args.sigreg_projections,
        max_seq_len=max(64, seq_len + 8),
    )
    if args.pred_dim is not None:
        cfg_kwargs["pred_dim"] = args.pred_dim
        cfg_kwargs["action_emb_dim"] = args.pred_dim
    if args.pred_heads is not None:
        cfg_kwargs["pred_heads"] = args.pred_heads
    cfg = LeWMConfig(**cfg_kwargs)
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

    # bf16 autocast on Ampere+ (RTX 3070 supports bf16). bf16 needs no GradScaler.
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        print("AMP: bf16 autocast enabled")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for obs, actions in loader:
            obs = obs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(obs, actions)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 20 == 0:
                dt = time.time() - t0
                msg = (
                    f"epoch {epoch:02d} step {step:05d}  "
                    f"loss {out['loss'].item():.4f}  "
                    f"pred {out['pred_loss'].item():.4f}  "
                    f"sigreg {out['sigreg'].item():.4f}"
                )
                if args.log_rank_every and step % args.log_rank_every == 0:
                    with torch.no_grad():
                        z_flat = out["z"].detach().reshape(-1, out["z"].shape[-1]).float()
                        zc = z_flat - z_flat.mean(dim=0, keepdim=True)
                        sv = torch.linalg.svdvals(zc)
                        p = sv / sv.sum().clamp_min(1e-8)
                        p = p[p > 1e-12]
                        eff_rank = float(torch.exp(-(p * p.log()).sum()).item())
                    msg += f"  eff_rank {eff_rank:.2f}"
                msg += f"  ({dt:.1f}s)"
                print(msg)
            step += 1

    # BN re-cal: the encoder's BN running stats lag the actual feature
    # distribution by end of training. Forward through the training set once
    # in train mode (with cumulative averaging) so eval-mode users of the
    # checkpoint get correctly-scaled embeddings without an extra pass.
    print("recalibrating BN running stats over training set...")
    obs_tensors = []
    for obs, _ in loader:
        # Reshape (B, T, C, H, W) -> (B*T, C, H, W); recal cares about per-frame stats
        obs_tensors.append(obs.reshape(-1, *obs.shape[2:]))
        if sum(t.shape[0] for t in obs_tensors) >= 4096:
            break
    recal_obs = torch.cat(obs_tensors, dim=0)[:4096].to(device)
    recalibrate_bn(model, recal_obs)

    import os
    ckpt = args.ckpt or f"checkpoints/lewm_{args.env}.pt"
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "env": args.env}, ckpt)
    print(f"saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
