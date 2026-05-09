"""Probing entry point: how well does a frozen encoder's latent decode env state?

    .venv/Scripts/python.exe probe.py --ckpt checkpoints/lewm_tworoom.pt
"""
from __future__ import annotations

import argparse
import time
import torch

from lewm import LeWMConfig, LeWorldModel, probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--env", type=str, default=None,
                   help="override env (default: read from checkpoint)")
    p.add_argument("--frames", type=int, default=4096)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--obs-size", type=int, default=None)
    p.add_argument("--skip-mlp", action="store_true",
                   help="skip the MLP probe (faster)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = LeWMConfig(**blob["cfg"])
    env_name = args.env or blob.get("env")
    if env_name is None:
        raise SystemExit("checkpoint has no 'env' key; pass --env explicitly")
    # Probing needs an env with get_state(). 'pusht_expert' is a dataset, not
    # an env — fall back to the underlying 'pusht' env for probe data.
    probe_env = "pusht" if env_name == "pusht_expert" else env_name
    if probe_env != env_name:
        print(f"loaded {args.ckpt}  ckpt_env={env_name}  probe_env={probe_env}  cfg.img_size={cfg.img_size}")
    else:
        print(f"loaded {args.ckpt}  env={env_name}  cfg.img_size={cfg.img_size}")

    model = LeWorldModel(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    t0 = time.time()
    res = probe(
        model,
        probe_env,
        num_frames=args.frames,
        obs_size=args.obs_size,
        train_frac=args.train_frac,
        seed=args.seed,
        skip_mlp=args.skip_mlp,
    )
    print()
    print(f"=== probe results ({time.time()-t0:.1f}s) ===")
    print(res)


if __name__ == "__main__":
    main()
