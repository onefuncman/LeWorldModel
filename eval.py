"""LeWorldModel control-evaluation entry point.

Loads a trained checkpoint and runs MPC goal-reaching on its env.

    .venv/Scripts/python.exe eval.py --ckpt checkpoints/lewm_tworoom.pt --episodes 8
"""
from __future__ import annotations

import argparse
import time
import torch

from lewm import LeWMConfig, LeWorldModel, evaluate, MPCConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--env", type=str, default=None,
                   help="override env (default: read from checkpoint)")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--goal-steps", type=int, default=12)
    p.add_argument("--obs-size", type=int, default=None)
    # MPC knobs (paper: 300/30/30 -- expensive; lower for quick eval)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--replan-every", type=int, default=1)
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-elites", type=int, default=30)
    p.add_argument("--cem-iters", type=int, default=30)
    p.add_argument("--success-threshold", type=float, default=None)
    p.add_argument("--bn-recal-frames", type=int, default=1024,
                   help="frames used to refresh BN running stats at load (0 to skip)")
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
    print(f"loaded {args.ckpt}  env={env_name}  cfg.img_size={cfg.img_size}")

    model = LeWorldModel(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    mpc = MPCConfig(
        horizon=args.horizon,
        replan_every=args.replan_every,
        n_samples=args.cem_samples,
        n_elites=args.cem_elites,
        n_iters=args.cem_iters,
    )
    print(f"MPC: horizon={mpc.horizon} replan_every={mpc.replan_every} "
          f"CEM(n={mpc.n_samples}, elites={mpc.n_elites}, iters={mpc.n_iters})")

    t0 = time.time()
    metrics = evaluate(
        model,
        env_name,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        goal_steps=args.goal_steps,
        obs_size=args.obs_size,
        mpc_cfg=mpc,
        success_threshold=args.success_threshold,
        seed=args.seed,
        bn_recal_frames=args.bn_recal_frames,
    )
    print()
    print(f"=== {env_name} ({args.episodes} episodes, {time.time()-t0:.1f}s total) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:>32s}: {v:.4f}")
        else:
            print(f"  {k:>32s}: {v}")


if __name__ == "__main__":
    main()
