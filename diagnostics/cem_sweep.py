"""Tier 2 #4 + #5: CEM hyperparameter sweep + success-threshold curve.

Loads a trained checkpoint and runs `evaluate()` over a small grid of
horizon/replan-every settings with paper-spec CEM (300 samples, 30 elites,
30 iters). Then re-evaluates the best config across a range of success
thresholds to produce an honest success-rate-vs-threshold curve.

Run:
  .venv/Scripts/python.exe -m diagnostics.cem_sweep --ckpt checkpoints/lewm_reacher.pt
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
    p.add_argument("--episodes", type=int, default=8,
                   help="episodes per grid point")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--goal-steps", type=int, default=12)
    p.add_argument("--horizons", type=str, default="4,8,12",
                   help="comma-sep horizons to sweep")
    p.add_argument("--replans", type=str, default="1,2,4",
                   help="comma-sep replan-every values to sweep")
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-elites", type=int, default=30)
    p.add_argument("--cem-iters", type=int, default=30)
    p.add_argument("--success-thresholds", type=str, default=None,
                   help="comma-sep thresholds for the curve, e.g. 0.05,0.08,0.12,0.20")
    p.add_argument("--bn-recal-frames", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load(args) -> tuple[LeWorldModel, str, dict]:
    device = torch.device(args.device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = LeWMConfig(**blob["cfg"])
    env_name = args.env or blob.get("env")
    if env_name is None:
        raise SystemExit("checkpoint has no 'env' key; pass --env explicitly")
    model = LeWorldModel(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, env_name, blob


def _fmt_metrics(m: dict) -> str:
    parts = [f"success={m['success_rate']*100:5.1f}%"]
    if "mean_state_dist_reduction" in m:
        parts.append(f"d_reduction={m['mean_state_dist_reduction']:+.3f}")
        parts.append(f"d_term={m['mean_terminal_state_dist']:.3f}")
    parts.append(f"latent_d={m['mean_terminal_latent_dist']:.3g}")
    return "  ".join(parts)


def main() -> None:
    args = parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    replans = [int(x) for x in args.replans.split(",")]

    # Flush prints so progress is visible when stdout is redirected to a file
    # (Python block-buffers non-tty stdout otherwise).
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    model, env_name, _ = _load(args)
    print(f"loaded {args.ckpt}  env={env_name}")
    print(f"sweep: horizons={horizons}  replans={replans}  CEM={args.cem_samples}/{args.cem_elites}/{args.cem_iters}  episodes/point={args.episodes}")
    print()

    results: list[tuple[int, int, dict]] = []
    t_start = time.time()
    for h in horizons:
        for r in replans:
            if r > h:
                continue  # pointless: replan_every shouldn't exceed horizon
            mpc = MPCConfig(
                horizon=h,
                replan_every=r,
                n_samples=args.cem_samples,
                n_elites=args.cem_elites,
                n_iters=args.cem_iters,
            )
            t0 = time.time()
            m = evaluate(
                model,
                env_name,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                goal_steps=args.goal_steps,
                mpc_cfg=mpc,
                seed=args.seed,
                bn_recal_frames=args.bn_recal_frames if not results else 0,  # only recal once
                verbose=False,
            )
            dt = time.time() - t0
            print(f"  H={h:>2d}  R={r:>1d}  {_fmt_metrics(m)}  ({dt:.1f}s)", flush=True)
            results.append((h, r, m))

    print()
    print(f"total sweep time: {time.time()-t_start:.1f}s")

    # Pick best by state_dist_reduction (or fall back to success_rate)
    if results and "mean_state_dist_reduction" in results[0][2]:
        best = max(results, key=lambda hrm: hrm[2]["mean_state_dist_reduction"])
        key = "mean_state_dist_reduction"
    else:
        best = max(results, key=lambda hrm: hrm[2]["success_rate"])
        key = "success_rate"
    bh, br, bm = best
    print(f"best by {key}: H={bh}  R={br}  {_fmt_metrics(bm)}")

    # Success-threshold curve at best config -- post-processed from per-episode
    # terminal state distances (no extra evaluate calls needed).
    if args.success_thresholds and "per_episode_terminal_state_dist" in bm:
        print()
        print(f"# success-threshold curve at H={bh}, R={br}  (post-processed from {args.episodes} eps):")
        print(f"  {'threshold':>10s}  {'success%':>10s}")
        thresholds = [float(x) for x in args.success_thresholds.split(",")]
        dists = bm["per_episode_terminal_state_dist"]
        for thr in thresholds:
            rate = sum(1 for d in dists if d < thr) / len(dists) * 100
            print(f"  {thr:>10.3f}  {rate:>9.1f}%", flush=True)


if __name__ == "__main__":
    main()
