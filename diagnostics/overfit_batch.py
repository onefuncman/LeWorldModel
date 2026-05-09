"""Diagnostic: can the model overfit a tiny fixed batch?

Take 4 trajectories from the synthetic moving-blob dataset, lock them as one
batch, train for N steps with the full two-term loss. We expect:
  - pred_loss collapses toward ~0 as the predictor memorizes z_{t+1} given
    (z_{1:t}, a_{1:t}). With sigreg lambda > 0, target z_{t+1} also moves
    (no stop-grad), so the floor isn't exactly zero, but it should be
    small relative to the init value (>10x reduction).
  - sigreg stays bounded (it's evaluated on 4 samples per step so will be noisy).

If pred_loss does NOT crash on a 4-trajectory batch, no training budget will
help on a real dataset.

Run:
  .venv/Scripts/python.exe -m diagnostics.overfit_batch
"""
from __future__ import annotations

import argparse
import time
import torch

from lewm import LeWMConfig, LeWorldModel
from lewm.data import MovingBlobDataset, SyntheticConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--img-size", type=int, default=56)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sigreg-lambda", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data_cfg = SyntheticConfig(
        img_size=args.img_size,
        seq_len=args.seq_len,
        num_trajectories=args.batch_size,
        action_dim=2,
    )
    ds = MovingBlobDataset(data_cfg, seed=args.seed)
    obs_list, act_list = [], []
    for i in range(args.batch_size):
        o, a = ds[i]
        obs_list.append(o)
        act_list.append(a)
    obs = torch.stack(obs_list).to(device)        # (B, T, C, H, W)
    actions = torch.stack(act_list).to(device)    # (B, T-1, A)
    print(f"fixed batch: obs {tuple(obs.shape)}, actions {tuple(actions.shape)}")

    cfg = LeWMConfig(
        img_size=args.img_size,
        action_dim=actions.shape[-1],
        sigreg_lambda=args.sigreg_lambda,
        max_seq_len=max(64, args.seq_len + 8),
    )
    model = LeWorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params/1e6:.2f}M")

    # Init reading
    with torch.no_grad():
        out0 = model(obs, actions)
    pred0 = float(out0["pred_loss"].item())
    print(f"step 000  pred {pred0:.4f}  sigreg {out0['sigreg'].item():.4f}")

    history = [(0, pred0, float(out0["sigreg"].item()))]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        out = model(obs, actions)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            history.append((step, float(out["pred_loss"].item()), float(out["sigreg"].item())))
            print(f"step {step:03d}  pred {history[-1][1]:.4f}  sigreg {history[-1][2]:.4f}")

    dt = time.time() - t0
    pred_final = history[-1][1]
    ratio = pred_final / max(pred0, 1e-12)
    print(f"\n{args.steps} steps in {dt:.1f}s")
    print(f"pred_loss: {pred0:.4f} -> {pred_final:.4f}  (final/init = {ratio:.4f})")

    # Bonus: also check that the predictor's outputs match z_target on this fixed batch.
    model.eval()
    with torch.no_grad():
        out_eval = model(obs, actions)
        per_token_mse = (out_eval["z_pred"] - out_eval["z"][:, 1:]).pow(2).mean(dim=-1)
    print(f"eval-mode pred_loss: {out_eval['pred_loss'].item():.4f}")
    print(f"per-step mse mean: {per_token_mse.mean(dim=0).cpu().numpy().round(4)}")

    # Heuristic verdict
    if ratio < 0.1:
        print("\nVERDICT: model overfits the fixed batch (>10x pred_loss reduction). OK.")
    elif ratio < 0.5:
        print("\nVERDICT: partial overfit (pred_loss reduced but not crushed). Worth investigating.")
    else:
        print("\nVERDICT: FAILED to overfit. The predictor cannot memorize 4 short trajectories.")
        print("         No training budget will fix this -- look for an architectural bug first.")


if __name__ == "__main__":
    main()
