"""End-to-end smoke test: tiny config, a few training steps, then CEM plan once.

    .venv/Scripts/python.exe smoke_test.py
"""
from __future__ import annotations

import time
import torch

from lewm import LeWMConfig, LeWorldModel, cem_plan
from lewm.data import MovingBlobDataset, SyntheticConfig


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Smaller-than-paper config so the smoke test runs in seconds.
    cfg = LeWMConfig(
        img_size=28,
        patch_size=14,
        enc_dim=96,
        enc_depth=4,
        enc_heads=3,
        pred_dim=96,
        pred_depth=2,
        pred_heads=4,
        pred_dropout=0.0,
        action_dim=2,
        action_emb_dim=96,
        max_seq_len=32,
        sigreg_projections=128,
        sigreg_quadrature=16,
    )
    model = LeWorldModel(cfg).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"smoke model params: {n/1e6:.2f}M")

    data_cfg = SyntheticConfig(
        img_size=cfg.img_size,
        seq_len=6,
        num_trajectories=64,
        action_dim=cfg.action_dim,
    )
    dataset = MovingBlobDataset(data_cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    # Forward once before training to confirm shapes and finite loss.
    obs0, act0 = next(iter(loader))
    obs0, act0 = obs0.to(device), act0.to(device)
    with torch.no_grad():
        out = model(obs0, act0)
    print(
        "init loss:",
        f"{out['loss'].item():.4f} (pred {out['pred_loss'].item():.4f}, sigreg {out['sigreg'].item():.4f})",
    )
    assert torch.isfinite(out["loss"]), "non-finite loss at init"
    assert out["z"].shape == (8, 6, cfg.pred_dim), f"unexpected z shape: {out['z'].shape}"
    assert out["z_pred"].shape == (8, 5, cfg.pred_dim), f"unexpected z_pred shape: {out['z_pred'].shape}"

    # A handful of optimizer steps.
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    t0 = time.time()
    losses = []
    for step in range(10):
        obs, actions = next(iter(loader))
        obs, actions = obs.to(device), actions.to(device)
        out = model(obs, actions)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append((out["loss"].item(), out["pred_loss"].item(), out["sigreg"].item()))
        print(
            f"step {step:02d}  loss {losses[-1][0]:.4f}  "
            f"pred {losses[-1][1]:.4f}  sigreg {losses[-1][2]:.4f}"
        )
    print(f"10-step training time: {time.time()-t0:.2f}s")
    assert all(torch.isfinite(torch.tensor(l[0])) for l in losses), "non-finite loss during training"

    # Run CEM once on a single (start, goal) pair drawn from the dataset.
    obs, _ = next(iter(loader))
    start_obs = obs[0, 0].to(device)
    goal_obs = obs[0, -1].to(device)
    t0 = time.time()
    plan = cem_plan(
        model,
        start_obs=start_obs,
        goal_obs=goal_obs,
        horizon=4,
        n_samples=64,
        n_elites=8,
        n_iters=4,
    )
    print(f"CEM done in {time.time()-t0:.2f}s; plan shape {tuple(plan.shape)}")
    assert plan.shape == (4, cfg.action_dim)
    assert torch.isfinite(plan).all()

    print("\nsmoke test PASSED")


if __name__ == "__main__":
    main()
