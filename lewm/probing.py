"""Probing experiments: does the encoder latent carry env-state structure?

Procedure:
1. Roll out the env with a random policy and record (obs, state) pairs.
2. Encode each obs with the *frozen* trained encoder -> latent z.
3. Fit two regressors from z to state:
   - linear closed-form (least-squares) — measures whether state is linearly
     decodable, the standard JEPA probing baseline.
   - small MLP (1 hidden layer, 256 units) — upper bound on decodability with
     a smooth nonlinear head.
4. Report R^2 on a held-out split.

A high R^2 means the embedding preserves the physical quantities of interest.
The paper validates the same property on the same envs.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .envs import make_env
from .model import LeWorldModel


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@dataclass
class ProbeResult:
    env: str
    state_dim: int
    embed_dim: int
    n_train: int
    n_test: int
    linear_r2: float
    linear_r2_per_dim: list
    mlp_r2: float
    mlp_r2_per_dim: list
    effective_rank: float
    top_singular_values: list

    def __str__(self) -> str:
        per_dim = ", ".join(f"{x:+.3f}" for x in self.linear_r2_per_dim)
        sv = ", ".join(f"{x:.1f}" for x in self.top_singular_values[:5])
        return (
            f"{self.env} (state_dim={self.state_dim}, embed_dim={self.embed_dim}, "
            f"n_train={self.n_train}, n_test={self.n_test})\n"
            f"  linear   R^2 = {self.linear_r2:+.4f}    per-dim: [{per_dim}]\n"
            f"  mlp(256) R^2 = {self.mlp_r2:+.4f}\n"
            f"  effective rank = {self.effective_rank:.2f} / {self.embed_dim}    "
            f"top-5 singular values: [{sv}]"
        )


def _probe_cache_path(env_name: str, num_frames: int, obs_size: int, seed: int) -> str:
    return os.path.join(
        CACHE_DIR, f"probe_{env_name}_n{num_frames}_S{obs_size}_seed{seed}.pt"
    )


def collect_probe_dataset(
    env_name: str,
    num_frames: int = 4096,
    obs_size: int | None = None,
    seed: int = 0,
    episode_len: int = 32,
) -> dict:
    """Collect (obs, state) pairs across many short random rollouts."""
    default_size = {"tworoom": 48, "reacher": 48, "pusht": 96, "cube": 64}
    obs_size = obs_size or default_size.get(env_name, 48)

    path = _probe_cache_path(env_name, num_frames, obs_size, seed)
    if os.path.exists(path):
        print(f"loading cached probe data {path}")
        return torch.load(path, weights_only=False)

    print(f"collecting probe data for {env_name} (n={num_frames}, S={obs_size})...")
    rng = np.random.default_rng(seed)

    # Probe one env once to learn shapes
    env = make_env(env_name, obs_size=obs_size, seed=int(rng.integers(0, 2**31 - 1)))
    if not (hasattr(env, "get_state") and hasattr(env, "state_distance")):
        raise ValueError(f"env {env_name} does not expose get_state(); cannot probe")
    env.reset()
    s0 = env.get_state()
    state_dim = int(np.asarray(s0).reshape(-1).shape[0])
    action_dim = env.action_dim

    obs_buf = np.empty((num_frames, obs_size, obs_size, 3), dtype=np.uint8)
    state_buf = np.empty((num_frames, state_dim), dtype=np.float32)

    t0 = time.time()
    i = 0
    while i < num_frames:
        env = make_env(env_name, obs_size=obs_size, seed=int(rng.integers(0, 2**31 - 1)))
        obs = env.reset()
        for t in range(episode_len):
            if i >= num_frames:
                break
            obs_buf[i] = obs
            state_buf[i] = np.asarray(env.get_state()).reshape(-1)
            i += 1
            a = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            obs, done, _ = env.step(a)
            if done:
                break
        if i % max(1, num_frames // 10) < episode_len:
            print(f"  collected {i}/{num_frames} ({time.time()-t0:.1f}s)")

    blob = {"obs": obs_buf, "state": state_buf, "obs_size": obs_size, "env": env_name}
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(blob, path)
    print(f"saved -> {path}")
    return blob


@torch.no_grad()
def recalibrate_bn(model: LeWorldModel, obs: np.ndarray, batch_size: int = 256) -> None:
    """Refresh BN running stats by forwarding through `obs` in train mode.

    The encoder's projector uses BatchNorm1d with affine=False. The running
    mean/var stored after training lag the actual feature distribution
    (we observe eval-mode std~4 vs the trained std~1), so eval-mode
    embeddings end up badly scaled. Doing one forward pass over a
    representative sample of obs in train mode resets the running stats to
    the current model's actual feature distribution; subsequent eval-mode
    calls then produce well-scaled embeddings.

    Accepts either a uint8 (N, H, W, 3) array or a float (N, 3, H, W) tensor.
    """
    device = next(model.parameters()).device
    # Reset running stats to a clean state, then accumulate fresh ones.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None  # use cumulative average instead of EMA
    model.train()
    n = obs.shape[0]
    for s in range(0, n, batch_size):
        chunk = obs[s : s + batch_size]
        if isinstance(chunk, np.ndarray):
            t = torch.from_numpy(chunk).float().permute(0, 3, 1, 2) / 255.0
        else:
            t = chunk
        model.encoder(t.to(device))
    # Restore default momentum and switch back to eval mode for downstream use
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = 0.1
    model.eval()


@torch.no_grad()
def recalibrate_bn_from_env(
    model: LeWorldModel,
    env_name: str,
    num_frames: int = 1024,
    obs_size: int | None = None,
    episode_len: int = 32,
    seed: int = 0,
) -> None:
    """Run BN recalibration using fresh random rollouts in `env_name`.

    Convenience wrapper for eval/CEM call sites that don't already have a
    cached observation buffer on hand.
    """
    blob = collect_probe_dataset(
        env_name, num_frames=num_frames, obs_size=obs_size,
        seed=seed, episode_len=episode_len,
    )
    recalibrate_bn(model, blob["obs"])


@torch.no_grad()
def _encode_all(model: LeWorldModel, obs: np.ndarray, batch_size: int = 256) -> torch.Tensor:
    device = next(model.parameters()).device
    model.eval()
    n = obs.shape[0]
    outs = []
    for s in range(0, n, batch_size):
        chunk = obs[s : s + batch_size]
        t = torch.from_numpy(chunk).float().permute(0, 3, 1, 2) / 255.0
        z = model.encoder(t.to(device))
        outs.append(z.cpu())
    return torch.cat(outs, dim=0)


def _r2(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, list]:
    """Coefficient of determination. Returns (mean, per-dim list)."""
    ss_res = ((pred - target) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-8)
    r2 = 1.0 - ss_res / ss_tot
    return float(r2.mean().item()), [float(x) for x in r2]


def _linear_probe(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_test: torch.Tensor,
    y_test: torch.Tensor,
    ridge: float = 1e-3,
) -> tuple[float, list]:
    """Closed-form ridge regression — stable when d > n (which is typical here).

    Solves W = (X^T X + lambda I)^-1 X^T Y on the augmented feature matrix
    [z, 1] (so the bias term is handled but not penalized as heavily).
    """
    x_tr = torch.cat([z_train, torch.ones(z_train.shape[0], 1, device=z_train.device)], dim=1)
    x_te = torch.cat([z_test, torch.ones(z_test.shape[0], 1, device=z_test.device)], dim=1)
    d = x_tr.shape[1]
    A = x_tr.t() @ x_tr + ridge * torch.eye(d, device=x_tr.device)
    B = x_tr.t() @ y_train
    W = torch.linalg.solve(A, B)
    pred = x_te @ W
    return _r2(pred, y_test)


def _mlp_probe(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_test: torch.Tensor,
    y_test: torch.Tensor,
    hidden: int = 256,
    epochs: int = 200,
    lr: float = 3e-3,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> tuple[float, list]:
    device = device or z_train.device
    in_dim = z_train.shape[1]
    out_dim = y_train.shape[1]
    net = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    z_train, y_train = z_train.to(device), y_train.to(device)
    z_test, y_test = z_test.to(device), y_test.to(device)
    n = z_train.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            b = idx[s : s + batch_size]
            pred = net(z_train[b])
            loss = F.mse_loss(pred, y_train[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return _r2(net(z_test), y_test)


def probe(
    model: LeWorldModel,
    env_name: str,
    num_frames: int = 4096,
    obs_size: int | None = None,
    train_frac: float = 0.8,
    seed: int = 0,
    skip_mlp: bool = False,
) -> ProbeResult:
    blob = collect_probe_dataset(env_name, num_frames=num_frames, obs_size=obs_size, seed=seed)
    obs = blob["obs"]
    state = blob["state"]

    print(f"  recalibrating BN running stats over probe set...")
    recalibrate_bn(model, obs)
    print(f"  encoding {obs.shape[0]} frames...")
    z = _encode_all(model, obs)            # (N, D), CPU
    y = torch.from_numpy(state).float()    # (N, state_dim), CPU

    # Split
    n = z.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    z, y = z[perm], y[perm]
    n_tr = int(n * train_frac)
    z_tr, y_tr = z[:n_tr], y[:n_tr]
    z_te, y_te = z[n_tr:], y[n_tr:]

    # Drop near-constant state dimensions (they explode under standardization
    # and contribute noise to mean R^2). Keep dims with std > 1e-3 in raw units.
    raw_std = y_tr.std(dim=0)
    keep = raw_std > 1e-3
    if not bool(keep.all()):
        n_drop = int((~keep).sum().item())
        print(f"  dropped {n_drop}/{y.shape[1]} near-constant state dims for probing")
    y_tr = y_tr[:, keep]
    y_te = y_te[:, keep]

    # Standardize y so per-dim R^2 is comparable
    y_mu = y_tr.mean(dim=0, keepdim=True)
    y_sd = y_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_tr_s = (y_tr - y_mu) / y_sd
    y_te_s = (y_te - y_mu) / y_sd

    print("  fitting linear probe...")
    lin_r2, lin_per = _linear_probe(z_tr, y_tr_s, z_te, y_te_s)

    if skip_mlp:
        mlp_r2, mlp_per = float("nan"), [float("nan")] * y.shape[1]
    else:
        print("  fitting MLP probe...")
        device = next(model.parameters()).device
        mlp_r2, mlp_per = _mlp_probe(z_tr, y_tr_s, z_te, y_te_s, device=device)

    # Spectral diagnostics on the centered training embeddings
    zc = z_tr - z_tr.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(zc)
    p = sv / sv.sum().clamp_min(1e-8)
    p = p[p > 1e-12]
    eff_rank = float(torch.exp(-(p * p.log()).sum()).item())

    return ProbeResult(
        env=env_name,
        state_dim=int(keep.sum().item()),
        embed_dim=z.shape[1],
        n_train=z_tr.shape[0],
        n_test=z_te.shape[0],
        linear_r2=lin_r2,
        linear_r2_per_dim=lin_per,
        mlp_r2=mlp_r2,
        mlp_r2_per_dim=mlp_per,
        effective_rank=eff_rank,
        top_singular_values=[float(x) for x in sv[:8]],
    )
