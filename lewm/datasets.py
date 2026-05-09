"""Trajectory datasets for the four envs.

For each env we collect random-policy rollouts and cache them to disk. For
OGBench-Cube specifically, where a state-based offline dataset already exists
upstream, we additionally support replaying its action stream through the env
and re-rendering pixels (`cube_replay`) so the action distribution matches
ogbench's published trajectories.

All datasets return torch tensors:
    obs:     (T, C, H, W)  float32 in [0, 1]
    actions: (T-1, A)      float32

A canonical filename is used for the cache so a second invocation with the
same args is a fast load.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .envs import make_env


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@dataclass
class TrajSpec:
    env_name: str
    num_trajectories: int
    seq_len: int
    obs_size: int
    action_dim: int
    source: str = "random"   # "random" or "cube_replay"
    seed: int = 0


def _cache_path(spec: TrajSpec) -> str:
    fname = (
        f"{spec.env_name}_{spec.source}_n{spec.num_trajectories}"
        f"_T{spec.seq_len}_S{spec.obs_size}_seed{spec.seed}.pt"
    )
    return os.path.join(CACHE_DIR, fname)


def _collect_random(spec: TrajSpec) -> dict:
    rng = np.random.default_rng(spec.seed)
    obs_buf = np.empty((spec.num_trajectories, spec.seq_len, spec.obs_size, spec.obs_size, 3), dtype=np.uint8)
    act_buf = np.empty((spec.num_trajectories, spec.seq_len - 1, spec.action_dim), dtype=np.float32)

    t0 = time.time()
    for i in range(spec.num_trajectories):
        env = make_env(spec.env_name, obs_size=spec.obs_size, seed=int(rng.integers(0, 2**31 - 1)))
        obs = env.reset()
        obs_buf[i, 0] = obs
        for t in range(spec.seq_len - 1):
            a = rng.uniform(-1.0, 1.0, size=spec.action_dim).astype(np.float32)
            act_buf[i, t] = a
            obs, done, _ = env.step(a)
            obs_buf[i, t + 1] = obs
            if done:
                # Pad remainder with last frame and zero action
                for tt in range(t + 1, spec.seq_len - 1):
                    act_buf[i, tt] = 0.0
                    obs_buf[i, tt + 1] = obs
                break
        if (i + 1) % max(1, spec.num_trajectories // 10) == 0:
            print(f"  collected {i+1}/{spec.num_trajectories} ({time.time()-t0:.1f}s)")
    return {"obs": obs_buf, "actions": act_buf}


def _collect_cube_replay(spec: TrajSpec) -> dict:
    """Replay ogbench's offline action stream through the env to get pixels.

    ogbench's cube-single-play-v0 dataset is a flat (N, ...) array with
    `terminals` marking episode boundaries. We segment trajectories of length
    `seq_len` from contiguous chunks of the dataset and replay actions in the
    env, rendering each frame.
    """
    import warnings; warnings.filterwarnings("ignore")
    import ogbench
    env, train_ds, _ = ogbench.make_env_and_datasets("cube-single-play-v0", compact_dataset=False)

    actions = train_ds["actions"].astype(np.float32)
    terminals = train_ds["terminals"].astype(np.float32)
    n = actions.shape[0]
    assert spec.action_dim == actions.shape[1], f"action_dim mismatch: {spec.action_dim} vs {actions.shape[1]}"

    # Find episode start indices (right after a terminal or at index 0)
    starts = [0] + (np.where(terminals == 1.0)[0] + 1).tolist()
    starts = [s for s in starts if s + spec.seq_len <= n]
    rng = np.random.default_rng(spec.seed)
    rng.shuffle(starts)
    if len(starts) < spec.num_trajectories:
        raise RuntimeError(
            f"only {len(starts)} valid {spec.seq_len}-step segments; reduce num/seq_len"
        )
    starts = starts[: spec.num_trajectories]

    obs_buf = np.empty(
        (spec.num_trajectories, spec.seq_len, spec.obs_size, spec.obs_size, 3), dtype=np.uint8
    )
    act_buf = np.empty((spec.num_trajectories, spec.seq_len - 1, spec.action_dim), dtype=np.float32)

    import cv2
    t0 = time.time()
    for i, s in enumerate(starts):
        env.reset()
        # Render initial frame
        img = env.render()
        if img.shape[0] != spec.obs_size:
            img = cv2.resize(img, (spec.obs_size, spec.obs_size), interpolation=cv2.INTER_AREA)
        obs_buf[i, 0] = img
        for t in range(spec.seq_len - 1):
            a = actions[s + t]
            act_buf[i, t] = a
            env.step(a)
            img = env.render()
            if img.shape[0] != spec.obs_size:
                img = cv2.resize(img, (spec.obs_size, spec.obs_size), interpolation=cv2.INTER_AREA)
            obs_buf[i, t + 1] = img
        if (i + 1) % max(1, spec.num_trajectories // 10) == 0:
            print(f"  cube replay {i+1}/{spec.num_trajectories} ({time.time()-t0:.1f}s)")
    return {"obs": obs_buf, "actions": act_buf}


def _action_dim_for(env_name: str) -> int:
    # Cheap one-shot probe
    env = make_env(env_name, obs_size=8)
    return env.action_dim


def make_dataset(
    env_name: str,
    num_trajectories: int = 256,
    seq_len: int = 16,
    obs_size: int | None = None,
    seed: int = 0,
    source: str | None = None,
) -> "TrajectoryDataset":
    """Load (or collect + cache) a trajectory dataset for the named env."""
    default_size = {"tworoom": 48, "reacher": 48, "pusht": 96, "cube": 64}
    obs_size = obs_size or default_size.get(env_name, 48)
    if source is None:
        source = "cube_replay" if env_name == "cube" else "random"
    action_dim = _action_dim_for(env_name)

    spec = TrajSpec(
        env_name=env_name,
        num_trajectories=num_trajectories,
        seq_len=seq_len,
        obs_size=obs_size,
        action_dim=action_dim,
        source=source,
        seed=seed,
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(spec)

    if os.path.exists(path):
        print(f"loading cached dataset {path}")
        blob = torch.load(path, weights_only=False)
    else:
        print(f"collecting {env_name} ({source}, n={num_trajectories}, T={seq_len}, S={obs_size})...")
        if source == "random":
            blob = _collect_random(spec)
        elif source == "cube_replay":
            blob = _collect_cube_replay(spec)
        else:
            raise ValueError(f"unknown source {source!r}")
        torch.save(blob, path)
        print(f"saved -> {path}")

    return TrajectoryDataset(blob["obs"], blob["actions"])


class TrajectoryDataset(Dataset):
    def __init__(self, obs: np.ndarray, actions: np.ndarray):
        # obs: (N, T, H, W, 3) uint8; actions: (N, T-1, A) float32
        self.obs = obs
        self.actions = actions

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx: int):
        obs = self.obs[idx]
        # uint8 (T, H, W, 3) -> float32 (T, 3, H, W) in [0, 1]
        obs_t = torch.from_numpy(obs).float().permute(0, 3, 1, 2) / 255.0
        actions_t = torch.from_numpy(self.actions[idx]).float()
        return obs_t, actions_t
