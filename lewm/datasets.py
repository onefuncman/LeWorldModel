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


_PUSHT_EXPERT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
_PUSHT_EXPERT_DIR = os.path.expanduser("~/.lewm/pusht_expert")


def _find_zarr_root(base_dir: str) -> str | None:
    """Walk base_dir looking for a directory containing a 'data' subdir with zarr arrays."""
    for root, dirs, _ in os.walk(base_dir):
        if "data" in dirs and "meta" in dirs:
            return root
    return None


def _download_pusht_expert() -> str:
    """Download + unzip the Diffusion-Policy Push-T expert dataset.

    Returns the local path to the unzipped zarr root (the directory that
    contains `data/img`, `data/action`, `meta/episode_ends`).
    """
    import urllib.request
    import zipfile

    os.makedirs(_PUSHT_EXPERT_DIR, exist_ok=True)
    zarr_root = _find_zarr_root(_PUSHT_EXPERT_DIR)
    if zarr_root is not None:
        return zarr_root

    zip_path = os.path.join(_PUSHT_EXPERT_DIR, "pusht.zip")
    if not os.path.exists(zip_path):
        print(f"downloading Push-T expert dataset ({_PUSHT_EXPERT_URL})...")
        with urllib.request.urlopen(_PUSHT_EXPERT_URL) as resp, open(zip_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            chunk = 1 << 20
            read = 0
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                read += len(buf)
                if total:
                    print(f"  {read/total*100:5.1f}%  ({read/1e6:.1f}/{total/1e6:.1f} MB)", end="\r")
        print()
    print(f"unzipping into {_PUSHT_EXPERT_DIR}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(_PUSHT_EXPERT_DIR)
    zarr_root = _find_zarr_root(_PUSHT_EXPERT_DIR)
    if zarr_root is None:
        raise RuntimeError(f"could not find a zarr root in {_PUSHT_EXPERT_DIR} after unzip")
    return zarr_root


def _collect_pusht_expert(spec: TrajSpec) -> dict:
    """Convert Diffusion-Policy's pusht_cchi_v7_replay.zarr into our format.

    The upstream dataset stores all transitions in flat arrays:
      data/img    : (N, 96, 96, 3) uint8
      data/action : (N, 2)         float32  (target end-effector position in world units)
      meta/episode_ends : (E,)     int       (cumulative end indices)

    We slide a length-`seq_len` window across each episode and sample
    `num_trajectories` windows uniformly at random.
    """
    import zarr
    import cv2

    zarr_root = _download_pusht_expert()
    store = zarr.open(zarr_root, mode="r")
    imgs = store["data/img"]                     # (N, 96, 96, 3) uint8
    acts = store["data/action"]                  # (N, 2) float32
    ep_ends = np.asarray(store["meta/episode_ends"][:], dtype=np.int64)
    if spec.action_dim != acts.shape[1]:
        raise ValueError(
            f"pusht_expert action_dim mismatch: spec={spec.action_dim} vs zarr={acts.shape[1]}"
        )

    # Build (start, end] episode ranges
    starts = np.concatenate([[0], ep_ends[:-1]])
    rng = np.random.default_rng(spec.seed)

    # Randomly sample (episode, offset) pairs whose window fits.
    candidates: list[tuple[int, int]] = []
    for s, e in zip(starts, ep_ends):
        if e - s >= spec.seq_len:
            candidates.extend((s, off) for off in range(e - s - spec.seq_len + 1))
    if len(candidates) < spec.num_trajectories:
        raise RuntimeError(
            f"only {len(candidates)} valid {spec.seq_len}-step windows in expert data;"
            f" reduce num/seq_len"
        )
    idx = rng.choice(len(candidates), size=spec.num_trajectories, replace=False)
    chosen = [candidates[i] for i in idx]

    obs_buf = np.empty(
        (spec.num_trajectories, spec.seq_len, spec.obs_size, spec.obs_size, 3), dtype=np.uint8
    )
    act_buf = np.empty(
        (spec.num_trajectories, spec.seq_len - 1, spec.action_dim), dtype=np.float32
    )

    t0 = time.time()
    for i, (s, off) in enumerate(chosen):
        base = s + off
        # Read the seq_len consecutive frames + (seq_len-1) actions
        win_imgs = np.asarray(imgs[base : base + spec.seq_len])
        win_acts = np.asarray(acts[base : base + spec.seq_len - 1], dtype=np.float32)
        if win_imgs.shape[1] != spec.obs_size:
            # Resize to requested resolution
            for t in range(spec.seq_len):
                obs_buf[i, t] = cv2.resize(
                    win_imgs[t], (spec.obs_size, spec.obs_size), interpolation=cv2.INTER_AREA
                )
        else:
            obs_buf[i] = win_imgs
        act_buf[i] = win_acts
        if (i + 1) % max(1, spec.num_trajectories // 10) == 0:
            print(f"  pusht_expert {i+1}/{spec.num_trajectories} ({time.time()-t0:.1f}s)")

    # Normalize actions to [-1, 1] using the dataset action range. The zarr
    # actions are in pixel-coords (~[0, 512]); re-scale to match our env
    # convention so a model trained here is consistent with our pusht env.
    a_lo, a_hi = act_buf.min(), act_buf.max()
    print(f"  pusht_expert action range: [{a_lo:.1f}, {a_hi:.1f}] -> normalizing to [-1, 1]")
    a_mid = 0.5 * (a_lo + a_hi)
    a_half = 0.5 * (a_hi - a_lo)
    act_buf = ((act_buf - a_mid) / max(a_half, 1e-6)).astype(np.float32)

    return {"obs": obs_buf, "actions": act_buf}


def make_dataset(
    env_name: str,
    num_trajectories: int = 256,
    seq_len: int = 16,
    obs_size: int | None = None,
    seed: int = 0,
    source: str | None = None,
) -> "TrajectoryDataset":
    """Load (or collect + cache) a trajectory dataset for the named env."""
    default_size = {"tworoom": 48, "reacher": 48, "pusht": 96, "cube": 64, "pusht_expert": 96}
    obs_size = obs_size or default_size.get(env_name, 48)
    if source is None:
        if env_name == "cube":
            source = "cube_replay"
        elif env_name == "pusht_expert":
            source = "pusht_expert"
        else:
            source = "random"
    if env_name == "pusht_expert":
        # No real env exists for this name; use pusht's action dim (2).
        action_dim = _action_dim_for("pusht")
    else:
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
        elif source == "pusht_expert":
            blob = _collect_pusht_expert(spec)
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
