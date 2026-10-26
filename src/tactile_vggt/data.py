"""Small memory-mapped cache and episode-safe action chunk dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def episode_split(episode_index, seed=43, val_fraction=0.1):
    episodes = np.unique(episode_index).astype(int)
    if len(episodes) < 2:
        raise ValueError("At least two original episodes are required for a held-out split")
    shuffled = np.random.default_rng(seed).permutation(episodes)
    n_val = min(len(episodes) - 1, max(1, int(round(len(episodes) * val_fraction))))
    return {"train": sorted(shuffled[n_val:].tolist()), "val": sorted(shuffled[:n_val].tolist())}


def training_statistics(state, action, tactile, episode_index, train_episodes):
    mask = np.isin(episode_index, train_episodes)
    if not mask.any():
        raise ValueError("No training frames")
    stats = {}
    for name, values in (("state", state), ("action", action)):
        train = np.asarray(values[mask], dtype=np.float64)
        stats[name + "_mean"] = train.mean(0).tolist()
        stats[name + "_std"] = np.maximum(train.std(0), 1e-3).tolist()
    positive = np.log1p(np.maximum(np.asarray(tactile[mask]), 0))
    stats["pressure_scale"] = float(max(np.quantile(positive, 0.99), 1e-3))
    stats["training_frames"] = int(mask.sum())
    return stats


def action_chunk(actions, episode_index, start, horizon):
    """Full-rate labels, repeat-padding only within this original episode."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    stop = start + 1
    while stop < min(start + horizon, len(actions)) and episode_index[stop] == episode_index[start]:
        stop += 1
    indices = np.minimum(np.arange(start, start + horizon), stop - 1)
    return np.asarray(actions[indices]).copy(), np.arange(start, start + horizon) >= stop


class CachedTactileDataset:
    """Torch-compatible dataset; cache arrays stay memory mapped and read-only.

    Observation stride affects sample selection only. Each action chunk retains
    the source dataset's complete command stream and episode boundaries.
    Recording FPS is wall time; simulated command spacing is recorded in the
    audited cache manifest (v2: one 0.01-second physics step per command).
    """
    def __init__(self, cache_dir, split="train", horizon=16, normalize=True):
        self.cache_dir = Path(cache_dir)
        self.metadata = json.loads((self.cache_dir / "manifest.json").read_text())
        if not self.metadata.get("complete"):
            raise ValueError("Cache is incomplete; resume preparation before training")
        self.stats = self.metadata["normalization"]
        self.horizon = int(horizon)
        self.normalize = bool(normalize)
        for name in ("points", "state", "action", "tactile", "episode_index", "frame_index", "obs_frame_indices"):
            setattr(self, name, np.load(self.cache_dir / f"{name}.npy", mmap_mode="r"))
        if split not in ("train", "val", "all"):
            raise ValueError("split must be train, val or all")
        self.observation_indices = np.arange(len(self.obs_frame_indices))
        if split != "all":
            selected = np.isin(self.episode_index[self.obs_frame_indices], self.metadata["split"][split])
            self.observation_indices = self.observation_indices[selected]
        if not len(self.observation_indices):
            raise ValueError(f"No cached observations in {split} split")
        self.indices = self.observation_indices

    def __len__(self):
        return len(self.observation_indices)

    def __getitem__(self, index):
        import torch
        cache_index = int(self.observation_indices[index])
        frame = int(self.obs_frame_indices[cache_index])
        points = np.array(self.points[cache_index], dtype=np.float32, copy=True)
        state = np.array(self.state[frame], dtype=np.float32, copy=True)
        tactile = np.array(self.tactile[frame], dtype=np.float32, copy=True)
        action, pad = action_chunk(self.action, self.episode_index, frame, self.horizon)
        if self.normalize:
            state = (state - np.asarray(self.stats["state_mean"], dtype=np.float32)) / np.asarray(self.stats["state_std"], dtype=np.float32)
            action = (action - np.asarray(self.stats["action_mean"], dtype=np.float32)) / np.asarray(self.stats["action_std"], dtype=np.float32)
            tactile = np.log1p(np.maximum(tactile, 0)) / self.stats["pressure_scale"]
            points[:, 6] = np.log1p(np.maximum(points[:, 6], 0)) / self.stats["pressure_scale"]
        return {
            "observation.points": torch.from_numpy(points),
            "observation.state": torch.from_numpy(state),
            "observation.tactile": torch.from_numpy(tactile.astype(np.float32)),
            "action": torch.from_numpy(action.astype(np.float32)),
            "action_is_pad": torch.from_numpy(pad),
            "episode_index": int(self.episode_index[frame]),
            "frame_index": int(self.frame_index[frame]),
        }
