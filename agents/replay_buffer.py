"""Replay buffer for off-policy reinforcement learning."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch


class ReplayBuffer:
    MEMMAP_THRESHOLD_BYTES = 4 * 1024**3

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        storage_dir: str | Path | None = None,
        memmap_threshold_bytes: int | None = None,
    ):
        self.capacity = int(capacity)
        threshold = self.MEMMAP_THRESHOLD_BYTES if memmap_threshold_bytes is None else int(memmap_threshold_bytes)
        estimated_bytes = self.capacity * (2 * int(obs_dim) + int(act_dim) + 2) * np.dtype(np.float32).itemsize
        self.uses_memmap = estimated_bytes > threshold
        self._temporary_storage_path: Path | None = None
        if self.uses_memmap:
            if storage_dir is None:
                storage_path = (
                    Path.cwd()
                    / "outputs"
                    / "replay_buffers"
                    / f"ra_sac_replay_{os.getpid()}_{uuid.uuid4().hex[:8]}"
                )
                storage_path.mkdir(parents=True, exist_ok=False)
                self._temporary_storage_path = storage_path
            else:
                storage_path = Path(storage_dir)
                storage_path.mkdir(parents=True, exist_ok=True)
            self.obs = self._array(storage_path / "obs.dat", (self.capacity, obs_dim))
            self.next_obs = self._array(storage_path / "next_obs.dat", (self.capacity, obs_dim))
            self.actions = self._array(storage_path / "actions.dat", (self.capacity, act_dim))
            self.rewards = self._array(storage_path / "rewards.dat", (self.capacity, 1))
            self.dones = self._array(storage_path / "dones.dat", (self.capacity, 1))
        else:
            self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
            self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
            self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
            self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
            self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    @staticmethod
    def _array(path: Path, shape: tuple[int, int]) -> np.memmap:
        return np.memmap(path, dtype=np.float32, mode="w+", shape=shape)

    def close(self) -> None:
        for name in ("obs", "next_obs", "actions", "rewards", "dones"):
            array = getattr(self, name, None)
            if isinstance(array, np.memmap):
                array.flush()
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        if self._temporary_storage_path is not None:
            shutil.rmtree(self._temporary_storage_path, ignore_errors=True)
            self._temporary_storage_path = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_obs": torch.as_tensor(self.next_obs[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }
