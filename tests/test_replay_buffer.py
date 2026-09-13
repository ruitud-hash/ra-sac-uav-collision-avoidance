"""Replay buffer storage regression tests."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import torch

from agents.replay_buffer import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_overwrite_order_matches_ring_buffer_rule(self) -> None:
        replay = ReplayBuffer(obs_dim=1, act_dim=1, capacity=3)
        for index in range(5):
            value = np.array([index], dtype=np.float32)
            replay.add(value, value, float(index), value + 1.0, done=False)

        self.assertEqual(replay.ptr, 2)
        self.assertEqual(replay.size, 3)
        np.testing.assert_array_equal(replay.obs[:, 0], np.array([3.0, 4.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(replay.rewards[:, 0], np.array([3.0, 4.0, 2.0], dtype=np.float32))

    def test_memmap_and_memory_sampling_match_with_same_indices(self) -> None:
        directory = Path("outputs") / f"test_replay_buffer_memmap_match_{uuid.uuid4().hex[:8]}"
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        memmap_replay = None
        try:
            memory_replay = ReplayBuffer(obs_dim=3, act_dim=2, capacity=5, memmap_threshold_bytes=10**9)
            memmap_replay = ReplayBuffer(
                obs_dim=3,
                act_dim=2,
                capacity=5,
                storage_dir=directory,
                memmap_threshold_bytes=1,
            )

            for index in range(5):
                obs = np.array([index, index + 1, index + 2], dtype=np.float32)
                action = np.array([index * 0.1, index * 0.2], dtype=np.float32)
                for replay in (memory_replay, memmap_replay):
                    replay.add(obs, action, float(index), obs + 1.0, done=index % 2 == 0)

            np.random.seed(1234)
            memory_batch = memory_replay.sample(batch_size=4, device=torch.device("cpu"))
            np.random.seed(1234)
            memmap_batch = memmap_replay.sample(batch_size=4, device=torch.device("cpu"))

            for key in ("obs", "actions", "rewards", "next_obs", "dones"):
                torch.testing.assert_close(memory_batch[key], memmap_batch[key])
        finally:
            if memmap_replay is not None:
                memmap_replay.close()
            shutil.rmtree(directory, ignore_errors=True)

    def test_memmap_storage_adds_and_samples(self) -> None:
        directory = Path("outputs") / f"test_replay_buffer_memmap_{uuid.uuid4().hex[:8]}"
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        try:
            replay = ReplayBuffer(
                obs_dim=3,
                act_dim=2,
                capacity=4,
                storage_dir=directory,
                memmap_threshold_bytes=1,
            )
            self.assertTrue(replay.uses_memmap)

            for index in range(4):
                obs = np.array([index, index + 1, index + 2], dtype=np.float32)
                action = np.array([index * 0.1, index * 0.2], dtype=np.float32)
                replay.add(obs, action, float(index), obs + 1.0, done=index % 2 == 0)

            batch = replay.sample(batch_size=2, device=torch.device("cpu"))
            self.assertEqual(batch["obs"].shape, (2, 3))
            self.assertEqual(batch["actions"].shape, (2, 2))
            self.assertEqual(batch["rewards"].shape, (2, 1))
            self.assertEqual(batch["next_obs"].shape, (2, 3))
            self.assertEqual(batch["dones"].shape, (2, 1))
            replay.close()
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
