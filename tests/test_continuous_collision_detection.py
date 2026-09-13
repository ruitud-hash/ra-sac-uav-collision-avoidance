"""Tests for swept collision detection over one simulation step."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from envs.uav_2d_env import DynamicUAV, UAV2DEnv
from utils.geometry import CircleObstacle, RectObstacle


ROOT = Path(__file__).resolve().parents[1]


def base_config() -> dict:
    with (ROOT / "configs" / "env_sanity_open.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["obstacles"]["dynamic_count"] = 0
    config["obstacles"]["static_count"] = 0
    config["uav"]["body_radius_m"] = 1.0
    config["uav"]["start_speed_mps"] = 15.0
    return config


def prepare_env() -> UAV2DEnv:
    env = UAV2DEnv(base_config())
    env.reset(seed=5)
    env.position = np.array([500.0, 500.0], dtype=np.float64)
    env.previous_position = env.position.copy()
    env.goal = np.array([900.0, 500.0], dtype=np.float64)
    env.heading = 0.0
    env.speed = 15.0
    env.previous_goal_distance = env._goal_distance()
    env.metrics.trajectory = [env.position.copy()]
    return env


class ContinuousCollisionDetectionTests(unittest.TestCase):
    def test_detects_circle_crossed_between_safe_endpoints(self) -> None:
        env = prepare_env()
        env.static_obstacles = [
            CircleObstacle(center=np.array([507.5, 500.0]), radius=1.0),
        ]

        _, _, done, info = env.step(np.zeros(2))

        self.assertTrue(done)
        self.assertEqual(info["outcome"], "static_collision")
        self.assertGreater(np.linalg.norm(env.previous_position - env.static_obstacles[0].center), 2.0)
        self.assertGreater(np.linalg.norm(env.position - env.static_obstacles[0].center), 2.0)

    def test_detects_rectangle_crossed_between_safe_endpoints(self) -> None:
        env = prepare_env()
        env.static_obstacles = [
            RectObstacle(center=np.array([507.5, 500.0]), length=1.0, width=4.0),
        ]

        _, _, done, info = env.step(np.zeros(2))

        self.assertTrue(done)
        self.assertEqual(info["outcome"], "static_collision")

    def test_rectangle_corner_outside_swept_radius_is_not_false_positive(self) -> None:
        env = prepare_env()
        env.static_obstacles = [
            RectObstacle(center=np.array([507.5, 502.0]), length=1.0, width=1.0),
        ]

        _, _, done, info = env.step(np.zeros(2))

        self.assertFalse(done)
        self.assertEqual(info["outcome"], "running")

    def test_detects_dynamic_uavs_that_swap_sides_in_one_step(self) -> None:
        env = prepare_env()
        env.dynamic_uavs = [
            DynamicUAV(
                position=np.array([515.0, 500.0], dtype=np.float64),
                velocity=np.array([-15.0, 0.0], dtype=np.float64),
                radius=1.0,
            )
        ]
        env.previous_dynamic_positions = [env.dynamic_uavs[0].position.copy()]

        _, _, done, info = env.step(np.zeros(2))

        self.assertTrue(done)
        self.assertEqual(info["outcome"], "dynamic_collision")
        self.assertLessEqual(info["min_separation_m"], 0.0)

    def test_parallel_motion_remains_collision_free(self) -> None:
        env = prepare_env()
        env.dynamic_uavs = [
            DynamicUAV(
                position=np.array([500.0, 510.0], dtype=np.float64),
                velocity=np.array([15.0, 0.0], dtype=np.float64),
                radius=1.0,
            )
        ]
        env.previous_dynamic_positions = [env.dynamic_uavs[0].position.copy()]

        _, _, done, info = env.step(np.zeros(2))

        self.assertFalse(done)
        self.assertEqual(info["outcome"], "running")
        self.assertAlmostEqual(info["min_separation_m"], 8.0)


if __name__ == "__main__":
    unittest.main()
