"""Tests for ownship wind and actuation disturbances."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np
import yaml

from envs.uav_2d_env import UAV2DEnv


ROOT = Path(__file__).resolve().parents[1]


def load_base_config() -> dict:
    with (ROOT / "configs" / "env_sanity_open.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["obstacles"]["dynamic_count"] = 0
    config["obstacles"]["static_count"] = 0
    return config


def prepare_env(config: dict, position: np.ndarray, heading: float, speed: float) -> UAV2DEnv:
    env = UAV2DEnv(config)
    env.reset(seed=7)
    env.position = position.astype(np.float64)
    env.goal = np.array([900.0, 500.0], dtype=np.float64)
    env.heading = heading
    env.speed = speed
    env.previous_goal_distance = env._goal_distance()
    env.metrics.trajectory = [env.position.copy()]
    return env


class UAVDisturbanceTests(unittest.TestCase):
    def test_missing_disturbance_config_preserves_nominal_transition(self) -> None:
        config = load_base_config()
        env = prepare_env(config, np.array([500.0, 500.0]), heading=0.0, speed=8.0)

        env.step(np.array([1.0, 0.0]))

        self.assertAlmostEqual(env.speed, 9.0)
        np.testing.assert_allclose(env.position, [509.0, 500.0])
        np.testing.assert_allclose(env.actual_action, [1.0, 0.0])
        np.testing.assert_allclose(env.wind_velocity, [0.0, 0.0])

    def test_constant_wind_changes_ground_displacement(self) -> None:
        config = load_base_config()
        config["uav"]["disturbances"] = {
            "wind": {
                "enabled": True,
                "base_speed_mps": 2.0,
                "base_direction_deg": 90.0,
                "period_s": 60.0,
            }
        }
        env = prepare_env(config, np.array([500.0, 500.0]), heading=0.0, speed=8.0)

        env.step(np.zeros(2))

        np.testing.assert_allclose(env.wind_velocity, [0.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(env.position, [508.0, 502.0], atol=1e-12)
        np.testing.assert_allclose(env.ground_velocity(), [8.0, 2.0], atol=1e-12)

    def test_actuation_bias_is_applied_before_state_transition(self) -> None:
        config = load_base_config()
        config["uav"]["disturbances"] = {
            "actuation_error": {
                "enabled": True,
                "accel_bias_mps2": 0.5,
                "omega_bias_degps": 5.0,
            }
        }
        env = prepare_env(config, np.array([500.0, 500.0]), heading=0.0, speed=8.0)

        env.step(np.zeros(2))

        self.assertAlmostEqual(env.actual_action[0], 0.5)
        self.assertAlmostEqual(env.actual_action[1], np.deg2rad(5.0))
        self.assertAlmostEqual(env.speed, 8.5)
        self.assertAlmostEqual(env.heading, np.deg2rad(5.0))

    def test_random_actuation_error_is_reproducible_after_seeded_reset(self) -> None:
        config = load_base_config()
        config["uav"]["disturbances"] = {
            "actuation_error": {
                "enabled": True,
                "accel_std_mps2": 0.1,
                "omega_std_degps": 0.5,
            }
        }
        first = prepare_env(copy.deepcopy(config), np.array([500.0, 500.0]), 0.0, 8.0)
        second = prepare_env(copy.deepcopy(config), np.array([500.0, 500.0]), 0.0, 8.0)

        first.step(np.zeros(2))
        second.step(np.zeros(2))

        np.testing.assert_allclose(first.actual_action, second.actual_action)
        np.testing.assert_allclose(first.position, second.position)


if __name__ == "__main__":
    unittest.main()
