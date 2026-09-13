"""Consistency tests for disturbance-aware Safety Gate rollouts."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np
import yaml

from envs.uav_2d_env import UAV2DEnv
from utils.action_safety import ActionSafetyConfig, _predict_candidate, _rollout_candidate
from utils.geometry import CircleObstacle


ROOT = Path(__file__).resolve().parents[1]


def disturbed_config() -> dict:
    with (ROOT / "configs" / "env_sanity_open.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["obstacles"]["dynamic_count"] = 0
    config["obstacles"]["static_count"] = 0
    config["uav"]["disturbances"] = {
        "wind": {
            "enabled": True,
            "base_speed_mps": 2.0,
            "speed_variation_mps": 1.0,
            "base_direction_deg": 90.0,
            "direction_variation_deg": 15.0,
            "period_s": 12.0,
        },
        "actuation_error": {
            "enabled": True,
            "accel_bias_mps2": 0.25,
            "omega_bias_degps": 1.0,
            "accel_std_mps2": 0.0,
            "omega_std_degps": 0.0,
        },
    }
    return config


def prepare_env(config: dict) -> UAV2DEnv:
    env = UAV2DEnv(config)
    env.reset(seed=19)
    env.position = np.array([500.0, 500.0], dtype=np.float64)
    env.goal = np.array([900.0, 700.0], dtype=np.float64)
    env.heading = 0.0
    env.speed = 8.0
    env.previous_goal_distance = env._goal_distance()
    env.metrics.trajectory = [env.position.copy()]
    return env


class ActionSafetyDisturbanceTests(unittest.TestCase):
    def test_single_step_prediction_matches_environment_transition(self) -> None:
        config = disturbed_config()
        predicted_env = prepare_env(copy.deepcopy(config))
        actual_env = prepare_env(copy.deepcopy(config))
        candidate = np.array([0.5, np.deg2rad(2.0)], dtype=np.float64)

        predicted_position, _, _ = _predict_candidate(predicted_env, candidate)
        actual_env.step(candidate)

        np.testing.assert_allclose(predicted_position, actual_env.position, atol=1e-12)

    def test_multistep_rollout_matches_deterministic_environment(self) -> None:
        config = disturbed_config()
        predicted_env = prepare_env(copy.deepcopy(config))
        actual_env = prepare_env(copy.deepcopy(config))
        candidate = np.array([0.2, np.deg2rad(-1.5)], dtype=np.float64)
        safety = ActionSafetyConfig(enabled=True, lookahead_steps=4)

        rollout = _rollout_candidate(predicted_env, candidate, safety)
        for _ in range(safety.lookahead_steps):
            actual_env.step(candidate)

        np.testing.assert_allclose(rollout.position, actual_env.position, atol=1e-12)

    def test_safety_prediction_does_not_consume_random_noise(self) -> None:
        config = disturbed_config()
        config["uav"]["disturbances"]["actuation_error"]["accel_std_mps2"] = 0.1
        config["uav"]["disturbances"]["actuation_error"]["omega_std_degps"] = 0.5
        predicted_env = prepare_env(copy.deepcopy(config))
        untouched_env = prepare_env(copy.deepcopy(config))
        candidate = np.zeros(2, dtype=np.float64)

        _rollout_candidate(predicted_env, candidate, ActionSafetyConfig(enabled=True, lookahead_steps=3))
        predicted_env.step(candidate)
        untouched_env.step(candidate)

        np.testing.assert_allclose(predicted_env.actual_action, untouched_env.actual_action)

    def test_rollout_detects_static_obstacle_crossed_within_step(self) -> None:
        config = disturbed_config()
        env = prepare_env(config)
        env.body_radius = 1.0
        env.speed = 15.0
        env.static_obstacles = [
            CircleObstacle(center=np.array([507.5, 500.0]), radius=1.0),
        ]

        rollout = _rollout_candidate(
            env,
            np.zeros(2, dtype=np.float64),
            ActionSafetyConfig(enabled=True, lookahead_steps=1),
        )

        self.assertLessEqual(rollout.static_clearance, 0.0)


if __name__ == "__main__":
    unittest.main()
