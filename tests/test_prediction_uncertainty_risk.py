"""Tests for ground-speed and uncertainty-aware conservative risk."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np
import yaml

from agents.observation import attention_observation
from envs.uav_2d_env import DynamicUAV, UAV2DEnv
from utils.action_safety import ActionSafetyConfig, _rollout_candidate
from utils.risk import conservative_risk_score, prediction_position_uncertainty


ROOT = Path(__file__).resolve().parents[1]


def risk_config() -> dict:
    with (ROOT / "configs" / "env_sanity_open.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["obstacles"]["dynamic_count"] = 0
    config["obstacles"]["static_count"] = 0
    config["uav"]["disturbances"] = {
        "wind": {
            "enabled": True,
            "base_speed_mps": 2.0,
            "base_direction_deg": 0.0,
            "period_s": 60.0,
        },
        "actuation_error": {
            "enabled": True,
            "accel_std_mps2": 0.2,
            "omega_std_degps": 1.0,
        },
    }
    config["risk"] = {
        "prediction_uncertainty": {
            "enabled": True,
            "alpha_tcpa": 1.0,
            "beta_dcpa": 1.0,
            "gamma_sigma": 0.05,
            "base_position_std_m": 1.0,
            "intruder_velocity_std_mps": 0.5,
            "wind_velocity_std_mps": 0.4,
            "horizon_max_s": 20.0,
            "confidence_k": 1.96,
            "log_scale": 3.0,
        }
    }
    return config


def prepare_env(config: dict) -> UAV2DEnv:
    env = UAV2DEnv(config)
    env.reset(seed=31)
    env.position = np.array([500.0, 500.0], dtype=np.float64)
    env.goal = np.array([900.0, 500.0], dtype=np.float64)
    env.heading = 0.0
    env.speed = 8.0
    env.wind_velocity = env._wind_velocity_at(0.0)
    env.dynamic_uavs = [
        DynamicUAV(
            position=np.array([560.0, 500.0], dtype=np.float64),
            velocity=np.array([-4.0, 0.0], dtype=np.float64),
            radius=1.0,
        )
    ]
    env.previous_dynamic_positions = [env.dynamic_uavs[0].position.copy()]
    env.previous_goal_distance = env._goal_distance()
    env.metrics.trajectory = [env.position.copy()]
    return env


class PredictionUncertaintyRiskTests(unittest.TestCase):
    def test_risk_increases_with_prediction_uncertainty(self) -> None:
        low = conservative_risk_score(5.0, 20.0, 1.0, gamma=0.1)
        high = conservative_risk_score(5.0, 20.0, 5.0, gamma=0.1)
        self.assertGreater(high, low)
        self.assertGreater(prediction_position_uncertainty(10.0, 1.0), 9.9)

    def test_dynamic_token_uses_ground_relative_velocity_and_risk_channel(self) -> None:
        env = prepare_env(risk_config())

        observation = env._observation()
        token = observation["tokens"][0]

        self.assertEqual(env.token_dim, 11)
        np.testing.assert_allclose(token[2:4], [-14.0, 0.0], atol=1e-12)
        self.assertGreater(token[10], 0.0)
        packed = attention_observation(observation, env.world_size)
        self.assertTrue(np.all(np.isfinite(packed)))

    def test_legacy_environment_keeps_ten_dimensional_tokens(self) -> None:
        config = risk_config()
        del config["risk"]
        env = prepare_env(config)

        self.assertEqual(env.token_dim, 10)
        self.assertEqual(env._observation()["tokens"].shape[1], 10)

    def test_safety_rollout_tightens_dcpa_using_uncertainty(self) -> None:
        uncertain_env = prepare_env(risk_config())
        nominal_config = copy.deepcopy(risk_config())
        nominal_config["risk"]["prediction_uncertainty"]["enabled"] = False
        nominal_env = prepare_env(nominal_config)
        safety = ActionSafetyConfig(enabled=True, dynamic_enabled=True, lookahead_steps=3)

        uncertain = _rollout_candidate(uncertain_env, np.zeros(2), safety)
        nominal = _rollout_candidate(nominal_env, np.zeros(2), safety)

        self.assertLess(uncertain.dynamic_clearance, nominal.dynamic_clearance)
        self.assertGreater(uncertain.dynamic_risk, 0.0)


if __name__ == "__main__":
    unittest.main()
