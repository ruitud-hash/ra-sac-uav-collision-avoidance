"""Two-dimensional UAV collision-avoidance simulation environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import radians
from typing import Any

import numpy as np

from utils.geometry import (
    CircleObstacle,
    RectObstacle,
    angle_normalize,
    heading_to_vector,
    nearest_vector_to_circle,
    nearest_vector_to_rect,
    point_to_segment_distance,
    swept_circle_static_clearance,
)
from utils.risk import (
    closest_approach_metrics,
    conservative_risk_score,
    normalized_conservative_risk,
    prediction_position_uncertainty,
    risk_score,
)
from utils.experiment_protocol import observation_config


@dataclass
class DynamicUAV:
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    target: np.ndarray | None = None
    kind: str = "generic"
    motion_model: str = "linear_bounce"
    speed_min: float = 0.0
    speed_max: float = float("inf")
    heading_noise_std: float = 0.0
    speed_noise_std: float = 0.0
    noise_interval_steps: int = 1
    turn_rate: float = 0.0
    acceleration: float = 0.0
    maneuver_step: int = -1
    maneuver_heading_delta: float = 0.0
    maneuver_speed_scale: float = 1.0
    maneuver_executed: bool = False
    scenario_role: str = "background"
    velocity_uncertainty_std: float = 0.0


@dataclass
class EpisodeMetrics:
    steps: int = 0
    path_length_m: float = 0.0
    fhp_count: int = 0
    min_separation_m: float = float("inf")
    min_ttc_s: float = float("inf")
    min_dcpa_m: float = float("inf")
    min_goal_distance_m: float = float("inf")
    min_path_static_clearance_m: float = float("inf")
    max_prediction_sigma_m: float = 0.0
    max_conservative_risk: float = 0.0
    outcome: str = "running"
    trajectory: list[np.ndarray] = field(default_factory=list)
    dynamic_trajectories: list[list[np.ndarray]] = field(default_factory=list)
    fhp_events: list[np.ndarray] = field(default_factory=list)
    min_clearance_point: np.ndarray | None = None
    min_clearance_other_position: np.ndarray | None = None
    min_clearance_kind: str | None = None


class UAV2DEnv:
    """A compact environment for validating the paper's 2D problem setup."""

    token_dim = 10
    DYNAMIC_TYPE_ORDER = (
        "slow_uav",
        "fast_uav",
        "non_cooperative_aircraft",
        "maneuvering_intruder",
    )
    TERMINAL_OUTCOMES = (
        "out_of_bounds",
        "static_collision",
        "dynamic_collision",
        "success",
        "timeout",
    )

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rng = np.random.default_rng(config.get("seed", None))
        self.world_size = float(config["world"]["size_m"])
        self.margin = float(config["world"]["boundary_margin_m"])
        self.goal_radius = float(config["task"]["goal_radius_m"])
        self.max_steps = int(config["uav"]["max_steps"])
        self.dt = float(config["uav"]["dt_s"])
        self.body_radius = float(config["uav"].get("body_radius_m", 0.0))
        self.v_min = float(config["uav"]["v_min_mps"])
        self.v_max = float(config["uav"]["v_max_mps"])
        self.a_max = float(config["uav"]["a_max_mps2"])
        self.omega_max = radians(float(config["uav"]["omega_max_degps"]))
        disturbance_cfg = config["uav"].get("disturbances", {})
        wind_cfg = disturbance_cfg.get("wind", {})
        self.wind_enabled = bool(wind_cfg.get("enabled", False))
        self.wind_base_speed = float(wind_cfg.get("base_speed_mps", 0.0))
        self.wind_speed_variation = float(wind_cfg.get("speed_variation_mps", 0.0))
        self.wind_base_direction = radians(float(wind_cfg.get("base_direction_deg", 0.0)))
        self.wind_direction_variation = radians(float(wind_cfg.get("direction_variation_deg", 0.0)))
        self.wind_period = max(float(wind_cfg.get("period_s", 1.0)), self.dt)
        self.wind_phase = radians(float(wind_cfg.get("phase_deg", 0.0)))
        self.wind_sensitivity = float(wind_cfg.get("sensitivity", 1.0))

        actuation_cfg = disturbance_cfg.get("actuation_error", {})
        self.actuation_error_enabled = bool(actuation_cfg.get("enabled", False))
        self.accel_error_bias = float(actuation_cfg.get("accel_bias_mps2", 0.0))
        self.accel_error_std = float(actuation_cfg.get("accel_std_mps2", 0.0))
        self.omega_error_bias = radians(float(actuation_cfg.get("omega_bias_degps", 0.0)))
        self.omega_error_std = radians(float(actuation_cfg.get("omega_std_degps", 0.0)))
        self.sense_radius = float(config["perception"]["sense_radius_m"])
        self.max_tokens = int(config["perception"]["max_tokens"])
        self.ttc_max = float(config["perception"]["ttc_max_s"])
        self.dynamic_radius = float(config["obstacles"]["dynamic_protect_radius_m"])
        self.warning_radius = float(config["obstacles"]["dynamic_warning_radius_m"])
        self.static_safe_margin = float(config["obstacles"]["static_safe_margin_m"])
        self.dynamic_motion_model = str(config["obstacles"].get("dynamic_motion_model", "linear_bounce"))
        self.dynamic_noise_interval_steps = max(1, int(config["obstacles"].get("dynamic_noise_interval_steps", 5)))
        self.dynamic_heading_noise_std = radians(float(config["obstacles"].get("dynamic_heading_noise_std_deg", 0.0)))
        self.dynamic_speed_noise_std = float(config["obstacles"].get("dynamic_speed_noise_std_mps", 0.0))
        self.dynamic_speed_min = float(config["obstacles"]["dynamic_speed_min_mps"])
        self.dynamic_speed_max = float(config["obstacles"]["dynamic_speed_max_mps"])
        self.dynamic_crossing_opposite_prob = float(
            config["obstacles"].get("dynamic_crossing_opposite_prob", 0.7)
        )
        heterogeneous_cfg = config["obstacles"].get("heterogeneous_dynamic", {})
        self.heterogeneous_dynamic_enabled = bool(heterogeneous_cfg.get("enabled", False))
        self.dynamic_type_profiles = self._parse_dynamic_type_profiles(heterogeneous_cfg.get("types", {}))
        self.dynamic_type_names = list(self.dynamic_type_profiles)
        self.nonlinear_scenario_config = config["obstacles"].get("nonlinear_scenarios", {})
        self.nonlinear_scenarios_enabled = bool(self.nonlinear_scenario_config.get("enabled", False))
        self.domain_randomization_config = config.get("domain_randomization", {})
        self.domain_randomization_enabled = bool(self.domain_randomization_config.get("enabled", False))
        observation_cfg = observation_config(config)
        self.actuator_state_mode = observation_cfg["actuator_state_mode"]
        self.observation_max_delay_steps = observation_cfg["max_delay_steps"]
        self.control_delay_steps = int(disturbance_cfg.get("control_delay_steps", 0))
        self.perception_position_std = 0.0
        self.perception_speed_std = 0.0
        self.perception_heading_std = 0.0
        self.perception_intruder_position_std = 0.0
        self.perception_intruder_velocity_std = 0.0
        self.action_delay_queue: list[np.ndarray] = []
        self.applied_command = np.zeros(2, dtype=np.float64)
        self.domain_parameters: dict[str, Any] = {}
        self._base_domain_parameters = {
            "wind_enabled": self.wind_enabled,
            "wind_base_speed_mps": self.wind_base_speed,
            "wind_speed_variation_mps": self.wind_speed_variation,
            "wind_base_direction_rad": self.wind_base_direction,
            "wind_direction_variation_rad": self.wind_direction_variation,
            "wind_period_s": self.wind_period,
            "wind_phase_rad": self.wind_phase,
            "accel_error_std_mps2": self.accel_error_std,
            "omega_error_std_radps": self.omega_error_std,
            "actuation_error_enabled": self.actuation_error_enabled,
            "control_delay_steps": self.control_delay_steps,
            "dynamic_speed_min_mps": self.dynamic_speed_min,
            "dynamic_speed_max_mps": self.dynamic_speed_max,
            "dynamic_motion_model": self.dynamic_motion_model,
            "dynamic_heading_noise_std_rad": self.dynamic_heading_noise_std,
            "dynamic_speed_noise_std_mps": self.dynamic_speed_noise_std,
            "dynamic_noise_interval_steps": self.dynamic_noise_interval_steps,
            "dynamic_crossing_opposite_prob": self.dynamic_crossing_opposite_prob,
        }
        risk_cfg = config.get("risk", {}).get("prediction_uncertainty", {})
        self.prediction_risk_enabled = bool(risk_cfg.get("enabled", False))
        self.risk_alpha = float(risk_cfg.get("alpha_tcpa", 1.0))
        self.risk_beta = float(risk_cfg.get("beta_dcpa", 1.0))
        self.risk_gamma = float(risk_cfg.get("gamma_sigma", 1.0))
        self.risk_time_epsilon = float(risk_cfg.get("time_epsilon_s", 0.5))
        self.risk_distance_epsilon = float(risk_cfg.get("distance_epsilon_m", 5.0))
        self.risk_log_scale = float(risk_cfg.get("log_scale", 6.0))
        self.risk_confidence_k = float(risk_cfg.get("confidence_k", 1.96))
        self.risk_base_position_std = float(risk_cfg.get("base_position_std_m", 0.0))
        self.risk_intruder_velocity_std = float(
            risk_cfg.get("intruder_velocity_std_mps", self.dynamic_speed_noise_std)
        )
        self.risk_wind_velocity_std = float(risk_cfg.get("wind_velocity_std_mps", 0.0))
        self.risk_horizon_max = float(risk_cfg.get("horizon_max_s", self.ttc_max))
        self.has_risk_channel = self.prediction_risk_enabled or self.heterogeneous_dynamic_enabled
        self.token_dim = 10 + int(self.has_risk_channel) + (
            len(self.dynamic_type_names) if self.heterogeneous_dynamic_enabled else 0
        )
        self._validate_protocol_config()

        self.position = np.zeros(2, dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        self.speed = 0.0
        self.heading = 0.0
        self.previous_action = np.zeros(2, dtype=np.float64)
        self.actual_action = np.zeros(2, dtype=np.float64)
        self.wind_velocity = np.zeros(2, dtype=np.float64)
        self.previous_goal_distance = 0.0
        self.dynamic_uavs: list[DynamicUAV] = []
        self.previous_position = np.zeros(2, dtype=np.float64)
        self.previous_dynamic_positions: list[np.ndarray] = []
        self.static_obstacles: list[CircleObstacle | RectObstacle] = []
        self.metrics = EpisodeMetrics()

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._sample_episode_domain()
        self.static_obstacles = self._sample_static_obstacles()
        self.position, self.goal = self._sample_start_and_goal()
        self.heading = float(np.arctan2(*(self.goal - self.position)[::-1]))
        self.speed = float(self.config["uav"]["start_speed_mps"])
        self.previous_action = np.zeros(2, dtype=np.float64)
        self.applied_command = np.zeros(2, dtype=np.float64)
        self.actual_action = np.zeros(2, dtype=np.float64)
        self.action_delay_queue = [
            np.zeros(2, dtype=np.float64) for _ in range(self.control_delay_steps)
        ]
        self.wind_velocity = self._wind_velocity_at(0.0)
        self.dynamic_uavs = self._sample_dynamic_uavs()
        self.previous_position = self.position.copy()
        self.previous_dynamic_positions = [other.position.copy() for other in self.dynamic_uavs]
        self.previous_goal_distance = self._goal_distance()
        self.metrics = EpisodeMetrics(
            trajectory=[self.position.copy()],
            dynamic_trajectories=[
                [other.position.copy()] for other in self.dynamic_uavs
            ],
            min_goal_distance_m=self.previous_goal_distance,
        )
        self.metrics.min_path_static_clearance_m = self.straight_path_static_clearance(self.position, self.goal)
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, [-self.a_max, -self.omega_max], [self.a_max, self.omega_max])
        self.applied_command = self._apply_control_delay(action)
        self.actual_action = self._actual_action(self.applied_command)
        simulation_time = self.metrics.steps * self.dt
        self.wind_velocity = self._wind_velocity_at(simulation_time)

        previous_position = self.position.copy()
        self.previous_position = previous_position
        self.speed = float(np.clip(self.speed + self.actual_action[0] * self.dt, self.v_min, self.v_max))
        self.heading = angle_normalize(self.heading + float(self.actual_action[1]) * self.dt)
        self.position = self.position + self.ground_velocity() * self.dt
        self.previous_dynamic_positions = self._move_dynamic_uavs()

        self.metrics.steps += 1
        self.metrics.path_length_m += float(np.linalg.norm(self.position - previous_position))
        self.metrics.trajectory.append(self.position.copy())
        for index, other in enumerate(self.dynamic_uavs):
            if index < len(self.metrics.dynamic_trajectories):
                self.metrics.dynamic_trajectories[index].append(other.position.copy())
        self.metrics.min_goal_distance_m = min(self.metrics.min_goal_distance_m, self._goal_distance())

        reward, done, outcome = self._reward_and_done(action)
        self.previous_action = action.copy()
        self.previous_goal_distance = self._goal_distance()
        self.metrics.outcome = outcome
        return self._observation(), reward, done, self.info()

    def info(self) -> dict[str, Any]:
        return {
            "outcome": self.metrics.outcome,
            "steps": self.metrics.steps,
            "path_length_m": self.metrics.path_length_m,
            "fhp_count": self.metrics.fhp_count,
            "min_separation_m": self.metrics.min_separation_m,
            "min_ttc_s": self.metrics.min_ttc_s,
            "min_dcpa_m": self.metrics.min_dcpa_m,
            "goal_distance_m": self._goal_distance(),
            "min_goal_distance_m": self.metrics.min_goal_distance_m,
            "min_path_static_clearance_m": self.metrics.min_path_static_clearance_m,
            "max_prediction_sigma_m": self.metrics.max_prediction_sigma_m,
            "max_conservative_risk": self.metrics.max_conservative_risk,
            "commanded_action": self.previous_action.copy(),
            "applied_command": self.applied_command.copy(),
            "actual_action": self.actual_action.copy(),
            "actuator_state": self._actuator_state(normalized=False),
            "wind_velocity_mps": self.wind_velocity.copy(),
            "ground_velocity_mps": self.ground_velocity(),
            "domain_parameters": dict(self.domain_parameters),
            "episode_randomization": self.episode_randomization(),
            "dynamic_type_counts": self._dynamic_type_counts(),
            "dynamic_scenario_counts": self._dynamic_scenario_counts(),
            "start": self.metrics.trajectory[0] if self.metrics.trajectory else None,
            "goal": self.goal.copy(),
        }

    def episode_randomization(self) -> dict[str, Any]:
        """Return all sampled episode parameters needed for exact audit trails."""
        dynamic_realizations = []
        for index, other in enumerate(self.dynamic_uavs):
            dynamic_realizations.append(
                {
                    "index": index,
                    "kind": other.kind,
                    "scenario_role": other.scenario_role,
                    "motion_model": other.motion_model,
                    "protect_radius_m": other.radius,
                    "speed_min_mps": other.speed_min,
                    "speed_max_mps": other.speed_max,
                    "heading_noise_std_deg": float(np.degrees(other.heading_noise_std)),
                    "speed_noise_std_mps": other.speed_noise_std,
                    "noise_interval_steps": other.noise_interval_steps,
                    "turn_rate_degps": float(np.degrees(other.turn_rate)),
                    "acceleration_mps2": other.acceleration,
                    "maneuver_step": other.maneuver_step,
                    "maneuver_heading_delta_deg": float(
                        np.degrees(other.maneuver_heading_delta)
                    ),
                    "maneuver_speed_scale": other.maneuver_speed_scale,
                    "velocity_uncertainty_std_mps": other.velocity_uncertainty_std,
                }
            )
        return {
            "domain_parameters": dict(self.domain_parameters),
            "dynamic_realizations": dynamic_realizations,
            "dynamic_type_counts": self._dynamic_type_counts(),
            "dynamic_scenario_counts": self._dynamic_scenario_counts(),
        }

    def ground_velocity(self) -> np.ndarray:
        """Return ownship ground-relative velocity, including wind drift."""
        air_velocity = heading_to_vector(self.heading) * self.speed
        return air_velocity + self.wind_velocity

    def _actual_action(self, commanded_action: np.ndarray) -> np.ndarray:
        actual_action = self.expected_actual_action(commanded_action)
        if self.actuation_error_enabled and self.accel_error_std > 0.0:
            actual_action[0] += float(self.rng.normal(0.0, self.accel_error_std))
        if self.actuation_error_enabled and self.omega_error_std > 0.0:
            actual_action[1] += float(self.rng.normal(0.0, self.omega_error_std))
        return np.clip(
            actual_action,
            [-self.a_max, -self.omega_max],
            [self.a_max, self.omega_max],
        )

    def _apply_control_delay(self, commanded_action: np.ndarray) -> np.ndarray:
        commanded_action = np.asarray(commanded_action, dtype=np.float64)
        if self.control_delay_steps <= 0:
            return commanded_action.copy()
        self.action_delay_queue.append(commanded_action.copy())
        return self.action_delay_queue.pop(0)

    def _actuator_state(self, *, normalized: bool) -> dict[str, Any]:
        max_delay = self.observation_max_delay_steps
        if len(self.action_delay_queue) > max_delay:
            raise ValueError(
                f"control delay queue length {len(self.action_delay_queue)} exceeds "
                f"observation.max_delay_steps={max_delay}"
            )
        queue = np.zeros((max_delay, 2), dtype=np.float32)
        mask = np.zeros(max_delay, dtype=np.float32)
        if self.action_delay_queue:
            valid = len(self.action_delay_queue)
            queue[:valid] = np.asarray(self.action_delay_queue, dtype=np.float32)
            mask[:valid] = 1.0
        applied = self.applied_command.astype(np.float32, copy=True)
        if normalized:
            action_scale = np.array([self.a_max, self.omega_max], dtype=np.float32)
            applied /= action_scale
            queue /= action_scale
        return {
            "control_delay_steps": int(self.control_delay_steps),
            "normalized_control_delay": np.array(
                [self.control_delay_steps / max_delay], dtype=np.float32
            ),
            "current_applied_command": applied,
            "pending_action_queue": queue,
            "pending_queue_mask": mask,
        }

    def expected_delayed_actions(self, candidate: np.ndarray, steps: int) -> list[np.ndarray]:
        """Predict applied commands when a candidate is held over a rollout."""
        candidate = np.asarray(candidate, dtype=np.float64)
        queue = [queued.copy() for queued in self.action_delay_queue]
        delayed_actions: list[np.ndarray] = []
        for _ in range(max(int(steps), 1)):
            if self.control_delay_steps <= 0:
                delayed_actions.append(candidate.copy())
                continue
            queue.append(candidate.copy())
            delayed_actions.append(queue.pop(0))
        return delayed_actions

    def expected_actual_action(self, commanded_action: np.ndarray) -> np.ndarray:
        """Return the deterministic actuation model used for safety prediction."""
        commanded_action = np.asarray(commanded_action, dtype=np.float64)
        if not self.actuation_error_enabled:
            return commanded_action.copy()
        bias = np.array([self.accel_error_bias, self.omega_error_bias], dtype=np.float64)
        return np.clip(
            commanded_action + bias,
            [-self.a_max, -self.omega_max],
            [self.a_max, self.omega_max],
        )

    def _wind_velocity_at(self, time_s: float) -> np.ndarray:
        if not self.wind_enabled:
            return np.zeros(2, dtype=np.float64)

        phase = 2.0 * np.pi * time_s / self.wind_period + self.wind_phase
        speed = max(0.0, self.wind_base_speed + self.wind_speed_variation * np.sin(phase))
        direction = self.wind_base_direction + self.wind_direction_variation * np.sin(phase)
        return self.wind_sensitivity * heading_to_vector(direction) * speed

    def _sample_episode_domain(self) -> None:
        self._restore_base_domain_parameters()
        self.perception_position_std = 0.0
        self.perception_speed_std = 0.0
        self.perception_heading_std = 0.0
        self.perception_intruder_position_std = 0.0
        self.perception_intruder_velocity_std = 0.0

        if self.domain_randomization_enabled:
            cfg = self.domain_randomization_config
            wind_cfg = cfg.get("wind", {})
            self.wind_enabled = bool(wind_cfg.get("enabled", self.wind_enabled))
            self.wind_base_speed = self._sample_float(wind_cfg, "base_speed_mps", self.wind_base_speed)
            self.wind_speed_variation = self._sample_float(
                wind_cfg, "speed_variation_mps", self.wind_speed_variation
            )
            self.wind_base_direction = radians(
                self._sample_float(wind_cfg, "base_direction_deg", np.degrees(self.wind_base_direction))
            )
            self.wind_direction_variation = radians(
                self._sample_float(
                    wind_cfg,
                    "direction_variation_deg",
                    np.degrees(self.wind_direction_variation),
                )
            )
            self.wind_period = max(self._sample_float(wind_cfg, "period_s", self.wind_period), self.dt)
            self.wind_phase = radians(
                self._sample_float(wind_cfg, "phase_deg", np.degrees(self.wind_phase))
            )

            actuation_cfg = cfg.get("actuation_error", {})
            self.actuation_error_enabled = bool(
                actuation_cfg.get("enabled", self.actuation_error_enabled)
            )
            self.accel_error_std = self._sample_float(
                actuation_cfg, "accel_std_mps2", self.accel_error_std
            )
            self.omega_error_std = radians(
                self._sample_float(
                    actuation_cfg,
                    "omega_std_degps",
                    np.degrees(self.omega_error_std),
                )
            )

            perception_cfg = cfg.get("perception_noise", {})
            self.perception_position_std = self._sample_float(
                perception_cfg, "own_position_std_m", 0.0
            )
            self.perception_speed_std = self._sample_float(
                perception_cfg, "own_speed_std_mps", 0.0
            )
            self.perception_heading_std = radians(
                self._sample_float(perception_cfg, "own_heading_std_deg", 0.0)
            )
            self.perception_intruder_position_std = self._sample_float(
                perception_cfg, "intruder_position_std_m", 0.0
            )
            self.perception_intruder_velocity_std = self._sample_float(
                perception_cfg, "intruder_velocity_std_mps", 0.0
            )

            self.control_delay_steps = self._sample_int(
                cfg, "control_delay_steps", self.control_delay_steps
            )
            dynamic_cfg = cfg.get("dynamic_obstacles", {})
            self.dynamic_speed_min = self._sample_float(
                dynamic_cfg, "speed_min_mps", self.dynamic_speed_min
            )
            self.dynamic_speed_max = self._sample_float(
                dynamic_cfg, "speed_max_mps", self.dynamic_speed_max
            )
            if self.dynamic_speed_min > self.dynamic_speed_max:
                self.dynamic_speed_min, self.dynamic_speed_max = (
                    self.dynamic_speed_max,
                    self.dynamic_speed_min,
                )
            self.dynamic_heading_noise_std = radians(
                self._sample_float(
                    dynamic_cfg,
                    "heading_noise_std_deg",
                    np.degrees(self.dynamic_heading_noise_std),
                )
            )
            self.dynamic_speed_noise_std = self._sample_float(
                dynamic_cfg, "speed_noise_std_mps", self.dynamic_speed_noise_std
            )
            self.dynamic_noise_interval_steps = max(
                1,
                self._sample_int(
                    dynamic_cfg,
                    "noise_interval_steps",
                    self.dynamic_noise_interval_steps,
                ),
            )
            self.dynamic_crossing_opposite_prob = self._sample_float(
                dynamic_cfg,
                "crossing_opposite_prob",
                self.dynamic_crossing_opposite_prob,
            )
            motion_models = dynamic_cfg.get("motion_models", [])
            if motion_models:
                self.dynamic_motion_model = str(self.rng.choice(motion_models))

        self.domain_parameters = {
            "wind_enabled": self.wind_enabled,
            "wind_base_speed_mps": self.wind_base_speed,
            "wind_speed_variation_mps": self.wind_speed_variation,
            "wind_base_direction_deg": float(np.degrees(self.wind_base_direction)),
            "wind_direction_variation_deg": float(np.degrees(self.wind_direction_variation)),
            "wind_period_s": self.wind_period,
            "wind_phase_deg": float(np.degrees(self.wind_phase)),
            "accel_error_std_mps2": self.accel_error_std,
            "omega_error_std_degps": float(np.degrees(self.omega_error_std)),
            "actuation_error_enabled": self.actuation_error_enabled,
            "own_position_std_m": self.perception_position_std,
            "own_speed_std_mps": self.perception_speed_std,
            "own_heading_std_deg": float(np.degrees(self.perception_heading_std)),
            "intruder_position_std_m": self.perception_intruder_position_std,
            "intruder_velocity_std_mps": self.perception_intruder_velocity_std,
            "control_delay_steps": self.control_delay_steps,
            "dynamic_speed_min_mps": self.dynamic_speed_min,
            "dynamic_speed_max_mps": self.dynamic_speed_max,
            "dynamic_motion_model": self.dynamic_motion_model,
            "dynamic_heading_noise_std_deg": float(np.degrees(self.dynamic_heading_noise_std)),
            "dynamic_speed_noise_std_mps": self.dynamic_speed_noise_std,
            "dynamic_noise_interval_steps": self.dynamic_noise_interval_steps,
            "dynamic_crossing_opposite_prob": self.dynamic_crossing_opposite_prob,
        }

    def _restore_base_domain_parameters(self) -> None:
        base = self._base_domain_parameters
        self.wind_enabled = bool(base["wind_enabled"])
        self.wind_base_speed = float(base["wind_base_speed_mps"])
        self.wind_speed_variation = float(base["wind_speed_variation_mps"])
        self.wind_base_direction = float(base["wind_base_direction_rad"])
        self.wind_direction_variation = float(base["wind_direction_variation_rad"])
        self.wind_period = float(base["wind_period_s"])
        self.wind_phase = float(base["wind_phase_rad"])
        self.accel_error_std = float(base["accel_error_std_mps2"])
        self.omega_error_std = float(base["omega_error_std_radps"])
        self.actuation_error_enabled = bool(base["actuation_error_enabled"])
        self.control_delay_steps = int(base["control_delay_steps"])
        self.dynamic_speed_min = float(base["dynamic_speed_min_mps"])
        self.dynamic_speed_max = float(base["dynamic_speed_max_mps"])
        self.dynamic_motion_model = str(base["dynamic_motion_model"])
        self.dynamic_heading_noise_std = float(base["dynamic_heading_noise_std_rad"])
        self.dynamic_speed_noise_std = float(base["dynamic_speed_noise_std_mps"])
        self.dynamic_noise_interval_steps = int(base["dynamic_noise_interval_steps"])
        self.dynamic_crossing_opposite_prob = float(base["dynamic_crossing_opposite_prob"])

    def _sample_float(self, config: dict[str, Any], key: str, default: float) -> float:
        value = config.get(key, default)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(self.rng.uniform(float(value[0]), float(value[1])))
        return float(value)

    def _sample_int(self, config: dict[str, Any], key: str, default: int) -> int:
        value = config.get(key, default)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = sorted((int(value[0]), int(value[1])))
            return int(self.rng.integers(low, high + 1))
        return int(value)

    def sample_random_action(self) -> np.ndarray:
        return self.rng.uniform([-self.a_max, -self.omega_max], [self.a_max, self.omega_max])

    def sample_goal_seeking_action(self) -> np.ndarray:
        desired_heading = float(np.arctan2(*(self.goal - self.position)[::-1]))
        heading_error = angle_normalize(desired_heading - self.heading)
        omega = np.clip(heading_error / self.dt, -self.omega_max, self.omega_max)
        accel = 0.7 if self.speed < 0.85 * self.v_max else 0.0
        return np.array([accel, omega], dtype=np.float64)

    def _observation(self) -> dict[str, np.ndarray]:
        perceived_position = self.position.copy()
        perceived_speed = float(self.speed)
        perceived_heading = float(self.heading)
        if self.perception_position_std > 0.0:
            perceived_position += self.rng.normal(0.0, self.perception_position_std, size=2)
        if self.perception_speed_std > 0.0:
            perceived_speed = float(
                np.clip(
                    perceived_speed + self.rng.normal(0.0, self.perception_speed_std),
                    self.v_min,
                    self.v_max,
                )
            )
        if self.perception_heading_std > 0.0:
            perceived_heading = angle_normalize(
                perceived_heading + float(self.rng.normal(0.0, self.perception_heading_std))
            )

        goal_delta = self.goal - perceived_position
        goal_distance = float(np.linalg.norm(goal_delta))
        goal_bearing = angle_normalize(float(np.arctan2(goal_delta[1], goal_delta[0])) - perceived_heading)
        boundary = np.array(
            [
                perceived_position[0],
                self.world_size - perceived_position[0],
                perceived_position[1],
                self.world_size - perceived_position[1],
            ],
            dtype=np.float32,
        )
        perceived_own_velocity = heading_to_vector(perceived_heading) * perceived_speed + self.wind_velocity
        tokens, mask = self._tokens(perceived_position, perceived_own_velocity)
        observation = {
            "uav": np.array(
                [
                    perceived_position[0],
                    perceived_position[1],
                    perceived_speed,
                    perceived_heading,
                    self.previous_action[0],
                    self.previous_action[1],
                ],
                dtype=np.float32,
            ),
            "goal": np.array([goal_delta[0], goal_delta[1], goal_distance, goal_bearing], dtype=np.float32),
            "boundary": boundary,
            "tokens": tokens.astype(np.float32),
            "mask": mask,
        }
        if self.actuator_state_mode == "queue_observable":
            actuator_state = self._actuator_state(normalized=True)
            observation.update(
                {
                    key: actuator_state[key]
                    for key in (
                        "normalized_control_delay",
                        "current_applied_command",
                        "pending_action_queue",
                        "pending_queue_mask",
                    )
                }
            )
        return observation

    def _tokens(
        self,
        own_position: np.ndarray | None = None,
        own_velocity: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        own_position = self.position if own_position is None else own_position
        own_velocity = self.ground_velocity() if own_velocity is None else own_velocity
        token_rows: list[tuple[float, np.ndarray]] = []

        for other in self.dynamic_uavs:
            perceived_other_position = other.position.copy()
            perceived_other_velocity = other.velocity.copy()
            if self.perception_intruder_position_std > 0.0:
                perceived_other_position += self.rng.normal(
                    0.0, self.perception_intruder_position_std, size=2
                )
            if self.perception_intruder_velocity_std > 0.0:
                perceived_other_velocity += self.rng.normal(
                    0.0, self.perception_intruder_velocity_std, size=2
                )
            rel_pos = perceived_other_position - own_position
            distance = float(np.linalg.norm(rel_pos))
            if distance > self.sense_radius:
                continue
            rel_vel = perceived_other_velocity - own_velocity
            combined_radius = other.radius + self.body_radius
            clear = distance - combined_radius
            ttc, tcpa, dcpa = closest_approach_metrics(rel_pos, rel_vel, combined_radius, self.ttc_max)
            risk_level, _ = self._prediction_risk(tcpa, dcpa, other)
            score = risk_level if self.prediction_risk_enabled else risk_score(clear, ttc, dcpa)
            token_values = [
                rel_pos[0],
                rel_pos[1],
                rel_vel[0],
                rel_vel[1],
                clear,
                ttc,
                tcpa,
                dcpa,
                combined_radius,
                1.0,
            ]
            if self.has_risk_channel:
                token_values.append(risk_level)
            if self.heterogeneous_dynamic_enabled:
                token_values.extend(self._dynamic_type_encoding(other.kind))
            token = np.array(token_values, dtype=np.float64)
            token_rows.append((score, token))

        for obstacle in self.static_obstacles:
            if isinstance(obstacle, CircleObstacle):
                rel_near, clear = self._static_clearance_at_radius(own_position, obstacle, self.body_radius)
                size = obstacle.radius
            else:
                rel_near, clear = self._static_clearance_at_radius(own_position, obstacle, self.body_radius)
                size = max(obstacle.length, obstacle.width)
            if float(np.linalg.norm(rel_near)) > self.sense_radius and clear > self.sense_radius:
                continue
            rel_vel = -own_velocity
            ttc, tcpa, dcpa = closest_approach_metrics(rel_near, rel_vel, self.body_radius, self.ttc_max)
            risk_level = self._normalized_prediction_risk(tcpa, dcpa, 0.0)
            score = risk_level if self.prediction_risk_enabled else risk_score(clear, ttc, dcpa)
            token_values = [rel_near[0], rel_near[1], rel_vel[0], rel_vel[1], clear, ttc, tcpa, dcpa, size, 0.0]
            if self.has_risk_channel:
                token_values.append(risk_level)
            if self.heterogeneous_dynamic_enabled:
                token_values.extend([0.0] * len(self.dynamic_type_names))
            token = np.array(token_values, dtype=np.float64)
            token_rows.append((score, token))

        token_rows.sort(key=lambda item: item[0], reverse=True)
        tokens = np.zeros((self.max_tokens, self.token_dim), dtype=np.float64)
        mask = np.zeros(self.max_tokens, dtype=bool)
        for idx, (_, token) in enumerate(token_rows[: self.max_tokens]):
            tokens[idx] = token
            mask[idx] = True
        return tokens, mask

    def _relative_velocity_std(self, other: DynamicUAV | None = None) -> float:
        own_accel_std = self.accel_error_std * self.dt
        own_turn_std = abs(self.speed) * self.omega_error_std * self.dt
        own_velocity_std = float(
            np.sqrt(own_accel_std**2 + own_turn_std**2 + self.risk_wind_velocity_std**2)
        )
        intruder_velocity_std = self.risk_intruder_velocity_std if other is not None else 0.0
        if other is not None:
            intruder_velocity_std = max(intruder_velocity_std, other.velocity_uncertainty_std)
        heading_noise_std = other.heading_noise_std if other is not None else 0.0
        if other is not None and heading_noise_std > 0.0:
            intruder_speed = float(np.linalg.norm(other.velocity))
            intruder_turn_std = intruder_speed * heading_noise_std
            intruder_velocity_std = float(np.hypot(intruder_velocity_std, intruder_turn_std))
        return float(np.hypot(own_velocity_std, intruder_velocity_std))

    def _dynamic_type_encoding(self, kind: str) -> list[float]:
        return [1.0 if kind == type_name else 0.0 for type_name in self.dynamic_type_names]

    def _dynamic_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for other in self.dynamic_uavs:
            counts[other.kind] = counts.get(other.kind, 0) + 1
        return counts

    def _dynamic_scenario_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for other in self.dynamic_uavs:
            counts[other.scenario_role] = counts.get(other.scenario_role, 0) + 1
        return counts

    def _parse_dynamic_type_profiles(self, profiles: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if self.heterogeneous_dynamic_enabled:
            configured = set(profiles)
            expected = set(self.DYNAMIC_TYPE_ORDER)
            if configured != expected:
                missing = sorted(expected - configured)
                unexpected = sorted(configured - expected)
                raise ValueError(
                    "heterogeneous_dynamic types must match the canonical four-class set; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            profile_items = ((name, profiles[name]) for name in self.DYNAMIC_TYPE_ORDER)
        else:
            profile_items = profiles.items()

        parsed: dict[str, dict[str, Any]] = {}
        for name, values in profile_items:
            parsed[str(name)] = {
                "probability": float(values.get("probability", 1.0)),
                "speed_min_mps": float(values.get("speed_min_mps", self.dynamic_speed_min)),
                "speed_max_mps": float(values.get("speed_max_mps", self.dynamic_speed_max)),
                "protect_radius_m": float(values.get("protect_radius_m", self.dynamic_radius)),
                "motion_model": str(values.get("motion_model", self.dynamic_motion_model)),
                "heading_noise_std_deg": float(values.get("heading_noise_std_deg", 0.0)),
                "speed_noise_std_mps": float(values.get("speed_noise_std_mps", 0.0)),
                "noise_interval_steps": max(1, int(values.get("noise_interval_steps", 5))),
                "turn_rate_min_degps": float(values.get("turn_rate_min_degps", 0.0)),
                "turn_rate_max_degps": float(values.get("turn_rate_max_degps", 0.0)),
                "acceleration_min_mps2": float(values.get("acceleration_min_mps2", 0.0)),
                "acceleration_max_mps2": float(values.get("acceleration_max_mps2", 0.0)),
                "maneuver_step_min": int(values.get("maneuver_step_min", -1)),
                "maneuver_step_max": int(values.get("maneuver_step_max", -1)),
                "maneuver_heading_min_deg": float(values.get("maneuver_heading_min_deg", 0.0)),
                "maneuver_heading_max_deg": float(values.get("maneuver_heading_max_deg", 0.0)),
                "maneuver_speed_scale_min": float(values.get("maneuver_speed_scale_min", 1.0)),
                "maneuver_speed_scale_max": float(values.get("maneuver_speed_scale_max", 1.0)),
                "velocity_uncertainty_std_mps": float(values.get("velocity_uncertainty_std_mps", 0.0)),
            }
        return parsed

    def _validate_protocol_config(self) -> None:
        protocol = self.config.get("protocol", {})
        if not protocol:
            return

        protocol_name = str(protocol.get("name", ""))
        difficulty = str(protocol.get("difficulty", "")).lower()
        declared_token_dim = int(protocol.get("token_dim", self.token_dim))
        declared_max_tokens = int(protocol.get("max_tokens", self.max_tokens))
        declared_type_order = tuple(protocol.get("dynamic_type_order", self.dynamic_type_names))
        declared_outcomes = tuple(protocol.get("terminal_outcomes", self.TERMINAL_OUTCOMES))

        if protocol_name not in {"unified_15d_id", "unified_15d_ood"}:
            raise ValueError(
                f"unsupported environment protocol {protocol_name!r}"
            )
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(
                f"protocol difficulty must be easy, medium, or hard, got {difficulty!r}"
            )
        required_features = {
            "domain_randomization": self.domain_randomization_enabled,
            "heterogeneous_dynamic": self.heterogeneous_dynamic_enabled,
            "nonlinear_scenarios": self.nonlinear_scenarios_enabled,
            "prediction_uncertainty": self.prediction_risk_enabled,
        }
        disabled_features = [
            name for name, enabled in required_features.items() if not enabled
        ]
        if disabled_features:
            raise ValueError(
                "unified_15d_id requires all formal environment features; "
                f"disabled={disabled_features}"
            )
        if declared_token_dim != 15 or declared_max_tokens != 48:
            raise ValueError(
                "unified 15-D protocols require token_dim=15 and max_tokens=48, "
                f"got token_dim={declared_token_dim}, max_tokens={declared_max_tokens}"
            )
        obstacle_capacity = int(self.config["obstacles"]["dynamic_count"]) + int(
            self.config["obstacles"]["static_count"]
        )
        if self.max_tokens < obstacle_capacity:
            raise ValueError(
                "formal environment token capacity must cover every configured obstacle; "
                f"max_tokens={self.max_tokens}, obstacle_count={obstacle_capacity}"
            )
        if protocol_name == "unified_15d_ood":
            ood_axis = str(protocol.get("ood_axis", "")).lower()
            if str(protocol.get("split", "")).lower() != "ood":
                raise ValueError("unified_15d_ood requires protocol split='ood'")
            if ood_axis not in {"disturbance", "behavior", "composition", "combined"}:
                raise ValueError(f"unsupported OOD axis {ood_axis!r}")
            if protocol.get("evaluation_only") is not True:
                raise ValueError("OOD environments must declare evaluation_only=true")
            if protocol.get("checkpoint_selection_eligible") is not False:
                raise ValueError(
                    "OOD environments must declare checkpoint_selection_eligible=false"
                )
        if declared_token_dim != self.token_dim:
            raise ValueError(
                f"protocol token_dim={declared_token_dim} does not match environment token_dim={self.token_dim}"
            )
        if declared_max_tokens != self.max_tokens:
            raise ValueError(
                f"protocol max_tokens={declared_max_tokens} does not match perception max_tokens={self.max_tokens}"
            )
        if declared_type_order != self.DYNAMIC_TYPE_ORDER:
            raise ValueError(
                "protocol dynamic_type_order must be "
                f"{list(self.DYNAMIC_TYPE_ORDER)}, got {list(declared_type_order)}"
            )
        if tuple(self.dynamic_type_names) != self.DYNAMIC_TYPE_ORDER:
            raise ValueError(
                "environment dynamic type order must be "
                f"{list(self.DYNAMIC_TYPE_ORDER)}, got {self.dynamic_type_names}"
            )
        if declared_outcomes != self.TERMINAL_OUTCOMES:
            raise ValueError(
                "protocol terminal_outcomes must preserve the environment termination order "
                f"{list(self.TERMINAL_OUTCOMES)}, got {list(declared_outcomes)}"
            )

    def _prediction_risk(self, tcpa: float, dcpa: float, other: DynamicUAV | None) -> tuple[float, float]:
        relative_velocity_std = self._relative_velocity_std(other)
        sigma = prediction_position_uncertainty(
            tcpa,
            relative_velocity_std,
            self.risk_base_position_std,
            self.risk_horizon_max,
        )
        return self._normalized_prediction_risk(tcpa, dcpa, sigma), sigma

    def _normalized_prediction_risk(self, tcpa: float, dcpa: float, sigma: float) -> float:
        raw_risk = conservative_risk_score(
            tcpa,
            dcpa,
            sigma,
            alpha=self.risk_alpha,
            beta=self.risk_beta,
            gamma=self.risk_gamma,
            time_epsilon=self.risk_time_epsilon,
            distance_epsilon=self.risk_distance_epsilon,
        )
        return normalized_conservative_risk(raw_risk, self.risk_log_scale)

    def _reward_and_done(self, action: np.ndarray) -> tuple[float, bool, str]:
        reward_cfg = self.config["reward"]
        goal_distance = self._goal_distance()
        progress = self.previous_goal_distance - goal_distance
        reward = float(reward_cfg["progress_scale"]) * progress / max(self.previous_goal_distance, 1.0)
        reward += self._safe_progress_reward(progress)
        reward -= float(reward_cfg["step_penalty"])
        reward -= float(reward_cfg["smooth_penalty_scale"]) * float(np.sum((action - self.previous_action) ** 2))

        if self._out_of_bounds():
            return reward - float(reward_cfg["boundary_penalty"]), True, "out_of_bounds"

        if self._static_collision():
            return reward - float(reward_cfg["static_collision_penalty"]), True, "static_collision"

        dynamic_collision, warning_penalty = self._dynamic_risk_penalty()
        reward -= warning_penalty
        if dynamic_collision:
            return reward - float(reward_cfg["dynamic_collision_penalty"]), True, "dynamic_collision"

        static_near_penalty = self._static_near_penalty()
        reward -= static_near_penalty
        reward -= self._boundary_near_penalty()

        if goal_distance <= self.goal_radius:
            return reward + float(reward_cfg["goal_reward"]), True, "success"

        if self.metrics.steps >= self.max_steps:
            return reward - float(reward_cfg["timeout_penalty"]), True, "timeout"

        return reward, False, "running"

    def _safe_progress_reward(self, progress: float) -> float:
        safe_progress_cfg = self.config["reward"].get("safe_progress", {})
        if not safe_progress_cfg or not bool(safe_progress_cfg.get("enabled", False)):
            return 0.0

        reference = float(safe_progress_cfg.get("progress_reference_m", self.v_max * self.dt))
        scale = float(safe_progress_cfg.get("scale", 0.0))
        if reference <= 0.0 or scale <= 0.0:
            return 0.0

        if progress >= 0.0:
            weight = self._safe_progress_weight(safe_progress_cfg)
            return scale * weight * progress / reference

        negative_scale = float(safe_progress_cfg.get("negative_scale", 0.25))
        return scale * negative_scale * progress / reference

    def _safe_progress_weight(self, safe_progress_cfg: dict[str, Any]) -> float:
        clear_reference = float(safe_progress_cfg.get("clearance_reference_m", self.static_safe_margin))
        boundary_reference = float(safe_progress_cfg.get("boundary_reference_m", self.margin))
        ttc_reference = float(safe_progress_cfg.get("ttc_reference_s", self.ttc_max))

        weights = []
        if clear_reference > 0.0:
            weights.append(np.clip(self._min_static_clearance() / clear_reference, 0.0, 1.0))
            dynamic_clear = self._min_dynamic_clearance()
            if np.isfinite(dynamic_clear):
                weights.append(np.clip(dynamic_clear / clear_reference, 0.0, 1.0))

        if boundary_reference > 0.0:
            weights.append(np.clip(self._min_boundary_distance() / boundary_reference, 0.0, 1.0))

        if ttc_reference > 0.0:
            min_ttc = self._min_dynamic_ttc()
            if np.isfinite(min_ttc):
                weights.append(np.clip(min_ttc / ttc_reference, 0.0, 1.0))

        if not weights:
            return 1.0
        return float(np.min(weights))

    def _dynamic_risk_penalty(self) -> tuple[bool, float]:
        penalty = 0.0
        collision = False
        for idx, other in enumerate(self.dynamic_uavs):
            distance = float(np.linalg.norm(other.position - self.position))
            combined_radius = other.radius + self.body_radius
            clear = distance - combined_radius
            previous_other_position = (
                self.previous_dynamic_positions[idx]
                if idx < len(self.previous_dynamic_positions)
                else other.position
            )
            relative_start = previous_other_position - self.previous_position
            relative_end = other.position - self.position
            swept_clear = point_to_segment_distance(
                np.zeros(2, dtype=np.float64),
                relative_start,
                relative_end,
            ) - combined_radius
            candidate_clearance = min(clear, swept_clear)
            if candidate_clearance < self.metrics.min_separation_m:
                self.metrics.min_separation_m = candidate_clearance
                self.metrics.min_clearance_point = self.position.copy()
                self.metrics.min_clearance_other_position = other.position.copy()
                self.metrics.min_clearance_kind = other.kind
            rel_vel = other.velocity - self.ground_velocity()
            ttc, tcpa, dcpa = closest_approach_metrics(
                other.position - self.position,
                rel_vel,
                combined_radius,
                self.ttc_max,
            )
            risk_level, sigma = self._prediction_risk(tcpa, dcpa, other)
            if self.prediction_risk_enabled:
                self.metrics.max_prediction_sigma_m = max(self.metrics.max_prediction_sigma_m, sigma)
                self.metrics.max_conservative_risk = max(self.metrics.max_conservative_risk, risk_level)
            self.metrics.min_ttc_s = min(self.metrics.min_ttc_s, ttc)
            self.metrics.min_dcpa_m = min(self.metrics.min_dcpa_m, dcpa)
            if swept_clear <= 0.0:
                collision = True
            elif swept_clear < self.warning_radius - self.dynamic_radius:
                penalty += float(self.config["reward"]["warning_penalty"])
                self.metrics.fhp_count += 1
                self.metrics.fhp_events.append(self.position.copy())
        return collision, penalty

    def _static_near_penalty(self) -> float:
        penalty = 0.0
        for obstacle in self.static_obstacles:
            _, clear = self._static_clearance(obstacle)
            if clear < self.metrics.min_separation_m:
                self.metrics.min_separation_m = clear
                self.metrics.min_clearance_point = self.position.copy()
                self.metrics.min_clearance_other_position = None
                self.metrics.min_clearance_kind = "static"
            if 0.0 < clear < self.static_safe_margin:
                ratio = (self.static_safe_margin - clear) / self.static_safe_margin
                penalty += float(self.config["reward"]["static_near_penalty"]) * ratio * ratio
        return penalty

    def _boundary_near_penalty(self) -> float:
        reward_cfg = self.config["reward"]
        safe_margin = float(reward_cfg.get("boundary_safe_margin_m", 0.0))
        penalty_scale = float(reward_cfg.get("boundary_near_penalty", 0.0))
        if safe_margin <= 0.0 or penalty_scale <= 0.0:
            return 0.0
        distances = np.array(
            [
                self.position[0],
                self.world_size - self.position[0],
                self.position[1],
                self.world_size - self.position[1],
            ],
            dtype=np.float64,
        )
        min_distance = float(np.min(distances))
        if min_distance >= safe_margin:
            return 0.0
        ratio = (safe_margin - max(min_distance, 0.0)) / safe_margin
        return penalty_scale * ratio * ratio

    def _min_static_clearance(self) -> float:
        if not self.static_obstacles:
            return float("inf")
        return min(self._static_clearance(obstacle)[1] for obstacle in self.static_obstacles)

    def _min_dynamic_clearance(self) -> float:
        if not self.dynamic_uavs:
            return float("inf")
        own_position = self.position
        return min(
            float(np.linalg.norm(other.position - own_position)) - (other.radius + self.body_radius)
            for other in self.dynamic_uavs
        )

    def _min_dynamic_ttc(self) -> float:
        if not self.dynamic_uavs:
            return float("inf")
        own_velocity = self.ground_velocity()
        min_ttc = float("inf")
        for other in self.dynamic_uavs:
            combined_radius = other.radius + self.body_radius
            rel_pos = other.position - self.position
            rel_vel = other.velocity - own_velocity
            ttc, _, _ = closest_approach_metrics(rel_pos, rel_vel, combined_radius, self.ttc_max)
            min_ttc = min(min_ttc, ttc)
        return min_ttc

    def _min_boundary_distance(self) -> float:
        distances = np.array(
            [
                self.position[0],
                self.world_size - self.position[0],
                self.position[1],
                self.world_size - self.position[1],
            ],
            dtype=np.float64,
        )
        return float(np.min(distances))

    def _static_collision(self) -> bool:
        collision = False
        for obstacle in self.static_obstacles:
            swept_clearance = swept_circle_static_clearance(
                self.previous_position,
                self.position,
                self.body_radius,
                obstacle,
            )
            self.metrics.min_separation_m = min(self.metrics.min_separation_m, swept_clearance)
            collision = collision or swept_clearance <= 0.0
        return collision

    def _out_of_bounds(self) -> bool:
        return bool(np.any(self.position < 0.0) or np.any(self.position > self.world_size))

    def _move_dynamic_uavs(self) -> list[np.ndarray]:
        previous_positions: list[np.ndarray] = []
        for other in self.dynamic_uavs:
            if other.motion_model in {"noisy_linear", "non_cooperative", "maneuvering"}:
                self._perturb_dynamic_velocity(other)
            other.velocity, other.maneuver_executed = self.predict_dynamic_velocity(
                other,
                other.velocity,
                self.metrics.steps,
                other.maneuver_executed,
            )
            previous_position = other.position.copy()
            other.position = other.position + other.velocity * self.dt

            if other.motion_model in {"random_crossing", "non_cooperative"}:
                if self._dynamic_crossing_should_respawn(other):
                    self._respawn_crossing_dynamic_uav(other)
                    previous_positions.append(other.position.copy())
                elif self._point_blocked_for_radius(other.position, other.radius):
                    other.position = previous_position
                    self._respawn_crossing_dynamic_uav(other)
                    previous_positions.append(other.position.copy())
                else:
                    previous_positions.append(previous_position)
                continue

            for axis in (0, 1):
                if other.position[axis] < self.margin or other.position[axis] > self.world_size - self.margin:
                    other.velocity[axis] *= -1.0
                    other.position[axis] = np.clip(other.position[axis], self.margin, self.world_size - self.margin)
            if self._point_blocked_for_radius(other.position, other.radius):
                other.position = previous_position
                normal = self._nearest_static_normal(other.position)
                if normal is None:
                    other.velocity *= -1.0
                else:
                    other.velocity = other.velocity - 2.0 * float(np.dot(other.velocity, normal)) * normal
            previous_positions.append(previous_position)
        return previous_positions

    def predict_dynamic_velocity(
        self,
        other: DynamicUAV,
        velocity: np.ndarray,
        step: int,
        maneuver_executed: bool,
        anticipate_sudden_maneuver: bool = True,
    ) -> tuple[np.ndarray, bool]:
        """Deterministically advance an intruder velocity model by one step."""
        speed = float(np.linalg.norm(velocity))
        heading = float(np.arctan2(velocity[1], velocity[0])) if speed > 1e-9 else 0.0

        if other.motion_model in {"maneuvering", "constant_turn"} and abs(other.turn_rate) > 0.0:
            heading = angle_normalize(heading + other.turn_rate * self.dt)

        if other.motion_model in {"accelerating", "accelerating_turn"}:
            speed = float(np.clip(speed + other.acceleration * self.dt, other.speed_min, other.speed_max))
            if other.motion_model == "accelerating_turn" and abs(other.turn_rate) > 0.0:
                heading = angle_normalize(heading + other.turn_rate * self.dt)

        if (
            other.motion_model == "sudden_maneuver"
            and anticipate_sudden_maneuver
            and not maneuver_executed
            and other.maneuver_step >= 0
            and step >= other.maneuver_step
        ):
            heading = angle_normalize(heading + other.maneuver_heading_delta)
            speed = float(
                np.clip(
                    speed * other.maneuver_speed_scale,
                    other.speed_min,
                    other.speed_max,
                )
            )
            maneuver_executed = True

        return heading_to_vector(heading) * speed, maneuver_executed

    def _perturb_dynamic_velocity(self, other: DynamicUAV) -> None:
        if self.metrics.steps % max(other.noise_interval_steps, 1) != 0:
            return

        speed_min = other.speed_min
        speed_max = other.speed_max
        speed = float(np.linalg.norm(other.velocity))
        heading = float(np.arctan2(other.velocity[1], other.velocity[0]))
        if other.heading_noise_std > 0.0:
            heading = angle_normalize(heading + float(self.rng.normal(0.0, other.heading_noise_std)))
        if other.speed_noise_std > 0.0:
            speed = float(np.clip(speed + self.rng.normal(0.0, other.speed_noise_std), speed_min, speed_max))
        other.velocity = heading_to_vector(heading) * speed

    def _dynamic_crossing_should_respawn(self, other: DynamicUAV) -> bool:
        if np.any(other.position < -self.margin) or np.any(other.position > self.world_size + self.margin):
            return True
        if other.target is not None and float(np.dot(other.target - other.position, other.velocity)) < 0.0:
            return True
        return False

    def _respawn_crossing_dynamic_uav(self, other: DynamicUAV) -> None:
        position, velocity, target = self._sample_crossing_dynamic_state(
            speed_min=other.speed_min,
            speed_max=other.speed_max,
            radius=other.radius,
        )
        other.position = position
        other.velocity = velocity
        other.target = target

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def _sample_point(self) -> np.ndarray:
        return self.rng.uniform(self.margin, self.world_size - self.margin, size=2)

    def _sample_start_and_goal(self) -> tuple[np.ndarray, np.ndarray]:
        min_dist = float(self.config["task"]["min_start_goal_distance_m"])
        max_dist = float(self.config["task"]["max_start_goal_distance_m"])
        require_direct_path = bool(self.config["task"].get("require_direct_path", False))
        for _ in range(5000):
            start = self._sample_point()
            goal = self._sample_point()
            distance = float(np.linalg.norm(goal - start))
            path_is_clear = not require_direct_path or self.straight_path_static_clearance(start, goal) >= float(
                self.config["task"].get("direct_path_clearance_m", 0.0)
            )
            if min_dist <= distance <= max_dist and not self._point_blocked(start) and not self._point_blocked(goal) and path_is_clear:
                return start, goal
        raise RuntimeError("Failed to sample a valid start-goal pair.")

    def _point_blocked(self, point: np.ndarray) -> bool:
        return self._point_blocked_for_radius(point, self.body_radius)

    def _point_blocked_for_radius(self, point: np.ndarray, radius: float) -> bool:
        return any(self._static_clearance_at_radius(point, obstacle, radius)[1] <= 0.0 for obstacle in self.static_obstacles)

    def _sample_dynamic_uavs(self) -> list[DynamicUAV]:
        count = int(self.config["obstacles"]["dynamic_count"])
        dynamic_uavs = self._sample_scripted_dynamic_uavs(count)
        for _ in range(len(dynamic_uavs), count):
            kind, profile = self._sample_dynamic_profile()
            speed_min = float(profile["speed_min_mps"])
            speed_max = float(profile["speed_max_mps"])
            radius = float(profile["protect_radius_m"])
            motion_model = str(profile["motion_model"])
            if motion_model in {"random_crossing", "non_cooperative"}:
                position, velocity, target = self._sample_crossing_dynamic_state(
                    dynamic_uavs,
                    speed_min=speed_min,
                    speed_max=speed_max,
                    radius=radius,
                )
                dynamic_uavs.append(
                    self._make_dynamic_uav(kind, profile, position, velocity, target)
                )
                continue
            position = self._sample_free_dynamic_position(dynamic_uavs, radius=radius)
            heading = self.rng.uniform(-np.pi, np.pi)
            speed = self.rng.uniform(speed_min, speed_max)
            velocity = heading_to_vector(float(heading)) * float(speed)
            dynamic_uavs.append(self._make_dynamic_uav(kind, profile, position, velocity))
        return dynamic_uavs

    def _sample_scripted_dynamic_uavs(self, limit: int) -> list[DynamicUAV]:
        if not self.nonlinear_scenarios_enabled or limit <= 0:
            return []
        scenarios = self.nonlinear_scenario_config.get("scenarios", {})
        dynamic_uavs: list[DynamicUAV] = []
        for role in (
            "head_on",
            "crossing",
            "constant_turn",
            "accelerating",
            "sudden_maneuver",
            "coordinated_pincer",
        ):
            scenario = scenarios.get(role, {})
            count = min(max(int(scenario.get("count", 0)), 0), limit - len(dynamic_uavs))
            for index in range(count):
                created = self._make_scripted_intruder(role, index, count, scenario, dynamic_uavs)
                if created is not None:
                    dynamic_uavs.append(created)
            if len(dynamic_uavs) >= limit:
                break
        return dynamic_uavs

    def _make_scripted_intruder(
        self,
        role: str,
        index: int,
        count: int,
        scenario: dict[str, Any],
        existing: list[DynamicUAV],
    ) -> DynamicUAV | None:
        type_name = str(scenario.get("type", self.dynamic_type_names[0] if self.dynamic_type_names else "generic"))
        if type_name in self.dynamic_type_profiles:
            profile = dict(self.dynamic_type_profiles[type_name])
        else:
            type_name, profile = self._sample_dynamic_profile()
            profile = dict(profile)
        profile["motion_model"] = {
            "constant_turn": "constant_turn",
            "accelerating": "accelerating",
            "sudden_maneuver": "sudden_maneuver",
        }.get(role, profile["motion_model"])

        route = self.goal - self.position
        route_distance = max(float(np.linalg.norm(route)), 1e-6)
        route_direction = route / route_distance
        perpendicular = np.array([-route_direction[1], route_direction[0]], dtype=np.float64)
        encounter_fraction = float(scenario.get("encounter_fraction", 0.45))
        encounter = self.position + np.clip(encounter_fraction, 0.15, 0.85) * route
        offset = float(scenario.get("spawn_offset_m", 700.0))
        radius = float(profile["protect_radius_m"])
        speed = float(
            self.rng.uniform(float(profile["speed_min_mps"]), float(profile["speed_max_mps"]))
        )

        if role == "head_on":
            position = encounter + route_direction * offset * (1.0 + 0.15 * index)
        elif role in {"crossing", "sudden_maneuver"}:
            side = -1.0 if index % 2 == 0 else 1.0
            position = encounter + side * perpendicular * offset
        elif role == "coordinated_pincer":
            angle = 2.0 * np.pi * index / max(count, 1) + np.pi / 4.0
            direction = np.cos(angle) * route_direction + np.sin(angle) * perpendicular
            position = encounter + direction * offset
        else:
            angle = self.rng.uniform(-np.pi, np.pi)
            position = encounter + heading_to_vector(float(angle)) * offset

        position = np.clip(position, self.margin, self.world_size - self.margin)
        if self._point_blocked_for_radius(position, radius) or any(
            np.linalg.norm(position - other.position) < other.radius + radius + self.warning_radius
            for other in existing
        ):
            position = self._sample_free_dynamic_position(existing, radius=radius)
        direction = encounter - position
        direction_norm = max(float(np.linalg.norm(direction)), 1e-6)
        velocity = direction / direction_norm * speed
        intruder = self._make_dynamic_uav(type_name, profile, position, velocity, encounter)
        intruder.scenario_role = role
        if role == "head_on":
            intruder.motion_model = "linear_bounce"
        elif role == "crossing":
            intruder.motion_model = "linear_bounce"
        elif role == "coordinated_pincer":
            intruder.motion_model = str(scenario.get("motion_model", "accelerating"))
            intruder.acceleration = float(scenario.get("acceleration_mps2", intruder.acceleration))
        return intruder

    def _sample_dynamic_profile(self) -> tuple[str, dict[str, Any]]:
        if not self.heterogeneous_dynamic_enabled:
            return "generic", {
                "speed_min_mps": self.dynamic_speed_min,
                "speed_max_mps": self.dynamic_speed_max,
                "protect_radius_m": self.dynamic_radius,
                "motion_model": self.dynamic_motion_model,
                "heading_noise_std_deg": float(np.degrees(self.dynamic_heading_noise_std)),
                "speed_noise_std_mps": self.dynamic_speed_noise_std,
                "noise_interval_steps": self.dynamic_noise_interval_steps,
                "turn_rate_min_degps": 0.0,
                "turn_rate_max_degps": 0.0,
                "acceleration_min_mps2": 0.0,
                "acceleration_max_mps2": 0.0,
                "maneuver_step_min": -1,
                "maneuver_step_max": -1,
                "maneuver_heading_min_deg": 0.0,
                "maneuver_heading_max_deg": 0.0,
                "maneuver_speed_scale_min": 1.0,
                "maneuver_speed_scale_max": 1.0,
                "velocity_uncertainty_std_mps": self.risk_intruder_velocity_std,
            }
        weights = np.array(
            [max(self.dynamic_type_profiles[name]["probability"], 0.0) for name in self.dynamic_type_names],
            dtype=np.float64,
        )
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(weights)
        weights /= weights.sum()
        kind = str(self.rng.choice(self.dynamic_type_names, p=weights))
        return kind, self.dynamic_type_profiles[kind]

    def _make_dynamic_uav(
        self,
        kind: str,
        profile: dict[str, Any],
        position: np.ndarray,
        velocity: np.ndarray,
        target: np.ndarray | None = None,
    ) -> DynamicUAV:
        turn_min = radians(float(profile["turn_rate_min_degps"]))
        turn_max = radians(float(profile["turn_rate_max_degps"]))
        if turn_min > turn_max:
            turn_min, turn_max = turn_max, turn_min
        turn_rate = float(self.rng.uniform(turn_min, turn_max))
        if abs(turn_rate) > 0.0 and self.rng.random() < 0.5:
            turn_rate *= -1.0
        acceleration_min = float(profile["acceleration_min_mps2"])
        acceleration_max = float(profile["acceleration_max_mps2"])
        if acceleration_min > acceleration_max:
            acceleration_min, acceleration_max = acceleration_max, acceleration_min
        acceleration = float(self.rng.uniform(acceleration_min, acceleration_max))
        maneuver_step_min = int(profile["maneuver_step_min"])
        maneuver_step_max = int(profile["maneuver_step_max"])
        if maneuver_step_min > maneuver_step_max:
            maneuver_step_min, maneuver_step_max = maneuver_step_max, maneuver_step_min
        maneuver_step = (
            int(self.rng.integers(maneuver_step_min, maneuver_step_max + 1))
            if maneuver_step_min >= 0
            else -1
        )
        maneuver_heading_min = radians(float(profile["maneuver_heading_min_deg"]))
        maneuver_heading_max = radians(float(profile["maneuver_heading_max_deg"]))
        if maneuver_heading_min > maneuver_heading_max:
            maneuver_heading_min, maneuver_heading_max = maneuver_heading_max, maneuver_heading_min
        maneuver_heading_delta = float(self.rng.uniform(maneuver_heading_min, maneuver_heading_max))
        if abs(maneuver_heading_delta) > 0.0 and self.rng.random() < 0.5:
            maneuver_heading_delta *= -1.0
        speed_scale_min = float(profile["maneuver_speed_scale_min"])
        speed_scale_max = float(profile["maneuver_speed_scale_max"])
        if speed_scale_min > speed_scale_max:
            speed_scale_min, speed_scale_max = speed_scale_max, speed_scale_min
        return DynamicUAV(
            position=position,
            velocity=velocity,
            radius=float(profile["protect_radius_m"]),
            target=target,
            kind=kind,
            motion_model=str(profile["motion_model"]),
            speed_min=float(profile["speed_min_mps"]),
            speed_max=float(profile["speed_max_mps"]),
            heading_noise_std=radians(float(profile["heading_noise_std_deg"])),
            speed_noise_std=float(profile["speed_noise_std_mps"]),
            noise_interval_steps=max(1, int(profile["noise_interval_steps"])),
            turn_rate=turn_rate,
            acceleration=acceleration,
            maneuver_step=maneuver_step,
            maneuver_heading_delta=maneuver_heading_delta,
            maneuver_speed_scale=float(self.rng.uniform(speed_scale_min, speed_scale_max)),
            velocity_uncertainty_std=float(profile["velocity_uncertainty_std_mps"]),
        )

    def _sample_crossing_dynamic_state(
        self,
        existing: list[DynamicUAV] | None = None,
        speed_min: float | None = None,
        speed_max: float | None = None,
        radius: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        existing = existing or []
        speed_min = self.dynamic_speed_min if speed_min is None else float(speed_min)
        speed_max = self.dynamic_speed_max if speed_max is None else float(speed_max)
        radius = self.dynamic_radius if radius is None else float(radius)
        edge_offset = float(self.config["obstacles"].get("dynamic_crossing_edge_offset_m", self.margin * 0.5))
        for _ in range(3000):
            edge = int(self.rng.integers(0, 4))
            position = self._sample_crossing_edge_point(edge, edge_offset)
            target = self._sample_crossing_target(edge)
            if self._point_blocked_for_radius(position, radius):
                continue
            if any(np.linalg.norm(position - other.position) < 2.0 * self.warning_radius for other in existing):
                continue
            direction = target - position
            distance = float(np.linalg.norm(direction))
            if distance <= 1e-6:
                continue
            speed = float(self.rng.uniform(speed_min, speed_max))
            velocity = direction / distance * speed
            return position, velocity, target
        return self._sample_fallback_crossing_dynamic_state(speed_min, radius)

    def _sample_crossing_edge_point(self, edge: int, edge_offset: float) -> np.ndarray:
        coord = float(self.rng.uniform(self.margin, self.world_size - self.margin))
        if edge == 0:
            return np.array([self.margin + edge_offset, coord], dtype=np.float64)
        if edge == 1:
            return np.array([self.world_size - self.margin - edge_offset, coord], dtype=np.float64)
        if edge == 2:
            return np.array([coord, self.margin + edge_offset], dtype=np.float64)
        return np.array([coord, self.world_size - self.margin - edge_offset], dtype=np.float64)

    def _sample_crossing_target(self, entry_edge: int) -> np.ndarray:
        opposite_edges = {0: 1, 1: 0, 2: 3, 3: 2}
        if self.rng.random() < self.dynamic_crossing_opposite_prob:
            target_edge = opposite_edges[entry_edge]
        else:
            choices = [edge for edge in range(4) if edge != entry_edge]
            target_edge = int(self.rng.choice(choices))
        return self._sample_crossing_edge_point(target_edge, -self.margin * 0.5)

    def _sample_fallback_crossing_dynamic_state(
        self,
        speed_min: float | None = None,
        radius: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        speed_min = self.dynamic_speed_min if speed_min is None else float(speed_min)
        radius = self.dynamic_radius if radius is None else float(radius)
        position = self._sample_free_dynamic_position([], radius=radius)
        target = self._sample_point()
        direction = target - position
        distance = max(float(np.linalg.norm(direction)), 1e-6)
        return position, direction / distance * speed_min, target

    def _sample_static_obstacles(self) -> list[CircleObstacle | RectObstacle]:
        count = int(self.config["obstacles"]["static_count"])
        obstacles: list[CircleObstacle | RectObstacle] = []
        attempts = 0
        idx = 0
        while len(obstacles) < count and attempts < count * 300:
            attempts += 1
            center = self._sample_point()
            if idx % 4 == 0:
                radius = float(self.rng.uniform(80.0, 180.0)) + self.static_safe_margin
                candidate: CircleObstacle | RectObstacle = CircleObstacle(
                    center=center,
                    radius=radius,
                    kind="no_fly_zone",
                )
            else:
                length = float(self.rng.uniform(160.0, 450.0)) + 2.0 * self.static_safe_margin
                width = float(self.rng.uniform(120.0, 360.0)) + 2.0 * self.static_safe_margin
                candidate = RectObstacle(center=center, length=length, width=width)
            if not self._static_obstacle_overlaps(candidate, obstacles):
                obstacles.append(candidate)
                idx += 1
        return obstacles

    def _sample_free_dynamic_position(
        self,
        existing: list[DynamicUAV],
        radius: float | None = None,
    ) -> np.ndarray:
        radius = self.dynamic_radius if radius is None else float(radius)
        for _ in range(3000):
            position = self._sample_point()
            if self._point_blocked_for_radius(position, radius):
                continue
            if np.linalg.norm(position - self.position) < 2.0 * self.warning_radius:
                continue
            if any(np.linalg.norm(position - other.position) < 2.0 * self.warning_radius for other in existing):
                continue
            return position
        for _ in range(3000):
            position = self._sample_point()
            if not self._point_blocked_for_radius(position, radius):
                return position
        raise RuntimeError("Failed to sample a valid dynamic UAV position.")

    def _static_clearance(self, obstacle: CircleObstacle | RectObstacle) -> tuple[np.ndarray, float]:
        return self._static_clearance_at(self.position, obstacle)

    def _static_clearance_at(
        self,
        point: np.ndarray,
        obstacle: CircleObstacle | RectObstacle,
    ) -> tuple[np.ndarray, float]:
        return self._static_clearance_at_radius(point, obstacle, self.body_radius)

    @staticmethod
    def _static_clearance_at_radius(
        point: np.ndarray,
        obstacle: CircleObstacle | RectObstacle,
        radius: float,
    ) -> tuple[np.ndarray, float]:
        if isinstance(obstacle, CircleObstacle):
            rel_near, boundary_clear = nearest_vector_to_circle(point, obstacle)
        else:
            rel_near, boundary_clear = nearest_vector_to_rect(point, obstacle)
        return rel_near, boundary_clear - radius

    def _nearest_static_normal(self, point: np.ndarray) -> np.ndarray | None:
        if not self.static_obstacles:
            return None
        nearest_vector = None
        nearest_clearance = float("inf")
        for obstacle in self.static_obstacles:
            rel_near, clearance = self._static_clearance_at_radius(point, obstacle, 0.0)
            if clearance < nearest_clearance:
                nearest_vector = rel_near
                nearest_clearance = clearance
        if nearest_vector is None:
            return None
        norm = float(np.linalg.norm(nearest_vector))
        if norm < 1e-9:
            return None
        return -nearest_vector / norm

    def straight_path_static_clearance(self, start: np.ndarray, goal: np.ndarray) -> float:
        samples = int(self.config["task"].get("direct_path_samples", 64))
        samples = max(samples, 2)
        min_clearance = float("inf")
        for alpha in np.linspace(0.0, 1.0, samples):
            point = (1.0 - alpha) * start + alpha * goal
            if self.static_obstacles:
                point_clearance = min(self._static_clearance_at(point, obstacle)[1] for obstacle in self.static_obstacles)
            else:
                point_clearance = float("inf")
            min_clearance = min(min_clearance, point_clearance)
        return min_clearance

    def _static_obstacle_overlaps(
        self,
        candidate: CircleObstacle | RectObstacle,
        existing: list[CircleObstacle | RectObstacle],
    ) -> bool:
        candidate_radius = self._obstacle_bounding_radius(candidate)
        for obstacle in existing:
            if self._no_fly_static_overlap_allowed(candidate, obstacle):
                continue
            center_distance = float(np.linalg.norm(candidate.center - obstacle.center))
            min_distance = candidate_radius + self._obstacle_bounding_radius(obstacle) + self.static_safe_margin
            if center_distance < min_distance:
                return True
        return False

    @staticmethod
    def _no_fly_static_overlap_allowed(
        first: CircleObstacle | RectObstacle,
        second: CircleObstacle | RectObstacle,
    ) -> bool:
        first_is_no_fly = isinstance(first, CircleObstacle) and first.kind == "no_fly_zone"
        second_is_no_fly = isinstance(second, CircleObstacle) and second.kind == "no_fly_zone"
        first_is_static_rect = isinstance(first, RectObstacle)
        second_is_static_rect = isinstance(second, RectObstacle)
        return (first_is_no_fly and second_is_static_rect) or (second_is_no_fly and first_is_static_rect)

    @staticmethod
    def _obstacle_bounding_radius(obstacle: CircleObstacle | RectObstacle) -> float:
        if isinstance(obstacle, CircleObstacle):
            return obstacle.radius
        return float(np.hypot(obstacle.length, obstacle.width) * 0.5)
