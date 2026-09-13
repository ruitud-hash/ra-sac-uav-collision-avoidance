"""Local action-safety filtering for static obstacles and boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.geometry import angle_normalize, heading_to_vector, point_to_segment_distance, swept_circle_static_clearance
from utils.risk import closest_approach_metrics


@dataclass(frozen=True)
class ActionSafetyConfig:
    enabled: bool = False
    static_boundary_enabled: bool = True
    static_margin_m: float = 50.0
    boundary_margin_m: float = 120.0
    dynamic_margin_m: float = 40.0
    dynamic_ttc_margin_s: float = 4.0
    dynamic_risk_threshold: float = 0.5
    omega_samples: int = 9
    lookahead_steps: int = 1
    dynamic_enabled: bool = False
    action_deviation_weight: float = 0.25
    clearance_weight: float = 0.03
    safety_violation_weight: float = 80.0


def filter_static_boundary_action(env, action: np.ndarray, config: ActionSafetyConfig) -> tuple[np.ndarray, bool]:
    """Return a locally safer action if the proposed one violates safety margins."""
    if not config.enabled:
        return action, False

    action = np.asarray(action, dtype=np.float64)
    action = np.clip(action, [-env.a_max, -env.omega_max], [env.a_max, env.omega_max])
    if _candidate_is_safe(env, action, config):
        return action, False

    best_action = action
    best_score = -float("inf")
    original_goal_distance = float(np.linalg.norm(env.goal - env.position))
    accel_candidates = _accel_candidates(env, action)
    omega_candidates = _omega_candidates(env, action, config.omega_samples)
    for accel in accel_candidates:
        for omega in omega_candidates:
            candidate = np.array([accel, omega], dtype=np.float64)
            score = _candidate_score(env, candidate, action, config, original_goal_distance)
            if score > best_score:
                best_score = score
                best_action = candidate

    return best_action, bool(np.linalg.norm(best_action - action) > 1e-9)


def _accel_candidates(env, action: np.ndarray) -> np.ndarray:
    values = np.array([action[0], min(action[0], 0.0), 0.0, -0.5 * env.a_max, -env.a_max], dtype=np.float64)
    return np.unique(np.clip(values, -env.a_max, env.a_max))


def _omega_candidates(env, action: np.ndarray, samples: int) -> np.ndarray:
    samples = max(int(samples), 3)
    values = np.concatenate(
        [
            np.array([action[1], 0.0], dtype=np.float64),
            np.linspace(-env.omega_max, env.omega_max, samples, dtype=np.float64),
        ]
    )
    return np.unique(np.clip(values, -env.omega_max, env.omega_max))


def _candidate_score(
    env,
    candidate: np.ndarray,
    original_action: np.ndarray,
    config: ActionSafetyConfig,
    original_goal_distance: float,
) -> float:
    rollout = _rollout_candidate(env, candidate, config)
    next_position = rollout.position
    static_clearance = rollout.static_clearance
    boundary_distance = rollout.boundary_distance
    dynamic_clearance = rollout.dynamic_clearance
    dynamic_ttc = rollout.dynamic_ttc
    dynamic_risk = rollout.dynamic_risk
    safe_static = (not config.static_boundary_enabled) or static_clearance >= config.static_margin_m
    safe_boundary = (not config.static_boundary_enabled) or boundary_distance >= config.boundary_margin_m
    safe_dynamic = (not config.dynamic_enabled) or (
        dynamic_clearance >= config.dynamic_margin_m
        and dynamic_ttc >= config.dynamic_ttc_margin_s
        and dynamic_risk <= config.dynamic_risk_threshold
    )
    safety_bonus = 1000.0 if safe_static and safe_boundary and safe_dynamic else 0.0
    progress = original_goal_distance - float(np.linalg.norm(env.goal - next_position))
    action_scale = np.array([max(env.a_max, 1e-6), max(env.omega_max, 1e-6)], dtype=np.float64)
    deviation = float(np.sum(((candidate - original_action) / action_scale) ** 2))
    clearance_terms = []
    if config.static_boundary_enabled:
        clearance_terms.extend(
            [static_clearance - config.static_margin_m, boundary_distance - config.boundary_margin_m]
        )
    violation = _safety_violation(rollout, config)
    if config.dynamic_enabled:
        clearance_terms.append(dynamic_clearance - config.dynamic_margin_m)
        clearance_terms.append(10.0 * (dynamic_ttc - config.dynamic_ttc_margin_s))
        clearance_terms.append(100.0 * (config.dynamic_risk_threshold - dynamic_risk))
    clearance_term = min(clearance_terms, default=0.0)
    return (
        safety_bonus
        + progress
        + config.clearance_weight * clearance_term
        - config.action_deviation_weight * deviation
        - config.safety_violation_weight * violation
    )


def _safety_violation(rollout: "_RolloutResult", config: ActionSafetyConfig) -> float:
    terms = []
    if config.static_boundary_enabled:
        terms.extend(
            [
                _normalized_shortfall(config.static_margin_m - rollout.static_clearance, config.static_margin_m),
                _normalized_shortfall(config.boundary_margin_m - rollout.boundary_distance, config.boundary_margin_m),
            ]
        )
    if config.dynamic_enabled:
        terms.append(_normalized_shortfall(config.dynamic_margin_m - rollout.dynamic_clearance, config.dynamic_margin_m))
        terms.append(_normalized_shortfall(config.dynamic_ttc_margin_s - rollout.dynamic_ttc, config.dynamic_ttc_margin_s))
        terms.append(
            _normalized_shortfall(
                rollout.dynamic_risk - config.dynamic_risk_threshold,
                config.dynamic_risk_threshold,
            )
        )
    return float(sum(term * term for term in terms))


def _normalized_shortfall(shortfall: float, reference: float) -> float:
    if shortfall <= 0.0:
        return 0.0
    return float(shortfall / max(reference, 1e-6))


def _candidate_is_safe(env, candidate: np.ndarray, config: ActionSafetyConfig) -> bool:
    rollout = _rollout_candidate(env, candidate, config)
    if config.static_boundary_enabled and (
        rollout.static_clearance < config.static_margin_m
        or rollout.boundary_distance < config.boundary_margin_m
    ):
        return False
    if config.dynamic_enabled:
        return (
            rollout.dynamic_clearance >= config.dynamic_margin_m
            and rollout.dynamic_ttc >= config.dynamic_ttc_margin_s
            and rollout.dynamic_risk <= config.dynamic_risk_threshold
        )
    return True


def _predict_candidate(env, candidate: np.ndarray) -> tuple[np.ndarray, float, float]:
    delayed_candidate = env.expected_delayed_actions(candidate, 1)[0]
    actual_candidate = env.expected_actual_action(delayed_candidate)
    speed = float(np.clip(env.speed + actual_candidate[0] * env.dt, env.v_min, env.v_max))
    heading = angle_normalize(env.heading + float(actual_candidate[1]) * env.dt)
    wind_velocity = env._wind_velocity_at(env.metrics.steps * env.dt)
    ground_velocity = heading_to_vector(heading) * speed + wind_velocity
    position = env.position + ground_velocity * env.dt
    return position, _static_clearance_at(env, position), _boundary_distance(env, position)


@dataclass(frozen=True)
class _RolloutResult:
    position: np.ndarray
    static_clearance: float
    boundary_distance: float
    dynamic_clearance: float
    dynamic_ttc: float
    dynamic_risk: float
    dynamic_positions: tuple[np.ndarray, ...] = ()
    dynamic_velocities: tuple[np.ndarray, ...] = ()


def _rollout_candidate(env, candidate: np.ndarray, config: ActionSafetyConfig) -> _RolloutResult:
    position = env.position.copy()
    speed = float(env.speed)
    heading = float(env.heading)
    min_static_clearance = float("inf")
    min_boundary_distance = float("inf")
    min_dynamic_clearance = float("inf")
    min_dynamic_ttc = float("inf")
    max_dynamic_risk = 0.0
    dynamic_positions = [other.position.copy() for other in env.dynamic_uavs]
    dynamic_velocities = [other.velocity.copy() for other in env.dynamic_uavs]
    dynamic_radii = [other.radius for other in env.dynamic_uavs]
    dynamic_maneuver_executed = [other.maneuver_executed for other in env.dynamic_uavs]
    start_step = int(env.metrics.steps)
    rollout_steps = max(int(config.lookahead_steps), 1)
    delayed_candidates = env.expected_delayed_actions(candidate, rollout_steps)
    for offset in range(rollout_steps):
        actual_candidate = env.expected_actual_action(delayed_candidates[offset])
        previous_position = position.copy()
        speed = float(np.clip(speed + actual_candidate[0] * env.dt, env.v_min, env.v_max))
        heading = angle_normalize(heading + float(actual_candidate[1]) * env.dt)
        wind_velocity = env._wind_velocity_at((start_step + offset) * env.dt)
        own_velocity = heading_to_vector(heading) * speed + wind_velocity
        position = position + own_velocity * env.dt
        if config.static_boundary_enabled:
            if env.static_obstacles:
                swept_static_clearance = min(
                    swept_circle_static_clearance(
                        previous_position,
                        position,
                        env.body_radius,
                        obstacle,
                    )
                    for obstacle in env.static_obstacles
                )
            else:
                swept_static_clearance = float("inf")
            min_static_clearance = min(min_static_clearance, swept_static_clearance)
            min_boundary_distance = min(min_boundary_distance, _boundary_distance(env, position))
        if config.dynamic_enabled:
            for idx, other_position in enumerate(dynamic_positions):
                previous_other_position = other_position.copy()
                other = env.dynamic_uavs[idx]
                dynamic_velocities[idx], dynamic_maneuver_executed[idx] = env.predict_dynamic_velocity(
                    other,
                    dynamic_velocities[idx],
                    start_step + offset,
                    dynamic_maneuver_executed[idx],
                    anticipate_sudden_maneuver=False,
                )
                other_position = other_position + dynamic_velocities[idx] * env.dt
                dynamic_positions[idx] = _reflect_dynamic_position(
                    env,
                    previous_other_position,
                    other_position,
                    dynamic_velocities[idx],
                    dynamic_radii[idx],
                )
                relative_start = previous_other_position - previous_position
                rel_pos = dynamic_positions[idx] - position
                rel_vel = dynamic_velocities[idx] - own_velocity
                combined_radius = dynamic_radii[idx] + env.body_radius
                clear = float(np.linalg.norm(rel_pos)) - combined_radius
                swept_clear = point_to_segment_distance(
                    np.zeros(2, dtype=np.float64),
                    relative_start,
                    rel_pos,
                ) - combined_radius
                ttc, tcpa, dcpa = closest_approach_metrics(rel_pos, rel_vel, combined_radius, env.ttc_max)
                if env.prediction_risk_enabled:
                    risk_level, sigma = env._prediction_risk(tcpa, dcpa, env.dynamic_uavs[idx])
                    conservative_dcpa = dcpa - env.risk_confidence_k * sigma
                    max_dynamic_risk = max(max_dynamic_risk, risk_level)
                else:
                    conservative_dcpa = dcpa
                min_dynamic_clearance = min(min_dynamic_clearance, clear, swept_clear, conservative_dcpa)
                min_dynamic_ttc = min(min_dynamic_ttc, ttc)
    return _RolloutResult(
        position,
        min_static_clearance,
        min_boundary_distance,
        min_dynamic_clearance,
        min_dynamic_ttc,
        max_dynamic_risk,
        tuple(position.copy() for position in dynamic_positions),
        tuple(velocity.copy() for velocity in dynamic_velocities),
    )


def _reflect_dynamic_position(
    env,
    previous_position: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    radius: float,
) -> np.ndarray:
    reflected = position.copy()
    for axis in (0, 1):
        if reflected[axis] < env.margin or reflected[axis] > env.world_size - env.margin:
            velocity[axis] *= -1.0
            reflected[axis] = np.clip(reflected[axis], env.margin, env.world_size - env.margin)
    if env._point_blocked_for_radius(reflected, radius):
        normal = env._nearest_static_normal(previous_position)
        if normal is None:
            velocity *= -1.0
        else:
            velocity[:] = velocity - 2.0 * float(np.dot(velocity, normal)) * normal
        reflected = previous_position.copy()
    return reflected


def _static_clearance_at(env, point: np.ndarray) -> float:
    if not env.static_obstacles:
        return float("inf")
    return min(env._static_clearance_at(point, obstacle)[1] for obstacle in env.static_obstacles)


def _boundary_distance(env, point: np.ndarray) -> float:
    distances = np.array(
        [
            point[0],
            env.world_size - point[0],
            point[1],
            env.world_size - point[1],
        ],
        dtype=np.float64,
    )
    return float(np.min(distances))
