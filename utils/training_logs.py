"""Shared training and periodic-evaluation log helpers."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


TERMINAL_OUTCOMES = {"success", "static_collision", "dynamic_collision", "timeout", "out_of_bounds"}

EPISODE_CORE_FIELDS = [
    "step",
    "episode",
    "episode_reward",
    "episode_steps",
    "outcome",
    "success",
    "collision",
    "static_collision",
    "dynamic_collision",
    "timeout",
    "oob",
]

EVAL_FIELDNAMES = [
    "step",
    "eval_id",
    "eval_episodes",
    "eval_seed_start",
    "eval_seed_end",
    "eval_return_mean",
    "eval_return_std",
    "eval_success_rate",
    "eval_collision_rate",
    "eval_timeout_rate",
    "eval_oob_rate",
    "eval_safety_failure_rate",
    "eval_path_length_success_mean",
    "eval_fhp_count_mean",
    "eval_fhp_rate_mean",
    "eval_sif_mean",
    "eval_sir_mean",
    "eval_scm_mean",
]
ACTUATOR_AUDIT_FIELDS = (
    "control_delay_steps",
    "applied_command",
    "pending_action_queue",
    "pending_queue_mask",
)

DOMAIN_PARAMETER_FIELDS = {
    "wind_enabled",
    "wind_base_speed_mps",
    "wind_speed_variation_mps",
    "wind_base_direction_deg",
    "wind_direction_variation_deg",
    "wind_period_s",
    "wind_phase_deg",
    "accel_error_std_mps2",
    "omega_error_std_degps",
    "actuation_error_enabled",
    "own_position_std_m",
    "own_speed_std_mps",
    "own_heading_std_deg",
    "intruder_position_std_m",
    "intruder_velocity_std_mps",
    "control_delay_steps",
    "dynamic_speed_min_mps",
    "dynamic_speed_max_mps",
    "dynamic_motion_model",
    "dynamic_heading_noise_std_deg",
    "dynamic_speed_noise_std_mps",
    "dynamic_noise_interval_steps",
    "dynamic_crossing_opposite_prob",
}


def outcome_flags(outcome: str) -> dict[str, int]:
    if outcome not in TERMINAL_OUTCOMES:
        raise ValueError(f"Unexpected terminal outcome: {outcome!r}")
    return {
        "success": int(outcome == "success"),
        "collision": int(outcome in ("static_collision", "dynamic_collision")),
        "static_collision": int(outcome == "static_collision"),
        "dynamic_collision": int(outcome == "dynamic_collision"),
        "timeout": int(outcome == "timeout"),
        "oob": int(outcome == "out_of_bounds"),
    }


def make_episode_row(
    *,
    step: int | str,
    episode: int,
    episode_reward: float,
    episode_steps: int,
    outcome: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": step,
        "episode": episode,
        "episode_reward": float(episode_reward),
        "episode_steps": int(episode_steps),
        "outcome": outcome,
        **outcome_flags(outcome),
    }
    if extra:
        row.update(extra)
    validate_episode_row(row)
    return row


def validate_episode_row(row: dict[str, Any], *, require_step: bool = True) -> None:
    if require_step and row.get("step", "") == "":
        raise ValueError("Episode row has an empty step value")
    terminal_sum = (
        int(row["success"])
        + int(row["static_collision"])
        + int(row["dynamic_collision"])
        + int(row["timeout"])
        + int(row["oob"])
    )
    if terminal_sum != 1:
        raise ValueError(f"Episode terminal indicators must sum to 1, got {terminal_sum}: {row}")
    expected_collision = int(bool(int(row["static_collision"]) or int(row["dynamic_collision"])))
    if int(row["collision"]) != expected_collision:
        raise ValueError(f"collision must equal static_collision OR dynamic_collision: {row}")
    if "outcome" in row:
        expected_flags = outcome_flags(str(row["outcome"]))
        for key, expected_value in expected_flags.items():
            if int(row[key]) != expected_value:
                raise ValueError(f"{key} does not match outcome={row['outcome']!r}: {row}")


def validate_eval_row(row: dict[str, Any], *, sum_tolerance: float = 1e-6) -> float:
    finite_fields = (
        "eval_return_mean",
        "eval_return_std",
        "eval_success_rate",
        "eval_collision_rate",
        "eval_timeout_rate",
        "eval_oob_rate",
        "eval_safety_failure_rate",
        "eval_fhp_count_mean",
        "eval_fhp_rate_mean",
        "eval_sif_mean",
        "eval_sir_mean",
        "eval_scm_mean",
    )
    for key in finite_fields:
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite, got {value}: {row}")

    for key in (
        "eval_success_rate",
        "eval_collision_rate",
        "eval_timeout_rate",
        "eval_oob_rate",
        "eval_safety_failure_rate",
        "eval_fhp_rate_mean",
        "eval_sir_mean",
    ):
        value = float(row[key])
        if value < -sum_tolerance or value > 1.0 + sum_tolerance:
            raise ValueError(f"{key} must be in [0, 1], got {value}: {row}")
    if float(row["eval_scm_mean"]) < -sum_tolerance:
        raise ValueError(f"eval_scm_mean must be non-negative: {row}")

    terminal_sum = (
        float(row["eval_success_rate"])
        + float(row["eval_collision_rate"])
        + float(row["eval_timeout_rate"])
        + float(row["eval_oob_rate"])
    )
    if abs(terminal_sum - 1.0) > sum_tolerance:
        raise ValueError(f"Evaluation terminal rates must sum to 1, got {terminal_sum}: {row}")
    return terminal_sum


def summarize_eval_episodes(rows: list[dict[str, Any]], eval_id: int, eval_episodes: int) -> dict[str, Any]:
    total = int(eval_episodes)
    if total <= 0:
        raise ValueError(f"eval_episodes must be positive, got {eval_episodes}")
    if len(rows) != total:
        raise ValueError(
            f"Expected {total} completed evaluation episodes, got {len(rows)}"
        )
    returns = [float(row["total_reward"]) for row in rows]
    steps = [int(row["steps"]) for row in rows]
    fhp_counts = [int(row["fhp_count"]) for row in rows]
    success_rows = [row for row in rows if row["outcome"] == "success"]
    outcomes: dict[str, int] = {}
    for row in rows:
        outcome = str(row["outcome"])
        if outcome not in TERMINAL_OUTCOMES:
            raise ValueError(f"Unexpected evaluation outcome: {outcome!r}")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    collisions = outcomes.get("dynamic_collision", 0) + outcomes.get("static_collision", 0)
    return {
        "eval_id": int(eval_id),
        "eval_episodes": int(eval_episodes),
        "eval_return_mean": _mean(returns),
        "eval_return_std": _std(returns),
        "eval_success_rate": outcomes.get("success", 0) / total,
        "eval_collision_rate": collisions / total,
        "eval_timeout_rate": outcomes.get("timeout", 0) / total,
        "eval_oob_rate": outcomes.get("out_of_bounds", 0) / total,
        "eval_safety_failure_rate": (
            collisions + outcomes.get("out_of_bounds", 0)
        )
        / total,
        "eval_path_length_success_mean": _mean([float(row["path_length_m"]) for row in success_rows]),
        "eval_fhp_count_mean": _mean(fhp_counts),
        "eval_fhp_rate_mean": _mean([fhp / max(step_count, 1) for fhp, step_count in zip(fhp_counts, steps)]),
        "eval_sif_mean": _mean([float(row.get("shield_intervention_count", 0.0)) for row in rows]),
        "eval_sir_mean": _mean([float(row.get("shield_intervention_rate", 0.0)) for row in rows]),
        "eval_scm_mean": _weighted_shield_correction_mean(rows),
    }


def actuator_audit_fields(info: dict[str, Any]) -> dict[str, Any]:
    state = info["actuator_state"]
    return {
        "control_delay_steps": int(state["control_delay_steps"]),
        "applied_command": json.dumps(
            np.asarray(state["current_applied_command"]).tolist(), separators=(",", ":")
        ),
        "pending_action_queue": json.dumps(
            np.asarray(state["pending_action_queue"]).tolist(), separators=(",", ":")
        ),
        "pending_queue_mask": json.dumps(
            np.asarray(state["pending_queue_mask"]).tolist(), separators=(",", ":")
        ),
    }


def evaluation_audit_fields(info: dict[str, Any]) -> dict[str, Any]:
    """Return stable per-episode audit fields for formal checkpoint evaluation."""
    snapshot = info["episode_randomization"]
    return {
        **actuator_audit_fields(info),
        "max_prediction_sigma_m": float(info.get("max_prediction_sigma_m", 0.0)),
        "max_conservative_risk": float(info.get("max_conservative_risk", 0.0)),
        "domain_parameters": json.dumps(
            info.get("domain_parameters", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "dynamic_type_counts": json.dumps(
            info.get("dynamic_type_counts", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "dynamic_scenario_counts": json.dumps(
            info.get("dynamic_scenario_counts", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "episode_randomization": serialize_episode_randomization(snapshot),
    }


def eval_selection_score(metrics: dict[str, Any], *, fhp_weight: float = 0.0) -> float:
    return (
        float(metrics["eval_success_rate"])
        - float(metrics["eval_collision_rate"])
        - 0.5 * float(metrics["eval_oob_rate"])
        - 0.25 * float(metrics["eval_timeout_rate"])
        - fhp_weight * float(metrics["eval_fhp_count_mean"])
    )


def eval_selection_key(
    metrics: dict[str, Any], *, step: int, fhp_weight: float = 0.0
) -> tuple[float, float, float, int]:
    """Frozen order: score, success, lower safety failure, earlier checkpoint."""
    return (
        eval_selection_score(metrics, fhp_weight=fhp_weight),
        float(metrics["eval_success_rate"]),
        -float(metrics["eval_safety_failure_rate"]),
        -int(step),
    )


def serialize_episode_randomization(snapshot: dict[str, Any]) -> str:
    """Validate and serialize a complete per-episode randomization snapshot."""
    domain_parameters = snapshot.get("domain_parameters", {})
    missing = sorted(DOMAIN_PARAMETER_FIELDS - set(domain_parameters))
    if missing:
        raise ValueError(f"Episode randomization is missing domain parameters: {missing}")
    for key in ("dynamic_realizations", "dynamic_type_counts", "dynamic_scenario_counts"):
        if key not in snapshot:
            raise ValueError(f"Episode randomization is missing {key!r}")
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"))


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _std(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.std(values, ddof=0))


def _weighted_shield_correction_mean(rows: list[dict[str, Any]]) -> float:
    correction_sum = sum(float(row.get("shield_correction_sum", 0.0)) for row in rows)
    intervention_count = sum(int(row.get("shield_intervention_count", 0)) for row in rows)
    if intervention_count <= 0:
        return 0.0
    value = correction_sum / intervention_count
    if math.isnan(value):
        return 0.0
    return float(value)
