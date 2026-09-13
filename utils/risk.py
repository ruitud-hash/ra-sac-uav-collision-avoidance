"""Risk metrics used by the 2D collision-avoidance environment."""

from __future__ import annotations

import numpy as np


def closest_approach_metrics(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    radius: float,
    ttc_max: float,
    eps: float = 1e-6,
) -> tuple[float, float, float]:
    """Return TTC, TCPA, and DCPA for a relative-motion pair."""
    distance = float(np.linalg.norm(relative_position))
    unit_rel = relative_position / (distance + eps)
    closing_speed = -float(unit_rel @ relative_velocity)

    clear = distance - radius
    if closing_speed > 0.0:
        ttc = max(0.0, clear) / (closing_speed + eps)
        ttc = min(float(ttc), ttc_max)
    else:
        ttc = float(ttc_max)

    speed_sq = float(relative_velocity @ relative_velocity)
    tcpa = max(0.0, -float(relative_position @ relative_velocity) / (speed_sq + eps))
    tcpa = min(float(tcpa), ttc_max)
    dcpa = float(np.linalg.norm(relative_position + relative_velocity * tcpa) - radius)
    return ttc, tcpa, dcpa


def risk_score(clearance: float, ttc: float, dcpa: float, eps: float = 1e-3) -> float:
    safe_clearance = max(clearance, 0.0)
    safe_dcpa = max(dcpa, 0.0)
    return float(1.0 / (safe_clearance + eps) + 2.0 / (ttc + eps) + 1.0 / (safe_dcpa + eps))


def prediction_position_uncertainty(
    tcpa: float,
    relative_velocity_std: float,
    base_position_std: float = 0.0,
    horizon_max: float | None = None,
) -> float:
    """Propagate relative-velocity uncertainty to the closest-approach horizon."""
    horizon = max(float(tcpa), 0.0)
    if horizon_max is not None:
        horizon = min(horizon, max(float(horizon_max), 0.0))
    return float(np.hypot(max(base_position_std, 0.0), max(relative_velocity_std, 0.0) * horizon))


def conservative_risk_score(
    tcpa: float,
    dcpa: float,
    prediction_sigma: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    time_epsilon: float = 0.5,
    distance_epsilon: float = 5.0,
) -> float:
    """Return alpha/TCPA + beta/DCPA + gamma*sigma with stable denominators."""
    safe_tcpa = max(float(tcpa), 0.0)
    safe_dcpa = max(float(dcpa), 0.0)
    safe_sigma = max(float(prediction_sigma), 0.0)
    return float(
        alpha / (safe_tcpa + max(time_epsilon, 1e-6))
        + beta / (safe_dcpa + max(distance_epsilon, 1e-6))
        + gamma * safe_sigma
    )


def normalized_conservative_risk(risk: float, log_scale: float = 6.0) -> float:
    """Map an unbounded conservative risk score to [0, 1]."""
    return float(np.clip(np.log1p(max(float(risk), 0.0)) / max(log_scale, 1e-6), 0.0, 1.0))
