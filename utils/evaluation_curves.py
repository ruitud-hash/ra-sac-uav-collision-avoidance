"""Aggregation helpers for paper training curves from periodic evaluation logs."""

from __future__ import annotations

from typing import Iterable

import numpy as np


RARE_EVENT_METRICS = {
    "eval_collision_rate",
    "eval_timeout_rate",
    "eval_safety_failure_rate",
}
ALLOWED_SMOOTH_WINDOWS = {1, 3, 5, 50, 100}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered moving average without shortening the curve."""
    values = np.asarray(values, dtype=np.float64)
    if window not in ALLOWED_SMOOTH_WINDOWS:
        raise ValueError(
            "moving-average window must be one of "
            f"{sorted(ALLOWED_SMOOTH_WINDOWS)}, got {window}"
        )
    if window == 1 or values.size <= 1:
        return values.copy()

    left_radius = (window - 1) // 2
    right_radius = window - left_radius
    smoothed = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index - left_radius)
        end = min(values.size, index + right_radius)
        smoothed[index] = float(np.mean(values[start:end]))
    return smoothed


def aggregate_seed_curves(
    rows_by_seed: Iterable[list[dict[str, float]]],
    metric: str,
    *,
    smooth_window: int,
) -> dict[str, np.ndarray]:
    """Align seed curves, smooth each seed, then compute mean and standard error."""
    return aggregate_seed_series(
        rows_by_seed,
        x_key="step",
        metric=metric,
        smooth_window=smooth_window,
    )


def aggregate_seed_series(
    rows_by_seed: Iterable[list[dict[str, float]]],
    *,
    x_key: str,
    metric: str,
    smooth_window: int,
) -> dict[str, np.ndarray]:
    """Align an arbitrary per-seed series and compute mean and standard error."""
    seed_rows = list(rows_by_seed)
    if not seed_rows:
        raise ValueError("at least one seed log is required")

    x_sets = [
        {float(row[x_key]) for row in rows if x_key in row and metric in row}
        for rows in seed_rows
    ]
    common_x = sorted(set.intersection(*x_sets))
    if not common_x:
        raise ValueError(f"no common {x_key} values found for metric {metric!r}")

    raw_seed_curves = []
    for rows in seed_rows:
        by_x = {
            float(row[x_key]): float(row[metric])
            for row in rows
            if x_key in row and metric in row
        }
        raw_seed_curves.append([by_x[x_value] for x_value in common_x])

    raw = np.asarray(raw_seed_curves, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError(f"non-finite value found in {metric!r}")
    smoothed = np.vstack([moving_average(seed_curve, smooth_window) for seed_curve in raw])
    seed_count = smoothed.shape[0]
    standard_error = (
        np.std(smoothed, axis=0, ddof=1) / np.sqrt(seed_count)
        if seed_count > 1
        else np.zeros(smoothed.shape[1], dtype=np.float64)
    )
    return {
        "steps": np.asarray(common_x, dtype=np.float64),
        "raw_mean": np.mean(raw, axis=0),
        "mean": np.mean(smoothed, axis=0),
        "standard_error": standard_error,
        "seed_count": np.asarray([seed_count], dtype=np.int64),
    }
