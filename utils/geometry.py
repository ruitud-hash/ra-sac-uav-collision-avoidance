"""Geometry helpers for the 2D UAV delivery airspace."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np


@dataclass(frozen=True)
class CircleObstacle:
    center: np.ndarray
    radius: float
    kind: str = "circle"


@dataclass(frozen=True)
class RectObstacle:
    center: np.ndarray
    length: float
    width: float


def angle_normalize(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def heading_to_vector(heading: float) -> np.ndarray:
    return np.array([cos(heading), sin(heading)], dtype=np.float64)


def point_in_circle(point: np.ndarray, circle: CircleObstacle) -> bool:
    return float(np.linalg.norm(point - circle.center)) <= circle.radius


def point_in_rect(point: np.ndarray, rect: RectObstacle) -> bool:
    half = np.array([rect.length * 0.5, rect.width * 0.5], dtype=np.float64)
    return bool(np.all(np.abs(point - rect.center) <= half))


def nearest_vector_to_circle(point: np.ndarray, circle: CircleObstacle) -> tuple[np.ndarray, float]:
    delta = circle.center - point
    distance_to_center = float(np.linalg.norm(delta))
    if distance_to_center < 1e-9:
        return np.array([0.0, 0.0], dtype=np.float64), -circle.radius
    clear = distance_to_center - circle.radius
    nearest = delta * max(clear, 0.0) / distance_to_center
    return nearest, clear


def nearest_vector_to_rect(point: np.ndarray, rect: RectObstacle) -> tuple[np.ndarray, float]:
    half = np.array([rect.length * 0.5, rect.width * 0.5], dtype=np.float64)
    local = point - rect.center
    clamped = np.clip(local, -half, half)
    nearest_point = rect.center + clamped
    outward = nearest_point - point

    if point_in_rect(point, rect):
        penetration = half - np.abs(local)
        axis = int(np.argmin(penetration))
        clear = -float(penetration[axis])
        direction = np.array([0.0, 0.0], dtype=np.float64)
        direction[axis] = 1.0 if local[axis] >= 0.0 else -1.0
        return direction * clear, clear

    clear = float(np.linalg.norm(outward))
    return outward, clear


def point_collides_with_static(point: np.ndarray, obstacle: CircleObstacle | RectObstacle) -> bool:
    if isinstance(obstacle, CircleObstacle):
        return point_in_circle(point, obstacle)
    return point_in_rect(point, obstacle)


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Return the Euclidean distance from a point to a closed line segment."""
    segment = end - start
    length_sq = float(segment @ segment)
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    alpha = float(np.clip((point - start) @ segment / length_sq, 0.0, 1.0))
    closest = start + alpha * segment
    return float(np.linalg.norm(point - closest))


def segment_intersects_aabb(
    start: np.ndarray,
    end: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    """Return whether a closed segment intersects an axis-aligned box."""
    direction = end - start
    t_min = 0.0
    t_max = 1.0
    for axis in (0, 1):
        if abs(float(direction[axis])) <= 1e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        inv_direction = 1.0 / float(direction[axis])
        t_near = float((lower[axis] - start[axis]) * inv_direction)
        t_far = float((upper[axis] - start[axis]) * inv_direction)
        if t_near > t_far:
            t_near, t_far = t_far, t_near
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return False
    return True


def _segments_intersect(start_a: np.ndarray, end_a: np.ndarray, start_b: np.ndarray, end_b: np.ndarray) -> bool:
    def cross(first: np.ndarray, second: np.ndarray) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    direction_a = end_a - start_a
    direction_b = end_b - start_b
    denominator = cross(direction_a, direction_b)
    offset = start_b - start_a
    if abs(denominator) <= 1e-12:
        if abs(cross(offset, direction_a)) > 1e-12:
            return False
        axis = int(np.argmax(np.abs(direction_a)))
        if abs(float(direction_a[axis])) <= 1e-12:
            return bool(np.linalg.norm(start_a - start_b) <= 1e-12)
        a_min, a_max = sorted((float(start_a[axis]), float(end_a[axis])))
        b_min, b_max = sorted((float(start_b[axis]), float(end_b[axis])))
        return max(a_min, b_min) <= min(a_max, b_max) + 1e-12

    t = cross(offset, direction_b) / denominator
    u = cross(offset, direction_a) / denominator
    return -1e-12 <= t <= 1.0 + 1e-12 and -1e-12 <= u <= 1.0 + 1e-12


def _segment_to_segment_distance(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
) -> float:
    if _segments_intersect(start_a, end_a, start_b, end_b):
        return 0.0
    return min(
        point_to_segment_distance(start_a, start_b, end_b),
        point_to_segment_distance(end_a, start_b, end_b),
        point_to_segment_distance(start_b, start_a, end_a),
        point_to_segment_distance(end_b, start_a, end_a),
    )


def segment_to_aabb_distance(
    start: np.ndarray,
    end: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Return the exact minimum distance between a segment and a 2D AABB."""
    if segment_intersects_aabb(start, end, lower, upper):
        return 0.0
    corners = (
        np.array([lower[0], lower[1]], dtype=np.float64),
        np.array([upper[0], lower[1]], dtype=np.float64),
        np.array([upper[0], upper[1]], dtype=np.float64),
        np.array([lower[0], upper[1]], dtype=np.float64),
    )
    return min(
        _segment_to_segment_distance(start, end, corners[index], corners[(index + 1) % 4])
        for index in range(4)
    )


def swept_circle_static_clearance(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    obstacle: CircleObstacle | RectObstacle,
) -> float:
    """Return minimum swept-body clearance to a static obstacle."""
    if isinstance(obstacle, CircleObstacle):
        return point_to_segment_distance(obstacle.center, start, end) - radius - obstacle.radius

    half = np.array([obstacle.length * 0.5, obstacle.width * 0.5], dtype=np.float64)
    return segment_to_aabb_distance(start, end, obstacle.center - half, obstacle.center + half) - radius


def swept_circle_collides_with_static(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    obstacle: CircleObstacle | RectObstacle,
) -> bool:
    """Check a swept circular body against a circle or axis-aligned rectangle."""
    return swept_circle_static_clearance(start, end, radius, obstacle) <= 0.0
