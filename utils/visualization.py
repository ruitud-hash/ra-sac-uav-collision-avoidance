"""Publication-style trajectory visualization for UAV collision avoidance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils.geometry import CircleObstacle, RectObstacle


KM = 1000.0
COLORS = {
    "ownship": "#2F6DB3",
    "start": "#2CA25F",
    "end": "#225EA8",
    "goal": "#F39C12",
    "static_fill": "#D9D9D9",
    "static_edge": "#7F8C8D",
    "no_fly": "#C0392B",
    "dynamic": "#B03A2E",
    "dynamic_light": "#D9A39D",
    "risk": "#D33682",
    "text": "#263238",
}


def plot_episode(
    env,
    save_path: str | Path,
    title: str | None = None,
    paper_style: bool = True,
) -> Path:
    """Draw a local risk view with a global 15 km x 15 km inset."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_m = np.asarray(env.metrics.trajectory, dtype=np.float64)
    trajectory = trajectory_m / KM

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    ):
        fig, ax = plt.subplots(figsize=(7.25, 6.2), dpi=220)
        local_limits = _local_limits(env, trajectory_m) if paper_style else (
            0.0,
            env.world_size,
            0.0,
            env.world_size,
        )

        _draw_static_obstacles(ax, env, local_limits, alpha=0.75)
        _draw_dynamic_traffic(ax, env, local_limits)
        _draw_ownship_trajectory(ax, env, trajectory)
        _draw_goal(ax, env)
        _draw_risk_annotations(ax, env)

        ax.set_xlim(local_limits[0] / KM, local_limits[1] / KM)
        ax.set_ylim(local_limits[2] / KM, local_limits[3] / KM)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title(title or f"UAV collision-avoidance episode ({env.metrics.outcome})", pad=8)
        ax.grid(False)
        ax.tick_params(length=3.2)
        for spine in ax.spines.values():
            spine.set_color("#4D4D4D")
        _add_legend(ax)
        _add_global_inset(ax, env, trajectory, local_limits)

        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.92)
        fig.savefig(save_path, facecolor="white")
        plt.close(fig)
    return save_path


def _draw_ownship_trajectory(ax, env, trajectory: np.ndarray) -> None:
    if len(trajectory) == 0:
        return
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color=COLORS["ownship"],
        linewidth=2.2,
        solid_capstyle="round",
        label="Ownship trajectory",
        zorder=7,
    )
    _draw_quadrotor(
        ax,
        trajectory[-1],
        radius_m=max(float(env.body_radius), 15.0),
        color=COLORS["end"],
        zorder=10,
    )
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        s=34,
        color=COLORS["start"],
        edgecolor="white",
        linewidth=0.6,
        zorder=11,
    )


def _draw_dynamic_traffic(ax, env, limits_m: tuple[float, float, float, float]) -> None:
    histories = getattr(env.metrics, "dynamic_trajectories", [])
    for index, other in enumerate(env.dynamic_uavs):
        history_m = (
            np.asarray(histories[index], dtype=np.float64)
            if index < len(histories) and histories[index]
            else np.asarray([other.position], dtype=np.float64)
        )
        if not _polyline_intersects_limits(history_m, limits_m, margin_m=other.radius):
            continue
        history = history_m / KM
        ax.plot(
            history[:, 0],
            history[:, 1],
            color=COLORS["dynamic_light"],
            linewidth=1.1,
            linestyle=(0, (3, 3)),
            alpha=0.65,
            zorder=3,
        )
        center = np.asarray(other.position, dtype=np.float64) / KM
        # The configured radius is the physical protected occupancy region.
        ax.add_patch(
            Circle(
                center,
                other.radius / KM,
                facecolor=COLORS["dynamic"],
                edgecolor=COLORS["dynamic"],
                alpha=0.16,
                linewidth=0.9,
                zorder=4,
            )
        )
        ax.add_patch(
            Circle(
                center,
                (other.radius + env.body_radius) / KM,
                fill=False,
                edgecolor=COLORS["dynamic"],
                alpha=0.42,
                linewidth=0.9,
                linestyle=(0, (2, 2)),
                zorder=4,
            )
        )
        _draw_quadrotor(
            ax,
            center,
            radius_m=max(0.55 * other.radius, 9.0),
            color=COLORS["dynamic"],
            zorder=6,
        )


def _draw_quadrotor(ax, center_km: np.ndarray, radius_m: float, color: str, zorder: int) -> None:
    """Draw a scale-aware top-view quadrotor as simple vector geometry."""
    arm = max(radius_m / KM, 0.010)
    rotor = max(0.22 * arm, 0.003)
    x, y = float(center_km[0]), float(center_km[1])
    ax.plot([x - arm, x + arm], [y - arm, y + arm], color=color, linewidth=1.05, zorder=zorder)
    ax.plot([x - arm, x + arm], [y + arm, y - arm], color=color, linewidth=1.05, zorder=zorder)
    ax.add_patch(Circle((x, y), 0.23 * arm, facecolor=color, edgecolor="white", linewidth=0.35, zorder=zorder + 1))
    for dx, dy in ((-arm, -arm), (-arm, arm), (arm, -arm), (arm, arm)):
        ax.add_patch(
            Circle(
                (x + dx, y + dy),
                rotor,
                facecolor="white",
                edgecolor=color,
                linewidth=0.75,
                zorder=zorder + 1,
            )
        )


def _draw_static_obstacles(
    ax,
    env,
    limits_m: tuple[float, float, float, float] | None,
    alpha: float,
) -> None:
    for obstacle in env.static_obstacles:
        if limits_m is not None and not _obstacle_intersects_limits(obstacle, limits_m):
            continue
        if isinstance(obstacle, CircleObstacle):
            is_no_fly = obstacle.kind == "no_fly_zone"
            ax.add_patch(
                Circle(
                    obstacle.center / KM,
                    obstacle.radius / KM,
                    facecolor="none" if is_no_fly else COLORS["static_fill"],
                    edgecolor=COLORS["no_fly"] if is_no_fly else COLORS["static_edge"],
                    alpha=0.9 if is_no_fly else alpha,
                    linewidth=1.05 if is_no_fly else 0.9,
                    linestyle=(0, (4, 3)) if is_no_fly else "-",
                    zorder=2,
                )
            )
        elif isinstance(obstacle, RectObstacle):
            lower_left = (
                obstacle.center
                - np.array([obstacle.length * 0.5, obstacle.width * 0.5])
            ) / KM
            ax.add_patch(
                Rectangle(
                    lower_left,
                    obstacle.length / KM,
                    obstacle.width / KM,
                    facecolor=COLORS["static_fill"],
                    edgecolor=COLORS["static_edge"],
                    alpha=alpha,
                    linewidth=0.9,
                    zorder=2,
                )
            )


def _draw_goal(ax, env) -> None:
    goal = np.asarray(env.goal) / KM
    ax.scatter(
        goal[0],
        goal[1],
        s=90,
        marker="*",
        color=COLORS["goal"],
        edgecolor="white",
        linewidth=0.6,
        zorder=11,
    )
    ax.add_patch(
        Circle(
            goal,
            env.goal_radius / KM,
            fill=False,
            edgecolor=COLORS["goal"],
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            alpha=0.8,
            zorder=5,
        )
    )


def _draw_risk_annotations(ax, env) -> None:
    events = np.asarray(getattr(env.metrics, "fhp_events", []), dtype=np.float64)
    if events.size:
        events = events.reshape(-1, 2) / KM
        ax.scatter(
            events[:, 0],
            events[:, 1],
            s=18,
            facecolor=COLORS["risk"],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
            zorder=12,
        )

    point = getattr(env.metrics, "min_clearance_point", None)
    if point is None or not np.isfinite(env.metrics.min_separation_m):
        return
    point_km = np.asarray(point) / KM
    ax.add_patch(
        Circle(
            point_km,
            0.055,
            fill=False,
            edgecolor=COLORS["risk"],
            linewidth=1.4,
            zorder=13,
        )
    )
    label = f"Min. clearance = {env.metrics.min_separation_m:.1f} m"
    ax.annotate(
        label,
        xy=point_km,
        xytext=(16, 18),
        textcoords="offset points",
        fontsize=8,
        color=COLORS["text"],
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": COLORS["risk"], "lw": 0.7, "alpha": 0.94},
        arrowprops={"arrowstyle": "->", "color": COLORS["risk"], "lw": 0.9},
        zorder=14,
    )


def _add_global_inset(ax, env, trajectory: np.ndarray, local_limits_m) -> None:
    inset = inset_axes(ax, width="31%", height="31%", loc="upper left", borderpad=1.0)
    _draw_static_obstacles(inset, env, None, alpha=0.48)
    if len(trajectory):
        inset.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=COLORS["ownship"],
            linewidth=1.15,
            zorder=5,
        )
        inset.scatter(trajectory[0, 0], trajectory[0, 1], s=10, color=COLORS["start"], zorder=6)
    inset.scatter(env.goal[0] / KM, env.goal[1] / KM, s=22, marker="*", color=COLORS["goal"], zorder=6)
    inset.add_patch(
        Rectangle(
            (local_limits_m[0] / KM, local_limits_m[2] / KM),
            (local_limits_m[1] - local_limits_m[0]) / KM,
            (local_limits_m[3] - local_limits_m[2]) / KM,
            fill=False,
            edgecolor=COLORS["risk"],
            linewidth=0.85,
            linestyle=(0, (3, 2)),
            zorder=7,
        )
    )
    inset.set_xlim(0.0, env.world_size / KM)
    inset.set_ylim(0.0, env.world_size / KM)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([0, env.world_size / (2 * KM), env.world_size / KM])
    inset.set_yticks([0, env.world_size / (2 * KM), env.world_size / KM])
    inset.tick_params(labelsize=6, length=2, pad=1)
    inset.set_title("Global scene (km)", fontsize=7, pad=2)
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#5F6368")


def _local_limits(env, trajectory_m: np.ndarray) -> tuple[float, float, float, float]:
    risk_point = getattr(env.metrics, "min_clearance_point", None)
    if risk_point is not None and np.isfinite(env.metrics.min_separation_m):
        center = np.asarray(risk_point, dtype=np.float64)
        side = max(2200.0, 3.2 * env.sense_radius)
    elif len(trajectory_m):
        center = np.mean(trajectory_m, axis=0)
        span = np.ptp(trajectory_m, axis=0)
        side = max(2200.0, 1.18 * float(np.max(span)))
    else:
        center = np.asarray(env.goal, dtype=np.float64)
        side = 2200.0
    side = min(side, env.world_size)
    half = 0.5 * side
    x_min = float(np.clip(center[0] - half, 0.0, env.world_size - side))
    y_min = float(np.clip(center[1] - half, 0.0, env.world_size - side))
    return x_min, x_min + side, y_min, y_min + side


def _polyline_intersects_limits(points: np.ndarray, limits, margin_m: float = 0.0) -> bool:
    x_min, x_max, y_min, y_max = limits
    return bool(
        np.any(
            (points[:, 0] >= x_min - margin_m)
            & (points[:, 0] <= x_max + margin_m)
            & (points[:, 1] >= y_min - margin_m)
            & (points[:, 1] <= y_max + margin_m)
        )
    )


def _obstacle_intersects_limits(obstacle, limits) -> bool:
    x_min, x_max, y_min, y_max = limits
    if isinstance(obstacle, CircleObstacle):
        return (
            obstacle.center[0] + obstacle.radius >= x_min
            and obstacle.center[0] - obstacle.radius <= x_max
            and obstacle.center[1] + obstacle.radius >= y_min
            and obstacle.center[1] - obstacle.radius <= y_max
        )
    half_length = obstacle.length * 0.5
    half_width = obstacle.width * 0.5
    return (
        obstacle.center[0] + half_length >= x_min
        and obstacle.center[0] - half_length <= x_max
        and obstacle.center[1] + half_width >= y_min
        and obstacle.center[1] - half_width <= y_max
    )


def _add_legend(ax) -> None:
    handles = [
        Line2D([0], [0], color=COLORS["ownship"], linewidth=2.2, label="Ownship trajectory"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["start"], markeredgecolor="white", label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["goal"], markeredgecolor="white", markersize=10, label="Goal"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["static_fill"], markeredgecolor=COLORS["static_edge"], label="Static obstacle"),
        Line2D([0], [0], color=COLORS["no_fly"], linestyle="--", linewidth=1.0, label="No-fly zone"),
        Line2D([0], [0], color=COLORS["dynamic_light"], linestyle="--", linewidth=1.1, marker="o", markerfacecolor=COLORS["dynamic"], markeredgecolor=COLORS["dynamic"], label="Dynamic UAV / trail"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["risk"], markeredgecolor="white", label="FHP event"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        edgecolor="#B0B0B0",
        fontsize=7.5,
        borderpad=0.55,
        handlelength=2.2,
    )
