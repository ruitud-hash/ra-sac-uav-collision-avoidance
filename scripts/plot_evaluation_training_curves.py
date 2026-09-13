"""Plot formal convergence curves from fixed periodic-evaluation logs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation_curves import (
    RARE_EVENT_METRICS,
    aggregate_seed_curves,
    aggregate_seed_series,
)


METRIC_LABELS = {
    "eval_return_mean": ("Evaluation Episodic Return", "Return"),
    "eval_success_rate": ("Evaluation Success Rate", "Rate"),
    "eval_collision_rate": ("Evaluation Collision Rate", "Rate"),
    "eval_timeout_rate": ("Evaluation Timeout Rate", "Rate"),
    "eval_safety_failure_rate": ("Evaluation Safety Failure Rate", "Rate"),
}
Y_METRIC_ALIASES = {
    "eval_return": "eval_return_mean",
    "eval_success": "eval_success_rate",
    "eval_collision": "eval_collision_rate",
    "eval_timeout": "eval_timeout_rate",
    "eval_safety_failure": "eval_safety_failure_rate",
    "average_episode_reward": "episode_reward",
}
DEFAULT_COLORS = ("#c43c39", "#3268a8", "#3d8b5f", "#7656a5", "#c68132")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("METHOD", "SEED", "LOG"),
        required=True,
        help="Repeat exactly five times per method. Use eval_log.csv in steps mode and episode_log.csv in episode mode.",
    )
    parser.add_argument("--x-axis", choices=("steps", "episode"), default="steps")
    parser.add_argument("--y-metric", choices=tuple(Y_METRIC_ALIASES), default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(METRIC_LABELS),
        default=["eval_return_mean", "eval_success_rate"],
    )
    parser.add_argument("--smooth-window", type=int, choices=(1, 3, 5, 50, 100), default=3)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--shade-alpha", type=float, default=0.18)
    parser.add_argument("--output", default=None)
    parser.add_argument("--source-data", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_numeric_log(
    path: Path,
    *,
    required_x: str,
    required_metric: str,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        for required in (required_x, required_metric):
            if required not in fieldnames:
                raise ValueError(f"{path} is missing required column {required!r}")
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value in {"", None}:
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            rows.append(parsed)
    return rows


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def group_runs(run_args: list[list[str]], expected_seeds: int) -> dict[str, list[tuple[str, Path]]]:
    grouped: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for method, seed, raw_path in run_args:
        grouped[method].append((seed, resolve_path(raw_path)))
    for method, runs in grouped.items():
        seeds = [seed for seed, _ in runs]
        if len(runs) != expected_seeds or len(set(seeds)) != expected_seeds:
            raise ValueError(
                f"{method} requires {expected_seeds} unique seeds, got {seeds}"
            )
        missing = [path for _, path in runs if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{method} has missing training logs: {missing}")
    return dict(grouped)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111827",
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
        }
    )


def build_figure(
    grouped_runs: dict[str, list[tuple[str, Path]]],
    metrics: list[str],
    *,
    x_axis: str,
    smooth_window: int,
    shade_alpha: float,
) -> tuple[plt.Figure, list[dict[str, object]]]:
    if not 0.15 <= shade_alpha <= 0.20:
        raise ValueError("shade alpha must be between 0.15 and 0.20 for formal curves")
    setup_style()
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5.2 * len(metrics), 3.8),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()
    source_rows: list[dict[str, object]] = []

    for method_index, (method, runs) in enumerate(grouped_runs.items()):
        color = DEFAULT_COLORS[method_index % len(DEFAULT_COLORS)]
        for axis, metric in zip(axes, metrics):
            x_key = "step" if x_axis == "steps" else "episode"
            rows_by_seed = [
                read_numeric_log(
                    path,
                    required_x=x_key,
                    required_metric=metric,
                )
                for _, path in runs
            ]
            curve = (
                aggregate_seed_curves(
                    rows_by_seed,
                    metric,
                    smooth_window=smooth_window,
                )
                if x_axis == "steps"
                else aggregate_seed_series(
                    rows_by_seed,
                    x_key="episode",
                    metric=metric,
                    smooth_window=smooth_window,
                )
            )
            x = curve["steps"]
            mean = curve["mean"]
            se = curve["standard_error"]
            if metric in RARE_EVENT_METRICS:
                axis.scatter(
                    x,
                    curve["raw_mean"],
                    color=color,
                    alpha=0.25,
                    s=12,
                    linewidths=0,
                    zorder=2,
                )
            axis.plot(x, mean, color=color, linewidth=2.0, label=method, zorder=3)
            axis.fill_between(
                x,
                mean - se,
                mean + se,
                color=color,
                alpha=shade_alpha,
                linewidth=0,
                zorder=1,
            )
            for index, x_value in enumerate(x):
                source_rows.append(
                    {
                        "method": method,
                        "x_axis": x_axis,
                        "x_value": float(x_value),
                        "metric": metric,
                        "raw_mean": float(curve["raw_mean"][index]),
                        "smoothed_mean": float(mean[index]),
                        "standard_error": float(se[index]),
                        "band_lower": float(mean[index] - se[index]),
                        "band_upper": float(mean[index] + se[index]),
                        "seed_count": int(curve["seed_count"][0]),
                        "smooth_window": int(smooth_window),
                    }
                )

    for axis, metric in zip(axes, metrics):
        title, y_label = (
            ("Average Episode Reward", "Average episode reward")
            if metric == "episode_reward"
            else METRIC_LABELS[metric]
        )
        axis.set_title(title)
        axis.set_xlabel("Environment Steps" if x_axis == "steps" else "Episode")
        axis.set_ylabel(y_label)
        axis.grid(True, color="#e5e7eb", linewidth=0.8)
        if metric in {
            "eval_success_rate",
            "eval_collision_rate",
            "eval_timeout_rate",
            "eval_safety_failure_rate",
        }:
            axis.set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=True)
    return fig, source_rows


def main() -> None:
    args = parse_args()
    metrics = (
        [Y_METRIC_ALIASES[args.y_metric]]
        if args.y_metric is not None
        else list(args.metrics)
    )
    if args.x_axis == "steps":
        if "episode_reward" in metrics:
            raise ValueError("average_episode_reward requires --x-axis episode")
        if args.smooth_window not in {1, 3, 5}:
            raise ValueError("periodic evaluation curves require smooth window 1, 3, or 5")
    elif metrics != ["episode_reward"]:
        raise ValueError(
            "episode mode is reserved for --y-metric average_episode_reward"
        )
    grouped = group_runs(args.run, int(args.expected_seeds))
    figure, source_rows = build_figure(
        grouped,
        metrics,
        x_axis=args.x_axis,
        smooth_window=int(args.smooth_window),
        shade_alpha=float(args.shade_alpha),
    )
    default_output = (
        "outputs/paper_figures/fig3_evaluation_return_success_vs_environment_steps.png"
        if args.x_axis == "steps"
        else "outputs/paper_figures/figA1_average_episode_reward_vs_training_episodes.png"
    )
    output = resolve_path(args.output or default_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(args.dpi))
    plt.close(figure)
    default_source = (
        output.parent / "fig3_source_data.csv"
        if args.x_axis == "steps"
        else output.parent / "figA1_source_data.csv"
    )
    source_path = resolve_path(args.source_data) if args.source_data else default_source
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)
    print(f"Evaluation training curves: {output}")
    print(f"Source data: {source_path}")
    print(f"X-axis: {'Environment Steps' if args.x_axis == 'steps' else 'Episode'}")
    print(f"Smoothing: centered moving average, window={args.smooth_window}")
    print(f"Band: standard error across {args.expected_seeds} seeds, alpha={args.shade_alpha:.2f}")
    if args.x_axis == "episode":
        print(
            "Average episode reward is the episode-level accumulated reward "
            "averaged across training seeds."
        )
        if args.smooth_window in {3, 5}:
            print("Note: raw training reward usually benefits from window 50 or 100.")


if __name__ == "__main__":
    main()
