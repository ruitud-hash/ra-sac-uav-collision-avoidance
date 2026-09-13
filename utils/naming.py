"""Naming helpers for generated experiment artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def precise_timestamp() -> str:
    """Return a filesystem-safe timestamp that distinguishes concurrent runs."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def config_stem(config_path: str | Path) -> str:
    return Path(config_path).stem.replace("eval_", "").replace("env_", "")


def algorithm_display_name(algorithm_name: str) -> str:
    """Return a stable human-readable method name from a training run name."""
    normalized = algorithm_name.lower()
    if normalized.startswith("ra_sac_v3"):
        return "RA-SAC-v3"
    if normalized.startswith("ra_sac_v2"):
        return "RA-SAC-v2"
    if normalized.startswith("sac_attention"):
        return "SAC-Attention"
    if normalized.startswith("sac_mlp"):
        return "SAC-MLP"
    if normalized.startswith("td3"):
        return "TD3-MLP"
    if normalized.startswith("ppo"):
        return "PPO-MLP"
    return algorithm_name


def checkpoint_role(checkpoint_path: str | Path) -> str:
    """Extract best/final/step identity from a checkpoint filename."""
    stem = Path(checkpoint_path).stem
    if stem.endswith("_best"):
        return "best"
    if stem.endswith("_final"):
        return "final"
    marker = "_step_"
    if marker in stem:
        return stem[stem.rfind(marker) + 1 :]
    return "checkpoint"


def episode_figure_name(
    *,
    config_name: str,
    purpose: str,
    episode_index: int | None,
    outcome: str,
    policy_name: str,
    created_at: str,
) -> str:
    episode_part = "single" if episode_index is None else f"ep{episode_index:03d}"
    return f"{config_name}_{policy_name}_{purpose}_{episode_part}_{outcome}_{created_at}.png"
