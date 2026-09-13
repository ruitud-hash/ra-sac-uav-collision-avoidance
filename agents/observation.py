"""Observation preprocessing for MLP baselines."""

from __future__ import annotations

import numpy as np
_ACTUATOR_OBSERVATION_KEYS = (
    "normalized_control_delay",
    "current_applied_command",
    "pending_action_queue",
    "pending_queue_mask",
)


def _actuator_features(observation: dict[str, np.ndarray]) -> np.ndarray:
    present = [key for key in _ACTUATOR_OBSERVATION_KEYS if key in observation]
    if not present:
        return np.empty(0, dtype=np.float32)
    if len(present) != len(_ACTUATOR_OBSERVATION_KEYS):
        missing = sorted(set(_ACTUATOR_OBSERVATION_KEYS) - set(present))
        raise ValueError(f"Incomplete queue-observable actuator state: missing={missing}")
    return np.concatenate(
        [observation[key].astype(np.float32).reshape(-1) for key in _ACTUATOR_OBSERVATION_KEYS],
        dtype=np.float32,
    )


def flatten_observation(observation: dict[str, np.ndarray], world_size: float) -> np.ndarray:
    """Flatten dict observations and apply simple scale normalization."""
    scale = max(float(world_size), 1.0)
    uav = observation["uav"].astype(np.float32).copy()
    goal = observation["goal"].astype(np.float32).copy()
    boundary = observation["boundary"].astype(np.float32).copy()
    tokens = observation["tokens"].astype(np.float32).copy()
    mask = observation["mask"].astype(np.float32)

    uav[0:2] /= scale
    uav[2] /= 20.0
    goal[0:3] /= scale
    boundary /= scale

    tokens[:, 0:2] /= scale
    tokens[:, 2:4] /= 40.0
    tokens[:, 4] /= scale
    tokens[:, 5:7] /= 60.0
    tokens[:, 7] /= scale
    tokens[:, 8] /= 200.0
    if tokens.shape[1] > 10:
        tokens[:, 10] = np.clip(tokens[:, 10], 0.0, 1.0)
    tokens *= mask[:, None]

    actuator = _actuator_features(observation)
    return np.concatenate([uav, goal, boundary, actuator, tokens.reshape(-1), mask], dtype=np.float32)


def flat_observation_dim(env) -> int:
    observation = env.reset(seed=0)
    return int(flatten_observation(observation, env.world_size).shape[0])


def attention_observation(observation: dict[str, np.ndarray], world_size: float) -> np.ndarray:
    """Pack global features, token features, and mask for attention networks."""
    scale = max(float(world_size), 1.0)
    uav = observation["uav"].astype(np.float32).copy()
    goal = observation["goal"].astype(np.float32).copy()
    boundary = observation["boundary"].astype(np.float32).copy()
    tokens = observation["tokens"].astype(np.float32).copy()
    mask = observation["mask"].astype(np.float32)

    uav[0:2] /= scale
    uav[2] /= 20.0
    goal[0:3] /= scale
    boundary /= scale

    tokens[:, 0:2] /= scale
    tokens[:, 2:4] /= 40.0
    tokens[:, 4] /= scale
    tokens[:, 5:7] /= 60.0
    tokens[:, 7] /= scale
    tokens[:, 8] /= 200.0
    if tokens.shape[1] > 10:
        tokens[:, 10] = np.clip(tokens[:, 10], 0.0, 1.0)
    tokens *= mask[:, None]

    global_features = np.concatenate([uav, goal, boundary, _actuator_features(observation)], dtype=np.float32)
    return np.concatenate([global_features, tokens.reshape(-1), mask], dtype=np.float32)


def attention_observation_spec(env) -> dict[str, int]:
    observation = env.reset(seed=0)
    packed = attention_observation(observation, env.world_size)
    return {
        "obs_dim": int(packed.shape[0]),
        "global_dim": int(
            observation["uav"].shape[0]
            + observation["goal"].shape[0]
            + observation["boundary"].shape[0]
            + _actuator_features(observation).shape[0]
        ),
        "max_tokens": int(observation["tokens"].shape[0]),
        "token_dim": int(observation["tokens"].shape[1]),
    }
