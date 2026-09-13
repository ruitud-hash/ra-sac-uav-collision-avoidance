"""Checkpoint observation-contract metadata and validation."""

from __future__ import annotations

import math
from typing import Any


def observation_spec_metadata(
    *,
    architecture: str,
    obs_dim: int,
    global_dim: int | None = None,
    max_tokens: int | None = None,
    token_dim: int | None = None,
    action_scale: Any | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "architecture": architecture,
        "obs_dim": int(obs_dim),
    }
    if global_dim is not None:
        metadata["global_dim"] = int(global_dim)
    if max_tokens is not None:
        metadata["max_tokens"] = int(max_tokens)
    if token_dim is not None:
        metadata["token_dim"] = int(token_dim)
    if action_scale is not None:
        metadata["action_scale"] = [
            float(value) for value in action_scale
        ]
    return metadata


def validate_checkpoint_observation_spec(
    checkpoint: dict[str, Any],
    expected: dict[str, Any],
    *,
    checkpoint_label: str,
    allow_smoke: bool = False,
) -> None:
    """Reject checkpoints created for a different observation contract."""
    if checkpoint.get("artifact_class") == "smoke_checkpoint" and not allow_smoke:
        raise ValueError(f"Smoke checkpoint is not eligible for formal use: {checkpoint_label}")

    observed = checkpoint.get("observation_spec")
    if observed is None:
        observed = _infer_legacy_observation_spec(checkpoint, expected["architecture"])
    else:
        observed = dict(observed)
        inferred = _infer_legacy_observation_spec(
            checkpoint, expected["architecture"]
        )
        for key, value in inferred.items():
            observed.setdefault(key, value)

    missing = []
    mismatches = []
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if observed_value is None:
            missing.append(key)
        elif not _contract_values_equal(observed_value, expected_value):
            mismatches.append(f"{key}: checkpoint={observed_value}, environment={expected_value}")
    if missing or mismatches:
        details = []
        if missing:
            details.append(f"unverifiable fields={missing}")
        details.extend(mismatches)
        raise ValueError(
            f"Incompatible checkpoint observation contract for {checkpoint_label}: {'; '.join(details)}. "
            "v2 and v3 checkpoints must be trained and evaluated with matching environment protocols."
        )


def _infer_legacy_observation_spec(
    checkpoint: dict[str, Any],
    architecture: str,
) -> dict[str, Any]:
    if architecture == "sac_attention":
        actor = checkpoint.get("actor", {})
        token_weight = actor.get("encoder.token_net.0.weight")
        global_weight = actor.get("encoder.global_net.0.weight")
        inferred: dict[str, Any] = {"architecture": architecture}
        if token_weight is not None:
            inferred["token_dim"] = int(token_weight.shape[1])
        if global_weight is not None:
            inferred["global_dim"] = int(global_weight.shape[1])
        action_scale = actor.get("action_scale")
        if action_scale is not None:
            inferred["action_scale"] = _tensor_values(action_scale)
        return inferred

    state_key = "model" if architecture == "ppo_mlp" else "actor"
    state = checkpoint.get(state_key, {})
    inferred = {"architecture": architecture}
    action_scale = state.get("action_scale")
    if action_scale is not None:
        inferred["action_scale"] = _tensor_values(action_scale)
    for key, value in state.items():
        if key.endswith("weight") and getattr(value, "ndim", 0) == 2:
            inferred["obs_dim"] = int(value.shape[1])
            return inferred
    return inferred


def _tensor_values(value: Any) -> list[float]:
    values = value.detach().cpu().reshape(-1).tolist()
    return [float(item) for item in values]


def _contract_values_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple)):
        if not isinstance(observed, (list, tuple)) or len(observed) != len(expected):
            return False
        return all(
            math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-8)
            for left, right in zip(observed, expected)
        )
    return observed == expected
