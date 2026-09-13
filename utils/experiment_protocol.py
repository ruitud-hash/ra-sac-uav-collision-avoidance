"""Guards that prevent evaluation-only environments from leaking into training."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


FORMAL_TRAINING_NAMESPACE = "formal_v3_training"
FORMAL_TRAINING_SEED_BASE = 1_000_000_000_000
FORMAL_TRAINING_SEED_STRIDE = 10_000_000
FORMAL_FINAL_TEST_SEED = 19_000
FORMAL_FINAL_TEST_EPISODES = 200
FORMAL_SHIELD_STATIC_MARGIN_M = 100.0
FORMAL_SHIELD_BOUNDARY_MARGIN_M = 120.0
FORMAL_SHIELD_DYNAMIC_MARGIN_M = 50.0
FORMAL_SHIELD_DYNAMIC_TTC_MARGIN_S = 4.0
FORMAL_SHIELD_DYNAMIC_RISK_THRESHOLD = 0.5
FORMAL_SHIELD_OMEGA_SAMPLES = 9
FORMAL_SHIELD_LOOKAHEAD_STEPS = 4
FORMAL_SHIELD_VIOLATION_WEIGHT = 80.0
QUEUE_OBSERVABLE_MAX_DELAY_STEPS = 5

_STAGE1_ARMS = {
    "C0": ("legacy", "none", 0.0, None, False, 0.0),
    "C1": ("queue_observable", "none", 0.0, None, False, 0.0),
    "R0": ("legacy", "log1p_risk", 0.25, 1.5, False, 0.0),
    "R1": ("queue_observable", "log1p_risk", 0.25, 1.5, False, 0.0),
    "D0": ("legacy", "log1p_risk", 0.25, 1.5, True, 0.03),
}
_STAGE3_ARMS = {
    "C0": ("none", 0.0, None),
    "R0": ("log1p_risk", 0.25, 1.5),
}
STAGE3_TRAINING_SEEDS = (68_207, 78_207, 88_207, 98_207, 108_207, 118_207, 128_207, 148_207)
STAGE3_MONITOR_SEEDS = (147_500, 147_519)
STAGE3_CONFIRM_SEEDS = (147_200, 147_399)


def observation_config(config: dict[str, Any]) -> dict[str, Any]:
    configured = config.get("observation", {})
    mode = str(configured.get("actuator_state_mode", "legacy"))
    if mode not in {"legacy", "queue_observable"}:
        raise ValueError(f"Unsupported observation.actuator_state_mode: {mode!r}")
    max_delay = int(configured.get("max_delay_steps", QUEUE_OBSERVABLE_MAX_DELAY_STEPS))
    if max_delay <= 0:
        raise ValueError(f"observation.max_delay_steps must be positive, got {max_delay}")
    if mode == "queue_observable" and max_delay != QUEUE_OBSERVABLE_MAX_DELAY_STEPS:
        raise ValueError(
            "queue_observable requires observation.max_delay_steps=5, "
            f"got {max_delay}"
        )
    return {"actuator_state_mode": mode, "max_delay_steps": max_delay}


def resolved_environment_config(
    env_config: dict[str, Any], train_config: dict[str, Any]
) -> dict[str, Any]:
    """Apply the explicit training observation contract to an environment."""
    resolved = deepcopy(env_config)
    if "observation" in train_config:
        resolved["observation"] = deepcopy(train_config["observation"])
    resolved["observation"] = observation_config(resolved)
    return resolved


def stage1_arm_metadata(train_config: dict[str, Any]) -> dict[str, Any] | None:
    algorithm = train_config.get("algorithm", {})
    arm = algorithm.get("stage1_arm")
    if arm is None:
        return None
    arm = str(arm).upper()
    if arm not in _STAGE1_ARMS:
        raise ValueError(f"Unknown algorithm.stage1_arm: {arm!r}")
    if "observation" not in train_config:
        raise ValueError("Stage-1 configs must explicitly declare observation")
    seed = int(train_config["seed"])
    total_steps = int(train_config.get("training", {}).get("total_steps", 0))
    if seed not in {28_207, 38_207}:
        raise ValueError(f"Stage-1 diagnostic training seed is not authorized: {seed}")
    if not 1_500_000 <= total_steps <= 2_000_000:
        raise ValueError(
            f"Stage-1 diagnostic total_steps must be in [1500000, 2000000], got {total_steps}"
        )

    mode, risk_mode, risk_scale, risk_clip, guidance_enabled, guidance_lambda = _STAGE1_ARMS[arm]
    observed = observation_config(train_config)
    guidance = algorithm.get("goal_guidance", {})
    actual = (
        observed["actuator_state_mode"],
        str(algorithm.get("risk_bias_mode", "none")),
        float(algorithm.get("risk_bias_scale", 0.0)),
        None if algorithm.get("risk_bias_clip") is None else float(algorithm["risk_bias_clip"]),
        bool(guidance.get("enabled", False)),
        float(guidance.get("lambda", 0.0)),
    )
    expected = (mode, risk_mode, risk_scale, risk_clip, guidance_enabled, guidance_lambda)
    if actual != expected:
        raise ValueError(f"Stage-1 arm {arm} mismatch: actual={actual}, expected={expected}")
    if "enabled" not in guidance or "lambda" not in guidance:
        raise ValueError("Stage-1 goal_guidance must explicitly declare enabled and lambda")

    return {
        "stage1_arm": arm,
        "observation": observed,
        "risk_bias_mode": risk_mode,
        "risk_bias_scale": risk_scale,
        "risk_bias_clip": risk_clip,
        "goal_guidance_enabled": guidance_enabled,
        "goal_guidance_lambda": guidance_lambda,
        "goal_guidance_loss_computed": guidance_enabled and guidance_lambda > 0.0,
    }


def stage3_arm_metadata(train_config: dict[str, Any]) -> dict[str, Any] | None:
    algorithm = train_config.get("algorithm", {})
    arm = algorithm.get("stage3_arm")
    if arm is None:
        return None
    arm = str(arm).upper()
    if arm not in _STAGE3_ARMS:
        raise ValueError(f"Unknown algorithm.stage3_arm: {arm!r}")
    if algorithm.get("stage1_arm") is not None:
        raise ValueError("A config cannot declare both stage1_arm and stage3_arm")
    if "observation" not in train_config:
        raise ValueError("Stage-3 configs must explicitly declare observation")

    expected_protocol = "configs/stage3/r0_c0_confirmatory_protocol.yaml"
    if train_config.get("stage3_protocol") != expected_protocol:
        raise ValueError(f"Stage-3 config must reference {expected_protocol}")

    seed = int(train_config["seed"])
    training = train_config.get("training", {})
    if seed not in STAGE3_TRAINING_SEEDS:
        raise ValueError(f"Stage-3 confirmatory training seed is not authorized: {seed}")
    if int(training.get("total_steps", 0)) != 2_000_000:
        raise ValueError("Stage-3 confirmatory total_steps must equal 2000000")
    if (
        int(training.get("eval_seed", -1)),
        int(training.get("eval_episodes", -1)),
        training.get("env_seed_namespace"),
    ) != (STAGE3_MONITOR_SEEDS[0], 20, FORMAL_TRAINING_NAMESPACE):
        raise ValueError("Stage-3 monitoring seeds or training namespace mismatch")

    risk_mode, risk_scale, risk_clip = _STAGE3_ARMS[arm]
    observed = observation_config(train_config)
    guidance = algorithm.get("goal_guidance", {})
    actual = (
        observed["actuator_state_mode"],
        int(observed["max_delay_steps"]),
        str(algorithm.get("risk_bias_mode", "none")),
        float(algorithm.get("risk_bias_scale", 0.0)),
        None if algorithm.get("risk_bias_clip") is None else float(algorithm["risk_bias_clip"]),
        bool(guidance.get("enabled", False)),
        float(guidance.get("lambda", 0.0)),
    )
    expected = ("legacy", 5, risk_mode, risk_scale, risk_clip, False, 0.0)
    if actual != expected:
        raise ValueError(f"Stage-3 arm {arm} mismatch: actual={actual}, expected={expected}")
    if train_config.get("init_checkpoint") is not None:
        raise ValueError("Stage-3 runs must start from scratch")

    return {
        "stage3_arm": arm,
        "observation": observed,
        "risk_bias_mode": risk_mode,
        "risk_bias_scale": risk_scale,
        "risk_bias_clip": risk_clip,
        "goal_guidance_enabled": False,
        "goal_guidance_lambda": 0.0,
        "goal_guidance_loss_computed": False,
        "monitor_seed_start": STAGE3_MONITOR_SEEDS[0],
        "monitor_seed_end": STAGE3_MONITOR_SEEDS[1],
        "id_confirmation_seed_start": STAGE3_CONFIRM_SEEDS[0],
        "id_confirmation_seed_end": STAGE3_CONFIRM_SEEDS[1],
    }


def ensure_training_environment_allowed(config: dict[str, Any]) -> None:
    """Reject OOD/evaluation-only configs before training or best-checkpoint selection."""
    protocol = config.get("protocol", {})
    is_ood = str(protocol.get("split", "")).lower() == "ood"
    evaluation_only = protocol.get("evaluation_only") is True
    selection_forbidden = protocol.get("checkpoint_selection_eligible") is False
    if is_ood or evaluation_only or selection_forbidden:
        axis = protocol.get("ood_axis", "unspecified")
        raise ValueError(
            "OOD environments are evaluation-only and cannot be used for training "
            f"or checkpoint selection (ood_axis={axis!r})"
        )


def evaluation_protocol_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Return auditable split metadata for checkpoint-evaluation artifacts."""
    protocol = config.get("protocol", {})
    split = str(protocol.get("split", "id")).lower()
    is_ood = split == "ood"
    return {
        "name": protocol.get("name", "legacy"),
        "split": split,
        "ood_axis": protocol.get("ood_axis"),
        "evaluation_only": bool(protocol.get("evaluation_only", is_ood)),
        "checkpoint_selection_eligible": bool(
            protocol.get("checkpoint_selection_eligible", not is_ood)
        ),
        "reference_environment": protocol.get("reference_environment"),
    }


def ensure_final_test_seeds_disjoint(
    train_config: dict[str, Any],
    *,
    final_seed: int,
    final_episodes: int,
) -> dict[str, int | str]:
    """Ensure final-test scenarios do not overlap periodic validation scenarios."""
    training = train_config.get("training", {})
    validation_seed = training.get("eval_seed")
    validation_episodes = training.get("eval_episodes")
    final_start = int(final_seed)
    final_end = final_start + int(final_episodes) - 1

    metadata: dict[str, int | str] = {
        "role": "final_test",
        "seed_start": final_start,
        "seed_end": final_end,
        "episodes": int(final_episodes),
    }
    training_range = training_environment_seed_range(train_config)
    if training_range is not None:
        training_start, training_end = training_range
        training_overlaps = max(final_start, training_start) <= min(
            final_end, training_end
        )
        if training_overlaps:
            raise ValueError(
                "Final-test seeds overlap training-environment seeds: "
                f"final={final_start}-{final_end}, "
                f"training={training_start}-{training_end}."
            )
        metadata["training_seed_start"] = training_start
        metadata["training_seed_end"] = training_end
    if validation_seed is None or validation_episodes is None:
        return metadata

    validation_start = int(validation_seed)
    validation_end = validation_start + int(validation_episodes) - 1
    if training_range is not None:
        training_start, training_end = training_range
        training_validation_overlap = max(
            validation_start, training_start
        ) <= min(validation_end, training_end)
        if training_validation_overlap:
            raise ValueError(
                "Periodic-validation seeds overlap training-environment seeds: "
                f"validation={validation_start}-{validation_end}, "
                f"training={training_start}-{training_end}."
            )
    overlaps = max(final_start, validation_start) <= min(final_end, validation_end)
    if overlaps:
        raise ValueError(
            "Final-test seeds overlap periodic-validation seeds: "
            f"final={final_start}-{final_end}, "
            f"validation={validation_start}-{validation_end}. "
            "Checkpoint selection and paper evaluation must use independent scenarios."
        )
    metadata["validation_seed_start"] = validation_start
    metadata["validation_seed_end"] = validation_end
    return metadata


def training_environment_seed(
    train_config: dict[str, Any],
    episode_index: int,
) -> int:
    """Return the environment seed for one training episode."""
    index = int(episode_index)
    if index < 0:
        raise ValueError(f"episode_index must be non-negative, got {index}")
    run_seed = int(train_config["seed"])
    namespace = train_config.get("training", {}).get("env_seed_namespace")
    if namespace is None:
        return run_seed + index
    if namespace != FORMAL_TRAINING_NAMESPACE:
        raise ValueError(f"Unsupported training environment seed namespace: {namespace!r}")
    _validate_formal_training_seed_budget(train_config)
    return FORMAL_TRAINING_SEED_BASE + run_seed * FORMAL_TRAINING_SEED_STRIDE + index


def training_environment_seed_range(
    train_config: dict[str, Any],
) -> tuple[int, int] | None:
    """Return a conservative formal-training seed range, if namespaced."""
    training = train_config.get("training", {})
    if training.get("env_seed_namespace") is None:
        return None
    _validate_formal_training_seed_budget(train_config)
    start = training_environment_seed(train_config, 0)
    max_episodes = int(training["total_steps"])
    return start, start + max_episodes - 1


def _validate_formal_training_seed_budget(train_config: dict[str, Any]) -> None:
    training = train_config.get("training", {})
    total_steps = int(training.get("total_steps", 0))
    if total_steps <= 0:
        raise ValueError(f"training.total_steps must be positive, got {total_steps}")
    if total_steps >= FORMAL_TRAINING_SEED_STRIDE:
        raise ValueError(
            "Formal training total_steps must be below the per-run seed stride "
            f"{FORMAL_TRAINING_SEED_STRIDE}, got {total_steps}"
        )
