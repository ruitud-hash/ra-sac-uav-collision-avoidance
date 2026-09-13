"""Write standalone protocol and compatibility audit records."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from utils.experiment_protocol import (
    ensure_final_test_seeds_disjoint,
    evaluation_protocol_metadata,
    training_environment_seed_range,
    observation_config,
    stage1_arm_metadata,
    stage3_arm_metadata,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_provenance(
    *,
    checkpoint_path: Path,
    checkpoint_step: int | None,
    model_config_path: Path,
    environment_config_path: Path,
    train_config: dict[str, Any],
    training_metadata_path: Path | None,
    training_config_path: Path | None,
    method: str,
    evaluation_mode: str,
    evaluation_seed_start: int,
    episodes: int,
) -> dict[str, Any]:
    metadata_seed = None
    if training_metadata_path is not None:
        metadata = _load_yaml(training_metadata_path)
        if metadata.get("stage") != "training" or "training_seed" not in metadata:
            raise ValueError("Training metadata must contain stage=training and training_seed")
        if checkpoint_path.resolve().parent.parent != training_metadata_path.resolve().parent:
            raise ValueError("Training metadata does not belong to the checkpoint run directory")
        metadata_seed = int(metadata["training_seed"])

    config_seed = None
    if training_config_path is not None:
        config_seed = int(_load_yaml(training_config_path)["seed"])
    if metadata_seed is not None and config_seed is not None and metadata_seed != config_seed:
        raise ValueError(
            f"Training-seed provenance mismatch: metadata={metadata_seed}, config={config_seed}"
        )

    training_seed = metadata_seed if metadata_seed is not None else config_seed
    if training_seed is None:
        training_seed = int(train_config["seed"])
        seed_source = "legacy_model_config"
    else:
        seed_source = "training_metadata" if metadata_seed is not None else "training_config"

    checkpoint_role = checkpoint_path.stem.rsplit("_", 1)[-1]
    if checkpoint_step is not None and checkpoint_role != str(checkpoint_step):
        raise ValueError(
            f"Checkpoint-step mismatch: declared={checkpoint_step}, path={checkpoint_path.name}"
        )

    evaluation_seed_end = int(evaluation_seed_start) + int(episodes) - 1
    return {
        "training_seed": training_seed,
        "training_seed_source": seed_source,
        "evaluation_seed_start": int(evaluation_seed_start),
        "evaluation_seed_end": evaluation_seed_end,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "training_config_path": None if training_config_path is None else training_config_path.as_posix(),
        "training_config_sha256": None if training_config_path is None else file_sha256(training_config_path),
        "training_metadata_path": None if training_metadata_path is None else training_metadata_path.as_posix(),
        "training_metadata_sha256": None if training_metadata_path is None else file_sha256(training_metadata_path),
        "model_config_path": model_config_path.as_posix(),
        "model_config_sha256": file_sha256(model_config_path),
        "environment_config_path": environment_config_path.as_posix(),
        "environment_config_sha256": file_sha256(environment_config_path),
        "method": method,
        "evaluation_mode": evaluation_mode,
    }


def write_training_audit_records(
    run_dir: Path,
    *,
    env_config: dict[str, Any],
    train_config: dict[str, Any],
    observation_spec: dict[str, Any],
    execution: dict[str, Any] | None = None,
    artifact_directory: Path | None = None,
    source_provenance: dict[str, Any] | None = None,
    runtime_environment: dict[str, Any] | None = None,
    risk_bias: dict[str, Any] | None = None,
) -> None:
    training = train_config.get("training", {})
    validation_seed = int(training.get("eval_seed", int(train_config["seed"]) + 100000))
    validation_episodes = int(training.get("eval_episodes", 0))
    training_seed_range = training_environment_seed_range(train_config)
    stage3 = stage3_arm_metadata(train_config)
    stage_d_arm = train_config.get("algorithm", {}).get("stage_d_arm")
    _write_yaml(
        run_dir / "protocol_metadata.yaml",
        {
            "stage": "training",
            "execution": execution,
            "artifact_directory": None if artifact_directory is None else str(artifact_directory),
            "source_provenance": source_provenance,
            "runtime_environment": runtime_environment,
            "risk_bias": risk_bias,
            "environment": evaluation_protocol_metadata(env_config),
            "algorithm_name": train_config.get("algorithm", {}).get("name", "unknown"),
            "observation": observation_config(env_config),
            "stage1_ablation": stage1_arm_metadata(train_config),
            "stage3_confirmation": stage3,
            "stage_d_confirmation": (
                None
                if stage_d_arm is None
                else {
                    "arm": str(stage_d_arm),
                    "protocol": train_config.get("confirmation_protocol"),
                    "fixed_checkpoint_step": int(training.get("total_steps", 0)),
                }
            ),
            "output_directory": str(train_config["output"]["directory"]),
            "training_seed": int(train_config["seed"]),
            "training_environment_seeds": (
                None
                if training_seed_range is None
                else {
                    "seed_start": training_seed_range[0],
                    "seed_end_upper_bound": training_seed_range[1],
                    "namespace": training.get("env_seed_namespace"),
                }
            ),
            "periodic_validation": {
                "role": "monitoring_only" if stage3 is not None or stage_d_arm is not None else "checkpoint_selection",
                "seed_start": validation_seed,
                "seed_end": validation_seed + validation_episodes - 1,
                "episodes": validation_episodes,
            },
        },
    )
    _write_yaml(run_dir / "config_snapshot.yaml", train_config)
    _write_yaml(run_dir / "resolved_environment_config.yaml", env_config)
    _write_yaml(
        run_dir / "checkpoint_compatibility_record.yaml",
        {
            "status": "declared",
            "observation_spec": observation_spec,
            "rule": "checkpoint and evaluation environment observation specifications must match",
        },
    )
    final_seed, final_episodes = 19000, 200
    if stage3 is not None:
        final_seed = int(stage3["id_confirmation_seed_start"])
        final_episodes = int(stage3["id_confirmation_seed_end"]) - final_seed + 1
    if train_config.get("evaluation_protocol"):
        protocol_path = Path(__file__).resolve().parents[1] / train_config["evaluation_protocol"]
        with protocol_path.open(encoding="utf-8") as file:
            final_section = yaml.safe_load(file)["final_test"]
        final_seed = int(final_section["seed_start"])
        final_episodes = int(final_section["episodes"])
    final_protocol = ensure_final_test_seeds_disjoint(
        train_config,
        final_seed=final_seed,
        final_episodes=final_episodes,
    )
    planned_role = "planned_id_confirmation" if stage3 is not None else "planned_final_test"
    if stage3 is not None:
        final_protocol["role"] = "id_confirmation"
    _write_yaml(
        run_dir / "seed_overlap_check_record.yaml",
        {
            "status": "passed",
            "periodic_validation": {
                "role": "monitoring_only" if stage3 is not None or stage_d_arm is not None else "checkpoint_selection",
                "seed_start": validation_seed,
                "seed_end": validation_seed + validation_episodes - 1,
            },
            planned_role: final_protocol,
            "overlap": False,
        },
    )


def write_evaluation_audit_records(
    output_dir: Path,
    *,
    env_config: dict[str, Any],
    train_config: dict[str, Any],
    observation_spec: dict[str, Any],
    checkpoint_path: Path,
    final_seed: int,
    final_episodes: int,
    provenance: dict[str, Any] | None = None,
) -> None:
    final_protocol = ensure_final_test_seeds_disjoint(
        train_config,
        final_seed=final_seed,
        final_episodes=final_episodes,
    )
    _write_yaml(
        output_dir / "protocol_metadata.yaml",
        {
            "stage": "final_evaluation",
            "environment": evaluation_protocol_metadata(env_config),
            "observation": observation_config(env_config),
            "stage1_ablation": stage1_arm_metadata(train_config),
            "evaluation": final_protocol,
        },
    )
    _write_yaml(
        output_dir / "checkpoint_compatibility_record.yaml",
        {
            "status": "passed",
            "checkpoint": str(checkpoint_path),
            "observation_spec": observation_spec,
        },
    )
    _write_yaml(
        output_dir / "seed_overlap_check_record.yaml",
        {
            "status": "passed",
            "evaluation": final_protocol,
            "overlap": False,
        },
    )
    if provenance is not None:
        _write_yaml(output_dir / "evaluation_provenance.yaml", provenance)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
