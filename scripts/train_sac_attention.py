"""Train a SAC-Attention baseline agent."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
import yaml
from tqdm import trange


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGE1_ARMS = ("c0", "c1", "r0", "r1", "d0")
STAGE1_SEEDS = (28_207, 38_207)
STAGE3_ARMS = ("c0", "r0")
STAGE3_SEEDS = (68_207, 78_207, 88_207, 98_207, 108_207, 118_207, 128_207, 148_207)
STAGE3_FORMAL_AMENDMENT_PATH = Path(
    "configs/stage3/r0_c0_formal_training_amendment.yaml"
)
STAGE3_FORMAL_FREEZE_PATH = Path(
    "configs/stage3/r0_c0_formal_training_freeze.sha256"
)
STAGE3_SERVER_SMOKE_FREEZE_PATH = Path(
    "configs/stage3/r0_c0_server_smoke_freeze.sha256"
)
STAGE3_PARALLEL_SMOKE_AMENDMENT_PATH = Path(
    "configs/stage3/r0_c0_parallel_execution_smoke_amendment.yaml"
)
STAGE3_PARALLEL_SMOKE_FREEZE_PATH = Path(
    "configs/stage3/r0_c0_parallel_execution_smoke_freeze.sha256"
)
STAGE3_PARALLEL_SERVER_SMOKE_FREEZE_PATH = Path(
    "configs/stage3/r0_c0_parallel_server_smoke_freeze.sha256"
)
STAGE3_PARALLEL_FORMAL_AMENDMENT_PATH = Path(
    "configs/stage3/r0_c0_parallel_formal_training_amendment.yaml"
)
STAGE3_PARALLEL_FORMAL_FREEZE_PATH = Path(
    "configs/stage3/r0_c0_parallel_formal_training_freeze.sha256"
)
STAGE1_SOURCE_FILES = (
    "agents/observation.py",
    "agents/replay_buffer.py",
    "agents/sac_attention.py",
    "envs/uav_2d_env.py",
    "scripts/train_sac_attention.py",
    "utils/artifact_audit.py",
    "utils/checkpoint_compat.py",
    "utils/experiment_protocol.py",
    "utils/naming.py",
    "utils/randomness.py",
    "utils/training_logs.py",
)


from agents.observation import attention_observation, attention_observation_spec
from agents.replay_buffer import ReplayBuffer
from agents.sac_attention import RISK_DIAGNOSTIC_FIELDS, SACAttentionAgent
from envs import UAV2DEnv
from utils.artifact_audit import write_training_audit_records
from utils.checkpoint_compat import validate_checkpoint_observation_spec
from utils.experiment_protocol import (
    FORMAL_TRAINING_NAMESPACE,
    ensure_training_environment_allowed,
    observation_config,
    resolved_environment_config,
    stage1_arm_metadata,
    stage3_arm_metadata,
    training_environment_seed,
    training_environment_seed_range,
)
from utils.naming import timestamp
from utils.randomness import preserve_global_rng_state
from utils.training_logs import (
    EPISODE_CORE_FIELDS,
    ACTUATOR_AUDIT_FIELDS,
    actuator_audit_fields,
    EVAL_FIELDNAMES,
    eval_selection_key,
    make_episode_row,
    serialize_episode_randomization,
    summarize_eval_episodes,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC-Attention on the UAV 2D environment.")
    parser.add_argument("--config", default="configs/train_sac_attention_tiny.yaml")
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint used to initialize the actor.")
    parser.add_argument("--check-config", action="store_true", help="Validate and print the resolved protocol, then exit.")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help="Run an isolated Stage-1/Stage-3 preflight smoke test without changing the formal YAML budget.",
    )
    parser.add_argument(
        "--execution-amendment",
        default=None,
        help="Frozen execution-only amendment for Stage-3 parallel smoke/formal runs.",
    )
    return parser.parse_args()

def resolved_execution_config(
    train_config: dict,
    smoke_steps: int | None,
    execution_amendment: dict | None = None,
) -> tuple[dict, dict]:
    """Return effective loop settings and auditable execution metadata."""
    formal = train_config["training"]
    effective = deepcopy(formal)
    stage3 = stage3_arm_metadata(train_config)
    stage_d = train_config.get("algorithm", {}).get("stage_d_arm") is not None
    if execution_amendment is not None and stage3 is None:
        raise ValueError("Execution amendments are restricted to Stage-3")
    if smoke_steps is None:
        authorized = stage3 is None or bool(
            execution_amendment
            and execution_amendment["authorization"]["parallel_formal_training"]
        )
        execution = {
            "mode": "formal_training",
            "formal_config_total_steps": int(formal["total_steps"]),
            "planned_total_steps": int(formal["total_steps"]),
            "formal_training_authorized": authorized,
            "checkpoint_selection_eligible": stage3 is None and not stage_d,
            "formal_result_eligible": authorized,
            "confirmatory_eligible": stage3 is None,
            "execution_amendment": None if execution_amendment is None else execution_amendment["_path"],
        }
        if execution_amendment is not None:
            execution.update(
                {
                    "execution_amendment_sha256": execution_amendment["_sha256"],
                    "execution_classification": execution_amendment["classification"][
                        "replacement_label"
                    ],
                    "concurrent_processes": int(
                        execution_amendment["execution"]["concurrent_processes"]
                    ),
                    "confirmatory_eligible": False,
                }
            )
        return effective, execution

    if stage1_arm_metadata(train_config) is None and stage3 is None:
        raise ValueError("--smoke-steps is restricted to declared Stage-1/Stage-3 arms")
    if smoke_steps < 64:
        raise ValueError("--smoke-steps must be at least 64 so one gradient update is exercised")

    smoke_batch_size = min(int(formal["batch_size"]), 64)
    effective.update(
        total_steps=int(smoke_steps),
        start_steps=min(int(formal["start_steps"]), 16),
        update_after=min(int(formal["update_after"]), smoke_batch_size),
        batch_size=smoke_batch_size,
        replay_size=max(int(smoke_steps), smoke_batch_size),
        eval_interval_steps=int(smoke_steps) + 1,
        checkpoint_interval_steps=int(smoke_steps) + 1,
    )
    execution = {
        "mode": "smoke",
        "purpose": "preflight_only",
        "formal_config_total_steps": int(formal["total_steps"]),
        "planned_total_steps": int(smoke_steps),
        "effective_start_steps": int(effective["start_steps"]),
        "effective_update_after": int(effective["update_after"]),
        "effective_batch_size": int(effective["batch_size"]),
        "effective_replay_size": int(effective["replay_size"]),
        "runtime_overrides": {
            "smoke_steps": int(smoke_steps),
            "start_steps": int(effective["start_steps"]),
            "learning_starts": int(effective["update_after"]),
            "batch_size": int(effective["batch_size"]),
            "replay_size": int(effective["replay_size"]),
            "periodic_evaluation": False,
        },
        "periodic_validation_enabled": False,
        "checkpoint_selection_eligible": False,
        "formal_result_eligible": False,
    }
    if execution_amendment is not None:
        authorization = execution_amendment["authorization"]
        required_steps = int(execution_amendment["smoke"]["steps_per_process"])
        if not authorization.get("parallel_smoke"):
            raise PermissionError("Stage-3 parallel smoke is not authorized")
        if int(smoke_steps) != required_steps:
            raise ValueError(
                f"Stage-3 parallel smoke requires exactly {required_steps} steps"
            )
        execution.update(
            {
                "execution_amendment": execution_amendment["_path"],
                "execution_amendment_sha256": execution_amendment["_sha256"],
                "execution_classification": execution_amendment["classification"][
                    "replacement_label"
                ],
                "concurrent_processes": int(
                    execution_amendment["execution"]["concurrent_processes"]
                ),
                "confirmatory_eligible": False,
            }
        )
    return effective, execution



def ensure_formal_training_authorized(
    train_config: dict, execution_amendment: dict | None = None
) -> None:
    if stage3_arm_metadata(train_config) is not None:
        if execution_amendment is None:
            raise PermissionError("Stage-3 execution amendment is required")
        if not execution_amendment["authorization"].get("parallel_formal_training"):
            raise PermissionError("Stage-3 parallel formal training is not authorized")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage3_formal_training_amendment() -> dict:
    amendment = load_yaml(PROJECT_ROOT / STAGE3_FORMAL_AMENDMENT_PATH)
    if amendment.get("status") != "frozen":
        raise PermissionError("Stage-3 formal training amendment is not frozen")
    base = amendment["base_protocol_freeze"]
    if (
        Path(base["path"]).as_posix() != Path("configs/stage3/r0_c0_freeze.sha256").as_posix()
        or _file_sha256(PROJECT_ROOT / base["path"]) != base["sha256"]
    ):
        raise PermissionError("Stage-3 protocol freeze mismatch")
    smoke = amendment["server_smoke_gate"]
    if (
        smoke.get("status") != "passed"
        or Path(smoke["freeze_path"]).as_posix() != STAGE3_SERVER_SMOKE_FREEZE_PATH.as_posix()
        or _file_sha256(PROJECT_ROOT / smoke["freeze_path"]) != smoke["freeze_sha256"]
        or _file_sha256(PROJECT_ROOT / smoke["audit_path"]) != smoke["audit_sha256"]
    ):
        raise PermissionError("Stage-3 server smoke gate mismatch")
    authorization = amendment["authorization"]
    if not authorization.get("formal_training"):
        raise PermissionError("Stage-3 formal training is not authorized")
    prohibited = (
        "checkpoint_selection",
        "id_confirmation",
        "ood_evaluation",
        "shield_evaluation",
        "final_test",
    )
    if any(authorization.get(key) for key in prohibited):
        raise PermissionError("Stage-3 formal amendment exceeds the authorized boundary")
    return amendment


def load_stage3_execution_amendment(path: str | Path) -> dict:
    relative = Path(path)
    if relative.is_absolute():
        relative = relative.relative_to(PROJECT_ROOT)
    allowed_paths = {
        STAGE3_PARALLEL_SMOKE_AMENDMENT_PATH.as_posix(),
        STAGE3_PARALLEL_FORMAL_AMENDMENT_PATH.as_posix(),
    }
    if relative.as_posix() not in allowed_paths:
        raise PermissionError("Unexpected Stage-3 execution amendment")
    amendment_path = PROJECT_ROOT / relative
    amendment = load_yaml(amendment_path)
    if amendment.get("status") != "frozen":
        raise PermissionError("Stage-3 execution amendment is not frozen")
    references = (
        ("base_formal_training_amendment", STAGE3_FORMAL_AMENDMENT_PATH),
        ("base_formal_training_freeze", STAGE3_FORMAL_FREEZE_PATH),
    )
    for key, expected_path in references:
        record = amendment[key]
        if (
            Path(record["path"]).as_posix() != expected_path.as_posix()
            or _file_sha256(PROJECT_ROOT / record["path"]) != record["sha256"]
        ):
            raise PermissionError(f"Stage-3 execution amendment reference mismatch: {key}")
    classification = amendment["classification"]
    if (
        classification.get("original_confirmatory_design") is not False
        or classification.get("confirmatory_claim_eligible") is not False
    ):
        raise PermissionError("Parallel execution must remain non-confirmatory")
    authorization = amendment["authorization"]
    if relative == STAGE3_PARALLEL_SMOKE_AMENDMENT_PATH:
        if not authorization.get("parallel_smoke"):
            raise PermissionError("Stage-3 parallel smoke is not authorized")
        prohibited = (
            "parallel_formal_training",
            "checkpoint_selection",
            "id_confirmation",
            "ood_evaluation",
            "shield_evaluation",
            "final_test",
        )
    else:
        gate = amendment["parallel_server_smoke_gate"]
        audit_path = Path("tables/stage3_parallel_server_smoke_audit.yaml")
        if (
            gate.get("status") != "passed"
            or Path(gate["freeze_path"]).as_posix()
            != STAGE3_PARALLEL_SERVER_SMOKE_FREEZE_PATH.as_posix()
            or _file_sha256(PROJECT_ROOT / gate["freeze_path"])
            != gate["freeze_sha256"]
            or Path(gate["audit_path"]).as_posix() != audit_path.as_posix()
            or _file_sha256(PROJECT_ROOT / gate["audit_path"])
            != gate["audit_sha256"]
        ):
            raise PermissionError("Stage-3 parallel server smoke gate mismatch")
        audit = load_yaml(PROJECT_ROOT / audit_path)
        if (
            audit.get("status") != "passed"
            or audit.get("confirmatory_eligible") is not False
            or audit["parallelism"].get("completed_processes") != 5
            or audit["parallelism"].get("common_five_process_overlap_seconds", 0)
            <= 0
        ):
            raise PermissionError("Stage-3 parallel server smoke evidence did not pass")
        if not authorization.get("parallel_formal_training"):
            raise PermissionError("Stage-3 parallel formal training is not authorized")
        expected_runs = {
            (arm.upper(), seed) for arm in STAGE3_ARMS for seed in STAGE3_SEEDS
        }
        run_sequence = [
            (run["arm"].upper(), int(run["seed"]))
            for terminal in amendment["execution"]["terminal_distribution"].values()
            for run in terminal["run_sequence"]
        ]
        if (
            len(run_sequence) != 16
            or set(run_sequence) != expected_runs
            or amendment["execution"].get("concurrent_processes") != 5
            or amendment["execution"].get("each_process_runs_serially") is not True
            or amendment["execution"].get("global_gpu_lock") is not False
        ):
            raise PermissionError("Stage-3 parallel formal execution matrix mismatch")
        prohibited = (
            "parallel_smoke",
            "checkpoint_selection",
            "id_confirmation",
            "ood_evaluation",
            "shield_evaluation",
            "final_test",
        )
    if any(authorization.get(key) for key in prohibited):
        raise PermissionError("Stage-3 execution amendment exceeds its authorization")
    amendment["_path"] = relative.as_posix()
    amendment["_sha256"] = _file_sha256(amendment_path)
    return amendment


def _manifest_sha256(relative_paths: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((PROJECT_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_metadata() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        return {"git_commit": None, "working_tree_clean": None, "git_status": "unavailable"}
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "git_commit": commit.stdout.strip(),
        "working_tree_clean": status.returncode == 0 and not status.stdout.strip(),
        "git_status": "available" if status.returncode == 0 else "unavailable",
    }


def stage1_source_provenance() -> dict:
    config_paths = [
        f"configs/stage1/train_stage1_{arm}_medium_seed_{seed}.yaml"
        for arm in STAGE1_ARMS
        for seed in STAGE1_SEEDS
    ]
    stage0 = load_yaml(PROJECT_ROOT / "configs/stage1/stage0_checkpoint_selection_protocol.yaml")
    env_path = Path("configs/env_medium_v3_unified_id.yaml")
    return {
        "stage0_audit_source_sha256": stage0["source_manifest"]["sha256"],
        "stage0_hash_scope": "frozen_stage0_partial_manifest",
        "stage1_source_sha256": _manifest_sha256(STAGE1_SOURCE_FILES),
        "stage1_source_files": list(STAGE1_SOURCE_FILES),
        "stage1_config_sha256": {
            relative: _file_sha256(PROJECT_ROOT / relative) for relative in config_paths
        },
        "environment_config": str(env_path).replace("\\", "/"),
        "environment_config_sha256": _file_sha256(PROJECT_ROOT / env_path),
        **_git_metadata(),
    }


def stage3_source_provenance(execution_amendment: dict | None = None) -> dict:
    config_paths = [
        f"configs/stage3/train_stage3_{arm}_medium_seed_{seed}.yaml"
        for arm in STAGE3_ARMS
        for seed in STAGE3_SEEDS
    ]
    env_path = Path("configs/env_medium_v3_unified_id.yaml")
    freeze_path = Path("configs/stage3/r0_c0_freeze.sha256")
    amendment = load_stage3_formal_training_amendment()
    provenance = {
        "stage3_protocol_freeze_manifest": str(freeze_path).replace("\\", "/"),
        "stage3_protocol_freeze_sha256": _file_sha256(PROJECT_ROOT / freeze_path),
        "stage3_formal_training_amendment": STAGE3_FORMAL_AMENDMENT_PATH.as_posix(),
        "stage3_formal_training_amendment_sha256": _file_sha256(
            PROJECT_ROOT / STAGE3_FORMAL_AMENDMENT_PATH
        ),
        "stage3_formal_freeze_manifest": STAGE3_FORMAL_FREEZE_PATH.as_posix(),
        "stage3_formal_freeze_sha256": _file_sha256(
            PROJECT_ROOT / STAGE3_FORMAL_FREEZE_PATH
        ),
        "stage3_server_smoke_freeze_sha256": amendment["server_smoke_gate"]["freeze_sha256"],
        "stage3_source_sha256": _manifest_sha256(STAGE1_SOURCE_FILES),
        "stage3_source_files": list(STAGE1_SOURCE_FILES),
        "stage3_config_sha256": {
            relative: _file_sha256(PROJECT_ROOT / relative) for relative in config_paths
        },
        "environment_config": str(env_path).replace("\\", "/"),
        "environment_config_sha256": _file_sha256(PROJECT_ROOT / env_path),
        **_git_metadata(),
    }
    if execution_amendment is not None:
        provenance.update(
            {
                "stage3_execution_amendment": execution_amendment["_path"],
                "stage3_execution_amendment_sha256": execution_amendment["_sha256"],
                "stage3_parallel_smoke_freeze_manifest": STAGE3_PARALLEL_SMOKE_FREEZE_PATH.as_posix(),
                "stage3_parallel_smoke_freeze_sha256": _file_sha256(
                    PROJECT_ROOT / STAGE3_PARALLEL_SMOKE_FREEZE_PATH
                ),
                "confirmatory_eligible": False,
            }
        )
        if execution_amendment["_path"] == STAGE3_PARALLEL_FORMAL_AMENDMENT_PATH.as_posix():
            provenance.update(
                {
                    "stage3_parallel_server_smoke_freeze_manifest": STAGE3_PARALLEL_SERVER_SMOKE_FREEZE_PATH.as_posix(),
                    "stage3_parallel_server_smoke_freeze_sha256": _file_sha256(
                        PROJECT_ROOT / STAGE3_PARALLEL_SERVER_SMOKE_FREEZE_PATH
                    ),
                    "stage3_parallel_formal_freeze_manifest": STAGE3_PARALLEL_FORMAL_FREEZE_PATH.as_posix(),
                    "stage3_parallel_formal_freeze_sha256": _file_sha256(
                        PROJECT_ROOT / STAGE3_PARALLEL_FORMAL_FREEZE_PATH
                    ),
                }
            )
    return provenance


def runtime_environment_metadata(device: torch.device) -> dict:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available": cuda,
        "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn": None if not cuda else torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def validate_stage1_config_matrix() -> dict:
    configs: dict[tuple[str, int], dict] = {}
    outputs = set()
    for arm in STAGE1_ARMS:
        for seed in STAGE1_SEEDS:
            relative = f"configs/stage1/train_stage1_{arm}_medium_seed_{seed}.yaml"
            config = load_yaml(PROJECT_ROOT / relative)
            metadata = stage1_arm_metadata(config)
            if metadata["stage1_arm"] != arm.upper():
                raise ValueError(f"Stage-1 matrix arm mismatch: {relative}")
            if config.get("init_checkpoint") is not None:
                raise ValueError(f"Stage-1 runs must start from scratch: {relative}")
            output = config["output"]["directory"]
            if output in outputs:
                raise ValueError(f"Duplicate Stage-1 output directory: {output}")
            outputs.add(output)
            configs[(arm, seed)] = config

    def without_arm_fields(config: dict) -> dict:
        value = deepcopy(config)
        value.pop("observation")
        value["output"].pop("directory")
        for key in (
            "name", "stage1_arm", "risk_bias_mode", "risk_bias_scale",
            "risk_bias_clip", "goal_guidance",
        ):
            value["algorithm"].pop(key, None)
        return value

    for seed in STAGE1_SEEDS:
        reference = without_arm_fields(configs[("c0", seed)])
        for arm in STAGE1_ARMS[1:]:
            if without_arm_fields(configs[(arm, seed)]) != reference:
                raise ValueError(f"Stage-1 non-whitelisted config drift: arm={arm}, seed={seed}")

    for arm in STAGE1_ARMS:
        left = deepcopy(configs[(arm, STAGE1_SEEDS[0])])
        right = deepcopy(configs[(arm, STAGE1_SEEDS[1])])
        for value in (left, right):
            value.pop("seed")
            value["algorithm"].pop("name")
            value["output"].pop("directory")
        if left != right:
            raise ValueError(f"Stage-1 seed configs differ beyond seed/name/output: arm={arm}")
    v32c = load_yaml(
        PROJECT_ROOT / "configs/v3_2_tuning/train_ra_sac_v3_2c_medium_seed_18207.yaml"
    )
    reference_algorithm = v32c["algorithm"]
    for seed in STAGE1_SEEDS:
        d0_algorithm = configs[("d0", seed)]["algorithm"]
        for key in ("risk_bias_mode", "risk_bias_scale", "risk_bias_clip", "goal_guidance"):
            if d0_algorithm[key] != reference_algorithm[key]:
                raise ValueError(f"Stage-1 D0 does not match frozen v3.2-c {key}: seed={seed}")

    return {"status": "passed", "arms": 5, "training_seeds": 2, "runs": 10, "d0_v32c_match": True}


def validate_stage3_config_matrix() -> dict:
    configs: dict[tuple[str, int], dict] = {}
    outputs = set()
    for arm in STAGE3_ARMS:
        for seed in STAGE3_SEEDS:
            relative = f"configs/stage3/train_stage3_{arm}_medium_seed_{seed}.yaml"
            config = load_yaml(PROJECT_ROOT / relative)
            metadata = stage3_arm_metadata(config)
            if metadata["stage3_arm"] != arm.upper():
                raise ValueError(f"Stage-3 matrix arm mismatch: {relative}")
            output = config["output"]["directory"]
            if output in outputs:
                raise ValueError(f"Duplicate Stage-3 output directory: {output}")
            outputs.add(output)
            configs[(arm, seed)] = config

    def without_arm_fields(config: dict) -> dict:
        value = deepcopy(config)
        value["output"].pop("directory")
        for key in (
            "name", "stage3_arm", "risk_bias_mode", "risk_bias_scale", "risk_bias_clip",
        ):
            value["algorithm"].pop(key, None)
        return value

    for seed in STAGE3_SEEDS:
        if without_arm_fields(configs[("r0", seed)]) != without_arm_fields(configs[("c0", seed)]):
            raise ValueError(f"Stage-3 non-whitelisted config drift: seed={seed}")

    for arm in STAGE3_ARMS:
        reference = deepcopy(configs[(arm, STAGE3_SEEDS[0])])
        reference.pop("seed")
        reference["algorithm"].pop("name")
        reference["output"].pop("directory")
        for seed in STAGE3_SEEDS[1:]:
            candidate = deepcopy(configs[(arm, seed)])
            candidate.pop("seed")
            candidate["algorithm"].pop("name")
            candidate["output"].pop("directory")
            if candidate != reference:
                raise ValueError(
                    f"Stage-3 seed configs differ beyond seed/name/output: arm={arm}, seed={seed}"
                )

    return {
        "status": "passed",
        "arms": 2,
        "training_seeds": len(STAGE3_SEEDS),
        "runs": len(configs),
        "only_arm_difference": "risk_bias_off_vs_on",
    }


def risk_bias_audit(agent, *, mode: str, scale: float, clip: float | None) -> dict:
    encoders = {
        "actor": agent.actor.encoder,
        "critic1": agent.q1.encoder,
        "critic2": agent.q2.encoder,
        "target_critic1": agent.q1_target.encoder,
        "target_critic2": agent.q2_target.encoder,
    }
    expected = (mode, float(scale), None if clip is None else float(clip))
    for name, encoder in encoders.items():
        actual = (encoder.risk_bias_mode, float(encoder.risk_bias_scale), encoder.risk_bias_clip)
        if actual != expected:
            raise ValueError(f"Risk-bias path mismatch for {name}: actual={actual}, expected={expected}")
    enabled = mode != "none"
    return {
        "configured_enabled": enabled,
        **{f"{name}_enabled": encoder.risk_bias_mode != "none" for name, encoder in encoders.items()},
        "mode": mode,
        "scale": float(scale),
        "clip": None if clip is None else float(clip),
        "status": "passed",
    }


def actuator_audit_snapshot(
    info: dict,
    observation: dict,
    action_scale: np.ndarray,
    *,
    expect_reset: bool = False,
    expected_next: np.ndarray | None = None,
    expected_applied: np.ndarray | None = None,
) -> dict:
    state = info["actuator_state"]
    delay = int(state["control_delay_steps"])
    applied = np.asarray(state["current_applied_command"], dtype=np.float64)
    queue = np.asarray(state["pending_action_queue"], dtype=np.float64)
    mask = np.asarray(state["pending_queue_mask"], dtype=np.float64)
    expected_mask = np.array([1.0] * delay + [0.0] * (len(mask) - delay))
    arrays = (applied, queue, mask)
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Non-finite actuator audit value")
    if queue.shape != (len(mask), 2) or delay > len(mask):
        raise ValueError("Invalid actuator queue shape or delay")
    if not np.array_equal(mask, expected_mask):
        raise ValueError("Actuator queue mask does not match delay")
    if delay < len(mask) and not np.allclose(queue[delay:], 0.0):
        raise ValueError("Inactive actuator queue slots must be zero-filled")
    if not np.allclose(applied, info["applied_command"]):
        raise ValueError("Actuator audit command is not the command used by the transition")
    if expect_reset and (not np.allclose(applied, 0.0) or not np.allclose(queue, 0.0)):
        raise ValueError("Actuator state was not cleared by reset")
    if expected_next is not None and not np.allclose(queue[0], expected_next):
        raise ValueError("pending_action_queue[0] is not the next FIFO command")
    if expected_applied is not None and not np.allclose(applied, expected_applied):
        raise ValueError("Unexpected applied command")
    if delay == 0 and (mask.any() or queue.any()):
        raise ValueError("delay=0 must not expose pending commands")

    normalized = None
    if "current_applied_command" in observation:
        normalized_applied = np.asarray(observation["current_applied_command"])
        normalized_queue = np.asarray(observation["pending_action_queue"])
        normalized_mask = np.asarray(observation["pending_queue_mask"])
        if not all(np.isfinite(value).all() for value in (normalized_applied, normalized_queue, normalized_mask)):
            raise ValueError("Non-finite normalized actuator observation")
        if not np.allclose(normalized_applied * action_scale, applied, atol=1e-6):
            raise ValueError("Applied-command normalization mismatch")
        if not np.allclose(normalized_queue * action_scale, queue, atol=1e-6):
            raise ValueError("Queue normalization mismatch")
        normalized = {
            "current_applied_command": normalized_applied.tolist(),
            "pending_action_queue": normalized_queue.tolist(),
            "pending_queue_mask": normalized_mask.tolist(),
        }
    return {
        "control_delay_steps": delay,
        "current_applied_command": applied.tolist(),
        "pending_action_queue": queue.tolist(),
        "pending_queue_mask": mask.tolist(),
        "normalized_observation": normalized,
        "semantic_validation": "passed",
    }


def resolved_training_protocol(train_config: dict) -> dict:
    training = train_config["training"]
    seed = int(train_config["seed"])
    validation_seed = int(training.get("eval_seed", seed + 100000))
    validation_episodes = int(training["eval_episodes"])
    namespace = training.get("env_seed_namespace")

    evaluation_protocol = None
    protocol_path = train_config.get("evaluation_protocol")
    if protocol_path:
        missing = [
            key
            for key in ("eval_seed", "eval_episodes", "env_seed_namespace")
            if key not in training
        ]
        if missing:
            raise ValueError(f"Formal training config is missing required keys: {missing}")
        evaluation_protocol = load_yaml(PROJECT_ROOT / protocol_path)
        checkpoint_selection = evaluation_protocol["checkpoint_selection"]
        expected = (
            str(checkpoint_selection["env_config"]),
            int(checkpoint_selection["seed_start"]),
            int(checkpoint_selection["episodes"]),
            FORMAL_TRAINING_NAMESPACE,
        )
        actual = (
            str(train_config["env_config"]),
            validation_seed,
            validation_episodes,
            namespace,
        )
        if actual != expected:
            raise ValueError(
                "Training protocol does not match checkpoint_selection: "
                f"actual={actual}, expected={expected}"
            )

    environment_range = training_environment_seed_range(train_config)
    return {
        "algorithm_name": train_config.get("algorithm", {}).get("name", "sac_attention"),
        "observation": observation_config(train_config),
        "stage1_ablation": stage1_arm_metadata(train_config),
        "stage3_confirmation": stage3_arm_metadata(train_config),
        "training_seed": seed,
        "training_environment_namespace": namespace,
        "training_environment_seed_start": None if environment_range is None else environment_range[0],
        "training_environment_seed_end_upper_bound": None if environment_range is None else environment_range[1],
        "checkpoint_validation_seed_start": validation_seed,
        "checkpoint_validation_seed_end": validation_seed + validation_episodes - 1,
        "checkpoint_validation_episodes": validation_episodes,
        "development_screen_seed_start": (
            None if evaluation_protocol is None else int(evaluation_protocol["development_screen"]["seed_start"])
        ),
        "final_test_seed_start": (
            None if evaluation_protocol is None else int(evaluation_protocol["final_test"]["seed_start"])
        ),
        "output_directory": str(train_config["output"]["directory"]),
    }


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@preserve_global_rng_state
def evaluate(agent: SACAttentionAgent, env_config: dict, episodes: int, seed: int, eval_id: int) -> dict:
    rows = []
    for idx in range(episodes):
        env = UAV2DEnv(deepcopy(env_config))
        obs_dict = env.reset(seed=seed + idx)
        obs = attention_observation(obs_dict, env.world_size)
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            next_obs_dict, reward, done, info = env.step(action)
            obs = attention_observation(next_obs_dict, env.world_size)
            total_reward += reward
        rows.append(
            {
                "outcome": info["outcome"],
                "total_reward": total_reward,
                "steps": info["steps"],
                "path_length_m": info["path_length_m"],
                "fhp_count": info["fhp_count"],
            }
        )

    return summarize_eval_episodes(rows, eval_id=eval_id, eval_episodes=episodes)


def main() -> None:
    args = parse_args()
    train_config = load_yaml(PROJECT_ROOT / args.config)
    env_config = resolved_environment_config(
        load_yaml(PROJECT_ROOT / train_config["env_config"]), train_config
    )
    ensure_training_environment_allowed(env_config)
    protocol = resolved_training_protocol(train_config)
    execution_amendment = (
        None
        if args.execution_amendment is None
        else load_stage3_execution_amendment(args.execution_amendment)
    )
    cfg, execution = resolved_execution_config(
        train_config, args.smoke_steps, execution_amendment
    )
    formal_started_at = datetime.now(timezone.utc).isoformat()
    matrix_audit = None
    source_provenance = None
    source_hash_key = None
    if protocol["stage1_ablation"] is not None:
        matrix_audit, source_provenance = validate_stage1_config_matrix(), stage1_source_provenance()
        source_hash_key = "stage1_source_sha256"
    elif protocol["stage3_confirmation"] is not None:
        matrix_audit, source_provenance = (
            validate_stage3_config_matrix(),
            stage3_source_provenance(execution_amendment),
        )
        source_hash_key = "stage3_source_sha256"
    elif train_config.get("source_freeze_manifest"):
        source_manifest_path = PROJECT_ROOT / train_config["source_freeze_manifest"]
        if source_manifest_path.is_file():
            source_provenance = {
                "label": "hash-frozen source snapshot",
                "manifest_path": train_config["source_freeze_manifest"],
                "manifest_sha256": _file_sha256(source_manifest_path),
            }
        elif not args.check_config:
            raise FileNotFoundError(f"Required source freeze is missing: {source_manifest_path}")
    if matrix_audit is not None:
        execution["config_matrix_audit"] = matrix_audit
    print("Resolved training protocol:")
    print(yaml.safe_dump(protocol, sort_keys=False).strip())
    print("Resolved execution:")
    print(yaml.safe_dump(execution, sort_keys=False).strip())
    if source_provenance is not None:
        print("Source provenance:")
        print(yaml.safe_dump(source_provenance, sort_keys=False).strip())
    if args.check_config:
        return
    if args.smoke_steps is None:
        ensure_formal_training_authorized(train_config, execution_amendment)
    algorithm_config = train_config.get("algorithm", {})
    algorithm_name = algorithm_config.get("name", "sac_attention")
    risk_bias_mode = algorithm_config.get("risk_bias_mode", "mlp" if algorithm_config.get("use_risk_bias", False) else "none")
    risk_bias_scale = float(algorithm_config.get("risk_bias_scale", 1.0))
    risk_bias_clip_value = algorithm_config.get("risk_bias_clip")
    risk_bias_clip = None if risk_bias_clip_value is None else float(risk_bias_clip_value)
    goal_guidance = algorithm_config.get("goal_guidance", {})
    seed = int(train_config["seed"])
    init_checkpoint = args.init_checkpoint or train_config.get("init_checkpoint")
    if args.smoke_steps is not None and init_checkpoint:
        raise ValueError("Smoke tests must start from scratch; initialization checkpoints are forbidden")
    set_seed(seed)
    device = select_device(str(train_config["device"]))
    runtime_environment = runtime_environment_metadata(device)

    run_id = f"{timestamp()}_{algorithm_name}{'_smoke' if args.smoke_steps is not None else ''}"
    smoke_root = (
        Path("outputs/stage3_smoke")
        if protocol["stage3_confirmation"] is not None
        else Path("outputs/stage1_smoke")
    )
    output_directory = (
        smoke_root / algorithm_name
        if args.smoke_steps is not None
        else Path(train_config["output"]["directory"])
    )
    run_dir = PROJECT_ROOT / output_directory / run_id
    table_dir = run_dir / "tables"
    checkpoint_dir = run_dir / "checkpoints"
    table_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = UAV2DEnv(deepcopy(env_config))
    spec = attention_observation_spec(env)
    obs_dict = env.reset(seed=training_environment_seed(train_config, 0))
    obs = attention_observation(obs_dict, env.world_size)
    action_scale = np.array([env.a_max, env.omega_max], dtype=np.float32)
    actuator_samples = {}
    smoke_actions: list[np.ndarray] = []
    smoke_delay = int(env.control_delay_steps)
    if args.smoke_steps is not None:
        actuator_samples["after_reset"] = actuator_audit_snapshot(
            env.info(),
            obs_dict,
            action_scale,
            expect_reset=True,
        )
    eval_seed = int(protocol["checkpoint_validation_seed_start"])

    agent = SACAttentionAgent(
        obs_dim=spec["obs_dim"],
        global_dim=spec["global_dim"],
        max_tokens=spec["max_tokens"],
        token_dim=spec["token_dim"],
        act_dim=2,
        action_scale=action_scale,
        hidden_sizes=cfg["hidden_sizes"],
        embed_dim=int(cfg["embed_dim"]),
        risk_bias_mode=risk_bias_mode,
        risk_bias_scale=risk_bias_scale,
        risk_bias_clip=risk_bias_clip,
        risk_gate_boundary_norm=float(algorithm_config.get("risk_gate_boundary_norm", 0.06)),
        action_residual_config=algorithm_config.get("action_residual"),
        goal_guidance=goal_guidance,
        actor_lr=float(cfg["actor_lr"]),
        critic_lr=float(cfg["critic_lr"]),
        alpha_lr=float(cfg["alpha_lr"]),
        gamma=float(cfg["gamma"]),
        tau=float(cfg["tau"]),
        device=device,
    )
    risk_audit = risk_bias_audit(
        agent,
        mode=risk_bias_mode,
        scale=risk_bias_scale,
        clip=risk_bias_clip,
    )
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.is_absolute():
            init_path = PROJECT_ROOT / init_path
        agent.load_actor(str(init_path))
        print(f"Initialized actor from checkpoint: {init_path}")
    write_training_audit_records(
        run_dir,
        env_config=env_config,
        train_config=train_config,
        observation_spec=agent.observation_spec,
        execution=execution,
        artifact_directory=run_dir,
        source_provenance=source_provenance,
        runtime_environment=runtime_environment,
        risk_bias=risk_audit,
    )
    replay = ReplayBuffer(spec["obs_dim"], 2, int(cfg["replay_size"]))

    episode_log_path = table_dir / "episode_log.csv"
    eval_log_path = table_dir / "eval_log.csv"
    with episode_log_path.open("w", encoding="utf-8", newline="") as episode_file, eval_log_path.open(
        "w", encoding="utf-8", newline=""
    ) as eval_file:
        episode_writer = csv.DictWriter(
            episode_file,
            fieldnames=[
                *EPISODE_CORE_FIELDS,
                "actor_loss",
                "critic_loss",
                "alpha_loss",
                "alpha",
                "q_mean",
                "goal_guidance_loss",
                "goal_guidance_loss_computed",
                "safety_gate_mean",
                *RISK_DIAGNOSTIC_FIELDS,
                *ACTUATOR_AUDIT_FIELDS,
                "domain_parameters",
                "episode_randomization",
            ],
        )
        eval_writer = csv.DictWriter(
            eval_file,
            fieldnames=EVAL_FIELDNAMES,
        )
        episode_writer.writeheader()
        eval_writer.writeheader()

        episode = 0
        episode_reward = 0.0
        episode_steps = 0
        last_losses = None
        best_eval_key = None
        best_eval_metrics = None
        eval_id = 0
        update_counts = {
            "critic_updates": 0,
            "actor_updates": 0,
            "alpha_updates": 0,
            "target_updates": 0,
            "goal_guidance_loss_computations": 0,
        }
        progress = trange(
            1,
            int(cfg["total_steps"]) + 1,
            desc=f"{algorithm_name} train on {device}",
            mininterval=1.0,
        )
        for step in progress:
            action = env.sample_random_action() if step <= int(cfg["start_steps"]) else agent.act(obs, deterministic=False)
            if args.smoke_steps is not None and len(smoke_actions) < max(smoke_delay, 1):
                smoke_actions.append(np.asarray(action, dtype=np.float64).copy())
            next_obs_dict, reward, done, info = env.step(action)
            next_obs = attention_observation(next_obs_dict, env.world_size)
            if (
                args.smoke_steps is not None
                and "after_queue_fill" not in actuator_samples
                and len(smoke_actions) == max(smoke_delay, 1)
            ):
                actuator_samples["after_queue_fill"] = actuator_audit_snapshot(
                    info,
                    next_obs_dict,
                    action_scale,
                    expected_next=smoke_actions[0] if smoke_delay > 0 else None,
                    expected_applied=(
                        np.zeros(2) if smoke_delay > 0 else smoke_actions[-1]
                    ),
                )
            replay.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            if step >= int(cfg["update_after"]) and replay.size >= int(cfg["batch_size"]):
                update_every = int(cfg["update_every"])
                for update_index in range(update_every):
                    batch = replay.sample(int(cfg["batch_size"]), device)
                    losses = agent.update(
                        batch,
                        collect_metrics=done and update_index == update_every - 1,
                    )
                    for key in ("critic_updates", "actor_updates", "alpha_updates", "target_updates"):
                        update_counts[key] += 1
                    if agent.goal_guidance_loss_computed:
                        update_counts["goal_guidance_loss_computations"] += 1
                    if losses is not None:
                        last_losses = losses

            if step % int(cfg["eval_interval_steps"]) == 0:
                eval_id += 1
                eval_metrics = evaluate(agent, env_config, int(cfg["eval_episodes"]), eval_seed, eval_id)
                eval_metrics["eval_seed_start"] = eval_seed
                eval_metrics["eval_seed_end"] = eval_seed + int(cfg["eval_episodes"]) - 1
                eval_writer.writerow({"step": step, **eval_metrics})
                eval_file.flush()
                eval_key = eval_selection_key(eval_metrics, step=step, fhp_weight=0.02)
                eval_score = eval_key[0]
                if best_eval_key is None or eval_key > best_eval_key:
                    best_eval_key = eval_key
                    best_eval_metrics = {"step": step, "score": eval_score, **eval_metrics}
                    agent.save(str(checkpoint_dir / f"{algorithm_name}_best.pt"))
                progress.set_postfix(
                    success=f"{eval_metrics['eval_success_rate']:.2f}",
                    collision=f"{eval_metrics['eval_collision_rate']:.2f}",
                )

            if step % int(cfg["checkpoint_interval_steps"]) == 0:
                agent.save(str(checkpoint_dir / f"{algorithm_name}_step_{step}.pt"))

            if done:
                episode_writer.writerow(
                    make_episode_row(
                        step=step,
                        episode=episode,
                        episode_reward=episode_reward,
                        episode_steps=episode_steps,
                        outcome=info["outcome"],
                        extra={
                        "actor_loss": "" if last_losses is None else last_losses.actor_loss,
                        "critic_loss": "" if last_losses is None else last_losses.critic_loss,
                        "alpha_loss": "" if last_losses is None else last_losses.alpha_loss,
                        "alpha": "" if last_losses is None else last_losses.alpha,
                        "q_mean": "" if last_losses is None else last_losses.q_mean,
                        "goal_guidance_loss": "" if last_losses is None else last_losses.goal_guidance_loss,
                        "goal_guidance_loss_computed": agent.goal_guidance_loss_computed,
                        "safety_gate_mean": "" if last_losses is None else last_losses.safety_gate_mean,
                        **{
                            field: "" if last_losses is None else getattr(last_losses, field)
                            for field in RISK_DIAGNOSTIC_FIELDS
                        },
                        **actuator_audit_fields(info),
                        "domain_parameters": json.dumps(
                            info.get("domain_parameters", {}),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "episode_randomization": serialize_episode_randomization(
                            info["episode_randomization"]
                        ),
                        },
                    )
                )
                episode_file.flush()
                episode += 1
                obs = attention_observation(
                    env.reset(seed=training_environment_seed(train_config, episode)),
                    env.world_size,
                )
                episode_reward = 0.0
                episode_steps = 0

    if args.smoke_steps is not None:
        required_updates = ("critic_updates", "actor_updates", "alpha_updates", "target_updates")
        if any(update_counts[key] == 0 for key in required_updates):
            raise RuntimeError(f"Smoke test missed an update path: {update_counts}")
        expected_guidance_calls = (
            update_counts["actor_updates"] if agent.goal_guidance_loss_computed else 0
        )
        if update_counts["goal_guidance_loss_computations"] != expected_guidance_calls:
            raise RuntimeError("Goal-guidance computation count mismatch")
        if "after_queue_fill" not in actuator_samples:
            raise RuntimeError("Smoke test did not capture the queue-fill actuator sample")
        actuator_samples["final_step"] = actuator_audit_snapshot(
            info, next_obs_dict, action_scale
        )

    final_checkpoint = checkpoint_dir / f"{algorithm_name}_final.pt"
    agent.save(str(final_checkpoint))
    if args.smoke_steps is not None:
        checkpoint = torch.load(final_checkpoint, map_location="cpu")
        source_hash = source_provenance[source_hash_key]
        checkpoint.update(
            {
                "artifact_class": "smoke_checkpoint",
                "formal_training_eligible": False,
                "checkpoint_selection_eligible": False,
                "purpose": "preflight_only",
                "source_manifest_sha256": source_hash,
                source_hash_key: source_hash,
                "training_config_sha256": _file_sha256((PROJECT_ROOT / args.config).resolve()),
            }
        )
        torch.save(checkpoint, final_checkpoint)
        checkpoint = torch.load(final_checkpoint, map_location="cpu")
        validate_checkpoint_observation_spec(
            checkpoint,
            agent.observation_spec,
            checkpoint_label=str(final_checkpoint),
            allow_smoke=True,
        )
        incompatible_spec = dict(agent.observation_spec)
        incompatible_spec["obs_dim"] += 18 if incompatible_spec["obs_dim"] == 782 else -18
        incompatible_spec["global_dim"] += 18 if incompatible_spec["global_dim"] == 14 else -18
        try:
            validate_checkpoint_observation_spec(
                checkpoint,
                incompatible_spec,
                checkpoint_label=str(final_checkpoint),
                allow_smoke=True,
            )
        except ValueError:
            pass
        else:
            raise RuntimeError("Smoke checkpoint accepted an incompatible observation contract")
        try:
            validate_checkpoint_observation_spec(
                checkpoint, agent.observation_spec, checkpoint_label=str(final_checkpoint)
            )
        except ValueError:
            pass
        else:
            raise RuntimeError("Smoke checkpoint was accepted for formal use")

        with (run_dir / "smoke_result.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                {
                    "status": "passed",
                    "purpose": "preflight_only",
                    "formal_result_eligible": False,
                    "checkpoint_selection_eligible": False,
                    "planned_total_steps": int(cfg["total_steps"]),
                    "start_time": formal_started_at,
                    "end_time": datetime.now(timezone.utc).isoformat(),
                    "runtime_overrides": execution["runtime_overrides"],
                    "updates": update_counts,
                    "gradient_updates": update_counts["actor_updates"],
                    "episodes_completed": episode,
                    "final_checkpoint": str(final_checkpoint),
                    "checkpoint": {
                        "artifact_class": checkpoint["artifact_class"],
                        "homogeneous_observation_load": "passed",
                        "heterogeneous_observation_load": "rejected",
                        "formal_use": "rejected",
                    },
                    "observation_spec": agent.observation_spec,
                    "risk_bias": risk_audit,
                    "goal_guidance": {
                        "configured_enabled": agent.goal_guidance_enabled,
                        "lambda": agent.goal_guidance_lambda,
                        "loss_computations": update_counts["goal_guidance_loss_computations"],
                    },
                    "actuator_audit_samples": actuator_samples,
                    "config_matrix_audit": matrix_audit,
                    "execution": execution,
                    "source_provenance": source_provenance,
                    "runtime_environment": runtime_environment,
                },
                file,
                sort_keys=False,
            )
    if best_eval_metrics is not None:
        best_eval_path = table_dir / "best_eval.yaml"
        with best_eval_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(best_eval_metrics, file, sort_keys=False)
        print(f"Best eval: {best_eval_metrics}")
        print(f"Best checkpoint: {checkpoint_dir / f'{algorithm_name}_best.pt'}")
    if args.smoke_steps is None and protocol["stage3_confirmation"] is not None:
        with (run_dir / "formal_run_result.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                {
                    "status": "complete",
                    "artifact_class": "stage3_formal_training",
                    "formal_result_eligible": execution["formal_result_eligible"],
                    "confirmatory_eligible": execution["confirmatory_eligible"],
                    "checkpoint_selection_eligible": False,
                    "algorithm_name": algorithm_name,
                    "training_seed": seed,
                    "source_manifest_sha256": source_provenance[
                        "stage3_formal_freeze_sha256"
                    ],
                    "config_sha256": _file_sha256((PROJECT_ROOT / args.config).resolve()),
                    "environment_sha256": source_provenance["environment_config_sha256"],
                    "gpu": runtime_environment["gpu"],
                    "cpu": {
                        "platform": runtime_environment["platform"],
                        "machine": runtime_environment["machine"],
                        "count": runtime_environment["cpu_count"],
                    },
                    "start_time": formal_started_at,
                    "end_time": datetime.now(timezone.utc).isoformat(),
                    "exit_status": 0,
                    "completed_steps": int(cfg["total_steps"]),
                    "init_checkpoint_argument": args.init_checkpoint,
                    "resume_argument": None,
                },
                file,
                sort_keys=False,
            )
    print(f"Training run directory: {run_dir}")
    print(f"Episode log: {episode_log_path}")
    print(f"Eval log: {eval_log_path}")


if __name__ == "__main__":
    main()
