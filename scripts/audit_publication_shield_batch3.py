"""Fail-closed integrity audit for the Batch-3 Shield dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("RA-SAC-v3.2-c", "SAC-Attention-v3")
SEEDS = (18207, 28207, 38207, 48207, 58207)
ENVIRONMENTS = {
    "env_medium_v3_unified_id.yaml": "medium_id",
    "env_medium_v3_combined_ood.yaml": "combined_ood",
}
MODES = ("static_boundary_only", "dynamic_ttc_only", "full_shield")
OUTCOMES = ("success", "dynamic_collision", "static_collision", "timeout", "out_of_bounds")
EXPECTED_EVAL_SEEDS = set(range(149000, 149200))
FROZEN_HASHES = {
    "configs/publication_confirmation/protocol.yaml": "4f251cba613cabe77826936033d2c876d4831faae73c65542bc2df9b57a4145c",
    "configs/publication_confirmation/implementation_manifest.yaml": "883f91d8ffc00aa61247aff0b5d425a3b3f9863f7ae58b9aa252751c62c4aee1",
    "configs/publication_confirmation/deployment_safety_extension_v1.yaml": "cf9c584396e6e2635f8c26dc3020c4190a88024c9ff07b415b572e9f7d069266",
    "scripts/evaluate_sac_attention_checkpoint.py": "61bc38c5a1d076bb7181c9311a6ececd4f81a6123fc7151bac90496df5a590b1",
    "envs/uav_2d_env.py": "d34f655d1f86aca474919d56519a55139e901d09e845a1ee30f79659e4db0130",
    "utils/action_safety.py": "5f7266399b4068525e4079dd364cf0d3f768ce664ef313c41b0c12c427e5685d",
    "utils/artifact_audit.py": "c93085f6b80ca7e6afeaa74b7b08fe19d79753898d88334125613ff7fff62db2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def mode_flags(mode: str) -> dict[str, bool]:
    return {
        "enabled": True,
        "static_boundary_enabled": mode in {"static_boundary_only", "full_shield"},
        "dynamic_enabled": mode in {"dynamic_ttc_only", "full_shield"},
    }


def logical_cell_id(environment: str, method: str, seed: int, mode: str) -> str:
    slug = method.lower().replace("-", "_").replace(".", "_")
    return f"{environment}_{slug}_seed_{seed}_{mode}"


def audit(args: argparse.Namespace) -> tuple[dict, dict]:
    for relative, expected_hash in FROZEN_HASHES.items():
        assert sha256(ROOT / relative) == expected_hash, f"Frozen hash mismatch: {relative}"

    inventory_path = ROOT / "outputs/publication_confirmation_v1/formal_freeze/execution_asset_inventory.yaml"
    inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
    for method in METHODS:
        for seed in SEEDS:
            item = inventory["checkpoints"][method][seed]
            assert sha256(ROOT / item["path"]) == item["sha256"], f"Checkpoint mismatch: {method}/{seed}"

    launcher = yaml.safe_load(args.launcher_status.read_text(encoding="utf-8"))
    recovery = yaml.safe_load(args.recovery_status.read_text(encoding="utf-8"))
    launch_counts = Counter(item["status"] for item in launcher["cells"])
    assert launcher["status"] == "INCIDENT" and launch_counts == {"COMPLETE": 46, "FAILED": 14}
    assert recovery["status"] == "COMPLETE"
    assert recovery["scientific_arguments_unchanged"] is True
    assert recovery["partial_outputs_reused"] is False
    assert recovery["concatenation_performed"] is False
    recovery_ids = {item["cell_id"] for item in recovery["cells"]}
    failed_ids = {item["cell_id"] for item in launcher["cells"] if item["status"] != "COMPLETE"}
    assert recovery_ids == failed_ids and len(recovery_ids) == 14
    assert all(item["status"] == "COMPLETE" and item["returncode"] == 0 for item in recovery["cells"])

    expected_keys = {
        (method, environment, seed, mode)
        for method in METHODS
        for environment in ENVIRONMENTS.values()
        for seed in SEEDS
        for mode in MODES
    }
    complete: dict[tuple[str, str, int, str], dict] = {}
    scenario_reference: dict[tuple[str, int], object] = {}
    completed_dirs = set()

    for summary_path in args.input_root.glob("*/tables/summary.yaml"):
        data = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
        environment = ENVIRONMENTS.get(Path(data["environment_config_path"]).name)
        key = (data["method"], environment, int(data["training_seed"]), data["evaluation_mode"])
        assert key in expected_keys, f"Unexpected completed cell: {key}"
        assert key not in complete, f"Duplicate completed cell: {key}"
        assert data["training_seed_source"] == "training_metadata"
        assert data["evaluation_seed_start"] == 149000 and data["evaluation_seed_end"] == 149199
        assert data["episodes"] == 200 and data["checkpoint_step"] == 1_000_000
        checkpoint = inventory["checkpoints"][key[0]][key[2]]
        assert data["checkpoint_sha256"] == checkpoint["sha256"]
        env_item = inventory["environments"][key[1]]
        assert data["environment_config_sha256"] == env_item["sha256"]
        assert data["evaluation_protocol"]["role"] == "publication_confirmation"
        assert {name: bool(data["action_safety_filter"][name]) for name in mode_flags(key[3])} == mode_flags(key[3])
        assert finite_tree(data), f"NaN/Inf in summary: {summary_path}"

        episode_path = summary_path.with_name("episodes.csv")
        assert episode_path.is_file()
        with episode_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        seeds = [int(row["seed"]) for row in rows]
        assert len(rows) == 200 and len(set(seeds)) == 200 and set(seeds) == EXPECTED_EVAL_SEEDS
        outcomes = Counter(row["outcome"] for row in rows)
        assert set(outcomes) <= set(OUTCOMES) and sum(outcomes.values()) == 200
        for row in rows:
            outcome = row["outcome"]
            flags = {
                "success": int(row["success"]),
                "dynamic_collision": int(row["dynamic_collision"]),
                "static_collision": int(row["static_collision"]),
                "timeout": int(row["timeout"]),
                "out_of_bounds": int(row["oob"]),
            }
            assert flags[outcome] == 1 and sum(flags.values()) == 1
            assert int(row["collision"]) == flags["dynamic_collision"] + flags["static_collision"]
            for field, value in row.items():
                if field and value and field not in {
                    "outcome", "applied_command", "pending_action_queue", "pending_queue_mask",
                    "domain_parameters", "dynamic_type_counts", "dynamic_scenario_counts", "episode_randomization",
                }:
                    try:
                        assert math.isfinite(float(value))
                    except ValueError:
                        pass
            scenario_key = (key[1], int(row["seed"]))
            scenario = json.loads(row["episode_randomization"])
            if scenario_key in scenario_reference:
                assert scenario_reference[scenario_key] == scenario, f"Scenario mismatch: {scenario_key}"
            else:
                scenario_reference[scenario_key] = scenario

        assert data["outcomes"] == dict(outcomes)
        assert math.isclose(data["collision_rate"], (outcomes["dynamic_collision"] + outcomes["static_collision"]) / 200)
        assert math.isclose(data["safety_failure_rate"], (outcomes["dynamic_collision"] + outcomes["static_collision"] + outcomes["out_of_bounds"]) / 200)
        root_artifacts = {path.name for path in summary_path.parents[1].iterdir()}
        for required in (
            "checkpoint_compatibility_record.yaml", "evaluation_provenance.yaml",
            "protocol_metadata.yaml", "seed_overlap_check_record.yaml",
        ):
            assert required in root_artifacts

        cell_id = logical_cell_id(key[1], key[0], key[2], key[3])
        log_root = args.recovery_logs if cell_id in recovery_ids else args.original_logs
        stdout = log_root / f"{cell_id}.stdout.log"
        stderr = log_root / f"{cell_id}.stderr.log"
        assert stdout.is_file() and stderr.is_file() and stderr.stat().st_size == 0
        directory = summary_path.parents[1]
        completed_dirs.add(directory.resolve())
        complete[key] = {
            "method": key[0],
            "environment": key[1],
            "training_seed": key[2],
            "mode": key[3],
            "execution_source": "recovery" if cell_id in recovery_ids else "original",
            "directory": directory.relative_to(ROOT).as_posix(),
            "summary_sha256": sha256(summary_path),
            "episodes_sha256": sha256(episode_path),
            "stdout_sha256": sha256(stdout),
            "stderr_sha256": sha256(stderr),
        }

    assert set(complete) == expected_keys and len(complete) == 60
    all_dirs = {path.resolve() for path in args.input_root.iterdir() if path.is_dir()}
    excluded_dirs = sorted(all_dirs - completed_dirs)
    assert len(excluded_dirs) == 14
    for directory in excluded_dirs:
        assert not (directory / "tables/summary.yaml").exists()
        assert not (directory / "tables/episodes.csv").exists()

    for log_root in (args.original_logs, args.recovery_logs):
        assert all(path.stat().st_size == 0 for path in log_root.glob("*.stderr.log"))

    p0_allowlist = ROOT / "outputs/publication_confirmation_v1/formal_freeze/authorized_scientific_cells.yaml"
    assert sha256(p0_allowlist) == "208adfb2c39fba7945d63bd61db9502da6d2aca265da3096090bb1e0c996a514"
    p0 = yaml.safe_load(p0_allowlist.read_text(encoding="utf-8"))
    p0_cells = [cell for cell in p0["cells"] if cell["environment"] in {"medium_id", "combined_ood"} and cell["method"] in METHODS]
    assert len(p0_cells) == 20

    allowlist = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN",
        "purpose": "exclusive Batch-3 Shield scientific-analysis allowlist",
        "cell_count": 60,
        "policy_only_reference_cell_count": 20,
        "complete_four_mode_analysis_cell_count": 80,
        "excluded_failed_directory_count": 14,
        "cells": [complete[key] for key in sorted(complete)],
    }
    audit_record = {
        "schema_version": 1,
        "status": "PASS",
        "audit_role": "Batch-3 Shield integrity and authorization",
        "performance_interpretation_performed": False,
        "new_cell_count": 60,
        "new_episode_count": 12000,
        "policy_only_reference_cells": 20,
        "complete_analysis_cells": 80,
        "original_complete_cells": 46,
        "recovery_complete_cells": 14,
        "excluded_failed_directories": [path.relative_to(ROOT).as_posix() for path in excluded_dirs],
        "checks": {
            "logical_matrix_exact": True,
            "episodes_and_seed_coverage_exact": True,
            "checkpoint_environment_and_source_hashes_match": True,
            "shield_mode_flags_match": True,
            "terminal_outcomes_conserve": True,
            "collision_and_safety_identities_match": True,
            "paired_scenario_randomization_match": True,
            "nan_or_inf_count": 0,
            "nonempty_stderr_count": 0,
            "partial_outputs_reused": False,
            "failed_directories_excluded": True,
        },
        "limitations": [
            "Full-shield intervention source cannot be separated because the frozen evaluator records total modifications only.",
            "The 20 Combined-OOD isolated-layer cells remain a policy-result-conditioned prospective extension.",
        ],
        "audit_script_sha256": sha256(Path(__file__)),
        "execution_inventory_sha256": sha256(inventory_path),
        "server_execution_manifest_sha256": sha256(args.execution_manifest),
        "launcher_status_sha256": sha256(args.launcher_status),
        "recovery_status_sha256": sha256(args.recovery_status),
        "policy_only_reference_allowlist_sha256": sha256(p0_allowlist),
    }
    return audit_record, allowlist


def self_test() -> None:
    assert mode_flags("static_boundary_only") == {"enabled": True, "static_boundary_enabled": True, "dynamic_enabled": False}
    assert mode_flags("dynamic_ttc_only") == {"enabled": True, "static_boundary_enabled": False, "dynamic_enabled": True}
    assert len({(m, e, s, q) for m in METHODS for e in ENVIRONMENTS.values() for s in SEEDS for q in MODES}) == 60
    print("SELF_TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--original-logs", type=Path)
    parser.add_argument("--recovery-logs", type=Path)
    parser.add_argument("--launcher-status", type=Path)
    parser.add_argument("--recovery-status", type=Path)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--allowlist-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = ("input_root", "original_logs", "recovery_logs", "launcher_status", "recovery_status", "execution_manifest", "audit_output", "allowlist_output")
    if any(getattr(args, name) is None for name in required):
        parser.error("all audit paths are required")
    for name in required:
        path = getattr(args, name)
        setattr(args, name, (ROOT / path).resolve() if not path.is_absolute() else path)
    if args.audit_output.exists() or args.allowlist_output.exists():
        raise FileExistsError("Audit output already exists")
    audit_record, allowlist = audit(args)
    args.audit_output.write_text(yaml.safe_dump(audit_record, sort_keys=False), encoding="utf-8", newline="\n")
    args.allowlist_output.write_text(yaml.safe_dump(allowlist, sort_keys=False), encoding="utf-8", newline="\n")
    print("BATCH3 INTEGRITY AUDIT: PASS (60 cells, 12000 episodes; 14 failed directories excluded)")


if __name__ == "__main__":
    main()
