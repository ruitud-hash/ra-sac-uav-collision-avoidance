"""Prepare and execute the frozen Batch-3 Shield matrix."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs/publication_confirmation_v1/formal_freeze"
INVENTORY = FREEZE / "execution_asset_inventory.yaml"
EXTENSION = ROOT / "configs/publication_confirmation/deployment_safety_extension_v1.yaml"
OUTPUT_ROOT = "outputs/publication_confirmation_v1/formal_blind_batch3_shield_decomposition"
LOG_ROOT = "outputs/publication_confirmation_v1/formal_blind_batch3_shield_decomposition_logs"
METHODS = ("RA-SAC-v3.2-c", "SAC-Attention-v3")
SEEDS = (18207, 28207, 38207, 48207, 58207)
ENVIRONMENTS = {
    "medium_id": "configs/env_medium_v3_unified_id.yaml",
    "combined_ood": "configs/env_medium_v3_combined_ood.yaml",
}
MODES = ("static_boundary_only", "dynamic_ttc_only", "full_shield")
FROZEN_HASHES = {
    "configs/publication_confirmation/protocol.yaml": "4f251cba613cabe77826936033d2c876d4831faae73c65542bc2df9b57a4145c",
    "configs/publication_confirmation/implementation_manifest.yaml": "883f91d8ffc00aa61247aff0b5d425a3b3f9863f7ae58b9aa252751c62c4aee1",
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


def verify_frozen_inputs() -> dict:
    for relative, expected in FROZEN_HASHES.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen hash mismatch: {relative}")
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    if inventory.get("status") != "FROZEN_BEFORE_BLIND_OPENING":
        raise ValueError("Execution inventory is not frozen")
    for environment, config in ENVIRONMENTS.items():
        entry = inventory["environments"][environment]
        if entry["path"] != config or sha256(ROOT / config) != entry["sha256"]:
            raise ValueError(f"Environment inventory mismatch: {environment}")
    return inventory


def method_paths(method: str, seed: int, inventory: dict) -> tuple[str, str, str | None, str]:
    checkpoint = inventory["checkpoints"][method][seed]["path"]
    metadata = (ROOT / checkpoint).parent.parent / "protocol_metadata.yaml"
    if not metadata.is_file():
        raise ValueError(f"Missing training metadata: {metadata}")
    if method == "RA-SAC-v3.2-c":
        train_config = "configs/v3_2_tuning/train_ra_sac_v3_2c_medium_seed_18207.yaml"
        candidate = ROOT / f"configs/v3_2_tuning/train_ra_sac_v3_2c_medium_seed_{seed}.yaml"
        provenance = candidate.relative_to(ROOT).as_posix() if candidate.is_file() else None
    else:
        train_config = f"configs/train_sac_attention_v3_medium_seed_{seed}.yaml"
        provenance = train_config
    return checkpoint, metadata.relative_to(ROOT).as_posix(), provenance, train_config


def build_cells(inventory: dict) -> list[dict]:
    cells = []
    for environment, env_config in ENVIRONMENTS.items():
        for method in METHODS:
            for seed in SEEDS:
                checkpoint, metadata, provenance, train_config = method_paths(method, seed, inventory)
                for mode in MODES:
                    slug = method.lower().replace("-", "_").replace(".", "_")
                    cell_id = f"{environment}_{slug}_seed_{seed}_{mode}"
                    argv = [
                        sys.executable,
                        "-B",
                        "scripts/evaluate_sac_attention_checkpoint.py",
                        "--checkpoint",
                        checkpoint,
                        "--env-config",
                        env_config,
                        "--train-config",
                        train_config,
                        "--episodes",
                        "200",
                        "--seed",
                        "149000",
                        "--evaluation-role",
                        "publication_confirmation",
                        "--shield-mode",
                        mode,
                        "--method",
                        method,
                        "--training-metadata",
                        metadata,
                        "--checkpoint-step",
                        "1000000",
                        "--output-root",
                        OUTPUT_ROOT,
                        "--device",
                        "cpu",
                        "--progress-interval",
                        "20",
                    ]
                    if provenance:
                        argv.extend(("--training-config-provenance", provenance))
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "method": method,
                            "training_seed": seed,
                            "environment": environment,
                            "mode": mode,
                            "authorization": (
                                "policy_result_conditioned_extension"
                                if environment == "combined_ood" and mode != "full_shield"
                                else "original_frozen_protocol"
                            ),
                            "checkpoint_sha256": inventory["checkpoints"][method][seed]["sha256"],
                            "environment_sha256": inventory["environments"][environment]["sha256"],
                            "argv": argv,
                        }
                    )
    if len(cells) != 60 or len({cell["cell_id"] for cell in cells}) != 60:
        raise AssertionError("Batch-3 matrix must contain 60 unique new cells")
    if sum(cell["authorization"] == "original_frozen_protocol" for cell in cells) != 40:
        raise AssertionError("Expected 40 originally authorized cells")
    return cells


def prepare(manifest_path: Path) -> None:
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if (ROOT / OUTPUT_ROOT).exists() or (ROOT / LOG_ROOT).exists():
        raise FileExistsError("Batch-3 output or log root already exists")
    inventory = verify_frozen_inputs()
    cells = build_cells(inventory)
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_EXECUTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "batch3_shield_decomposition",
        "scientific_result_inspected": False,
        "extension_protocol_path": str(EXTENSION.relative_to(ROOT).as_posix()),
        "extension_protocol_sha256": sha256(EXTENSION),
        "runner_sha256": sha256(Path(__file__)),
        "execution_inventory_sha256": sha256(INVENTORY),
        "policy_only_reference_allowlist_sha256": "208adfb2c39fba7945d63bd61db9502da6d2aca265da3096090bb1e0c996a514",
        "new_cell_count": len(cells),
        "new_episode_count": 12000,
        "policy_only_reference_cells": 20,
        "complete_analysis_cell_count": 80,
        "output_root": OUTPUT_ROOT,
        "log_root": LOG_ROOT,
        "cells": cells,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8", newline="\n")
    print(f"PREPARE: PASS ({len(cells)} cells; original=40, extension=20)")


def run_cell(cell: dict, log_root: Path, timeout_seconds: int) -> dict:
    stdout_path = log_root / f"{cell['cell_id']}.stdout.log"
    stderr_path = log_root / f"{cell['cell_id']}.stderr.log"
    started = datetime.now(timezone.utc).isoformat()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                cell["argv"],
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                timeout=None if timeout_seconds == 0 else timeout_seconds,
                check=False,
            )
        status, returncode = ("COMPLETE", completed.returncode) if completed.returncode == 0 else ("FAILED", completed.returncode)
    except subprocess.TimeoutExpired:
        status, returncode = "TIMEOUT", None
    return {
        "cell_id": cell["cell_id"],
        "status": status,
        "returncode": returncode,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_log": str(stdout_path.relative_to(ROOT).as_posix()),
        "stderr_log": str(stderr_path.relative_to(ROOT).as_posix()),
    }


def execute(manifest_path: Path, max_workers: int, timeout_seconds: int) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_EXECUTION" or manifest.get("new_cell_count") != 60:
        raise ValueError("Execution manifest is not authorized")
    if manifest["runner_sha256"] != sha256(Path(__file__)) or manifest["extension_protocol_sha256"] != sha256(EXTENSION):
        raise ValueError("Runner or extension protocol changed after manifest freeze")
    verify_frozen_inputs()
    output_root, log_root = ROOT / manifest["output_root"], ROOT / manifest["log_root"]
    if output_root.exists() or log_root.exists():
        raise FileExistsError("Batch-3 output or log root already exists")
    log_root.mkdir(parents=True)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_cell, cell, log_root, timeout_seconds): cell for cell in manifest["cells"]}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{len(results):02d}/60 {result['cell_id']}: {result['status']}", flush=True)
    status_path = log_root / "batch3_launcher_status.yaml"
    status_path.write_text(
        yaml.safe_dump(
            {
                "status": "COMPLETE" if all(item["status"] == "COMPLETE" for item in results) else "INCIDENT",
                "max_workers": max_workers,
                "timeout_seconds": timeout_seconds,
                "cells": sorted(results, key=lambda item: item["cell_id"]),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )


def self_test() -> None:
    cells = build_cells(verify_frozen_inputs())
    assert len(cells) == 60
    assert sum(cell["authorization"] == "policy_result_conditioned_extension" for cell in cells) == 20
    assert all("policy_only" not in cell["cell_id"] for cell in cells)
    print("SELF_TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=0, help="0 disables the external wall-clock timeout")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.prepare:
        prepare((ROOT / args.prepare).resolve() if not args.prepare.is_absolute() else args.prepare)
    elif args.run:
        if args.max_workers < 1 or args.timeout_seconds < 0:
            parser.error("--max-workers must be positive and --timeout-seconds non-negative")
        execute((ROOT / args.run).resolve() if not args.run.is_absolute() else args.run, args.max_workers, args.timeout_seconds)
    else:
        parser.error("choose --self-test, --prepare, or --run")


if __name__ == "__main__":
    main()
