"""Prepare and execute frozen broad-baseline publication extensions."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs/publication_confirmation_v1/formal_freeze"
INVENTORY = FREEZE / "execution_asset_inventory.yaml"
PROTOCOL = ROOT / "configs/publication_confirmation/broad_baseline_extension_v1.yaml"
EVALUATOR = "scripts/evaluate_publication_mlp_baseline.py"
METHODS = ("SAC-MLP", "TD3-MLP", "PPO-MLP")
SEEDS = (18207, 28207, 38207, 48207, 58207)
BATCHES = {
    "batch4": {
        "mode": "policy_only",
        "environments": ("medium_id", "disturbance_ood", "behavior_ood", "composition_ood", "combined_ood"),
        "output_root": "outputs/publication_confirmation_v1/formal_extension_batch4_broad_baseline_policy_only",
        "log_root": "outputs/publication_confirmation_v1/formal_extension_batch4_broad_baseline_policy_only_logs",
    },
    "batch5": {
        "mode": "full_shield",
        "environments": ("medium_id", "combined_ood"),
        "output_root": "outputs/publication_confirmation_v1/formal_extension_batch5_broad_baseline_full_shield",
        "log_root": "outputs/publication_confirmation_v1/formal_extension_batch5_broad_baseline_full_shield_logs",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> tuple[dict, dict]:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_BASELINE_RESULT_OPENING":
        raise ValueError("Broad-baseline extension protocol is not frozen")
    for relative, expected in protocol["frozen_source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen source mismatch: {relative}")
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    if inventory.get("status") != "FROZEN_BEFORE_BLIND_OPENING":
        raise ValueError("Checkpoint inventory is not frozen")
    for method in METHODS:
        for seed in SEEDS:
            entry = inventory["checkpoints"][method][seed]
            if sha256(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Checkpoint mismatch: {method} {seed}")
    for environment in set(BATCHES["batch4"]["environments"]):
        entry = inventory["environments"][environment]
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Environment mismatch: {environment}")
    return protocol, inventory


def method_files(method: str, seed: int, inventory: dict) -> tuple[str, str, str, int]:
    checkpoint = inventory["checkpoints"][method][seed]["path"]
    metadata = (ROOT / checkpoint).parent.parent / "protocol_metadata.yaml"
    if method == "SAC-MLP":
        config = f"configs/train_sac_mlp_v3_medium_seed_{seed}.yaml"
        step = 1_000_000
    elif method == "TD3-MLP":
        config = f"configs/train_td3_mlp_v3_medium_seed_{seed}.yaml"
        step = 1_000_000
    else:
        config = f"configs/train_ppo_mlp_v3_medium_seed_{seed}.yaml"
        step = 1_003_520
    if not metadata.is_file() or not (ROOT / config).is_file():
        raise FileNotFoundError(f"Missing provenance for {method} seed {seed}")
    metadata_data = yaml.safe_load(metadata.read_text(encoding="utf-8"))
    if metadata_data.get("stage") != "training" or int(metadata_data["training_seed"]) != seed:
        raise ValueError(f"Training metadata mismatch: {method} seed {seed}")
    if int(yaml.safe_load((ROOT / config).read_text(encoding="utf-8"))["seed"]) != seed:
        raise ValueError(f"Training config mismatch: {method} seed {seed}")
    return checkpoint, metadata.relative_to(ROOT).as_posix(), config, step


def build_cells(batch: str, inventory: dict) -> list[dict]:
    spec = BATCHES[batch]
    cells = []
    for environment in spec["environments"]:
        env = inventory["environments"][environment]
        for method in METHODS:
            for seed in SEEDS:
                checkpoint, metadata, config, step = method_files(method, seed, inventory)
                slug = method.lower().replace("-", "_")
                cell_id = f"{environment}_{slug}_seed_{seed}_{spec['mode']}"
                cells.append({
                    "cell_id": cell_id,
                    "method": method,
                    "training_seed": seed,
                    "environment": environment,
                    "mode": spec["mode"],
                    "authorization": "original_frozen_protocol" if environment == "medium_id" else "post_opening_prospective_extension",
                    "checkpoint_sha256": inventory["checkpoints"][method][seed]["sha256"],
                    "training_config_sha256": sha256(ROOT / config),
                    "training_metadata_sha256": sha256(ROOT / metadata),
                    "environment_sha256": env["sha256"],
                    "argv": [
                        "python", "-B", EVALUATOR,
                        "--method", method,
                        "--checkpoint", checkpoint,
                        "--checkpoint-step", str(step),
                        "--train-config", config,
                        "--training-metadata", metadata,
                        "--env-config", env["path"],
                        "--training-seed", str(seed),
                        "--episodes", "200",
                        "--seed", "149000",
                        "--shield-mode", spec["mode"],
                        "--output-root", spec["output_root"],
                        "--device", "cpu",
                        "--progress-interval", "20",
                    ],
                })
    expected = 75 if batch == "batch4" else 30
    if len(cells) != expected or len({cell["cell_id"] for cell in cells}) != expected:
        raise AssertionError(f"{batch} matrix must contain {expected} unique cells")
    return cells


def prepare(batch: str, manifest_path: Path) -> None:
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    spec = BATCHES[batch]
    if (ROOT / spec["output_root"]).exists() or (ROOT / spec["log_root"]).exists():
        raise FileExistsError(f"{batch} output or log root already exists")
    protocol, inventory = verify_inputs()
    cells = build_cells(batch, inventory)
    original = sum(cell["authorization"] == "original_frozen_protocol" for cell in cells)
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_EXECUTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": batch,
        "scientific_result_inspected": False,
        "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(Path(__file__)),
        "evaluator_sha256": sha256(ROOT / EVALUATOR),
        "execution_inventory_sha256": sha256(INVENTORY),
        "cell_count": len(cells),
        "episode_count": len(cells) * 200,
        "original_protocol_cells": original,
        "post_opening_extension_cells": len(cells) - original,
        "output_root": spec["output_root"],
        "log_root": spec["log_root"],
        "cells": cells,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8", newline="\n")
    print(f"PREPARE {batch}: PASS ({len(cells)} cells, {len(cells) * 200} episodes)")


def run_cell(cell: dict, log_root: Path, timeout_seconds: int) -> dict:
    stdout_path = log_root / f"{cell['cell_id']}.stdout.log"
    stderr_path = log_root / f"{cell['cell_id']}.stderr.log"
    started = datetime.now(timezone.utc).isoformat()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                cell["argv"], cwd=ROOT, stdout=stdout, stderr=stderr,
                timeout=None if timeout_seconds == 0 else timeout_seconds, check=False,
            )
        status = "COMPLETE" if completed.returncode == 0 else "FAILED"
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        status, returncode = "TIMEOUT", None
    return {
        "cell_id": cell["cell_id"], "status": status, "returncode": returncode,
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_log": stdout_path.relative_to(ROOT).as_posix(),
        "stderr_log": stderr_path.relative_to(ROOT).as_posix(),
    }


def execute(manifest_path: Path, max_workers: int, timeout_seconds: int) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    batch = manifest.get("experiment")
    if batch not in BATCHES or manifest.get("status") != "FROZEN_BEFORE_EXECUTION":
        raise ValueError("Execution manifest is not authorized")
    protocol, inventory = verify_inputs()
    if manifest["protocol_sha256"] != sha256(PROTOCOL) or manifest["runner_sha256"] != sha256(Path(__file__)):
        raise ValueError("Protocol or runner changed after manifest freeze")
    if manifest["evaluator_sha256"] != sha256(ROOT / EVALUATOR):
        raise ValueError("Evaluator changed after manifest freeze")
    if manifest["cells"] != build_cells(batch, inventory):
        raise ValueError("Manifest cells no longer match frozen inputs")
    output_root, log_root = ROOT / manifest["output_root"], ROOT / manifest["log_root"]
    if output_root.exists() or log_root.exists():
        raise FileExistsError(f"{batch} output or log root already exists")
    log_root.mkdir(parents=True)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_cell, cell, log_root, timeout_seconds): cell for cell in manifest["cells"]}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{len(results):02d}/{manifest['cell_count']} {result['cell_id']}: {result['status']}", flush=True)
    (log_root / f"{batch}_launcher_status.yaml").write_text(
        yaml.safe_dump({
            "status": "COMPLETE" if all(item["status"] == "COMPLETE" for item in results) else "INCIDENT",
            "max_workers": max_workers,
            "timeout_seconds": timeout_seconds,
            "cells": sorted(results, key=lambda item: item["cell_id"]),
        }, sort_keys=False), encoding="utf-8", newline="\n",
    )


def preflight_load() -> None:
    _, inventory = verify_inputs()
    for method in METHODS:
        for seed in SEEDS:
            checkpoint, metadata, config, step = method_files(method, seed, inventory)
            command = [
                "python", "-B", EVALUATOR,
                "--method", method, "--checkpoint", checkpoint, "--checkpoint-step", str(step),
                "--train-config", config, "--training-metadata", metadata,
                "--env-config", inventory["environments"]["medium_id"]["path"],
                "--training-seed", str(seed), "--episodes", "200", "--seed", "149000",
                "--shield-mode", "policy_only", "--output-root", "outputs/publication_confirmation_v1/preflight_unused",
                "--device", "cpu", "--load-only",
            ]
            subprocess.run(command, cwd=ROOT, check=True)
    print("LOAD PREFLIGHT: PASS (15 checkpoints)")


def self_test() -> None:
    _, inventory = verify_inputs()
    assert len(build_cells("batch4", inventory)) == 75
    assert len(build_cells("batch5", inventory)) == 30
    assert sum(cell["authorization"] == "original_frozen_protocol" for cell in build_cells("batch4", inventory)) == 15
    assert sum(cell["authorization"] == "original_frozen_protocol" for cell in build_cells("batch5", inventory)) == 15
    print("SELF_TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", choices=tuple(BATCHES))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--max-workers", type=int, default=14)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--preflight-load", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.preflight_load:
        preflight_load()
    elif args.prepare:
        if args.manifest is None:
            parser.error("--prepare requires --manifest")
        prepare(args.prepare, ROOT / args.manifest if not args.manifest.is_absolute() else args.manifest)
    elif args.run:
        if args.max_workers < 1 or args.timeout_seconds < 0:
            parser.error("invalid worker or timeout setting")
        execute(ROOT / args.run if not args.run.is_absolute() else args.run, args.max_workers, args.timeout_seconds)
    else:
        parser.error("choose --self-test, --preflight-load, --prepare, or --run")


if __name__ == "__main__":
    main()
