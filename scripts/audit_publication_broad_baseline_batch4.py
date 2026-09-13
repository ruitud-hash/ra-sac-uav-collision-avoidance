"""Fail-closed integrity audit for the 75-cell Batch-4 baseline extension."""

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
METHODS = ("SAC-MLP", "TD3-MLP", "PPO-MLP")
ENVIRONMENTS = ("medium_id", "disturbance_ood", "behavior_ood", "composition_ood", "combined_ood")
SEEDS = (18207, 28207, 38207, 48207, 58207)
OUTCOMES = ("success", "dynamic_collision", "static_collision", "timeout", "out_of_bounds")
EVAL_SEEDS = set(range(149000, 149200))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def load_reference_cells(allowlist: Path) -> list[dict]:
    data = yaml.safe_load(allowlist.read_text(encoding="utf-8"))
    assert data["status"] == "FROZEN" and data["cell_count"] == len(data["cells"])
    return data["cells"]


def audit(args: argparse.Namespace) -> tuple[dict, dict]:
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    launcher = yaml.safe_load(args.launcher_status.read_text(encoding="utf-8"))
    inventory_path = ROOT / "outputs/publication_confirmation_v1/formal_freeze/execution_asset_inventory.yaml"
    inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
    protocol = ROOT / "configs/publication_confirmation/broad_baseline_extension_v1.yaml"
    evaluator = ROOT / "scripts/evaluate_publication_mlp_baseline.py"
    runner = ROOT / "scripts/run_publication_broad_baselines.py"
    assert manifest["status"] == "FROZEN_BEFORE_EXECUTION" and manifest["experiment"] == "batch4"
    assert manifest["cell_count"] == 75 and manifest["episode_count"] == 15000
    assert manifest["protocol_sha256"] == sha256(protocol)
    assert manifest["evaluator_sha256"] == sha256(evaluator)
    assert manifest["runner_sha256"] == sha256(runner)
    assert manifest["execution_inventory_sha256"] == sha256(inventory_path)
    assert launcher["status"] == "COMPLETE" and len(launcher["cells"]) == 75
    assert all(item["status"] == "COMPLETE" and item["returncode"] == 0 for item in launcher["cells"])
    launch_by_id = {item["cell_id"]: item for item in launcher["cells"]}
    assert len(launch_by_id) == 75 and set(launch_by_id) == {item["cell_id"] for item in manifest["cells"]}

    expected = {(method, environment, seed) for method in METHODS for environment in ENVIRONMENTS for seed in SEEDS}
    manifest_by_key = {(item["method"], item["environment"], int(item["training_seed"])): item for item in manifest["cells"]}
    assert set(manifest_by_key) == expected
    complete, scenarios = {}, {}
    for summary_path in args.input_root.glob("*/tables/summary.yaml"):
        summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
        environment = next(name for name, item in inventory["environments"].items() if item["sha256"] == summary["environment_config_sha256"])
        key = (summary["method"], environment, int(summary["training_seed"]))
        assert key in expected and key not in complete
        planned = manifest_by_key[key]
        assert summary["evaluation_mode"] == "policy_only"
        assert summary["evaluation_protocol"]["role"] == "publication_confirmation_extension"
        assert summary["evaluation_seed_start"] == 149000 and summary["evaluation_seed_end"] == 149199
        assert summary["episodes"] == 200 and summary["training_seed_source"] == "training_metadata"
        assert summary["checkpoint_sha256"] == planned["checkpoint_sha256"]
        assert summary["training_config_sha256"] == planned["training_config_sha256"]
        assert summary["training_metadata_sha256"] == planned["training_metadata_sha256"]
        assert summary["environment_config_sha256"] == planned["environment_sha256"]
        assert {name: bool(summary["action_safety_filter"][name]) for name in ("enabled", "static_boundary_enabled", "dynamic_enabled")} == {
            "enabled": False, "static_boundary_enabled": False, "dynamic_enabled": False,
        }
        assert summary["total_safety_filter_modifications"] == 0 and summary["shield_metrics_applicable"] is False
        assert finite_tree(summary)
        episodes_path = summary_path.with_name("episodes.csv")
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        seeds = [int(row["seed"]) for row in rows]
        assert len(rows) == 200 and len(set(seeds)) == 200 and set(seeds) == EVAL_SEEDS
        outcomes = Counter(row["outcome"] for row in rows)
        assert set(outcomes) <= set(OUTCOMES) and sum(outcomes.values()) == 200
        for row in rows:
            flags = {
                "success": int(row["success"]), "dynamic_collision": int(row["dynamic_collision"]),
                "static_collision": int(row["static_collision"]), "timeout": int(row["timeout"]),
                "out_of_bounds": int(row["oob"]),
            }
            assert sum(flags.values()) == 1 and flags[row["outcome"]] == 1
            assert int(row["collision"]) == flags["dynamic_collision"] + flags["static_collision"]
            assert int(row["shield_intervention_count"]) == 0
            scenario_key = (environment, int(row["seed"]))
            scenario = json.loads(row["episode_randomization"])
            if scenario_key in scenarios:
                assert scenarios[scenario_key] == scenario
            else:
                scenarios[scenario_key] = scenario
        assert summary["outcomes"] == {name: outcomes[name] for name in OUTCOMES}
        assert math.isclose(summary["collision_rate"], (outcomes["dynamic_collision"] + outcomes["static_collision"]) / 200)
        assert math.isclose(summary["safety_failure_rate"], (outcomes["dynamic_collision"] + outcomes["static_collision"] + outcomes["out_of_bounds"]) / 200)
        directory = summary_path.parents[1]
        assert {"checkpoint_compatibility_record.yaml", "evaluation_provenance.yaml", "protocol_metadata.yaml", "seed_overlap_check_record.yaml"} <= {p.name for p in directory.iterdir()}
        cell_id = planned["cell_id"]
        stdout = args.logs / f"{cell_id}.stdout.log"
        stderr = args.logs / f"{cell_id}.stderr.log"
        assert stdout.is_file() and stderr.is_file() and stderr.stat().st_size == 0
        complete[key] = {
            "method": key[0], "environment": key[1], "training_seed": key[2], "mode": "policy_only",
            "authorization": planned["authorization"],
            "directory": directory.relative_to(ROOT).as_posix(),
            "summary_sha256": sha256(summary_path), "episodes_sha256": sha256(episodes_path),
            "stdout_sha256": sha256(stdout), "stderr_sha256": sha256(stderr),
        }
    assert set(complete) == expected and len(list(args.input_root.iterdir())) == 75
    assert len(list(args.logs.glob("*.stdout.log"))) == 75 and len(list(args.logs.glob("*.stderr.log"))) == 75

    reference_paths = [
        ROOT / "outputs/publication_confirmation_v1/formal_freeze/authorized_scientific_cells.yaml",
        ROOT / "outputs/publication_confirmation_v1/formal_freeze/batch2_authorized_scientific_cells.yaml",
    ]
    reference_cells = sum((load_reference_cells(path) for path in reference_paths), [])
    assert len(reference_cells) == 50
    for cell in reference_cells:
        summary_path = ROOT / cell["directory"] / "tables/summary.yaml"
        episodes_path = summary_path.with_name("episodes.csv")
        assert sha256(summary_path) == cell["summary_sha256"] and sha256(episodes_path) == cell["episodes_sha256"]
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (cell["environment"], int(row["seed"]))
                assert scenarios[key] == json.loads(row["episode_randomization"]), f"Cross-method scenario mismatch: {key}"

    allowlist = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "FROZEN",
        "purpose": "exclusive Batch-4 broad-baseline scientific-analysis allowlist",
        "cell_count": 75, "reference_cell_count": 50, "complete_five_method_cell_count": 125,
        "cells": [complete[key] for key in sorted(complete)],
    }
    record = {
        "schema_version": 1, "status": "PASS", "audit_role": "Batch-4 broad-baseline integrity and authorization",
        "performance_interpretation_performed": False, "new_cell_count": 75, "new_episode_count": 15000,
        "reference_cells": 50, "complete_five_method_cells": 125,
        "checks": {
            "logical_matrix_exact": True, "episodes_and_seed_coverage_exact": True,
            "checkpoint_environment_and_provenance_hashes_match": True,
            "policy_only_interventions_zero": True, "terminal_outcomes_conserve": True,
            "collision_and_safety_identities_match": True,
            "five_method_scenario_randomization_match": True,
            "nan_or_inf_count": 0, "nonempty_stderr_count": 0,
            "launcher_complete_cells": 75,
        },
        "limitations": [
            "The 60 non-Medium cells are a post-opening prospective baseline extension, not part of the original blind opening.",
            "Breadth baselines support hierarchy and robustness context, not deep causal mechanism claims.",
        ],
        "audit_script_sha256": sha256(Path(__file__)), "execution_manifest_sha256": sha256(args.manifest),
        "launcher_status_sha256": sha256(args.launcher_status), "execution_inventory_sha256": sha256(inventory_path),
        "reference_allowlist_sha256": {path.name: sha256(path) for path in reference_paths},
    }
    return record, allowlist


def self_test() -> None:
    assert len({(m, e, s) for m in METHODS for e in ENVIRONMENTS for s in SEEDS}) == 75
    print("SELF_TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--launcher-status", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--allowlist-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("input_root", "logs", "manifest", "launcher_status", "audit_output", "allowlist_output"):
        path = getattr(args, name)
        setattr(args, name, path if path.is_absolute() else ROOT / path)
    record, allowlist = audit(args)
    args.audit_output.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8", newline="\n")
    args.allowlist_output.write_text(yaml.safe_dump(allowlist, sort_keys=False), encoding="utf-8", newline="\n")
    print("BATCH4 INTEGRITY AUDIT: PASS (75 cells, 15000 episodes; 125-cell five-method population)")


if __name__ == "__main__":
    main()
