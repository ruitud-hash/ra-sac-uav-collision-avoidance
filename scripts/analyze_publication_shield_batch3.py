"""Fail-closed frozen analysis for the authorized 80-cell Batch-3 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml

from analyze_publication_confirmation import paired_statistics, percentile


METHODS = ("RA-SAC-v3.2-c", "SAC-Attention-v3")
ENVIRONMENTS = ("medium_id", "combined_ood")
SEEDS = (18207, 28207, 38207, 48207, 58207)
MODES = ("policy_only", "static_boundary_only", "dynamic_ttc_only", "full_shield")
SHIELD_MODES = MODES[1:]
OUTCOMES = ("success", "dynamic_collision", "static_collision", "timeout", "out_of_bounds")
PERFORMANCE_METRICS = {
    "success_rate": "higher",
    "dynamic_collision_rate": "lower",
    "static_collision_rate": "lower",
    "collision_rate": "lower",
    "safety_failure_rate": "lower",
    "timeout_rate": "lower",
    "out_of_bounds_rate": "lower",
    "avg_reward": "higher",
    "avg_fhp": "lower",
}
BURDEN_METRICS = {
    "modified_episode_count": "lower",
    "modified_episode_rate": "lower",
    "modifications_per_episode": "lower",
    "SIF": "lower",
    "SIR": "lower",
    "SCM": "lower",
    "median_interventions_per_episode": "lower",
    "p95_interventions_per_episode": "lower",
}
METRICS = {**PERFORMANCE_METRICS, **BURDEN_METRICS}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_allowlist(path: Path, mode: str | None) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["status"] == "FROZEN"
    assert data["cell_count"] == len(data["cells"])
    cells = []
    for item in data["cells"]:
        cell = dict(item)
        cell["mode"] = mode or cell["mode"]
        cells.append(cell)
    return cells


def burden(rows: list[dict]) -> dict[str, float]:
    interventions = [int(row["shield_intervention_count"]) for row in rows]
    total = sum(interventions)
    correction_sum = sum(float(row["safety_filter_correction_sum"]) for row in rows)
    total_steps = sum(int(row["steps"]) for row in rows)
    modified = sum(value > 0 for value in interventions)
    return {
        "modified_episode_count": modified,
        "modified_episode_rate": modified / len(rows),
        "modifications_per_episode": total / len(rows),
        "SIF": total / len(rows),
        "SIR": total / total_steps,
        "SCM": correction_sum / total if total else 0.0,
        "median_interventions_per_episode": statistics.median(interventions),
        "p95_interventions_per_episode": percentile(interventions, 0.95),
    }


def load_cells(workspace: Path, p0_allowlist: Path, batch3_allowlist: Path) -> dict:
    entries = load_allowlist(p0_allowlist, "policy_only") + load_allowlist(batch3_allowlist, None)
    assert len(entries) == 80
    cells = {}
    for entry in entries:
        key = (entry["method"], entry["environment"], int(entry["training_seed"]), entry["mode"])
        assert key not in cells, f"duplicate allowlisted cell: {key}"
        directory = workspace / entry["directory"]
        summary_path = directory / "tables/summary.yaml"
        episodes_path = directory / "tables/episodes.csv"
        assert sha256(summary_path) == entry["summary_sha256"]
        assert sha256(episodes_path) == entry["episodes_sha256"]
        summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            episodes = list(csv.DictReader(handle))
        assert summary["method"] == key[0]
        assert int(summary["training_seed"]) == key[2]
        assert summary["evaluation_mode"] == key[3]
        assert summary["episodes"] == 200 and len(episodes) == 200
        assert {int(row["seed"]) for row in episodes} == set(range(149000, 149200))
        metrics = {name: float(summary[name]) for name in PERFORMANCE_METRICS}
        metrics.update(burden(episodes))
        cells[key] = {"metrics": metrics, "episodes": {int(row["seed"]): row for row in episodes}}
    expected = {
        (method, environment, seed, mode)
        for method in METHODS
        for environment in ENVIRONMENTS
        for seed in SEEDS
        for mode in MODES
    }
    assert set(cells) == expected
    return cells


def aggregate(cells: dict) -> dict:
    result = {}
    for method in METHODS:
        result[method] = {}
        for environment in ENVIRONMENTS:
            result[method][environment] = {}
            for mode in MODES:
                result[method][environment][mode] = {}
                for metric, direction in METRICS.items():
                    values = {seed: cells[(method, environment, seed, mode)]["metrics"][metric] for seed in SEEDS}
                    worst = min(values, key=values.get) if direction == "higher" else max(values, key=values.get)
                    result[method][environment][mode][metric] = {
                        "mean": statistics.mean(values.values()),
                        "sample_sd": statistics.stdev(values.values()),
                        "worst_seed": worst,
                        "worst_seed_value": values[worst],
                    }
    return result


def method_effects(cells: dict) -> dict:
    result = {}
    for environment in ENVIRONMENTS:
        result[environment] = {}
        for mode in MODES:
            result[environment][mode] = {}
            for metric, direction in METRICS.items():
                candidate = {seed: cells[(METHODS[0], environment, seed, mode)]["metrics"][metric] for seed in SEEDS}
                baseline = {seed: cells[(METHODS[1], environment, seed, mode)]["metrics"][metric] for seed in SEEDS}
                result[environment][mode][metric] = paired_statistics(candidate, baseline, direction)
    return result


def shield_effects(cells: dict) -> dict:
    result = {}
    for method in METHODS:
        result[method] = {}
        for environment in ENVIRONMENTS:
            result[method][environment] = {}
            for mode in SHIELD_MODES:
                result[method][environment][mode] = {}
                for metric, direction in METRICS.items():
                    shield = {seed: cells[(method, environment, seed, mode)]["metrics"][metric] for seed in SEEDS}
                    policy = {seed: cells[(method, environment, seed, "policy_only")]["metrics"][metric] for seed in SEEDS}
                    result[method][environment][mode][metric] = paired_statistics(shield, policy, direction)
    return result


def transitions(cells: dict) -> tuple[list[dict], list[dict]]:
    counts, derived = [], []
    unsafe = {"dynamic_collision", "static_collision", "out_of_bounds"}
    collision = {"dynamic_collision", "static_collision"}
    for method in METHODS:
        for environment in ENVIRONMENTS:
            for seed in SEEDS:
                policy = cells[(method, environment, seed, "policy_only")]["episodes"]
                for mode in SHIELD_MODES:
                    shield = cells[(method, environment, seed, mode)]["episodes"]
                    pairs = [(policy[evaluation_seed]["outcome"], shield[evaluation_seed]["outcome"]) for evaluation_seed in sorted(policy)]
                    for source in OUTCOMES:
                        for target in OUTCOMES:
                            counts.append({
                                "method": method, "environment": environment, "training_seed": seed,
                                "shield_mode": mode, "policy_outcome": source, "shield_outcome": target,
                                "episode_count": sum(a == source and b == target for a, b in pairs),
                            })
                    derived.append({
                        "method": method, "environment": environment, "training_seed": seed, "shield_mode": mode,
                        "shield_rescued_unsafe_episodes": sum(a in unsafe and b == "success" for a, b in pairs),
                        "collision_to_other_failure": sum(a in collision and b in {"timeout", "out_of_bounds"} for a, b in pairs),
                        "shield_induced_regressions": sum(a == "success" and b != "success" for a, b in pairs),
                        "failure_redistribution": sum(a != "success" and b != "success" and a != b for a, b in pairs),
                    })
    return counts, derived


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(cells: dict, output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_rows = [
        {"method": key[0], "environment": key[1], "training_seed": key[2], "mode": key[3], **value["metrics"]}
        for key, value in sorted(cells.items())
    ]
    write_csv(output_dir / "seed_level_metrics.csv", seed_rows)
    write_csv(output_dir / "seed_48207_four_mode.csv", [row for row in seed_rows if row["training_seed"] == 48207])
    transition_counts, transition_summary = transitions(cells)
    write_csv(output_dir / "episode_transition_counts.csv", transition_counts)
    write_csv(output_dir / "episode_transition_summary.csv", transition_summary)
    for name, value in {
        "aggregate_metrics.yaml": aggregate(cells),
        "paired_method_effects.yaml": method_effects(cells),
        "paired_shield_effects.yaml": shield_effects(cells),
    }.items():
        (output_dir / name).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8", newline="\n")
    outputs = sorted(path for path in output_dir.iterdir() if path.name != "analysis_manifest.yaml")
    manifest = {
        **metadata,
        "analysis_completed_at": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "COMPLETE",
        "performance_interpretation_performed_before_analysis": False,
        "analysis_population_cells": 80,
        "analysis_population_episodes": 16000,
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "limitations": [
            "Full-shield intervention source attribution is unavailable.",
            "Combined-OOD isolated-layer cells are a policy-result-conditioned prospective extension.",
        ],
    }
    (output_dir / "analysis_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8", newline="\n"
    )


def self_test() -> None:
    rows = [
        {"shield_intervention_count": "0", "safety_filter_correction_sum": "0", "steps": "10"},
        {"shield_intervention_count": "2", "safety_filter_correction_sum": "1.0", "steps": "10"},
    ]
    result = burden(rows)
    assert result["modified_episode_rate"] == 0.5
    assert result["SIF"] == 1.0 and result["SIR"] == 0.1 and result["SCM"] == 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--policy-allowlist", type=Path)
    parser.add_argument("--batch3-allowlist", type=Path)
    parser.add_argument("--analysis-protocol", type=Path)
    parser.add_argument("--integrity-audit", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("SELF_TEST: PASS")
        return
    required = (args.policy_allowlist, args.batch3_allowlist, args.analysis_protocol, args.integrity_audit, args.output_dir)
    if any(value is None for value in required):
        parser.error("all analysis inputs and --output-dir are required")
    protocol = yaml.safe_load(args.analysis_protocol.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_BEFORE_PERFORMANCE_OPENING"
    assert protocol["analysis_script_sha256"] == sha256(Path(__file__))
    assert protocol["policy_allowlist_sha256"] == sha256(args.policy_allowlist)
    assert protocol["batch3_allowlist_sha256"] == sha256(args.batch3_allowlist)
    assert protocol["integrity_audit_sha256"] == sha256(args.integrity_audit)
    assert yaml.safe_load(args.integrity_audit.read_text(encoding="utf-8"))["status"] == "PASS"
    started = datetime.now(timezone.utc).isoformat()
    cells = load_cells(args.workspace, args.policy_allowlist, args.batch3_allowlist)
    write_outputs(cells, args.output_dir, {
        "schema_version": 1,
        "analysis_started_at": started,
        "analysis_script_sha256": sha256(Path(__file__)),
        "analysis_protocol_sha256": sha256(args.analysis_protocol),
        "policy_allowlist_sha256": sha256(args.policy_allowlist),
        "batch3_allowlist_sha256": sha256(args.batch3_allowlist),
        "integrity_audit_sha256": sha256(args.integrity_audit),
    })
    print(f"BATCH3 FROZEN ANALYSIS: COMPLETE ({len(cells)} cells)")


if __name__ == "__main__":
    main()
