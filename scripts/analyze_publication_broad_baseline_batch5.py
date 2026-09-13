"""Frozen five-method policy-vs-full-shield deployment analysis for Batch 5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml

from analyze_publication_confirmation import paired_statistics
from analyze_publication_shield_batch3 import burden


METHODS = ("RA-SAC-v3.2-c", "SAC-Attention-v3", "SAC-MLP", "TD3-MLP", "PPO-MLP")
ENVIRONMENTS = ("medium_id", "combined_ood")
MODES = ("policy_only", "full_shield")
SEEDS = (18207, 28207, 38207, 48207, 58207)
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


def load_allowlist(path: Path, source: str) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["status"] == "FROZEN" and data["cell_count"] == len(data["cells"])
    return [{**item, "source": source, "mode": item.get("mode", "policy_only")} for item in data["cells"]]


def load_cells(workspace: Path, allowlists: dict[str, Path]) -> dict:
    entries = []
    for source, path in allowlists.items():
        for item in load_allowlist(path, source):
            if item["environment"] not in ENVIRONMENTS:
                continue
            if source == "batch3_shield" and item["mode"] != "full_shield":
                continue
            entries.append(item)
    assert len(entries) == 100
    cells = {}
    for item in entries:
        key = (item["method"], item["environment"], int(item["training_seed"]), item["mode"])
        assert key not in cells
        directory = workspace / item["directory"]
        summary_path = directory / "tables/summary.yaml"
        episodes_path = directory / "tables/episodes.csv"
        assert sha256(summary_path) == item["summary_sha256"]
        assert sha256(episodes_path) == item["episodes_sha256"]
        summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            episodes = list(csv.DictReader(handle))
        assert summary["method"] == key[0] and int(summary["training_seed"]) == key[2]
        assert summary["evaluation_mode"] == key[3] and len(episodes) == 200
        assert {int(row["seed"]) for row in episodes} == set(range(149000, 149200))
        metrics = {name: float(summary[name]) for name in PERFORMANCE_METRICS}
        metrics.update(burden(episodes))
        cells[key] = {"metrics": metrics, "episodes": {int(row["seed"]): row for row in episodes}}
    expected = {
        (method, environment, seed, mode)
        for method in METHODS for environment in ENVIRONMENTS for seed in SEEDS for mode in MODES
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
                        "mean": statistics.mean(values.values()), "sample_sd": statistics.stdev(values.values()),
                        "worst_seed": worst, "worst_seed_value": values[worst],
                    }
    return result


def shield_effects(cells: dict) -> dict:
    result = {}
    for method in METHODS:
        result[method] = {}
        for environment in ENVIRONMENTS:
            result[method][environment] = {}
            for metric, direction in METRICS.items():
                shield = {seed: cells[(method, environment, seed, "full_shield")]["metrics"][metric] for seed in SEEDS}
                policy = {seed: cells[(method, environment, seed, "policy_only")]["metrics"][metric] for seed in SEEDS}
                result[method][environment][metric] = paired_statistics(shield, policy, direction)
    return result


def method_effects(cells: dict) -> dict:
    result = {}
    for comparator in METHODS[1:]:
        result[comparator] = {}
        for environment in ENVIRONMENTS:
            result[comparator][environment] = {}
            for mode in MODES:
                result[comparator][environment][mode] = {}
                for metric, direction in METRICS.items():
                    candidate = {seed: cells[(METHODS[0], environment, seed, mode)]["metrics"][metric] for seed in SEEDS}
                    baseline = {seed: cells[(comparator, environment, seed, mode)]["metrics"][metric] for seed in SEEDS}
                    result[comparator][environment][mode][metric] = paired_statistics(candidate, baseline, direction)
    return result


def transitions(cells: dict) -> tuple[list[dict], list[dict]]:
    counts, summary = [], []
    unsafe = {"dynamic_collision", "static_collision", "out_of_bounds"}
    collision = {"dynamic_collision", "static_collision"}
    for method in METHODS:
        for environment in ENVIRONMENTS:
            for seed in SEEDS:
                policy = cells[(method, environment, seed, "policy_only")]["episodes"]
                shield = cells[(method, environment, seed, "full_shield")]["episodes"]
                pairs = [(policy[e]["outcome"], shield[e]["outcome"]) for e in sorted(policy)]
                for source in OUTCOMES:
                    for target in OUTCOMES:
                        counts.append({
                            "method": method, "environment": environment, "training_seed": seed,
                            "policy_outcome": source, "full_shield_outcome": target,
                            "episode_count": sum(a == source and b == target for a, b in pairs),
                        })
                summary.append({
                    "method": method, "environment": environment, "training_seed": seed,
                    "rescued_unsafe_to_success": sum(a in unsafe and b == "success" for a, b in pairs),
                    "collision_to_timeout_or_oob": sum(a in collision and b in {"timeout", "out_of_bounds"} for a, b in pairs),
                    "shield_induced_success_regressions": sum(a == "success" and b != "success" for a, b in pairs),
                    "failure_redistribution": sum(a != "success" and b != "success" and a != b for a, b in pairs),
                })
    return counts, summary


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
    write_csv(output_dir / "seed_48207_deployment.csv", [row for row in seed_rows if row["training_seed"] == 48207])
    aggregates = aggregate(cells)
    ranking_rows, pareto_rows = [], []
    for environment in ENVIRONMENTS:
        for mode in MODES:
            for metric, direction in METRICS.items():
                if mode == "policy_only" and metric in BURDEN_METRICS:
                    continue
                ordered = sorted(METHODS, key=lambda m: aggregates[m][environment][mode][metric]["mean"], reverse=direction == "higher")
                for rank, method in enumerate(ordered, 1):
                    item = aggregates[method][environment][mode][metric]
                    ranking_rows.append({"environment": environment, "mode": mode, "metric": metric, "direction": direction,
                                         "rank": rank, "method": method, **item})
            points = {method: (aggregates[method][environment][mode]["success_rate"]["mean"],
                               aggregates[method][environment][mode]["safety_failure_rate"]["mean"]) for method in METHODS}
            for method, (success, safety) in points.items():
                dominated = any(other != method and s >= success and f <= safety and (s > success or f < safety)
                                for other, (s, f) in points.items())
                pareto_rows.append({"environment": environment, "mode": mode, "method": method,
                                    "mean_success_rate": success, "mean_safety_failure_rate": safety,
                                    "pareto_optimal": not dominated})
    write_csv(output_dir / "deployment_rankings.csv", ranking_rows)
    write_csv(output_dir / "success_safety_pareto.csv", pareto_rows)
    transition_counts, transition_summary = transitions(cells)
    write_csv(output_dir / "episode_transition_counts.csv", transition_counts)
    write_csv(output_dir / "episode_transition_summary.csv", transition_summary)
    for name, value in {
        "aggregate_metrics.yaml": aggregates,
        "paired_full_shield_minus_policy.yaml": shield_effects(cells),
        "paired_ra_sac_minus_comparators.yaml": method_effects(cells),
    }.items():
        (output_dir / name).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8", newline="\n")
    outputs = sorted(path for path in output_dir.iterdir() if path.name != "analysis_manifest.yaml")
    manifest = {
        **metadata, "analysis_completed_at": datetime.now(timezone.utc).isoformat(), "analysis_status": "COMPLETE",
        "performance_interpretation_performed_before_analysis": False,
        "analysis_population_cells": 100, "analysis_population_episodes": 20000,
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "limitations": [
            "The 15 Batch-5 Combined-OOD breadth-baseline cells are a post-opening prospective extension.",
            "Full-shield telemetry supports total intervention burden only, not static/dynamic source attribution.",
            "Five paired training seeds support effect-size and consistency reporting, not significance-centered claims.",
        ],
    }
    (output_dir / "analysis_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8", newline="\n")


def self_test() -> None:
    assert len({(m, e, s, mode) for m in METHODS for e in ENVIRONMENTS for s in SEEDS for mode in MODES}) == 100
    assert PERFORMANCE_METRICS["success_rate"] == "higher" and PERFORMANCE_METRICS["safety_failure_rate"] == "lower"
    print("SELF_TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--analysis-protocol", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    protocol = yaml.safe_load(args.analysis_protocol.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_BEFORE_BATCH5_PERFORMANCE_OPENING"
    assert protocol["analysis_script_sha256"] == sha256(Path(__file__))
    dependencies = {Path(path).name: sha256(args.workspace / path) for path in protocol["analysis_dependencies"]}
    assert dependencies == protocol["analysis_dependency_sha256"]
    allowlists = {name: args.workspace / path for name, path in protocol["allowlists"].items()}
    assert {name: sha256(path) for name, path in allowlists.items()} == protocol["allowlist_sha256"]
    integrity = args.workspace / protocol["integrity_audit"]
    assert sha256(integrity) == protocol["integrity_audit_sha256"]
    assert yaml.safe_load(integrity.read_text(encoding="utf-8"))["status"] == "PASS"
    cells = load_cells(args.workspace, allowlists)
    write_outputs(cells, args.output_dir, {
        "schema_version": 1, "analysis_started_at": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": sha256(Path(__file__)), "analysis_protocol_sha256": sha256(args.analysis_protocol),
        "analysis_dependency_sha256": dependencies,
        "allowlist_sha256": {name: sha256(path) for name, path in allowlists.items()},
        "integrity_audit_sha256": sha256(integrity),
    })
    print("BATCH5 FROZEN ANALYSIS: COMPLETE (100 cells, five methods, two environments, two modes)")


if __name__ == "__main__":
    main()
