"""Frozen five-method, five-environment policy-only analysis for Batch 4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml

from analyze_publication_confirmation import paired_statistics


METHODS = ("RA-SAC-v3.2-c", "SAC-Attention-v3", "SAC-MLP", "TD3-MLP", "PPO-MLP")
ENVIRONMENTS = ("medium_id", "disturbance_ood", "behavior_ood", "composition_ood", "combined_ood")
SEEDS = (18207, 28207, 38207, 48207, 58207)
METRICS = {
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cells(workspace: Path, allowlists: list[Path]) -> dict:
    cells = {}
    for allowlist in allowlists:
        data = yaml.safe_load(allowlist.read_text(encoding="utf-8"))
        assert data["status"] == "FROZEN" and data["cell_count"] == len(data["cells"])
        for item in data["cells"]:
            key = (item["method"], item["environment"], int(item["training_seed"]))
            assert key not in cells
            summary_path = workspace / item["directory"] / "tables/summary.yaml"
            assert sha256(summary_path) == item["summary_sha256"]
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            assert summary["evaluation_mode"] == "policy_only"
            cells[key] = {metric: float(summary[metric]) for metric in METRICS}
    expected = {(method, environment, seed) for method in METHODS for environment in ENVIRONMENTS for seed in SEEDS}
    assert set(cells) == expected
    return cells


def aggregate(cells: dict) -> dict:
    result = {}
    for method in METHODS:
        result[method] = {}
        for environment in ENVIRONMENTS:
            result[method][environment] = {}
            for metric, direction in METRICS.items():
                values = {seed: cells[(method, environment, seed)][metric] for seed in SEEDS}
                worst = min(values, key=values.get) if direction == "higher" else max(values, key=values.get)
                result[method][environment][metric] = {
                    "mean": statistics.mean(values.values()),
                    "sample_sd": statistics.stdev(values.values()),
                    "worst_seed": worst,
                    "worst_seed_value": values[worst],
                }
    return result


def paired_vs_ra_sac(cells: dict) -> dict:
    result = {}
    for comparator in METHODS[1:]:
        result[comparator] = {}
        for environment in ENVIRONMENTS:
            result[comparator][environment] = {}
            for metric, direction in METRICS.items():
                candidate = {seed: cells[(METHODS[0], environment, seed)][metric] for seed in SEEDS}
                baseline = {seed: cells[(comparator, environment, seed)][metric] for seed in SEEDS}
                result[comparator][environment][metric] = paired_statistics(candidate, baseline, direction)
    return result


def ood_degradation(cells: dict) -> dict:
    result = {}
    for method in METHODS:
        result[method] = {}
        for environment in ENVIRONMENTS[1:]:
            result[method][environment] = {}
            for metric, direction in METRICS.items():
                ood = {seed: cells[(method, environment, seed)][metric] for seed in SEEDS}
                medium = {seed: cells[(method, "medium_id", seed)][metric] for seed in SEEDS}
                result[method][environment][metric] = paired_statistics(ood, medium, direction)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(cells: dict, output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_rows = [
        {"method": key[0], "environment": key[1], "training_seed": key[2], **value}
        for key, value in sorted(cells.items())
    ]
    write_csv(output_dir / "seed_level_metrics.csv", seed_rows)
    aggregates = aggregate(cells)
    ranking_rows = []
    for environment in ENVIRONMENTS:
        for metric, direction in METRICS.items():
            ordered = sorted(METHODS, key=lambda method: aggregates[method][environment][metric]["mean"], reverse=direction == "higher")
            for rank, method in enumerate(ordered, 1):
                item = aggregates[method][environment][metric]
                ranking_rows.append({
                    "environment": environment, "metric": metric, "direction": direction,
                    "rank": rank, "method": method, "mean": item["mean"], "sample_sd": item["sample_sd"],
                    "worst_seed": item["worst_seed"], "worst_seed_value": item["worst_seed_value"],
                })
    write_csv(output_dir / "environment_rankings.csv", ranking_rows)
    degradation = ood_degradation(cells)
    degradation_rows = []
    for method in METHODS:
        for environment in ENVIRONMENTS[1:]:
            for metric in METRICS:
                item = degradation[method][environment][metric]
                degradation_rows.append({
                    "method": method, "environment": environment, "metric": metric,
                    "ood_minus_medium_mean": item["mean_raw_difference"],
                    "ci_low": item["paired_bootstrap_95pct_CI"][0],
                    "ci_high": item["paired_bootstrap_95pct_CI"][1],
                    "worst_favorable_difference": item["worst_favorable_difference"],
                })
    write_csv(output_dir / "ood_degradation_map.csv", degradation_rows)
    for name, value in {
        "aggregate_metrics.yaml": aggregates,
        "paired_ra_sac_minus_comparators.yaml": paired_vs_ra_sac(cells),
        "paired_ood_minus_medium.yaml": degradation,
    }.items():
        (output_dir / name).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8", newline="\n")
    outputs = sorted(output_dir.iterdir())
    manifest = {
        **metadata,
        "analysis_completed_at": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "COMPLETE",
        "performance_interpretation_performed_before_analysis": False,
        "analysis_population_cells": 125,
        "analysis_population_episodes": 25000,
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "limitations": [
            "The 60 non-Medium breadth-baseline cells are a post-opening prospective extension.",
            "Five paired training seeds support effect-size and consistency reporting, not significance-centered claims.",
        ],
    }
    (output_dir / "analysis_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8", newline="\n")


def self_test() -> None:
    assert len({(m, e, s) for m in METHODS for e in ENVIRONMENTS for s in SEEDS}) == 125
    assert METRICS["success_rate"] == "higher" and METRICS["safety_failure_rate"] == "lower"
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
    assert protocol["status"] == "FROZEN_BEFORE_BATCH4_PERFORMANCE_OPENING"
    assert protocol["analysis_script_sha256"] == sha256(Path(__file__))
    allowlists = [args.workspace / path for path in protocol["allowlists"]]
    for path, expected in zip(allowlists, protocol["allowlist_sha256"]):
        assert sha256(path) == expected
    integrity = args.workspace / protocol["integrity_audit"]
    assert sha256(integrity) == protocol["integrity_audit_sha256"]
    assert yaml.safe_load(integrity.read_text(encoding="utf-8"))["status"] == "PASS"
    cells = load_cells(args.workspace, allowlists)
    write_outputs(cells, args.output_dir, {
        "schema_version": 1,
        "analysis_started_at": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "analysis_protocol_sha256": sha256(args.analysis_protocol),
        "allowlist_sha256": [sha256(path) for path in allowlists],
        "integrity_audit_sha256": sha256(integrity),
    })
    print("BATCH4 FROZEN ANALYSIS: COMPLETE (125 cells, five methods, five environments)")


if __name__ == "__main__":
    main()
