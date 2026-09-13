"""Fail-closed paired-training-seed analysis for publication confirmation."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import yaml


METRICS = {
    "success_rate": "higher",
    "dynamic_collision_rate": "lower",
    "static_collision_rate": "lower",
    "safety_failure_rate": "lower",
    "timeout_rate": "lower",
    "out_of_bounds_rate": "lower",
    "avg_reward": "higher",
    "avg_fhp": "lower",
}
ENVIRONMENTS = {
    "env_medium_v3_unified_id.yaml": "medium_id",
    "env_medium_v3_disturbance_ood.yaml": "disturbance_ood",
    "env_medium_v3_behavior_ood.yaml": "behavior_ood",
    "env_medium_v3_composition_ood.yaml": "composition_ood",
    "env_medium_v3_combined_ood.yaml": "combined_ood",
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_statistics(
    candidate: dict[int, float], baseline: dict[int, float], direction: str
) -> dict:
    seeds = sorted(candidate)
    if seeds != sorted(baseline):
        raise ValueError("Candidate and baseline training seeds do not match")
    raw = [candidate[seed] - baseline[seed] for seed in seeds]
    favorable = raw if direction == "higher" else [-value for value in raw]
    rng = random.Random(20260809)
    boot = [statistics.mean(rng.choices(raw, k=len(raw))) for _ in range(20000)]
    tolerance = 1e-12
    return {
        "training_seeds": seeds,
        "raw_candidate_minus_baseline": dict(zip(seeds, raw)),
        "mean_raw_difference": statistics.mean(raw),
        "median_raw_difference": statistics.median(raw),
        "paired_bootstrap_95pct_CI": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "win_tie_loss": {
            "win": sum(value > tolerance for value in favorable),
            "tie": sum(abs(value) <= tolerance for value in favorable),
            "loss": sum(value < -tolerance for value in favorable),
        },
        "worst_favorable_difference": min(favorable),
        "positive_means_candidate_better": direction == "higher",
    }


def load_cells(root: Path, mode: str) -> dict[tuple[str, str, int], dict]:
    cells = {}
    for path in root.glob("**/tables/summary.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data.get("evaluation_protocol", {}).get("role") != "publication_confirmation":
            continue
        if data.get("evaluation_mode") != mode:
            continue
        environment = ENVIRONMENTS.get(Path(data["environment_config_path"]).name)
        if environment is None:
            raise ValueError(f"Unknown environment config in {path}")
        key = (data["method"], environment, int(data["training_seed"]))
        if key in cells:
            raise ValueError(f"Duplicate completed cell: {key}")
        if (
            data["evaluation_seed_start"] != 149000
            or data["evaluation_seed_end"] != 149199
            or data["episodes"] != 200
        ):
            raise ValueError(f"Protocol mismatch in {path}")
        cells[key] = data
    return cells


def analyze(cells: dict, methods: list[str], environments: list[str]) -> dict:
    seeds = [18207, 28207, 38207, 48207, 58207]
    expected = {(method, env, seed) for method in methods for env in environments for seed in seeds}
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        extra = sorted(set(cells) - expected)
        raise ValueError(f"Incomplete or unexpected matrix; missing={missing}, extra={extra}")
    result = {"unit": "paired_training_seed", "methods": {}, "paired_vs_sac_attention": {}}
    for method in methods:
        result["methods"][method] = {}
        for environment in environments:
            metric_results = {}
            for metric, direction in METRICS.items():
                values = {seed: float(cells[(method, environment, seed)][metric]) for seed in seeds}
                worst_seed = min(values, key=values.get) if direction == "higher" else max(values, key=values.get)
                metric_results[metric] = {
                    "mean": statistics.mean(values.values()),
                    "sample_sd": statistics.stdev(values.values()),
                    "worst_seed": worst_seed,
                    "worst_seed_value": values[worst_seed],
                }
            result["methods"][method][environment] = metric_results
    if "RA-SAC-v3.2-c" in methods and "SAC-Attention-v3" in methods:
        for environment in environments:
            result["paired_vs_sac_attention"][environment] = {}
            for metric, direction in METRICS.items():
                candidate = {seed: float(cells[("RA-SAC-v3.2-c", environment, seed)][metric]) for seed in seeds}
                baseline = {seed: float(cells[("SAC-Attention-v3", environment, seed)][metric]) for seed in seeds}
                result["paired_vs_sac_attention"][environment][metric] = paired_statistics(candidate, baseline, direction)
    return result


def write_outputs(cells: dict, analysis: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for (method, environment, seed), data in sorted(cells.items()):
        rows.append({"method": method, "environment": environment, "training_seed": seed, **{metric: data[metric] for metric in METRICS}})
    with (output_dir / "seed_level_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "paired_seed_analysis.yaml").write_text(
        yaml.safe_dump(analysis, sort_keys=False), encoding="utf-8", newline="\n"
    )


def self_test() -> None:
    candidate = {seed: value for seed, value in zip(range(5), [2, 3, 4, 5, 6])}
    baseline = {seed: value for seed, value in zip(range(5), [1, 2, 3, 4, 5])}
    result = paired_statistics(candidate, baseline, "higher")
    assert result["mean_raw_difference"] == 1
    assert result["win_tie_loss"] == {"win": 5, "tie": 0, "loss": 0}
    assert result["paired_bootstrap_95pct_CI"] == [1.0, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", default="policy_only")
    parser.add_argument("--methods", nargs="+", default=["RA-SAC-v3.2-c", "SAC-Attention-v3"])
    parser.add_argument("--environments", nargs="+", default=["medium_id", "combined_ood"])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("SELF_TEST: PASS")
        return
    if args.input_root is None or args.output_dir is None:
        parser.error("--input-root and --output-dir are required")
    cells = load_cells(args.input_root, args.mode)
    write_outputs(cells, analyze(cells, args.methods, args.environments), args.output_dir)


if __name__ == "__main__":
    main()
