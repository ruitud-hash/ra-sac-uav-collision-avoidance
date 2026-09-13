"""Post-hoc paired-episode failure-transition diagnostics for frozen data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


METHODS = ("SAC-Attention-v3", "RA-SAC-v3.2-c")
OUTCOMES = ("success", "dynamic_collision", "static_collision", "timeout", "out_of_bounds")
ENVIRONMENTS = ("medium_id", "disturbance_ood", "behavior_ood", "composition_ood", "combined_ood")
TRAINING_SEEDS = (18207, 28207, 38207, 48207, 58207)
EVALUATION_SEEDS = set(range(149000, 149200))
JSON_SCENARIO_FIELDS = (
    "domain_parameters",
    "dynamic_type_counts",
    "dynamic_scenario_counts",
    "episode_randomization",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transition_class(baseline: str, candidate: str) -> str:
    if baseline == candidate == "success":
        return "stable_success"
    if baseline == "success":
        return "lost_success"
    if candidate == "success":
        return "rescued_failure"
    if baseline == candidate:
        return "same_failure"
    return "redistributed_failure"


def read_cells(project_root: Path, allowlists: list[Path]) -> tuple[dict, dict[str, str]]:
    cells: dict[tuple[str, str, int], dict[int, dict[str, str]]] = {}
    allowlist_hashes = {}
    for allowlist in allowlists:
        data = yaml.safe_load(allowlist.read_text(encoding="utf-8"))
        if data.get("status") != "FROZEN" or data.get("cell_count") != len(data.get("cells", [])):
            raise ValueError(f"Invalid or unfrozen allowlist: {allowlist}")
        allowlist_hashes[str(allowlist)] = sha256(allowlist)
        for cell in data["cells"]:
            key = (cell["method"], cell["environment"], int(cell["training_seed"]))
            if key in cells:
                raise ValueError(f"Duplicate authorized cell: {key}")
            cell_dir = project_root / Path(cell["directory"])
            episodes_path = cell_dir / "tables" / "episodes.csv"
            summary_path = cell_dir / "tables" / "summary.yaml"
            if sha256(episodes_path) != cell["episodes_sha256"] or sha256(summary_path) != cell["summary_sha256"]:
                raise ValueError(f"Authorized input hash mismatch: {key}")
            with episodes_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            by_seed = {int(row["seed"]): row for row in rows}
            if len(rows) != 200 or len(by_seed) != 200 or set(by_seed) != EVALUATION_SEEDS:
                raise ValueError(f"Episode coverage mismatch: {key}")
            if any(row["outcome"] not in OUTCOMES for row in rows):
                raise ValueError(f"Unknown terminal outcome: {key}")
            cells[key] = by_seed
    expected = {
        (method, environment, seed)
        for method in METHODS
        for environment in ENVIRONMENTS
        for seed in TRAINING_SEEDS
    }
    if set(cells) != expected:
        raise ValueError(f"Authorized matrix mismatch; missing={sorted(expected-set(cells))}, extra={sorted(set(cells)-expected)}")
    return cells, allowlist_hashes


def validate_scenario_pair(baseline: dict[str, str], candidate: dict[str, str]) -> None:
    if baseline["control_delay_steps"] != candidate["control_delay_steps"]:
        raise ValueError("Paired control delay mismatch")
    for field in JSON_SCENARIO_FIELDS:
        if json.loads(baseline[field]) != json.loads(candidate[field]):
            raise ValueError(f"Paired scenario mismatch in {field}")


def summarize(rows: list[dict]) -> dict:
    counts = Counter(row["transition_class"] for row in rows)
    lost = Counter(row["candidate_outcome"] for row in rows if row["transition_class"] == "lost_success")
    rescued = Counter(row["baseline_outcome"] for row in rows if row["transition_class"] == "rescued_failure")
    result = {
        "episodes": len(rows),
        "stable_success": counts["stable_success"],
        "lost_success_total": counts["lost_success"],
        "lost_to_dynamic_collision": lost["dynamic_collision"],
        "lost_to_static_collision": lost["static_collision"],
        "lost_to_timeout": lost["timeout"],
        "lost_to_out_of_bounds": lost["out_of_bounds"],
        "rescued_failure_total": counts["rescued_failure"],
        "rescued_from_dynamic_collision": rescued["dynamic_collision"],
        "rescued_from_static_collision": rescued["static_collision"],
        "rescued_from_timeout": rescued["timeout"],
        "rescued_from_out_of_bounds": rescued["out_of_bounds"],
        "same_failure": counts["same_failure"],
        "redistributed_failure": counts["redistributed_failure"],
        "net_success_change": counts["rescued_failure"] - counts["lost_success"],
    }
    baseline_success = sum(row["baseline_outcome"] == "success" for row in rows)
    candidate_success = sum(row["candidate_outcome"] == "success" for row in rows)
    assert result["net_success_change"] == candidate_success - baseline_success
    return result


def build_outputs(cells: dict) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    paired_rows = []
    for environment in ENVIRONMENTS:
        for training_seed in TRAINING_SEEDS:
            baseline = cells[(METHODS[0], environment, training_seed)]
            candidate = cells[(METHODS[1], environment, training_seed)]
            for evaluation_seed in sorted(EVALUATION_SEEDS):
                b_row, c_row = baseline[evaluation_seed], candidate[evaluation_seed]
                validate_scenario_pair(b_row, c_row)
                paired_rows.append(
                    {
                        "environment": environment,
                        "training_seed": training_seed,
                        "evaluation_seed": evaluation_seed,
                        "baseline_outcome": b_row["outcome"],
                        "candidate_outcome": c_row["outcome"],
                        "transition_class": transition_class(b_row["outcome"], c_row["outcome"]),
                    }
                )

    grouped: dict[tuple[str, int | str], list[dict]] = defaultdict(list)
    for row in paired_rows:
        grouped[(row["environment"], row["training_seed"])].append(row)
        grouped[(row["environment"], "all")].append(row)

    matrices, summaries = [], []
    for (environment, training_seed), rows in grouped.items():
        matrix = Counter((row["baseline_outcome"], row["candidate_outcome"]) for row in rows)
        for baseline_outcome in OUTCOMES:
            for candidate_outcome in OUTCOMES:
                count = matrix[(baseline_outcome, candidate_outcome)]
                matrices.append(
                    {
                        "environment": environment,
                        "training_seed": training_seed,
                        "baseline_outcome": baseline_outcome,
                        "candidate_outcome": candidate_outcome,
                        "count": count,
                        "fraction_of_episodes": count / len(rows),
                    }
                )
        summaries.append({"environment": environment, "training_seed": training_seed, **summarize(rows)})

    diagnostic_rows = []
    for row in paired_rows:
        if row["training_seed"] != 48207:
            continue
        environment, evaluation_seed = row["environment"], row["evaluation_seed"]
        b_row = cells[(METHODS[0], environment, 48207)][evaluation_seed]
        c_row = cells[(METHODS[1], environment, 48207)][evaluation_seed]
        domain = json.loads(b_row["domain_parameters"])
        type_counts = json.loads(b_row["dynamic_type_counts"])
        scenario_counts = json.loads(b_row["dynamic_scenario_counts"])
        diagnostic_rows.append(
            {
                **row,
                "control_delay_steps": int(b_row["control_delay_steps"]),
                "dynamic_obstacle_count": sum(type_counts.values()),
                "wind_base_speed_mps": domain.get("wind_base_speed_mps"),
                "wind_speed_variation_mps": domain.get("wind_speed_variation_mps"),
                "accel_error_std_mps2": domain.get("accel_error_std_mps2"),
                "intruder_position_std_m": domain.get("intruder_position_std_m"),
                "intruder_velocity_std_mps": domain.get("intruder_velocity_std_mps"),
                "dynamic_type_counts": json.dumps(type_counts, sort_keys=True, separators=(",", ":")),
                "dynamic_scenario_counts": json.dumps(scenario_counts, sort_keys=True, separators=(",", ":")),
                "baseline_steps": int(b_row["steps"]),
                "candidate_steps": int(c_row["steps"]),
                "baseline_reward": float(b_row["total_reward"]),
                "candidate_reward": float(c_row["total_reward"]),
                "baseline_min_static_clearance_m": float(b_row["min_path_static_clearance_m"]),
                "candidate_min_static_clearance_m": float(c_row["min_path_static_clearance_m"]),
            }
        )

    strata = []
    for field in ("control_delay_steps", "dynamic_obstacle_count"):
        buckets: dict[tuple[str, object], list[dict]] = defaultdict(list)
        for row in diagnostic_rows:
            buckets[(row["environment"], row[field])].append(row)
        for (environment, value), rows in buckets.items():
            strata.append(
                {
                    "environment": environment,
                    "stratum": field,
                    "value": value,
                    **summarize(rows),
                }
            )
    return paired_rows, matrices, summaries, diagnostic_rows, strata


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    rows = [
        {"baseline_outcome": "success", "candidate_outcome": "dynamic_collision", "transition_class": "lost_success"},
        {"baseline_outcome": "timeout", "candidate_outcome": "success", "transition_class": "rescued_failure"},
        {"baseline_outcome": "success", "candidate_outcome": "success", "transition_class": "stable_success"},
    ]
    result = summarize(rows)
    assert result["lost_to_dynamic_collision"] == 1
    assert result["rescued_from_timeout"] == 1
    assert result["net_success_change"] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--allowlists", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("SELF_TEST: PASS")
        return
    if not args.allowlists or args.output_dir is None:
        parser.error("--allowlists and --output-dir are required")
    project_root = args.project_root.resolve()
    allowlists = [(project_root / path).resolve() if not path.is_absolute() else path for path in args.allowlists]
    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    cells, allowlist_hashes = read_cells(project_root, allowlists)
    outputs = build_outputs(cells)
    output_dir.mkdir(parents=True, exist_ok=False)
    names = (
        "paired_episode_outcomes.csv",
        "transition_matrix.csv",
        "transition_summary.csv",
        "seed_48207_episode_diagnostics.csv",
        "seed_48207_stratification.csv",
    )
    for name, rows in zip(names, outputs):
        write_csv(output_dir / name, rows)
    output_hashes = {name: sha256(output_dir / name) for name in names}
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "analysis_role": "POST_HOC_EXPLORATORY_DIAGNOSTIC",
        "causal_claims_authorized": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods": list(METHODS),
        "environments": list(ENVIRONMENTS),
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seed_range": [149000, 149199],
        "authorized_cell_count": len(cells),
        "paired_episode_count": len(outputs[0]),
        "scenario_pair_validation": "PASS",
        "allowlist_sha256": allowlist_hashes,
        "analysis_script_sha256": sha256(Path(__file__)),
        "output_sha256": output_hashes,
        "telemetry_limitations": [
            "no step-level TTC/DCPA",
            "no realized relative-speed trace",
            "no boundary-distance trace",
        ],
    }
    (output_dir / "diagnostic_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8", newline="\n"
    )
    print(f"DIAGNOSTIC: COMPLETE ({len(cells)} cells, {len(outputs[0])} paired episodes)")


if __name__ == "__main__":
    main()
