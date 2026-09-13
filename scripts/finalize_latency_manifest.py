"""Validate completed latency CSVs and recover a missing manifest without rerunning timing."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from benchmark_publication_latency import (
    E2E_MODES,
    METHODS,
    ROOT,
    SHIELD_MODES,
    hardware_metadata,
    load_yaml,
    sha256,
    split_obstacles,
    summarize,
)


def key(row: dict) -> tuple:
    return (
        row["device"], int(row["active_tokens"]), int(row["dynamic_obstacles"]),
        int(row["static_obstacles"]), row["method"], row["component"], row["shield_mode"],
    )


def expected_cells(protocol: dict, device: str) -> dict[tuple, int]:
    fast = int(protocol["timing"]["fast_iterations"])
    shield = int(protocol["timing"]["shield_iterations"])
    end_to_end = int(protocol["timing"]["end_to_end_iterations"])
    expected = {}
    for token_count in protocol["active_token_counts"]:
        dynamic, static = split_obstacles(int(token_count))
        expected[(device, token_count, dynamic, static, "shared", "observation_raw", "")] = fast
        for method in METHODS:
            for component in ("observation_pack", "observation_total", "policy"):
                expected[(device, token_count, dynamic, static, method, component, "")] = fast
            for mode in SHIELD_MODES:
                expected[(device, token_count, dynamic, static, method, "shield", mode)] = shield
            for mode in E2E_MODES:
                expected[(device, token_count, dynamic, static, method, "end_to_end", mode)] = end_to_end
    return expected


def validate_outputs(protocol: dict, device: str, output_dir: Path) -> tuple[Path, Path]:
    raw_path = output_dir / "raw_latency_samples.csv"
    summary_path = output_dir / "latency_summary.csv"
    assert raw_path.is_file() and summary_path.is_file()
    expected = expected_cells(protocol, device)
    samples = defaultdict(list)
    with raw_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = float(row["latency_ms"])
            assert math.isfinite(value) and value >= 0.0
            samples[key(row)].append(value)
    assert set(samples) == set(expected)
    assert all(len(samples[cell]) == iterations for cell, iterations in expected.items())

    summaries = {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cell = key(row)
            assert cell not in summaries
            summaries[cell] = row
    assert set(summaries) == set(expected)
    for cell, row in summaries.items():
        assert int(row["iterations"]) == expected[cell]
        recomputed = summarize(samples[cell])
        for field, value in recomputed.items():
            assert math.isclose(float(row[field]), value, rel_tol=1e-12, abs_tol=1e-12), (cell, field)
        if cell[5] == "end_to_end":
            assert math.isclose(float(row["p95_compatible_hz"]), 1000.0 / recomputed["p95_ms"], rel_tol=1e-12)
            assert math.isclose(float(row["p99_compatible_hz"]), 1000.0 / recomputed["p99_ms"], rel_tol=1e-12)
    return raw_path, summary_path


def finalize(args: argparse.Namespace) -> None:
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    incident_path = args.incident_record if args.incident_record.is_absolute() else ROOT / args.incident_record
    manifest_path = output_dir / "benchmark_manifest.yaml"
    assert not manifest_path.exists(), "Refusing to overwrite an existing benchmark manifest"
    assert sha256(protocol_path) == args.timing_protocol_sha256
    protocol = load_yaml(protocol_path)
    for relative, expected in protocol["asset_sha256"].items():
        assert sha256(ROOT / relative) == expected, f"Asset hash mismatch: {relative}"
    raw_path, summary_path = validate_outputs(protocol, args.device, output_dir)
    torch.set_num_threads(int(protocol["runtime"]["torch_threads"]))
    torch.set_num_interop_threads(int(protocol["runtime"]["torch_interop_threads"]))
    metadata = {
        "schema_version": 1,
        "status": "COMPLETE_RECOVERED_AFTER_MANIFEST_SERIALIZATION_INCIDENT",
        "manifest_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "timing_protocol_sha256": args.timing_protocol_sha256,
        "timing_script_sha256": args.timing_script_sha256,
        "manifest_finalizer_sha256": sha256(Path(__file__)),
        "metadata_dependency_benchmark_script_sha256": sha256(ROOT / "scripts/benchmark_publication_latency.py"),
        "execution_incident_sha256": sha256(incident_path),
        "timing_outputs_reused": True,
        "timing_rerun_performed": False,
        "timing_logic_changed": False,
        "recovery_scope": "metadata serialization and output-integrity validation only",
        "hardware": hardware_metadata(torch.device(args.device), protocol),
        "active_token_definition": "physical obstacle tokens with a fixed 48-slot padded network input",
        "ra_sac_internal_extra_tokens": 1,
        "ra_sac_internal_extra_token_role": "boundary token in the residual risk branch",
        "state_corpus": protocol["state_corpus"],
        "shield_parameters": protocol["shield_parameters"],
        "validated_counts": {
            "timing_cells": len(expected_cells(protocol, args.device)),
            "raw_samples": sum(expected_cells(protocol, args.device).values()),
        },
        "output_sha256": {
            raw_path.name: sha256(raw_path),
            summary_path.name: sha256(summary_path),
        },
    }
    manifest_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8", newline="\n")
    print(f"LATENCY MANIFEST RECOVERY: PASS device={args.device} output={output_dir}")


def self_test() -> None:
    protocol = {
        "active_token_counts": [8, 16, 24, 32, 40, 48],
        "timing": {"fast_iterations": 2000, "shield_iterations": 500, "end_to_end_iterations": 500},
    }
    expected = expected_cells(protocol, "cpu")
    assert len(expected) == 186 and sum(expected.values()) == 183000
    print("SELF_TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timing-script-sha256")
    parser.add_argument("--timing-protocol-sha256")
    parser.add_argument("--incident-record", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = (args.protocol, args.output_dir, args.timing_script_sha256, args.timing_protocol_sha256, args.incident_record)
    if any(value is None for value in required):
        parser.error("all recovery inputs are required")
    finalize(args)


if __name__ == "__main__":
    main()
