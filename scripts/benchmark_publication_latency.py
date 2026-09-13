"""Batch-1 online latency/scalability benchmark for publication reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

from agents.observation import attention_observation, flatten_observation  # noqa: E402
from envs import UAV2DEnv  # noqa: E402
from evaluate_publication_mlp_baseline import build_agent as build_mlp_agent  # noqa: E402
from evaluate_sac_attention_checkpoint import build_agent as build_attention_agent  # noqa: E402
from utils.action_safety import ActionSafetyConfig, filter_static_boundary_action  # noqa: E402
from utils.experiment_protocol import (  # noqa: E402
    FORMAL_SHIELD_BOUNDARY_MARGIN_M,
    FORMAL_SHIELD_DYNAMIC_MARGIN_M,
    FORMAL_SHIELD_DYNAMIC_RISK_THRESHOLD,
    FORMAL_SHIELD_DYNAMIC_TTC_MARGIN_S,
    FORMAL_SHIELD_LOOKAHEAD_STEPS,
    FORMAL_SHIELD_OMEGA_SAMPLES,
    FORMAL_SHIELD_STATIC_MARGIN_M,
    FORMAL_SHIELD_VIOLATION_WEIGHT,
    resolved_environment_config,
)


METHODS = ("SAC-MLP", "SAC-Attention-v3", "RA-SAC-v3.2-c")
SHIELD_MODES = ("static_boundary_only", "dynamic_ttc_only", "full_shield")
E2E_MODES = ("policy_only", *SHIELD_MODES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(call, *, warmup: int, iterations: int, device: torch.device) -> list[float]:
    for index in range(warmup):
        call(index)
    synchronize(device)
    samples = []
    for index in range(iterations):
        synchronize(device)
        started = perf_counter_ns()
        call(index)
        synchronize(device)
        samples.append((perf_counter_ns() - started) / 1_000_000.0)
    return samples


def split_obstacles(token_count: int) -> tuple[int, int]:
    dynamic = round(token_count * 14 / 32)
    return dynamic, token_count - dynamic


def make_environment_config(base: dict, token_count: int) -> tuple[dict, int, int]:
    dynamic, static = split_obstacles(token_count)
    config = deepcopy(base)
    config["obstacles"]["dynamic_count"] = dynamic
    config["obstacles"]["static_count"] = static
    config["perception"]["sense_radius_m"] = float(config["world"]["size_m"]) * 2.0
    return config, dynamic, static


def build_states(config: dict, token_count: int, state_count: int, seed: int) -> list[UAV2DEnv]:
    states = []
    for index in range(state_count):
        env = UAV2DEnv(deepcopy(config))
        env.reset(seed=seed + token_count * 1000 + index)
        for _ in range(index % 8):
            _, _, done, _ = env.step(env.sample_goal_seeking_action())
            if done:
                env.reset(seed=seed + token_count * 1000 + index + 100_000)
        raw = env._observation()
        assert len(env.dynamic_uavs) + len(env.static_obstacles) == token_count
        assert int(raw["mask"].sum()) == token_count
        states.append(env)
    return states


def shield_config(mode: str) -> ActionSafetyConfig:
    return ActionSafetyConfig(
        enabled=True,
        static_boundary_enabled=mode in {"static_boundary_only", "full_shield"},
        static_margin_m=FORMAL_SHIELD_STATIC_MARGIN_M,
        boundary_margin_m=FORMAL_SHIELD_BOUNDARY_MARGIN_M,
        dynamic_margin_m=FORMAL_SHIELD_DYNAMIC_MARGIN_M,
        dynamic_ttc_margin_s=FORMAL_SHIELD_DYNAMIC_TTC_MARGIN_S,
        dynamic_risk_threshold=FORMAL_SHIELD_DYNAMIC_RISK_THRESHOLD,
        omega_samples=FORMAL_SHIELD_OMEGA_SAMPLES,
        lookahead_steps=FORMAL_SHIELD_LOOKAHEAD_STEPS,
        dynamic_enabled=mode in {"dynamic_ttc_only", "full_shield"},
        safety_violation_weight=FORMAL_SHIELD_VIOLATION_WEIGHT,
    )


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def nvidia_value(query: str) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.splitlines()[0].strip()
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        return None


def hardware_metadata(device: torch.device, protocol: dict) -> dict:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "process_cpu_affinity": affinity,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_driver_version": nvidia_value("driver_version") if device.type == "cuda" else None,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "cuda_synchronized_each_sample": device.type == "cuda",
        "batch_size": 1,
        "single_process": True,
        "warmup_iterations": int(protocol["timing"]["warmup_iterations"]),
        "fast_iterations": int(protocol["timing"]["fast_iterations"]),
        "shield_iterations": int(protocol["timing"]["shield_iterations"]),
        "end_to_end_iterations": int(protocol["timing"]["end_to_end_iterations"]),
        "thread_environment": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def load_agents(protocol: dict, base_env: dict, device: torch.device) -> dict:
    agents = {}
    for method in METHODS:
        spec = protocol["methods"][method]
        train_config = load_yaml(ROOT / spec["train_config"])
        checkpoint = ROOT / spec["checkpoint"]
        assert sha256(checkpoint) == spec["checkpoint_sha256"]
        if method == "SAC-MLP":
            agent = build_mlp_agent(method, base_env, train_config, checkpoint, device)
        else:
            env_config = resolved_environment_config(deepcopy(base_env), train_config)
            agent = build_attention_agent(env_config, train_config, checkpoint, device)
        agent.actor.eval()
        agents[method] = agent
    return agents


def pack(method: str, observation: dict, world_size: float) -> np.ndarray:
    return (
        flatten_observation(observation, world_size)
        if method == "SAC-MLP"
        else attention_observation(observation, world_size)
    )


def add_cell(
    raw_writer: csv.DictWriter,
    summary_rows: list[dict],
    *,
    samples: list[float],
    device: torch.device,
    token_count: int,
    dynamic_count: int,
    static_count: int,
    method: str,
    component: str,
    shield_mode: str,
    modified_state_rate: float | None = None,
) -> None:
    for index, latency in enumerate(samples):
        raw_writer.writerow({
            "device": device.type, "active_tokens": token_count, "dynamic_obstacles": dynamic_count,
            "static_obstacles": static_count, "method": method, "component": component,
            "shield_mode": shield_mode, "iteration": index, "latency_ms": latency,
        })
    stats = summarize(samples)
    summary_rows.append({
        "device": device.type, "active_tokens": token_count, "dynamic_obstacles": dynamic_count,
        "static_obstacles": static_count, "method": method, "component": component,
        "shield_mode": shield_mode, "iterations": len(samples), **stats,
        "p95_compatible_hz": 1000.0 / stats["p95_ms"] if component == "end_to_end" else "",
        "p99_compatible_hz": 1000.0 / stats["p99_ms"] if component == "end_to_end" else "",
        "shield_modified_state_rate": "" if modified_state_rate is None else modified_state_rate,
    })


def benchmark(protocol: dict, device: torch.device, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    base_env = load_yaml(ROOT / protocol["environment_config"])
    assert sha256(ROOT / protocol["environment_config"]) == protocol["environment_config_sha256"]
    agents = load_agents(protocol, base_env, device)
    warmup = int(protocol["timing"]["warmup_iterations"])
    fast_iterations = int(protocol["timing"]["fast_iterations"])
    shield_iterations = int(protocol["timing"]["shield_iterations"])
    e2e_iterations = int(protocol["timing"]["end_to_end_iterations"])
    state_count = int(protocol["state_corpus"]["states_per_token_count"])
    state_seed = int(protocol["state_corpus"]["seed"])
    shield_configs = {mode: shield_config(mode) for mode in SHIELD_MODES}
    summary_rows = []
    raw_path = output_dir / "raw_latency_samples.csv"
    raw_fields = (
        "device", "active_tokens", "dynamic_obstacles", "static_obstacles", "method",
        "component", "shield_mode", "iteration", "latency_ms",
    )
    with raw_path.open("w", newline="", encoding="utf-8") as raw_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_fields)
        raw_writer.writeheader()
        for token_count in protocol["active_token_counts"]:
            env_config, dynamic_count, static_count = make_environment_config(base_env, int(token_count))
            states = build_states(env_config, int(token_count), state_count, state_seed)
            raw_observations = [env._observation() for env in states]
            packed = {
                method: [pack(method, observation, env.world_size) for observation, env in zip(raw_observations, states)]
                for method in METHODS
            }
            actions = {
                method: [agents[method].act(observation, deterministic=True) for observation in packed[method]]
                for method in METHODS
            }
            synchronize(device)

            samples = measure(
                lambda i: states[i % state_count]._observation(),
                warmup=warmup, iterations=fast_iterations, device=device,
            )
            add_cell(raw_writer, summary_rows, samples=samples, device=device, token_count=token_count,
                     dynamic_count=dynamic_count, static_count=static_count, method="shared",
                     component="observation_raw", shield_mode="")

            for method in METHODS:
                samples = measure(
                    lambda i, m=method: pack(m, raw_observations[i % state_count], states[i % state_count].world_size),
                    warmup=warmup, iterations=fast_iterations, device=device,
                )
                add_cell(raw_writer, summary_rows, samples=samples, device=device, token_count=token_count,
                         dynamic_count=dynamic_count, static_count=static_count, method=method,
                         component="observation_pack", shield_mode="")
                samples = measure(
                    lambda i, m=method: pack(m, states[i % state_count]._observation(), states[i % state_count].world_size),
                    warmup=warmup, iterations=fast_iterations, device=device,
                )
                add_cell(raw_writer, summary_rows, samples=samples, device=device, token_count=token_count,
                         dynamic_count=dynamic_count, static_count=static_count, method=method,
                         component="observation_total", shield_mode="")
                samples = measure(
                    lambda i, m=method: agents[m].act(packed[m][i % state_count], deterministic=True),
                    warmup=warmup, iterations=fast_iterations, device=device,
                )
                add_cell(raw_writer, summary_rows, samples=samples, device=device, token_count=token_count,
                         dynamic_count=dynamic_count, static_count=static_count, method=method,
                         component="policy", shield_mode="")

                modified_rates = {}
                for mode, config in shield_configs.items():
                    modified_rates[mode] = float(np.mean([
                        filter_static_boundary_action(env, action, config)[1]
                        for env, action in zip(states, actions[method])
                    ]))
                    samples = measure(
                        lambda i, m=method, c=config: filter_static_boundary_action(
                            states[i % state_count], actions[m][i % state_count], c
                        ),
                        warmup=warmup, iterations=shield_iterations, device=device,
                    )
                    add_cell(raw_writer, summary_rows, samples=samples, device=device, token_count=token_count,
                             dynamic_count=dynamic_count, static_count=static_count, method=method,
                             component="shield", shield_mode=mode,
                             modified_state_rate=modified_rates[mode])

                for mode in E2E_MODES:
                    config = shield_configs.get(mode)

                    def online_path(index: int, m: str = method, c: ActionSafetyConfig | None = config):
                        env = states[index % state_count]
                        observation = pack(m, env._observation(), env.world_size)
                        action = agents[m].act(observation, deterministic=True)
                        return action if c is None else filter_static_boundary_action(env, action, c)[0]

                    samples = measure(
                        online_path, warmup=warmup, iterations=e2e_iterations, device=device,
                    )
                    add_cell(raw_writer, summary_rows, samples=samples, device=device, token_count=token_count,
                             dynamic_count=dynamic_count, static_count=static_count, method=method,
                             component="end_to_end", shield_mode=mode,
                             modified_state_rate=modified_rates.get(mode))

    summary_path = output_dir / "latency_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    metadata = {
        "schema_version": 1,
        "status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(Path(protocol["_path"])),
        "benchmark_script_sha256": sha256(Path(__file__)),
        "hardware": hardware_metadata(device, protocol),
        "active_token_definition": "physical obstacle tokens with a fixed 48-slot padded network input",
        "ra_sac_internal_extra_tokens": 1,
        "ra_sac_internal_extra_token_role": "boundary token in the residual risk branch",
        "state_corpus": protocol["state_corpus"],
        "shield_parameters": protocol["shield_parameters"],
        "output_sha256": {
            raw_path.name: sha256(raw_path),
            summary_path.name: sha256(summary_path),
        },
    }
    (output_dir / "benchmark_manifest.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8", newline="\n"
    )


def preflight(protocol: dict, device: torch.device) -> None:
    base_env = load_yaml(ROOT / protocol["environment_config"])
    agents = load_agents(protocol, base_env, device)
    for token_count in (8, 48):
        config, _, _ = make_environment_config(base_env, token_count)
        env = build_states(config, token_count, 1, int(protocol["state_corpus"]["seed"]))[0]
        raw = env._observation()
        for method, agent in agents.items():
            observation = pack(method, raw, env.world_size)
            action = agent.act(observation, deterministic=True)
            for mode in SHIELD_MODES:
                filter_static_boundary_action(env, action, shield_config(mode))
    synchronize(device)
    print(f"LATENCY PREFLIGHT: PASS device={device.type}")


def self_test() -> None:
    assert [split_obstacles(value) for value in (8, 16, 24, 32, 40, 48)] == [
        (4, 4), (7, 9), (10, 14), (14, 18), (18, 22), (21, 27)
    ]
    values = [1.0, 2.0, 3.0, 4.0]
    result = summarize(values)
    assert result["mean_ms"] == 2.5 and result["median_ms"] == 2.5
    print("SELF_TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.protocol or not args.device or (not args.preflight and not args.output_dir):
        parser.error("--protocol and --device are required; --output-dir is required unless --preflight is used")
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    output_dir = None if args.output_dir is None else (args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir)
    protocol = load_yaml(protocol_path)
    protocol["_path"] = str(protocol_path)
    assert protocol["status"] == "FROZEN_BEFORE_BENCHMARK"
    assert protocol["benchmark_script_sha256"] == sha256(Path(__file__))
    for relative, expected in protocol["asset_sha256"].items():
        assert sha256(ROOT / relative) == expected, f"Asset hash mismatch: {relative}"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")
    threads = int(protocol["runtime"]["torch_threads"])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(int(protocol["runtime"]["torch_interop_threads"]))
    device = torch.device(args.device)
    if args.preflight:
        preflight(protocol, device)
        return
    benchmark(protocol, device, output_dir)
    print(f"LATENCY BENCHMARK: COMPLETE device={device.type} output={output_dir}")


if __name__ == "__main__":
    main()
