"""Publication-grade evaluator for frozen SAC-MLP, TD3-MLP, and PPO-MLP checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.observation import flatten_observation
from agents.ppo_mlp import PPOMLPAgent
from agents.sac_mlp import SACMLPAgent
from agents.td3_mlp import TD3MLPAgent
from envs import UAV2DEnv
from utils.action_safety import ActionSafetyConfig, filter_static_boundary_action
from utils.artifact_audit import evaluation_provenance, write_evaluation_audit_records
from utils.checkpoint_compat import observation_spec_metadata, validate_checkpoint_observation_spec
from utils.experiment_protocol import (
    FORMAL_SHIELD_BOUNDARY_MARGIN_M,
    FORMAL_SHIELD_DYNAMIC_MARGIN_M,
    FORMAL_SHIELD_DYNAMIC_RISK_THRESHOLD,
    FORMAL_SHIELD_DYNAMIC_TTC_MARGIN_S,
    FORMAL_SHIELD_LOOKAHEAD_STEPS,
    FORMAL_SHIELD_OMEGA_SAMPLES,
    FORMAL_SHIELD_STATIC_MARGIN_M,
    FORMAL_SHIELD_VIOLATION_WEIGHT,
    ensure_final_test_seeds_disjoint,
    evaluation_protocol_metadata,
)
from utils.naming import checkpoint_role, config_stem, precise_timestamp
from utils.training_logs import evaluation_audit_fields, outcome_flags


METHODS = ("SAC-MLP", "TD3-MLP", "PPO-MLP")
MODES = ("policy_only", "full_shield")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--training-metadata", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=149000)
    parser.add_argument("--shield-mode", choices=MODES, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--safety-static-margin", type=float, default=FORMAL_SHIELD_STATIC_MARGIN_M)
    parser.add_argument("--safety-boundary-margin", type=float, default=FORMAL_SHIELD_BOUNDARY_MARGIN_M)
    parser.add_argument("--safety-dynamic-margin", type=float, default=FORMAL_SHIELD_DYNAMIC_MARGIN_M)
    parser.add_argument("--safety-dynamic-ttc-margin", type=float, default=FORMAL_SHIELD_DYNAMIC_TTC_MARGIN_S)
    parser.add_argument("--safety-dynamic-risk-threshold", type=float, default=FORMAL_SHIELD_DYNAMIC_RISK_THRESHOLD)
    parser.add_argument("--safety-omega-samples", type=int, default=FORMAL_SHIELD_OMEGA_SAMPLES)
    parser.add_argument("--safety-lookahead-steps", type=int, default=FORMAL_SHIELD_LOOKAHEAD_STEPS)
    parser.add_argument("--safety-violation-weight", type=float, default=FORMAL_SHIELD_VIOLATION_WEIGHT)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def build_agent(method: str, env_config: dict, train_config: dict, checkpoint: Path, device: torch.device):
    env = UAV2DEnv(deepcopy(env_config))
    obs = flatten_observation(env.reset(seed=0), env.world_size)
    scale = np.array([env.a_max, env.omega_max], dtype=np.float32)
    cfg = train_config["training"]
    common = dict(obs_dim=obs.shape[0], act_dim=2, action_scale=scale, hidden_sizes=list(cfg["hidden_sizes"]), device=device)
    if method == "SAC-MLP":
        agent = SACMLPAgent(
            **common,
            actor_lr=float(cfg["actor_lr"]), critic_lr=float(cfg["critic_lr"]),
            alpha_lr=float(cfg["alpha_lr"]), gamma=float(cfg["gamma"]), tau=float(cfg["tau"]),
        )
        architecture = "sac_mlp"
    elif method == "TD3-MLP":
        agent = TD3MLPAgent(
            **common,
            actor_lr=float(cfg["actor_lr"]), critic_lr=float(cfg["critic_lr"]),
            gamma=float(cfg["gamma"]), tau=float(cfg["tau"]),
            policy_noise=float(cfg["policy_noise"]), noise_clip=float(cfg["noise_clip"]),
            policy_delay=int(cfg["policy_delay"]),
        )
        architecture = "td3_mlp"
    else:
        agent = PPOMLPAgent(
            **common,
            lr=float(cfg["lr"]), clip_ratio=float(cfg["clip_ratio"]),
            value_coef=float(cfg["value_coef"]), entropy_coef=float(cfg["entropy_coef"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
        )
        architecture = "ppo_mlp"
    saved = torch.load(checkpoint, map_location=device)
    validate_checkpoint_observation_spec(
        saved,
        observation_spec_metadata(architecture=architecture, obs_dim=obs.shape[0]),
        checkpoint_label=str(checkpoint),
    )
    agent.load(str(checkpoint))
    return agent


def act(agent, method: str, observation: np.ndarray) -> np.ndarray:
    value = agent.act(observation, deterministic=True)
    return value[0] if method == "PPO-MLP" else value


def mean(rows: list[dict], field: str) -> float:
    return float(np.mean([row[field] for row in rows]))


def mean_or_none(rows: list[dict], field: str) -> float | None:
    return None if not rows else mean(rows, field)


def assert_finite(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN/Inf in evaluation output")


def self_test() -> None:
    assert set(METHODS) == {"SAC-MLP", "TD3-MLP", "PPO-MLP"}
    assert MODES == ("policy_only", "full_shield")
    assert mean([{"x": 1.0}, {"x": 3.0}], "x") == 2.0
    print("SELF_TEST: PASS")


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    args = parse_args()
    if args.episodes != 200 or args.seed != 149000:
        raise ValueError("Publication extension requires exactly seeds 149000-149199")
    checkpoint = resolve(args.checkpoint)
    train_config_path = resolve(args.train_config)
    metadata_path = resolve(args.training_metadata)
    environment_path = resolve(args.env_config)
    train_config, env_config = load_yaml(train_config_path), load_yaml(environment_path)
    provenance = evaluation_provenance(
        checkpoint_path=checkpoint,
        checkpoint_step=args.checkpoint_step,
        model_config_path=train_config_path,
        environment_config_path=environment_path,
        train_config=train_config,
        training_metadata_path=metadata_path,
        training_config_path=train_config_path,
        method=args.method,
        evaluation_mode=args.shield_mode,
        evaluation_seed_start=args.seed,
        episodes=args.episodes,
    )
    if int(provenance["training_seed"]) != args.training_seed:
        raise ValueError("Declared training seed does not match frozen provenance")
    train_config["seed"] = args.training_seed
    final_protocol = ensure_final_test_seeds_disjoint(train_config, final_seed=args.seed, final_episodes=args.episodes)
    final_protocol.update({
        "role": "publication_confirmation_extension",
        "env_config": Path(args.env_config).as_posix(),
        "evaluation_mode": args.shield_mode,
    })
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    agent = build_agent(args.method, env_config, train_config, checkpoint, device)
    if args.load_only:
        print(f"LOAD_ONLY: PASS {args.method} seed={args.training_seed}")
        return

    enabled = args.shield_mode == "full_shield"
    safety = ActionSafetyConfig(
        enabled=enabled,
        static_boundary_enabled=enabled,
        static_margin_m=float(args.safety_static_margin),
        boundary_margin_m=float(args.safety_boundary_margin),
        dynamic_margin_m=float(args.safety_dynamic_margin),
        dynamic_ttc_margin_s=float(args.safety_dynamic_ttc_margin),
        dynamic_risk_threshold=float(args.safety_dynamic_risk_threshold),
        omega_samples=int(args.safety_omega_samples),
        lookahead_steps=int(args.safety_lookahead_steps),
        dynamic_enabled=enabled,
        safety_violation_weight=float(args.safety_violation_weight),
    )
    method_slug = args.method.lower().replace("-", "_")
    artifact = f"{config_stem(args.env_config)}_{method_slug}_seed_{args.training_seed}_{args.shield_mode}_{precise_timestamp()}"
    output_dir = resolve(args.output_root) / artifact
    tables = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=False)
    tables.mkdir()
    write_evaluation_audit_records(
        output_dir,
        env_config=env_config,
        train_config=train_config,
        observation_spec=agent.observation_spec,
        checkpoint_path=checkpoint,
        final_seed=args.seed,
        final_episodes=args.episodes,
        provenance=provenance,
    )

    rows = []
    started = perf_counter()
    for episode in range(args.episodes):
        env = UAV2DEnv(deepcopy(env_config))
        obs_dict = env.reset(seed=args.seed + episode)
        obs = flatten_observation(obs_dict, env.world_size)
        done, reward_sum, modifications, correction_sum = False, 0.0, 0, 0.0
        info = env.info()
        while not done:
            raw_action = act(agent, args.method, obs)
            action, modified = filter_static_boundary_action(env, raw_action, safety)
            modifications += int(modified)
            if modified:
                correction_sum += float(np.linalg.norm(np.asarray(action) - np.asarray(raw_action)))
            next_obs, reward, done, info = env.step(action)
            obs = flatten_observation(next_obs, env.world_size)
            reward_sum += float(reward)
        correction_mean = correction_sum / modifications if modifications else 0.0
        rows.append({
            "episode": episode,
            "seed": args.seed + episode,
            "outcome": str(info["outcome"]),
            **outcome_flags(str(info["outcome"])),
            "steps": int(info["steps"]),
            "path_length_m": float(info["path_length_m"]),
            "fhp_count": int(info["fhp_count"]),
            "fhp_rate": int(info["fhp_count"]) / max(int(info["steps"]), 1),
            "goal_distance_m": float(info["goal_distance_m"]),
            "min_goal_distance_m": float(info["min_goal_distance_m"]),
            "min_path_static_clearance_m": float(info["min_path_static_clearance_m"]),
            "safety_filter_modifications": modifications,
            "safety_filter_intervention_rate": modifications / max(int(info["steps"]), 1),
            "safety_filter_correction_sum": correction_sum,
            "safety_filter_correction_mean": correction_mean,
            "shield_intervention_count": modifications,
            "shield_intervention_rate": modifications / max(int(info["steps"]), 1),
            "shield_correction_magnitude": correction_mean,
            "total_reward": reward_sum,
            **evaluation_audit_fields(info),
        })
        if args.progress_interval and (episode + 1) % args.progress_interval == 0:
            elapsed = perf_counter() - started
            print(f"Progress {episode + 1}/{args.episodes}; elapsed={elapsed:.1f}s", flush=True)

    outcomes = {name: sum(row["outcome"] == name for row in rows) for name in ("success", "dynamic_collision", "static_collision", "timeout", "out_of_bounds")}
    collisions = outcomes["dynamic_collision"] + outcomes["static_collision"]
    timeouts = [row for row in rows if row["outcome"] == "timeout"]
    successes = [row for row in rows if row["outcome"] == "success"]
    total_modifications = sum(row["shield_intervention_count"] for row in rows)
    summary = {
        **provenance,
        "method": args.method,
        "algorithm_name": str(train_config["algorithm"]["name"]),
        "training_seed": args.training_seed,
        "checkpoint_role": checkpoint_role(checkpoint),
        "evaluation_mode": args.shield_mode,
        "environment_protocol": evaluation_protocol_metadata(env_config),
        "evaluation_protocol": final_protocol,
        "episodes": args.episodes,
        "success_rate": outcomes["success"] / args.episodes,
        "collision_rate": collisions / args.episodes,
        "dynamic_collision_rate": outcomes["dynamic_collision"] / args.episodes,
        "static_collision_rate": outcomes["static_collision"] / args.episodes,
        "timeout_rate": outcomes["timeout"] / args.episodes,
        "out_of_bounds_rate": outcomes["out_of_bounds"] / args.episodes,
        "safety_failure_rate": (collisions + outcomes["out_of_bounds"]) / args.episodes,
        "avg_reward": mean(rows, "total_reward"),
        "avg_steps": mean(rows, "steps"),
        "avg_path_length_m": mean(rows, "path_length_m"),
        "avg_fhp": mean(rows, "fhp_count"),
        "avg_fhp_rate": mean(rows, "fhp_rate"),
        "avg_max_prediction_sigma_m": mean(rows, "max_prediction_sigma_m"),
        "max_prediction_sigma_m": max(row["max_prediction_sigma_m"] for row in rows),
        "avg_max_conservative_risk": mean(rows, "max_conservative_risk"),
        "max_conservative_risk": max(row["max_conservative_risk"] for row in rows),
        "avg_final_goal_distance_m": mean(rows, "goal_distance_m"),
        "avg_min_goal_distance_m": mean(rows, "min_goal_distance_m"),
        "avg_min_path_static_clearance_m": mean(rows, "min_path_static_clearance_m"),
        "avg_safety_filter_modifications": mean(rows, "safety_filter_modifications"),
        "total_safety_filter_modifications": total_modifications,
        "shield_intervention_frequency": mean(rows, "shield_intervention_count"),
        "avg_shield_intervention_count": mean(rows, "shield_intervention_count"),
        "shield_intervention_rate": total_modifications / max(sum(row["steps"] for row in rows), 1),
        "avg_shield_intervention_rate": mean(rows, "shield_intervention_rate"),
        "shield_correction_magnitude": sum(row["safety_filter_correction_sum"] for row in rows) / max(total_modifications, 1),
        "avg_shield_correction_magnitude": mean(rows, "shield_correction_magnitude"),
        "shield_metrics_applicable": enabled,
        "timeout_avg_final_goal_distance_m": mean_or_none(timeouts, "goal_distance_m"),
        "timeout_avg_min_goal_distance_m": mean_or_none(timeouts, "min_goal_distance_m"),
        "success_avg_steps": mean_or_none(successes, "steps"),
        "outcomes": outcomes,
        "action_safety_filter": {
            "enabled": enabled,
            "static_boundary_enabled": enabled,
            "dynamic_enabled": enabled,
            "static_margin_m": safety.static_margin_m,
            "boundary_margin_m": safety.boundary_margin_m,
            "dynamic_margin_m": safety.dynamic_margin_m,
            "dynamic_ttc_margin_s": safety.dynamic_ttc_margin_s,
            "dynamic_risk_threshold": safety.dynamic_risk_threshold,
            "omega_samples": safety.omega_samples,
            "lookahead_steps": safety.lookahead_steps,
            "safety_violation_weight": safety.safety_violation_weight,
        },
    }
    if abs(summary["collision_rate"] - summary["dynamic_collision_rate"] - summary["static_collision_rate"]) > 1e-12:
        raise AssertionError("collision identity failed")
    if abs(summary["safety_failure_rate"] - summary["collision_rate"] - summary["out_of_bounds_rate"]) > 1e-12:
        raise AssertionError("safety identity failed")
    assert_finite(summary)
    with (tables / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (tables / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8", newline="\n")
    print(f"COMPLETE: {output_dir.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
