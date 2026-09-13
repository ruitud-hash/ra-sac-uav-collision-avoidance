"""Evaluate a saved attention-based SAC checkpoint."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.observation import attention_observation, attention_observation_spec
from agents.sac_attention import RISK_DIAGNOSTIC_FIELDS, SACAttentionAgent
from envs import UAV2DEnv
from utils.action_safety import ActionSafetyConfig, filter_static_boundary_action
from utils.artifact_audit import evaluation_provenance, write_evaluation_audit_records
from utils.checkpoint_compat import (
    observation_spec_metadata,
    validate_checkpoint_observation_spec,
)
from utils.experiment_protocol import (
    FORMAL_FINAL_TEST_EPISODES,
    FORMAL_FINAL_TEST_SEED,
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
    resolved_environment_config,
    stage1_arm_metadata,
)
from utils.naming import (
    algorithm_display_name,
    checkpoint_role,
    config_stem,
    episode_figure_name,
    precise_timestamp,
)
from utils.training_logs import evaluation_audit_fields, outcome_flags
from utils.visualization import plot_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an attention-based SAC checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--env-config", default="configs/env_sanity_shaped.yaml", help="Environment config path.")
    parser.add_argument("--train-config", default="configs/train_sac_attention_sanity_shaped.yaml")
    parser.add_argument("--episodes", type=int, default=FORMAL_FINAL_TEST_EPISODES)
    parser.add_argument("--seed", type=int, default=FORMAL_FINAL_TEST_SEED)
    parser.add_argument(
        "--evaluation-role",
        choices=("development_screen", "final_test", "publication_preflight", "publication_confirmation"),
        default="final_test",
    )
    parser.add_argument(
        "--shield-mode",
        choices=("policy_only", "static_boundary_only", "dynamic_ttc_only", "full_shield"),
        help="Explicit publication mode; cannot be combined with legacy shield flags.",
    )
    parser.add_argument("--method", help="Canonical method label required for publication roles.")
    parser.add_argument("--training-metadata", help="Training-time protocol_metadata.yaml for seed provenance.")
    parser.add_argument("--training-config-provenance", help="Actual frozen per-seed training config, when retained.")
    parser.add_argument("--checkpoint-step", type=int, help="Declared fixed-budget checkpoint step.")
    parser.add_argument("--output-root", help="Override the default outputs/checkpoint_eval root.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-figures", action="store_true")
    parser.add_argument("--action-safety-filter", action="store_true", help="Filter actions using one-step static/boundary checks.")
    parser.add_argument("--safety-static-margin", type=float, default=FORMAL_SHIELD_STATIC_MARGIN_M)
    parser.add_argument("--safety-boundary-margin", type=float, default=FORMAL_SHIELD_BOUNDARY_MARGIN_M)
    parser.add_argument("--dynamic-safety-filter", action="store_true", help="Extend action safety filtering to dynamic UAV risks.")
    parser.add_argument("--safety-dynamic-margin", type=float, default=FORMAL_SHIELD_DYNAMIC_MARGIN_M)
    parser.add_argument("--safety-dynamic-ttc-margin", type=float, default=FORMAL_SHIELD_DYNAMIC_TTC_MARGIN_S)
    parser.add_argument("--safety-dynamic-risk-threshold", type=float, default=FORMAL_SHIELD_DYNAMIC_RISK_THRESHOLD)
    parser.add_argument("--safety-omega-samples", type=int, default=FORMAL_SHIELD_OMEGA_SAMPLES)
    parser.add_argument("--safety-lookahead-steps", type=int, default=FORMAL_SHIELD_LOOKAHEAD_STEPS)
    parser.add_argument(
        "--safety-violation-weight",
        type=float,
        default=FORMAL_SHIELD_VIOLATION_WEIGHT,
        help="Penalty weight for normalized safety-margin violations when choosing shield actions.",
    )
    parser.add_argument("--progress-interval", type=int, default=10, help="Print progress every N episodes; use 0 to disable.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configured_evaluation_protocol(
    train_config: dict,
    *,
    role: str,
    env_config_path: str,
    seed: int,
    episodes: int,
    evaluation_mode: str,
) -> dict:
    protocol_path = train_config.get("evaluation_protocol")
    if not protocol_path:
        return {"role": role}
    protocol = load_yaml(PROJECT_ROOT / protocol_path)
    expected = protocol[role]
    actual = {
        "env_config": Path(env_config_path).as_posix(),
        "seed_start": int(seed),
        "episodes": int(episodes),
        "evaluation_mode": evaluation_mode,
    }
    for field, value in actual.items():
        if field in expected and expected[field] != value:
            raise ValueError(
                f"Evaluation protocol mismatch for {role}.{field}: "
                f"expected {expected[field]!r}, got {value!r}"
            )
    return {"role": expected["role"], "protocol_config": str(protocol_path), **actual}


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_agent(env_config: dict, train_config: dict, checkpoint_path: Path, device: torch.device) -> SACAttentionAgent:
    env = UAV2DEnv(deepcopy(env_config))
    spec = attention_observation_spec(env)
    action_scale = np.array([env.a_max, env.omega_max], dtype=np.float32)
    cfg = train_config["training"]
    algorithm_config = train_config.get("algorithm", {})
    risk_bias_mode = algorithm_config.get("risk_bias_mode", "mlp" if algorithm_config.get("use_risk_bias", False) else "none")
    risk_bias_clip_value = algorithm_config.get("risk_bias_clip")
    risk_bias_clip = None if risk_bias_clip_value is None else float(risk_bias_clip_value)
    goal_guidance = algorithm_config.get("goal_guidance", {})
    agent = SACAttentionAgent(
        obs_dim=spec["obs_dim"],
        global_dim=spec["global_dim"],
        max_tokens=spec["max_tokens"],
        token_dim=spec["token_dim"],
        act_dim=2,
        action_scale=action_scale,
        hidden_sizes=cfg["hidden_sizes"],
        embed_dim=int(cfg["embed_dim"]),
        risk_bias_mode=risk_bias_mode,
        risk_bias_scale=float(algorithm_config.get("risk_bias_scale", 1.0)),
        risk_bias_clip=risk_bias_clip,
        risk_gate_boundary_norm=float(algorithm_config.get("risk_gate_boundary_norm", 0.06)),
        action_residual_config=algorithm_config.get("action_residual"),
        goal_guidance=goal_guidance,
        actor_lr=float(cfg["actor_lr"]),
        critic_lr=float(cfg["critic_lr"]),
        alpha_lr=float(cfg["alpha_lr"]),
        gamma=float(cfg["gamma"]),
        tau=float(cfg["tau"]),
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    validate_checkpoint_observation_spec(
        checkpoint,
        observation_spec_metadata(
            architecture="sac_attention",
            obs_dim=spec["obs_dim"],
            global_dim=spec["global_dim"],
            max_tokens=spec["max_tokens"],
            token_dim=spec["token_dim"],
        ),
        checkpoint_label=str(checkpoint_path),
    )
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.q1.load_state_dict(checkpoint["q1"])
    agent.q2.load_state_dict(checkpoint["q2"])
    agent.q1_target.load_state_dict(agent.q1.state_dict())
    agent.q2_target.load_state_dict(agent.q2.state_dict())
    agent.log_alpha.data.copy_(checkpoint["log_alpha"].to(device))
    return agent


def main() -> None:
    args = parse_args()
    publication_role = args.evaluation_role.startswith("publication_")
    if publication_role and (not args.training_metadata or not args.method or args.checkpoint_step is None):
        raise ValueError("Publication evaluation requires --training-metadata, --method, and --checkpoint-step")
    if args.shield_mode and (args.action_safety_filter or args.dynamic_safety_filter):
        raise ValueError("--shield-mode cannot be combined with legacy shield flags")
    if args.dynamic_safety_filter and not args.action_safety_filter:
        raise ValueError("Legacy --dynamic-safety-filter requires --action-safety-filter")

    model_config_path = (PROJECT_ROOT / args.train_config).resolve()
    environment_config_path = (PROJECT_ROOT / args.env_config).resolve()
    train_config = load_yaml(model_config_path)
    stage1_arm_metadata(train_config)
    env_config = resolved_environment_config(
        load_yaml(environment_config_path), train_config
    )
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    if args.shield_mode:
        static_boundary_enabled = args.shield_mode in {"static_boundary_only", "full_shield"}
        dynamic_enabled = args.shield_mode in {"dynamic_ttc_only", "full_shield"}
        shield_enabled = args.shield_mode != "policy_only"
        evaluation_mode = args.shield_mode
    else:
        static_boundary_enabled = True
        dynamic_enabled = bool(args.dynamic_safety_filter)
        shield_enabled = bool(args.action_safety_filter)
        evaluation_mode = "shielded" if shield_enabled else "policy_only"

    training_metadata_path = None if args.training_metadata is None else (PROJECT_ROOT / args.training_metadata).resolve()
    training_config_path = (
        None
        if args.training_config_provenance is None
        else (PROJECT_ROOT / args.training_config_provenance).resolve()
    )
    method_name = args.method or algorithm_display_name(str(train_config.get("algorithm", {}).get("name", "sac_attention")))
    provenance = evaluation_provenance(
        checkpoint_path=checkpoint_path,
        checkpoint_step=args.checkpoint_step,
        model_config_path=model_config_path,
        environment_config_path=environment_config_path,
        train_config=train_config,
        training_metadata_path=training_metadata_path,
        training_config_path=training_config_path,
        method=method_name,
        evaluation_mode=evaluation_mode,
        evaluation_seed_start=args.seed,
        episodes=args.episodes,
    )
    train_config["seed"] = int(provenance["training_seed"])
    final_test_protocol = ensure_final_test_seeds_disjoint(
        train_config,
        final_seed=args.seed,
        final_episodes=args.episodes,
    )
    device = select_device(args.device)

    agent = build_agent(env_config, train_config, checkpoint_path, device)
    safety_config = ActionSafetyConfig(
        enabled=shield_enabled,
        static_boundary_enabled=static_boundary_enabled,
        static_margin_m=float(args.safety_static_margin),
        boundary_margin_m=float(args.safety_boundary_margin),
        dynamic_margin_m=float(args.safety_dynamic_margin),
        dynamic_ttc_margin_s=float(args.safety_dynamic_ttc_margin),
        dynamic_risk_threshold=float(args.safety_dynamic_risk_threshold),
        omega_samples=int(args.safety_omega_samples),
        lookahead_steps=int(args.safety_lookahead_steps),
        dynamic_enabled=dynamic_enabled,
        safety_violation_weight=float(args.safety_violation_weight),
    )
    final_test_protocol.update(
        configured_evaluation_protocol(
            train_config,
            role=args.evaluation_role,
            env_config_path=args.env_config,
            seed=args.seed,
            episodes=args.episodes,
            evaluation_mode=evaluation_mode,
        )
    )
    run_timestamp = precise_timestamp()
    env_name = config_stem(args.env_config)
    algorithm_name = str(train_config.get("algorithm", {}).get("name", "sac_attention"))
    train_seed = int(train_config["seed"])
    role = checkpoint_role(checkpoint_path)
    if publication_role:
        method_slug = method_name.lower().replace("-", "_").replace(".", "_")
        artifact_name = f"{env_name}_{method_slug}_seed_{train_seed}_{evaluation_mode}_{run_timestamp}"
    else:
        artifact_name = f"{env_name}_{algorithm_name}_{role}_{evaluation_mode}_{run_timestamp}"
    output_root = PROJECT_ROOT / (args.output_root or "outputs/checkpoint_eval")
    output_dir = output_root / artifact_name
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir()
    tables_dir.mkdir()
    write_evaluation_audit_records(
        output_dir,
        env_config=env_config,
        train_config=train_config,
        observation_spec=agent.observation_spec,
        checkpoint_path=checkpoint_path,
        final_seed=args.seed,
        final_episodes=args.episodes,
        provenance=provenance,
    )

    rows = []
    example_envs = {}
    best_fhp_env = None
    best_fhp_episode = None
    best_fhp = -1
    started_at = perf_counter()
    last_progress_at = started_at
    for episode in range(args.episodes):
        env = UAV2DEnv(deepcopy(env_config))
        obs_dict = env.reset(seed=args.seed + episode)
        obs = attention_observation(obs_dict, env.world_size)
        done = False
        total_reward = 0.0
        safety_filter_modifications = 0
        safety_filter_correction_sum = 0.0
        step_risk_diagnostics = []
        info = env.info()
        while not done:
            raw_action = agent.act(obs, deterministic=True)
            diagnostics = agent.actor.encoder.risk_diagnostics()
            if diagnostics:
                step_risk_diagnostics.append(diagnostics)
            action, modified = filter_static_boundary_action(env, raw_action, safety_config)
            safety_filter_modifications += int(modified)
            if modified:
                safety_filter_correction_sum += float(np.linalg.norm(np.asarray(action) - np.asarray(raw_action)))
            next_obs_dict, reward, done, info = env.step(action)
            obs = attention_observation(next_obs_dict, env.world_size)
            total_reward += reward

        safety_filter_correction_mean = (
            safety_filter_correction_sum / safety_filter_modifications if safety_filter_modifications > 0 else 0.0
        )
        row = {
            "episode": episode,
            "seed": args.seed + episode,
            "outcome": info["outcome"],
            **outcome_flags(info["outcome"]),
            "steps": info["steps"],
            "path_length_m": info["path_length_m"],
            "fhp_count": info["fhp_count"],
            "fhp_rate": info["fhp_count"] / max(int(info["steps"]), 1),
            "goal_distance_m": info["goal_distance_m"],
            "min_goal_distance_m": info["min_goal_distance_m"],
            "min_path_static_clearance_m": info["min_path_static_clearance_m"],
            "safety_filter_modifications": safety_filter_modifications,
            "safety_filter_intervention_rate": safety_filter_modifications / max(int(info["steps"]), 1),
            "safety_filter_correction_sum": safety_filter_correction_sum,
            "safety_filter_correction_mean": safety_filter_correction_mean,
            "shield_intervention_count": safety_filter_modifications,
            "shield_intervention_rate": safety_filter_modifications / max(int(info["steps"]), 1),
            "shield_correction_magnitude": safety_filter_correction_mean,
            "total_reward": total_reward,
            **_summarize_risk_diagnostics(step_risk_diagnostics),
            **evaluation_audit_fields(info),
        }
        rows.append(row)
        example_envs.setdefault(info["outcome"], {"episode": episode, "env": env})
        if info["fhp_count"] > best_fhp:
            best_fhp = int(info["fhp_count"])
            best_fhp_env = env
            best_fhp_episode = episode
        if args.progress_interval > 0 and ((episode + 1) % args.progress_interval == 0 or episode + 1 == args.episodes):
            last_progress_at = _print_progress(
                rows=rows,
                completed=episode + 1,
                total=args.episodes,
                started_at=started_at,
                last_progress_at=last_progress_at,
            )

    outcomes = {row["outcome"]: sum(1 for item in rows if item["outcome"] == row["outcome"]) for row in rows}
    collisions = outcomes.get("dynamic_collision", 0) + outcomes.get("static_collision", 0)
    timeout_rows = [row for row in rows if row["outcome"] == "timeout"]
    success_rows = [row for row in rows if row["outcome"] == "success"]
    summary = {
        **provenance,
        "method": method_name,
        "algorithm_name": algorithm_name,
        "training_seed": train_seed,
        "checkpoint_role": role,
        "checkpoint_path": str(checkpoint_path),
        "evaluation_mode": evaluation_mode,
        "environment_protocol": evaluation_protocol_metadata(env_config),
        "evaluation_protocol": final_test_protocol,
        "episodes": args.episodes,
        "success_rate": outcomes.get("success", 0) / args.episodes,
        "collision_rate": collisions / args.episodes,
        "dynamic_collision_rate": outcomes.get("dynamic_collision", 0) / args.episodes,
        "static_collision_rate": outcomes.get("static_collision", 0) / args.episodes,
        "timeout_rate": outcomes.get("timeout", 0) / args.episodes,
        "out_of_bounds_rate": outcomes.get("out_of_bounds", 0) / args.episodes,
        "safety_failure_rate": (collisions + outcomes.get("out_of_bounds", 0)) / args.episodes,
        "avg_reward": float(np.mean([row["total_reward"] for row in rows])),
        "avg_steps": float(np.mean([row["steps"] for row in rows])),
        "avg_path_length_m": float(np.mean([row["path_length_m"] for row in rows])),
        "avg_fhp": float(np.mean([row["fhp_count"] for row in rows])),
        "avg_fhp_rate": float(np.mean([row["fhp_rate"] for row in rows])),
        "avg_max_prediction_sigma_m": float(np.mean([row["max_prediction_sigma_m"] for row in rows])),
        "max_prediction_sigma_m": float(np.max([row["max_prediction_sigma_m"] for row in rows])),
        "avg_max_conservative_risk": float(np.mean([row["max_conservative_risk"] for row in rows])),
        "max_conservative_risk": float(np.max([row["max_conservative_risk"] for row in rows])),
        "avg_final_goal_distance_m": float(np.mean([row["goal_distance_m"] for row in rows])),
        "avg_min_goal_distance_m": float(np.mean([row["min_goal_distance_m"] for row in rows])),
        "avg_min_path_static_clearance_m": float(np.mean([row["min_path_static_clearance_m"] for row in rows])),
        "avg_safety_filter_modifications": float(np.mean([row["safety_filter_modifications"] for row in rows])),
        "total_safety_filter_modifications": int(sum(row["safety_filter_modifications"] for row in rows)),
        "shield_intervention_frequency": float(np.mean([row["safety_filter_modifications"] for row in rows])),
        "avg_shield_intervention_count": float(np.mean([row["shield_intervention_count"] for row in rows])),
        "shield_intervention_rate": float(
            sum(row["safety_filter_modifications"] for row in rows) / max(sum(row["steps"] for row in rows), 1)
        ),
        "avg_shield_intervention_rate": float(np.mean([row["shield_intervention_rate"] for row in rows])),
        "shield_correction_magnitude": float(
            sum(row["safety_filter_correction_sum"] for row in rows)
            / max(sum(row["safety_filter_modifications"] for row in rows), 1)
        ),
        "avg_shield_correction_magnitude": float(np.mean([row["shield_correction_magnitude"] for row in rows])),
        "shield_metrics_applicable": bool(safety_config.enabled),
        "timeout_avg_final_goal_distance_m": _mean_or_none(timeout_rows, "goal_distance_m"),
        "timeout_avg_min_goal_distance_m": _mean_or_none(timeout_rows, "min_goal_distance_m"),
        "timeout_avg_min_path_static_clearance_m": _mean_or_none(timeout_rows, "min_path_static_clearance_m"),
        "success_avg_steps": _mean_or_none(success_rows, "steps"),
        "outcomes": outcomes,
        "action_safety_filter": {
            "enabled": safety_config.enabled,
            "static_boundary_enabled": safety_config.static_boundary_enabled,
            "static_margin_m": safety_config.static_margin_m,
            "boundary_margin_m": safety_config.boundary_margin_m,
            "omega_samples": safety_config.omega_samples,
            "lookahead_steps": safety_config.lookahead_steps,
            "dynamic_enabled": safety_config.dynamic_enabled,
            "dynamic_margin_m": safety_config.dynamic_margin_m,
            "dynamic_ttc_margin_s": safety_config.dynamic_ttc_margin_s,
            "dynamic_risk_threshold": safety_config.dynamic_risk_threshold,
            "safety_violation_weight": safety_config.safety_violation_weight,
        },
    }

    if sum(outcomes.values()) != args.episodes:
        raise RuntimeError("Terminal outcome counts do not equal the requested episodes")
    if not np.isclose(
        summary["collision_rate"],
        summary["dynamic_collision_rate"] + summary["static_collision_rate"],
    ):
        raise RuntimeError("Collision-rate partition mismatch")
    if not np.isclose(
        summary["safety_failure_rate"],
        summary["collision_rate"] + summary["out_of_bounds_rate"],
    ):
        raise RuntimeError("Safety-failure definition mismatch")
    _assert_finite(summary)

    if rows and "risk_gate_mean" in rows[0]:
        summary["risk_diagnostics"] = {
            field: _nanmean([row[field] for row in rows])
            for field in RISK_DIAGNOSTIC_FIELDS[:-1]
        }

    csv_path = tables_dir / ("episodes.csv" if publication_role else f"{artifact_name}_eval.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = tables_dir / ("summary.yaml" if publication_role else f"{artifact_name}_summary.yaml")
    with summary_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(summary, file, sort_keys=False)

    if args.save_figures and best_fhp_env is not None:
        figure_name = episode_figure_name(
            config_name=env_name,
            policy_name=algorithm_name,
            purpose="highest_fhp",
            episode_index=best_fhp_episode,
            outcome=best_fhp_env.metrics.outcome,
            created_at=run_timestamp,
        )
        plot_episode(best_fhp_env, figures_dir / figure_name, title=f"{env_name} | {method_name} checkpoint | highest FHP")
        for outcome, example in example_envs.items():
            env = example["env"]
            example_name = episode_figure_name(
                config_name=env_name,
                policy_name=algorithm_name,
                purpose="outcome_example",
                episode_index=example["episode"],
                outcome=outcome,
                created_at=run_timestamp,
            )
            plot_episode(env, figures_dir / example_name, title=f"{env_name} | {method_name} checkpoint | {outcome}")

    print(f"{method_name} checkpoint evaluation")
    print(f"Algorithm run: {algorithm_name}")
    print(f"Training seed: {train_seed}")
    print(f"Evaluation mode: {evaluation_mode}")
    print(f"Device: {device}")
    print(f"Success rate: {summary['success_rate']:.3f}")
    print(f"Collision rate: {summary['collision_rate']:.3f}")
    print(f"  Dynamic collision rate: {summary['dynamic_collision_rate']:.3f}")
    print(f"  Static collision rate: {summary['static_collision_rate']:.3f}")
    print(f"Timeout rate: {summary['timeout_rate']:.3f}")
    print(f"Out-of-bounds rate: {summary['out_of_bounds_rate']:.3f}")
    print(f"Average reward: {summary['avg_reward']:.3f}")
    print(f"Average steps: {summary['avg_steps']:.1f}")
    print(f"Average path length: {summary['avg_path_length_m']:.1f} m")
    print(f"Average FHP: {summary['avg_fhp']:.2f}")
    print(f"Average direct-path static clearance: {summary['avg_min_path_static_clearance_m']:.1f} m")
    print(f"Action safety filter: {'enabled' if safety_config.enabled else 'disabled'}")
    print(f"Dynamic safety filter: {'enabled' if safety_config.dynamic_enabled else 'disabled'}")
    print(f"Average action safety modifications: {summary['avg_safety_filter_modifications']:.2f}")
    print(f"Total action safety modifications: {summary['total_safety_filter_modifications']}")
    print(f"Timeout avg final goal distance: {summary['timeout_avg_final_goal_distance_m']}")
    print(f"Timeout avg min goal distance: {summary['timeout_avg_min_goal_distance_m']}")
    print("Outcome counts:")
    for outcome, count in sorted(summary["outcomes"].items()):
        print(f"  {outcome}: {count}")
    print(f"CSV file: {csv_path}")
    print(f"Summary file: {summary_path}")
    if args.save_figures:
        print(f"Figures directory: {figures_dir}")
    print(f"Output directory: {output_dir}")


def _nanmean(values: list[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _assert_finite(value, path: str = "summary") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise RuntimeError(f"Non-finite evaluation output: {path}={value}")


def _summarize_risk_diagnostics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    gates = np.asarray([record["risk_gate_mean"] for record in records], dtype=np.float64)
    summary = {
        field: _nanmean([record[field] for record in records])
        for field in RISK_DIAGNOSTIC_FIELDS[:-1]
    }
    summary.update(
        risk_gate_mean=float(np.mean(gates)),
        risk_gate_p10=float(np.quantile(gates, 0.1)),
        risk_gate_p50=float(np.quantile(gates, 0.5)),
        risk_gate_p90=float(np.quantile(gates, 0.9)),
    )
    return summary


def _mean_or_none(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([row[key] for row in rows]))


def _print_progress(
    rows: list[dict],
    completed: int,
    total: int,
    started_at: float,
    last_progress_at: float,
) -> float:
    now = perf_counter()
    elapsed = now - started_at
    recent_elapsed = now - last_progress_at
    outcomes = {row["outcome"]: sum(1 for item in rows if item["outcome"] == row["outcome"]) for row in rows}
    collisions = outcomes.get("dynamic_collision", 0) + outcomes.get("static_collision", 0)
    completed = max(completed, 1)
    avg_seconds = elapsed / completed
    remaining = max(total - completed, 0) * avg_seconds
    print(
        "Progress "
        f"{completed}/{total} | "
        f"success={outcomes.get('success', 0) / completed:.3f} | "
        f"collision={collisions / completed:.3f} | "
        f"timeout={outcomes.get('timeout', 0) / completed:.3f} | "
        f"recent={recent_elapsed:.1f}s | "
        f"eta={remaining / 60.0:.1f}min",
        flush=True,
    )
    return now


if __name__ == "__main__":
    main()
