"""Train a PPO-MLP baseline on the 2D UAV environment."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml
from tqdm import trange


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.observation import flatten_observation
from agents.ppo_mlp import PPOMLPAgent
from envs import UAV2DEnv
from utils.artifact_audit import write_training_audit_records
from utils.experiment_protocol import (
    ensure_training_environment_allowed,
    training_environment_seed,
)
from utils.naming import timestamp
from utils.randomness import preserve_global_rng_state
from utils.training_logs import (
    EPISODE_CORE_FIELDS,
    EVAL_FIELDNAMES,
    eval_selection_score,
    make_episode_row,
    serialize_episode_randomization,
    summarize_eval_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO-MLP on the UAV 2D environment.")
    parser.add_argument("--config", default="configs/train_ppo_mlp_tiny.yaml")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional PPO checkpoint used to initialize the actor-critic model.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@preserve_global_rng_state
def evaluate(agent: PPOMLPAgent, env_config: dict, episodes: int, seed: int, eval_id: int) -> dict:
    rows = []
    for idx in range(episodes):
        env = UAV2DEnv(deepcopy(env_config))
        obs_dict = env.reset(seed=seed + idx)
        obs = flatten_observation(obs_dict, env.world_size)
        done = False
        total_reward = 0.0
        while not done:
            action, _, _ = agent.act(obs, deterministic=True)
            next_obs_dict, reward, done, info = env.step(action)
            obs = flatten_observation(next_obs_dict, env.world_size)
            total_reward += float(reward)
        rows.append(
            {
                "outcome": info["outcome"],
                "total_reward": total_reward,
                "steps": info["steps"],
                "path_length_m": info["path_length_m"],
                "fhp_count": info["fhp_count"],
            }
        )
    return summarize_eval_episodes(rows, eval_id=eval_id, eval_episodes=episodes)


def main() -> None:
    args = parse_args()
    train_config = load_yaml(PROJECT_ROOT / args.config)
    env_config = load_yaml(PROJECT_ROOT / train_config["env_config"])
    ensure_training_environment_allowed(env_config)
    seed = int(train_config["seed"])
    set_seed(seed)
    device = select_device(str(train_config["device"]))
    cfg = train_config["training"]
    eval_seed = int(cfg.get("eval_seed", seed + 100000))

    algorithm_name = str(train_config.get("algorithm", {}).get("name", "ppo_mlp"))
    run_id = f"{timestamp()}_{algorithm_name}"
    run_dir = PROJECT_ROOT / train_config["output"]["directory"] / run_id
    table_dir = run_dir / "tables"
    checkpoint_dir = run_dir / "checkpoints"
    table_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = UAV2DEnv(deepcopy(env_config))
    obs_dict = env.reset(seed=training_environment_seed(train_config, 0))
    obs = flatten_observation(obs_dict, env.world_size)
    obs_dim = obs.shape[0]
    action_scale = np.array([env.a_max, env.omega_max], dtype=np.float32)
    agent = PPOMLPAgent(
        obs_dim=obs_dim,
        act_dim=2,
        action_scale=action_scale,
        hidden_sizes=list(cfg["hidden_sizes"]),
        lr=float(cfg["lr"]),
        clip_ratio=float(cfg["clip_ratio"]),
        value_coef=float(cfg["value_coef"]),
        entropy_coef=float(cfg["entropy_coef"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        device=device,
    )
    init_checkpoint = args.init_checkpoint or train_config.get("init_checkpoint")
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.is_absolute():
            init_path = PROJECT_ROOT / init_path
        agent.load(str(init_path))
        print(f"Initialized actor-critic from checkpoint: {init_path}")
    write_training_audit_records(
        run_dir,
        env_config=env_config,
        train_config=train_config,
        observation_spec=agent.observation_spec,
    )

    episode_log_path = table_dir / "episode_log.csv"
    eval_log_path = table_dir / "eval_log.csv"
    with episode_log_path.open("w", encoding="utf-8", newline="") as episode_file, eval_log_path.open(
        "w", encoding="utf-8", newline=""
    ) as eval_file:
        episode_writer = csv.DictWriter(
            episode_file,
            fieldnames=[
                *EPISODE_CORE_FIELDS,
                "actor_loss",
                "critic_loss",
                "alpha_loss",
                "alpha",
                "q_mean",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "episode_randomization",
            ],
        )
        episode_writer.writeheader()
        eval_writer = csv.DictWriter(
            eval_file,
            fieldnames=EVAL_FIELDNAMES,
        )
        eval_writer.writeheader()

        best_eval_score = -float("inf")
        best_eval_metrics = None
        episode = 0
        episode_reward = 0.0
        episode_steps = 0
        last_losses = None
        eval_id = 0
        progress = trange(
            0,
            int(cfg["total_steps"]),
            int(cfg["rollout_steps"]),
            desc=f"{algorithm_name} train on {device}",
            mininterval=1.0,
        )
        global_step = 0
        for _ in progress:
            rollout = _collect_rollout(
                agent, env, obs, train_config, cfg, device, global_step
            )
            obs = rollout["last_obs_np"]
            global_step += len(rollout["rewards_np"])

            losses = agent.update(
                obs=rollout["obs"],
                actions=rollout["actions"],
                old_log_probs=rollout["log_probs"],
                returns=rollout["returns"],
                advantages=rollout["advantages"],
                epochs=int(cfg["update_epochs"]),
                minibatch_size=int(cfg["minibatch_size"]),
            )
            last_losses = losses

            for episode_row in rollout["episodes"]:
                episode_row.update(
                    {
                        "actor_loss": losses.policy_loss,
                        "critic_loss": losses.value_loss,
                        "alpha_loss": "",
                        "alpha": "",
                        "q_mean": "",
                        "policy_loss": losses.policy_loss,
                        "value_loss": losses.value_loss,
                        "entropy": losses.entropy,
                        "approx_kl": losses.approx_kl,
                    }
                )
                episode_writer.writerow(episode_row)
                episode = max(episode, int(episode_row["episode"]) + 1)
                episode_reward = 0.0
                episode_steps = 0
            episode_file.flush()

            if global_step % int(cfg["eval_interval_steps"]) < int(cfg["rollout_steps"]) or global_step >= int(cfg["total_steps"]):
                eval_id += 1
                eval_metrics = evaluate(agent, env_config, int(cfg["eval_episodes"]), eval_seed, eval_id)
                eval_metrics["eval_seed_start"] = eval_seed
                eval_metrics["eval_seed_end"] = eval_seed + int(cfg["eval_episodes"]) - 1
                eval_writer.writerow({"step": global_step, **eval_metrics})
                eval_file.flush()
                eval_score = eval_selection_score(eval_metrics)
                if eval_score > best_eval_score or (
                    np.isclose(eval_score, best_eval_score)
                    and best_eval_metrics is not None
                    and eval_metrics["eval_return_mean"] > best_eval_metrics["eval_return_mean"]
                ):
                    best_eval_score = eval_score
                    best_eval_metrics = {"step": global_step, "score": eval_score, **eval_metrics}
                    agent.save(str(checkpoint_dir / f"{algorithm_name}_best.pt"))
                progress.set_postfix(
                    success=f"{eval_metrics['eval_success_rate']:.2f}",
                    collision=f"{eval_metrics['eval_collision_rate']:.2f}",
                )

            if global_step % int(cfg["checkpoint_interval_steps"]) < int(cfg["rollout_steps"]):
                agent.save(str(checkpoint_dir / f"{algorithm_name}_step_{global_step}.pt"))

            if global_step >= int(cfg["total_steps"]):
                break

    agent.save(str(checkpoint_dir / f"{algorithm_name}_final.pt"))
    if best_eval_metrics is not None:
        best_eval_path = table_dir / "best_eval.yaml"
        with best_eval_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(best_eval_metrics, file, sort_keys=False)
        print(f"Best eval: {best_eval_metrics}")
        print(f"Best eval file: {best_eval_path}")
        print(f"Best checkpoint: {checkpoint_dir / f'{algorithm_name}_best.pt'}")
    print(f"Training run directory: {run_dir}")
    print(f"Episode log: {episode_log_path}")
    print(f"Eval log: {eval_log_path}")


def _collect_rollout(
    agent: PPOMLPAgent,
    env: UAV2DEnv,
    obs: np.ndarray,
    train_config: dict,
    cfg: dict,
    device: torch.device,
    global_step_start: int,
) -> dict:
    rollout_steps = int(cfg["rollout_steps"])
    gamma = float(cfg["gamma"])
    gae_lambda = float(cfg["gae_lambda"])
    obs_rows = []
    action_rows = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    episodes = []
    episode_reward = float(getattr(env, "_ppo_episode_reward", 0.0))
    episode_steps = int(getattr(env, "_ppo_episode_steps", 0))
    episode_index = int(getattr(env, "_ppo_episode_index", 0))

    for local_step in range(rollout_steps):
        action, log_prob, value = agent.act(obs, deterministic=False)
        next_obs_dict, reward, done, info = env.step(action)
        next_obs = flatten_observation(next_obs_dict, env.world_size)
        obs_rows.append(obs)
        action_rows.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(float(reward))
        dones.append(float(done))
        episode_reward += float(reward)
        episode_steps += 1
        obs = next_obs
        if done:
            episodes.append(
                make_episode_row(
                    step=global_step_start + local_step + 1,
                    episode=episode_index,
                    episode_reward=episode_reward,
                    episode_steps=episode_steps,
                    outcome=info["outcome"],
                    extra={
                        "episode_randomization": serialize_episode_randomization(
                            info["episode_randomization"]
                        )
                    },
                )
            )
            episode_index += 1
            obs = flatten_observation(
                env.reset(
                    seed=training_environment_seed(train_config, episode_index)
                ),
                env.world_size,
            )
            episode_reward = 0.0
            episode_steps = 0

    setattr(env, "_ppo_episode_index", episode_index)
    setattr(env, "_ppo_episode_reward", episode_reward)
    setattr(env, "_ppo_episode_steps", episode_steps)
    last_value = agent.value(obs)
    advantages_np = _compute_gae(np.array(rewards), np.array(values), np.array(dones), last_value, gamma, gae_lambda)
    returns_np = advantages_np + np.array(values, dtype=np.float32)
    advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)
    return {
        "obs": torch.as_tensor(np.asarray(obs_rows), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.asarray(action_rows), dtype=torch.float32, device=device),
        "log_probs": torch.as_tensor(np.asarray(log_probs), dtype=torch.float32, device=device),
        "returns": torch.as_tensor(returns_np, dtype=torch.float32, device=device),
        "advantages": torch.as_tensor(advantages_np, dtype=torch.float32, device=device),
        "rewards_np": rewards,
        "last_obs_np": obs,
        "episodes": episodes,
    }


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_value = float(last_value)
    next_advantage = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages


if __name__ == "__main__":
    main()

