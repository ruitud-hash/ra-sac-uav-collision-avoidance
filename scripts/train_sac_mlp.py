"""Train a baseline SAC-MLP agent on the 2D UAV environment."""

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
from agents.replay_buffer import ReplayBuffer
from agents.sac_mlp import SACMLPAgent
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
    parser = argparse.ArgumentParser(description="Train SAC-MLP on the UAV 2D environment.")
    parser.add_argument(
        "--config",
        default="configs/train_sac_mlp_sanity.yaml",
        help="Training config path relative to the project root.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional SAC checkpoint used to initialize the actor.",
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
def evaluate(agent: SACMLPAgent, env_config: dict, episodes: int, seed: int, eval_id: int) -> dict:
    rows = []
    for idx in range(episodes):
        env = UAV2DEnv(deepcopy(env_config))
        obs_dict = env.reset(seed=seed + idx)
        obs = flatten_observation(obs_dict, env.world_size)
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            next_obs_dict, reward, done, info = env.step(action)
            obs = flatten_observation(next_obs_dict, env.world_size)
            total_reward += reward
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

    run_id = f"{timestamp()}_sac_mlp"
    run_dir = PROJECT_ROOT / train_config["output"]["directory"] / run_id
    log_dir = run_dir / "logs"
    table_dir = run_dir / "tables"
    checkpoint_dir = run_dir / "checkpoints"
    for directory in (log_dir, table_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = UAV2DEnv(deepcopy(env_config))
    obs_dict = env.reset(seed=training_environment_seed(train_config, 0))
    obs = flatten_observation(obs_dict, env.world_size)
    obs_dim = obs.shape[0]
    act_dim = 2
    action_scale = np.array([env.a_max, env.omega_max], dtype=np.float32)

    cfg = train_config["training"]
    eval_seed = int(cfg.get("eval_seed", seed + 100000))
    agent = SACMLPAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_scale=action_scale,
        hidden_sizes=cfg["hidden_sizes"],
        actor_lr=float(cfg["actor_lr"]),
        critic_lr=float(cfg["critic_lr"]),
        alpha_lr=float(cfg["alpha_lr"]),
        gamma=float(cfg["gamma"]),
        tau=float(cfg["tau"]),
        device=device,
    )
    init_checkpoint = args.init_checkpoint or train_config.get("init_checkpoint")
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.is_absolute():
            init_path = PROJECT_ROOT / init_path
        agent.load_actor(str(init_path))
        print(f"Initialized actor from checkpoint: {init_path}")
    write_training_audit_records(
        run_dir,
        env_config=env_config,
        train_config=train_config,
        observation_spec=agent.observation_spec,
    )
    replay = ReplayBuffer(obs_dim, act_dim, int(cfg["replay_size"]))

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
                "episode_randomization",
            ],
        )
        episode_writer.writeheader()
        eval_writer = csv.DictWriter(
            eval_file,
            fieldnames=EVAL_FIELDNAMES,
        )
        eval_writer.writeheader()

        episode = 0
        episode_reward = 0.0
        episode_steps = 0
        last_losses = None
        best_eval_score = -float("inf")
        best_eval_metrics = None
        eval_id = 0
        progress = trange(
            1,
            int(cfg["total_steps"]) + 1,
            desc=f"SAC-MLP train on {device}",
            mininterval=1.0,
        )
        for step in progress:
            if step <= int(cfg["start_steps"]):
                action = env.sample_random_action()
            else:
                action = agent.act(obs, deterministic=False)

            next_obs_dict, reward, done, info = env.step(action)
            next_obs = flatten_observation(next_obs_dict, env.world_size)
            replay.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            if step >= int(cfg["update_after"]) and replay.size >= int(cfg["batch_size"]):
                update_every = int(cfg["update_every"])
                for update_index in range(update_every):
                    batch = replay.sample(int(cfg["batch_size"]), device)
                    losses = agent.update(
                        batch,
                        collect_metrics=done and update_index == update_every - 1,
                    )
                    if losses is not None:
                        last_losses = losses

            if step % int(cfg["eval_interval_steps"]) == 0:
                eval_id += 1
                eval_metrics = evaluate(agent, env_config, int(cfg["eval_episodes"]), eval_seed, eval_id)
                eval_metrics["eval_seed_start"] = eval_seed
                eval_metrics["eval_seed_end"] = eval_seed + int(cfg["eval_episodes"]) - 1
                eval_writer.writerow({"step": step, **eval_metrics})
                eval_file.flush()
                eval_score = eval_selection_score(eval_metrics)
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_eval_metrics = {"step": step, "score": eval_score, **eval_metrics}
                    agent.save(str(checkpoint_dir / "sac_mlp_best.pt"))
                progress.set_postfix(
                    success=f"{eval_metrics['eval_success_rate']:.2f}",
                    collision=f"{eval_metrics['eval_collision_rate']:.2f}",
                )

            if step % int(cfg["checkpoint_interval_steps"]) == 0:
                agent.save(str(checkpoint_dir / f"sac_mlp_step_{step}.pt"))

            if done:
                row = make_episode_row(
                    step=step,
                    episode=episode,
                    episode_reward=episode_reward,
                    episode_steps=episode_steps,
                    outcome=info["outcome"],
                    extra={
                        "actor_loss": "" if last_losses is None else last_losses.actor_loss,
                        "critic_loss": "" if last_losses is None else last_losses.critic_loss,
                        "alpha_loss": "" if last_losses is None else last_losses.alpha_loss,
                        "alpha": "" if last_losses is None else last_losses.alpha,
                        "q_mean": "" if last_losses is None else last_losses.q_mean,
                        "episode_randomization": serialize_episode_randomization(
                            info["episode_randomization"]
                        ),
                    },
                )
                episode_writer.writerow(row)
                episode_file.flush()
                episode += 1
                obs_dict = env.reset(
                    seed=training_environment_seed(train_config, episode)
                )
                obs = flatten_observation(obs_dict, env.world_size)
                episode_reward = 0.0
                episode_steps = 0

    agent.save(str(checkpoint_dir / "sac_mlp_final.pt"))
    if best_eval_metrics is not None:
        best_eval_path = table_dir / "best_eval.yaml"
        with best_eval_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(best_eval_metrics, file, sort_keys=False)
        print(f"Best eval: {best_eval_metrics}")
        print(f"Best eval file: {best_eval_path}")
        print(f"Best checkpoint: {checkpoint_dir / 'sac_mlp_best.pt'}")
    print(f"Training run directory: {run_dir}")
    print(f"Episode log: {episode_log_path}")
    print(f"Eval log: {eval_log_path}")


if __name__ == "__main__":
    main()
