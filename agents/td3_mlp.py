"""Twin Delayed DDPG with MLP actor and critic networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.metrics import scalar_values
from utils.checkpoint_compat import (
    observation_spec_metadata,
    validate_checkpoint_observation_spec,
)

from agents.sac_mlp import mlp


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], action_scale: np.ndarray):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes, act_dim], nn.ReLU, nn.Identity)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.action_scale


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden_sizes, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


@dataclass
class TD3Losses:
    actor_loss: float
    critic_loss: float
    q_mean: float


class TD3MLPAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_scale: np.ndarray,
        hidden_sizes: Sequence[int],
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_delay: int,
        device: torch.device,
    ):
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = max(int(policy_delay), 1)
        self.device = device
        self.update_step = 0
        self.observation_spec = observation_spec_metadata(
            architecture="td3_mlp",
            obs_dim=obs_dim,
            action_scale=action_scale,
        )

        self.action_scale_np = np.asarray(action_scale, dtype=np.float32)
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=device)

        self.actor = DeterministicActor(obs_dim, act_dim, hidden_sizes, action_scale).to(device)
        self.actor_target = DeterministicActor(obs_dim, act_dim, hidden_sizes, action_scale).to(device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=float(critic_lr))

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True, noise_std: float = 0.0) -> np.ndarray:
        del deterministic
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_tensor).squeeze(0).cpu().numpy()
        if noise_std > 0.0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape) * self.action_scale_np
        return np.clip(action, -self.action_scale_np, self.action_scale_np)

    def update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        collect_metrics: bool = True,
    ) -> TD3Losses | None:
        self.update_step += 1
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise * self.action_scale
            noise = torch.clamp(noise, -self.noise_clip * self.action_scale, self.noise_clip * self.action_scale)
            next_actions = torch.clamp(self.actor_target(next_obs) + noise, -self.action_scale, self.action_scale)
            q_next = torch.min(self.q1_target(next_obs, next_actions), self.q2_target(next_obs, next_actions))
            target_q = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        actor_loss = None
        if self.update_step % self.policy_delay == 0:
            actor_loss = -self.q1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)

        if not collect_metrics:
            return None

        metric_actor_loss = actor_loss if actor_loss is not None else critic_loss.new_zeros(())
        metric_values = scalar_values(metric_actor_loss, critic_loss, q1.mean())
        return TD3Losses(
            actor_loss=metric_values[0],
            critic_loss=metric_values[1],
            q_mean=metric_values[2],
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for src_param, tgt_param in zip(source.parameters(), target.parameters()):
                tgt_param.data.mul_(1.0 - self.tau).add_(self.tau * src_param.data)

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "update_step": self.update_step,
                "observation_spec": self.observation_spec,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        validate_checkpoint_observation_spec(
            checkpoint,
            self.observation_spec,
            checkpoint_label=path,
        )
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint.get("actor_target", checkpoint["actor"]))
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint.get("q1_target", checkpoint["q1"]))
        self.q2_target.load_state_dict(checkpoint.get("q2_target", checkpoint["q2"]))
        self.update_step = int(checkpoint.get("update_step", 0))

    def load_actor(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        validate_checkpoint_observation_spec(
            checkpoint,
            self.observation_spec,
            checkpoint_label=path,
        )
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint.get("actor_target", checkpoint["actor"]))
        self.update_step = int(checkpoint.get("update_step", 0))

