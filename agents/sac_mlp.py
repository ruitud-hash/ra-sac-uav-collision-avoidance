"""Soft Actor-Critic with MLP actor and critic networks."""

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


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(sizes: Sequence[int], activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(sizes) - 1):
        act = activation if idx < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[idx], sizes[idx + 1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], action_scale: np.ndarray):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes], nn.ReLU, nn.ReLU)
        self.mu = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], act_dim)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mu = self.mu(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        raw_action = mu if deterministic else dist.rsample()
        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.action_scale

        log_prob = dist.log_prob(raw_action) - torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden_sizes, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


@dataclass
class SACLosses:
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float
    q_mean: float


class SACMLPAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_scale: np.ndarray,
        hidden_sizes: Sequence[int],
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        device: torch.device,
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.target_entropy = -float(act_dim)
        self.observation_spec = observation_spec_metadata(
            architecture="sac_mlp",
            obs_dim=obs_dim,
            action_scale=action_scale,
        )

        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes, action_scale).to(device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(obs_tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        collect_metrics: bool = True,
    ) -> SACLosses | None:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_obs)
            q_next = torch.min(self.q1_target(next_obs, next_actions), self.q2_target(next_obs, next_actions))
            target_q = rewards + self.gamma * (1.0 - dones) * (q_next - self.alpha.detach() * next_log_probs)

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        new_actions, log_probs = self.actor(obs)
        q_new = torch.min(self.q1(obs, new_actions), self.q2(obs, new_actions))
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        if not collect_metrics:
            return None

        metric_values = scalar_values(
            actor_loss,
            critic_loss,
            alpha_loss,
            self.alpha,
            q_new.mean(),
        )
        return SACLosses(
            actor_loss=metric_values[0],
            critic_loss=metric_values[1],
            alpha_loss=metric_values[2],
            alpha=metric_values[3],
            q_mean=metric_values[4],
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for src_param, tgt_param in zip(source.parameters(), target.parameters()):
                tgt_param.data.mul_(1.0 - self.tau).add_(self.tau * src_param.data)

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
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
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))

    def load_actor(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        validate_checkpoint_observation_spec(
            checkpoint,
            self.observation_spec,
            checkpoint_label=path,
        )
        self.actor.load_state_dict(checkpoint["actor"])
