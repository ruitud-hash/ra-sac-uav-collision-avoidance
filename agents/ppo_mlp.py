"""Proximal Policy Optimization with MLP actor-critic networks."""

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


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], action_scale: np.ndarray):
        super().__init__()
        self.actor_body = mlp([obs_dim, *hidden_sizes], nn.Tanh, nn.Tanh)
        self.mu = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.value_net = mlp([obs_dim, *hidden_sizes, 1], nn.Tanh, nn.Identity)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        features = self.actor_body(obs)
        mu = self.mu(features)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs).squeeze(-1)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action) * self.action_scale
        log_prob = _squashed_log_prob(dist, raw_action, action, self.action_scale)
        value = self.value(obs)
        return action, log_prob, value

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(obs)
        return torch.tanh(dist.mean) * self.action_scale

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        normalized = torch.clamp(actions / self.action_scale, -0.999999, 0.999999)
        raw_action = torch.atanh(normalized)
        log_prob = _squashed_log_prob(dist, raw_action, actions, self.action_scale)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value


def _squashed_log_prob(
    dist: torch.distributions.Normal,
    raw_action: torch.Tensor,
    action: torch.Tensor,
    action_scale: torch.Tensor,
) -> torch.Tensor:
    tanh_action = torch.clamp(action / action_scale, -0.999999, 0.999999)
    correction = torch.log(action_scale * (1.0 - tanh_action.pow(2)) + 1e-6)
    return (dist.log_prob(raw_action) - correction).sum(dim=-1)


@dataclass
class PPOLosses:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float


class PPOMLPAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_scale: np.ndarray,
        hidden_sizes: Sequence[int],
        lr: float,
        clip_ratio: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        device: torch.device,
    ):
        self.device = device
        self.clip_ratio = float(clip_ratio)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.observation_spec = observation_spec_metadata(
            architecture="ppo_mlp",
            obs_dim=obs_dim,
            action_scale=action_scale,
        )
        self.model = PPOActorCritic(obs_dim, act_dim, hidden_sizes, action_scale).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            action = self.model.deterministic_action(obs_tensor)
            value = self.model.value(obs_tensor)
            log_prob = torch.zeros_like(value)
        else:
            action, log_prob, value = self.model.sample(obs_tensor)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.squeeze(0).cpu()),
            float(value.squeeze(0).cpu()),
        )

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.model.value(obs_tensor).squeeze(0).cpu())

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        epochs: int,
        minibatch_size: int,
    ) -> PPOLosses:
        batch_size = obs.shape[0]
        last_losses = PPOLosses(0.0, 0.0, 0.0, 0.0)
        indices = np.arange(batch_size)
        last_metric_tensors = None
        for _ in range(int(epochs)):
            np.random.shuffle(indices)
            for start in range(0, batch_size, int(minibatch_size)):
                batch_idx = torch.as_tensor(indices[start : start + int(minibatch_size)], dtype=torch.long, device=self.device)
                log_probs, entropy, values = self.model.evaluate_actions(obs[batch_idx], actions[batch_idx])
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                unclipped = ratio * advantages[batch_idx]
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[batch_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns[batch_idx])
                entropy_loss = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = (old_log_probs[batch_idx] - log_probs).mean().detach()
                last_metric_tensors = (policy_loss, value_loss, entropy_loss, approx_kl)
        if last_metric_tensors is not None:
            metric_values = scalar_values(*last_metric_tensors)
            last_losses = PPOLosses(
                policy_loss=metric_values[0],
                value_loss=metric_values[1],
                entropy=metric_values[2],
                approx_kl=metric_values[3],
            )
        return last_losses

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
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
        self.model.load_state_dict(checkpoint["model"])
