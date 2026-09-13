"""Soft Actor-Critic with a token-attention observation encoder."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.metrics import scalar_values
from agents.sac_mlp import LOG_STD_MAX, LOG_STD_MIN, mlp
from utils.checkpoint_compat import (
    observation_spec_metadata,
    validate_checkpoint_observation_spec,
)


RISK_DIAGNOSTIC_FIELDS = (
    "risk_gate_mean",
    "risk_gate_p10",
    "risk_gate_p50",
    "risk_gate_p90",
    "risk_gate_low_mean",
    "risk_gate_medium_mean",
    "risk_gate_high_mean",
    "residual_base_context_norm_ratio",
    "boundary_token_attention_weight",
    "boundary_attention_near",
    "boundary_attention_medium",
    "boundary_attention_far",
    "risk_residual_gradient_norm",
)


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        global_dim: int,
        max_tokens: int,
        token_dim: int,
        embed_dim: int = 128,
        risk_bias_mode: str = "none",
        risk_bias_scale: float = 1.0,
        risk_bias_clip: float | None = None,
        risk_gate_boundary_norm: float = 0.06,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.risk_bias_mode = risk_bias_mode
        self.risk_bias_scale = risk_bias_scale
        self.risk_bias_clip = None if risk_bias_clip is None else float(risk_bias_clip)
        self.risk_gate_boundary_norm = float(risk_gate_boundary_norm)
        if self.risk_bias_clip is not None and self.risk_bias_clip <= 0.0:
            raise ValueError("risk_bias_clip must be positive when provided")
        expected_obs_dim = global_dim + max_tokens * token_dim + max_tokens
        if obs_dim != expected_obs_dim:
            raise ValueError(
                f"obs_dim must equal global_dim + max_tokens * token_dim + max_tokens, "
                f"got {obs_dim} != {expected_obs_dim}"
            )

        if self.risk_bias_mode not in {"none", "mlp", "prior", "log1p_risk", "residual"}:
            raise ValueError(f"Unknown risk_bias_mode: {self.risk_bias_mode}")
        if self.risk_bias_mode == "residual" and token_dim <= 10:
            raise ValueError("residual risk attention requires an explicit risk channel at token index 10")
        if self.risk_gate_boundary_norm <= 0.0:
            raise ValueError("risk_gate_boundary_norm must be positive")

        self.global_net = mlp([global_dim, embed_dim, embed_dim], nn.ReLU, nn.ReLU)
        self.token_net = mlp([token_dim, embed_dim, embed_dim], nn.ReLU, nn.ReLU)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        risk_feature_dim = 5 if token_dim > 10 else 4
        self.risk_bias_net = mlp([risk_feature_dim, embed_dim // 2, 1], nn.ReLU, nn.Identity)
        if self.risk_bias_mode == "residual":
            self.risk_residual = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(self.risk_residual.weight)
            nn.init.zeros_(self.risk_residual.bias)
        self._last_risk_diagnostics: dict[str, torch.Tensor] | None = None
        self.output_dim = embed_dim * 2

    def forward(
        self,
        obs: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        disable_risk_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(
                f"attention observation must have shape (batch, {self.obs_dim}), "
                f"got {tuple(obs.shape)}"
            )
        global_features, tokens, mask = self.unpack(obs)
        global_embedding = self.global_net(global_features)
        token_embedding = self.token_net(tokens)

        query = self.query(global_embedding).unsqueeze(1)
        key = self.key(token_embedding)
        value = self.value(token_embedding)
        base_logits = self.attention_logits(query, key, tokens)
        logits = base_logits.masked_fill(mask <= 0.0, -1e9)

        no_valid_token = mask.sum(dim=-1, keepdim=True) <= 0.0
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        weights = torch.where(no_valid_token.unsqueeze(-1), torch.zeros_like(weights), weights)
        context = torch.sum(weights * value, dim=1)
        base_context = context
        diagnostics: dict[str, torch.Tensor] = {}
        if self.risk_bias_mode == "residual":
            boundary_token = self._boundary_token(global_features)
            risk_tokens = torch.cat([tokens, boundary_token], dim=1)
            risk_embedding = torch.cat([token_embedding, self.token_net(boundary_token)], dim=1)
            risk_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            risk_content_logits = torch.sum(query * self.key(risk_embedding), dim=-1) / sqrt(self.embed_dim)
            risk_bias = self.risk_bias_scale * self._risk_bias(risk_tokens)
            if self.risk_bias_clip is not None:
                risk_bias = torch.clamp(risk_bias, min=-self.risk_bias_clip, max=self.risk_bias_clip)
            risk_logits = risk_content_logits + risk_bias
            risk_weights = torch.softmax(
                risk_logits.masked_fill(risk_mask <= 0.0, -1e9), dim=-1
            ).unsqueeze(-1)
            risk_values = self.value(risk_embedding)
            weighted_risk_values = risk_weights * risk_values
            risk_context = torch.sum(weighted_risk_values, dim=1)
            risk_gate = torch.max(
                torch.where(risk_mask > 0.0, risk_tokens[:, :, 10], torch.zeros_like(risk_mask)),
                dim=1,
                keepdim=True,
            ).values.clamp(0.0, 1.0)
            residual_context = risk_gate * self.risk_residual(risk_context)
            self._last_risk_diagnostics = {
                "risk_gate": risk_gate.detach().flatten(),
                "base_context_norm": context.detach().norm(dim=1),
                "residual_context_norm": residual_context.detach().norm(dim=1),
                "boundary_attention": risk_weights[:, -1, 0].detach(),
                "boundary_distance": boundary_token[:, 0, 4].detach(),
            }
            if return_diagnostics:
                # Projection bias belongs to the summed residual once, not to every token.
                projected_token_contributions = risk_gate.unsqueeze(1) * F.linear(
                    weighted_risk_values, self.risk_residual.weight, bias=None
                )
                diagnostics = {
                    "base_logits": base_logits.detach(),
                    "base_attention_weights": weights.squeeze(-1).detach(),
                    "residual_logits": risk_logits.detach(),
                    "residual_attention_weights": risk_weights.squeeze(-1).detach(),
                    "per_token_value_vectors": risk_values.detach(),
                    "per_token_weighted_value_norms": weighted_risk_values.norm(dim=-1).detach(),
                    "per_token_projected_contribution_norms": projected_token_contributions.norm(dim=-1).detach(),
                    "risk_gate": risk_gate.detach(),
                    "base_context": base_context.detach(),
                    "residual_context": residual_context.detach(),
                    "base_context_norm": base_context.norm(dim=-1).detach(),
                    "residual_context_norm": residual_context.norm(dim=-1).detach(),
                    "valid_physical_token_count": mask.sum(dim=-1).detach(),
                    "physical_token_mask": mask.detach(),
                    "boundary_token_index": torch.full(
                        (obs.shape[0],), risk_tokens.shape[1] - 1, dtype=torch.long, device=obs.device
                    ),
                }
            if not disable_risk_residual:
                context = context + residual_context
        encoded = torch.cat([global_embedding, context], dim=-1)
        return (encoded, diagnostics) if return_diagnostics else encoded

    def attention_logits(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute content attention plus the configured risk logit prior."""
        logits = torch.sum(query * key, dim=-1) / sqrt(self.embed_dim)
        if self.risk_bias_mode not in {"none", "residual"}:
            risk_logits = self.risk_bias_scale * self._risk_bias(tokens)
            if self.risk_bias_clip is not None:
                risk_logits = torch.clamp(risk_logits, min=-self.risk_bias_clip, max=self.risk_bias_clip)
            logits = logits + risk_logits
        return logits

    def _risk_bias(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.risk_bias_mode in {"log1p_risk", "residual"}:
            if self.token_dim <= 10:
                raise ValueError("log1p_risk requires an explicit risk channel at token index 10")
            risk = torch.clamp(tokens[:, :, 10], min=0.0)
            return torch.log1p(risk)

        risk_features = tokens[:, :, 4:8]
        if self.token_dim > 10:
            risk_features = torch.cat([risk_features, tokens[:, :, 10:11]], dim=-1)
        if self.risk_bias_mode == "mlp":
            return self.risk_bias_net(risk_features).squeeze(-1)

        if self.token_dim > 10:
            return torch.clamp(6.0 * tokens[:, :, 10], min=0.0, max=6.0)

        clearance = torch.clamp(risk_features[:, :, 0], min=0.0)
        ttc = torch.clamp(risk_features[:, :, 1], min=0.0)
        dcpa = torch.clamp(risk_features[:, :, 3], min=0.0)
        distance_risk = 1.0 / (clearance + 0.02)
        ttc_risk = 1.0 / (ttc + 0.05)
        dcpa_risk = 1.0 / (dcpa + 0.02)
        prior = distance_risk + 2.0 * ttc_risk + dcpa_risk
        return torch.clamp(torch.log1p(prior), min=0.0, max=6.0)

    def risk_diagnostics(self) -> dict[str, float]:
        values = self._last_risk_diagnostics
        if values is None:
            return {}
        gate = values["risk_gate"]
        boundary_attention = values["boundary_attention"]
        boundary_distance = values["boundary_distance"]

        def grouped_mean(data: torch.Tensor, mask: torch.Tensor) -> float:
            selected = data[mask]
            return float(selected.mean()) if selected.numel() else float("nan")

        ratio = values["residual_context_norm"] / values["base_context_norm"].clamp_min(1e-6)
        return {
            "risk_gate_mean": float(gate.mean()),
            "risk_gate_p10": float(torch.quantile(gate, 0.1)),
            "risk_gate_p50": float(torch.quantile(gate, 0.5)),
            "risk_gate_p90": float(torch.quantile(gate, 0.9)),
            "risk_gate_low_mean": grouped_mean(gate, gate < 1.0 / 3.0),
            "risk_gate_medium_mean": grouped_mean(gate, (gate >= 1.0 / 3.0) & (gate < 2.0 / 3.0)),
            "risk_gate_high_mean": grouped_mean(gate, gate >= 2.0 / 3.0),
            "residual_base_context_norm_ratio": float(ratio.mean()),
            "boundary_token_attention_weight": float(boundary_attention.mean()),
            "boundary_attention_near": grouped_mean(
                boundary_attention, boundary_distance <= self.risk_gate_boundary_norm
            ),
            "boundary_attention_medium": grouped_mean(
                boundary_attention,
                (boundary_distance > self.risk_gate_boundary_norm)
                & (boundary_distance <= 2.0 * self.risk_gate_boundary_norm),
            ),
            "boundary_attention_far": grouped_mean(
                boundary_attention, boundary_distance > 2.0 * self.risk_gate_boundary_norm
            ),
        }

    def _boundary_token(self, global_features: torch.Tensor) -> torch.Tensor:
        distances = global_features[:, -4:]
        nearest_distance, nearest_side = distances.min(dim=1, keepdim=True)
        vectors = torch.stack(
            [
                torch.stack([-distances[:, 0], torch.zeros_like(distances[:, 0])], dim=1),
                torch.stack([distances[:, 1], torch.zeros_like(distances[:, 1])], dim=1),
                torch.stack([torch.zeros_like(distances[:, 2]), -distances[:, 2]], dim=1),
                torch.stack([torch.zeros_like(distances[:, 3]), distances[:, 3]], dim=1),
            ],
            dim=1,
        )
        token = global_features.new_zeros((global_features.shape[0], 1, self.token_dim))
        token[:, 0, 0:2] = vectors.gather(
            1, nearest_side.unsqueeze(-1).expand(-1, -1, 2)
        ).squeeze(1)
        token[:, 0, 4] = nearest_distance.squeeze(1)
        token[:, 0, 7] = nearest_distance.squeeze(1)
        token[:, 0, 10] = (
            1.0 - nearest_distance / self.risk_gate_boundary_norm
        ).clamp(0.0, 1.0).squeeze(1)
        return token

    def unpack(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_features = obs[:, : self.global_dim]
        token_start = self.global_dim
        token_end = token_start + self.max_tokens * self.token_dim
        tokens = obs[:, token_start:token_end].reshape(-1, self.max_tokens, self.token_dim)
        mask = obs[:, token_end : token_end + self.max_tokens]
        return global_features, tokens, mask


class AttentionActor(nn.Module):
    def __init__(
        self,
        encoder: AttentionEncoder,
        act_dim: int,
        hidden_sizes: Sequence[int],
        action_scale: np.ndarray,
    ):
        super().__init__()
        self.encoder = encoder
        self.net = mlp([encoder.output_dim, *hidden_sizes], nn.ReLU, nn.ReLU)
        self.mu = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], act_dim)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        *,
        return_diagnostics: bool = False,
        disable_risk_residual: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encoder(
            obs, return_diagnostics=return_diagnostics, disable_risk_residual=disable_risk_residual
        )
        if return_diagnostics:
            encoded, diagnostics = encoded
        features = self.net(encoded)
        mu = self.mu(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        raw_action = mu if deterministic else dist.rsample()
        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.action_scale
        log_prob = dist.log_prob(raw_action) - torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + 1e-6)
        result = (action, log_prob.sum(dim=-1, keepdim=True))
        if return_diagnostics:
            diagnostics["actor_mean"] = mu.detach()
            return (*result, diagnostics)
        return result



class DelayAwareActionResidual(nn.Module):
    """Independent, bounded pre-tanh action residual for RA-SAC-v4-A2."""

    def __init__(
        self,
        encoder: AttentionEncoder,
        act_dim: int,
        action_scale: np.ndarray,
        config: dict,
    ):
        super().__init__()
        if act_dim != 2:
            raise ValueError("delay-aware directional residual requires [acceleration, yaw_rate]")
        self.encoder = encoder
        self.execution_delay_steps = int(config.get("execution_delay_steps", 5))
        self.dt = float(config.get("dt_s", 1.0))
        self.world_size = float(config.get("world_size_m", 15000.0))
        self.v_max = float(config.get("v_max_mps", 15.0))
        self.boundary_margin = float(config.get("boundary_margin_norm", 0.06))
        hidden_dim = int(config.get("hidden_dim", 64))
        if self.execution_delay_steps < 0 or min(self.dt, self.world_size, self.v_max, self.boundary_margin) <= 0.0:
            raise ValueError("invalid action_residual geometry configuration")
        self.register_buffer("dynamic_scale", torch.as_tensor(config.get("dynamic_scale", [1.0, 1.0]), dtype=torch.float32))
        self.register_buffer("boundary_scale", torch.as_tensor(config.get("boundary_scale", [1.0, 1.0]), dtype=torch.float32))
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        embed_dim = encoder.embed_dim
        self.dynamic_query = nn.Linear(embed_dim, embed_dim)
        self.dynamic_key = nn.Linear(embed_dim, embed_dim)
        self.dynamic_value = nn.Linear(embed_dim, embed_dim)
        self.dynamic_output = nn.Linear(embed_dim, act_dim)
        self.boundary_mlp = mlp([14, hidden_dim, act_dim], nn.ReLU, nn.Identity)
        nn.init.zeros_(self.dynamic_output.weight)
        nn.init.zeros_(self.dynamic_output.bias)
        nn.init.zeros_(self.boundary_mlp[-2].weight)
        nn.init.zeros_(self.boundary_mlp[-2].bias)

    @staticmethod
    def _zero_at_origin_magnitude(raw: torch.Tensor) -> torch.Tensor:
        shifted = F.softplus(raw) - np.log(2.0)
        return torch.where(raw >= 0.0, shifted, -shifted)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        global_features, tokens, mask = self.encoder.unpack(obs)
        horizon_s = self.execution_delay_steps * self.dt
        delayed_tokens = tokens.clone()
        delayed_tokens[:, :, 0:2] = tokens[:, :, 0:2] + (
            tokens[:, :, 2:4] * (40.0 * horizon_s / self.world_size)
        )

        # Reuse the base representation without allowing the residual objective to alter it.
        dynamic_global = self.encoder.global_net(global_features).detach()
        dynamic_tokens = self.encoder.token_net(delayed_tokens).detach()
        query = self.dynamic_query(dynamic_global).unsqueeze(1)
        logits = torch.sum(query * self.dynamic_key(dynamic_tokens), dim=-1) / sqrt(self.encoder.embed_dim)
        if self.encoder.token_dim > 10:
            logits = logits + torch.log1p(torch.clamp(tokens[:, :, 10], min=0.0))
        masked_logits = logits.masked_fill(mask <= 0.0, -1e9)
        no_tokens = mask.sum(dim=-1, keepdim=True) <= 0.0
        weights = torch.softmax(masked_logits, dim=-1)
        weights = torch.where(no_tokens, torch.zeros_like(weights), weights)
        context = torch.sum(weights.unsqueeze(-1) * self.dynamic_value(dynamic_tokens), dim=1)
        if self.encoder.token_dim > 10:
            dynamic_gate = torch.max(
                torch.where(mask > 0.0, tokens[:, :, 10], torch.zeros_like(mask)), dim=1, keepdim=True
            ).values.clamp(0.0, 1.0)
        else:
            dynamic_gate = torch.zeros((obs.shape[0], 1), dtype=obs.dtype, device=obs.device)
        dynamic_delta = dynamic_gate * torch.tanh(self.dynamic_output(context)) * self.dynamic_scale

        distances = global_features[:, -4:]
        speed = torch.clamp(global_features[:, 2] * 20.0, 0.0, self.v_max)
        heading = global_features[:, 3]
        previous_accel = global_features[:, 4]
        previous_yaw = global_features[:, 5]
        speed_exec = torch.clamp(speed + previous_accel * horizon_s, 0.0, self.v_max)
        heading_exec = heading + previous_yaw * horizon_s
        mean_speed = 0.5 * (speed + speed_exec)
        mean_heading = heading + 0.5 * previous_yaw * horizon_s
        displacement_x = mean_speed * torch.cos(mean_heading) * horizon_s / self.world_size
        displacement_y = mean_speed * torch.sin(mean_heading) * horizon_s / self.world_size
        predicted_distances = torch.stack(
            (
                distances[:, 0] + displacement_x,
                distances[:, 1] - displacement_x,
                distances[:, 2] + displacement_y,
                distances[:, 3] - displacement_y,
            ),
            dim=1,
        )
        velocity_x = speed_exec * torch.cos(heading_exec)
        velocity_y = speed_exec * torch.sin(heading_exec)
        outward_velocity = torch.stack((-velocity_x, velocity_x, -velocity_y, velocity_y), dim=1)
        boundary_weights = torch.clamp((self.boundary_margin - predicted_distances) / self.boundary_margin, 0.0, 1.0)
        inward_x = boundary_weights[:, 0] - boundary_weights[:, 1]
        inward_y = boundary_weights[:, 2] - boundary_weights[:, 3]
        inward_norm = torch.sqrt(inward_x.square() + inward_y.square()).clamp_min(1e-6)
        inward_heading = torch.atan2(inward_y / inward_norm, inward_x / inward_norm)
        heading_error = torch.atan2(torch.sin(inward_heading - heading_exec), torch.cos(inward_heading - heading_exec))
        minimum_distance = predicted_distances.min(dim=1, keepdim=True).values
        boundary_gate = torch.clamp((self.boundary_margin - minimum_distance) / self.boundary_margin, 0.0, 1.0)
        outward_positive = torch.clamp(outward_velocity, min=0.0)
        weighted_outward = (
            (boundary_weights * outward_positive).sum(dim=1, keepdim=True)
            / boundary_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        )
        time_to_boundary = torch.where(
            outward_positive > 1e-6,
            predicted_distances * self.world_size / outward_positive.clamp_min(1e-6),
            torch.full_like(predicted_distances, 60.0),
        ).min(dim=1, keepdim=True).values.clamp(0.0, 60.0)
        boundary_features = torch.cat(
            (
                predicted_distances,
                outward_velocity / self.v_max,
                minimum_distance,
                time_to_boundary / 60.0,
                (speed_exec / self.v_max).unsqueeze(1),
                torch.sin(heading_error).unsqueeze(1),
                torch.cos(heading_error).unsqueeze(1),
                torch.full_like(minimum_distance, self.execution_delay_steps / 5.0),
            ),
            dim=1,
        )
        magnitude = torch.tanh(self._zero_at_origin_magnitude(self.boundary_mlp(boundary_features)))
        boundary_delta = torch.cat(
            (
                -boundary_gate * magnitude[:, 0:1] * (weighted_outward / self.v_max),
                boundary_gate * magnitude[:, 1:2] * torch.sign(heading_error).unsqueeze(1),
            ),
            dim=1,
        ) * self.boundary_scale
        return dynamic_delta, boundary_delta, {
            "dynamic_gate": dynamic_gate,
            "boundary_gate": boundary_gate,
            "dynamic_attention_weights": weights,
            "dynamic_delta_mean": dynamic_delta,
            "boundary_delta_mean": boundary_delta,
            "predicted_boundary_distances": predicted_distances,
            "heading_error_to_inward": heading_error,
        }


class DelayAwareAttentionActor(AttentionActor):
    def __init__(
        self,
        encoder: AttentionEncoder,
        act_dim: int,
        hidden_sizes: Sequence[int],
        action_scale: np.ndarray,
        action_residual_config: dict,
    ):
        super().__init__(encoder, act_dim, hidden_sizes, action_scale)
        self.action_residual = DelayAwareActionResidual(
            encoder, act_dim, action_scale, action_residual_config
        )

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        *,
        return_diagnostics: bool = False,
        disable_risk_residual: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        features = self.net(self.encoder(obs))
        base_mean = self.mu(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        dynamic_delta, boundary_delta, diagnostics = self.action_residual(obs)
        total_mean = base_mean if disable_risk_residual else base_mean + dynamic_delta + boundary_delta
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(total_mean, std)
        raw_action = total_mean if deterministic else dist.rsample()
        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.action_scale
        log_prob = dist.log_prob(raw_action) - torch.log(
            self.action_scale * (1.0 - tanh_action.pow(2)) + 1e-6
        )
        result = (action, log_prob.sum(dim=-1, keepdim=True))
        if return_diagnostics:
            diagnostics = {key: value.detach() for key, value in diagnostics.items()}
            diagnostics.update(
                actor_base_mean=base_mean.detach(),
                actor_mean=total_mean.detach(),
                actor_log_std=log_std.detach(),
            )
            return (*result, diagnostics)
        return result
class AttentionQNetwork(nn.Module):
    def __init__(self, encoder: AttentionEncoder, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.encoder = encoder
        self.q = mlp([encoder.output_dim + act_dim, *hidden_sizes, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([self.encoder(obs), action], dim=-1))


@dataclass
class AttentionSACLosses:
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float
    q_mean: float
    goal_guidance_loss: float = 0.0
    safety_gate_mean: float = 0.0
    risk_gate_mean: float = float("nan")
    risk_gate_p10: float = float("nan")
    risk_gate_p50: float = float("nan")
    risk_gate_p90: float = float("nan")
    risk_gate_low_mean: float = float("nan")
    risk_gate_medium_mean: float = float("nan")
    risk_gate_high_mean: float = float("nan")
    residual_base_context_norm_ratio: float = float("nan")
    boundary_token_attention_weight: float = float("nan")
    boundary_attention_near: float = float("nan")
    boundary_attention_medium: float = float("nan")
    boundary_attention_far: float = float("nan")
    risk_residual_gradient_norm: float = 0.0


class SACAttentionAgent:
    def __init__(
        self,
        obs_dim: int,
        global_dim: int,
        max_tokens: int,
        token_dim: int,
        act_dim: int,
        action_scale: np.ndarray,
        hidden_sizes: Sequence[int],
        embed_dim: int,
        risk_bias_mode: str,
        risk_bias_scale: float,
        goal_guidance: dict | None,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        device: torch.device,
        risk_bias_clip: float | None = None,
        risk_gate_boundary_norm: float = 0.06,
        action_residual_config: dict | None = None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.target_entropy = -float(act_dim)
        self.observation_spec = observation_spec_metadata(
            architecture="sac_attention",
            obs_dim=obs_dim,
            global_dim=global_dim,
            max_tokens=max_tokens,
            token_dim=token_dim,
            action_scale=action_scale,
        )
        self.goal_guidance = goal_guidance or {}
        self.goal_guidance_enabled = bool(self.goal_guidance.get("enabled", False))
        self.goal_guidance_type = self.goal_guidance.get("type", "heading")
        self.goal_guidance_lambda = float(self.goal_guidance.get("lambda", 0.0))
        if not self.goal_guidance_enabled and self.goal_guidance_lambda != 0.0:
            raise ValueError("Disabled goal guidance requires lambda=0.0")
        self.goal_guidance_loss_computed = (
            self.goal_guidance_enabled and self.goal_guidance_lambda > 0.0
        )
        self.goal_guidance_dt = float(self.goal_guidance.get("dt_s", 1.0))
        self.goal_guidance_speed_scale = float(self.goal_guidance.get("speed_scale_mps", 20.0))
        self.goal_guidance_v_min = float(self.goal_guidance.get("v_min_mps", 0.0))
        self.goal_guidance_v_max = float(self.goal_guidance.get("v_max_mps", 15.0))
        self.safe_clearance_threshold = float(self.goal_guidance.get("safe_clearance_norm", 0.02))
        self.safe_ttc_threshold = float(self.goal_guidance.get("safe_ttc_norm", 0.25))
        self.safe_risk_threshold = float(self.goal_guidance.get("safe_risk_norm", 0.5))
        self.safe_gate_temperature = float(self.goal_guidance.get("gate_temperature", 0.05))
        action_residual_enabled = risk_bias_mode == "action_residual"
        encoder_risk_mode = "none" if action_residual_enabled else risk_bias_mode
        actor_encoder = AttentionEncoder(
            obs_dim, global_dim, max_tokens, token_dim, embed_dim, encoder_risk_mode, risk_bias_scale,
            risk_bias_clip, risk_gate_boundary_norm
        )
        if action_residual_enabled:
            self.actor = DelayAwareAttentionActor(
                actor_encoder, act_dim, hidden_sizes, action_scale, action_residual_config or {}
            ).to(device)
        else:
            self.actor = AttentionActor(actor_encoder, act_dim, hidden_sizes, action_scale).to(device)
        self.q1 = AttentionQNetwork(
            AttentionEncoder(
                obs_dim, global_dim, max_tokens, token_dim, embed_dim, encoder_risk_mode, risk_bias_scale,
                risk_bias_clip, risk_gate_boundary_norm
            ),
            act_dim,
            hidden_sizes,
        ).to(device)
        self.q2 = AttentionQNetwork(
            AttentionEncoder(
                obs_dim, global_dim, max_tokens, token_dim, embed_dim, encoder_risk_mode, risk_bias_scale,
                risk_bias_clip, risk_gate_boundary_norm
            ),
            act_dim,
            hidden_sizes,
        ).to(device)
        self.q1_target = AttentionQNetwork(
            AttentionEncoder(
                obs_dim, global_dim, max_tokens, token_dim, embed_dim, encoder_risk_mode, risk_bias_scale,
                risk_bias_clip, risk_gate_boundary_norm
            ),
            act_dim,
            hidden_sizes,
        ).to(device)
        self.q2_target = AttentionQNetwork(
            AttentionEncoder(
                obs_dim, global_dim, max_tokens, token_dim, embed_dim, encoder_risk_mode, risk_bias_scale,
                risk_bias_clip, risk_gate_boundary_norm
            ),
            act_dim,
            hidden_sizes,
        ).to(device)
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
    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        *,
        return_diagnostics: bool = False,
        disable_risk_residual: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        result = self.actor(
            obs_tensor,
            deterministic=deterministic,
            return_diagnostics=return_diagnostics,
            disable_risk_residual=disable_risk_residual,
        )
        if return_diagnostics:
            action, _, diagnostics = result
            return action.squeeze(0).cpu().numpy(), {
                key: value.squeeze(0).cpu().numpy() for key, value in diagnostics.items()
            }
        action, _ = result
        return action.squeeze(0).cpu().numpy()

    def update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        collect_metrics: bool = True,
    ) -> AttentionSACLosses | None:
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
        goal_guidance_loss = torch.zeros((), device=self.device)
        safety_gate = torch.ones((obs.shape[0], 1), device=self.device)
        if self.goal_guidance_loss_computed:
            goal_guidance_loss, safety_gate = self._goal_guidance_loss(obs, new_actions)
            actor_loss = actor_loss + self.goal_guidance_lambda * goal_guidance_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        risk_residual_gradient_norm = torch.zeros((), device=self.device)
        if collect_metrics and hasattr(self.actor.encoder, "risk_residual"):
            gradients = [
                parameter.grad.detach().square().sum()
                for parameter in self.actor.encoder.risk_residual.parameters()
                if parameter.grad is not None
            ]
            if gradients:
                risk_residual_gradient_norm = torch.stack(gradients).sum().sqrt()
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
            goal_guidance_loss,
            safety_gate.mean(),
            risk_residual_gradient_norm,
        )
        risk_diagnostics = self.actor.encoder.risk_diagnostics()
        return AttentionSACLosses(
            actor_loss=metric_values[0],
            critic_loss=metric_values[1],
            alpha_loss=metric_values[2],
            alpha=metric_values[3],
            q_mean=metric_values[4],
            goal_guidance_loss=metric_values[5],
            safety_gate_mean=metric_values[6],
            risk_residual_gradient_norm=metric_values[7],
            **risk_diagnostics,
        )

    def _goal_guidance_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        goal_bearing = obs[:, 9:10]
        candidate_accel = actions[:, 0:1]
        candidate_omega = actions[:, 1:2]
        predicted_bearing = goal_bearing - candidate_omega * self.goal_guidance_dt
        safety_gate = self._safety_gate(obs).detach()

        if self.goal_guidance_type == "heading":
            guidance_loss = 1.0 - torch.cos(predicted_bearing)
        elif self.goal_guidance_type == "velocity_projection":
            current_speed = obs[:, 2:3] * self.goal_guidance_speed_scale
            candidate_speed = torch.clamp(
                current_speed + candidate_accel * self.goal_guidance_dt,
                min=self.goal_guidance_v_min,
                max=self.goal_guidance_v_max,
            )
            candidate_speed_norm = candidate_speed / max(self.goal_guidance_speed_scale, 1e-6)
            projected_speed = candidate_speed_norm * torch.cos(predicted_bearing)
            guidance_loss = 1.0 - projected_speed
        else:
            raise ValueError(f"Unknown goal_guidance type: {self.goal_guidance_type}")

        return (safety_gate * guidance_loss).mean(), safety_gate

    def _safety_gate(self, obs: torch.Tensor) -> torch.Tensor:
        token_start = self.actor.encoder.global_dim
        token_end = token_start + self.actor.encoder.max_tokens * self.actor.encoder.token_dim
        tokens = obs[:, token_start:token_end].reshape(-1, self.actor.encoder.max_tokens, self.actor.encoder.token_dim)
        mask = obs[:, token_end : token_end + self.actor.encoder.max_tokens]
        valid = mask > 0.0
        if tokens.shape[1] == 0:
            return torch.ones((obs.shape[0], 1), device=obs.device)

        clearance = tokens[:, :, 4]
        ttc = tokens[:, :, 5]
        clearance_danger = torch.sigmoid((self.safe_clearance_threshold - clearance) / self.safe_gate_temperature)
        ttc_danger = torch.sigmoid((self.safe_ttc_threshold - ttc) / self.safe_gate_temperature)
        danger = torch.maximum(clearance_danger, ttc_danger)
        if self.actor.encoder.token_dim > 10:
            risk_danger = torch.sigmoid(
                (tokens[:, :, 10] - self.safe_risk_threshold) / self.safe_gate_temperature
            )
            danger = torch.maximum(danger, risk_danger)
        danger = torch.where(valid, danger, torch.zeros_like(danger))
        max_danger = torch.max(danger, dim=1, keepdim=True).values
        return torch.clamp(1.0 - max_danger, 0.0, 1.0)

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

    def load_actor(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        validate_checkpoint_observation_spec(
            checkpoint,
            self.observation_spec,
            checkpoint_label=path,
        )
        missing, unexpected = self.actor.load_state_dict(checkpoint["actor"], strict=False)
        if missing:
            print(f"Actor checkpoint missing keys ignored: {missing}")
        if unexpected:
            print(f"Actor checkpoint unexpected keys ignored: {unexpected}")
