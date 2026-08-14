from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ACTRLConfig
from ..networks import ObservationEncoder, TwinChunkQ, ValueNetwork
from ..types import ChunkBatch


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.square()).mean()


class IQLACTLearner:
    """IQL critic plus an advantage-weighted, directly updated ACT policy."""

    def __init__(self, actor_adapter: nn.Module, config: ACTRLConfig, device: str = "cpu"):
        config.validate()
        self.cfg, self.device = config, torch.device(device)
        n, h = config.network, config.network.hidden_dims
        self.encoder = ObservationEncoder(n.state_dim, n.use_images, n.image_feature_dim).to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.q = TwinChunkQ(self.encoder.output_dim, n.critic_horizon, n.action_dim, h).to(self.device)
        self.target_q = copy.deepcopy(self.q).requires_grad_(False)
        self.value = ValueNetwork(self.encoder.output_dim, h).to(self.device)
        self.actor = actor_adapter.to(self.device)
        critic_params = list(self.encoder.parameters()) + list(self.q.parameters())
        self.q_optimizer = torch.optim.AdamW(critic_params, lr=config.iql.critic_lr)
        self.v_optimizer = torch.optim.AdamW(self.value.parameters(), lr=config.iql.value_lr)
        groups = actor_adapter.parameter_groups(config.iql.actor_lr, config.iql.backbone_lr)
        self.actor_optimizer = torch.optim.AdamW(groups)

    def _obs(self, batch: ChunkBatch, next_obs: bool = False) -> torch.Tensor:
        state = batch.next_state if next_obs else batch.state
        images = batch.next_images if next_obs else batch.images
        encoder = self.target_encoder if next_obs else self.encoder
        return encoder(state, images)

    def update_critic(self, batch: ChunkBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        obs = self._obs(batch)
        with torch.no_grad():
            next_obs = self._obs(batch, True)
            target = batch.reward + batch.discount * (1.0 - batch.terminated) * self.value(next_obs)
        critic_action = batch.action[:, : self.cfg.network.critic_horizon]
        q1, q2 = self.q(obs, critic_action)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.q.parameters()), self.cfg.iql.grad_clip_norm)
        self.q_optimizer.step()

        with torch.no_grad():
            tq1, tq2 = self.target_q(obs.detach(), critic_action)
            target_q = torch.minimum(tq1, tq2)
        value = self.value(obs.detach())
        v_loss = expectile_loss(target_q - value, self.cfg.iql.expectile)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.iql.grad_clip_norm)
        self.v_optimizer.step()
        self._soft_update()
        return {"q_loss": q_loss.item(), "v_loss": v_loss.item(), "q_mean": target_q.mean().item(), "v_mean": value.mean().item()}

    def update_actor(self, batch: ChunkBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        with torch.no_grad():
            obs = self._obs(batch)
            q1, q2 = self.target_q(obs, batch.action[:, : self.cfg.network.critic_horizon])
            advantage = torch.minimum(q1, q2) - self.value(obs)
            weights = torch.exp(advantage * self.cfg.iql.temperature).clamp(max=self.cfg.iql.max_advantage_weight)
        loss, metrics = self.actor.weighted_loss(batch, weights, prior_weight=self.cfg.iql.prior_weight)
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.iql.grad_clip_norm)
        self.actor_optimizer.step()
        metrics.update({"actor_loss": loss.item(), "advantage_mean": advantage.mean().item(), "weight_mean": weights.mean().item()})
        return metrics

    def update(self, batch: ChunkBatch, update_actor: bool = True) -> dict[str, float]:
        metrics = self.update_critic(batch)
        if update_actor:
            metrics.update(self.update_actor(batch))
        return metrics

    @torch.no_grad()
    def _soft_update(self) -> None:
        tau = self.cfg.iql.target_tau
        for target, source in zip(self.target_q.parameters(), self.q.parameters(), strict=True):
            target.lerp_(source, tau)
        for target, source in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.lerp_(source, tau)

    def state_dict(self) -> dict[str, Any]:
        actor_state = self.actor.policy.state_dict() if hasattr(self.actor, "policy") else self.actor.state_dict()
        return {"actor": actor_state, "encoder": self.encoder.state_dict(),
                "target_encoder": self.target_encoder.state_dict(), "q": self.q.state_dict(),
                "target_q": self.target_q.state_dict(), "value": self.value.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(), "v_optimizer": self.v_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(), "config": self.cfg.to_dict()}

    def load_state_dict(self, state: dict[str, Any], load_optimizers: bool = True) -> None:
        actor = self.actor.policy if hasattr(self.actor, "policy") else self.actor
        actor.load_state_dict(state["actor"], strict=True)
        self.encoder.load_state_dict(state["encoder"], strict=True)
        self.target_encoder.load_state_dict(state.get("target_encoder", state["encoder"]), strict=True)
        self.q.load_state_dict(state["q"], strict=True)
        self.target_q.load_state_dict(state["target_q"], strict=True)
        self.value.load_state_dict(state["value"], strict=True)
        if load_optimizers:
            self.q_optimizer.load_state_dict(state["q_optimizer"])
            self.v_optimizer.load_state_dict(state["v_optimizer"])
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
