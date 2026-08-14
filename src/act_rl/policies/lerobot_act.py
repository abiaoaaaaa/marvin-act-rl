from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from ..types import ChunkBatch


class LeRobotACTAdapter(nn.Module):
    """Adds per-sample weighted loss to an unmodified LeRobot ACTPolicy."""

    def __init__(self, policy: nn.Module, keep_reference: bool = True):
        super().__init__()
        self.policy = policy
        self.reference = copy.deepcopy(policy).eval().requires_grad_(False) if keep_reference else None

    def parameter_groups(self, actor_lr: float, backbone_lr: float) -> list[dict]:
        backbone, other = [], []
        for name, parameter in self.policy.named_parameters():
            if not parameter.requires_grad:
                continue
            (backbone if name.startswith("model.backbone") else other).append(parameter)
        groups = [{"params": other, "lr": actor_lr}]
        if backbone and backbone_lr > 0:
            groups.append({"params": backbone, "lr": backbone_lr})
        else:
            for parameter in backbone:
                parameter.requires_grad_(False)
        return groups

    def _policy_batch(self, batch: ChunkBatch) -> dict[str, torch.Tensor]:
        result = {"observation.state": batch.state, "action": batch.action, "action_is_pad": batch.action_is_pad}
        result.update(batch.images)
        return result

    def _predict_train(self, policy: nn.Module, data: dict[str, torch.Tensor]):
        if policy.config.image_features:
            data = dict(data)
            data["observation.images"] = [data[key] for key in policy.config.image_features]
        return policy.model(data)

    def weighted_loss(self, batch: ChunkBatch, weights: torch.Tensor, prior_weight: float = 0.0):
        # ACT only runs its CVAE encoder when the module is in training mode.
        # from_pretrained() may leave it in eval mode after a deployment run.
        self.policy.train()
        data = self._policy_batch(batch)
        actions_hat, (mu, log_var) = self._predict_train(self.policy, data)
        valid = (~batch.action_is_pad).unsqueeze(-1)
        per_sample_den = valid.sum((1, 2)).clamp_min(1) * actions_hat.shape[-1]
        per_sample_l1 = (torch.abs(actions_hat - batch.action) * valid).sum((1, 2)) / per_sample_den
        loss = (weights.detach() * per_sample_l1).sum() / weights.sum().clamp_min(1e-6)
        kld = torch.zeros((), device=loss.device)
        if self.policy.config.use_vae and log_var is not None:
            kld = (-0.5 * (1 + log_var - mu.square() - log_var.exp())).sum(-1).mean()
            loss = loss + self.policy.config.kl_weight * kld
        prior = torch.zeros((), device=loss.device)
        if self.reference is not None and prior_weight > 0:
            with torch.no_grad():
                reference_actions = self._predict_train(self.reference, data)[0]
            prior = ((actions_hat - reference_actions).abs() * valid).sum() / (
                valid.sum().clamp_min(1) * actions_hat.shape[-1]
            )
            loss = loss + prior_weight * prior
        return loss, {"weighted_l1": per_sample_l1.mean().item(), "kld": kld.item(), "prior": prior.item()}

    @torch.no_grad()
    def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.policy.predict_action_chunk(observation)
