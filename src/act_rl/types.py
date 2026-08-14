from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ChunkTransition:
    state: np.ndarray
    action_chunk: np.ndarray
    next_state: np.ndarray
    reward: float
    discount: float
    terminated: bool
    truncated: bool = False
    action_is_pad: np.ndarray | None = None
    images: dict[str, np.ndarray] = field(default_factory=dict)
    next_images: dict[str, np.ndarray] = field(default_factory=dict)
    success: bool = False
    intervention: bool = False
    safety_abort: bool = False
    source: str = "offline"
    episode_id: int = -1
    timestep: int = -1
    executed_steps: int = 1
    policy_version: str = "dataset"

    def validate(self, state_dim: int, chunk_size: int, action_dim: int) -> None:
        if np.asarray(self.state).shape != (state_dim,):
            raise ValueError(f"state must be {(state_dim,)}, got {np.asarray(self.state).shape}")
        if np.asarray(self.next_state).shape != (state_dim,):
            raise ValueError(f"next_state must be {(state_dim,)}, got {np.asarray(self.next_state).shape}")
        if np.asarray(self.action_chunk).shape != (chunk_size, action_dim):
            raise ValueError(
                f"action_chunk must be {(chunk_size, action_dim)}, got {np.asarray(self.action_chunk).shape}"
            )
        if not np.isfinite(self.reward) or not 0.0 <= self.discount <= 1.0:
            raise ValueError("reward must be finite and discount must be in [0, 1]")


@dataclass
class ChunkBatch:
    state: Any
    action: Any
    next_state: Any
    reward: Any
    discount: Any
    terminated: Any
    action_is_pad: Any
    images: dict[str, Any] = field(default_factory=dict)
    next_images: dict[str, Any] = field(default_factory=dict)
    source: Any = None

    def to(self, device: Any) -> "ChunkBatch":
        move = lambda x: x.to(device) if hasattr(x, "to") else x
        return ChunkBatch(
            state=move(self.state), action=move(self.action), next_state=move(self.next_state),
            reward=move(self.reward), discount=move(self.discount), terminated=move(self.terminated),
            action_is_pad=move(self.action_is_pad), images={k: move(v) for k, v in self.images.items()},
            next_images={k: move(v) for k, v in self.next_images.items()}, source=self.source,
        )


def concatenate_batches(*batches: ChunkBatch) -> ChunkBatch:
    import torch

    if not batches:
        raise ValueError("at least one batch is required")
    common_images = set.intersection(*(set(batch.images) for batch in batches))
    common_next_images = set.intersection(*(set(batch.next_images) for batch in batches))
    cat = lambda name: torch.cat([getattr(batch, name) for batch in batches], dim=0)
    sources = []
    for batch in batches:
        sources.extend(batch.source if isinstance(batch.source, list) else [batch.source] * len(batch.state))
    return ChunkBatch(
        state=cat("state"), action=cat("action"), next_state=cat("next_state"), reward=cat("reward"),
        discount=cat("discount"), terminated=cat("terminated"), action_is_pad=cat("action_is_pad"),
        images={key: torch.cat([batch.images[key] for batch in batches]) for key in common_images},
        next_images={key: torch.cat([batch.next_images[key] for batch in batches]) for key in common_next_images},
        source=sources,
    )
