from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence

import numpy as np

from .types import ChunkBatch, ChunkTransition


class ReplayBuffer:
    """Dependency-light replay; images remain optional to keep offline tests cheap."""

    def __init__(self, capacity: int | None = None, seed: int = 0):
        self._items = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: ChunkTransition) -> None:
        self._items.append(transition)

    def extend(self, transitions: Sequence[ChunkTransition]) -> None:
        self._items.extend(transitions)

    def sample_items(self, batch_size: int) -> list[ChunkTransition]:
        if not self._items:
            raise RuntimeError("cannot sample an empty replay buffer")
        return self._rng.choices(tuple(self._items), k=batch_size)

    def sample(self, batch_size: int, device: str = "cpu") -> ChunkBatch:
        return collate_transitions(self.sample_items(batch_size), device)


class MixedReplay:
    def __init__(self, offline: ReplayBuffer, online: ReplayBuffer, offline_fraction: float = 0.5):
        if not 0 <= offline_fraction <= 1:
            raise ValueError("offline_fraction must be in [0, 1]")
        self.offline, self.online, self.offline_fraction = offline, online, offline_fraction

    def sample(self, batch_size: int, device: str = "cpu") -> ChunkBatch:
        n_offline = round(batch_size * self.offline_fraction)
        if len(self.online) == 0:
            n_offline = batch_size
        if len(self.offline) == 0:
            n_offline = 0
        items = self.offline.sample_items(n_offline) if n_offline else []
        items += self.online.sample_items(batch_size - n_offline) if batch_size > n_offline else []
        random.shuffle(items)
        return collate_transitions(items, device)


def collate_transitions(items: Sequence[ChunkTransition], device: str = "cpu") -> ChunkBatch:
    import torch

    def tensor(values, dtype=torch.float32):
        return torch.as_tensor(np.stack(values), dtype=dtype, device=device)

    image_keys = set.intersection(*(set(x.images) for x in items)) if items else set()
    next_image_keys = set.intersection(*(set(x.next_images) for x in items)) if items else set()
    masks = [
        np.zeros(x.action_chunk.shape[0], dtype=bool) if x.action_is_pad is None else x.action_is_pad
        for x in items
    ]
    return ChunkBatch(
        state=tensor([x.state for x in items]), action=tensor([x.action_chunk for x in items]),
        next_state=tensor([x.next_state for x in items]),
        reward=tensor([np.asarray(x.reward, np.float32) for x in items]),
        discount=tensor([np.asarray(x.discount, np.float32) for x in items]),
        terminated=tensor([np.asarray(x.terminated, np.float32) for x in items]),
        action_is_pad=tensor(masks, torch.bool),
        images={k: tensor([x.images[k] for x in items]) for k in sorted(image_keys)},
        next_images={k: tensor([x.next_images[k] for x in items]) for k in sorted(next_image_keys)},
        source=[x.source for x in items],
    )
