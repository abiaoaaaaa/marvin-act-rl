from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class NetworkConfig:
    state_dim: int = 24
    action_dim: int = 5
    chunk_size: int = 30
    executed_steps: int = 10
    critic_horizon: int = 10
    hidden_dims: tuple[int, ...] = (512, 512, 256)
    use_images: bool = False
    image_keys: tuple[str, ...] = (
        "observation.images.side_policy",
        "observation.images.wrist_1buchong",
        "observation.images.wrist_1",
        "observation.images.wrist_2",
    )
    image_feature_dim: int = 128


@dataclass
class IQLConfig:
    gamma: float = 0.99
    expectile: float = 0.7
    temperature: float = 3.0
    max_advantage_weight: float = 100.0
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    actor_lr: float = 3e-5
    backbone_lr: float = 0.0
    target_tau: float = 0.005
    prior_weight: float = 0.1
    grad_clip_norm: float = 10.0
    offline_fraction: float = 0.5


@dataclass
class OnlineConfig:
    """Conservative defaults for staged real-robot replay fine-tuning."""

    low_online_threshold: int = 1000
    high_online_threshold: int = 5000
    offline_fraction_low_data: float = 0.75
    offline_fraction_mid_data: float = 0.50
    offline_fraction_high_data: float = 0.25
    recent_ratio: float = 0.50
    recent_window: int = 10_000
    balance_outcomes: bool = True
    intervention_weight: float = 2.0
    critic_to_actor_ratio: int = 2
    max_updates_per_online_transition: float = 4.0


@dataclass
class ACTRLConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    iql: IQLConfig = field(default_factory=IQLConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    batch_size: int = 64
    seed: int = 1000

    def validate(self) -> None:
        n, i, online = self.network, self.iql, self.online
        if n.state_dim <= 0 or n.action_dim <= 0 or n.chunk_size <= 0:
            raise ValueError("state_dim, action_dim and chunk_size must be positive")
        if not 1 <= n.executed_steps <= n.chunk_size:
            raise ValueError("executed_steps must be in [1, chunk_size]")
        if not 1 <= n.critic_horizon <= n.executed_steps:
            raise ValueError("critic_horizon must be in [1, executed_steps]")
        if not 0.5 <= i.expectile < 1.0:
            raise ValueError("IQL expectile must be in [0.5, 1)")
        if not 0.0 <= i.offline_fraction <= 1.0:
            raise ValueError("offline_fraction must be in [0, 1]")
        if not 0 < online.low_online_threshold <= online.high_online_threshold:
            raise ValueError("online thresholds must satisfy 0 < low <= high")
        for value in (online.offline_fraction_low_data, online.offline_fraction_mid_data,
                      online.offline_fraction_high_data, online.recent_ratio):
            if not 0.0 <= value <= 1.0:
                raise ValueError("online fractions and recent_ratio must be in [0, 1]")
        if online.critic_to_actor_ratio < 1:
            raise ValueError("critic_to_actor_ratio must be >= 1")
        if online.max_updates_per_online_transition <= 0:
            raise ValueError("max_updates_per_online_transition must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ACTRLConfig":
        raw = json.loads(Path(path).read_text())
        cfg = cls(
            network=NetworkConfig(**raw.get("network", {})),
            iql=IQLConfig(**raw.get("iql", {})),
            online=OnlineConfig(**raw.get("online", {})),
        )
        cfg.batch_size = int(raw.get("batch_size", cfg.batch_size))
        cfg.seed = int(raw.get("seed", cfg.seed))
        cfg.validate()
        return cfg
