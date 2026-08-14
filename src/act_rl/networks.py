from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(last, width), nn.LayerNorm(width), nn.Mish()))
        last = width
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class SharedImageEncoder(nn.Module):
    """Small shared four-camera encoder; replaceable without changing IQL."""

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, output_dim), nn.LayerNorm(output_dim),
        )

    def forward(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        if not images:
            raise ValueError("image encoder enabled but batch has no images")
        features = []
        for key in sorted(images):
            image = images[key]
            if image.ndim != 4:
                raise ValueError(f"{key} must be BCHW or BHWC, got {tuple(image.shape)}")
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            image = image.float()
            if image.max().detach() > 1.5:
                image = image / 255.0
            features.append(self.net(image))
        return torch.stack(features, dim=1).mean(dim=1)


class ObservationEncoder(nn.Module):
    def __init__(self, state_dim: int, use_images: bool = False, image_feature_dim: int = 128):
        super().__init__()
        self.use_images = use_images
        self.image_encoder = SharedImageEncoder(image_feature_dim) if use_images else None
        self.output_dim = state_dim + (image_feature_dim if use_images else 0)

    def forward(self, state: torch.Tensor, images: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if not self.use_images:
            return state
        return torch.cat((state, self.image_encoder(images or {})), dim=-1)


class TwinChunkQ(nn.Module):
    def __init__(self, obs_dim: int, chunk_size: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        input_dim = obs_dim + chunk_size * action_dim
        self.q1, self.q2 = mlp(input_dim, hidden_dims, 1), mlp(input_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((obs, action_chunk.flatten(1)), dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
