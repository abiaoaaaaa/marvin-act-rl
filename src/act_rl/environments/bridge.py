from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StepResult:
    observation: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class BridgeEnv:
    """Gym-like adapter over the existing fail-safe MarvinBridgeClient."""

    def __init__(self, client):
        self.client = client

    def connect(self) -> None:
        self.client.connect()

    def reset(self):
        return self.client.reset()

    def step(self, action: np.ndarray) -> StepResult:
        result = self.client.step(action)
        return StepResult(
            observation=result["observation"], reward=float(result.get("reward", 0.0)),
            terminated=bool(result.get("terminated", result.get("done", False))),
            truncated=bool(result.get("truncated", False)), info=dict(result.get("info", {})),
        )

    def close(self) -> None:
        self.client.disconnect()
