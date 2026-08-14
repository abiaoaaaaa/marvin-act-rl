"""RL-100 checkpoint loading and Marvin observation adaptation.

Heavy RL-100 dependencies are imported lazily so the HIL-SERL bridge can use
this package without installing torch, Hydra or RL-100.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CAMERA_KEYS, IMAGE_SHAPE
from .errors import SafetyError
from .schema import validate_observation


LOG = logging.getLogger(__name__)
RL100_CAMERA_KEYS = tuple("rgb_" + key for key in CAMERA_KEYS)


class MarvinObservationHistory:
    """Convert Marvin observations into RL-100's two-step BCHW contract."""

    def __init__(self, n_obs_steps: int = 2):
        if n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        self.n_obs_steps = n_obs_steps
        self._observations = deque(maxlen=n_obs_steps)

    def reset(self, observation: dict) -> None:
        observation = validate_observation(observation)
        self._observations.clear()
        for _ in range(self.n_obs_steps):
            self._observations.append(observation)

    def append(self, observation: dict) -> None:
        observation = validate_observation(observation)
        if not self._observations:
            self.reset(observation)
            return
        self._observations.append(observation)

    def as_numpy(self) -> dict:
        if len(self._observations) != self.n_obs_steps:
            raise RuntimeError("Observation history must be reset before inference")
        observations = list(self._observations)
        result = {
            "agent_pos": np.ascontiguousarray(
                np.stack([item["state"] for item in observations], axis=0)[None],
                dtype=np.float32,
            )
        }
        for bridge_key, policy_key in zip(CAMERA_KEYS, RL100_CAMERA_KEYS):
            thwc = np.stack([item[bridge_key] for item in observations], axis=0)
            result[policy_key] = np.ascontiguousarray(
                thwc.transpose(0, 3, 1, 2)[None],
                dtype=np.uint8,
            )
        return result


def sanitize_action_chunk(action: Any) -> tuple[np.ndarray, int]:
    """Validate an RL-100 chunk and clip finite extrapolation to [-1, 1]."""
    chunk = np.asarray(action, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim != 2 or chunk.shape[1] != 5 or chunk.shape[0] == 0:
        raise SafetyError(f"RL-100 action chunk must have shape (N, 5), got {chunk.shape}")
    if not np.all(np.isfinite(chunk)):
        raise SafetyError("RL-100 action chunk contains NaN or Inf")
    clipped = np.clip(chunk, -1.0, 1.0)
    clipped_rows = int(np.count_nonzero(np.any(clipped != chunk, axis=1)))
    return np.ascontiguousarray(clipped), clipped_rows


class RL100PolicyRunner:
    """Load only the policy (preferably EMA), not the full training workspace."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        checkpoint_source: str = "auto",
        deterministic: bool = True,
    ):
        try:
            import dill
            import hydra
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "RL100PolicyRunner requires torch, dill, hydra-core and the rl_100 package"
            ) from exc

        if checkpoint_source not in {"auto", "ema", "model"}:
            raise ValueError("checkpoint_source must be auto, ema or model")
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RL-100 checkpoint not found: {checkpoint_path}")

        self.torch = torch
        self.device = torch.device(device)
        self.deterministic = deterministic
        started = time.monotonic()
        with checkpoint_path.open("rb") as stream:
            payload = torch.load(stream, pickle_module=dill, map_location="cpu")
        if not isinstance(payload, dict) or "cfg" not in payload or "state_dicts" not in payload:
            raise ValueError("Not an RL-100 workspace checkpoint: missing cfg/state_dicts")

        self.cfg = payload["cfg"]
        self._validate_config(self.cfg)
        self.policy = hydra.utils.instantiate(self.cfg.policy)
        state_dicts = payload["state_dicts"]
        if checkpoint_source == "auto":
            selected = "ema_model" if "ema_model" in state_dicts else "model"
        else:
            selected = checkpoint_source + "_model" if checkpoint_source == "ema" else "model"
        if selected not in state_dicts:
            raise KeyError(f"Checkpoint has no {selected!r}; available={sorted(state_dicts)}")
        self.policy.load_state_dict(state_dicts[selected], strict=True)
        self.policy.to(self.device).eval()
        self.n_obs_steps = int(self.cfg.n_obs_steps)
        self.n_action_steps = int(self.cfg.n_action_steps)
        self.history = MarvinObservationHistory(self.n_obs_steps)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_source = selected
        self.last_inference_seconds = 0.0
        LOG.info(
            "Loaded RL-100 %s from %s in %.2fs (obs_steps=%d action_steps=%d)",
            selected,
            checkpoint_path,
            time.monotonic() - started,
            self.n_obs_steps,
            self.n_action_steps,
        )

    @staticmethod
    def _validate_config(cfg: Any) -> None:
        if str(cfg.feature_type) != "2D":
            raise ValueError(f"Marvin deployment requires feature_type=2D, got {cfg.feature_type}")
        if int(cfg.n_obs_steps) != 2:
            raise ValueError(f"Expected n_obs_steps=2, got {cfg.n_obs_steps}")
        if int(cfg.shape_meta.action.shape[0]) != 5:
            raise ValueError(f"Expected action dimension 5, got {cfg.shape_meta.action.shape}")
        obs_meta = cfg.shape_meta.obs
        if int(obs_meta.agent_pos.shape[0]) != 24:
            raise ValueError(f"Expected agent_pos dimension 24, got {obs_meta.agent_pos.shape}")
        for key in RL100_CAMERA_KEYS:
            if key not in obs_meta:
                raise ValueError(f"Checkpoint observation schema is missing {key}")
            if tuple(int(value) for value in obs_meta[key].shape) != (3, *IMAGE_SHAPE[:2]):
                raise ValueError(f"Checkpoint camera {key} has incompatible shape {obs_meta[key].shape}")

    def reset(self, observation: dict, seed: int = 1000) -> None:
        self.history.reset(observation)
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)

    def observe(self, observation: dict) -> None:
        self.history.append(observation)

    def predict_chunk(self) -> tuple[np.ndarray, int]:
        numpy_obs = self.history.as_numpy()
        torch_obs = {
            key: self.torch.from_numpy(value).to(self.device, non_blocking=True)
            for key, value in numpy_obs.items()
        }
        started = time.monotonic()
        with self.torch.inference_mode():
            result = self.policy.predict_action(
                torch_obs,
                deterministic=self.deterministic,
            )
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        self.last_inference_seconds = time.monotonic() - started
        if not isinstance(result, dict) or "action" not in result:
            raise SafetyError("RL-100 predict_action did not return an action chunk")
        chunk, clipped_rows = sanitize_action_chunk(result["action"].detach().cpu().numpy())
        if chunk.shape[0] != self.n_action_steps:
            raise SafetyError(
                f"Checkpoint declares {self.n_action_steps} action steps but returned {chunk.shape[0]}"
            )
        return chunk, clipped_rows
