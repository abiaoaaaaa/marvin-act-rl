"""RL-100 checkpoint adapter for the 16-DoF, three-camera Zadai task."""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SafetyError


LOG = logging.getLogger(__name__)
ZADAI_CAMERA_KEYS = ("top", "wrist_L", "wrist_R")
ZADAI_POLICY_CAMERA_KEYS = tuple(f"rgb_{key}" for key in ZADAI_CAMERA_KEYS)
ZADAI_STATE_DIM = 16


def _image(value: Any, expected_shape: tuple[int, int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape or array.dtype != np.uint8:
        raise SafetyError(f"{name} must be uint8 HWC {expected_shape}, got {array.dtype} {array.shape}")
    return np.ascontiguousarray(array)


class ZadaiObservationHistory:
    """Turn HTTP robot observations into RL-100 BCHW input tensors."""

    def __init__(self, n_obs_steps: int, image_shapes: dict[str, tuple[int, int, int]]):
        if n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        self.n_obs_steps = n_obs_steps
        self.image_shapes = image_shapes
        self._items: deque[dict[str, np.ndarray]] = deque(maxlen=n_obs_steps)

    def _validate(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        state = np.asarray(observation.get("state"), dtype=np.float32)
        if state.shape != (ZADAI_STATE_DIM,) or not np.isfinite(state).all():
            raise SafetyError(f"state must be finite {ZADAI_STATE_DIM}-D, got {state.shape}")
        result = {"state": np.ascontiguousarray(state)}
        for key in ZADAI_CAMERA_KEYS:
            if key not in observation:
                raise SafetyError(f"Robot observation is missing camera {key!r}")
            result[key] = _image(observation[key], self.image_shapes[key], key)
        return result

    def reset(self, observation: dict[str, Any]) -> None:
        item = self._validate(observation)
        self._items.clear()
        self._items.extend(item.copy() for _ in range(self.n_obs_steps))

    def append(self, observation: dict[str, Any]) -> None:
        if not self._items:
            self.reset(observation)
        else:
            self._items.append(self._validate(observation))

    def as_numpy(self) -> dict[str, np.ndarray]:
        if len(self._items) != self.n_obs_steps:
            raise RuntimeError("Observation history must be reset before inference")
        items = list(self._items)
        result = {
            "agent_pos": np.ascontiguousarray(
                np.stack([item["state"] for item in items])[None], dtype=np.float32
            )
        }
        for robot_key, policy_key in zip(ZADAI_CAMERA_KEYS, ZADAI_POLICY_CAMERA_KEYS):
            thwc = np.stack([item[robot_key] for item in items])
            # The encoder normalizes float [0,255] to [0,1].  Passing uint8
            # fails for an all-black (max==0) frame before that conversion.
            result[policy_key] = np.ascontiguousarray(
                thwc.transpose(0, 3, 1, 2)[None], dtype=np.float32
            )
        return result


def sanitize_zadai_chunk(action: Any, expected_steps: int) -> np.ndarray:
    """Reject malformed physical-unit actions; policy output is not normalized."""
    chunk = np.asarray(action, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.shape != (expected_steps, ZADAI_STATE_DIM):
        raise SafetyError(
            f"RL-100 Zadai action must be ({expected_steps}, {ZADAI_STATE_DIM}), got {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise SafetyError("RL-100 Zadai action contains NaN or Inf")
    return np.ascontiguousarray(chunk)


class ZadaiRL100PolicyRunner:
    """Load the trained Zadai RL-100 diffusion policy without a workspace."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        checkpoint_source: str = "model",
        deterministic: bool = True,
        num_inference_steps: int | None = None,
    ):
        try:
            import dill
            import hydra
            import torch
        except ImportError as exc:
            raise RuntimeError("Zadai deployment requires torch, dill, hydra-core and rl_100") from exc
        if checkpoint_source not in {"auto", "ema", "model"}:
            raise ValueError("checkpoint_source must be auto, ema or model")
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RL-100 checkpoint not found: {path}")
        self.torch = torch
        self.device = torch.device(device)
        self.deterministic = deterministic
        started = time.monotonic()
        with path.open("rb") as stream:
            payload = torch.load(stream, pickle_module=dill, map_location="cpu")
        if not isinstance(payload, dict) or {"cfg", "state_dicts"}.difference(payload):
            raise ValueError("Not an RL-100 workspace checkpoint: missing cfg/state_dicts")
        self.cfg = payload["cfg"]
        self.image_shapes = self._validate_config(self.cfg)
        self.policy = hydra.utils.instantiate(self.cfg.policy)
        state_dicts = payload["state_dicts"]
        selected = "ema_model" if checkpoint_source == "auto" and "ema_model" in state_dicts else "model"
        if checkpoint_source == "ema":
            selected = "ema_model"
        if selected not in state_dicts:
            raise KeyError(f"Checkpoint has no {selected!r}; available={sorted(state_dicts)}")
        self.policy.load_state_dict(state_dicts[selected], strict=True)
        if num_inference_steps is not None:
            if num_inference_steps <= 0:
                raise ValueError("num_inference_steps must be positive")
            self.policy.num_inference_steps = int(num_inference_steps)
        self.policy.to(self.device).eval()
        self.n_obs_steps = int(self.cfg.n_obs_steps)
        self.n_action_steps = int(self.cfg.n_action_steps)
        self.history = ZadaiObservationHistory(self.n_obs_steps, self.image_shapes)
        self.last_inference_seconds = 0.0
        LOG.info("Loaded %s from %s in %.2fs (obs=%d, actions=%d)", selected, path,
                 time.monotonic() - started, self.n_obs_steps, self.n_action_steps)

    @staticmethod
    def _validate_config(cfg: Any) -> dict[str, tuple[int, int, int]]:
        if str(cfg.feature_type) != "2D":
            raise ValueError(f"Zadai deployment requires feature_type=2D, got {cfg.feature_type}")
        if int(cfg.shape_meta.action.shape[0]) != ZADAI_STATE_DIM:
            raise ValueError(f"Expected {ZADAI_STATE_DIM}-D action, got {cfg.shape_meta.action.shape}")
        obs = cfg.shape_meta.obs
        if int(obs.agent_pos.shape[0]) != ZADAI_STATE_DIM:
            raise ValueError(f"Expected {ZADAI_STATE_DIM}-D state, got {obs.agent_pos.shape}")
        shapes = {}
        for key in ZADAI_POLICY_CAMERA_KEYS:
            if key not in obs:
                raise ValueError(f"Checkpoint observation schema is missing {key}")
            chw = tuple(int(x) for x in obs[key].shape)
            if len(chw) != 3 or chw[0] != 3:
                raise ValueError(f"Checkpoint camera {key} must be CHW RGB, got {chw}")
            shapes[key[4:]] = (chw[1], chw[2], chw[0])
        return shapes

    def reset(self, observation: dict[str, Any], seed: int = 1000) -> None:
        self.history.reset(observation)
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)

    def observe(self, observation: dict[str, Any]) -> None:
        self.history.append(observation)

    def predict_chunk(self) -> np.ndarray:
        torch_obs = {
            key: self.torch.from_numpy(value).to(self.device, non_blocking=True)
            for key, value in self.history.as_numpy().items()
        }
        started = time.monotonic()
        with self.torch.inference_mode():
            result = self.policy.predict_action(torch_obs, deterministic=self.deterministic)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        self.last_inference_seconds = time.monotonic() - started
        if not isinstance(result, dict) or "action" not in result:
            raise SafetyError("RL-100 predict_action did not return an action chunk")
        return sanitize_zadai_chunk(result["action"].detach().cpu().numpy(), self.n_action_steps)
