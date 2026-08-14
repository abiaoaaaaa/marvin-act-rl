"""Validation and horizon handling at the bridge boundary."""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import ACTION_HIGH, ACTION_LOW, CAMERA_KEYS, IMAGE_SHAPE
from .errors import ProtocolError, SafetyError


def remove_leading_horizon(value: Any, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (1, *expected_shape):
        array = array[0]
    if array.shape != expected_shape:
        raise ProtocolError(f"{name} shape must be {expected_shape} or (1, {expected_shape}), got {array.shape}")
    return array


def validate_observation(observation: dict) -> dict[str, np.ndarray]:
    if not isinstance(observation, dict):
        raise ProtocolError(f"Observation must be a dict, got {type(observation).__name__}")
    expected = {"state", *CAMERA_KEYS}
    missing = expected.difference(observation)
    if missing:
        raise ProtocolError(f"Observation missing fields: {sorted(missing)}")

    state = remove_leading_horizon(observation["state"], (24,), "state").astype(np.float32, copy=False)
    if not np.all(np.isfinite(state)):
        raise ProtocolError("Observation state contains NaN or Inf")

    result = {"state": np.ascontiguousarray(state)}
    for key in CAMERA_KEYS:
        image = remove_leading_horizon(observation[key], IMAGE_SHAPE, key)
        if image.dtype != np.uint8:
            raise ProtocolError(f"Camera {key} must be uint8, got {image.dtype}")
        result[key] = np.ascontiguousarray(image)
    return result


def validate_action(action: Any) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (5,):
        raise SafetyError(f"Action must have shape (5,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise SafetyError("Action contains NaN or Inf")
    low = np.asarray(ACTION_LOW, dtype=np.float32)
    high = np.asarray(ACTION_HIGH, dtype=np.float32)
    if np.any(array < low) or np.any(array > high):
        raise SafetyError(f"Normalized action outside [-1, 1]: {array.tolist()}")
    return np.ascontiguousarray(array)

