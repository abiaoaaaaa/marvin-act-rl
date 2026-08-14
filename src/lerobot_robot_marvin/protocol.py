"""Versioned JSON protocol with explicit NumPy array encoding."""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

from .errors import ProtocolError

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": base64.b64encode(array.tobytes()).decode("ascii"),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and "__ndarray__" in value:
        try:
            raw = base64.b64decode(value["__ndarray__"], validate=True)
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(dim) for dim in value["shape"])
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"Invalid ndarray envelope: {exc}") from exc
        if expected != len(raw):
            raise ProtocolError(f"Array byte count mismatch: expected {expected}, got {len(raw)}")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def dumps(value: Any) -> bytes:
    return json.dumps(_encode(value), separators=(",", ":"), allow_nan=False).encode("utf-8")


def loads(payload: bytes) -> Any:
    try:
        return _decode(json.loads(payload.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Invalid JSON payload: {exc}") from exc


def request(operation: str, payload: dict | None = None) -> dict:
    return {"version": PROTOCOL_VERSION, "operation": operation, "payload": payload or {}}

