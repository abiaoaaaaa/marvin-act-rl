"""Fail-safe bridge client using only the Python standard library."""

from __future__ import annotations

import socket
import urllib.error
import urllib.request

import numpy as np

from .errors import BridgeError
from .protocol import MAX_RESPONSE_BYTES, dumps, loads, request
from .schema import validate_action, validate_observation


class MarvinBridgeClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        timeout_s: float = 1.0,
        connect_timeout_s: float = 30.0,
        reset_timeout_s: float = 120.0,
    ):
        self.url = f"http://{host}:{port}/v1/rpc"
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.reset_timeout_s = reset_timeout_s
        self.connected = False

    def _rpc(self, operation: str, payload: dict | None = None, timeout_s: float | None = None) -> dict:
        body = dumps(request(operation, payload))
        http_request = urllib.request.Request(
            self.url, data=body, method="POST", headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(http_request, timeout=timeout_s or self.timeout_s) as response:
                length = int(response.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_RESPONSE_BYTES:
                    raise BridgeError(f"Invalid bridge response size: {length}")
                message = loads(response.read(length))
        except urllib.error.HTTPError as exc:
            detail = loads(exc.read()) if exc.fp else {}
            raise BridgeError(detail.get("error", f"bridge HTTP {exc.code}")) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            raise BridgeError(f"Bridge {operation} failed: {exc}") from exc
        if not isinstance(message, dict) or not message.get("ok"):
            raise BridgeError(str(message.get("error", "invalid bridge response")))
        return message["result"]

    def connect(self) -> dict:
        try:
            result = self._rpc("connect", timeout_s=self.connect_timeout_s)
        except Exception:
            # The server may have constructed the hardware environment even
            # when its HTTP response was lost or timed out.
            self.best_effort_stop("connect RPC failed or timed out")
            raise
        self.connected = True
        return result

    def reset(self) -> tuple[dict, dict]:
        try:
            result = self._rpc("reset", timeout_s=self.reset_timeout_s)
        except Exception:
            self.best_effort_stop("reset RPC failed or timed out")
            raise
        return validate_observation(result["observation"]), dict(result["info"])

    def observe(self) -> dict:
        return validate_observation(self._rpc("observe")["observation"])

    def step(self, action: np.ndarray) -> dict:
        action = validate_action(action)
        try:
            result = self._rpc("step", {"action": action})
        except Exception:
            self.best_effort_stop("step RPC failed or timed out")
            raise
        result["observation"] = validate_observation(result["observation"])
        result["actual_action"] = validate_action(result["actual_action"])
        return result

    def stop(self, reason: str = "client requested stop") -> dict:
        return self._rpc("stop", {"reason": reason}, timeout_s=min(self.timeout_s, 0.5))

    def best_effort_stop(self, reason: str) -> None:
        try:
            self.stop(reason)
        except Exception:
            pass

    def recover(self) -> dict:
        return self._rpc("recover")

    def status(self) -> dict:
        return self._rpc("status")

    def disconnect(self) -> None:
        if not self.connected:
            return
        try:
            # Hardware shutdown disables the gripper/arms, stops several
            # camera pipelines and releases the Marvin SDK.  It is expected to
            # take longer than a regular control RPC, so use the same generous
            # timeout as hardware construction instead of the 0.5 s emergency
            # stop timeout.
            self._rpc("disconnect", timeout_s=self.connect_timeout_s)
        finally:
            self.connected = False
