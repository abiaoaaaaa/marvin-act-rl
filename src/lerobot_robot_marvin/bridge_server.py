"""Local HTTP bridge.  Default backend is always mock."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .backends import Backend, HilserlBackend, MockBackend
from .errors import MarvinError, ProtocolError
from .protocol import MAX_REQUEST_BYTES, PROTOCOL_VERSION, dumps, loads
from .schema import validate_action, validate_observation

LOG = logging.getLogger(__name__)
HARDWARE_CONFIRMATION = "I_UNDERSTAND_THIS_CONTROLS_A_REAL_ROBOT"


class BridgeService:
    def __init__(self, backend: Backend, watchdog_timeout_s: float = 2.0):
        self.backend = backend
        self.watchdog_timeout_s = watchdog_timeout_s
        self._operation_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._connected = False
        self._last_observation: dict | None = None
        self._last_transition: dict | None = None
        self._last_contact = time.monotonic()
        self._watchdog_stop = threading.Event()
        self._watchdog_fired = False
        self._active_operations = 0
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def close(self) -> None:
        self._watchdog_stop.set()
        self._watchdog_thread.join(timeout=1.0)
        try:
            # A hardware backend is initialized on the main thread before the
            # service/watchdog is constructed.  It must still be released if
            # the server exits before an ACT client adopts the session.
            if self._connected or self.backend.status().get("connected", False):
                self.backend.stop("bridge server shutdown")
                self.backend.disconnect()
        finally:
            self._connected = False

    def _touch(self) -> None:
        with self._state_lock:
            self._last_contact = time.monotonic()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(0.05):
            with self._state_lock:
                elapsed = time.monotonic() - self._last_contact
                should_stop = (
                    self._connected
                    and not self._watchdog_fired
                    and self._active_operations == 0
                    and elapsed > self.watchdog_timeout_s
                )
                if should_stop:
                    self._watchdog_fired = True
            if should_stop:
                LOG.error("Bridge watchdog expired after %.3fs; issuing terminal stop", elapsed)
                try:
                    self.backend.stop(f"watchdog timeout ({elapsed:.3f}s)")
                except Exception:
                    LOG.exception("Watchdog stop failed")

    def dispatch(self, operation: str, payload: dict) -> dict:
        self._touch()
        with self._state_lock:
            self._active_operations += 1
        try:
            if operation == "stop":
                # Hardware shutdown can take several seconds.  Count it as an
                # active serialized operation so the watchdog cannot enter the
                # same terminal close path a second time while cameras, the
                # gripper and the SDK are still being released.
                reason = str(payload.get("reason", "client requested stop"))
                with self._operation_lock:
                    self.backend.stop(reason)
                with self._state_lock:
                    self._watchdog_fired = True
                return {"stopped": True, "reason": reason}
            return self._dispatch_serial(operation, payload)
        finally:
            with self._state_lock:
                self._active_operations -= 1
                self._last_contact = time.monotonic()

    def _dispatch_serial(self, operation: str, payload: dict) -> dict:
        with self._operation_lock:
            if operation == "connect":
                if self._connected:
                    return self.backend.status()
                status = self.backend.status()
                # Hardware is preconnected by main() so JAX and SpaceMouse can
                # initialize before the watchdog creates a background thread.
                # Mock backends retain the normal lazy-connect behavior.
                result = status if status.get("connected", False) else self.backend.connect()
                self._connected = True
                with self._state_lock:
                    self._watchdog_fired = False
                return result

            if operation == "status":
                result = self.backend.status()
                result["bridge_connected"] = self._connected
                result["watchdog_fired"] = self._watchdog_fired
                return result

            if not self._connected:
                raise ProtocolError("Bridge session is not connected")

            if operation == "reset":
                observation, info = self.backend.reset()
                self._last_observation = validate_observation(observation)
                self._last_transition = None
                with self._state_lock:
                    self._watchdog_fired = False
                return {"observation": self._last_observation, "info": info}

            if operation == "observe":
                if self._last_observation is None:
                    self._last_observation = validate_observation(self.backend.observe())
                return {"observation": self._last_observation}

            if operation == "step":
                action = validate_action(payload.get("action"))
                observation, reward, terminated, truncated, info = self.backend.step(action)
                self._last_observation = validate_observation(observation)
                actual_action = validate_action(info.get("intervene_action", action))
                transition = {
                    "observation": self._last_observation,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "succeed": bool(info.get("succeed", reward >= 0.5)),
                    "actual_action": actual_action,
                    "info": info,
                }
                self._last_transition = transition
                return transition

            if operation == "recover":
                result = self.backend.recover()
                with self._state_lock:
                    self._watchdog_fired = False
                return result

            if operation == "disconnect":
                self.backend.stop("client disconnect")
                self.backend.disconnect()
                self._connected = False
                self._last_observation = None
                self._last_transition = None
                return {"disconnected": True}

            raise ProtocolError(f"Unsupported operation: {operation!r}")


class BridgeHTTPServer(HTTPServer):
    """Single-client local server whose handlers run on the main thread.

    HIL-SERL constructs keyboard/SpaceMouse helpers that register Unix signal
    handlers.  Python only permits that from the main interpreter thread, so a
    ThreadingHTTPServer cannot safely perform the hardware ``connect`` RPC.
    The bridge is loopback-only for hardware and serializes operations anyway,
    therefore a single-threaded HTTPServer is the correct execution model.
    """

    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: BridgeService):
        self.service = service
        super().__init__(address, BridgeRequestHandler)

    def server_close(self) -> None:
        self.service.close()
        super().server_close()


class BridgeRequestHandler(BaseHTTPRequestHandler):
    server: BridgeHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.debug("%s - %s", self.client_address[0], fmt % args)

    def _reply(self, status: int, body: dict) -> None:
        encoded = dumps(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            LOG.warning("Client disconnected before RPC response was delivered")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/rpc":
            self._reply(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ProtocolError(f"Invalid request size: {length}")
            message = loads(self.rfile.read(length))
            if not isinstance(message, dict) or message.get("version") != PROTOCOL_VERSION:
                raise ProtocolError("Unsupported or missing protocol version")
            operation = message.get("operation")
            payload = message.get("payload", {})
            if not isinstance(operation, str) or not isinstance(payload, dict):
                raise ProtocolError("operation must be str and payload must be dict")
            result = self.server.service.dispatch(operation, payload)
            self._reply(200, {"ok": True, "version": PROTOCOL_VERSION, "result": result})
        except (MarvinError, ValueError, TypeError) as exc:
            self._reply(400, {"ok": False, "version": PROTOCOL_VERSION, "error": str(exc)})
        except Exception as exc:
            LOG.exception("Unhandled bridge operation failure")
            self._reply(500, {"ok": False, "version": PROTOCOL_VERSION, "error": f"internal error: {exc}"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marvin local bridge (defaults to safe mock backend)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", choices=("mock", "hardware"), default="mock")
    parser.add_argument("--watchdog-timeout", type=float, default=2.0)
    parser.add_argument("--mock-max-steps", type=int, default=200)
    parser.add_argument("--mock-success-at-step", type=int)
    parser.add_argument("--classifier", action="store_true")
    parser.add_argument(
        "--human-success-key",
        action="store_true",
        help="Without learned reward, use SPACE=success and ESC=failure on the bridge host",
    )
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--confirm-hardware-risk", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser


def make_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "mock":
        return MockBackend(max_steps=args.mock_max_steps, success_at_step=args.mock_success_at_step)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("REFUSED: hardware bridge must bind to a loopback address")
    if not args.allow_hardware or args.confirm_hardware_risk != HARDWARE_CONFIRMATION:
        raise SystemExit(
            "REFUSED: hardware backend needs both --allow-hardware and "
            f"--confirm-hardware-risk={HARDWARE_CONFIRMATION}"
        )
    if args.classifier and args.human_success_key:
        raise SystemExit("Choose either --classifier or --human-success-key, not both")
    return HilserlBackend(
        classifier=args.classifier,
        human_success_key=args.human_success_key,
    )


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    backend = make_backend(args)

    if args.backend == "hardware":
        # Match the known-good HIL-SERL run_test.sh defaults.  These must be
        # applied before importing the experiment config, which imports JAX.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".1")
        LOG.warning(
            "Initializing hardware backend on the main thread before starting the watchdog "
            "(XLA preallocate=%s, memory_fraction=%s)",
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"],
        )
        backend.connect()

    service = BridgeService(backend, watchdog_timeout_s=args.watchdog_timeout)
    try:
        server = BridgeHTTPServer((args.host, args.port), service)
    except Exception:
        service.close()
        raise
    LOG.warning("Bridge listening on http://%s:%d with backend=%s", args.host, args.port, args.backend)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        LOG.info("Bridge interrupted")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
