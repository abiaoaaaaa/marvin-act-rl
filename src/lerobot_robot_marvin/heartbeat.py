"""Watchdog heartbeat shared by long-running Marvin policy clients."""

from __future__ import annotations

import logging
import threading
import time

from .client import MarvinBridgeClient
from .errors import BridgeError


LOG = logging.getLogger(__name__)


class BridgeHeartbeat:
    """Keep the Bridge alive while inference, updates, or replay writes run."""

    def __init__(self, client: MarvinBridgeClient, interval_s: float = 0.5, max_consecutive_failures: int = 3):
        if interval_s <= 0:
            raise ValueError("heartbeat interval must be positive")
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")
        self.client = client
        self.interval_s = interval_s
        self.max_consecutive_failures = max_consecutive_failures
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="marvin-heartbeat", daemon=True)
        self._state_lock = threading.Lock()
        self._last_primary_contact = time.monotonic()
        self._consecutive_failures = 0
        self.last_error: Exception | None = None

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            with self._state_lock:
                primary_contact_age = time.monotonic() - self._last_primary_contact
            if primary_contact_age < self.interval_s:
                continue
            try:
                self.client.status()
            except Exception as exc:
                with self._state_lock:
                    self._consecutive_failures += 1
                    failures = self._consecutive_failures
                    if failures >= self.max_consecutive_failures:
                        self.last_error = exc
                if failures >= self.max_consecutive_failures:
                    LOG.error("Bridge heartbeat failed %d consecutive times: %s", failures, exc)
                else:
                    LOG.warning(
                        "Transient bridge heartbeat failure %d/%d: %s",
                        failures,
                        self.max_consecutive_failures,
                        exc,
                    )
                continue
            with self._state_lock:
                self._consecutive_failures = 0
                self.last_error = None

    def note_primary_contact(self) -> None:
        with self._state_lock:
            self._last_primary_contact = time.monotonic()
            self._consecutive_failures = 0
            self.last_error = None

    def check(self) -> None:
        with self._state_lock:
            error = self.last_error
        if error is not None:
            raise BridgeError(f"Bridge heartbeat failed: {error}")

    def close(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2))
