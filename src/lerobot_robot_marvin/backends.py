"""Mock and opt-in HIL-SERL backends.

No Marvin SDK, camera or motion implementation is duplicated here.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .constants import CAMERA_KEYS, FPS, IMAGE_SHAPE
from .errors import BridgeError
from .schema import validate_action, validate_observation

LOG = logging.getLogger(__name__)


class HumanOutcomeListener:
    """Global keyboard listener for supervised hardware evaluation."""

    def __init__(self):
        from pynput import keyboard

        self._keyboard = keyboard
        self._lock = threading.Lock()
        self._success = False
        self._failure = False
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()
        LOG.warning("Human outcome listener active: SPACE=success, ESC=failure")

    def _on_press(self, key) -> None:
        with self._lock:
            if key == self._keyboard.Key.esc:
                self._failure = True
                self._success = False
                LOG.warning("Human outcome: ESC pressed; failure will terminate on the next transition")
            elif key == self._keyboard.Key.space:
                self._success = True
                LOG.warning("Human outcome: SPACE pressed; success will terminate on the next transition")

    def reset(self) -> None:
        with self._lock:
            self._success = False
            self._failure = False

    def consume(self) -> str | None:
        with self._lock:
            outcome = "failure" if self._failure else "success" if self._success else None
            self._success = False
            self._failure = False
            return outcome

    def close(self) -> None:
        self._listener.stop()
        self._listener.join(timeout=1.0)


class Backend(ABC):
    @abstractmethod
    def connect(self) -> dict: ...

    @abstractmethod
    def reset(self) -> tuple[dict, dict]: ...

    @abstractmethod
    def observe(self) -> dict: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]: ...

    @abstractmethod
    def stop(self, reason: str) -> None: ...

    @abstractmethod
    def recover(self) -> dict: ...

    @abstractmethod
    def status(self) -> dict: ...

    @abstractmethod
    def disconnect(self) -> None: ...


class MockBackend(Backend):
    """Deterministic fake with real schema and instrumentation for tests."""

    def __init__(self, max_steps: int = 200, success_at_step: int | None = None, step_delay_s: float = 0.0):
        self.max_steps = max_steps
        self.success_at_step = success_at_step
        self.step_delay_s = step_delay_s
        self.connected = False
        self.episode_active = False
        self.step_count = 0
        self.reset_calls = 0
        self.observe_calls = 0
        self.step_calls = 0
        self.stop_calls = 0
        self.disconnect_calls = 0
        self.last_action = np.zeros(5, dtype=np.float32)
        self.last_stop_reason = ""
        self._lock = threading.Lock()

    def _observation(self) -> dict:
        state = np.zeros((1, 24), dtype=np.float32)
        state[0, 0] = 1.0
        state[0, 1:6] = self.last_action
        state[0, 9] = self.step_count / 100.0
        obs: dict[str, np.ndarray] = {"state": state}
        for index, key in enumerate(CAMERA_KEYS):
            value = (17 * index + self.step_count) % 256
            obs[key] = np.full((1, *IMAGE_SHAPE), value, dtype=np.uint8)
        return obs

    def connect(self) -> dict:
        with self._lock:
            self.connected = True
        return self.status()

    def reset(self) -> tuple[dict, dict]:
        with self._lock:
            if not self.connected:
                raise BridgeError("Mock backend is not connected")
            self.reset_calls += 1
            self.step_count = 0
            self.last_action.fill(0)
            self.episode_active = True
            return self._observation(), {"succeed": False}

    def observe(self) -> dict:
        with self._lock:
            self.observe_calls += 1
            if not self.episode_active:
                raise BridgeError("Reset the episode before observing")
            return self._observation()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        if self.step_delay_s:
            time.sleep(self.step_delay_s)
        with self._lock:
            if not self.connected or not self.episode_active:
                raise BridgeError("Mock episode is not active; connect and reset first")
            action = validate_action(action)
            self.step_calls += 1
            self.step_count += 1
            self.last_action = action.copy()
            succeed = self.success_at_step is not None and self.step_count >= self.success_at_step
            terminated = bool(succeed)
            truncated = self.step_count >= self.max_steps and not terminated
            if terminated or truncated:
                self.episode_active = False
            return self._observation(), float(succeed), terminated, truncated, {"succeed": succeed}

    def stop(self, reason: str) -> None:
        with self._lock:
            self.stop_calls += 1
            self.last_stop_reason = reason
            self.episode_active = False

    def recover(self) -> dict:
        if not self.connected:
            raise BridgeError("Mock backend is not connected")
        return {"recovered": True}

    def status(self) -> dict:
        return {
            "backend": "mock",
            "fps": FPS,
            "connected": self.connected,
            "episode_active": self.episode_active,
            "step_count": self.step_count,
            "stop_calls": self.stop_calls,
            "last_stop_reason": self.last_stop_reason,
        }

    def disconnect(self) -> None:
        with self._lock:
            self.disconnect_calls += 1
            self.episode_active = False
            self.connected = False


class HilserlBackend(Backend):
    """Thin adapter around TrainConfig.get_environment; hardware is opt-in."""

    def __init__(self, classifier: bool = False, human_success_key: bool = False):
        self.classifier = classifier
        self.human_success_key = human_success_key
        self.env: Any | None = None
        self._outcome_listener: HumanOutcomeListener | None = None
        self._last_observation: dict | None = None
        self._stopped = False

    def connect(self) -> dict:
        if self.env is not None:
            raise BridgeError("HIL-SERL environment is already constructed")
        module = importlib.import_module("examples.experiments.marvin_usb_insertion.config")
        config = module.TrainConfig()
        self.env = config.get_environment(fake_env=False, save_video=False, classifier=self.classifier)
        if self.human_success_key:
            self._outcome_listener = HumanOutcomeListener()
        self._stopped = False
        return self.status()

    def _require_env(self):
        if self.env is None or self._stopped:
            raise BridgeError("Hardware backend is unavailable or emergency-stopped")
        return self.env

    def reset(self) -> tuple[dict, dict]:
        if self._outcome_listener is not None:
            self._outcome_listener.reset()
        observation, info = self._require_env().reset()
        self._last_observation = validate_observation(observation)
        return self._last_observation, dict(info)

    def observe(self) -> dict:
        if self._last_observation is None:
            raise BridgeError("No wrapped observation exists; call reset first")
        return self._last_observation

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        if self._outcome_listener is not None:
            outcome = self._outcome_listener.consume()
            if outcome is not None:
                if self._last_observation is None:
                    raise BridgeError("Human outcome received before the first reset observation")
                reward = -0.5 if outcome == "failure" else 1.0
                info = {
                    "succeed": outcome == "success",
                    "human_outcome": outcome,
                }
                if outcome == "failure":
                    info["failure_reason"] = "keyboard_esc"
                return self._last_observation, reward, True, False, info

        observation, reward, terminated, truncated, info = self._require_env().step(validate_action(action))
        if self._outcome_listener is not None:
            outcome = self._outcome_listener.consume()
            if outcome == "failure":
                reward, terminated, truncated = -0.5, True, False
                info.update({"succeed": False, "human_outcome": "failure", "failure_reason": "keyboard_esc"})
            elif outcome == "success":
                reward, terminated, truncated = 1.0, True, False
                info.update({"succeed": True, "human_outcome": "success"})
        self._last_observation = validate_observation(observation)
        return self._last_observation, float(reward), bool(terminated), bool(truncated), dict(info)

    def _close_input_helpers(self) -> None:
        """Stop wrapper-owned input workers before their managers disappear."""
        wrapper = self.env
        visited: set[int] = set()
        while wrapper is not None and id(wrapper) not in visited:
            visited.add(id(wrapper))
            expert = getattr(wrapper, "expert", None)
            if expert is not None:
                try:
                    expert.close()
                    process = getattr(expert, "process", None)
                    if process is not None:
                        process.join(timeout=1.0)
                    manager = getattr(expert, "manager", None)
                    if manager is not None:
                        manager.shutdown()
                except Exception as exc:
                    LOG.warning("Failed to close intervention input helper cleanly: %s", exc)
            wrapper = getattr(wrapper, "env", None)

    def stop(self, reason: str) -> None:
        # HIL-SERL exposes no public non-terminal hold/stop API.  Its close()
        # disables both arms and releases SDK/cameras, so fail-safe stop is
        # deliberately terminal and requires a bridge restart.
        if self.env is not None and not self._stopped:
            self._stopped = True
            if self._outcome_listener is not None:
                self._outcome_listener.close()
                self._outcome_listener = None
            self._close_input_helpers()
            self.env.close()

    def recover(self) -> dict:
        env = self._require_env()
        env.unwrapped._recover()
        return {"recovered": True}

    def status(self) -> dict:
        return {
            "backend": "hardware",
            "fps": FPS,
            "connected": self.env is not None and not self._stopped,
            "stopped": self._stopped,
            "classifier": self.classifier,
            "human_success_key": self.human_success_key,
            "observation_ready": self._last_observation is not None,
        }

    def disconnect(self) -> None:
        if self.env is not None and not self._stopped:
            if self._outcome_listener is not None:
                self._outcome_listener.close()
                self._outcome_listener = None
            self._close_input_helpers()
            self.env.close()
        self._stopped = True
        self.env = None
        self._last_observation = None
