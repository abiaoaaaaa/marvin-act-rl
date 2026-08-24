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

    def __init__(self, input_mode: str = "keyboard"):
        from pynput import keyboard

        if input_mode not in {"keyboard", "footswitch"}:
            raise ValueError(f"Unsupported human outcome input mode: {input_mode}")
        self._keyboard = keyboard
        self._input_mode = input_mode
        self._lock = threading.Lock()
        self._success = False
        self._failure = False
        self._pressed_shifts: set[Any] = set()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()
        shortcuts = "SPACE=success; ESC=failure" if input_mode == "keyboard" else (
            "SHIFT+LEFT=success; SHIFT+RIGHT=failure/reset"
        )
        LOG.warning("Human outcome listener active: mode=%s; %s", input_mode, shortcuts)

    def _is_shift(self, key) -> bool:
        return key in {
            self._keyboard.Key.shift,
            self._keyboard.Key.shift_l,
            self._keyboard.Key.shift_r,
        }

    def _on_press(self, key) -> None:
        with self._lock:
            if self._is_shift(key):
                self._pressed_shifts.add(key)
            elif (
                self._input_mode == "keyboard" and key == self._keyboard.Key.esc
            ) or (
                self._input_mode == "footswitch"
                and key == self._keyboard.Key.right
                and self._pressed_shifts
            ):
                self._failure = True
                self._success = False
                LOG.warning("Human outcome: failure key pressed; failure/reset will occur on the next transition")
            elif (
                self._input_mode == "keyboard" and key == self._keyboard.Key.space
            ) or (
                self._input_mode == "footswitch"
                and key == self._keyboard.Key.left
                and self._pressed_shifts
            ):
                self._success = True
                LOG.warning("Human outcome: success key pressed; success will terminate on the next transition")

    def _on_release(self, key) -> None:
        with self._lock:
            if self._is_shift(key):
                self._pressed_shifts.discard(key)

    def reset(self) -> None:
        with self._lock:
            self._success = False
            self._failure = False
            self._pressed_shifts.clear()

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

    def __init__(
        self,
        classifier: bool = False,
        human_success_key: bool = False,
        human_outcome_input: str = "keyboard",
    ):
        self.classifier = classifier
        self.human_success_key = human_success_key
        self.human_outcome_input = human_outcome_input
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
            self._outcome_listener = HumanOutcomeListener(self.human_outcome_input)
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

    def _set_last_episode_result(self, success: bool) -> None:
        """Pass the human-labelled result to the unwrapped HIL-SERL env."""
        env = self._require_env()
        unwrapped = getattr(env, "unwrapped", env)
        callback = getattr(unwrapped, "set_last_episode_result", None)
        if callback is not None:
            callback(bool(success))

    def _human_failure_reason(self) -> str:
        return "footswitch_shift_right" if self.human_outcome_input == "footswitch" else "keyboard_esc"

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        if self._outcome_listener is not None:
            outcome = self._outcome_listener.consume()
            if outcome is not None:
                if self._last_observation is None:
                    raise BridgeError("Human outcome received before the first reset observation")
                reward = -0.5 if outcome == "failure" else 1.0
                self._set_last_episode_result(outcome == "success")
                info = {
                    "succeed": outcome == "success",
                    "human_outcome": outcome,
                }
                if outcome == "failure":
                    info["failure_reason"] = self._human_failure_reason()
                return self._last_observation, reward, True, False, info

        observation, reward, terminated, truncated, info = self._require_env().step(validate_action(action))
        if self._outcome_listener is not None:
            outcome = self._outcome_listener.consume()
            if outcome == "failure":
                self._set_last_episode_result(False)
                reward, terminated, truncated = -0.5, True, False
                info.update(
                    {
                        "succeed": False,
                        "human_outcome": "failure",
                        "failure_reason": self._human_failure_reason(),
                    }
                )
            elif outcome == "success":
                self._set_last_episode_result(True)
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
        env = self.env
        if env is not None and not self._stopped:
            self._stopped = True
            if self._outcome_listener is not None:
                self._outcome_listener.close()
                self._outcome_listener = None
            self._close_input_helpers()
            env.close()

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
            "human_outcome_input": self.human_outcome_input,
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
