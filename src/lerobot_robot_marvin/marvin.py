"""LeRobot Robot client; all transforms and physical safety remain server-side."""

from __future__ import annotations

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot

from .client import MarvinBridgeClient
from .config_marvin import MarvinRobotConfig
from .constants import CAMERA_KEYS, IMAGE_SHAPE, ROBOT_ACTION_KEYS, ROBOT_STATE_KEYS
from .errors import SafetyError
from .schema import validate_action, validate_observation


class MarvinRobot(Robot):
    config_class = MarvinRobotConfig
    name = "marvin"

    def __init__(self, config: MarvinRobotConfig, client: MarvinBridgeClient | None = None):
        super().__init__(config)
        self.config = config
        self.client = client or MarvinBridgeClient(
            config.host,
            config.port,
            config.timeout_s,
            config.connect_timeout_s,
            config.reset_timeout_s,
        )
        self._connected = False
        self._cached_observation: dict | None = None
        self.last_transition: dict | None = None

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**dict.fromkeys(ROBOT_STATE_KEYS, float), **dict.fromkeys(CAMERA_KEYS, IMAGE_SHAPE)}

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ROBOT_ACTION_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise RuntimeError("MarvinRobot is already connected")
        self.client.connect()
        self._connected = True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def reset_episode(self) -> dict:
        self._require_connected()
        observation, info = self.client.reset()
        self._cached_observation = observation
        self.last_transition = None
        return info

    def recover(self) -> dict:
        self._require_connected()
        return self.client.recover()

    def status(self) -> dict:
        return self.client.status()

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("MarvinRobot is not connected")

    @staticmethod
    def _to_robot_observation(observation: dict) -> RobotObservation:
        observation = validate_observation(observation)
        state = observation["state"]
        result: RobotObservation = {key: float(state[index]) for index, key in enumerate(ROBOT_STATE_KEYS)}
        result.update({key: observation[key] for key in CAMERA_KEYS})
        return result

    def get_observation(self) -> RobotObservation:
        self._require_connected()
        if self._cached_observation is None:
            observation = self.client.observe()
        else:
            observation = self._cached_observation
            self._cached_observation = None
        return self._to_robot_observation(observation)

    def _action_to_vector(self, action: RobotAction) -> np.ndarray:
        missing = set(ROBOT_ACTION_KEYS).difference(action)
        extra = set(action).difference(ROBOT_ACTION_KEYS)
        if missing or extra:
            raise SafetyError(f"Action keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        return validate_action([action[key] for key in ROBOT_ACTION_KEYS])

    def send_action(self, action: RobotAction) -> RobotAction:
        self._require_connected()
        try:
            vector = self._action_to_vector(action)
            transition = self.client.step(vector)
        except Exception:
            if self.config.fail_safe_on_invalid_action:
                self.client.best_effort_stop("MarvinRobot rejected or failed to send action")
            raise
        self.last_transition = transition
        self._cached_observation = transition["observation"]
        actual = transition["actual_action"]
        return {key: float(actual[index]) for index, key in enumerate(ROBOT_ACTION_KEYS)}

    def emergency_stop(self, reason: str = "MarvinRobot emergency stop") -> None:
        self.client.best_effort_stop(reason)

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            self.client.disconnect()
        finally:
            self._connected = False
            self._cached_observation = None
