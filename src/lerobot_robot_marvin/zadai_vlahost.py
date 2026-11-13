"""Small Python-3.8-compatible client for VLAHost's Zadai HTTP protocol."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests

from .errors import SafetyError
from .zadai_rl100_policy import ZADAI_STATE_DIM


class ZadaiVLAHostClient:
    """Subset of VLAHost's MarvainM6HttpRobot needed by the RL100 rollout."""

    def __init__(self, base_url: str, timeout: float, safety_stats_path: Optional[str], action_clip_margin: float, max_relative_target: float, default_gripper_pos: float):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.action_clip_margin, self.max_relative_target = action_clip_margin, max_relative_target
        self.default_gripper_pos = default_gripper_pos
        self.session = requests.Session()
        self.last_arm = self.observation_generation = None
        self.bounds = self._load_bounds(safety_stats_path)

    @staticmethod
    def _load_bounds(path: Optional[str]):
        if not path:
            return None
        stats_file = Path(path).expanduser() / "meta" / "stats.json"
        if not stats_file.is_file():
            raise FileNotFoundError("safety stats not found: {}".format(stats_file))
        with stats_file.open() as stream:
            stats = json.load(stream)
        values = stats.get("action", stats.get("observation.state"))
        if not isinstance(values, dict):
            raise ValueError("safety stats need action or observation.state")
        low, high = np.asarray(values["min"], np.float32), np.asarray(values["max"], np.float32)
        if low.shape != (ZADAI_STATE_DIM,) or high.shape != (ZADAI_STATE_DIM,):
            raise ValueError("safety stats must contain 16-D min/max")
        return np.stack([low, high], axis=1)

    def _state(self) -> dict:
        response = self.session.get(self.base_url + "/state", timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("VLAHost /state did not return an object")
        intervention = data.get("intervention")
        self.observation_generation = intervention.get("refresh_generation") if isinstance(intervention, dict) else None
        return data

    def connect(self) -> None:
        self._state()

    @staticmethod
    def _decode_image(encoded: str) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("VLAHost returned an invalid JPEG")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _mjpeg_frame(self, url: str) -> np.ndarray:
        if not url.startswith("http"):
            url = self.base_url + "/" + url.lstrip("/")
        response, data = self.session.get(url, stream=True, timeout=self.timeout), bytearray()
        try:
            for part in response.iter_content(4096):
                data.extend(part)
                begin, end = data.find(b"\xff\xd8"), data.find(b"\xff\xd9")
                if begin >= 0 and end > begin:
                    return self._decode_image(base64.b64encode(bytes(data[begin:end + 2])).decode())
        finally:
            response.close()
        raise RuntimeError("VLAHost MJPEG stream did not yield a full JPEG")

    @staticmethod
    def _split_quad(image: np.ndarray) -> dict:
        height, width = image.shape[:2]
        if height % 3 == 0 and width % 2 == 0:  # VLAHost's current 1+2 layout.
            top_height, lower_height, half_width = 2 * height // 3, height // 3, width // 2
            top = cv2.resize(image[:top_height], (half_width, lower_height), interpolation=cv2.INTER_AREA)
            return {"top": top, "wrist_L": image[top_height:, :half_width].copy(), "wrist_R": image[top_height:, half_width:].copy()}
        if height % 2 or width % 2:
            raise RuntimeError("VLAHost quad image is neither 1+2 nor 2x2")
        half_height, half_width = height // 2, width // 2
        return {"top": image[:half_height, :half_width].copy(), "wrist_L": image[half_height:, :half_width].copy(), "wrist_R": image[half_height:, half_width:].copy()}

    @staticmethod
    def _gripper(value: Any, fallback: float) -> float:
        if isinstance(value, dict) and value.get("position") is not None:
            return float(value["position"])
        if isinstance(value, list) and value:
            return float(value[0])
        return float(value) if isinstance(value, (int, float)) else fallback

    def get_observation(self) -> dict:
        data = self._state()
        joints = data.get("joint_states", {}).get("positions")
        if not isinstance(joints, list) or len(joints) != 14:
            raise RuntimeError("VLAHost /state must provide 14 joint_states.positions values")
        state = np.asarray(joints + [self._gripper(data.get("gripper_left"), self.default_gripper_pos), self._gripper(data.get("gripper_right"), self.default_gripper_pos)], np.float32)
        if not np.isfinite(state).all():
            raise SafetyError("VLAHost state contains NaN or Inf")
        self.last_arm = state[:14].copy()
        quad = data.get("quad_image")
        if not isinstance(quad, dict):
            raise RuntimeError("VLAHost /state is missing quad_image")
        image = self._decode_image(quad["data"]) if "data" in quad else self._mjpeg_frame(quad["stream_url"])
        result = self._split_quad(image)
        result["state"] = state
        return result

    def send_action_chunk(self, action: np.ndarray, source_hz: float) -> None:
        if action.ndim != 2 or action.shape[1] != ZADAI_STATE_DIM:
            raise SafetyError("action chunk must be [T,16]")
        payload_actions = []
        for row in action:
            arm = row[:14].astype(np.float32, copy=True)
            if self.bounds is not None:
                arm = np.clip(arm, self.bounds[:14, 0] - self.action_clip_margin, self.bounds[:14, 1] + self.action_clip_margin)
            if self.last_arm is not None and self.max_relative_target > 0:
                arm = self.last_arm + np.clip(arm - self.last_arm, -self.max_relative_target, self.max_relative_target)
            self.last_arm = arm.copy()
            payload_actions.append({"jointcmd_left": arm[:7].tolist(), "jointcmd_right": arm[7:].tolist(), "gripper_left": float(row[14]), "gripper_right": float(row[15])})
        body = {"actions": payload_actions, "source_hz": float(source_hz)}
        if self.observation_generation is not None:
            body["observation_generation"] = int(self.observation_generation)
        response = self.session.post(self.base_url + "/action_chunk", json=body, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or not result.get("success", False):
            raise RuntimeError("VLAHost rejected action chunk: {!r}".format(result))

    def close(self) -> None:
        self.session.close()
