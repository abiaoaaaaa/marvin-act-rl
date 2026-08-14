"""ACT checkpoint deployment loop with explicit episode reset semantics."""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import nullcontext

import numpy as np

from .config_marvin import MarvinRobotConfig
from .constants import CAMERA_KEYS, FPS, ROBOT_ACTION_KEYS, ROBOT_STATE_KEYS
from .episode import MarvinEpisodeController
from .errors import SafetyError
from .marvin import MarvinRobot
from .schema import validate_action

LOG = logging.getLogger(__name__)


def validate_bridge_preflight(status: dict, allow_real_robot: bool) -> None:
    if status.get("backend") == "hardware" and not allow_real_robot:
        raise RuntimeError("Hardware bridge detected; add --allow-real-robot only after completing mock checks")


def sanitize_policy_action(action: np.ndarray) -> tuple[np.ndarray, bool]:
    """Reject non-finite model output and clip finite output to the normalized action space."""
    vector = np.asarray(action, dtype=np.float32)
    if vector.shape != (5,):
        raise SafetyError(f"ACT output must have shape (5,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise SafetyError("ACT output contains NaN or Inf")
    clipped = np.clip(vector, -1.0, 1.0)
    return np.ascontiguousarray(clipped), not np.array_equal(clipped, vector)


class ACTPolicyRunner:
    def __init__(self, checkpoint: str, device: str, task: str, n_action_steps: int | None = None):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.device = torch.device(device)
        self.task = task
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if config.type != "act":
            raise ValueError(f"This deployment entry point requires ACT, checkpoint type is {config.type!r}")
        if n_action_steps is not None:
            if n_action_steps <= 0:
                raise ValueError("n_action_steps must be positive")
            if n_action_steps > config.chunk_size:
                raise ValueError(
                    f"n_action_steps={n_action_steps} exceeds checkpoint chunk_size={config.chunk_size}"
                )
            config.n_action_steps = n_action_steps
            LOG.warning(
                "Overriding checkpoint n_action_steps: execute %d actions from each predicted chunk of %d",
                config.n_action_steps,
                config.chunk_size,
            )
        config.device = device
        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(checkpoint, config=config).to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.action_clip_count = 0
        self.last_raw_action: np.ndarray | None = None
        self.last_action_was_clipped = False

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def action(self, observation: dict) -> dict[str, float]:
        from lerobot.policies.utils import prepare_observation_for_inference

        state = np.asarray([observation[key] for key in ROBOT_STATE_KEYS], dtype=np.float32)
        frame = {"observation.state": state}
        for key in CAMERA_KEYS:
            frame[f"observation.images.{key}"] = observation[key]
        frame = prepare_observation_for_inference(frame, self.device, self.task, "marvin")
        frame = self.preprocessor(frame)
        amp = (
            self.torch.autocast(device_type="cuda")
            if self.device.type == "cuda" and self.policy.config.use_amp
            else nullcontext()
        )
        with self.torch.inference_mode(), amp:
            action = self.policy.select_action(frame)
            action = self.postprocessor(action)
        raw_vector = action.squeeze(0).detach().cpu().numpy()
        vector, was_clipped = sanitize_policy_action(raw_vector)
        self.last_raw_action = np.asarray(raw_vector, dtype=np.float32).copy()
        self.last_action_was_clipped = was_clipped
        if was_clipped:
            self.action_clip_count += 1
            if self.action_clip_count <= 5 or self.action_clip_count % 100 == 0:
                LOG.warning(
                    "Clipped finite ACT output to [-1, 1] (count=%d, raw=%s)",
                    self.action_clip_count,
                    self.last_raw_action.tolist(),
                )
        vector = validate_action(vector)
        return {key: float(vector[index]) for index, key in enumerate(ROBOT_ACTION_KEYS)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy a trained LeRobot ACT policy through Marvin bridge")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--reset-timeout", type=float, default=120.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="Insert the USB plug")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override checkpoint n_action_steps (must be between 1 and chunk_size).",
    )
    parser.add_argument("--allow-real-robot", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    if args.fps != FPS:
        raise ValueError(f"This Marvin dataset/control contract requires --fps={FPS}, got {args.fps}")
    # Load the checkpoint before opening a bridge session.  A bad checkpoint
    # therefore cannot touch hardware.
    runner = ACTPolicyRunner(
        args.checkpoint,
        args.device,
        args.task,
        n_action_steps=args.n_action_steps,
    )
    robot = MarvinRobot(
        MarvinRobotConfig(
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
            connect_timeout_s=args.connect_timeout,
            reset_timeout_s=args.reset_timeout,
            id="marvin_usb",
        )
    )
    # The status RPC is deliberately available before connect.  Refuse a
    # hardware bridge before constructing its HIL-SERL environment.
    status = robot.status()
    validate_bridge_preflight(status, args.allow_real_robot)
    robot.connect()
    try:
        status = robot.status()
        if status.get("fps") != FPS:
            raise RuntimeError(f"Bridge reports incompatible fps={status.get('fps')}, expected {FPS}")
        controller = MarvinEpisodeController(robot, max_steps=args.max_steps)

        def policy_action(observation):
            # MarvinUSBEnv.step() already owns the 10 Hz execution window.
            # Sleeping here as well would approximately halve the control rate.
            action = runner.action(observation)
            return action

        for episode_index in range(args.episodes):
            started = time.monotonic()
            result = controller.run_episode(runner, policy_action)
            elapsed = time.monotonic() - started
            LOG.info("episode=%d result=%s", episode_index, result)
            LOG.info("episode=%d effective_control_hz=%.2f", episode_index, result.steps / max(elapsed, 1e-9))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
