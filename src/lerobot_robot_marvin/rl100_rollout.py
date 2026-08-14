"""Deploy an RL-100 2D diffusion-policy checkpoint through Marvin Bridge."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque

import numpy as np

from .client import MarvinBridgeClient
from .constants import FPS
from .errors import BridgeError, SafetyError
from .rl100_policy import RL100PolicyRunner, sanitize_action_chunk


LOG = logging.getLogger(__name__)


def validate_bridge_preflight(status: dict, allow_real_robot: bool) -> None:
    if status.get("backend") == "hardware" and not allow_real_robot:
        raise RuntimeError("Hardware bridge detected; add --allow-real-robot only after mock checks")


def scale_motion_action(action: np.ndarray, motion_scale: float) -> np.ndarray:
    if not 0.0 <= motion_scale <= 1.0:
        raise ValueError("motion_scale must be between 0 and 1")
    vector = np.asarray(action, dtype=np.float32).copy()
    if vector.shape != (5,):
        raise SafetyError(f"Action must have shape (5,), got {vector.shape}")
    vector[:4] *= motion_scale
    return np.ascontiguousarray(vector)


class BridgeHeartbeat:
    """Keep the bridge watchdog alive while diffusion inference is running."""

    def __init__(
        self,
        client: MarvinBridgeClient,
        interval_s: float = 0.5,
        max_consecutive_failures: int = 3,
    ):
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
            # A successful reset/step response is stronger evidence of bridge
            # health than a status probe.  Do not make the single-threaded
            # hardware HTTP server queue an extra status RPC while the primary
            # 10 Hz control stream is already keeping the watchdog alive.
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
                    LOG.error(
                        "Bridge heartbeat failed %d consecutive times: %s",
                        failures,
                        exc,
                    )
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
        """Record a successful reset/step response from the control client."""
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


class RL100RolloutController:
    """Policy-independent episode loop, testable without torch or RL-100."""

    def __init__(
        self,
        client,
        policy,
        max_steps: int,
        actions_per_inference: int,
        motion_scale: float,
        dry_run: bool = False,
        max_consecutive_clipped_chunks: int = 5,
        seed: int = 1000,
        bridge_contact_callback=None,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if actions_per_inference <= 0:
            raise ValueError("actions_per_inference must be positive")
        if actions_per_inference > policy.n_action_steps:
            raise ValueError(
                f"actions_per_inference={actions_per_inference} exceeds checkpoint "
                f"n_action_steps={policy.n_action_steps}"
            )
        if max_consecutive_clipped_chunks <= 0:
            raise ValueError("max_consecutive_clipped_chunks must be positive")
        if not 0.0 <= motion_scale <= 1.0:
            raise ValueError("motion_scale must be between 0 and 1")
        self.client = client
        self.policy = policy
        self.max_steps = max_steps
        self.actions_per_inference = actions_per_inference
        self.motion_scale = motion_scale
        self.dry_run = dry_run
        self.max_consecutive_clipped_chunks = max_consecutive_clipped_chunks
        self.seed = seed
        self.bridge_contact_callback = bridge_contact_callback

    def _note_bridge_contact(self) -> None:
        if self.bridge_contact_callback is not None:
            self.bridge_contact_callback()

    def run_episode(self, episode_index: int = 0) -> dict:
        observation, info = self.client.reset()
        self._note_bridge_contact()
        self.policy.reset(observation, seed=self.seed + episode_index)
        queue = deque()
        consecutive_clipped = 0
        total_reward = 0.0
        inference_calls = 0

        for step in range(1, self.max_steps + 1):
            if not queue:
                chunk, clipped_rows = self.policy.predict_chunk()
                chunk, extra_clipped_rows = sanitize_action_chunk(chunk)
                clipped_rows += extra_clipped_rows
                inference_calls += 1
                if clipped_rows:
                    consecutive_clipped += 1
                    LOG.warning(
                        "RL-100 chunk contained %d clipped actions (consecutive clipped chunks=%d)",
                        clipped_rows,
                        consecutive_clipped,
                    )
                else:
                    consecutive_clipped = 0
                if consecutive_clipped >= self.max_consecutive_clipped_chunks:
                    raise SafetyError("Too many consecutive RL-100 chunks required clipping")
                queue.extend(chunk[: self.actions_per_inference])
                LOG.info(
                    "episode=%d inference=%d latency=%.3fs queued=%d first_action=%s",
                    episode_index,
                    inference_calls,
                    getattr(self.policy, "last_inference_seconds", float("nan")),
                    len(queue),
                    queue[0].tolist(),
                )

            action = scale_motion_action(queue.popleft(), self.motion_scale)
            if self.dry_run:
                LOG.info("episode=%d dry_run_step=%d action=%s", episode_index, step, action.tolist())
                continue

            transition = self.client.step(action)
            self._note_bridge_contact()
            self.policy.observe(transition["observation"])
            total_reward += float(transition["reward"])
            LOG.info(
                "episode=%d step=%d action=%s reward=%.4f terminated=%s truncated=%s",
                episode_index,
                step,
                action.tolist(),
                float(transition["reward"]),
                transition["terminated"],
                transition["truncated"],
            )
            if transition["terminated"] or transition["truncated"]:
                return {
                    "steps": step,
                    "reward": total_reward,
                    "succeeded": bool(transition["succeed"]),
                    "terminated": bool(transition["terminated"]),
                    "truncated": bool(transition["truncated"]),
                    "inference_calls": inference_calls,
                }

        return {
            "steps": self.max_steps,
            "reward": total_reward,
            "succeeded": False,
            "terminated": False,
            "truncated": True,
            "inference_calls": inference_calls,
            "dry_run": self.dry_run,
            "reset_info": info,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy RL-100 Diffusion Policy through Marvin Bridge")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-source", choices=("auto", "ema", "model"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--reset-timeout", type=float, default=120.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--actions-per-inference", type=int, default=1)
    parser.add_argument("--motion-scale", type=float, default=0.2)
    parser.add_argument("--max-consecutive-clipped-chunks", type=int, default=5)
    parser.add_argument("--heartbeat-interval", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--allow-real-robot", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.fps != FPS:
        raise ValueError(f"Marvin control contract requires --fps={FPS}, got {args.fps}")

    # A malformed checkpoint must fail before any bridge or hardware contact.
    policy = RL100PolicyRunner(
        checkpoint=args.checkpoint,
        device=args.device,
        checkpoint_source=args.checkpoint_source,
        deterministic=not args.stochastic,
    )
    if args.inspect_only:
        LOG.info("Checkpoint inspection completed without contacting a bridge")
        return

    client = MarvinBridgeClient(
        args.host,
        args.port,
        args.timeout,
        args.connect_timeout,
        args.reset_timeout,
    )
    status = client.status()
    validate_bridge_preflight(status, args.allow_real_robot)
    client.connect()
    heartbeat_client = MarvinBridgeClient(args.host, args.port, args.timeout)
    heartbeat = BridgeHeartbeat(heartbeat_client, args.heartbeat_interval)
    heartbeat.start()
    try:
        controller = RL100RolloutController(
            client=client,
            policy=policy,
            max_steps=args.max_steps,
            actions_per_inference=args.actions_per_inference,
            motion_scale=args.motion_scale,
            dry_run=args.dry_run,
            max_consecutive_clipped_chunks=args.max_consecutive_clipped_chunks,
            seed=args.seed,
            bridge_contact_callback=heartbeat.note_primary_contact,
        )
        for episode_index in range(args.episodes):
            heartbeat.check()
            started = time.monotonic()
            result = controller.run_episode(episode_index)
            LOG.info(
                "episode=%d result=%s elapsed=%.2fs",
                episode_index,
                result,
                time.monotonic() - started,
            )
            # The next iteration checks bridge health before reset.  A check
            # here has no safety benefit and can turn an already-confirmed
            # terminal success into a false CLI failure during teardown.
    except Exception:
        client.best_effort_stop("RL-100 rollout exception")
        raise
    finally:
        heartbeat.close()
        client.disconnect()


if __name__ == "__main__":
    main()
