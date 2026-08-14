"""USB task episode boundary, intentionally separate from Robot.send_action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from lerobot.lerobot_types import RobotAction, RobotObservation

from .marvin import MarvinRobot


class ResettablePolicy(Protocol):
    def reset(self) -> None: ...


@dataclass(frozen=True)
class EpisodeResult:
    steps: int
    reward: float
    succeeded: bool
    terminated: bool
    truncated: bool


class MarvinEpisodeController:
    def __init__(self, robot: MarvinRobot, max_steps: int = 200):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.robot = robot
        self.max_steps = max_steps

    def run_episode(
        self,
        policy: ResettablePolicy,
        action_fn: Callable[[RobotObservation], RobotAction],
    ) -> EpisodeResult:
        # reset_episode owns task/reset semantics; send_action remains one step.
        self.robot.reset_episode()
        policy.reset()  # Clears ACT action queue and processor state through the runner.
        total_reward = 0.0
        try:
            for step in range(1, self.max_steps + 1):
                observation = self.robot.get_observation()
                action = action_fn(observation)
                self.robot.send_action(action)
                transition = self.robot.last_transition
                if transition is None:
                    raise RuntimeError("Bridge step returned no transition metadata")
                total_reward += float(transition["reward"])
                if transition["terminated"] or transition["truncated"]:
                    return EpisodeResult(
                        steps=step,
                        reward=total_reward,
                        succeeded=bool(transition["succeed"]),
                        terminated=bool(transition["terminated"]),
                        truncated=bool(transition["truncated"]),
                    )
            # Client-side horizon is a truncation even if the bridge environment
            # was configured with a larger horizon.
            return EpisodeResult(self.max_steps, total_reward, False, False, True)
        except Exception:
            self.robot.emergency_stop("episode controller exception")
            raise

