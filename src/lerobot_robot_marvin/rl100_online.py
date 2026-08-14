"""Marvin adapters for RL-100's original on-policy online update.

The algorithm remains in RL-100: diffusion trajectories/log-probabilities are
sampled by ``RL1002D.all_step_action_logprob`` and optimized by
``UniPPO.dp_align_update_no_share``.  This module only supplies the missing
Marvin Bridge environment contract and a four-camera on-policy buffer.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np

from .errors import SafetyError
from .rl100_policy import MarvinObservationHistory, RL100_CAMERA_KEYS
from .schema import validate_action


class MarvinBridgeOnlineEnv:
    """Expose Bridge control as RL-100's chunk-level ``step_online`` API."""

    def __init__(
        self,
        client,
        n_obs_steps: int = 2,
        n_action_steps: int = 5,
        max_episode_steps: int = 150,
        action_match_atol: float = 1e-6,
    ):
        if n_obs_steps <= 0 or n_action_steps <= 0 or max_episode_steps <= 0:
            raise ValueError("observation, action and episode lengths must be positive")
        if action_match_atol < 0:
            raise ValueError("action_match_atol must be non-negative")
        self.client = client
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.max_episode_steps = int(max_episode_steps)
        self.action_match_atol = float(action_match_atol)
        self.history = MarvinObservationHistory(self.n_obs_steps)
        self.episode_steps = 0

    def _policy_observation(self) -> dict[str, np.ndarray]:
        # MarvinObservationHistory includes a batch dimension for deployment;
        # RL-100's environment contract expects an unbatched observation.
        return {key: value[0] for key, value in self.history.as_numpy().items()}

    def reset(self) -> tuple[dict[str, np.ndarray], dict]:
        observation, info = self.client.reset()
        self.history.reset(observation)
        self.episode_steps = 0
        return self._policy_observation(), dict(info)

    def step_online(self, action_chunk, gamma: float = 0.99):
        chunk = np.asarray(action_chunk, dtype=np.float32)
        expected_shape = (self.n_action_steps, 5)
        if chunk.shape != expected_shape:
            raise SafetyError(f"online action chunk must have shape {expected_shape}, got {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            raise SafetyError("online action chunk contains NaN or Inf")
        if np.any(chunk < -1.0) or np.any(chunk > 1.0):
            raise SafetyError(
                "online action chunk exceeds [-1, 1] "
                f"(min={float(chunk.min()):.8f}, max={float(chunk.max()):.8f}); "
                "refusing to clip an on-policy action"
            )
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")

        rewards: list[float] = []
        actual_actions: list[np.ndarray] = []
        terminated = False
        truncated = False
        succeed = False
        final_info: dict = {}

        for policy_action in chunk:
            transition = self.client.step(validate_action(policy_action))
            actual_action = validate_action(transition["actual_action"])
            if not np.allclose(
                actual_action,
                policy_action,
                rtol=0.0,
                atol=self.action_match_atol,
            ):
                raise SafetyError(
                    "Bridge actual_action differs from the sampled policy action; "
                    "the transition is not on-policy"
                )
            self.history.append(transition["observation"])
            rewards.append(float(transition["reward"]))
            actual_actions.append(actual_action.copy())
            self.episode_steps += 1
            terminated = bool(transition["terminated"])
            truncated = bool(transition["truncated"])
            succeed = bool(transition["succeed"])
            final_info = dict(transition.get("info", {}))
            if not terminated and not truncated and self.episode_steps >= self.max_episode_steps:
                truncated = True
                final_info["client_time_limit"] = True
            if terminated or truncated:
                break

        discounted_reward = sum((gamma**index) * reward for index, reward in enumerate(rewards))
        info = {
            **final_info,
            "terminated": terminated,
            "truncated": truncated,
            "succeed": succeed,
            "executed_steps": len(rewards),
            "policy_action": np.ascontiguousarray(chunk[: len(rewards)]),
            "actual_action": np.ascontiguousarray(actual_actions, dtype=np.float32),
            "substep_rewards": np.asarray(rewards, dtype=np.float32),
        }
        return (
            self._policy_observation(),
            float(discounted_reward),
            bool(terminated or truncated),
            info,
        )


class MarvinOnlineReplayBuffer:
    """Four-camera on-policy storage compatible with UniPPO's tensor API.

    Arrays are allocated from the first sampled diffusion trajectory so the
    buffer preserves the policy's full horizon, not only the executed prefix.
    RGB observations remain uint8 in host memory and are converted to float on
    the training device only when an update starts.
    """

    def __init__(self, capacity: int, device, policy_version: int = 0):
        if capacity <= 1:
            raise ValueError("online buffer capacity must be greater than one")
        self.capacity = int(capacity)
        self.device = device
        self.policy_version = int(policy_version)
        self.count = 0
        self._allocated = False

    @staticmethod
    def _validate_observation(obs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        required = {"agent_pos", *RL100_CAMERA_KEYS}
        missing = required.difference(obs)
        extra = set(obs).difference(required)
        if missing or extra:
            raise KeyError(f"invalid online observation keys; missing={sorted(missing)} extra={sorted(extra)}")
        result = {"agent_pos": np.asarray(obs["agent_pos"], dtype=np.float32)}
        for key in RL100_CAMERA_KEYS:
            value = np.asarray(obs[key])
            if value.dtype != np.uint8:
                raise TypeError(f"{key} must remain uint8 in the online buffer, got {value.dtype}")
            result[key] = value
        return result

    def _allocate(self, obs, diffusion_path, old_logprob) -> None:
        self.obs = {
            "agent_pos": np.empty((self.capacity, *obs["agent_pos"].shape), dtype=np.float32),
            **{
                key: np.empty((self.capacity, *obs[key].shape), dtype=np.uint8)
                for key in RL100_CAMERA_KEYS
            },
        }
        self.next_obs = {
            key: np.empty_like(value) for key, value in self.obs.items()
        }
        self.action = np.empty((self.capacity, *diffusion_path.shape), dtype=np.float32)
        self.a_logprob = np.empty((self.capacity, *old_logprob.shape), dtype=np.float32)
        self.reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, 1), dtype=np.float32)
        self.dw = np.empty((self.capacity, 1), dtype=np.float32)
        self._allocated = True

    def store(self, obs, diffusion_path, old_logprob, reward, next_obs, done, dw) -> None:
        if self.count >= self.capacity:
            raise RuntimeError("online replay buffer is full")
        obs = self._validate_observation(obs)
        next_obs = self._validate_observation(next_obs)
        diffusion_path = np.asarray(diffusion_path, dtype=np.float32)
        old_logprob = np.asarray(old_logprob, dtype=np.float32)
        if diffusion_path.ndim != 3 or old_logprob.ndim != 3:
            raise ValueError("diffusion_path and old_logprob must be [denoise, horizon, action_dim]")
        if diffusion_path.shape[0] != old_logprob.shape[0] + 1:
            raise ValueError("diffusion_path must contain one more denoising state than logprob")
        if diffusion_path.shape[1:] != old_logprob.shape[1:]:
            raise ValueError("diffusion_path and old_logprob event shapes differ")
        if diffusion_path.shape[-1] != 5:
            raise ValueError("Marvin diffusion action dimension must be 5")
        if not np.all(np.isfinite(diffusion_path)) or not np.all(np.isfinite(old_logprob)):
            raise ValueError("diffusion trajectory contains NaN or Inf")
        if not self._allocated:
            self._allocate(obs, diffusion_path, old_logprob)
        if diffusion_path.shape != self.action.shape[1:] or old_logprob.shape != self.a_logprob.shape[1:]:
            raise ValueError("diffusion trajectory shape changed within one policy buffer")

        index = self.count
        for key in self.obs:
            if obs[key].shape != self.obs[key].shape[1:] or next_obs[key].shape != self.next_obs[key].shape[1:]:
                raise ValueError(f"observation shape changed for {key}")
            self.obs[key][index] = obs[key]
            self.next_obs[key][index] = next_obs[key]
        self.action[index] = diffusion_path
        self.a_logprob[index] = old_logprob
        self.reward[index, 0] = float(reward)
        self.done[index, 0] = float(bool(done))
        self.dw[index, 0] = float(bool(dw))
        self.count += 1

    def numpy_to_tensor(self):
        if self.count != self.capacity:
            raise RuntimeError(
                f"PPO update requires a full on-policy buffer: {self.count}/{self.capacity}"
            )
        import torch

        def obs_tensor(source):
            return {
                key: torch.as_tensor(value[: self.count], device=self.device, dtype=torch.float32)
                for key, value in source.items()
            }

        return (
            obs_tensor(self.obs),
            torch.as_tensor(self.action[: self.count], device=self.device),
            torch.as_tensor(self.a_logprob[: self.count], device=self.device),
            torch.as_tensor(self.reward[: self.count], device=self.device),
            obs_tensor(self.next_obs),
            torch.as_tensor(self.dw[: self.count], device=self.device),
            torch.as_tensor(self.done[: self.count], device=self.device),
        )

    def clear(self, policy_version: int) -> None:
        self.count = 0
        self.policy_version = int(policy_version)

    def state_dict(self) -> dict:
        payload = {
            "capacity": self.capacity,
            "count": self.count,
            "policy_version": self.policy_version,
            "allocated": self._allocated,
        }
        if self._allocated:
            payload.update(
                obs={key: value[: self.count] for key, value in self.obs.items()},
                next_obs={key: value[: self.count] for key, value in self.next_obs.items()},
                action=self.action[: self.count],
                a_logprob=self.a_logprob[: self.count],
                reward=self.reward[: self.count],
                done=self.done[: self.count],
                dw=self.dw[: self.count],
            )
        return payload

    def load_state_dict(self, payload: dict, expected_policy_version: int) -> None:
        if int(payload["capacity"]) != self.capacity:
            raise ValueError("saved online buffer capacity differs from current configuration")
        if int(payload["policy_version"]) != int(expected_policy_version):
            raise ValueError("saved online buffer belongs to a different policy version")
        count = int(payload["count"])
        if not 0 <= count <= self.capacity:
            raise ValueError("saved online buffer count is invalid")
        self.policy_version = int(expected_policy_version)
        self.count = 0
        if not payload.get("allocated", False) or count == 0:
            return
        first_obs = {key: value[0] for key, value in payload["obs"].items()}
        self._allocate(first_obs, payload["action"][0], payload["a_logprob"][0])
        for key in self.obs:
            self.obs[key][:count] = payload["obs"][key]
            self.next_obs[key][:count] = payload["next_obs"][key]
        for key in ("action", "a_logprob", "reward", "done", "dw"):
            getattr(self, key)[:count] = payload[key]
        self.count = count

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(self.state_dict(), stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    def load(self, path: str | Path, expected_policy_version: int) -> None:
        with Path(path).open("rb") as stream:
            self.load_state_dict(pickle.load(stream), expected_policy_version)


def write_latest_metadata(path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
