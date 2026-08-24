from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..types import ChunkBatch


class LeRobotChunkDataset(Dataset):
    """Turn a LeRobot frame dataset into h-step chunk transitions.

    Episode outcomes are deliberately external metadata: filtering only successful
    demonstrations is acceptable for BC, but must never silently masquerade as a
    failure-aware RL replay.
    """

    def __init__(
        self,
        dataset,
        image_keys: tuple[str, ...],
        executed_steps: int,
        gamma: float,
        outcomes: dict[int, dict[str, Any]],
    ):
        self.dataset = dataset
        self.image_keys = image_keys
        self.executed_steps = executed_steps
        self.gamma = gamma
        self.outcomes = outcomes
        self.episode_lengths = {int(row["episode_index"]): int(row["length"]) for row in dataset.meta.episodes}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        episode = int(item["episode_index"])
        frame = int(item["frame_index"])
        # There are T-frame transitions left, including the transition at
        # `frame`.  A horizon h reaches terminal iff T-frame <= h.
        transitions_left = self.episode_lengths[episode] - frame
        actual_h = min(self.executed_steps, transitions_left)
        terminal = transitions_left <= self.executed_steps
        outcome = self.outcomes.get(episode)
        if outcome is None:
            raise KeyError(f"episode {episode} missing from outcome manifest")
        success = bool(outcome.get("success", False))
        if "rl.reward" in item:
            rewards = torch.as_tensor(item["rl.reward"], dtype=torch.float32).reshape(-1)[:actual_h]
            powers = torch.pow(torch.tensor(self.gamma), torch.arange(actual_h, dtype=torch.float32))
            reward = (rewards * powers).sum()
            if "rl.terminated" in item:
                terminal = bool(torch.as_tensor(item["rl.terminated"]).reshape(-1)[:actual_h].any())
        else:
            reward = torch.tensor(
                float(outcome.get("terminal_reward", 1.0 if success else 0.0)) if terminal else 0.0,
                dtype=torch.float32,
            )
        result = {
            "observation.state": item["observation.state"][0],
            "next.observation.state": item["observation.state"][1],
            "action": item["action"],
            "action_is_pad": item["action_is_pad"],
            "reward": torch.as_tensor(reward, dtype=torch.float32),
            "discount": torch.tensor(self.gamma**actual_h, dtype=torch.float32),
            "terminated": torch.tensor(terminal, dtype=torch.float32),
            "truncated": torch.tensor(False, dtype=torch.float32),
            "success": torch.tensor(success, dtype=torch.float32),
            "intervention": torch.as_tensor(item.get("rl.intervention", False), dtype=torch.float32).reshape(-1)[0],
            "executed_steps": torch.tensor(actual_h, dtype=torch.int64),
            "episode_index": item["episode_index"],
            "frame_index": item["frame_index"],
            "source": str(outcome.get("source", "offline")),
        }
        for key in self.image_keys:
            result[key] = item[key][0]
            result[f"next.{key}"] = item[key][1]
        return result


def load_outcomes(path: str | Path) -> dict[int, dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    episodes = raw.get("episodes", raw)
    return {int(key): dict(value) for key, value in episodes.items()}


def make_delta_timestamps(dataset_root: str | Path, image_keys: tuple[str, ...], chunk_size: int,
                          executed_steps: int, fps: int = 10) -> dict[str, list[float]]:
    delta = {"action": [i / fps for i in range(chunk_size)]}
    for key in ("observation.state", *image_keys):
        delta[key] = [0.0, executed_steps / fps]
    info = json.loads((Path(dataset_root) / "meta" / "info.json").read_text())
    features = info.get("features", {})
    for key in ("rl.reward", "rl.terminated"):
        if key in features:
            delta[key] = [i / fps for i in range(executed_steps)]
    for key in ("rl.success", "rl.intervention", "rl.safety_abort"):
        if key in features:
            delta[key] = [0.0]
    return delta


def prepare_lerobot_batch(raw: dict[str, Any], preprocessor, image_keys: tuple[str, ...]) -> ChunkBatch:
    """Split current/next observations, then apply checkpoint normalization."""
    current = {"observation.state": raw["observation.state"], "action": raw["action"],
               "action_is_pad": raw["action_is_pad"]}
    next_frame = {"observation.state": raw["next.observation.state"]}
    for key in image_keys:
        current[key] = raw[key]
        next_frame[key] = raw[f"next.{key}"]
    current = preprocessor(current)
    next_frame = preprocessor(next_frame)
    critic_raw = raw.get("critic_action", raw["action"])
    critic_pad = raw.get("critic_action_is_pad", raw["action_is_pad"])
    policy_raw = raw.get("policy_action", raw["action"])
    # Apply the checkpoint's action normalizer to auxiliary action tensors.
    # Observation/image processing is intentionally omitted for these calls.
    critic_processed = preprocessor({
        "observation.state": raw["observation.state"],
        "action": critic_raw,
        "action_is_pad": critic_pad,
    })
    critic_action = critic_processed["action"]
    critic_action = critic_action.masked_fill(critic_pad.to(critic_action.device).unsqueeze(-1), 0.0)
    zeros = torch.zeros(len(raw["reward"]), dtype=torch.float32)
    return ChunkBatch(
        state=current["observation.state"], action=current["action"],
        next_state=next_frame["observation.state"], reward=raw["reward"],
        discount=raw["discount"], terminated=raw["terminated"],
        action_is_pad=current["action_is_pad"],
        critic_action=critic_action, policy_action=policy_raw.to(critic_action.device),
        critic_action_is_pad=critic_pad,
        truncated=raw.get("truncated", zeros), success=raw.get("success", zeros),
        intervention=raw.get("intervention", zeros),
        executed_steps=raw.get("executed_steps"),
        images={key: current[key] for key in image_keys},
        next_images={key: next_frame[key] for key in image_keys}, source=raw.get("source"),
    )
