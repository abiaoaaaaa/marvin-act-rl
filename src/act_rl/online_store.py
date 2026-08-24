from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

REPLAY_SCHEMA_VERSION = 2


class OnlineEpisodeWriter:
    """Append-only one-file-per-episode storage; a crash cannot corrupt old episodes."""

    def __init__(self, root: str | Path, image_keys: tuple[str, ...], chunk_size: int, action_dim: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_keys, self.chunk_size, self.action_dim = image_keys, chunk_size, action_dim

    def write(self, episode_id: int, records: list[dict], metadata: dict | None = None) -> Path:
        if not records:
            raise ValueError("refusing to write an empty online episode")
        action = np.stack([x["action_chunk"] for x in records]).astype(np.float32)
        critic_action = np.stack([x.get("critic_action_chunk", x["action_chunk"]) for x in records]).astype(np.float32)
        arrays = {
            "state": np.stack([x["observation"]["state"] for x in records]).astype(np.float32),
            "next_state": np.stack([x["next_observation"]["state"] for x in records]).astype(np.float32),
            "action": action,
            "policy_action": np.stack([x.get("policy_action_chunk", x["action_chunk"]) for x in records]).astype(np.float32),
            "command_action": np.stack([
                x.get("command_action_chunk", x.get("policy_action_chunk", x["action_chunk"]))
                for x in records
            ]).astype(np.float32),
            "critic_action": critic_action,
            "actual_action": np.stack([x.get("actual_action_chunk", critic_action[i]) for i, x in enumerate(records)]).astype(np.float32),
            "reward": np.asarray([x["reward"] for x in records], np.float32),
            "discount": np.asarray([x["discount"] for x in records], np.float32),
            "terminated": np.asarray([x["terminated"] for x in records], np.float32),
            "truncated": np.asarray([x.get("truncated", False) for x in records], np.float32),
            "success": np.asarray([x.get("success", False) for x in records], np.float32),
            "intervention": np.asarray([x.get("intervention", False) for x in records], np.float32),
            "safety_abort": np.asarray([x.get("safety_abort", False) for x in records], np.float32),
            "executed_steps": np.asarray([x.get("executed_steps", self.chunk_size) for x in records], np.int64),
            "action_is_pad": np.stack([x["action_is_pad"] for x in records]).astype(bool),
            "critic_action_is_pad": np.stack(
                [x.get("critic_action_is_pad", x["action_is_pad"]) for x in records]
            ).astype(bool),
            "executed_mask": np.stack(
                [x.get("executed_mask", ~x.get("critic_action_is_pad", x["action_is_pad"])) for x in records]
            ).astype(bool),
            "intervention_mask": np.stack(
                [x.get("intervention_mask", np.zeros(self.chunk_size, dtype=bool)) for x in records]
            ).astype(bool),
        }
        if arrays["action"].shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError(f"bad action chunk shape {arrays['action'].shape}")
        for key in ("policy_action", "command_action", "critic_action", "actual_action"):
            if arrays[key].shape != arrays["action"].shape:
                raise ValueError(f"bad {key} chunk shape {arrays[key].shape}")
        for key in self.image_keys:
            observation_key = key.removeprefix("observation.images.")
            arrays[key] = np.stack([x["observation"][observation_key] for x in records]).astype(np.uint8)
            arrays[f"next.{key}"] = np.stack(
                [x["next_observation"][observation_key] for x in records]
            ).astype(np.uint8)
        path = self.root / f"episode_{episode_id:06d}.npz"
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(f"online replay is append-only; episode already exists: {path}")
        temporary = self.root / f".episode_{episode_id:06d}.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        sidecar = path.with_suffix(".json")
        episode_metadata = dict(metadata or {})
        episode_metadata.setdefault("schema_version", REPLAY_SCHEMA_VERSION)
        episode_metadata.setdefault("episode_id", episode_id)
        episode_metadata.setdefault("transitions", len(records))
        sidecar.write_text(json.dumps(episode_metadata, indent=2, ensure_ascii=False))
        return path

    def next_episode_id(self) -> int:
        ids = [int(path.stem.split("_")[-1]) for path in self.root.glob("episode_*.npz")]
        return max(ids, default=-1) + 1


class OnlineNpzDataset(Dataset):
    def __init__(self, root: str | Path, image_keys: tuple[str, ...], require_schema_version: int | None = None):
        self.image_keys = image_keys
        self.files = sorted(Path(root).glob("episode_*.npz"))
        self.index: list[tuple[int, int]] = []
        self.row_metadata: list[dict[str, object]] = []
        self._cache_id, self._cache = None, None
        for file_id, path in enumerate(self.files):
            sidecar = path.with_suffix(".json")
            metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            schema_version = int(metadata.get("schema_version", 1))
            if require_schema_version is not None and schema_version < require_schema_version:
                raise ValueError(
                    f"online replay {path} uses schema v{schema_version}; "
                    f"stage-1 training requires v{require_schema_version}"
                )
            with np.load(path, allow_pickle=False) as data:
                size = len(data["reward"])
                for key in ("state", "next_state", "action", "reward", "discount", "terminated", "action_is_pad"):
                    if key not in data or len(data[key]) != size:
                        raise ValueError(f"invalid online replay {path}: missing or mis-sized {key}")
                for row in range(size):
                    self.index.append((file_id, row))
                    self.row_metadata.append({
                        "episode_id": int(metadata.get("episode_id", file_id)),
                        "schema_version": schema_version,
                        "policy_sha256": str(metadata.get("policy_sha256", "unknown")),
                        "success": bool(metadata.get("success", False)) or (
                            bool(data["success"][row]) if "success" in data else False
                        ),
                        "failure": bool(metadata.get("failure", False)),
                        "intervention": bool(data["intervention"][row]) if "intervention" in data else False,
                    })

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int):
        file_id, row = self.index[index]
        if self._cache_id != file_id:
            if self._cache is not None:
                self._cache.close()
            self._cache = np.load(self.files[file_id], allow_pickle=False)
            self._cache_id = file_id
        data = self._cache
        result = {key: torch.from_numpy(np.asarray(data[key][row])) for key in
                  ("state", "next_state", "action", "reward", "discount", "terminated", "action_is_pad")}
        result["observation.state"], result["next.observation.state"] = result.pop("state"), result.pop("next_state")
        result["critic_action"] = torch.from_numpy(np.asarray(data["critic_action"][row])) if "critic_action" in data else result["action"].clone()
        result["policy_action"] = torch.from_numpy(np.asarray(data["policy_action"][row])) if "policy_action" in data else result["action"].clone()
        result["critic_action_is_pad"] = (
            torch.from_numpy(np.asarray(data["critic_action_is_pad"][row]))
            if "critic_action_is_pad" in data else result["action_is_pad"].clone()
        )
        for key, dtype in (("truncated", torch.float32), ("success", torch.float32),
                           ("intervention", torch.float32), ("executed_steps", torch.int64)):
            value = data[key][row] if key in data else 0
            result[key] = torch.as_tensor(value, dtype=dtype)
        result["source"] = "online"
        for key in self.image_keys:
            result[key] = torch.from_numpy(np.asarray(data[key][row])).permute(2, 0, 1).float() / 255.0
            result[f"next.{key}"] = torch.from_numpy(np.asarray(data[f"next.{key}"][row])).permute(2, 0, 1).float() / 255.0
        return result

    def sampling_weights(
        self,
        recent_ratio: float = 0.5,
        recent_window: int = 10_000,
        balance_outcomes: bool = True,
        intervention_weight: float = 2.0,
    ) -> torch.Tensor:
        """Weights for replacement sampling: recent/uniform mixture plus outcome balance."""
        if not 0.0 <= recent_ratio <= 1.0:
            raise ValueError("recent_ratio must be in [0, 1]")
        count = len(self)
        if count == 0:
            return torch.empty(0, dtype=torch.double)
        recent_count = min(count, max(int(recent_window), 1))
        weights = np.full(count, (1.0 - recent_ratio) / count, dtype=np.float64)
        weights[-recent_count:] += recent_ratio / recent_count
        if balance_outcomes:
            success = np.asarray([bool(row["success"]) for row in self.row_metadata])
            failure = np.asarray([bool(row["failure"]) for row in self.row_metadata])
            for mask in (success, failure):
                if mask.any():
                    weights[mask] *= count / (2.0 * mask.sum())
        intervention = np.asarray([bool(row["intervention"]) for row in self.row_metadata])
        weights[intervention] *= max(float(intervention_weight), 0.0)
        weights /= weights.sum()
        return torch.from_numpy(weights)
