from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OnlineEpisodeWriter:
    """Append-only one-file-per-episode storage; a crash cannot corrupt old episodes."""

    def __init__(self, root: str | Path, image_keys: tuple[str, ...], chunk_size: int, action_dim: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_keys, self.chunk_size, self.action_dim = image_keys, chunk_size, action_dim

    def write(self, episode_id: int, records: list[dict], metadata: dict | None = None) -> Path:
        if not records:
            raise ValueError("refusing to write an empty online episode")
        arrays = {
            "state": np.stack([x["observation"]["state"] for x in records]).astype(np.float32),
            "next_state": np.stack([x["next_observation"]["state"] for x in records]).astype(np.float32),
            "action": np.stack([x["action_chunk"] for x in records]).astype(np.float32),
            "reward": np.asarray([x["reward"] for x in records], np.float32),
            "discount": np.asarray([x["discount"] for x in records], np.float32),
            "terminated": np.asarray([x["terminated"] for x in records], np.float32),
            "action_is_pad": np.stack([x["action_is_pad"] for x in records]).astype(bool),
        }
        if arrays["action"].shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError(f"bad action chunk shape {arrays['action'].shape}")
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
        sidecar.write_text(json.dumps(metadata or {}, indent=2, ensure_ascii=False))
        return path

    def next_episode_id(self) -> int:
        ids = [int(path.stem.split("_")[-1]) for path in self.root.glob("episode_*.npz")]
        return max(ids, default=-1) + 1


class OnlineNpzDataset(Dataset):
    def __init__(self, root: str | Path, image_keys: tuple[str, ...]):
        self.image_keys = image_keys
        self.files = sorted(Path(root).glob("episode_*.npz"))
        self.index: list[tuple[int, int]] = []
        self._cache_id, self._cache = None, None
        for file_id, path in enumerate(self.files):
            with np.load(path, allow_pickle=False) as data:
                self.index.extend((file_id, row) for row in range(len(data["reward"])))

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
        result["source"] = "online"
        for key in self.image_keys:
            result[key] = torch.from_numpy(np.asarray(data[key][row])).permute(2, 0, 1).float() / 255.0
            result[f"next.{key}"] = torch.from_numpy(np.asarray(data[f"next.{key}"][row])).permute(2, 0, 1).float() / 255.0
        return result
