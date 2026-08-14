"""Marvin adapters for RL-100's failure-aware offline training stage."""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


CAMERA_KEYS = (
    "rgb_side_policy",
    "rgb_wrist_1",
    "rgb_wrist_1buchong",
    "rgb_wrist_2",
)


def _rl100_imports():
    from rl_100.common.pytorch_util import dict_apply
    from rl_100.common.replay_buffer import ReplayBuffer
    from rl_100.common.sampler import SequenceSampler, downsample_mask, get_val_mask
    from rl_100.dataset.base_dataset import BaseDataset
    from rl_100.env_runner.base_runner import BaseRunner
    from rl_100.model.common.normalizer import LinearNormalizer

    return dict_apply, ReplayBuffer, SequenceSampler, downsample_mask, get_val_mask, BaseDataset, BaseRunner, LinearNormalizer


try:
    (
        dict_apply,
        ReplayBuffer,
        SequenceSampler,
        downsample_mask,
        get_val_mask,
        BaseDataset,
        BaseRunner,
        LinearNormalizer,
    ) = _rl100_imports()
except ImportError:
    # Keep the bridge package importable in the lightweight HIL-SERL process.
    BaseDataset = object
    BaseRunner = object


class MarvinOfflineDataset(BaseDataset):
    """RL-100 sequence dataset with a deployment-aligned K-step transition.

    The diffusion trajectory remains ``horizon`` steps long. Q/V and dynamics
    receive ``next_obs`` at exactly ``executed_steps`` after the decision-time
    observation, matching ``--actions-per-inference`` on the real robot.
    """

    def __init__(
        self,
        zarr_path,
        horizon=17,
        n_obs_steps=2,
        executed_steps=5,
        pad_before=1,
        pad_after=15,
        seed=1000,
        val_ratio=0.1,
        max_train_episodes=None,
        sequence_stride=1,
        normalizer_zarr_path=None,
    ):
        super().__init__()
        if executed_steps <= 0:
            raise ValueError("executed_steps must be positive")
        if executed_steps > horizon - n_obs_steps + 1:
            raise ValueError(
                f"executed_steps={executed_steps} does not fit horizon={horizon}, n_obs_steps={n_obs_steps}"
            )
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")
        required = {"state", "action", "reward", "done", "return", *CAMERA_KEYS}
        missing = required.difference(self.replay_buffer.keys())
        if missing:
            raise KeyError(f"Marvin offline Zarr is missing keys: {sorted(missing)}")

        val_mask = get_val_mask(self.replay_buffer.n_episodes, val_ratio, seed)
        train_mask = downsample_mask(~val_mask, max_train_episodes, seed)
        self._sampler_keys = ["state", "action", "reward", "done", "return", *CAMERA_KEYS]
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=self._sampler_keys,
            episode_mask=train_mask,
            sequence_stride=sequence_stride,
        )
        self.train_mask = train_mask
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.executed_steps = int(executed_steps)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.sequence_stride = int(sequence_stride)
        self.normalizer_zarr_path = normalizer_zarr_path

    def get_validation_dataset(self):
        dataset = copy.copy(self)
        dataset.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self._sampler_keys,
            episode_mask=~self.train_mask,
            sequence_stride=self.sequence_stride,
        )
        dataset.train_mask = ~self.train_mask
        return dataset

    def get_normalizer(self, mode="limits", **kwargs):
        replay = self.replay_buffer
        if self.normalizer_zarr_path:
            replay = ReplayBuffer.create_from_path(self.normalizer_zarr_path, mode="r")
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": replay["action"], "agent_pos": replay["state"]},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer

    def __len__(self):
        return len(self.sampler)

    @staticmethod
    def _images(sample: dict, indices) -> dict:
        result = {}
        for key in CAMERA_KEYS:
            result[key] = np.ascontiguousarray(
                sample[key][indices].transpose(0, 3, 1, 2), dtype=np.uint8
            )
        return result

    def _sample_to_data(self, sample: dict) -> dict:
        action_start = self.n_obs_steps - 1
        action_end = action_start + self.executed_steps
        next_start = action_end - 1
        next_end = next_start + self.n_obs_steps
        if next_end > self.horizon:
            raise RuntimeError("next observation window exceeds sampled horizon")

        rewards = sample["reward"].astype(np.float32, copy=True)
        dones = sample["done"].astype(np.bool_, copy=False)
        # SequenceSampler repeats terminal values for right padding. A terminal
        # reward is valid once; repeated padded copies must contribute zero.
        terminal_indices = np.flatnonzero(dones.reshape(self.horizon, -1).any(axis=1))
        if len(terminal_indices):
            rewards[terminal_indices[0] + 1 :] = 0.0

        obs = {
            "agent_pos": sample["state"][: self.n_obs_steps].astype(np.float32),
            **self._images(sample, slice(0, self.n_obs_steps)),
        }
        next_obs = {
            "agent_pos": sample["state"][next_start:next_end].astype(np.float32),
            **self._images(sample, slice(next_start, next_end)),
        }
        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": sample["action"].astype(np.float32),
            "reward": rewards,
            "not_done": 1.0 - dones.astype(np.float32),
            "return": sample["return"].astype(np.float32),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return dict_apply(self._sample_to_data(self.sampler.sample_sequence(index)), torch.from_numpy)

    def get_shape_info(self, n_action_steps, n_obs_steps):
        if int(n_obs_steps) != self.n_obs_steps:
            raise ValueError(
                f"Config n_obs_steps={n_obs_steps} differs from dataset n_obs_steps={self.n_obs_steps}"
            )
        return {
            "obs": {
                "agent_pos": (self.n_obs_steps, self.replay_buffer["state"].shape[-1]),
                **{
                    key: (
                        self.n_obs_steps,
                        self.replay_buffer[key].shape[-1],
                        self.replay_buffer[key].shape[1],
                        self.replay_buffer[key].shape[2],
                    )
                    for key in CAMERA_KEYS
                },
            },
            "action": (self.executed_steps, self.replay_buffer["action"].shape[-1]),
        }


class MarvinOfflineEvalRunner(BaseRunner):
    """Offline safety metric used when no simulator exists for Marvin."""

    def __init__(
        self,
        output_dir,
        zarr_path,
        normalizer_zarr_path,
        horizon=17,
        n_obs_steps=2,
        executed_steps=5,
        seed=1000,
        val_ratio=0.1,
        batch_size=4,
        max_batches=4,
        env_num=1,
        eval_episodes=1,
    ):
        super().__init__(output_dir)
        self.env = None
        dataset = MarvinOfflineDataset(
            zarr_path=zarr_path,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            executed_steps=executed_steps,
            pad_before=n_obs_steps - 1,
            pad_after=horizon - n_obs_steps,
            seed=seed,
            val_ratio=val_ratio,
            normalizer_zarr_path=normalizer_zarr_path,
        ).get_validation_dataset()
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.max_batches = int(max_batches)
        self.env_num = int(env_num)
        self.eval_episodes = int(eval_episodes)
        self.action_start = int(n_obs_steps) - 1
        self.executed_steps = int(executed_steps)

    def run(self, policy, **_kwargs) -> Dict[str, float]:
        device = next(policy.parameters()).device
        squared_error = 0.0
        element_count = 0
        policy.eval()
        with torch.inference_mode():
            for batch_index, batch in enumerate(self.loader):
                if batch_index >= self.max_batches:
                    break
                obs = {key: value.to(device) for key, value in batch["obs"].items()}
                predicted = policy.predict_action(obs, deterministic=True)["action"]
                target = batch["action"][:, self.action_start : self.action_start + self.executed_steps].to(device)
                error = (predicted[:, : self.executed_steps] - target).square()
                squared_error += float(error.sum().cpu())
                element_count += error.numel()
        if element_count == 0:
            raise RuntimeError("Offline validation dataset is empty")
        mse = squared_error / element_count
        return {"test_mean_score": -mse, "mean_returns": -mse, "offline_action_mse": mse}
