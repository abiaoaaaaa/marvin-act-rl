import tempfile
import unittest
import json
import io
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from act_rl.online_store import OnlineEpisodeWriter, OnlineNpzDataset, REPLAY_SCHEMA_VERSION
from act_rl.scripts.collect_online import _validate_fixed_policy_replay
from act_rl.scripts.audit_online_replay import main as audit_online_replay


class OnlineStoreTest(unittest.TestCase):
    def test_roundtrip(self):
        keys = ("cam",)
        obs = {"state": np.zeros(4, np.float32), "cam": np.zeros((8, 8, 3), np.uint8)}
        nxt = {"state": np.ones(4, np.float32), "cam": np.ones((8, 8, 3), np.uint8)}
        record = {"observation": obs, "next_observation": nxt,
                  "action_chunk": np.zeros((3, 2), np.float32), "action_is_pad": np.zeros(3, bool),
                  "reward": 1.0, "discount": 0.9, "terminated": True}
        with tempfile.TemporaryDirectory() as root:
            path = OnlineEpisodeWriter(root, keys, 3, 2).write(0, [record])
            self.assertTrue(path.exists())
            self.assertEqual(OnlineEpisodeWriter(root, keys, 3, 2).next_episode_id(), 1)
            with self.assertRaises(FileExistsError):
                OnlineEpisodeWriter(root, keys, 3, 2).write(0, [record])
            dataset = OnlineNpzDataset(root, keys, require_schema_version=REPLAY_SCHEMA_VERSION)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(dataset[0]["cam"].shape), (3, 8, 8))
            self.assertEqual(tuple(dataset[0]["policy_action"].shape), (3, 2))
            self.assertEqual(tuple(dataset[0]["critic_action"].shape), (3, 2))
            with np.load(path, allow_pickle=False) as stored:
                self.assertIn("command_action", stored)
            weights = dataset.sampling_weights(recent_ratio=0.5)
            self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_rejects_legacy_replay_when_v2_is_required(self):
        with tempfile.TemporaryDirectory() as root:
            np.savez_compressed(
                f"{root}/episode_000000.npz",
                state=np.zeros((1, 4), np.float32), next_state=np.zeros((1, 4), np.float32),
                action=np.zeros((1, 3, 2), np.float32), reward=np.zeros(1, np.float32),
                discount=np.ones(1, np.float32), terminated=np.ones(1, np.float32),
                action_is_pad=np.zeros((1, 3), bool),
            )
            with self.assertRaisesRegex(ValueError, "schema v1"):
                OnlineNpzDataset(root, (), require_schema_version=REPLAY_SCHEMA_VERSION)

    def test_collection_round_rejects_mixed_policy_versions(self):
        with tempfile.TemporaryDirectory() as root:
            sidecar = Path(root) / "episode_000000.json"
            sidecar.write_text(json.dumps({"policy_sha256": "actor-a"}))
            _validate_fixed_policy_replay(root, "actor-a")
            with self.assertRaisesRegex(ValueError, "different or unknown policy"):
                _validate_fixed_policy_replay(root, "actor-b")

    def test_strict_replay_audit_accepts_valid_v2_episode(self):
        image_keys = (
            "observation.images.side_policy", "observation.images.wrist_1buchong",
            "observation.images.wrist_1", "observation.images.wrist_2",
        )
        obs = {"state": np.zeros(24, np.float32)}
        nxt = {"state": np.ones(24, np.float32)}
        for key in image_keys:
            short = key.removeprefix("observation.images.")
            obs[short] = np.zeros((8, 8, 3), np.uint8)
            nxt[short] = np.ones((8, 8, 3), np.uint8)
        action = np.zeros((30, 5), np.float32)
        pad = np.arange(30) >= 10
        record = {
            "observation": obs, "next_observation": nxt, "action_chunk": action,
            "action_is_pad": pad, "reward": 1.0, "discount": 0.99**10,
            "terminated": True, "success": True, "executed_steps": 10,
            "critic_action_is_pad": np.arange(30) >= 10,
            "executed_mask": np.arange(30) < 10,
        }
        with tempfile.TemporaryDirectory() as root:
            OnlineEpisodeWriter(root, image_keys, 30, 5).write(0, [record], {
                "policy_sha256": "known-actor", "success": True, "failure": False,
            })
            argv = ["audit_online_replay", "--replay", root, "--config",
                    "configs/act_rl/marvin_online_round1.json"]
            output = io.StringIO()
            with patch("sys.argv", argv), redirect_stdout(output):
                audit_online_replay()
            self.assertTrue(json.loads(output.getvalue())["ok"])


if __name__ == "__main__":
    unittest.main()
