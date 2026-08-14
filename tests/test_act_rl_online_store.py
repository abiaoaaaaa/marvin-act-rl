import tempfile
import unittest

import numpy as np

from act_rl.online_store import OnlineEpisodeWriter, OnlineNpzDataset


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
            dataset = OnlineNpzDataset(root, keys)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(dataset[0]["cam"].shape), (3, 8, 8))


if __name__ == "__main__":
    unittest.main()
