import unittest

import numpy as np

from lerobot_robot_marvin.zadai_rl100_policy import ZadaiObservationHistory, sanitize_zadai_chunk


class ZadaiPolicyAdapterTest(unittest.TestCase):
    def test_history_and_chunk_contract(self):
        history = ZadaiObservationHistory(2, {key: (2, 3, 3) for key in ("top", "wrist_L", "wrist_R")})
        item = {"state": np.zeros(16, np.float32), **{key: np.zeros((2, 3, 3), np.uint8) for key in ("top", "wrist_L", "wrist_R")}}
        history.reset(item)
        result = history.as_numpy()
        self.assertEqual(result["agent_pos"].shape, (1, 2, 16))
        self.assertEqual(result["rgb_top"].shape, (1, 2, 3, 2, 3))
        self.assertEqual(sanitize_zadai_chunk(np.zeros((1, 16, 16)), 16).shape, (16, 16))


if __name__ == "__main__":
    unittest.main()
