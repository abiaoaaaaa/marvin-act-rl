import unittest

import numpy as np

from lerobot_robot_marvin.protocol import dumps, loads
from lerobot_robot_marvin.schema import validate_observation


class ProtocolTest(unittest.TestCase):
    def test_ndarray_round_trip(self):
        source = {"x": np.arange(12, dtype=np.float32).reshape(3, 4)}
        decoded = loads(dumps(source))
        np.testing.assert_array_equal(decoded["x"], source["x"])
        self.assertEqual(decoded["x"].dtype, np.float32)

    def test_horizon_is_removed(self):
        obs = {"state": np.zeros((1, 24), np.float32)}
        for key in ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2"):
            obs[key] = np.zeros((1, 128, 128, 3), np.uint8)
        flat = validate_observation(obs)
        self.assertEqual(flat["state"].shape, (24,))
        self.assertEqual(flat["side_policy"].shape, (128, 128, 3))


if __name__ == "__main__":
    unittest.main()

