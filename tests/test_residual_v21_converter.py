import tempfile
import unittest
from pathlib import Path

from convert_marvin_v3_to_residual_v21 import (
    CAMERA_KEYS,
    destination_features,
    resolve_mode,
    scalar,
    video_path_for_data,
)


class ResidualV21ConverterTest(unittest.TestCase):
    def setUp(self):
        self.info = {
            "features": {
                "observation.state": {"shape": [24], "names": None},
                "action": {"shape": [5], "names": None},
                **{key: {"shape": [128, 128, 3]} for key in CAMERA_KEYS},
            },
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }

    def test_auto_mode_and_rl_terminal_feature(self):
        columns = {
            "rl.reward",
            "rl.terminated",
            "rl.success",
            "rl.intervention",
            "rl.safety_abort",
        }
        self.assertEqual(resolve_mode("auto", columns), "rl")
        features = destination_features(self.info, "rl", "image")
        self.assertEqual(features["next.done"]["dtype"], "bool")
        self.assertEqual(features["next.done"]["shape"], (1,))
        self.assertTrue(all(features[key]["dtype"] == "image" for key in CAMERA_KEYS))
        self.assertTrue(all(features[key]["names"] == ["channels", "height", "width"] for key in CAMERA_KEYS))

    def test_bc_mode_does_not_leak_rl_fields(self):
        features = destination_features(self.info, "bc", "video")
        self.assertNotIn("next.done", features)
        self.assertNotIn("rl.reward", features)
        self.assertTrue(all(features[key]["dtype"] == "video" for key in CAMERA_KEYS))

    def test_video_path_uses_matching_chunk_and_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data/chunk-007/file-012.parquet"
            expected = root / "videos/observation.images.side_policy/chunk-007/file-012.mp4"
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(
                video_path_for_data(self.info, root, "observation.images.side_policy", data_path), expected
            )

    def test_scalar_preserves_one_element_shape(self):
        value = scalar(True, "bool")
        self.assertEqual(value.shape, (1,))
        self.assertTrue(bool(value[0]))


if __name__ == "__main__":
    unittest.main()
