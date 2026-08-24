#!/usr/bin/env python3
"""Convert a local Marvin LeRobot v3 dataset to ResFiT's LeRobot v2.1 layout.

The converter is intentionally executed with the ``residual`` Conda Python.  It
reads the v3 parquet/video files directly (without importing a modern LeRobot)
and writes the destination through ResFiT's pinned LeRobot API.  ``bc`` mode
keeps only policy inputs/actions.  ``rl`` mode additionally preserves the
failure-aware labels and adds ``next.done`` derived from episode boundaries,
which is the terminal key consumed by ResFiT's offline replay builder.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import re
import shutil
from contextlib import ExitStack
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


CAMERA_KEYS = (
    "observation.images.side_policy",
    "observation.images.wrist_1buchong",
    "observation.images.wrist_1",
    "observation.images.wrist_2",
)
RL_SOURCE_KEYS = (
    "rl.reward",
    "rl.terminated",
    "rl.success",
    "rl.intervention",
    "rl.safety_abort",
)
DATA_FILE_RE = re.compile(r"chunk-(?P<chunk>\d+)/file-(?P<file>\d+)\.parquet$")


class VideoFrameStream:
    """Sequential RGB decoder that owns and closes its PyAV container."""

    def __init__(self, path: Path):
        self.path = path
        self.container = av.open(str(path))
        self.frames = self.container.decode(video=0)
        self.count = 0

    def next_rgb(self) -> np.ndarray:
        try:
            frame = next(self.frames)
        except StopIteration as exc:
            raise RuntimeError(f"Video ended early after {self.count} frames: {self.path}") from exc
        self.count += 1
        return frame.to_ndarray(format="rgb24")

    def assert_exhausted(self) -> None:
        try:
            next(self.frames)
        except StopIteration:
            return
        raise RuntimeError(f"Video contains more frames than its paired parquet file: {self.path}")

    def close(self) -> None:
        self.container.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path, help="Local LeRobot v3 dataset root")
    parser.add_argument("--output-root", required=True, type=Path, help="New LeRobot v2.1 dataset root")
    parser.add_argument("--repo-id", required=True, help="Local dataset id stored in v2.1 metadata")
    parser.add_argument(
        "--mode",
        choices=("auto", "bc", "rl"),
        default="auto",
        help="auto selects rl when rl.reward exists, otherwise bc",
    )
    parser.add_argument(
        "--media-storage",
        choices=("image", "video"),
        default="image",
        help="image avoids TorchCodec/torchvision video ABI issues in the residual environment",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert only the first N episodes")
    parser.add_argument(
        "--exclude-episodes",
        type=int,
        nargs="*",
        default=[],
        help="Source episode indices to omit; destination indices remain contiguous",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output root first")
    return parser


def load_parquet_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def source_metadata(root: Path) -> tuple[dict, dict[int, str], dict[int, int]]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected a LeRobot v3.0 source, got {info.get('codebase_version')!r}")

    task_paths = sorted((root / "meta").glob("tasks*.parquet"))
    if not task_paths:
        raise FileNotFoundError(root / "meta" / "tasks.parquet")
    tasks = {int(row["task_index"]): str(row["task"]) for row in load_parquet_rows(task_paths)}

    episode_paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(root / "meta" / "episodes")
    lengths = {int(row["episode_index"]): int(row["length"]) for row in load_parquet_rows(episode_paths)}
    return info, tasks, lengths


def resolve_mode(requested: str, columns: set[str]) -> str:
    mode = "rl" if requested == "auto" and "rl.reward" in columns else requested
    if mode == "auto":
        mode = "bc"
    if mode == "rl":
        missing = set(RL_SOURCE_KEYS) - columns
        if missing:
            raise KeyError(f"RL conversion requires source columns: {sorted(missing)}")
    return mode


def destination_features(info: dict, mode: str, media_storage: str) -> dict:
    source_features = info["features"]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": tuple(source_features["observation.state"]["shape"]),
            "names": source_features["observation.state"].get("names"),
        },
        "action": {
            "dtype": "float32",
            "shape": tuple(source_features["action"]["shape"]),
            "names": source_features["action"].get("names"),
        },
    }
    for key in CAMERA_KEYS:
        shape = tuple(source_features[key]["shape"])
        if len(shape) != 3:
            raise ValueError(f"Expected HWC camera shape for {key}, got {shape}")
        height, width, channels = shape
        features[key] = {
            "dtype": media_storage,
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        }
    if mode == "rl":
        features.update(
            {
                "rl.reward": {"dtype": "float32", "shape": (1,), "names": None},
                "rl.terminated": {"dtype": "bool", "shape": (1,), "names": None},
                "rl.success": {"dtype": "bool", "shape": (1,), "names": None},
                "rl.intervention": {"dtype": "bool", "shape": (1,), "names": None},
                "rl.safety_abort": {"dtype": "bool", "shape": (1,), "names": None},
                "next.done": {"dtype": "bool", "shape": (1,), "names": None},
            }
        )
    return features


def video_path_for_data(info: dict, root: Path, key: str, data_path: Path) -> Path:
    relative = data_path.relative_to(root).as_posix()
    match = DATA_FILE_RE.search(relative)
    if match is None:
        raise ValueError(f"Cannot extract v3 chunk/file indices from {relative}")
    formatted = info["video_path"].format(
        video_key=key,
        chunk_index=int(match.group("chunk")),
        file_index=int(match.group("file")),
    )
    path = root / formatted
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def image_from_parquet(value: object, key: str) -> np.ndarray:
    """Decode a LeRobot v3 inline image struct into an HWC RGB array."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected inline image struct for {key}, got {type(value).__name__}")
    payload = value.get("bytes")
    if payload is None:
        path = value.get("path")
        raise FileNotFoundError(f"Inline image bytes missing for {key}; path={path!r}")
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def scalar(value, dtype) -> np.ndarray:
    return np.asarray([value], dtype=dtype)


def convert(args: argparse.Namespace) -> dict:
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if input_root == output_root:
        raise ValueError("Input and output roots must differ")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)

    info, tasks, episode_lengths = source_metadata(input_root)
    data_paths = sorted((input_root / "data").rglob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError(input_root / "data")
    first_columns = set(pq.read_schema(data_paths[0]).names)
    mode = resolve_mode(args.mode, first_columns)

    # Imported only here so --help and metadata-only failures remain useful
    # even when the pinned ResFiT LeRobot environment is not active.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(info["fps"]),
        features=destination_features(info, mode, args.media_storage),
        root=output_root,
        robot_type=str(info.get("robot_type") or "marvin"),
        use_videos=args.media_storage == "video",
        video_backend="pyav",
    )

    converted_episodes = 0
    converted_frames = 0
    current_source_episode: int | None = None
    source_episode_order: dict[int, int] = {}
    excluded_episodes = set(args.exclude_episodes)
    stopped_early = False

    for data_path in data_paths:
        table = pq.read_table(data_path)
        columns = set(table.column_names)
        file_mode = resolve_mode(args.mode, columns)
        if file_mode != mode:
            raise ValueError(f"Mixed BC/RL parquet schemas are not supported: {data_path}")

        with ExitStack() as stack:
            streams = {}
            inline_images = info.get("video_path") is None
            if not inline_images:
                for key in CAMERA_KEYS:
                    stream = VideoFrameStream(video_path_for_data(info, input_root, key, data_path))
                    streams[key] = stream
                    stack.callback(stream.close)

            rows_seen = 0
            for batch in table.to_batches(max_chunksize=512):
                for row in batch.to_pylist():
                    rows_seen += 1
                    source_episode = int(row["episode_index"])
                    if source_episode in excluded_episodes:
                        # Source videos are file-level streams.  Consume the
                        # matching frames even when this row is omitted so all
                        # following observations remain aligned.
                        if not inline_images:
                            for stream in streams.values():
                                stream.next_rgb()
                        continue
                    if source_episode not in source_episode_order:
                        source_episode_order[source_episode] = len(source_episode_order)
                    if args.max_episodes is not None and source_episode_order[source_episode] >= args.max_episodes:
                        stopped_early = True
                        break

                    if current_source_episode is not None and source_episode != current_source_episode:
                        dataset.save_episode()
                        converted_episodes += 1
                    current_source_episode = source_episode

                    task_index = int(row["task_index"])
                    if task_index not in tasks:
                        raise KeyError(f"Unknown task_index={task_index} in {data_path}")
                    frame_index = int(row["frame_index"])
                    expected_length = episode_lengths[source_episode]
                    if frame_index < 0 or frame_index >= expected_length:
                        raise ValueError(
                            f"Invalid frame_index={frame_index} for episode {source_episode} length={expected_length}"
                        )

                    frame = {
                        "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                        "action": np.asarray(row["action"], dtype=np.float32),
                    }
                    if inline_images:
                        for key in CAMERA_KEYS:
                            frame[key] = image_from_parquet(row[key], key)
                    else:
                        for key, stream in streams.items():
                            frame[key] = stream.next_rgb()
                    if mode == "rl":
                        frame.update(
                            {
                                "rl.reward": scalar(row["rl.reward"], np.float32),
                                "rl.terminated": scalar(row["rl.terminated"], np.bool_),
                                "rl.success": scalar(row["rl.success"], np.bool_),
                                "rl.intervention": scalar(row["rl.intervention"], np.bool_),
                                "rl.safety_abort": scalar(row["rl.safety_abort"], np.bool_),
                                "next.done": scalar(frame_index == expected_length - 1, np.bool_),
                            }
                        )
                    dataset.add_frame(frame, task=tasks[task_index], timestamp=frame_index / int(info["fps"]))
                    converted_frames += 1
                    current_source_episode = source_episode
                if stopped_early:
                    break

            if not stopped_early:
                if rows_seen != table.num_rows:
                    raise RuntimeError(f"Read {rows_seen} rows from {data_path}, expected {table.num_rows}")
                if not inline_images:
                    for stream in streams.values():
                        stream.assert_exhausted()
        if stopped_early:
            break

    if dataset.episode_buffer is not None and dataset.episode_buffer["size"]:
        dataset.save_episode()
        converted_episodes += 1

    report = {
        "schema_version": 1,
        "source_root": str(input_root),
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "mode": mode,
        "source_codebase_version": info["codebase_version"],
        "destination_codebase_version": "v2.1",
        "fps": int(info["fps"]),
        "episodes": converted_episodes,
        "frames": converted_frames,
        "camera_keys": list(CAMERA_KEYS),
        "media_storage": args.media_storage,
        "excluded_source_episodes": sorted(excluded_episodes),
        "rl_terminal_contract": "next.done is true on the final frame of every episode" if mode == "rl" else None,
    }
    (output_root / "residual_v21_conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = convert(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
