#!/usr/bin/env python3
"""Convert the Marvin LeRobot v3 dataset to an RL-100 Zarr replay buffer.

The converter streams every camera video in presentation order and writes
uint8 RGB frames to Zarr. It does not perform image or action normalization;
RL-100 fits its numeric normalizer and normalizes images in the encoder.

Required packages: av, numpy, numcodecs, pyarrow and zarr<3.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


DEFAULT_CAMERAS = ("side_policy", "wrist_1", "wrist_1buchong", "wrist_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-zarr", type=Path, required=True)
    parser.add_argument("--camera-keys", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--exclude-episodes", type=int, nargs="*", default=[])
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N selected episodes for a smoke test.",
    )
    parser.add_argument("--image-chunk-length", type=int, default=16)
    parser.add_argument("--numeric-chunk-length", type=int, default=2048)
    parser.add_argument(
        "--include-rl",
        action="store_true",
        help="Preserve rl.reward/terminated/success/intervention/safety_abort for offline RL.",
    )
    parser.add_argument("--return-gamma", type=float, default=0.99)
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if args.image_chunk_length <= 0 or args.numeric_chunk_length <= 0:
        parser.error("chunk lengths must be positive")
    if not 0.0 < args.return_gamma <= 1.0:
        parser.error("--return-gamma must be in (0, 1]")
    return args


def fixed_list_to_numpy(column, width: int) -> np.ndarray:
    array = column.combine_chunks()
    if array.type.list_size != width:
        raise ValueError(f"Expected fixed list width {width}, got {array.type}")
    return (
        array.values.to_numpy(zero_copy_only=False)
        .reshape(len(array), width)
        .astype(np.float32, copy=False)
    )


def _scalar_column_to_numpy(column, dtype) -> np.ndarray:
    array = column.combine_chunks()
    if hasattr(array.type, "list_size"):
        if array.type.list_size != 1:
            raise ValueError(f"Expected scalar or width-1 list, got {array.type}")
        values = array.values.to_numpy(zero_copy_only=False)
    else:
        values = array.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype).reshape(-1, 1)


def load_tabular_data(source_root: Path, include_rl: bool = False) -> dict[str, np.ndarray]:
    parquet_files = sorted((source_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {source_root / 'data'}")
    columns = ["observation.state", "action", "episode_index", "frame_index", "index"]
    if include_rl:
        columns.extend(
            ["rl.reward", "rl.terminated", "rl.success", "rl.intervention", "rl.safety_abort"]
        )
    table = pq.read_table(parquet_files, columns=columns)
    state_width = table.schema.field("observation.state").type.list_size
    action_width = table.schema.field("action").type.list_size
    result = {
        "state": fixed_list_to_numpy(table["observation.state"], state_width),
        "action": fixed_list_to_numpy(table["action"], action_width),
        "episode_index": np.asarray(table["episode_index"].combine_chunks(), dtype=np.int64),
        "frame_index": np.asarray(table["frame_index"].combine_chunks(), dtype=np.int64),
        "index": np.asarray(table["index"].combine_chunks(), dtype=np.int64),
    }
    if include_rl:
        result.update(
            {
                "reward": _scalar_column_to_numpy(table["rl.reward"], np.float32),
                "done": _scalar_column_to_numpy(table["rl.terminated"], np.bool_),
                "success": _scalar_column_to_numpy(table["rl.success"], np.bool_),
                "intervention": _scalar_column_to_numpy(table["rl.intervention"], np.bool_),
                "safety_abort": _scalar_column_to_numpy(table["rl.safety_abort"], np.bool_),
            }
        )
    frame_count = len(result["index"])
    if not all(len(value) == frame_count for value in result.values()):
        raise ValueError("Parquet columns have inconsistent lengths")
    if not np.array_equal(result["index"], np.arange(frame_count)):
        raise ValueError("Dataset index must be contiguous and match video frame order")
    if not np.isfinite(result["state"]).all() or not np.isfinite(result["action"]).all():
        raise ValueError("State/action contains NaN or infinity")
    return result


def compute_returns(reward: np.ndarray, done: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(reward, dtype=np.float32)
    running = 0.0
    for index in range(len(reward) - 1, -1, -1):
        running = float(reward[index, 0]) + gamma * running * (1.0 - float(done[index, 0]))
        returns[index, 0] = running
    return returns


def select_episodes(tabular: dict[str, np.ndarray], args: argparse.Namespace):
    episode_ids = tabular["episode_index"]
    available = np.unique(episode_ids)
    excluded = set(args.exclude_episodes)
    selected_ids = [int(ep) for ep in available if int(ep) not in excluded]
    if args.max_episodes is not None:
        selected_ids = selected_ids[: args.max_episodes]
    if not selected_ids:
        raise ValueError("No episodes selected")

    mask = np.isin(episode_ids, selected_ids)
    selected_source_indices = np.flatnonzero(mask)
    counts = np.array([np.count_nonzero(episode_ids == ep) for ep in selected_ids], dtype=np.int64)
    if np.any(counts <= 0):
        raise ValueError("Selected an empty episode")

    # Validate original episode-local ordering before dropping episode ids.
    for ep in selected_ids:
        local_frames = tabular["frame_index"][episode_ids == ep]
        if not np.array_equal(local_frames, np.arange(len(local_frames))):
            raise ValueError(f"Episode {ep} has non-contiguous frame_index")
    return selected_ids, selected_source_indices, np.cumsum(counts)


def camera_video_files(source_root: Path, camera: str) -> list[Path]:
    files = sorted((source_root / "videos" / f"observation.images.{camera}").glob("*/*.mp4"))
    if not files:
        raise FileNotFoundError(f"No video files found for camera {camera!r}")
    return files


def write_camera(
    destination,
    video_files: list[Path],
    selected_source_indices: np.ndarray,
) -> tuple[int, int]:
    selected_set = set(int(i) for i in selected_source_indices)
    last_needed = int(selected_source_indices[-1])
    source_index = 0
    output_index = 0
    pending: list[np.ndarray] = []
    pending_start = 0

    def flush() -> None:
        nonlocal pending, pending_start
        if pending:
            batch = np.stack(pending)
            destination[pending_start : pending_start + len(batch)] = batch
            pending = []

    for video_path in video_files:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                if source_index > last_needed:
                    break
                if source_index in selected_set:
                    image = frame.to_ndarray(format="rgb24")
                    if image.shape != destination.shape[1:]:
                        raise ValueError(
                            f"Unexpected frame shape {image.shape} in {video_path}; "
                            f"expected {destination.shape[1:]}"
                        )
                    if not pending:
                        pending_start = output_index
                    pending.append(image)
                    output_index += 1
                    if len(pending) >= destination.chunks[0]:
                        flush()
                source_index += 1
            if source_index > last_needed:
                break
    flush()
    if output_index != len(selected_source_indices):
        raise ValueError(f"Decoded {output_index} selected frames, expected {len(selected_source_indices)}")
    return source_index, output_index


def main() -> None:
    args = parse_args()
    try:
        import numcodecs
        import zarr
    except ImportError as exc:
        raise SystemExit("Install zarr<3 and numcodecs before running this converter") from exc

    source_root = args.source_root.resolve()
    output_zarr = args.output_zarr.resolve()
    if output_zarr.exists():
        raise FileExistsError(f"Output already exists: {output_zarr}")
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zarr.with_name(f".{output_zarr.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary path already exists: {temporary}")

    tabular = load_tabular_data(source_root, include_rl=args.include_rl)
    selected_ids, selected_source_indices, episode_ends = select_episodes(tabular, args)
    selected_frames = len(selected_source_indices)

    info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info["features"]
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
    )

    try:
        root = zarr.open_group(str(temporary), mode="w")
        data = root.create_group("data")
        meta = root.create_group("meta")
        numeric_chunk = min(args.numeric_chunk_length, selected_frames)
        data.create_dataset(
            "state",
            data=tabular["state"][selected_source_indices],
            chunks=(numeric_chunk, tabular["state"].shape[1]),
            compressor=compressor,
        )
        data.create_dataset(
            "action",
            data=tabular["action"][selected_source_indices],
            chunks=(numeric_chunk, tabular["action"].shape[1]),
            compressor=compressor,
        )
        if args.include_rl:
            selected_done = tabular["done"][selected_source_indices]
            selected_episode_ids = tabular["episode_index"][selected_source_indices]
            for episode_id in selected_ids:
                episode_rows = np.flatnonzero(selected_episode_ids == episode_id)
                if not np.array_equal(
                    selected_done[episode_rows, 0],
                    np.arange(len(episode_rows)) == len(episode_rows) - 1,
                ):
                    raise ValueError(
                        f"Episode {episode_id} must have rl.terminated=True only on its final frame"
                    )
            rl_arrays = {
                "reward": tabular["reward"][selected_source_indices],
                "done": selected_done,
                "timeout": np.zeros_like(selected_done, dtype=np.bool_),
                "success": tabular["success"][selected_source_indices],
                "intervention": tabular["intervention"][selected_source_indices],
                "safety_abort": tabular["safety_abort"][selected_source_indices],
            }
            rl_arrays["return"] = compute_returns(
                rl_arrays["reward"], rl_arrays["done"], args.return_gamma
            )
            for key, values in rl_arrays.items():
                data.create_dataset(
                    key,
                    data=values,
                    chunks=(numeric_chunk, 1),
                    compressor=compressor,
                )
        meta.create_dataset("episode_ends", data=episode_ends, compressor=None)
        meta.create_dataset("source_episode_index", data=np.asarray(selected_ids), compressor=None)

        camera_reports = {}
        for camera in args.camera_keys:
            feature_key = f"observation.images.{camera}"
            if feature_key not in features:
                raise KeyError(f"Camera feature not present in dataset: {feature_key}")
            height, width, channels = features[feature_key]["shape"]
            if channels != 3:
                raise ValueError(f"Only RGB cameras are supported, got {features[feature_key]['shape']}")
            image_chunk = min(args.image_chunk_length, selected_frames)
            array = data.create_dataset(
                f"rgb_{camera}",
                shape=(selected_frames, height, width, channels),
                chunks=(image_chunk, height, width, channels),
                dtype=np.uint8,
                compressor=compressor,
            )
            decoded, written = write_camera(
                array,
                camera_video_files(source_root, camera),
                selected_source_indices,
            )
            camera_reports[camera] = {"decoded_through_source_frame": decoded, "written": written}
            print(f"Converted camera {camera}: {written} frames", flush=True)

        report = {
            "source_root": str(source_root),
            "output_zarr": str(output_zarr),
            "fps": info["fps"],
            "episodes": len(selected_ids),
            "frames": selected_frames,
            "source_episode_indices": selected_ids,
            "excluded_episode_indices": sorted(set(args.exclude_episodes)),
            "state_shape": list(tabular["state"].shape[1:]),
            "action_shape": list(tabular["action"].shape[1:]),
            "camera_keys": list(args.camera_keys),
            "camera_reports": camera_reports,
            "include_rl": args.include_rl,
            "return_gamma": args.return_gamma if args.include_rl else None,
        }
        root.attrs.update(report)
        (temporary / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(temporary, output_zarr)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Created RL-100 Zarr: {output_zarr}")
    print(f"Episodes: {len(selected_ids)}, frames: {selected_frames}")


if __name__ == "__main__":
    main()
