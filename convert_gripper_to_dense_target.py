#!/usr/bin/env python3
"""Create a LeRobot dataset whose gripper action is a persistent target state.

The source Marvin dataset uses sparse gripper commands: -1 closes, +1 opens,
and values in the dead zone mean "do nothing".  ACT is a regression policy, so
this converter turns that event representation into a persistent target:

* closed target: -1 on every frame
* open target: +1 on every frame

The conversion is episode-local.  The initial target comes from
``observation.state[0]`` (``gripper_pose``), and subsequent sparse commands
update it.  ``observation.state[5]`` (``last_action.gripper``) is converted as
well so that policy inputs and action labels have the same semantics.

All other action/state dimensions and episode indices are unchanged.  Video
files are reused by hard link by default and are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ACTION_KEY = "action"
STATE_KEY = "observation.state"
EPISODE_KEY = "episode_index"
FRAME_KEY = "frame_index"
GRIPPER_ACTION_INDEX = 4
GRIPPER_POSE_INDEX = 0
LAST_GRIPPER_ACTION_INDEX = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Local LeRobot dataset id used when validating, e.g. xlb/marvin_usb_act_success_dense_gripper.",
    )
    parser.add_argument(
        "--command-threshold",
        type=float,
        default=0.5,
        help="Sparse action <= -threshold closes and >= threshold opens.",
    )
    parser.add_argument(
        "--pose-open-threshold",
        type=float,
        default=0.5,
        help="At an episode start, gripper_pose above this value means open.",
    )
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to reuse source video files. Hard links save disk space and remain local-file compatible.",
    )
    args = parser.parse_args()
    if args.command_threshold <= 0:
        parser.error("--command-threshold must be positive")
    if args.source_root.resolve() == args.output_root.resolve():
        parser.error("Source and output roots must differ; the source dataset is never edited in place")
    return args


def densify_episode_arrays(
    actions: np.ndarray,
    states: np.ndarray,
    command_threshold: float = 0.5,
    pose_open_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return copied arrays with persistent gripper targets for one episode."""
    actions = np.asarray(actions, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] <= GRIPPER_ACTION_INDEX:
        raise ValueError(f"Expected action shape [T, >=5], got {actions.shape}")
    if states.ndim != 2 or states.shape[1] <= LAST_GRIPPER_ACTION_INDEX:
        raise ValueError(f"Expected state shape [T, >=6], got {states.shape}")
    if len(actions) != len(states) or len(actions) == 0:
        raise ValueError(f"Action/state episode lengths must match and be nonzero: {len(actions)}, {len(states)}")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Action/state contains NaN or infinity")

    dense_actions = actions.copy()
    dense_states = states.copy()
    target = 1.0 if states[0, GRIPPER_POSE_INDEX] > pose_open_threshold else -1.0
    initial_target = target
    transitions = 0

    for frame in range(len(actions)):
        # State at t contains the target that preceded action_t.
        dense_states[frame, LAST_GRIPPER_ACTION_INDEX] = target
        sparse_command = float(actions[frame, GRIPPER_ACTION_INDEX])
        previous_target = target
        if sparse_command <= -command_threshold:
            target = -1.0
        elif sparse_command >= command_threshold:
            target = 1.0
        transitions += int(target != previous_target)
        dense_actions[frame, GRIPPER_ACTION_INDEX] = target

    details = {
        "initial_target": int(initial_target),
        "target_transitions": transitions,
        "changed_action_frames": int(
            np.count_nonzero(dense_actions[:, GRIPPER_ACTION_INDEX] != actions[:, GRIPPER_ACTION_INDEX])
        ),
        "changed_last_action_frames": int(
            np.count_nonzero(
                dense_states[:, LAST_GRIPPER_ACTION_INDEX] != states[:, LAST_GRIPPER_ACTION_INDEX]
            )
        ),
    }
    return dense_actions, dense_states, details


def fixed_list_to_numpy(column: pa.ChunkedArray, width: int) -> np.ndarray:
    array = column.combine_chunks()
    if not pa.types.is_fixed_size_list(array.type) or array.type.list_size != width:
        raise ValueError(f"Expected fixed-size-list[{width}], got {array.type}")
    return array.values.to_numpy(zero_copy_only=False).reshape(len(array), width).astype(np.float32, copy=False)


def replace_fixed_list_column(table: pa.Table, key: str, values: np.ndarray) -> pa.Table:
    flat = pa.array(values.reshape(-1), type=pa.float32())
    replacement = pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    return table.set_column(table.schema.get_field_index(key), key, replacement)


def classify_sparse(values: np.ndarray, threshold: float) -> Counter:
    return Counter(
        {
            "close": int(np.count_nonzero(values <= -threshold)),
            "hold": int(np.count_nonzero((values > -threshold) & (values < threshold))),
            "open": int(np.count_nonzero(values >= threshold)),
        }
    )


def classify_dense(values: np.ndarray) -> Counter:
    return Counter(
        {
            "closed_-1": int(np.count_nonzero(values == -1.0)),
            "open_+1": int(np.count_nonzero(values == 1.0)),
            "other": int(np.count_nonzero((values != -1.0) & (values != 1.0))),
        }
    )


def copy_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path, mode: str = "copy") -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            copy_file(path, target, mode)


def convert_parquet_file(source: Path, destination: Path, args: argparse.Namespace, totals: dict) -> None:
    parquet = pq.ParquetFile(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for row_group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(row_group_index)
            missing = {ACTION_KEY, STATE_KEY, EPISODE_KEY, FRAME_KEY} - set(table.column_names)
            if missing:
                raise ValueError(f"{source} row group {row_group_index} missing columns: {sorted(missing)}")

            episode_ids = np.asarray(table[EPISODE_KEY].combine_chunks())
            frame_ids = np.asarray(table[FRAME_KEY].combine_chunks())
            unique_episode_ids = np.unique(episode_ids)
            if len(unique_episode_ids) != 1:
                raise ValueError(f"Row group must contain exactly one episode, got {unique_episode_ids.tolist()}")
            if len(frame_ids) > 1 and np.any(np.diff(frame_ids) != 1):
                raise ValueError(f"Episode {int(unique_episode_ids[0])} frame_index is not contiguous")

            action_width = table.schema.field(ACTION_KEY).type.list_size
            state_width = table.schema.field(STATE_KEY).type.list_size
            actions = fixed_list_to_numpy(table[ACTION_KEY], action_width)
            states = fixed_list_to_numpy(table[STATE_KEY], state_width)
            sparse_counts = classify_sparse(actions[:, GRIPPER_ACTION_INDEX], args.command_threshold)
            dense_actions, dense_states, details = densify_episode_arrays(
                actions,
                states,
                command_threshold=args.command_threshold,
                pose_open_threshold=args.pose_open_threshold,
            )
            dense_counts = classify_dense(dense_actions[:, GRIPPER_ACTION_INDEX])

            table = replace_fixed_list_column(table, ACTION_KEY, dense_actions)
            table = replace_fixed_list_column(table, STATE_KEY, dense_states)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema)
            writer.write_table(table)

            totals["episodes"] += 1
            totals["frames"] += len(actions)
            totals["target_transitions"] += details["target_transitions"]
            totals["changed_action_frames"] += details["changed_action_frames"]
            totals["changed_last_action_frames"] += details["changed_last_action_frames"]
            totals["initial_open_episodes"] += int(details["initial_target"] == 1)
            totals["initial_closed_episodes"] += int(details["initial_target"] == -1)
            totals["source_action_distribution"].update(sparse_counts)
            totals["dense_action_distribution"].update(dense_counts)
    finally:
        if writer is not None:
            writer.close()


def recompute_numeric_stats(output_root: Path, repo_id: str) -> None:
    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=output_root, download_videos=False)
    recompute_stats(dataset, skip_image_video=True)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}. Choose a new directory or remove it explicitly after checking it."
        )
    data_files = sorted((source_root / "data").glob("*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {source_root / 'data'}")

    output_root.mkdir(parents=True)
    try:
        copy_tree(source_root / "meta", output_root / "meta", mode="copy")
        # v3 datasets may store camera frames as videos or inline/image files.
        # The latter is used by the human40 dataset to avoid video decoding ABI
        # issues in the residual environment.
        media_dir_name = "videos" if (source_root / "videos").is_dir() else "images"
        copy_tree(source_root / media_dir_name, output_root / media_dir_name, mode=args.video_mode)

        totals = {
            "episodes": 0,
            "frames": 0,
            "target_transitions": 0,
            "changed_action_frames": 0,
            "changed_last_action_frames": 0,
            "initial_open_episodes": 0,
            "initial_closed_episodes": 0,
            "source_action_distribution": Counter(),
            "dense_action_distribution": Counter(),
        }
        for source_file in data_files:
            destination = output_root / source_file.relative_to(source_root)
            convert_parquet_file(source_file, destination, args, totals)

        recompute_numeric_stats(output_root, args.repo_id)

        report = {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "repo_id": args.repo_id,
            "semantics": {
                "action.gripper": "persistent target: closed=-1, open=+1",
                "observation.state.last_action.gripper": "previous persistent target",
                "command_threshold": args.command_threshold,
                "pose_open_threshold": args.pose_open_threshold,
            },
            "media_dir": media_dir_name,
            "video_mode": args.video_mode if media_dir_name == "videos" else None,
            **{
                key: dict(value) if isinstance(value, Counter) else value
                for key, value in totals.items()
            },
        }
        (output_root / "dense_gripper_conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        # Final structural validation through LeRobot's own loader.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=args.repo_id, root=output_root, download_videos=False)
        if len(dataset) != totals["frames"]:
            raise RuntimeError(f"LeRobot loaded {len(dataset)} frames, expected {totals['frames']}")

        print(f"Created dense-gripper dataset: {output_root}")
        print(f"Episodes: {totals['episodes']}, frames: {totals['frames']}")
        print(f"Source gripper labels: {dict(totals['source_action_distribution'])}")
        print(f"Dense gripper labels: {dict(totals['dense_action_distribution'])}")
        print(f"Changed action frames: {totals['changed_action_frames']}")
        print(f"Report: {output_root / 'dense_gripper_conversion_report.json'}")
    except Exception:
        # The output did not pass conversion/validation, so do not leave a dataset
        # that could accidentally be used for training.
        shutil.rmtree(output_root)
        raise


if __name__ == "__main__":
    main()
