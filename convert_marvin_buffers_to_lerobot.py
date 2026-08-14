#!/usr/bin/env python3
"""Convert high-quality Marvin HIL-SERL episodes to a LeRobot ACT dataset.

The online ``buffer`` contains the complete transition stream.  The
``demo_buffer`` contains only intervention transitions, so it is used as a set
of markers: an online episode is considered a demo/intervention episode when
one or more of its transitions also occur in ``demo_buffer``.

Selection policy:

1. Keep only complete episodes (ending with ``dones=True``).
2. Keep only successful episodes (reward >= ``--success-threshold``).
3. Keep an online episode only when
   ``episode_length < max_demo_episode_length`` (strict inequality).
4. Optionally include successful episodes from standalone ``record_demos.py``
   pickle files.  These files are also the preferred source for computing
   ``max_demo_episode_length``.

The four default cameras match the current Marvin policy configuration:
``side_policy``, ``wrist_1buchong``, ``wrist_1`` and ``wrist_2``.

Example:

    PYTHONPATH=/home/xlb/code/lerobot/src \
    /home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
      convert_marvin_buffers_to_lerobot.py \
      --checkpoint-dir /path/to/checkpoints_scale10_acc300_alpha080_4cam_fresh_20260807 \
      --reference-demo-pkl /path/to/marvin_usb_insertion_20_demos_2026-08-07_10-16-21.pkl \
      --output-root /path/to/marvin_usb_act_success \
      --repo-id xlb/marvin_usb_act_success

Pickle is unsafe for untrusted input.  Only run this on locally generated,
trusted buffer files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


DEFAULT_CAMERAS = ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2")
STATE_NAMES_24 = (
    "gripper_pose",
    "last_action.dx",
    "last_action.dy",
    "last_action.dz",
    "last_action.drz",
    "last_action.gripper",
    "tcp_force.x",
    "tcp_force.y",
    "tcp_force.z",
    "tcp_pose.x",
    "tcp_pose.y",
    "tcp_pose.z",
    "tcp_pose.rx",
    "tcp_pose.ry",
    "tcp_pose.rz",
    "tcp_torque.x",
    "tcp_torque.y",
    "tcp_torque.z",
    "tcp_vel.x",
    "tcp_vel.y",
    "tcp_vel.z",
    "tcp_vel.rx",
    "tcp_vel.ry",
    "tcp_vel.rz",
)
ACTION_NAMES_5 = ("dx", "dy", "dz", "drz", "gripper")
SHARD_PATTERN = re.compile(r"^transitions_(\d+)\.pkl$")


@dataclass(frozen=True)
class TransitionRecord:
    transition: dict
    shard: str
    shard_step: int
    index_in_shard: int
    force_boundary_before: bool = False


@dataclass
class Episode:
    transitions: list[dict]
    start_shard: str
    start_index: int
    end_shard: str
    end_index: int
    boundary: str


@dataclass
class EpisodeSummary:
    stream_episode_index: int
    source: str
    length: int
    successful: bool
    complete: bool
    intervention_steps: int
    start_shard: str
    start_index: int
    end_shard: str
    end_index: int
    boundary: str
    selected: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Run directory containing buffer/ and demo_buffer/.",
    )
    parser.add_argument(
        "--reference-demo-pkl",
        type=Path,
        action="append",
        default=[],
        help=(
            "Standalone record_demos.py pickle. May be repeated. Its complete successful "
            "episodes define max_demo_episode_length and are included by default."
        ),
    )
    parser.add_argument(
        "--max-demo-episode-length",
        type=int,
        default=None,
        help=(
            "Explicit comparison threshold. If omitted, use standalone demo pickle(s); "
            "otherwise derive it from successful online episodes containing intervention transitions."
        ),
    )
    parser.add_argument("--output-root", type=Path, help="New local LeRobot dataset directory.")
    parser.add_argument("--repo-id", help="LeRobot dataset id, e.g. xlb/marvin_usb_act_success.")
    parser.add_argument("--task", default="Insert the USB plug into the port.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--camera-keys", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--continuity-camera", default="side_policy")
    parser.add_argument("--expected-state-dim", type=int, default=24)
    parser.add_argument("--expected-action-dim", type=int, default=5)
    parser.add_argument(
        "--image-storage",
        choices=("video", "image"),
        default="video",
        help="Video is recommended for the four-camera dataset.",
    )
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument(
        "--exclude-reference-demos",
        action="store_true",
        help="Use reference demos to set the length threshold but do not add them to the output dataset.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Debug option: inspect only the first N online/demo shards.",
    )
    parser.add_argument(
        "--max-online-episodes",
        type=int,
        default=None,
        help="Debug option: stop conversion after writing N selected online episodes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print the selection without creating a LeRobot dataset.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_demo_episode_length is not None and args.max_demo_episode_length <= 0:
        parser.error("--max-demo-episode-length must be positive")
    if not args.dry_run and (args.output_root is None or args.repo_id is None):
        parser.error("--output-root and --repo-id are required unless --dry-run is used")
    if args.continuity_camera not in args.camera_keys:
        parser.error("--continuity-camera must also appear in --camera-keys")
    return args


def shard_step(path: Path) -> int:
    match = SHARD_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected shard filename: {path}")
    return int(match.group(1))


def discover_shards(directory: Path, max_shards: int | None = None) -> list[Path]:
    if not directory.is_dir():
        return []
    shards = [path for path in directory.glob("transitions_*.pkl") if SHARD_PATTERN.match(path.name)]
    shards.sort(key=shard_step)
    if max_shards is not None:
        shards = shards[:max_shards]
    return shards


def nominal_shard_stride(shards: Sequence[Path]) -> int | None:
    if len(shards) < 2:
        return None
    steps = [shard_step(path) for path in shards]
    positive_diffs = [b - a for a, b in zip(steps, steps[1:], strict=False) if b > a]
    return min(positive_diffs) if positive_diffs else None


def load_pickle_list(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, list):
        raise TypeError(f"{path} contains {type(value).__name__}, expected a list")
    return value


def iter_shard_records(shards: Sequence[Path]) -> Iterator[TransitionRecord]:
    stride = nominal_shard_stride(shards)
    previous_step: int | None = None
    for path in shards:
        step = shard_step(path)
        gap_before = previous_step is not None and stride is not None and step - previous_step > stride
        transitions = load_pickle_list(path)
        print(f"Loading {path.name}: {len(transitions)} transitions", flush=True)
        for index, transition in enumerate(transitions):
            if not isinstance(transition, dict):
                raise TypeError(f"{path}[{index}] is not a transition dict")
            yield TransitionRecord(
                transition=transition,
                shard=path.name,
                shard_step=step,
                index_in_shard=index,
                force_boundary_before=gap_before and index == 0,
            )
        previous_step = step
        del transitions
        gc.collect()


def scalar_bool(value: object) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar bool, got shape {array.shape}")
    return bool(array.reshape(-1)[0])


def scalar_float(value: object) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar float, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def is_done(transition: dict) -> bool:
    return scalar_bool(transition.get("dones", False))


def is_successful(transitions: Sequence[dict], threshold: float) -> bool:
    for transition in transitions:
        if scalar_float(transition.get("rewards", 0.0)) >= threshold:
            return True
        infos = transition.get("infos")
        if isinstance(infos, dict) and scalar_bool(infos.get("succeed", False)):
            return True
    return False


def squeeze_observation(array: object) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def observations_are_contiguous(previous: dict, current: dict, camera_key: str) -> bool:
    """Check the actor-loop invariant: previous next_obs equals current obs."""
    previous_next = previous.get("next_observations")
    current_obs = current.get("observations")
    if not isinstance(previous_next, dict) or not isinstance(current_obs, dict):
        return False
    if "state" not in previous_next or "state" not in current_obs:
        return False
    if not np.array_equal(previous_next["state"], current_obs["state"]):
        return False
    if camera_key not in previous_next or camera_key not in current_obs:
        return False
    return np.array_equal(previous_next[camera_key], current_obs[camera_key])


def iter_episodes(records: Iterable[TransitionRecord], continuity_camera: str) -> Iterator[Episode]:
    pending: list[dict] = []
    start_shard = ""
    start_index = -1
    end_shard = ""
    end_index = -1

    for record in records:
        transition = record.transition
        discontinuity = bool(
            pending
            and (
                record.force_boundary_before
                or not observations_are_contiguous(pending[-1], transition, continuity_camera)
            )
        )
        if discontinuity:
            yield Episode(
                transitions=pending,
                start_shard=start_shard,
                start_index=start_index,
                end_shard=end_shard,
                end_index=end_index,
                boundary="shard_gap_or_observation_discontinuity",
            )
            pending = []

        if not pending:
            start_shard = record.shard
            start_index = record.index_in_shard
        pending.append(transition)
        end_shard = record.shard
        end_index = record.index_in_shard

        if is_done(transition):
            yield Episode(
                transitions=pending,
                start_shard=start_shard,
                start_index=start_index,
                end_shard=end_shard,
                end_index=end_index,
                boundary="done",
            )
            pending = []

    if pending:
        yield Episode(
            transitions=pending,
            start_shard=start_shard,
            start_index=start_index,
            end_shard=end_shard,
            end_index=end_index,
            boundary="end_of_input",
        )


def update_hash_with_array(digest: "hashlib._Hash", value: object) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def transition_signature(transition: dict) -> bytes:
    """Stable low-dimensional signature shared by online and intervention copies."""
    observations = transition["observations"]
    next_observations = transition["next_observations"]
    digest = hashlib.blake2b(digest_size=16)
    update_hash_with_array(digest, observations["state"])
    update_hash_with_array(digest, transition["actions"])
    update_hash_with_array(digest, next_observations["state"])
    return digest.digest()


def collect_intervention_signatures(shards: Sequence[Path]) -> set[bytes]:
    signatures: set[bytes] = set()
    for path in shards:
        transitions = load_pickle_list(path)
        print(f"Indexing interventions from {path.name}: {len(transitions)}", flush=True)
        for transition in transitions:
            signatures.add(transition_signature(transition))
        del transitions
        gc.collect()
    return signatures


def episode_intervention_steps(episode: Episode, signatures: set[bytes]) -> int:
    return sum(transition_signature(transition) in signatures for transition in episode.transitions)


def summarize_online_episodes(
    shards: Sequence[Path],
    intervention_signatures: set[bytes],
    continuity_camera: str,
    success_threshold: float,
    camera_keys: Sequence[str],
    expected_state_dim: int,
    expected_action_dim: int,
) -> tuple[list[EpisodeSummary], tuple[tuple[int, int, int], int, int]]:
    summaries: list[EpisodeSummary] = []
    schema: tuple[tuple[int, int, int], int, int] | None = None
    records = iter_shard_records(shards)
    for index, episode in enumerate(iter_episodes(records, continuity_camera)):
        if schema is None and episode.transitions:
            schema = inspect_transition_schema(
                episode.transitions[0],
                camera_keys,
                expected_state_dim,
                expected_action_dim,
            )
        summaries.append(
            EpisodeSummary(
                stream_episode_index=index,
                source="online_buffer",
                length=len(episode.transitions),
                successful=is_successful(episode.transitions, success_threshold),
                complete=episode.boundary == "done",
                intervention_steps=episode_intervention_steps(episode, intervention_signatures),
                start_shard=episode.start_shard,
                start_index=episode.start_index,
                end_shard=episode.end_shard,
                end_index=episode.end_index,
                boundary=episode.boundary,
            )
        )
    if schema is None:
        raise RuntimeError("The online buffer contains no transitions")
    return summaries, schema


def reference_demo_episodes(paths: Sequence[Path], continuity_camera: str) -> Iterator[tuple[Path, Episode]]:
    for path in paths:
        transitions = load_pickle_list(path)
        records = (
            TransitionRecord(
                transition=transition,
                shard=path.name,
                shard_step=0,
                index_in_shard=index,
            )
            for index, transition in enumerate(transitions)
        )
        for episode in iter_episodes(records, continuity_camera):
            yield path, episode
        del transitions
        gc.collect()


def inspect_transition_schema(
    transition: dict,
    camera_keys: Sequence[str],
    expected_state_dim: int,
    expected_action_dim: int,
) -> tuple[tuple[int, int, int], int, int]:
    observations = transition.get("observations")
    if not isinstance(observations, dict):
        raise ValueError("Transition is missing observations dict")

    image_shape: tuple[int, int, int] | None = None
    for camera in camera_keys:
        if camera not in observations:
            raise KeyError(f"Missing required camera {camera!r}; available={list(observations)}")
        image = squeeze_observation(observations[camera])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera {camera} has shape {image.shape}, expected HWC RGB")
        if image.dtype != np.uint8:
            raise ValueError(f"Camera {camera} has dtype {image.dtype}, expected uint8")
        current_shape = tuple(int(x) for x in image.shape)
        if image_shape is None:
            image_shape = current_shape
        elif current_shape != image_shape:
            raise ValueError(f"All ACT cameras must have one shape, got {image_shape} and {current_shape}")

    state = squeeze_observation(observations["state"]).reshape(-1)
    action = np.asarray(transition["actions"]).reshape(-1)
    if state.size != expected_state_dim:
        raise ValueError(f"State dim {state.size}, expected {expected_state_dim}")
    if action.size != expected_action_dim:
        raise ValueError(f"Action dim {action.size}, expected {expected_action_dim}")
    assert image_shape is not None
    return image_shape, int(state.size), int(action.size)


def make_features(
    camera_keys: Sequence[str],
    image_shape: tuple[int, int, int],
    state_dim: int,
    action_dim: int,
    image_storage: str,
) -> dict:
    state_names = list(STATE_NAMES_24) if state_dim == 24 else [f"state_{i}" for i in range(state_dim)]
    action_names = list(ACTION_NAMES_5) if action_dim == 5 else [f"action_{i}" for i in range(action_dim)]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"axes": state_names},
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": {"axes": action_names},
        },
    }
    for camera in camera_keys:
        features[f"observation.images.{camera}"] = {
            "dtype": image_storage,
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }
    return features


def transition_to_lerobot_frame(transition: dict, camera_keys: Sequence[str], task: str) -> dict:
    observations = transition["observations"]
    frame = {
        "task": task,
        "observation.state": squeeze_observation(observations["state"]).reshape(-1).astype(np.float32),
        "action": np.asarray(transition["actions"]).reshape(-1).astype(np.float32),
    }
    for camera in camera_keys:
        frame[f"observation.images.{camera}"] = squeeze_observation(observations[camera]).astype(
            np.uint8, copy=False
        )
    return frame


def selected_reference_summaries(
    paths: Sequence[Path],
    continuity_camera: str,
    success_threshold: float,
) -> tuple[list[EpisodeSummary], int | None]:
    summaries: list[EpisodeSummary] = []
    max_length: int | None = None
    for index, (path, episode) in enumerate(reference_demo_episodes(paths, continuity_camera)):
        successful = is_successful(episode.transitions, success_threshold)
        complete = episode.boundary == "done"
        selected = successful and complete
        if selected:
            max_length = max(max_length or 0, len(episode.transitions))
        summaries.append(
            EpisodeSummary(
                stream_episode_index=index,
                source=f"reference_demo:{path.name}",
                length=len(episode.transitions),
                successful=successful,
                complete=complete,
                intervention_steps=len(episode.transitions),
                start_shard=episode.start_shard,
                start_index=episode.start_index,
                end_shard=episode.end_shard,
                end_index=episode.end_index,
                boundary=episode.boundary,
                selected=selected,
            )
        )
    return summaries, max_length


def print_summary(
    online: Sequence[EpisodeSummary],
    reference: Sequence[EpisodeSummary],
    max_demo_length: int,
) -> None:
    complete_online = [item for item in online if item.complete]
    successful_online = [item for item in complete_online if item.successful]
    selected_online = [item for item in online if item.selected]
    intervention_episodes = [item for item in complete_online if item.intervention_steps > 0]
    print("\nSelection summary")
    print(f"  max_demo_episode_length: {max_demo_length}")
    print(f"  strict online filter: length < {max_demo_length}")
    print(f"  online episodes observed: {len(online)}")
    print(f"  online complete episodes: {len(complete_online)}")
    print(f"  online successful complete episodes: {len(successful_online)}")
    print(f"  online episodes containing intervention: {len(intervention_episodes)}")
    print(f"  selected online episodes: {len(selected_online)}")
    print(f"  selected online frames: {sum(item.length for item in selected_online)}")
    print(f"  selected reference demos: {sum(item.selected for item in reference)}")
    print(f"  selected reference frames: {sum(item.length for item in reference if item.selected)}")


def write_reference_episodes(dataset, args: argparse.Namespace) -> int:
    written = 0
    for _, episode in reference_demo_episodes(args.reference_demo_pkl, args.continuity_camera):
        if episode.boundary != "done" or not is_successful(episode.transitions, args.success_threshold):
            continue
        for transition in episode.transitions:
            dataset.add_frame(transition_to_lerobot_frame(transition, args.camera_keys, args.task))
        dataset.save_episode()
        written += 1
    return written


def write_online_episodes(
    dataset,
    args: argparse.Namespace,
    shards: Sequence[Path],
    selected_indices: set[int],
) -> int:
    written = 0
    records = iter_shard_records(shards)
    for index, episode in enumerate(iter_episodes(records, args.continuity_camera)):
        if index not in selected_indices:
            continue
        for transition in episode.transitions:
            dataset.add_frame(transition_to_lerobot_frame(transition, args.camera_keys, args.task))
        dataset.save_episode()
        written += 1
        if args.max_online_episodes is not None and written >= args.max_online_episodes:
            break
    return written


def write_report(
    output_root: Path,
    args: argparse.Namespace,
    online: Sequence[EpisodeSummary],
    reference: Sequence[EpisodeSummary],
    max_demo_length: int,
    online_shards: Sequence[Path],
    demo_shards: Sequence[Path],
) -> None:
    payload = {
        "selection_policy": {
            "complete_episode_required": True,
            "success_threshold": args.success_threshold,
            "online_length_rule": f"length < {max_demo_length}",
            "max_demo_episode_length": max_demo_length,
            "camera_keys": list(args.camera_keys),
            "fps": args.fps,
            "task": args.task,
        },
        "inputs": {
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "online_shards": [path.name for path in online_shards],
            "demo_buffer_shards": [path.name for path in demo_shards],
            "reference_demo_pickles": [str(path.resolve()) for path in args.reference_demo_pkl],
        },
        "counts": {
            "selected_online_episodes": sum(item.selected for item in online),
            "selected_reference_episodes": sum(item.selected for item in reference),
            "selected_online_frames": sum(item.length for item in online if item.selected),
            "selected_reference_frames": sum(item.length for item in reference if item.selected),
        },
        "online_episodes": [asdict(item) for item in online],
        "reference_episodes": [asdict(item) for item in reference],
    }
    with (output_root / "conversion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    online_shards = discover_shards(checkpoint_dir / "buffer", args.max_shards)
    demo_shards = discover_shards(checkpoint_dir / "demo_buffer", args.max_shards)
    if not online_shards:
        raise FileNotFoundError(f"No transitions_*.pkl found in {checkpoint_dir / 'buffer'}")

    print(f"Online shards: {len(online_shards)}")
    print(f"Demo/intervention shards: {len(demo_shards)}")
    print(f"Cameras: {args.camera_keys}")

    intervention_signatures = collect_intervention_signatures(demo_shards)
    print(f"Unique intervention transition signatures: {len(intervention_signatures)}")

    reference_summaries, reference_max = selected_reference_summaries(
        args.reference_demo_pkl,
        args.continuity_camera,
        args.success_threshold,
    )

    online_summaries, schema = summarize_online_episodes(
        online_shards,
        intervention_signatures,
        args.continuity_camera,
        args.success_threshold,
        args.camera_keys,
        args.expected_state_dim,
        args.expected_action_dim,
    )
    image_shape, state_dim, action_dim = schema
    print(f"Validated schema: images={image_shape}, state=({state_dim},), action=({action_dim},)")

    if args.max_demo_episode_length is not None:
        max_demo_length = args.max_demo_episode_length
    elif reference_max is not None:
        max_demo_length = reference_max
    else:
        intervention_lengths = [
            item.length
            for item in online_summaries
            if item.complete and item.successful and item.intervention_steps > 0
        ]
        if not intervention_lengths:
            raise RuntimeError(
                "Could not derive max demo episode length: no complete successful online episode "
                "contains a demo_buffer transition. Pass --reference-demo-pkl or "
                "--max-demo-episode-length explicitly."
            )
        max_demo_length = max(intervention_lengths)

    for item in online_summaries:
        item.selected = item.complete and item.successful and item.length < max_demo_length

    if args.exclude_reference_demos:
        for item in reference_summaries:
            item.selected = False

    if args.max_online_episodes is not None:
        selected_seen = 0
        for item in online_summaries:
            if not item.selected:
                continue
            selected_seen += 1
            if selected_seen > args.max_online_episodes:
                item.selected = False

    print_summary(online_summaries, reference_summaries, max_demo_length)
    if args.dry_run:
        print("Dry run complete; no dataset was written.")
        return 0

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output path already exists: {output_root}. Use a new path so an existing dataset is never overwritten."
        )

    # Import lazily so --dry-run can be used in the HIL-SERL environment.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = make_features(
        args.camera_keys,
        image_shape,
        state_dim,
        action_dim,
        args.image_storage,
    )
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=output_root,
        robot_type="marvin",
        features=features,
        use_videos=args.image_storage == "video",
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    try:
        reference_written = 0
        if not args.exclude_reference_demos:
            reference_written = write_reference_episodes(dataset, args)
        selected_online_indices = {item.stream_episode_index for item in online_summaries if item.selected}
        online_written = write_online_episodes(dataset, args, online_shards, selected_online_indices)
        dataset.finalize()
    except Exception:
        # Flush any completed episodes, while preserving the original exception.
        dataset.finalize()
        raise

    write_report(
        output_root,
        args,
        online_summaries,
        reference_summaries,
        max_demo_length,
        online_shards,
        demo_shards,
    )
    print(f"\nCreated dataset: {output_root}")
    print(f"Reference episodes written: {reference_written}")
    print(f"Online episodes written: {online_written}")
    print(f"Total episodes: {reference_written + online_written}")
    print(f"Report: {output_root / 'conversion_report.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
