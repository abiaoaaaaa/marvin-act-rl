#!/usr/bin/env python3
"""Convert Marvin HIL-SERL buffers into a failure-aware LeRobot ACT-RL dataset.

Unlike ``convert_marvin_buffers_to_lerobot.py`` (the behavior-cloning converter),
this converter keeps every *complete* online episode, including failures.  It
also stores per-transition rewards/termination/intervention labels and writes an
``meta/rl_outcomes.json`` manifest consumed by ``act_rl.scripts.train_offline``.

The source pickle files must be trusted local files.  Pickle is unsafe for
untrusted input.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

import numpy as np

from convert_gripper_to_dense_target import densify_episode_arrays
from convert_marvin_buffers_to_lerobot import (
    DEFAULT_CAMERAS,
    Episode,
    EpisodeSummary,
    collect_intervention_signatures,
    discover_shards,
    episode_intervention_steps,
    inspect_transition_schema,
    is_successful,
    iter_episodes,
    iter_shard_records,
    make_features,
    reference_demo_episodes,
    scalar_float,
    squeeze_observation,
    transition_signature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="HIL-SERL run directory containing buffer/ and demo_buffer/.")
    parser.add_argument("--reference-demo-pkl", type=Path, action="append", default=[],
                        help="Optional standalone successful demonstration pickle; may be repeated.")
    parser.add_argument("--output-root", type=Path, help="New local LeRobot dataset directory.")
    parser.add_argument("--repo-id", help="Dataset id, e.g. xlb/marvin_usb_act_rl_offline.")
    parser.add_argument("--task", default="Insert the USB plug into the port.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--failure-terminal-reward", type=float, default=0.0,
                        help="Manifest fallback only; recorded per-step rewards are preserved unchanged.")
    parser.add_argument("--camera-keys", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--continuity-camera", default="side_policy")
    parser.add_argument("--expected-state-dim", type=int, default=24)
    parser.add_argument("--expected-action-dim", type=int, default=5)
    parser.add_argument("--image-storage", choices=("video", "image"), default="video")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--exclude-reference-demos", action="store_true")
    parser.add_argument("--keep-sparse-gripper", action="store_true",
                        help="Do not convert sparse -1/0/+1 commands to persistent -1/+1 targets.")
    parser.add_argument("--gripper-command-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-pose-open-threshold", type=float, default=0.5)
    parser.add_argument("--max-episode-length", type=int, default=None,
                        help="Optional corruption guard. Complete episodes longer than this are excluded.")
    parser.add_argument("--max-shards", type=int, default=None, help="Debug: inspect the first N shards.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug: write at most N episodes total.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0 or args.gripper_command_threshold <= 0:
        parser.error("fps and gripper-command-threshold must be positive")
    if args.max_episode_length is not None and args.max_episode_length <= 0:
        parser.error("--max-episode-length must be positive")
    if not args.dry_run and (args.output_root is None or args.repo_id is None):
        parser.error("--output-root and --repo-id are required unless --dry-run is used")
    if args.continuity_camera not in args.camera_keys:
        parser.error("--continuity-camera must be one of --camera-keys")
    return args


def rl_features(base: dict) -> dict:
    result = dict(base)
    result.update(
        {
            "rl.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "rl.terminated": {"dtype": "bool", "shape": (1,), "names": None},
            "rl.success": {"dtype": "bool", "shape": (1,), "names": None},
            "rl.intervention": {"dtype": "bool", "shape": (1,), "names": None},
            "rl.safety_abort": {"dtype": "bool", "shape": (1,), "names": None},
        }
    )
    return result


def prepare_episode_arrays(episode: Episode, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    actions = np.stack([np.asarray(t["actions"]).reshape(-1) for t in episode.transitions]).astype(np.float32)
    states = np.stack(
        [squeeze_observation(t["observations"]["state"]).reshape(-1) for t in episode.transitions]
    ).astype(np.float32)
    if args.keep_sparse_gripper:
        return actions, states, {"dense_gripper": False, "changed_action_frames": 0}
    dense_actions, dense_states, details = densify_episode_arrays(
        actions,
        states,
        command_threshold=args.gripper_command_threshold,
        pose_open_threshold=args.gripper_pose_open_threshold,
    )
    details["dense_gripper"] = True
    return dense_actions, dense_states, details


def add_episode(
    dataset,
    episode: Episode,
    args: argparse.Namespace,
    output_episode_index: int,
    source: str,
    intervention_signatures: set[bytes],
    force_intervention: bool = False,
) -> tuple[dict, dict]:
    successful = is_successful(episode.transitions, args.success_threshold)
    actions, states, gripper_details = prepare_episode_arrays(episode, args)
    intervention_count = 0
    reward_sum = 0.0
    reward_max = float("-inf")
    for step, transition in enumerate(episode.transitions):
        intervened = force_intervention or transition_signature(transition) in intervention_signatures
        intervention_count += int(intervened)
        reward = scalar_float(transition.get("rewards", 0.0))
        reward_sum += reward
        reward_max = max(reward_max, reward)
        observation = transition["observations"]
        frame = {
            "task": args.task,
            "observation.state": states[step],
            "action": actions[step],
            "rl.reward": np.asarray([reward], dtype=np.float32),
            "rl.terminated": np.asarray([step == len(episode.transitions) - 1], dtype=bool),
            "rl.success": np.asarray([successful], dtype=bool),
            "rl.intervention": np.asarray([intervened], dtype=bool),
            # HIL-SERL's saved transition schema does not retain a distinct
            # safety-abort bit.  Do not infer one from an ordinary failure.
            "rl.safety_abort": np.asarray([False], dtype=bool),
        }
        for camera in args.camera_keys:
            frame[f"observation.images.{camera}"] = squeeze_observation(observation[camera]).astype(
                np.uint8, copy=False
            )
        dataset.add_frame(frame)
    dataset.save_episode()
    outcome = {
        "success": successful,
        "terminal_reward": 1.0 if successful else args.failure_terminal_reward,
        "source": source,
        "length": len(episode.transitions),
        "recorded_return": reward_sum,
        "max_step_reward": reward_max,
        "intervention_steps": intervention_count,
        "safety_abort": False,
        "complete": True,
    }
    report = {
        "output_episode_index": output_episode_index,
        "source": source,
        "length": len(episode.transitions),
        "successful": successful,
        "intervention_steps": intervention_count,
        "recorded_return": reward_sum,
        "gripper": gripper_details,
        "start_shard": episode.start_shard,
        "start_index": episode.start_index,
        "end_shard": episode.end_shard,
        "end_index": episode.end_index,
    }
    return outcome, report


def selected_online_episodes(shards: Sequence[Path], args: argparse.Namespace):
    for stream_index, episode in enumerate(iter_episodes(iter_shard_records(shards), args.continuity_camera)):
        complete = episode.boundary == "done"
        within_limit = args.max_episode_length is None or len(episode.transitions) <= args.max_episode_length
        if complete and within_limit:
            yield stream_index, episode


def analyze(args, online_shards, demo_signatures) -> tuple[list[EpisodeSummary], tuple]:
    summaries: list[EpisodeSummary] = []
    schema = None
    for stream_index, episode in enumerate(
        iter_episodes(iter_shard_records(online_shards), args.continuity_camera)
    ):
        if schema is None and episode.transitions:
            schema = inspect_transition_schema(
                episode.transitions[0], args.camera_keys, args.expected_state_dim, args.expected_action_dim
            )
        complete = episode.boundary == "done"
        within_limit = args.max_episode_length is None or len(episode.transitions) <= args.max_episode_length
        summaries.append(
            EpisodeSummary(
                stream_episode_index=stream_index,
                source="online_buffer",
                length=len(episode.transitions),
                successful=is_successful(episode.transitions, args.success_threshold),
                complete=complete,
                intervention_steps=episode_intervention_steps(episode, demo_signatures),
                start_shard=episode.start_shard,
                start_index=episode.start_index,
                end_shard=episode.end_shard,
                end_index=episode.end_index,
                boundary=episode.boundary,
                selected=complete and within_limit,
            )
        )
    if schema is None:
        raise RuntimeError("online buffer contains no transitions")
    return summaries, schema


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    online_shards = discover_shards(checkpoint_dir / "buffer", args.max_shards)
    demo_shards = discover_shards(checkpoint_dir / "demo_buffer", args.max_shards)
    if not online_shards:
        raise FileNotFoundError(f"No buffer shards under {checkpoint_dir / 'buffer'}")
    demo_signatures = collect_intervention_signatures(demo_shards)
    summaries, schema = analyze(args, online_shards, demo_signatures)
    selected = [x for x in summaries if x.selected]
    successes = [x for x in selected if x.successful]
    failures = [x for x in selected if not x.successful]
    incomplete = [x for x in summaries if not x.complete]
    print("\nACT-RL selection summary")
    print(f"  online shards: {len(online_shards)}")
    print(f"  intervention signatures: {len(demo_signatures)}")
    print(f"  observed episodes: {len(summaries)}")
    print(f"  selected complete episodes: {len(selected)}")
    print(f"  successes / failures: {len(successes)} / {len(failures)}")
    print(f"  incomplete excluded: {len(incomplete)}")
    print(f"  selected frames: {sum(x.length for x in selected)}")
    print(f"  schema: image={schema[0]}, state={schema[1]}, action={schema[2]}")
    if args.dry_run:
        return 0

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = rl_features(
        make_features(args.camera_keys, schema[0], schema[1], schema[2], args.image_storage)
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
    outcomes: dict[str, dict] = {}
    episode_reports: list[dict] = []
    output_index = 0
    try:
        if not args.exclude_reference_demos:
            for path, episode in reference_demo_episodes(args.reference_demo_pkl, args.continuity_camera):
                if episode.boundary != "done" or not is_successful(episode.transitions, args.success_threshold):
                    continue
                outcome, report = add_episode(
                    dataset, episode, args, output_index, f"reference_demo:{path.name}", demo_signatures,
                    force_intervention=True,
                )
                outcomes[str(output_index)] = outcome
                episode_reports.append(report)
                output_index += 1
                if args.max_episodes is not None and output_index >= args.max_episodes:
                    break
        if args.max_episodes is None or output_index < args.max_episodes:
            for stream_index, episode in selected_online_episodes(online_shards, args):
                outcome, report = add_episode(
                    dataset, episode, args, output_index, "online_buffer", demo_signatures
                )
                report["stream_episode_index"] = stream_index
                outcomes[str(output_index)] = outcome
                episode_reports.append(report)
                output_index += 1
                if args.max_episodes is not None and output_index >= args.max_episodes:
                    break
        dataset.finalize()

        manifest = {
            "format_version": 2,
            "reward_source": "recorded rl.reward feature; terminal_reward is fallback only",
            "safety_abort_limitation": "source buffer does not retain a distinct safety-abort label",
            "episodes": outcomes,
        }
        (output_root / "meta" / "rl_outcomes.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        report = {
            "inputs": {
                "checkpoint_dir": str(checkpoint_dir),
                "online_shards": [x.name for x in online_shards],
                "demo_buffer_shards": [x.name for x in demo_shards],
                "reference_demo_pickles": [str(x.resolve()) for x in args.reference_demo_pkl],
            },
            "policy": {
                "complete_online_episodes": True,
                "keep_successes": True,
                "keep_failures": True,
                "exclude_incomplete": True,
                "max_episode_length": args.max_episode_length,
                "dense_gripper": not args.keep_sparse_gripper,
                "success_threshold": args.success_threshold,
            },
            "analysis": [asdict(x) for x in summaries],
            "written_episodes": episode_reports,
        }
        (output_root / "act_rl_conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    except Exception:
        with np.errstate(all="ignore"):
            dataset.finalize()
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    finally:
        gc.collect()

    written_successes = sum(bool(x["success"]) for x in outcomes.values())
    print(f"\nCreated ACT-RL dataset: {output_root}")
    print(f"Episodes: {len(outcomes)} (success={written_successes}, failure={len(outcomes)-written_successes})")
    print(f"Outcome manifest: {output_root / 'meta' / 'rl_outcomes.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
