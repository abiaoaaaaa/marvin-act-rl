"""Offline RL-100 checkpoint replay against a converted Marvin Zarr episode."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .constants import CAMERA_KEYS
from .rl100_policy import RL100PolicyRunner


LOG = logging.getLogger(__name__)


def episode_bounds(episode_ends: np.ndarray, episode_index: int) -> tuple[int, int]:
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(ends) <= 0):
        raise ValueError("Invalid Zarr episode_ends")
    if not 0 <= episode_index < len(ends):
        raise IndexError(f"episode_index must be in [0, {len(ends) - 1}]")
    start = 0 if episode_index == 0 else int(ends[episode_index - 1])
    return start, int(ends[episode_index])


def zarr_observation(data, index: int) -> dict:
    observation = {"state": np.asarray(data["state"][index], dtype=np.float32)}
    for key in CAMERA_KEYS:
        observation[key] = np.asarray(data["rgb_" + key][index], dtype=np.uint8)
    return observation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay one Marvin Zarr episode through RL-100")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--checkpoint-source", choices=("auto", "ema", "model"), default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-predictions", type=int)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.max_predictions is not None and args.max_predictions <= 0:
        raise ValueError("max_predictions must be positive")
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Offline replay requires zarr<3") from exc

    root = zarr.open(str(Path(args.dataset).expanduser().resolve()), mode="r")
    data = root["data"]
    start, end = episode_bounds(root["meta"]["episode_ends"][:], args.episode_index)
    policy = RL100PolicyRunner(
        checkpoint=args.checkpoint,
        device=args.device,
        checkpoint_source=args.checkpoint_source,
        deterministic=not args.stochastic,
    )
    policy.reset(zarr_observation(data, start), seed=args.seed)

    frame_indices = []
    predicted_chunks = []
    target_chunks = []
    clipped_rows = []
    inference_seconds = []
    for frame_index in range(start, end):
        if frame_index > start:
            policy.observe(zarr_observation(data, frame_index))
        if (frame_index - start) % args.stride:
            continue
        prediction, clipped = policy.predict_chunk()
        target_end = min(frame_index + policy.n_action_steps, end)
        target = np.asarray(data["action"][frame_index:target_end], dtype=np.float32)
        if len(target) < policy.n_action_steps:
            target = np.concatenate(
                [target, np.repeat(target[-1:], policy.n_action_steps - len(target), axis=0)],
                axis=0,
            )
        predicted_chunks.append(prediction)
        target_chunks.append(target)
        frame_indices.append(frame_index)
        clipped_rows.append(clipped)
        inference_seconds.append(policy.last_inference_seconds)
        LOG.info(
            "frame=%d latency=%.3fs clipped_rows=%d pred0=%s target0=%s",
            frame_index,
            policy.last_inference_seconds,
            clipped,
            prediction[0].tolist(),
            target[0].tolist(),
        )
        if args.max_predictions is not None and len(frame_indices) >= args.max_predictions:
            break

    predictions = np.stack(predicted_chunks)
    targets = np.stack(target_chunks)
    first_action_mae = np.mean(np.abs(predictions[:, 0] - targets[:, 0]), axis=0)
    report = {
        "checkpoint": str(policy.checkpoint_path),
        "checkpoint_source": policy.checkpoint_source,
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "episode_index": args.episode_index,
        "episode_bounds": [start, end],
        "predictions": len(frame_indices),
        "stride": args.stride,
        "first_action_mae": first_action_mae.tolist(),
        "mean_inference_seconds": float(np.mean(inference_seconds)),
        "total_clipped_rows": int(np.sum(clipped_rows)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frame_indices=np.asarray(frame_indices),
        predicted_chunks=predictions,
        target_chunks=targets,
        clipped_rows=np.asarray(clipped_rows),
        inference_seconds=np.asarray(inference_seconds),
        report_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved replay result: {args.output}")


if __name__ == "__main__":
    main()

