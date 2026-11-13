"""Deploy a Zadai RL-100 checkpoint through VLAHost's HTTP API."""

from __future__ import annotations

import argparse
import logging
import signal
import time

import numpy as np

from .zadai_rl100_policy import ZADAI_STATE_DIM, ZadaiRL100PolicyRunner
from .zadai_vlahost import ZadaiVLAHostClient


LOG = logging.getLogger(__name__)


def checked_actions(chunk: np.ndarray, count: int, max_abs_arm: float) -> np.ndarray:
    selected = np.asarray(chunk[:count], dtype=np.float32)
    if selected.shape != (count, ZADAI_STATE_DIM) or not np.isfinite(selected).all():
        raise RuntimeError("Invalid Zadai action chunk {}".format(selected.shape))
    if np.abs(selected[:, :14]).max() > max_abs_arm:
        raise RuntimeError("Arm target exceeds --max-abs-arm={}; refusing to send".format(max_abs_arm))
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--http-base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--checkpoint-source", choices=("auto", "ema", "model"), default="model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--actions-per-inference", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-abs-arm", type=float, default=6.5)
    parser.add_argument("--default-gripper-pos", type=float, default=0.0)
    parser.add_argument("--safety-stats-path")
    parser.add_argument("--action-clip-margin", type=float, default=0.05)
    parser.add_argument("--max-relative-target", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-real-robot", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.max_chunks <= 0 or args.actions_per_inference <= 0 or args.fps <= 0 or args.max_abs_arm <= 0:
        raise ValueError("max-chunks, actions-per-inference, fps and max-abs-arm must be positive")
    policy = ZadaiRL100PolicyRunner(args.checkpoint, args.device, args.checkpoint_source, not args.stochastic, args.num_inference_steps)
    if args.actions_per_inference > policy.n_action_steps:
        raise ValueError("actions-per-inference={} exceeds checkpoint horizon {}".format(args.actions_per_inference, policy.n_action_steps))
    if args.inspect_only:
        LOG.info("Checkpoint inspection completed without any robot connection")
        return 0
    if not args.dry_run and not args.allow_real_robot:
        raise RuntimeError("Refusing to send real-robot actions; use --dry-run or --allow-real-robot")
    client = ZadaiVLAHostClient(args.http_base_url, args.timeout, args.safety_stats_path, args.action_clip_margin, args.max_relative_target, args.default_gripper_pos)
    stopped = [False]
    def stop(_signum, _frame):
        stopped[0] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    client.connect()
    try:
        for index in range(args.max_chunks):
            if stopped[0]:
                break
            observation = client.get_observation()
            if index == 0:
                policy.reset(observation, args.seed)
            else:
                policy.observe(observation)
            actions = checked_actions(policy.predict_chunk(), args.actions_per_inference, args.max_abs_arm)
            LOG.info("chunk=%d inference=%.3fs first_action=%s", index + 1, policy.last_inference_seconds, actions[0].tolist())
            if args.dry_run:
                continue
            started = time.monotonic()
            client.send_action_chunk(actions, args.fps)
            delay = len(actions) / args.fps - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
