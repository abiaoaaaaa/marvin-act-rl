"""Replay absolute Zadai actions from a LeRobot v3 episode through VLAHost."""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

import numpy as np

from .zadai_rl100_rollout import checked_actions
from .zadai_vlahost import ZadaiVLAHostClient


LOG = logging.getLogger(__name__)


def load_episode_actions(dataset_root: str, episode_index: int, start_frame: int) -> np.ndarray:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("LeRobot replay requires pyarrow") from exc
    files = sorted(Path(dataset_root).expanduser().glob("data/*/*.parquet"))
    if not files:
        raise FileNotFoundError("No LeRobot parquet files under {}/data".format(dataset_root))
    tables = []
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action"])
        table = table.filter(pc.equal(table["episode_index"], episode_index))
        if len(table):
            tables.append(table)
    if not tables:
        raise IndexError("Episode {} was not found".format(episode_index))
    table = tables[0] if len(tables) == 1 else __import__("pyarrow").concat_tables(tables)
    table = table.sort_by([("frame_index", "ascending")])
    action = np.asarray(table["action"].combine_chunks().values).reshape(len(table), -1).astype(np.float32)
    if action.shape[1] != 16 or not np.isfinite(action).all():
        raise RuntimeError("Expected finite [T,16] action, got {}".format(action.shape))
    if not 0 <= start_frame < len(action):
        raise IndexError("start-frame must be in [0, {}]".format(len(action) - 1))
    return action[start_frame:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--actions-per-chunk", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--http-base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-abs-arm", type=float, default=6.5)
    parser.add_argument("--safety-stats-path")
    parser.add_argument("--action-clip-margin", type=float, default=0.05)
    parser.add_argument("--max-relative-target", type=float, default=0.08)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-real-robot", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if min(args.actions_per_chunk, args.max_chunks, args.fps, args.max_abs_arm) <= 0:
        raise ValueError("actions-per-chunk, max-chunks, fps and max-abs-arm must be positive")
    actions = load_episode_actions(args.dataset_root, args.episode_index, args.start_frame)
    if not args.dry_run and not args.allow_real_robot:
        raise RuntimeError("Refusing to send real-robot actions; use --dry-run or --allow-real-robot")
    client = ZadaiVLAHostClient(args.http_base_url, args.timeout, args.safety_stats_path,
                                args.action_clip_margin, args.max_relative_target, 0.0)
    stopped = [False]
    def stop(_signum, _frame):
        stopped[0] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    client.connect()
    try:
        for chunk_index, start in enumerate(range(0, len(actions), args.actions_per_chunk), 1):
            if stopped[0] or chunk_index > args.max_chunks:
                break
            observation = client.get_observation()  # Validate live state and all three image views.
            chunk = checked_actions(actions[start:start + args.actions_per_chunk], min(args.actions_per_chunk, len(actions) - start), args.max_abs_arm)
            LOG.info("episode=%d frames=%d..%d state0=%.4f first_action=%s", args.episode_index,
                     args.start_frame + start, args.start_frame + start + len(chunk) - 1,
                     observation["state"][0], chunk[0].tolist())
            if args.dry_run:
                continue
            started = time.monotonic()
            client.send_action_chunk(chunk, args.fps)
            delay = len(chunk) / args.fps - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
