from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from act_rl.online_store import OnlineEpisodeWriter
from lerobot_robot_marvin.act_rollout import ACTPolicyRunner, sanitize_policy_action, validate_bridge_preflight
from lerobot_robot_marvin.client import MarvinBridgeClient
from lerobot_robot_marvin.constants import CAMERA_KEYS

LOG = logging.getLogger(__name__)


def _raw_chunk(runner: ACTPolicyRunner, observation: dict) -> np.ndarray:
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference
    frame = {"observation.state": np.asarray(observation["state"], np.float32)}
    for key in CAMERA_KEYS:
        frame[f"observation.images.{key}"] = observation[key]
    frame = prepare_observation_for_inference(frame, runner.device, runner.task, "marvin")
    frame = runner.preprocessor(frame)
    with torch.inference_mode():
        chunk = runner.postprocessor(runner.policy.predict_action_chunk(frame)).squeeze(0).cpu().numpy()
    return np.stack([sanitize_policy_action(action)[0] for action in chunk])


def main() -> None:
    p = argparse.ArgumentParser(description="Collect versioned ACT chunk transitions through Marvin Bridge")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-decisions", type=int, default=20)
    p.add_argument("--executed-steps", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--allow-real-robot", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runner = ACTPolicyRunner(args.checkpoint, args.device, "Insert the USB plug")
    if not 1 <= args.executed_steps <= runner.policy.config.chunk_size:
        raise ValueError("executed_steps must be in [1, checkpoint chunk_size]")
    client = MarvinBridgeClient(args.host, args.port)
    status = client.status()
    validate_bridge_preflight(status, args.allow_real_robot)
    model_image_keys = tuple(f"observation.images.{key}" for key in CAMERA_KEYS)
    writer = OnlineEpisodeWriter(args.output_dir, model_image_keys, runner.policy.config.chunk_size, 5)
    first_episode_id = writer.next_episode_id()
    client.connect()
    try:
        for episode_offset in range(args.episodes):
            episode = first_episode_id + episode_offset
            observation, _ = client.reset()
            runner.reset()
            records = []
            terminal = False
            for _decision in range(args.max_decisions):
                chunk = _raw_chunk(runner, observation)
                reward_sum, actual_h, next_observation = 0.0, 0, observation
                for action in chunk[: args.executed_steps]:
                    result = client.step(action)
                    reward_sum += (args.gamma**actual_h) * float(result.get("reward", 0.0))
                    actual_h += 1
                    next_observation = result["observation"]
                    terminal = bool(result.get("terminated", result.get("done", False)))
                    if terminal:
                        break
                records.append({
                    "observation": observation, "next_observation": next_observation,
                    "action_chunk": chunk, "action_is_pad": np.zeros(len(chunk), dtype=bool),
                    "reward": reward_sum, "discount": args.gamma**actual_h, "terminated": terminal,
                })
                observation = next_observation
                if terminal:
                    break
            path = writer.write(episode, records, {"checkpoint": str(Path(args.checkpoint).resolve()),
                                "executed_steps": args.executed_steps, "terminal": terminal})
            LOG.info("wrote episode=%d decisions=%d terminal=%s path=%s", episode, len(records), terminal, path)
    except Exception:
        client.best_effort_stop("ACT RL collector failure")
        raise
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
