from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np

from act_rl.online_store import OnlineEpisodeWriter
from lerobot_robot_marvin.act_rollout import ACTPolicyRunner, sanitize_policy_action, validate_bridge_preflight
from lerobot_robot_marvin.client import MarvinBridgeClient
from lerobot_robot_marvin.constants import CAMERA_KEYS

LOG = logging.getLogger(__name__)


def _checkpoint_sha256(checkpoint: str | Path) -> str:
    model = Path(checkpoint) / "model.safetensors"
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


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
    return np.asarray(chunk, dtype=np.float32)


def _validate_fixed_policy_replay(output_dir: str | Path, policy_sha256: str) -> None:
    """Prevent one collection round from silently mixing policy versions."""
    versions: set[str] = set()
    for sidecar in Path(output_dir).glob("episode_*.json"):
        import json

        metadata = json.loads(sidecar.read_text())
        versions.add(str(metadata.get("policy_sha256", "unknown")))
    if versions and versions != {policy_sha256}:
        raise ValueError(
            "output-dir already contains a different or unknown policy version; "
            f"existing={sorted(versions)}, requested={policy_sha256}. Use a new round directory."
        )


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
    policy_sha256 = _checkpoint_sha256(args.checkpoint)
    _validate_fixed_policy_replay(args.output_dir, policy_sha256)
    first_episode_id = writer.next_episode_id()
    client.connect()
    try:
        for episode_offset in range(args.episodes):
            episode = first_episode_id + episode_offset
            observation, _ = client.reset()
            runner.reset()
            records = []
            terminated = truncated = False
            final_info = {}
            for _decision in range(args.max_decisions):
                policy_chunk = _raw_chunk(runner, observation)
                command_chunk = np.stack([sanitize_policy_action(action)[0] for action in policy_chunk])
                behavior_chunk = command_chunk.copy()
                critic_chunk = np.zeros_like(policy_chunk)
                action_is_pad = np.ones(len(policy_chunk), dtype=bool)
                critic_action_is_pad = np.ones(len(policy_chunk), dtype=bool)
                intervention_mask = np.zeros(len(policy_chunk), dtype=bool)
                reward_sum, actual_h, next_observation = 0.0, 0, observation
                intervened = safety_abort = success = False
                for action in command_chunk[: args.executed_steps]:
                    result = client.step(action)
                    reward_sum += (args.gamma**actual_h) * float(result.get("reward", 0.0))
                    actual_action = np.asarray(result.get("actual_action", action), dtype=np.float32)
                    behavior_chunk[actual_h] = actual_action
                    critic_chunk[actual_h] = actual_action
                    action_is_pad[actual_h] = False
                    critic_action_is_pad[actual_h] = False
                    info = dict(result.get("info", {}))
                    step_intervened = not np.allclose(actual_action, action, atol=1e-6, rtol=0.0)
                    step_intervened = step_intervened or "intervene_action" in info
                    intervention_mask[actual_h] = step_intervened
                    intervened = intervened or step_intervened
                    safety_abort = safety_abort or bool(info.get("safety_abort", False))
                    success = success or bool(result.get("succeed", info.get("succeed", False)))
                    actual_h += 1
                    next_observation = result["observation"]
                    terminated = bool(result.get("terminated", result.get("done", False)))
                    truncated = bool(result.get("truncated", False))
                    if safety_abort:
                        terminated, truncated = True, False
                    final_info = info
                    if terminated or truncated:
                        break
                records.append({
                    "observation": observation, "next_observation": next_observation,
                    # ACT learns from the executed behavior prefix.  The full raw
                    # policy chunk is retained separately for strategy analysis.
                    "action_chunk": behavior_chunk, "policy_action_chunk": policy_chunk,
                    "command_action_chunk": command_chunk,
                    "critic_action_chunk": critic_chunk, "actual_action_chunk": critic_chunk.copy(),
                    "action_is_pad": action_is_pad,
                    "critic_action_is_pad": critic_action_is_pad, "executed_steps": actual_h,
                    "executed_mask": ~critic_action_is_pad, "intervention_mask": intervention_mask,
                    "reward": reward_sum, "discount": args.gamma**actual_h,
                    "terminated": terminated, "truncated": truncated,
                    "success": success, "intervention": intervened, "safety_abort": safety_abort,
                })
                observation = next_observation
                if terminated or truncated:
                    break
            if records and not (terminated or truncated):
                # Local collection horizon is a time-limit truncation.  Control
                # stops, while the learner may still bootstrap V(next_obs).
                truncated = True
                records[-1]["truncated"] = True
            episode_success = bool(records and terminated and records[-1]["success"])
            episode_failure = bool(records and terminated and not episode_success)
            for record in records:
                record["success"] = episode_success
            path = writer.write(episode, records, {
                "checkpoint": str(Path(args.checkpoint).resolve()), "policy_sha256": policy_sha256,
                "executed_steps": args.executed_steps, "terminated": terminated, "truncated": truncated,
                "success": episode_success, "failure": episode_failure,
                "interventions": sum(bool(record["intervention"]) for record in records),
                "final_info": _json_safe(final_info),
            })
            LOG.info(
                "wrote episode=%d decisions=%d terminated=%s truncated=%s success=%s path=%s",
                episode, len(records), terminated, truncated, episode_success, path,
            )
    except Exception:
        client.best_effort_stop("ACT RL collector failure")
        raise
    finally:
        try:
            client.disconnect()
        except Exception as exc:
            LOG.warning("Bridge disconnect failed after collection: %s", exc)


if __name__ == "__main__":
    main()
