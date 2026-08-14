from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from act_rl.algorithms import IQLACTLearner
from act_rl.checkpoint import save_checkpoint
from act_rl.config import ACTRLConfig
from act_rl.datasets.lerobot_chunk import (
    LeRobotChunkDataset,
    load_outcomes,
    make_delta_timestamps,
    prepare_lerobot_batch,
)
from act_rl.online_store import OnlineNpzDataset
from act_rl.policies import LeRobotACTAdapter
from act_rl.types import concatenate_batches

LOG = logging.getLogger(__name__)


def _next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    p = argparse.ArgumentParser(description="Offline/online mixed IQL updates for ACT (no robot connection)")
    p.add_argument("--config", required=True)
    p.add_argument("--model-checkpoint", required=True, help="pretrained_model matching learner checkpoint")
    p.add_argument("--learner-checkpoint", required=True, help="step_XXXXXXXX.pt from offline training")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo-id", default="xlb/marvin_usb_act_success_dense_gripper")
    p.add_argument("--outcomes", required=True)
    p.add_argument("--exclude-episodes", nargs="*", type=int, default=[])
    p.add_argument("--online-replay", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-freq", type=int, default=50)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = ACTRLConfig.from_json(args.config)
    n_offline = round(cfg.batch_size * cfg.iql.offline_fraction)
    n_online = cfg.batch_size - n_offline
    if n_offline <= 0 or n_online <= 0:
        raise ValueError("online mixed training requires both offline and online samples per batch")

    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import get_policy_class, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(args.model_checkpoint)
    policy_cfg.device = args.device
    policy = get_policy_class("act").from_pretrained(args.model_checkpoint, config=policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=args.model_checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    delta = make_delta_timestamps(
        args.dataset_root, cfg.network.image_keys, cfg.network.chunk_size, cfg.network.executed_steps
    )
    outcomes = load_outcomes(args.outcomes)
    excluded = set(args.exclude_episodes)
    unknown = excluded.difference(outcomes)
    if unknown:
        raise ValueError(f"excluded episodes are absent from outcome manifest: {sorted(unknown)}")
    selected_episodes = sorted(set(outcomes).difference(excluded))
    base = LeRobotDataset(
        args.repo_id, root=args.dataset_root, episodes=selected_episodes,
        delta_timestamps=delta, return_uint8=False,
    )
    offline = LeRobotChunkDataset(
        base, cfg.network.image_keys, cfg.network.executed_steps, cfg.iql.gamma, outcomes
    )
    online = OnlineNpzDataset(args.online_replay, cfg.network.image_keys)
    if len(online) < n_online:
        raise ValueError(f"online replay has {len(online)} transitions, need at least {n_online}")
    offline_loader = DataLoader(offline, n_offline, shuffle=True, num_workers=args.num_workers, drop_last=True)
    online_loader = DataLoader(online, n_online, shuffle=True, num_workers=args.num_workers, drop_last=True)
    offline_iter, online_iter = iter(offline_loader), iter(online_loader)

    learner = IQLACTLearner(LeRobotACTAdapter(policy), cfg, args.device)
    state = torch.load(args.learner_checkpoint, map_location=args.device, weights_only=False)
    learner.load_state_dict(state, load_optimizers=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    initial_step = int(state.get("step", 0))
    for local_step in range(1, args.steps + 1):
        offline_raw, offline_iter = _next(offline_iter, offline_loader)
        online_raw, online_iter = _next(online_iter, online_loader)
        batch = concatenate_batches(
            prepare_lerobot_batch(offline_raw, preprocessor, cfg.network.image_keys),
            prepare_lerobot_batch(online_raw, preprocessor, cfg.network.image_keys),
        )
        metrics = learner.update(batch, update_actor=True)
        step = initial_step + local_step
        if local_step == 1 or local_step % args.log_freq == 0:
            LOG.info("step=%d offline=%d online=%d %s", step, n_offline, n_online,
                     " ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
        if local_step % args.save_freq == 0 or local_step == args.steps:
            save_checkpoint(output / "checkpoints" / f"step_{step:08d}.pt", learner, step, metrics)
            model_dir = output / "checkpoints" / f"step_{step:08d}" / "pretrained_model"
            learner.actor.policy.save_pretrained(model_dir)
            preprocessor.save_pretrained(model_dir)
            postprocessor.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
