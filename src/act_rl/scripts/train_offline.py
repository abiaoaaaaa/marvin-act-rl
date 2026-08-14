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
from act_rl.policies import LeRobotACTAdapter

LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline IQL + advantage-weighted direct ACT fine-tuning")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo-id", default="xlb/marvin_usb_act_success_dense_gripper")
    p.add_argument("--outcomes", required=True)
    p.add_argument(
        "--exclude-episodes",
        nargs="*",
        type=int,
        default=[],
        help="Output episode indices to exclude after conversion, e.g. --exclude-episodes 556.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--critic-warmup-steps", type=int, default=1000)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-freq", type=int, default=50)
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = ACTRLConfig.from_json(args.config)
    torch.manual_seed(cfg.seed)
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import get_policy_class, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    if policy_cfg.type != "act":
        raise ValueError(f"checkpoint must be ACT, got {policy_cfg.type}")
    if policy_cfg.chunk_size != cfg.network.chunk_size:
        raise ValueError(f"chunk mismatch: checkpoint={policy_cfg.chunk_size}, RL={cfg.network.chunk_size}")
    policy_cfg.device = args.device
    policy = get_policy_class("act").from_pretrained(args.checkpoint, config=policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    delta = make_delta_timestamps(
        args.dataset_root, cfg.network.image_keys, cfg.network.chunk_size, cfg.network.executed_steps
    )
    # Match LeRobot BC training: decoded images are float tensors before the
    # checkpoint preprocessor.  uint8 here would also coerce normalization
    # statistics to uint8 in the current processor implementation.
    outcomes = load_outcomes(args.outcomes)
    excluded = set(args.exclude_episodes)
    unknown = excluded.difference(outcomes)
    if unknown:
        raise ValueError(f"excluded episodes are absent from outcome manifest: {sorted(unknown)}")
    selected_episodes = sorted(set(outcomes).difference(excluded))
    base = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=selected_episodes,
        delta_timestamps=delta,
        return_uint8=False,
    )
    if base.num_episodes != len(selected_episodes):
        raise ValueError(f"LeRobot selected {base.num_episodes} episodes, expected {len(selected_episodes)}")
    if excluded:
        LOG.warning("Excluded converted episode indices: %s", sorted(excluded))
    dataset = LeRobotChunkDataset(base, cfg.network.image_keys, cfg.network.executed_steps, cfg.iql.gamma, outcomes)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=args.device.startswith("cuda"), drop_last=True)
    learner = IQLACTLearner(LeRobotACTAdapter(policy), cfg, args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        batch = prepare_lerobot_batch(raw, preprocessor, cfg.network.image_keys)
        metrics = learner.update(batch, update_actor=step > args.critic_warmup_steps)
        if step == 1 or step % args.log_freq == 0:
            LOG.info("step=%d %s", step, " ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
        if step % args.save_freq == 0 or step == args.steps:
            save_checkpoint(output / "checkpoints" / f"step_{step:08d}.pt", learner, step, metrics)
            model_dir = output / "checkpoints" / f"step_{step:08d}" / "pretrained_model"
            learner.actor.policy.save_pretrained(model_dir)
            preprocessor.save_pretrained(model_dir)
            postprocessor.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
