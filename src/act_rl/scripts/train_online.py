from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from act_rl.algorithms import IQLACTLearner
from act_rl.scripts.audit_online_replay import audit_replay
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


def resolve_offline_fraction(cfg: ACTRLConfig, online_size: int) -> float:
    online = cfg.online
    if online_size < online.low_online_threshold:
        return online.offline_fraction_low_data
    if online_size < online.high_online_threshold:
        return online.offline_fraction_mid_data
    return online.offline_fraction_high_data


def validate_checkpoint_pair(policy, state: dict, cfg: ACTRLConfig) -> None:
    saved_network = state.get("config", {}).get("network")
    if saved_network is not None and json.dumps(saved_network, sort_keys=True) != json.dumps(
        cfg.to_dict()["network"], sort_keys=True
    ):
        raise ValueError("learner checkpoint network config does not match online config")
    actor_state = state.get("actor")
    if actor_state is None:
        raise ValueError("learner checkpoint has no actor state")
    model_state = policy.state_dict()
    if model_state.keys() != actor_state.keys():
        raise ValueError("model-checkpoint and learner-checkpoint actor keys do not match")
    mismatched = [
        key for key in model_state
        if model_state[key].shape != actor_state[key].shape
        or not torch.equal(model_state[key].detach().cpu(), actor_state[key].detach().cpu())
    ]
    if mismatched:
        raise ValueError(
            "model-checkpoint does not match learner-checkpoint actor; "
            f"first mismatched keys: {mismatched[:3]}"
        )


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
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--save-freq", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-freq", type=int, default=50)
    p.add_argument("--fixed-offline-fraction", type=float)
    p.add_argument("--critic-to-actor-ratio", type=int)
    p.add_argument("--allow-high-replay-reuse", action="store_true")
    p.add_argument("--allow-legacy-replay", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = ACTRLConfig.from_json(args.config)
    if not args.allow_legacy_replay:
        audit = audit_replay(args.online_replay, cfg)
        if not audit["ok"]:
            preview = "; ".join(audit["errors"][:5])
            raise ValueError(f"online replay audit failed before training: {preview}")
        LOG.info("strict replay audit passed: %s", audit["totals"])
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import get_policy_class, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(args.model_checkpoint)
    if policy_cfg.type != "act":
        raise ValueError(f"model-checkpoint must be ACT, got {policy_cfg.type}")
    if policy_cfg.chunk_size != cfg.network.chunk_size:
        raise ValueError(f"chunk mismatch: checkpoint={policy_cfg.chunk_size}, online={cfg.network.chunk_size}")
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
    online = OnlineNpzDataset(
        args.online_replay, cfg.network.image_keys,
        require_schema_version=None if args.allow_legacy_replay else 2,
    )
    offline_fraction = (
        float(args.fixed_offline_fraction)
        if args.fixed_offline_fraction is not None else resolve_offline_fraction(cfg, len(online))
    )
    if not 0.0 < offline_fraction < 1.0:
        raise ValueError("online mixed training requires offline fraction strictly between 0 and 1")
    n_offline = round(cfg.batch_size * offline_fraction)
    n_online = cfg.batch_size - n_offline
    if n_offline <= 0 or n_online <= 0:
        raise ValueError("batch_size is too small for the selected offline/online fraction")
    if len(online) < n_online:
        raise ValueError(f"online replay has {len(online)} transitions, need at least {n_online}")
    max_safe_updates = max(1, int(len(online) * cfg.online.max_updates_per_online_transition))
    if args.steps > max_safe_updates and not args.allow_high_replay_reuse:
        raise ValueError(
            f"requested {args.steps} updates for {len(online)} online transitions; "
            f"stage-1 safety limit is {max_safe_updates}. Collect more data, reduce --steps, "
            "or explicitly pass --allow-high-replay-reuse."
        )
    offline_loader = DataLoader(offline, n_offline, shuffle=True, num_workers=args.num_workers, drop_last=True)
    online_weights = online.sampling_weights(
        recent_ratio=cfg.online.recent_ratio, recent_window=cfg.online.recent_window,
        balance_outcomes=cfg.online.balance_outcomes,
        intervention_weight=cfg.online.intervention_weight,
    )
    online_sampler = WeightedRandomSampler(
        online_weights, num_samples=max(len(online), n_online), replacement=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    online_loader = DataLoader(
        online, n_online, sampler=online_sampler, num_workers=args.num_workers, drop_last=True,
    )
    offline_iter, online_iter = iter(offline_loader), iter(online_loader)

    state = torch.load(args.learner_checkpoint, map_location=args.device, weights_only=False)
    validate_checkpoint_pair(policy, state, cfg)
    learner = IQLACTLearner(LeRobotACTAdapter(policy), cfg, args.device)
    learner.load_state_dict(state, load_optimizers=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    actor_ratio = int(args.critic_to_actor_ratio or cfg.online.critic_to_actor_ratio)
    if actor_ratio < 1:
        raise ValueError("critic-to-actor ratio must be >= 1")
    resolved = cfg.to_dict()
    resolved["online_runtime"] = {
        "online_transitions": len(online), "offline_fraction": offline_fraction,
        "offline_batch": n_offline, "online_batch": n_online,
        "critic_to_actor_ratio": actor_ratio, "requested_updates": args.steps,
    }
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2))
    LOG.info(
        "stage-1 replay online=%d offline_fraction=%.2f batch=%d+%d critic_to_actor=%d",
        len(online), offline_fraction, n_offline, n_online, actor_ratio,
    )
    initial_step = int(state.get("step", 0))
    for local_step in range(1, args.steps + 1):
        offline_raw, offline_iter = _next(offline_iter, offline_loader)
        online_raw, online_iter = _next(online_iter, online_loader)
        batch = concatenate_batches(
            prepare_lerobot_batch(offline_raw, preprocessor, cfg.network.image_keys),
            prepare_lerobot_batch(online_raw, preprocessor, cfg.network.image_keys),
        )
        update_actor = local_step % actor_ratio == 0
        metrics = learner.update(batch, update_actor=update_actor)
        step = initial_step + local_step
        if local_step == 1 or local_step % args.log_freq == 0:
            LOG.info("step=%d offline=%d online=%d actor_update=%s %s", step, n_offline, n_online, update_actor,
                     " ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
        if local_step % args.save_freq == 0 or local_step == args.steps:
            save_checkpoint(output / "checkpoints" / f"step_{step:08d}.pt", learner, step, metrics)
            model_dir = output / "checkpoints" / f"step_{step:08d}" / "pretrained_model"
            learner.actor.policy.save_pretrained(model_dir)
            preprocessor.save_pretrained(model_dir)
            postprocessor.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
