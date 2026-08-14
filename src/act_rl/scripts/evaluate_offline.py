from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from act_rl.config import ACTRLConfig
from act_rl.datasets.lerobot_chunk import load_outcomes
from act_rl.networks import ObservationEncoder, TwinChunkQ, ValueNetwork


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only ACT-RL checkpoint evaluation on labeled offline data")
    p.add_argument("--config", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--learner-checkpoint", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--outcomes", required=True)
    p.add_argument("--exclude-episodes", nargs="*", type=int, default=[])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output", type=Path)
    return p


def _load_policy(checkpoint: str, device: str):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    policy = get_policy_class("act").from_pretrained(checkpoint, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def _select_indices(dataset, max_samples: int) -> list[int]:
    if max_samples <= 0 or max_samples >= len(dataset):
        return list(range(len(dataset)))
    return np.linspace(0, len(dataset) - 1, max_samples, dtype=np.int64).tolist()


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    args = parser().parse_args()
    cfg = ACTRLConfig.from_json(args.config)
    outcomes = load_outcomes(args.outcomes)
    selected_episodes = sorted(set(outcomes).difference(args.exclude_episodes))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    delta = {"action": [i / 10 for i in range(cfg.network.chunk_size)]}
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=selected_episodes,
        delta_timestamps=delta,
        return_uint8=False,
    )
    subset = Subset(dataset, _select_indices(dataset, args.max_samples))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model, pre, post = _load_policy(args.model_checkpoint, args.device)
    base, base_pre, base_post = _load_policy(args.base_checkpoint, args.device)

    state = torch.load(args.learner_checkpoint, map_location=args.device, weights_only=False)
    encoder = ObservationEncoder(
        cfg.network.state_dim, cfg.network.use_images, cfg.network.image_feature_dim
    ).to(args.device).eval()
    q = TwinChunkQ(
        encoder.output_dim, cfg.network.critic_horizon, cfg.network.action_dim, cfg.network.hidden_dims
    ).to(args.device).eval()
    value = ValueNetwork(encoder.output_dim, cfg.network.hidden_dims).to(args.device).eval()
    encoder.load_state_dict(state["encoder"], strict=True)
    q.load_state_dict(state["q"], strict=True)
    value.load_state_dict(state["value"], strict=True)

    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    image_keys = cfg.network.image_keys
    with torch.inference_mode():
        for raw in loader:
            episode_ids = raw["episode_index"].tolist()
            labels = ["success" if outcomes[int(ep)]["success"] else "failure" for ep in episode_ids]
            current = {"observation.state": raw["observation.state"], "action": raw["action"],
                       "action_is_pad": raw["action_is_pad"]}
            for key in image_keys:
                current[key] = raw[key]
            model_batch = pre(dict(current))
            base_batch = base_pre(dict(current))
            amp = torch.autocast("cuda") if str(args.device).startswith("cuda") else nullcontext()
            with amp:
                model_chunk = model.predict_action_chunk(model_batch)
                base_chunk = base.predict_action_chunk(base_batch)
            target = model_batch["action"]
            valid = (~model_batch["action_is_pad"]).unsqueeze(-1)
            denom = valid.sum((1, 2)).clamp_min(1) * target.shape[-1]
            model_l1 = ((model_chunk - target).abs() * valid).sum((1, 2)) / denom
            base_l1 = ((base_chunk - base_batch["action"]).abs() * valid).sum((1, 2)) / denom
            deviation = ((model_chunk - base_chunk).abs() * valid).sum((1, 2)) / denom
            obs = encoder(model_batch["observation.state"], {k: model_batch[k] for k in image_keys})
            q1, q2 = q(obs, target[:, : cfg.network.critic_horizon])
            q_min, v = torch.minimum(q1, q2), value(obs)
            # Postprocessors return physical normalized Marvin actions in this project.
            model_physical = post(model_chunk[:, 0]).detach()
            base_physical = base_post(base_chunk[:, 0]).detach()
            model_oob = (model_physical.abs() > 1.0).any(-1).float()
            base_oob = (base_physical.abs() > 1.0).any(-1).float()
            model_excess = (model_physical.abs() - 1.0).clamp_min(0).amax(-1)
            base_excess = (base_physical.abs() - 1.0).clamp_min(0).amax(-1)
            for i, label in enumerate(labels):
                for group in ("all", label):
                    bucket = metrics[group]
                    bucket["model_l1"].append(float(model_l1[i]))
                    bucket["base_l1"].append(float(base_l1[i]))
                    bucket["model_base_l1"].append(float(deviation[i]))
                    bucket["q_data"].append(float(q_min[i]))
                    bucket["value"].append(float(v[i]))
                    bucket["advantage"].append(float(q_min[i] - v[i]))
                    bucket["model_first_action_oob"].append(float(model_oob[i]))
                    bucket["base_first_action_oob"].append(float(base_oob[i]))
                    bucket["model_first_action_max_excess"].append(float(model_excess[i]))
                    bucket["base_first_action_max_excess"].append(float(base_excess[i]))

    report = {
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "model_checkpoint": str(Path(args.model_checkpoint).resolve()),
        "learner_checkpoint": str(Path(args.learner_checkpoint).resolve()),
        "samples": len(subset),
        "note": "Diagnostic replay on training-distribution data; not an unbiased hold-out or success rate.",
        "groups": {
            group: {"count": len(values["model_l1"]), **{key: _mean(items) for key, items in values.items()}}
            for group, values in metrics.items()
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
