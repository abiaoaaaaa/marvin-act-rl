#!/usr/bin/env python3
"""Offline ACT-vs-demonstration analysis for the dense human40 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


CAMERAS = ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2")
ACTION_NAMES = ("dx", "dy", "dz", "drz", "gripper")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from resfit.lerobot.policies.act.modeling_act import ACTPolicy

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(args.dataset_name, root=args.dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ACTPolicy.from_pretrained(args.checkpoint, dataset_stats=dataset.meta.stats).to(device)
    policy.eval()

    refs: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    batch_refs: list[np.ndarray] = []
    batch_eps: list[np.ndarray] = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        action = batch["action"].float()
        episode = batch["episode_index"].numpy().astype(np.int64)
        model_batch = {"observation.state": batch["observation.state"].to(device)}
        for key in CAMERAS:
            model_batch[f"observation.images.{key}"] = batch[f"observation.images.{key}"].to(device)
        # Each batch is independent: select_action returns the first action of
        # a fresh 20-action queue for every sample in the batch.
        policy.reset()
        with torch.inference_mode():
            prediction = policy.select_action(model_batch).detach().cpu()
        refs.append(action.numpy())
        preds.append(prediction.numpy())
        episodes.append(episode)

    ref = np.concatenate(refs)
    pred = np.concatenate(preds)
    ep = np.concatenate(episodes)
    err = pred - ref
    abs_err = np.abs(err)
    sq_err = err**2

    split = {"train_episodes": [], "eval_episodes": []}
    split_path = args.split_json
    if split_path is None:
        candidate = args.checkpoint.parent.parent / "dataset_split.json"
        if candidate.is_file():
            split_path = candidate
    if split_path and split_path.is_file():
        split = json.loads(split_path.read_text())
    groups = {"all": np.ones(len(ep), dtype=bool)}
    if split["train_episodes"]:
        groups["train"] = np.isin(ep, split["train_episodes"])
        groups["eval"] = np.isin(ep, split["eval_episodes"])

    rows = {}
    for name, mask in groups.items():
        rows[name] = {
            "samples": int(mask.sum()),
            "reference_mean": ref[mask].mean(0).tolist(),
            "prediction_mean": pred[mask].mean(0).tolist(),
            "reference_variance": ref[mask].var(0).tolist(),
            "prediction_variance": pred[mask].var(0).tolist(),
            "reference_sample_variance_ddof1": ref[mask].var(0, ddof=1).tolist(),
            "prediction_sample_variance_ddof1": pred[mask].var(0, ddof=1).tolist(),
            "reference_std": ref[mask].std(0).tolist(),
            "prediction_std": pred[mask].std(0).tolist(),
            "bias_prediction_minus_reference": err[mask].mean(0).tolist(),
            "mae": abs_err[mask].mean(0).tolist(),
            "rmse": np.sqrt(sq_err[mask].mean(0)).tolist(),
            "overall_mae": float(abs_err[mask].mean()),
            "overall_rmse": float(np.sqrt(sq_err[mask].mean())),
        }

    # Integrate the first four action dimensions within each episode to obtain
    # an action-induced trajectory; this is a diagnostic, not robot pose error.
    traj_errors = []
    endpoint_errors = []
    for episode_id in np.unique(ep):
        mask = ep == episode_id
        ref_path = np.cumsum(ref[mask, :4], axis=0)
        pred_path = np.cumsum(pred[mask, :4], axis=0)
        distance = np.linalg.norm(pred_path - ref_path, axis=1)
        traj_errors.extend(distance.tolist())
        endpoint_errors.append(float(distance[-1]))
    rows["trajectory"] = {
        "integrated_action_path_mean_l2": float(np.mean(traj_errors)),
        "integrated_action_path_p95_l2": float(np.percentile(traj_errors, 95)),
        "episode_endpoint_mean_l2": float(np.mean(endpoint_errors)),
        "episode_endpoint_p95_l2": float(np.percentile(endpoint_errors, 95)),
        "episodes": len(endpoint_errors),
    }
    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "chunk_size": int(policy.config.chunk_size),
        "comparison": "first action from a fresh ACT queue vs reference action at each frame",
        "groups": rows,
    }
    (args.output_dir / "act_offline_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "act_offline_source.npz", reference=ref, prediction=pred, episode=ep, error=err)

    labels = list(ACTION_NAMES)
    all_stats = rows["all"]
    x = np.arange(5)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].bar(x - 0.18, all_stats["reference_mean"], 0.36, label="reference")
    axes[0, 0].bar(x + 0.18, all_stats["prediction_mean"], 0.36, label="ACT prediction")
    axes[0, 0].set_title("Action mean")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].legend()
    axes[0, 1].bar(x - 0.18, all_stats["reference_variance"], 0.36, label="reference")
    axes[0, 1].bar(x + 0.18, all_stats["prediction_variance"], 0.36, label="ACT prediction")
    axes[0, 1].set_title("Action variance")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].legend()
    axes[1, 0].bar(x, all_stats["mae"], color="#2a9d8f")
    axes[1, 0].set_title(f"Per-dimension MAE (overall={all_stats['overall_mae']:.4f})")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 1].hist(np.linalg.norm(err[:, :4], axis=1), bins=40, color="#457b9d", alpha=0.85)
    axes[1, 1].set_title("Per-frame trajectory-action error ||prediction-reference||")
    axes[1, 1].set_xlabel("L2 error")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Dense human40 ACT offline analysis")
    fig.savefig(args.output_dir / "act_offline_analysis.png", dpi=180)
    fig.savefig(args.output_dir / "act_offline_analysis.svg")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
