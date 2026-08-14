#!/usr/bin/env python3
"""Merge RL-100 offline actor files into a deployment-compatible checkpoint."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import dill
import hydra
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--action-steps", type=int, required=True)
    args = parser.parse_args()
    if args.action_steps <= 0:
        parser.error("--action-steps must be positive")
    return args


def main():
    args = parse_args()
    if args.output_checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_checkpoint}")
    model_path = args.policy_dir / "model.pt"
    encoder_path = args.policy_dir / "encoder.pt"
    for path in (args.base_checkpoint, model_path, encoder_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = torch.load(args.base_checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = copy.deepcopy(base["cfg"])
    cfg.n_action_steps = args.action_steps
    cfg.policy.n_action_steps = args.action_steps
    cfg.chunk_as_single_action = True
    cfg.policy.chunk_as_single_action = True
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(base["state_dicts"]["model"], strict=True)
    policy.model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    policy.obs_encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"), strict=True)

    payload = {
        "cfg": cfg,
        "state_dicts": {"model": policy.state_dict()},
        "pickles": {},
        "marvin_offline_export": {
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "policy_dir": str(args.policy_dir.resolve()),
            "action_steps": args.action_steps,
        },
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_checkpoint.open("wb"), pickle_module=dill)
    print(f"Created deployment checkpoint: {args.output_checkpoint.resolve()}")


if __name__ == "__main__":
    main()
