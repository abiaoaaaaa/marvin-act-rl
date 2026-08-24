from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from act_rl.config import ACTRLConfig
from act_rl.online_store import REPLAY_SCHEMA_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit stage-1 ACT online replay before any learner update")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--allow-legacy-replay", action="store_true")
    return parser


def audit_replay(
    replay: str | Path,
    cfg: ACTRLConfig,
    allow_legacy_replay: bool = False,
) -> dict:
    root = Path(replay)
    files = sorted(root.glob("episode_*.npz"))
    errors: list[str] = []
    totals = {
        "episodes": len(files), "transitions": 0, "success_episodes": 0,
        "failure_episodes": 0, "truncated_episodes": 0, "intervention_transitions": 0,
    }
    policy_versions: set[str] = set()
    episodes = []
    for path in files:
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            errors.append(f"{path.name}: missing JSON sidecar")
            continue
        metadata = json.loads(sidecar.read_text())
        schema = int(metadata.get("schema_version", 1))
        if schema < REPLAY_SCHEMA_VERSION and not allow_legacy_replay:
            errors.append(f"{path.name}: schema v{schema}, require v{REPLAY_SCHEMA_VERSION}")
        policy_versions.add(str(metadata.get("policy_sha256", "unknown")))
        try:
            with np.load(path, allow_pickle=False) as data:
                required = (
                    "state", "next_state", "action", "policy_action", "command_action",
                    "critic_action", "actual_action",
                    "reward", "discount", "terminated", "truncated", "success",
                    "intervention", "executed_steps", "action_is_pad", "critic_action_is_pad",
                    "executed_mask", "intervention_mask",
                    *(key for image_key in cfg.network.image_keys for key in (image_key, f"next.{image_key}")),
                )
                missing = [key for key in required if key not in data]
                if missing:
                    errors.append(f"{path.name}: missing arrays {missing}")
                    continue
                count = len(data["reward"])
                totals["transitions"] += count
                expected_shapes = {
                    "state": (count, cfg.network.state_dim),
                    "next_state": (count, cfg.network.state_dim),
                    "action": (count, cfg.network.chunk_size, cfg.network.action_dim),
                    "policy_action": (count, cfg.network.chunk_size, cfg.network.action_dim),
                    "command_action": (count, cfg.network.chunk_size, cfg.network.action_dim),
                    "critic_action": (count, cfg.network.chunk_size, cfg.network.action_dim),
                    "actual_action": (count, cfg.network.chunk_size, cfg.network.action_dim),
                    "action_is_pad": (count, cfg.network.chunk_size),
                    "critic_action_is_pad": (count, cfg.network.chunk_size),
                    "executed_mask": (count, cfg.network.chunk_size),
                    "intervention_mask": (count, cfg.network.chunk_size),
                }
                for key, shape in expected_shapes.items():
                    if data[key].shape != shape:
                        errors.append(f"{path.name}: {key} shape={data[key].shape}, expected={shape}")
                for image_key in cfg.network.image_keys:
                    current_shape = data[image_key].shape
                    next_shape = data[f"next.{image_key}"].shape
                    if len(current_shape) != 4 or current_shape[0] != count or current_shape[-1] != 3:
                        errors.append(f"{path.name}: invalid HWC image batch {image_key}={current_shape}")
                    if next_shape != current_shape:
                        errors.append(
                            f"{path.name}: current/next image shape mismatch for {image_key}: "
                            f"{current_shape} vs {next_shape}"
                        )
                for key in ("state", "next_state", "action", "policy_action", "command_action",
                            "critic_action", "actual_action", "reward", "discount"):
                    if not np.isfinite(data[key]).all():
                        errors.append(f"{path.name}: non-finite values in {key}")
                if np.any((data["discount"] < 0) | (data["discount"] > 1)):
                    errors.append(f"{path.name}: discount outside [0,1]")
                if np.max(np.abs(data["command_action"]), initial=0.0) > 1.00001:
                    errors.append(f"{path.name}: command action outside physical [-1,1]")
                executed = data["executed_steps"].astype(int)
                if np.any((executed < 1) | (executed > cfg.network.executed_steps)):
                    errors.append(f"{path.name}: executed_steps outside [1,{cfg.network.executed_steps}]")
                for row, actual_h in enumerate(executed):
                    expected_pad = np.arange(cfg.network.chunk_size) >= actual_h
                    if not np.array_equal(data["action_is_pad"][row], expected_pad):
                        errors.append(f"{path.name}[{row}]: action_is_pad disagrees with executed_steps={actual_h}")
                    if not np.array_equal(data["critic_action_is_pad"][row], expected_pad):
                        errors.append(f"{path.name}[{row}]: critic_action_is_pad disagrees with executed_steps={actual_h}")
                    if not np.array_equal(data["executed_mask"][row], ~expected_pad):
                        errors.append(f"{path.name}[{row}]: executed_mask disagrees with executed_steps={actual_h}")
                    if np.any(data["intervention_mask"][row] & expected_pad):
                        errors.append(f"{path.name}[{row}]: intervention marked on an unexecuted action")
                    prefix = slice(0, actual_h)
                    suffix = slice(actual_h, cfg.network.chunk_size)
                    if not np.allclose(data["action"][row, prefix], data["actual_action"][row, prefix]):
                        errors.append(f"{path.name}[{row}]: ACT behavior target is not the actual executed prefix")
                    if not np.allclose(data["critic_action"][row, prefix], data["actual_action"][row, prefix]):
                        errors.append(f"{path.name}[{row}]: critic action is not the actual executed prefix")
                    if not np.allclose(data["action"][row, suffix], data["command_action"][row, suffix]):
                        errors.append(f"{path.name}[{row}]: behavior suffix does not preserve the command chunk")
                    if not np.allclose(data["critic_action"][row, suffix], 0.0):
                        errors.append(f"{path.name}[{row}]: padded critic action is nonzero")
                    action_changed = np.any(
                        np.abs(data["actual_action"][row, prefix] - data["command_action"][row, prefix]) > 1e-6,
                        axis=-1,
                    )
                    if np.any(action_changed & ~data["intervention_mask"][row, prefix]):
                        errors.append(f"{path.name}[{row}]: changed actual action lacks intervention mark")
                    if bool(data["intervention"][row]) != bool(data["intervention_mask"][row, prefix].any()):
                        errors.append(f"{path.name}[{row}]: intervention flag disagrees with intervention mask")
                terminated = data["terminated"].astype(bool)
                truncated = data["truncated"].astype(bool)
                if np.any(terminated & truncated):
                    errors.append(f"{path.name}: transition cannot be both terminated and truncated")
                if count and not (terminated[-1] or truncated[-1]):
                    errors.append(f"{path.name}: final transition is neither terminated nor truncated")
                if count and np.any(terminated[:-1] | truncated[:-1]):
                    errors.append(f"{path.name}: done marker appears before final transition")
                success = bool(data["success"].any())
                failure = bool(terminated[-1] and not success) if count else False
                if int(metadata.get("transitions", count)) != count:
                    errors.append(f"{path.name}: sidecar transition count disagrees with arrays")
                if bool(metadata.get("success", success)) != success:
                    errors.append(f"{path.name}: sidecar success disagrees with arrays")
                if bool(metadata.get("failure", failure)) != failure:
                    errors.append(f"{path.name}: sidecar failure disagrees with arrays")
                totals["success_episodes"] += int(success)
                totals["failure_episodes"] += int(failure)
                totals["truncated_episodes"] += int(bool(truncated[-1])) if count else 0
                totals["intervention_transitions"] += int(data["intervention"].astype(bool).sum())
                episodes.append({
                    "file": path.name, "schema_version": schema, "transitions": count,
                    "success": success, "failure": failure,
                    "truncated": bool(truncated[-1]) if count else False,
                    "interventions": int(data["intervention"].astype(bool).sum()),
                })
        except Exception as exc:
            errors.append(f"{path.name}: cannot read replay: {exc}")
    if not files:
        errors.append("replay contains no episode_*.npz files")
    if len(policy_versions) > 1 or "unknown" in policy_versions:
        errors.append(
            "one collection round must contain exactly one known policy_sha256; "
            f"found {sorted(policy_versions)}"
        )
    report = {
        "replay": str(root.resolve()), "schema_required": REPLAY_SCHEMA_VERSION,
        "ok": not errors, "totals": totals, "policy_sha256": sorted(policy_versions),
        "errors": errors, "episodes": episodes,
    }
    return report


def main() -> None:
    args = build_parser().parse_args()
    cfg = ACTRLConfig.from_json(args.config)
    report = audit_replay(args.replay, cfg, args.allow_legacy_replay)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
