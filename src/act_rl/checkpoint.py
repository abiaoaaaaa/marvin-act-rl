from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, learner, step: int, metrics: dict[str, float] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = learner.state_dict()
    payload.update({"step": int(step), "metrics": metrics or {}, "format_version": 1})
    torch.save(payload, path)
    path.with_suffix(".json").write_text(json.dumps({"step": step, "metrics": metrics or {}, "format_version": 1}, indent=2))
    return path


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {"format_version": data.get("format_version"), "step": data.get("step"),
            "metrics": data.get("metrics", {}), "config": data.get("config", {}),
            "keys": sorted(data.keys())}
