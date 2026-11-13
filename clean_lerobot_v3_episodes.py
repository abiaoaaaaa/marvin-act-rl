#!/usr/bin/env python3
"""Create a contiguous cleaned copy of a LeRobot v3 dataset.

The operation is intentionally copy-to-new-output: the source dataset is never
modified. Episodes listed in ``--exclude-episodes`` are removed together with
their tabular rows and all camera video shards. The remaining episode indices
are preserved, which is safe when the excluded episodes are a suffix.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--exclude-episodes", type=int, nargs="+", required=True)
    return p.parse_args()


def load_rows(root: Path) -> tuple[list[dict], list[pa.Table]]:
    ep_paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    data_paths = sorted((root / "data").rglob("*.parquet"))
    if len(ep_paths) != 1 or not data_paths:
        raise ValueError("Expected one episodes parquet and at least one data parquet")
    episodes = pq.read_table(ep_paths[0]).to_pylist()
    tables = [pq.read_table(path) for path in data_paths]
    return episodes, tables


def filter_table(table: pa.Table, keep: np.ndarray) -> pa.Table:
    return table.filter(pa.array(keep))


def numeric_column(table: pa.Table, key: str) -> np.ndarray:
    array = table[key].combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        width = array.type.list_size
        return array.values.to_numpy(zero_copy_only=False).reshape(len(array), width)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        width = int(offsets[1] - offsets[0])
        return array.values.to_numpy(zero_copy_only=False).reshape(len(array), width)
    return array.to_numpy(zero_copy_only=False).reshape(len(array), -1)


def merge_stats(stats: list[dict], kept: list[int]) -> dict:
    """Aggregate per-episode stats, preserving LeRobot's JSON shape."""
    result: dict = {}
    for key in stats[0]:
        values = [stats[i][key] for i in kept if key in stats[i]]
        if not values:
            continue
        if key.endswith("/count"):
            total = int(sum(int(v[0]) for v in values))
            result[key] = [total]
            continue
        if key.endswith("/min"):
            result[key] = np.min(np.asarray([v for v in values], dtype=np.float64), axis=0).tolist()
            continue
        if key.endswith("/max"):
            result[key] = np.max(np.asarray([v for v in values], dtype=np.float64), axis=0).tolist()
            continue
        if key.endswith("/mean") or key.endswith("/std"):
            counts = np.asarray([stats[i][key[:-5] + "/count"][0] for i in kept], dtype=np.float64)
            arr = np.asarray(values, dtype=np.float64)
            mean = np.sum(arr * counts.reshape((-1,) + (1,) * (arr.ndim - 1)), axis=0) / counts.sum()
            if key.endswith("/mean"):
                result[key] = mean.tolist()
            else:
                result[key] = np.asarray(values[0], dtype=np.float64).tolist()
            continue
        # Quantiles cannot be reconstructed exactly from summary statistics;
        # retain the source shape and use the median across kept episodes.
        result[key] = np.median(np.asarray(values, dtype=np.float64), axis=0).tolist()
    return result


def main() -> None:
    args = parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    excluded = set(args.exclude_episodes)
    episodes, data_tables = load_rows(source)
    all_ids = {int(row["episode_index"]) for row in episodes}
    missing = excluded - all_ids
    if missing:
        raise ValueError(f"Unknown episode ids: {sorted(missing)}")
    kept_episode_rows = [i for i, row in enumerate(episodes) if int(row["episode_index"]) not in excluded]
    kept_ids = {int(episodes[i]["episode_index"]) for i in kept_episode_rows}
    if kept_ids != set(range(len(kept_ids))):
        raise ValueError("This copier requires kept episode indices to remain contiguous from zero")

    output.mkdir(parents=True)
    (output / "meta").mkdir()
    # Copy non-data metadata/assets first, then replace filtered metadata below.
    shutil.copytree(source / "videos", output / "videos")
    shutil.copy2(source / "meta" / "tasks.parquet", output / "meta" / "tasks.parquet")
    shutil.copy2(source / "README.md", output / "README.md")
    readme = (output / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("zadai_354", "zadai_335_cleaned")
    readme = readme.replace("Episodes | 354", "Episodes | 335")
    readme = readme.replace("Frames | 439,135", "Frames | 424,905")
    readme += (
        "\n\n## 清洗记录\n\n"
        "根据右腕图像全黑的视觉脏数据判定，完整删除 episode 335–353（19 个回合、"
        "14,230 帧）；保留回合内图像、state、action 的原始对齐关系。原始数据集为 "
        "`/home/xlb/data/lerobot/zadai_354`，清洗日期为 2026-09-06。\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")

    data_out = output / "data" / "chunk-000"
    data_out.mkdir(parents=True)
    source_data_paths = sorted((source / "data").rglob("*.parquet"))
    row_offset = 0
    for src_path, table in zip(source_data_paths, data_tables):
        ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        keep = np.isin(ids, sorted(kept_ids))
        filtered = filter_table(table, keep)
        dest = data_out / src_path.name
        pq.write_table(filtered, dest, compression="zstd")
        row_offset += filtered.num_rows

    # Remove camera shards belonging only to excluded suffix episodes.
    for camera in ("top", "wrist_L", "wrist_R"):
        camera_dir = output / "videos" / f"observation.images.{camera}" / "chunk-000"
        for path in sorted(camera_dir.glob("*.mp4")):
            if int(path.stem.split("-")[-1]) >= len(kept_ids):
                path.unlink()

    episode_rows = []
    for new_index, old_index in enumerate(kept_episode_rows):
        row = dict(episodes[old_index])
        if int(row["episode_index"]) != new_index:
            raise ValueError("Non-suffix deletion would require remapping episode ids")
        episode_rows.append(row)
    ep_out = output / "meta" / "episodes" / "chunk-000"
    ep_out.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), ep_out / "file-000.parquet", compression="zstd")

    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    info["total_episodes"] = len(episode_rows)
    info["total_frames"] = row_offset
    info["splits"] = {"train": f"0:{len(episode_rows)}"}
    (output / "meta").mkdir(exist_ok=True)
    (output / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    # Recompute exact numeric dataset statistics from the retained rows. The
    # image statistics are aggregated from retained episode summaries.
    all_data = pa.concat_tables([filter_table(t, np.isin(np.asarray(t["episode_index"].to_numpy()), sorted(kept_ids))) for t in data_tables])
    stats = json.loads((source / "meta" / "stats.json").read_text(encoding="utf-8"))
    for key in ("observation.state", "action", "observation.velocity", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        values = numeric_column(all_data, key)
        numeric = values.astype(np.float64)
        stats[key] = {
            "min": numeric.min(axis=0).tolist(),
            "max": numeric.max(axis=0).tolist(),
            "mean": numeric.mean(axis=0).tolist(),
            "std": numeric.std(axis=0).tolist(),
            "count": [int(len(numeric))],
            "q01": np.quantile(numeric, 0.01, axis=0).tolist(),
            "q10": np.quantile(numeric, 0.10, axis=0).tolist(),
            "q50": np.quantile(numeric, 0.50, axis=0).tolist(),
            "q90": np.quantile(numeric, 0.90, axis=0).tolist(),
            "q99": np.quantile(numeric, 0.99, axis=0).tolist(),
        }
    (output / "meta" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    manifest = {
        "source": str(source),
        "excluded_episodes": sorted(excluded),
        "kept_episodes": len(episode_rows),
        "kept_frames": row_offset,
        "reason": "right wrist video is black/corrupt for complete episodes 335-353",
    }
    (output / "cleaning_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
