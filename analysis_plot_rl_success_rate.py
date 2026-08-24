#!/usr/bin/env python3
"""Plot the online success-rate history from a local W&B run file."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wandb.sdk.internal.datastore import DataStore
import wandb.proto.wandb_internal_pb2 as wandb_pb


RUN_FILE = Path(
    "/home/xlb/code/强化微调ACT/wandb/run-20260819_142651-marvinhuman40densec20e5/"
    "run-marvinhuman40densec20e5.wandb"
)
OUT_DIR = Path(
    "/home/xlb/code/强化微调ACT/outputs/marvin_usb_human40_residual_td3_chunk20_exec5/analysis"
)


def read_history(path: Path) -> list[dict[str, float]]:
    datastore = DataStore()
    datastore.open_for_scan(str(path))
    rows: list[dict[str, float]] = []
    while True:
        record_data = datastore.scan_record()
        if record_data is None:
            break
        record = wandb_pb.Record()
        try:
            record.ParseFromString(record_data[1])
        except Exception:
            continue
        if not record.HasField("history"):
            continue
        row: dict[str, float] = {}
        for item in record.history.item:
            if not item.nested_key:
                continue
            try:
                row[item.nested_key[0]] = float(json.loads(item.value_json))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if "online/success_rate" in row and "training/global_step" in row:
            rows.append(
                {
                    "global_step": row["training/global_step"],
                    "success_rate": row["online/success_rate"],
                    "episodes": row.get("online/episodes_total", np.nan),
                    "transitions": row.get("online/transitions_total", np.nan),
                    "runtime_s": row.get("_runtime", np.nan),
                }
            )
    rows.sort(key=lambda row: row["global_step"])
    return rows


def main() -> None:
    rows = read_history(RUN_FILE)
    if not rows:
        raise RuntimeError("No online/success_rate history found")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "rl_success_rate_history.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    steps = np.asarray([row["global_step"] for row in rows])
    rates = np.asarray([row["success_rate"] for row in rows])
    window = min(10, len(rates))
    smooth = np.convolve(rates, np.ones(window) / window, mode="valid")
    smooth_steps = steps[window - 1 :]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.5), constrained_layout=True)
    ax.plot(steps, rates * 100, color="#9aa3ad", linewidth=1.0, alpha=0.55, label="Cumulative rate")
    ax.plot(
        smooth_steps,
        smooth * 100,
        color="#1f5a85",
        linewidth=2.0,
        label=f"{window}-record moving average",
    )
    ax.scatter([steps[-1]], [rates[-1] * 100], color="#b44b3e", s=24, zorder=3)
    ax.annotate(
        f"Final: {rates[-1] * 100:.1f}%\n{int(rows[-1]['episodes'])} episodes",
        xy=(steps[-1], rates[-1] * 100),
        xytext=(-58, 14),
        textcoords="offset points",
        fontsize=8,
        color="#b44b3e",
        arrowprops={"arrowstyle": "-", "color": "#b44b3e", "lw": 0.8},
    )
    ax.set_xlabel("Learner global step")
    ax.set_ylabel("Online success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.grid(axis="y", color="#d9dee3", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(OUT_DIR / "rl_success_rate_curve.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "rl_success_rate_curve.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "rl_success_rate_curve.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "rl_success_rate_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"records={len(rows)}")
    print(f"initial={rates[0] * 100:.2f}% at step={int(steps[0])}")
    print(f"final={rates[-1] * 100:.2f}% at step={int(steps[-1])}")
    print(f"max={rates.max() * 100:.2f}%")
    print(f"runtime_hours={rows[-1]['runtime_s'] / 3600:.3f}")
    print(f"csv={csv_path}")
    print(f"figure={OUT_DIR / 'rl_success_rate_curve.png'}")


if __name__ == "__main__":
    main()
