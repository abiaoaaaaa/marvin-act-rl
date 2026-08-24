"""Quantitative residual-effect analysis on the cached offline state distribution.

The figure deliberately separates three offline diagnostics from the non-paired
real-robot outcome summary. Offline diagnostics do not claim causal improvement;
they describe how the learned residual changes ACT actions on a fixed state set.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


ROOT = Path("/home/xlb/code/强化微调ACT")
CACHE = ROOT / "outputs/marvin_residual_td3_chunk20_exec5/cache/offline_buffer_cache/e7f1719a/storage"
OUTPUT = ROOT / "outputs/residual_effect_analysis"
N_OFFLINE = 4096
BATCH_SIZE = 32
SEED = 1000
CAMERAS = [
    "side_policy",
    "wrist_1buchong",
    "wrist_1",
    "wrist_2",
]


def memmap(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def load_offline_subset() -> dict[str, np.ndarray]:
    meta = json.loads((CACHE / "obs/meta.json").read_text())
    n = int(meta["shape"][0])
    # Deterministic systematic subsampling of real replay rows for dense plots;
    # no synthetic observations are generated. The full cache remains untouched.
    ids = np.linspace(0, n - 1, num=min(N_OFFLINE, n), dtype=np.int64)
    result: dict[str, np.ndarray] = {"ids": ids}
    result["state"] = np.asarray(memmap(CACHE / "obs/observation.state.memmap", np.float32, (n, 24))[ids])
    result["base"] = np.asarray(memmap(CACHE / "obs/observation.base_action.memmap", np.float32, (n, 5))[ids])
    result["action"] = np.asarray(memmap(CACHE / "action.memmap", np.float32, (n, 5))[ids])
    for camera in CAMERAS:
        result[camera] = np.asarray(
            memmap(CACHE / f"obs/observation.images.{camera}.memmap", np.uint8, (n, 3, 128, 128))[ids]
        )
    return result


def load_actor(bundle_path: Path) -> QAgent:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    cfg = QAgentConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    cfg.actor.action_scale = float(bundle["actor_action_scale"])
    cfg.actor.actor_last_layer_init_scale = 0.0
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        agent = QAgent(
            obs_shape=(3, 128, 128),
            prop_shape=(24,),
            action_dim=5,
            rl_cameras=[f"observation.images.{camera}" for camera in CAMERAS],
            cfg=cfg,
            residual_actor=True,
        )
    agent.actor.load_state_dict(bundle["actor"])
    agent.encoders.load_state_dict(bundle["encoders"])
    agent.eval()
    agent.actor.eval()
    agent.encoders.eval()
    return agent


@torch.inference_mode()
def predict_residual(agent: QAgent, data: dict[str, np.ndarray]) -> np.ndarray:
    device = torch.device(agent.cfg.device)
    n = len(data["state"])
    output = []
    for start in range(0, n, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, n)
        obs = {
            "observation.state": torch.from_numpy(data["state"][start:stop]).to(device),
            "observation.base_action": torch.from_numpy(data["base"][start:stop]).to(device),
        }
        for camera in CAMERAS:
            obs[f"observation.images.{camera}"] = torch.from_numpy(data[camera][start:stop]).to(device)
        output.append(agent.act(obs, eval_mode=True, stddev=0.0, cpu=True).numpy())
    return np.concatenate(output, axis=0)


def wilson(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = success / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def read_success_rates() -> list[tuple[str, int, int, float, float, float]]:
    specs = [
        ("ACT base", ROOT / "outputs/eval_act_only_chunk20_exec5", 20),
        ("RL step 11000", ROOT / "outputs/eval_act_residual_step11000_chunk20_exec5", 20),
        ("RL step 11050", ROOT / "outputs/eval_act_residual_chunk20_exec5", 20),
    ]
    rows = []
    for label, directory, take_last in specs:
        files = sorted(directory.glob("episode_*.json"))[-take_last:]
        values = [json.loads(path.read_text()).get("success") for path in files]
        values = [bool(value) for value in values if value is not None]
        success = sum(values)
        lo, hi = wilson(success, len(values))
        rows.append((label, success, len(values), success / len(values), lo, hi))
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_offline_subset()
    demonstration_residual = data["action"] - data["base"]

    checkpoints = {
        "RL step 11000": ROOT / "outputs/marvin_residual_td3_chunk20_exec5/deployment/residual_actor_step_00011000.pt",
        "RL step 11050": ROOT / "outputs/marvin_residual_td3_chunk20_exec5/deployment/residual_actor_latest.pt",
    }
    predictions = {}
    for name, path in checkpoints.items():
        agent = load_actor(path)
        predictions[name] = predict_residual(agent, data)
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save numerical source data next to the figure for traceability.
    np.savez_compressed(
        OUTPUT / "residual_effect_source_data.npz",
        ids=data["ids"],
        base_action=data["base"],
        demonstration_action=data["action"],
        demonstration_residual=demonstration_residual,
        residual_step11000=predictions["RL step 11000"],
        residual_step11050=predictions["RL step 11050"],
    )
    with (OUTPUT / "real_robot_success_summary.json").open("w") as handle:
        json.dump(
            [
                {"model": row[0], "success": row[1], "total": row[2], "rate": row[3], "ci95": [row[4], row[5]]}
                for row in read_success_rates()
            ],
            handle,
            indent=2,
        )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
        }
    )
    colors = {"target": "#5B6472", "RL step 11000": "#2A9D8F", "RL step 11050": "#E76F51"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    norms = {
        "target": np.linalg.norm(demonstration_residual, axis=1),
        "RL step 11000": np.linalg.norm(predictions["RL step 11000"], axis=1),
        "RL step 11050": np.linalg.norm(predictions["RL step 11050"], axis=1),
    }
    bins = np.linspace(0, max(v.max() for v in norms.values()) * 1.02, 45)
    for key in ["target", "RL step 11000", "RL step 11050"]:
        label = "Demonstration residual" if key == "target" else key
        ax.hist(norms[key], bins=bins, density=True, histtype="step", linewidth=1.5, color=colors[key], label=label)
    ax.set(xlabel="Residual magnitude ||a - a_base||2", ylabel="Density", title="A  Residual magnitude")
    ax.legend(fontsize=6)

    ax = axes[0, 1]
    dim_labels = ["dx", "dy", "dz", "drz", "gripper"]
    x = np.arange(5)
    width = 0.24
    values = [
        np.mean(np.abs(demonstration_residual), axis=0),
        np.mean(np.abs(predictions["RL step 11000"]), axis=0),
        np.mean(np.abs(predictions["RL step 11050"]), axis=0),
    ]
    for i, (label, value) in enumerate(zip(["Demonstration target", "RL 11000", "RL 11050"], values)):
        color = colors["target"] if i == 0 else colors["RL step 11000"] if i == 1 else colors["RL step 11050"]
        ax.bar(x + (i - 1) * width, value, width, color=color, label=label)
    ax.axhline(0.09, color="#8D99AE", linestyle="--", linewidth=0.8, label="Residual cap (90%)")
    clip_rates = {}
    for label, key in [("RL 11000", "RL step 11000"), ("RL 11050", "RL step 11050")]:
        combined = data["base"] + predictions[key]
        clip_rates[label] = float(np.mean(np.any((combined < -1) | (combined > 1), axis=1)))
    ax.set_xticks(x, dim_labels)
    ax.set(
        ylabel="Mean absolute residual",
        title=f"B  Per-dimension correction\n(any-dim clipping: {clip_rates['RL 11000']:.1%} / {clip_rates['RL 11050']:.1%})",
    )
    ax.legend(fontsize=6)

    ax = axes[1, 0]
    target_norm = np.linalg.norm(demonstration_residual, axis=1)
    target_nonzero = target_norm > 1e-6
    for label in ["RL step 11000", "RL step 11050"]:
        pred = predictions[label]
        cosine = np.sum(pred * demonstration_residual, axis=1) / (
            np.linalg.norm(pred, axis=1) * target_norm + 1e-8
        )
        cosine = cosine[target_nonzero]
        ax.hist(cosine, bins=np.linspace(-1, 1, 41), density=True, histtype="step", linewidth=1.5, color=colors[label], label=label)
        ax.axvline(np.median(cosine), color=colors[label], linestyle=":", linewidth=1)
    ax.axvline(0, color="#8D99AE", linewidth=0.8)
    ax.set(xlabel="Cosine similarity to demonstration residual", ylabel="Density", title="C  Directional agreement")
    ax.legend(fontsize=6)

    ax = axes[1, 1]
    rows = read_success_rates()
    x = np.arange(len(rows))
    rates = [row[3] for row in rows]
    lower = [row[3] - row[4] for row in rows]
    upper = [row[5] - row[3] for row in rows]
    ax.errorbar(x, rates, yerr=[lower, upper], fmt="o", markersize=5, capsize=3, linewidth=1.2, color="#264653")
    ax.set_xticks(x, [row[0].replace(" ", "\n") for row in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("D  Real-robot outcome (last 20 episodes)")
    for idx, row in enumerate(rows):
        ax.text(idx, min(1.02, row[3] + 0.09), f"{row[1]}/{row[2]}", ha="center", va="bottom", fontsize=6)
    ax.text(
        0.02,
        0.03,
        "Wilson 95% CI; episodes are not paired across models",
        transform=ax.transAxes,
        fontsize=5.5,
        color="#4A5568",
    )

    fig.suptitle(
        "Residual policy effect on a fixed Marvin offline state distribution "
        f"(n={len(data['state'])}; real-robot episodes are non-paired)",
        fontsize=10,
    )
    for idx, ax in enumerate(axes.flat):
        ax.text(-0.12, 1.08, chr(ord("a") + idx), transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.savefig(OUTPUT / "residual_effect_analysis.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / "residual_effect_analysis.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "residual_effect_analysis.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT / "residual_effect_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures and source data to {OUTPUT}")


if __name__ == "__main__":
    main()
