#!/usr/bin/env python3
"""Marvin-specific RL-100 stage-2 entry point."""

from __future__ import annotations

import pathlib
import sys
import types

import dill
import hydra
import torch

# RL-100 imports its optional MuJoCo oracle unconditionally from the dynamics
# package. Marvin uses learned dynamics and has no simulator, so prevent that
# unused module from importing mujoco_py (which is intentionally absent).
_oracle_module_name = "rl_100.unidpg.transition_model.dynamics.mujoco_oracle_dynamics"
if _oracle_module_name not in sys.modules:
    oracle_stub = types.ModuleType(_oracle_module_name)
    oracle_stub.MujocoOracleDynamics = type("MujocoOracleDynamics", (), {})
    sys.modules[_oracle_module_name] = oracle_stub

# uni_ppo imports RL1003D only for an annotation. Loading the real 3D policy
# would require pytorch3d even though this run is strictly 2D.
_policy_3d_module_name = "rl_100.policy.rl100_3d"
if _policy_3d_module_name not in sys.modules:
    policy_3d_stub = types.ModuleType(_policy_3d_module_name)
    policy_3d_stub.RL1003D = type("RL1003D", (), {})
    sys.modules[_policy_3d_module_name] = policy_3d_stub

from train_real import TrainDP3Workspace


class MarvinOfflineWorkspace(TrainDP3Workspace):
    """Load only actor weights from stage 1 and retain the new output directory."""

    def get_stage1_artifact_dir(self):
        # Critic and dynamics belong to the stage-2 run, not the immutable BC run.
        return self.output_dir

    def get_stage1_checkpoint_path(self, tag="latest"):
        if tag == "latest":
            return pathlib.Path(self.cfg.marvin_offline.base_checkpoint)
        return super().get_stage1_checkpoint_path(tag=tag)

    def load_checkpoint(self, path=None, tag="latest", **_kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        source = str(self.cfg.marvin_offline.checkpoint_source)
        state_key = "ema_model" if source == "ema" else "model"
        if state_key not in payload.get("state_dicts", {}):
            raise KeyError(
                f"Stage-1 checkpoint has no {state_key!r}; available={sorted(payload.get('state_dicts', {}))}"
            )
        state = payload["state_dicts"][state_key]
        self.model.load_state_dict(state, strict=True)
        if self.ema_model is not None:
            self.ema_model.load_state_dict(state, strict=True)
        print(f"Loaded Marvin stage-1 {state_key} actor from {path}")
        return payload

    def finetune_dp3(self, dynamics, Q, value, iql, wandb, ema):
        # RL1002D.sample_action(get_np=False) returns its full 16-step denoised
        # trajectory. The generic dynamics OPE path assumes it already contains
        # exactly n_action_steps. Adapt only that OPE call; BPPO keeps the full
        # denoising trajectory and applies its own chunk slicing.
        original_rollout = dynamics.rollout
        executed_steps = int(self.cfg.n_action_steps)

        class KStepPolicyProxy:
            def __init__(self, policy):
                self._policy = policy

            def __getattr__(self, name):
                return getattr(self._policy, name)

            def sample_action(self, *args, **kwargs):
                action = self._policy.sample_action(*args, **kwargs)
                if torch.is_tensor(action) and action.ndim == 3:
                    return action[:, :executed_steps]
                return action

        def rollout_with_executed_chunk(policy, *args, **kwargs):
            return original_rollout(KStepPolicyProxy(policy), *args, **kwargs)

        dynamics.rollout = rollout_with_executed_chunk
        try:
            return super().finetune_dp3(dynamics, Q, value, iql, wandb, ema)
        finally:
            dynamics.rollout = original_rollout

    def run(self):
        import rl_100.unidpg.transition_model.utils.logger as dynamics_logger

        cfg = self.cfg
        if not (cfg.offline and not cfg.online and not cfg.only_bc):
            raise ValueError("Marvin stage 2 requires offline=true, online=false, only_bc=false")
        if int(cfg.training.num_epochs) != 0:
            raise ValueError("Set training.num_epochs=0 so stage 2 cannot accidentally repeat BC")
        if not cfg.training.resume or not cfg.marvin_offline.get("base_checkpoint"):
            raise ValueError("Stage 2 requires training.resume=true and marvin_offline.base_checkpoint")
        if int(cfg.n_action_steps) != int(cfg.task.dataset.executed_steps):
            raise ValueError("n_action_steps must equal task.dataset.executed_steps")
        dynamics_logger.ROOT_DIR = str(pathlib.Path(self.output_dir()) / "dynamics_logs")
        return super().run()


@hydra.main(
    version_base=None,
    config_path="../RL-100/RL-100/rl_100/config",
    config_name="rl100_2d_epsilon",
)
def main(cfg):
    MarvinOfflineWorkspace(cfg).run()


if __name__ == "__main__":
    main()
