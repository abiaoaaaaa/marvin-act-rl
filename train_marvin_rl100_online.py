#!/usr/bin/env python3
"""Run RL-100's original online diffusion PPO through Marvin Bridge."""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import time
import types

import dill
import hydra
import numpy as np
import torch

# Marvin is 2D and uses learned dynamics. Avoid importing optional 3D/MuJoCo
# dependencies that RL-100 references only through annotations or unused paths.
_oracle_name = "rl_100.unidpg.transition_model.dynamics.mujoco_oracle_dynamics"
if _oracle_name not in sys.modules:
    oracle_stub = types.ModuleType(_oracle_name)
    oracle_stub.MujocoOracleDynamics = type("MujocoOracleDynamics", (), {})
    sys.modules[_oracle_name] = oracle_stub

_policy_3d_name = "rl_100.policy.rl100_3d"
if _policy_3d_name not in sys.modules:
    policy_3d_stub = types.ModuleType(_policy_3d_name)
    policy_3d_stub.RL1003D = type("RL1003D", (), {})
    sys.modules[_policy_3d_name] = policy_3d_stub

from train_real import TrainDP3Workspace

from lerobot_robot_marvin.client import MarvinBridgeClient
from lerobot_robot_marvin.rl100_online import (
    MarvinBridgeOnlineEnv,
    MarvinOnlineReplayBuffer,
    write_latest_metadata,
)
from lerobot_robot_marvin.rl100_rollout import (
    BridgeHeartbeat,
    validate_bridge_preflight,
)


class MarvinOnlineWorkspace(TrainDP3Workspace):
    """Reuse RL-100 initialization while replacing only its robot adapters."""

    def __init__(self, cfg, output_dir=None):
        # Match the official 2D chunk-online recipe's variance clipping. This
        # constrains stochastic DDIM exploration before any action reaches the
        # Bridge; post-hoc action clipping would invalidate stored log-probs.
        cfg.clip_std_max = float(cfg.marvin_online.get("exploration_std_max", 0.1))
        super().__init__(cfg, output_dir=output_dir)

    def get_stage1_artifact_dir(self):
        return str(pathlib.Path(self.cfg.marvin_online.stage2_dir).expanduser().resolve())

    def get_stage1_checkpoint_path(self, tag="latest"):
        del tag
        return pathlib.Path(self.cfg.marvin_online.base_checkpoint).expanduser().resolve()

    def get_global_best_dir(self):
        return str(pathlib.Path(self.cfg.marvin_online.stage2_policy_dir).expanduser().resolve())

    def load_checkpoint(self, path=None, tag="latest", **_kwargs):
        if path is None:
            path = self.get_stage1_checkpoint_path(tag)
        path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        state = payload.get("state_dicts", {}).get("model")
        if state is None:
            raise KeyError(f"Online base checkpoint has no model state: {path}")
        self.model.load_state_dict(state, strict=True)
        if self.ema_model is not None:
            self.ema_model.load_state_dict(state, strict=True)
        print(f"Loaded Marvin online base actor from {path}")
        return payload

    def finetune_dp3(self, dynamics, Q, value, iql, wandb, ema):
        # ``offline=true`` is retained only so RL-100 loads the existing learned
        # dynamics artifact. Never repeat stage-2 BPPO in the online command.
        if self.cfg.online:
            print("Skipping offline BPPO; entering Marvin online stage")
            return None
        return super().finetune_dp3(dynamics, Q, value, iql, wandb, ema)

    def _sample_bounded_online_action(self, torch_obs):
        """Sample RL-100 and keep its final diffusion transition executable.

        RL-100 clips the predicted x0 in DDIM, but its stochastic final step can
        add noise outside the dataset/Bridge action support.  Projecting only at
        the Bridge would make the recorded action and old log-prob disagree.
        Instead, project the final diffusion state to the dataset action bounds
        and recompute that final transition's log-prob before storing it.
        """
        policy = self.unio4._policy
        action, all_x, old_logprob = policy.all_step_action_logprob(
            torch_obs,
            fix_encoder=True,
        )
        del action

        action_dim = int(policy.action_dim)
        raw_final = policy.normalizer["action"].unnormalize(
            all_x[-1][..., :action_dim]
        )
        bounded_final = raw_final.clamp(-1.0, 1.0)
        projected_values = int(torch.count_nonzero(raw_final != bounded_final).item())
        if projected_values:
            all_x = all_x.clone()
            old_logprob = old_logprob.clone()
            normalized_final = policy.normalizer["action"].normalize(bounded_final)
            all_x[-1, ..., :action_dim] = normalized_final

            _cond_data, _cond_mask, local_cond, global_cond, _features = policy.obs2feature(
                torch_obs,
                training=False,
                fix_encoder=True,
            )
            timestep = policy.noise_scheduler.timesteps[-1]
            if not torch.is_tensor(timestep):
                timesteps = torch.tensor([timestep], dtype=torch.long, device=self.device)
            elif timestep.ndim == 0:
                timesteps = timestep[None].to(self.device)
            else:
                timesteps = timestep.to(self.device)
            timesteps = timesteps.expand(all_x.shape[1])
            unet_timesteps = policy.get_unet_timesteps(timesteps)
            model_output = policy.model(
                sample=all_x[-2],
                timestep=unet_timesteps,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            eta = 1.0
            if getattr(policy, "diffusion_eta", False):
                eta = torch.mean(model_output[:, :, -1], dim=1, keepdim=True)
            final_logprob, _entropy = policy.noise_scheduler.step_forward_logprob_with_entropy(
                model_output,
                timesteps,
                all_x[-2],
                next_sample=all_x[-1],
                eta=eta,
            )
            old_logprob[-1] = final_logprob.detach()

        start = 0 if policy.no_pre_action else int(policy.n_obs_steps) - 1
        end = start + int(policy.n_action_steps)
        action = bounded_final[:, start:end]
        return action, all_x, old_logprob, projected_values

    def _save_online_update(self, update_root, version, total_steps, actor_loss, critic_loss):
        update_dir = update_root / f"update_{version:04d}"
        if update_dir.exists():
            raise FileExistsError(f"Refusing to overwrite online update: {update_dir}")
        policy_dir = update_dir / "policy"
        critic_dir = update_dir / "critic"
        policy_dir.mkdir(parents=True)
        critic_dir.mkdir(parents=True)
        self.unio4.save(str(policy_dir))
        self.unio4.save_critic(str(critic_dir))
        torch.save(
            {
                "actor": self.unio4.optimizer_actor.state_dict(),
                "critic": self.unio4.optimizer_critic.state_dict(),
            },
            update_dir / "optimizers.pt",
        )

        base_path = pathlib.Path(self.cfg.marvin_online.base_checkpoint).expanduser().resolve()
        payload = torch.load(base_path.open("rb"), pickle_module=dill, map_location="cpu")
        payload = copy.deepcopy(payload)
        payload["state_dicts"] = {
            "model": {
                key: value.detach().cpu()
                for key, value in self.unio4._policy.state_dict().items()
            }
        }
        payload["marvin_online_export"] = {
            "base_checkpoint": str(base_path),
            "policy_version": int(version),
            "total_chunk_transitions": int(total_steps),
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss),
            "online_action_projection": "final_diffusion_state_to_raw_minus1_plus1",
        }
        candidate_path = update_dir / "candidate.ckpt"
        torch.save(payload, candidate_path.open("wb"), pickle_module=dill)

        metadata = {
            "policy_version": int(version),
            "total_chunk_transitions": int(total_steps),
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss),
            "n_action_steps": int(self.cfg.n_action_steps),
            "batch_size": int(self.cfg.marvin_online.batch_size),
            "base_checkpoint": str(base_path),
            "update_dir": str(update_dir.resolve()),
            "candidate_checkpoint": str(candidate_path.resolve()),
        }
        write_latest_metadata(update_root / "latest.json", metadata)
        return metadata

    def _resume_online_update(self, update_root):
        latest_path = update_root / "latest.json"
        if not bool(self.cfg.marvin_online.resume_online) or not latest_path.is_file():
            return 0, 0
        metadata = json.loads(latest_path.read_text(encoding="utf-8"))
        expected_base = str(
            pathlib.Path(self.cfg.marvin_online.base_checkpoint).expanduser().resolve()
        )
        compatibility = {
            "n_action_steps": int(self.cfg.n_action_steps),
            "batch_size": int(self.cfg.marvin_online.batch_size),
            "base_checkpoint": expected_base,
        }
        for key, expected in compatibility.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Cannot resume online run: {key} changed from "
                    f"{metadata.get(key)!r} to {expected!r}"
                )
        update_dir = pathlib.Path(metadata["update_dir"])
        self.unio4.load(str(update_dir / "policy"))
        self.unio4.load_critic(str(update_dir / "critic"))
        optimizer_path = update_dir / "optimizers.pt"
        if optimizer_path.is_file():
            optimizers = torch.load(optimizer_path, map_location=self.device)
            self.unio4.optimizer_actor.load_state_dict(optimizers["actor"])
            self.unio4.optimizer_critic.load_state_dict(optimizers["critic"])
        version = int(metadata["policy_version"])
        total_steps = int(metadata["total_chunk_transitions"])
        print(f"Resumed Marvin online policy version {version} at transition {total_steps}")
        return version, total_steps

    def online_ft(self, dynamics, Q, value, iql, iql_online, copy_encoder, wandb, ema):
        del Q, value, iql_online, copy_encoder, ema
        cfg = self.cfg
        online_cfg = cfg.marvin_online
        if cfg.ppo.iql_ft:
            raise ValueError("Marvin online v1 keeps IQL Q/V frozen; set ppo.iql_ft=false")
        if cfg.distill_phase is not None:
            raise ValueError("Marvin online v1 does not combine policy distillation with PPO")
        if not cfg.ppo.fix_encoder:
            raise ValueError("Marvin online v1 requires ppo.fix_encoder=true")
        if int(cfg.n_action_steps) != int(online_cfg.actions_per_inference):
            raise ValueError("n_action_steps must equal Marvin actions_per_inference")
        if int(cfg.ppo.batch_size) != int(online_cfg.batch_size):
            raise ValueError("ppo.batch_size must equal marvin_online.batch_size")

        value_net = iql.get_online_value_buget(cfg)
        self.unio4.transfer2online(critic=value_net, dynamics=dynamics, cfg=cfg)
        update_root = pathlib.Path(self.output_dir) / "online_ft"
        update_root.mkdir(parents=True, exist_ok=True)
        version, total_steps = self._resume_online_update(update_root)

        buffer = MarvinOnlineReplayBuffer(
            capacity=int(online_cfg.batch_size),
            device=torch.device(self.device),
            policy_version=version,
        )
        buffer_path = update_root / "on_policy_buffer.pkl"
        if bool(online_cfg.resume_online) and buffer_path.is_file():
            buffer.load(buffer_path, expected_policy_version=version)
            print(f"Restored {buffer.count}/{buffer.capacity} on-policy transitions")

        client = MarvinBridgeClient(
            host=str(online_cfg.host),
            port=int(online_cfg.port),
            timeout_s=float(online_cfg.timeout),
            connect_timeout_s=float(online_cfg.connect_timeout),
            reset_timeout_s=float(online_cfg.reset_timeout),
        )
        status = client.status()
        validate_bridge_preflight(status, bool(online_cfg.allow_real_robot))
        client.connect()
        heartbeat_client = MarvinBridgeClient(
            host=str(online_cfg.host),
            port=int(online_cfg.port),
            timeout_s=float(online_cfg.timeout),
        )
        heartbeat = BridgeHeartbeat(
            heartbeat_client,
            interval_s=float(online_cfg.heartbeat_interval),
        )
        heartbeat.start()
        env = MarvinBridgeOnlineEnv(
            client,
            n_obs_steps=int(cfg.n_obs_steps),
            n_action_steps=int(cfg.n_action_steps),
            max_episode_steps=int(online_cfg.max_episode_steps),
            action_match_atol=float(online_cfg.action_match_atol),
        )

        max_updates = int(online_cfg.max_updates)
        collect_only = bool(online_cfg.collect_only)
        updates_this_run = 0
        episode_index = 0
        stop_requested = False
        try:
            while not stop_requested:
                heartbeat.check()
                obs, reset_info = env.reset()
                heartbeat.note_primary_contact()
                episode_reward = 0.0
                episode_chunks = 0
                done = False
                print(f"online episode={episode_index} reset_info={reset_info}")
                while not done:
                    heartbeat.check()
                    torch_obs = {}
                    for key, value in obs.items():
                        tensor = torch.from_numpy(value).unsqueeze(0)
                        # The encoder normalizes [0,255] float images itself.
                        # Always cast RGB here because an all-black uint8 mock
                        # frame cannot be detected by a max()>1 heuristic.
                        if key != "agent_pos":
                            tensor = tensor.float()
                        torch_obs[key] = tensor.to(self.device, non_blocking=True)
                    self.unio4._policy.eval()
                    started = time.monotonic()
                    with torch.no_grad():
                        action, all_x, old_logprob, projected_values = (
                            self._sample_bounded_online_action(torch_obs)
                        )
                    if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                        torch.cuda.synchronize(torch.device(self.device))
                    inference_seconds = time.monotonic() - started
                    action_np = action.squeeze(0).detach().cpu().numpy()
                    all_x_np = all_x.squeeze(1).detach().cpu().numpy()
                    old_logprob_np = old_logprob.squeeze(1).detach().cpu().numpy()

                    next_obs, reward, done, info = env.step_online(action_np, gamma=float(cfg.gamma))
                    heartbeat.note_primary_contact()
                    terminal = bool(info["terminated"])
                    buffer_boundary = buffer.count + 1 == buffer.capacity
                    stored_done = bool(done or (collect_only and buffer_boundary))
                    buffer.store(
                        obs,
                        all_x_np,
                        old_logprob_np,
                        reward,
                        next_obs,
                        stored_done,
                        terminal,
                    )
                    total_steps += 1
                    episode_chunks += 1
                    episode_reward += reward
                    obs = next_obs
                    print(
                        f"online episode={episode_index} chunk={episode_chunks} "
                        f"transition={total_steps} reward={reward:.4f} "
                        f"executed={info['executed_steps']} inference={inference_seconds:.3f}s "
                        f"projected_values={projected_values} done={done} "
                        f"buffer={buffer.count}/{buffer.capacity}"
                    )

                    if buffer.count == buffer.capacity:
                        if collect_only:
                            buffer.save(buffer_path)
                            print("Collect-only buffer is full; no optimizer step was executed")
                            stop_requested = True
                            break
                        actor_loss, critic_loss, bc_loss, distill_loss = self.unio4.dp_align_update_no_share(
                            buffer,
                            total_steps,
                        )
                        version += 1
                        updates_this_run += 1
                        metadata = self._save_online_update(
                            update_root,
                            version,
                            total_steps,
                            actor_loss,
                            critic_loss,
                        )
                        print(f"Saved online candidate: {metadata['candidate_checkpoint']}")
                        wandb.log(
                            {
                                "online/actor_loss": actor_loss,
                                "online/critic_loss": critic_loss,
                                "online/bc_loss": bc_loss,
                                "online/distill_loss": distill_loss,
                                "online/policy_version": version,
                                "online/chunk_transitions": total_steps,
                            }
                        )
                        buffer.clear(policy_version=version)
                        if buffer_path.exists():
                            buffer_path.unlink()
                        if updates_this_run >= max_updates:
                            stop_requested = True
                            break

                if buffer.count > 0:
                    buffer.save(buffer_path)
                wandb.log(
                    {
                        "online/episode_reward": episode_reward,
                        "online/episode_chunks": episode_chunks,
                        "online/episode_index": episode_index,
                    }
                )
                episode_index += 1
        except Exception:
            client.best_effort_stop("Marvin RL-100 online exception")
            raise
        finally:
            heartbeat.close()
            client.disconnect()

    def run(self):
        cfg = self.cfg
        if not (cfg.online and cfg.offline and not cfg.only_bc):
            raise ValueError("Marvin online requires online=true, offline=true, only_bc=false")
        if int(cfg.training.num_epochs) != 0:
            raise ValueError("Set training.num_epochs=0; online training must not repeat BC")
        if int(cfg.n_action_steps) != int(cfg.task.dataset.executed_steps):
            raise ValueError("online n_action_steps must match the stage-2 dataset executed_steps")
        # Reconstruct the exact learned-dynamics architecture used by the
        # completed Marvin stage-2 command before loading dynamics.pth.
        cfg.dynamics.dynamics_hidden_dims = [512, 512, 256, 256]
        for path in (
            pathlib.Path(cfg.marvin_online.base_checkpoint),
            pathlib.Path(cfg.marvin_online.stage2_dir) / "Q_bc_20.pt",
            pathlib.Path(cfg.marvin_online.stage2_dir) / "value_20.pt",
            pathlib.Path(cfg.marvin_online.stage2_dir) / "saved_models_full" / "dynamics.pth",
            pathlib.Path(cfg.marvin_online.stage2_policy_dir) / "model.pt",
            pathlib.Path(cfg.marvin_online.stage2_policy_dir) / "encoder.pt",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        return super().run()


@hydra.main(
    version_base=None,
    config_path="../RL-100/RL-100/rl_100/config",
    config_name="rl100_2d_epsilon",
)
def main(cfg):
    MarvinOnlineWorkspace(cfg).run()


if __name__ == "__main__":
    main()
