#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/home/xlb/code/强化微调ACT
RL100_ROOT=/home/xlb/code/RL-100/RL-100
PYTHON_BIN=/home/xlb/miniconda3/envs/rl100/bin/python

OFFLINE_DATASET=${OFFLINE_DATASET:-/home/xlb/code/RL-100/RL-100/data/marvin_usb_offline_rl.zarr}
BC_DATASET=${BC_DATASET:-/home/xlb/code/RL-100/RL-100/data/marvin_usb_bc.zarr}
BC_CHECKPOINT=${BC_CHECKPOINT:-/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_bc_seed1000_retrain_20260813/checkpoints/latest.ckpt}
EXECUTED_STEPS=${EXECUTED_STEPS:-5}
OUTPUT_DIR=${OUTPUT_DIR:-/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_offline_rl_k${EXECUTED_STEPS}_seed1000}
SEQUENCE_STRIDE=${SEQUENCE_STRIDE:-${EXECUTED_STEPS}}
DEVICE=${DEVICE:-cuda:0}
WANDB_LOG_MODE=${WANDB_LOG_MODE:-offline}
CRITIC_EPOCHS=${CRITIC_EPOCHS:-20}
DYNAMICS_EPOCHS=${DYNAMICS_EPOCHS:-50}
BPPO_STEPS=${BPPO_STEPS:-4000}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}

for required_path in "${OFFLINE_DATASET}" "${BC_DATASET}" "${BC_CHECKPOINT}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done
if (( EXECUTED_STEPS < 1 || EXECUTED_STEPS > 16 )); then
    echo "EXECUTED_STEPS must be between 1 and 16" >&2
    exit 1
fi
if (( BATCH_SIZE < 2 )); then
    echo "BATCH_SIZE must be at least 2 because BPPO normalizes advantages" >&2
    exit 1
fi

export PYTHONPATH="${WORKSPACE_ROOT}/src:${RL100_ROOT}:/home/xlb/code/RL-100/third_party/gym-0.21.0"
export TORCH_HOME=${TORCH_HOME:-/home/xlb/code/强化微调ACT/.cache/torch}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-rl100}
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-offline}

cd "${RL100_ROOT}"
"${PYTHON_BIN}" "${WORKSPACE_ROOT}/train_marvin_rl100_offline.py" --config-name=rl100_2d_epsilon task=marvin_usb hydra.run.dir="${OUTPUT_DIR}" training.device="${DEVICE}" training.seed=1000 training.resume=true training.num_epochs=0 training.num_critic_epochs="${CRITIC_EPOCHS}" training.use_ema=false training.rollout_every=1000000 training.checkpoint_every=1000000 training.val_every=1 training.sample_every=1000000 only_bc=false offline=true online=false eval=false distill_phase=null use_wandb=true logging.mode="${WANDB_LOG_MODE}" logging.project=marvin-rl100-offline-rl logging.name="marvin_usb_offline_rl_k${EXECUTED_STEPS}_seed1000" checkpoint.save_ckpt=false env_num=1 feature_type=2D with_pointcloud=false use_agent_pos=true action_norm=true chunk_as_single_action=true ft_all_actions=false horizon=17 n_obs_steps=2 n_action_steps="${EXECUTED_STEPS}" num_inference_steps=10 policy._target_=rl_100.policy.rl100_2d.RL1002D policy.scheduler_type=ddim policy.model=dp3 policy.w_pc=false policy.use_visual=true policy.use_aug=false policy.img_shape='[3,128,128]' policy.down_dims='[128,256,512]' policy.chunk_as_single_action=true encoder_type=resnet encoders.resnet.resize_shape='[128,128]' encoders.resnet.crop_shape=null encoders.resnet.random_crop=false encoders.resnet.share_rgb_model=true encoders.resnet.rgb_model.weights=IMAGENET1K_V1 encoders.resnet.imagenet_norm=true use_vib=false use_recon=false kl_annealing=false use_action_embed=false use_conv_action_embed=false task.dataset._target_=lerobot_robot_marvin.rl100_offline.MarvinOfflineDataset task.dataset.zarr_path="${OFFLINE_DATASET}" +task.dataset.normalizer_zarr_path="${BC_DATASET}" +task.dataset.executed_steps="${EXECUTED_STEPS}" task.dataset.sequence_stride="${SEQUENCE_STRIDE}" task.dataset.val_ratio=0.1 ++task.env_runner._target_=lerobot_robot_marvin.rl100_offline.MarvinOfflineEvalRunner ++task.env_runner.zarr_path="${OFFLINE_DATASET}" ++task.env_runner.normalizer_zarr_path="${BC_DATASET}" ++task.env_runner.horizon=17 ++task.env_runner.n_obs_steps=2 ++task.env_runner.executed_steps="${EXECUTED_STEPS}" ++task.env_runner.seed=1000 ++task.env_runner.val_ratio=0.1 ++task.env_runner.batch_size=4 ++task.env_runner.max_batches=8 ++task.env_runner.env_num=1 ++task.env_runner.eval_episodes=1 dataloader.batch_size="${BATCH_SIZE}" dataloader.num_workers="${NUM_WORKERS}" dataloader.pin_memory=false dataloader.persistent_workers=false val_dataloader.batch_size="${BATCH_SIZE}" val_dataloader.num_workers="${NUM_WORKERS}" val_dataloader.pin_memory=false val_dataloader.persistent_workers=false critic.q_hidden_dim=512 critic.q_depth=3 critic.q_lr=1e-4 critic.v_hidden_dim=512 critic.v_depth=3 critic.v_lr=1e-4 critic.omega=0.7 critic.gamma=0.99 critic.tau=0.005 critic.target_update_freq=2 critic.is_double_q=true critic.is_iql=true critic.is_share_encoder=true critic.fix_encoder=true critic.load_pretrain=true dynamics_type=mlp dynamics.prediction_mode=full dynamics.fix_encoder=true dynamics.dynamics_lr=3e-4 dynamics.dynamics_max_epochs="${DYNAMICS_EPOCHS}" dynamics.max_epochs_since_update=10 dynamics.dynamics_hidden_dims='[512,512,256,256]' predict_r=true offline_use_aug=false unio4.bppo_steps="${BPPO_STEPS}" unio4.bppo_lr=1e-6 unio4.clip_ratio=0.25 unio4.fix_encoder=true unio4.finetune_batch_size="${BATCH_SIZE}" unio4.bppo_batch_size="${BATCH_SIZE}" unio4.rollout_length=3 unio4.eval_step=50 unio4.eval_freq=500 unio4.idql_eval=false unio4.is_linear_decay=true unio4.is_update_old_policy=true offline_chunk_ratio_mode=scalar offline_chunk_adv_mode=scalar_iql bppo_chunk_level_ratio=true +marvin_offline.base_checkpoint="${BC_CHECKPOINT}" +marvin_offline.checkpoint_source=model

"${PYTHON_BIN}" "${WORKSPACE_ROOT}/export_rl100_offline_checkpoint.py" --base-checkpoint "${BC_CHECKPOINT}" --policy-dir "${OUTPUT_DIR}/last" --output-checkpoint "${OUTPUT_DIR}/checkpoints/offline_last.ckpt" --action-steps "${EXECUTED_STEPS}"
if [[ -f "${OUTPUT_DIR}/best/model.pt" && -f "${OUTPUT_DIR}/best/encoder.pt" ]]; then
    "${PYTHON_BIN}" "${WORKSPACE_ROOT}/export_rl100_offline_checkpoint.py" --base-checkpoint "${BC_CHECKPOINT}" --policy-dir "${OUTPUT_DIR}/best" --output-checkpoint "${OUTPUT_DIR}/checkpoints/offline_best.ckpt" --action-steps "${EXECUTED_STEPS}"
fi
