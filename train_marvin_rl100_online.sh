#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/home/xlb/code/强化微调ACT
RL100_ROOT=/home/xlb/code/RL-100/RL-100
PYTHON_BIN=/home/xlb/miniconda3/envs/rl100/bin/python

STAGE2_DIR=${STAGE2_DIR:-/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_offline_rl_k5_seed1000}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-${STAGE2_DIR}/checkpoints/offline_best.ckpt}
STAGE2_POLICY_DIR=${STAGE2_POLICY_DIR:-${STAGE2_DIR}/best}
OFFLINE_DATASET=${OFFLINE_DATASET:-/home/xlb/code/RL-100/RL-100/data/marvin_usb_offline_rl.zarr}
BC_DATASET=${BC_DATASET:-/home/xlb/code/RL-100/RL-100/data/marvin_usb_bc.zarr}
OUTPUT_DIR=${OUTPUT_DIR:-/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_online_rl_k5_seed1000}
EXECUTED_STEPS=${EXECUTED_STEPS:-5}
ONLINE_BATCH_SIZE=${ONLINE_BATCH_SIZE:-64}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
PPO_EPOCHS=${PPO_EPOCHS:-3}
MAX_UPDATES=${MAX_UPDATES:-10}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-150}
ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-4}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8765}
DEVICE=${DEVICE:-cuda:0}
COLLECT_ONLY=${COLLECT_ONLY:-true}
RESUME_ONLINE=${RESUME_ONLINE:-true}
WANDB_LOG_MODE=${WANDB_LOG_MODE:-offline}
ALLOW_REAL_ROBOT=${ALLOW_REAL_ROBOT:-false}
CONFIRM_HARDWARE_RISK=${CONFIRM_HARDWARE_RISK:-}

if [[ "${ALLOW_REAL_ROBOT}" == "true" && "${CONFIRM_HARDWARE_RISK}" != "I_UNDERSTAND_THIS_CONTROLS_A_REAL_ROBOT" ]]; then
    echo "Hardware online collection requires CONFIRM_HARDWARE_RISK=I_UNDERSTAND_THIS_CONTROLS_A_REAL_ROBOT" >&2
    exit 1
fi
if (( EXECUTED_STEPS != 5 )); then
    echo "This online run must match the trained stage-2 K=5 checkpoint" >&2
    exit 1
fi
if (( ONLINE_BATCH_SIZE <= 1 || MINI_BATCH_SIZE <= 0 || ONLINE_BATCH_SIZE % MINI_BATCH_SIZE != 0 )); then
    echo "ONLINE_BATCH_SIZE must be >1 and divisible by MINI_BATCH_SIZE" >&2
    exit 1
fi
for required_path in "${STAGE2_DIR}/Q_bc_20.pt" "${STAGE2_DIR}/value_20.pt" "${STAGE2_DIR}/saved_models_full/dynamics.pth" "${STAGE2_POLICY_DIR}/model.pt" "${STAGE2_POLICY_DIR}/encoder.pt" "${BASE_CHECKPOINT}" "${OFFLINE_DATASET}" "${BC_DATASET}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

export PYTHONPATH="${WORKSPACE_ROOT}/src:${RL100_ROOT}:/home/xlb/code/RL-100/third_party/gym-0.21.0"
export TORCH_HOME=${TORCH_HOME:-/home/xlb/code/强化微调ACT/.cache/torch}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-rl100-online}
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-offline}

cd "${RL100_ROOT}"
"${PYTHON_BIN}" "${WORKSPACE_ROOT}/train_marvin_rl100_online.py" --config-name=rl100_2d_epsilon task=marvin_usb hydra.run.dir="${OUTPUT_DIR}" training.device="${DEVICE}" training.seed=1000 training.resume=true training.num_epochs=0 training.num_critic_epochs=0 training.use_ema=false training.rollout_every=1000000 training.checkpoint_every=1000000 training.val_every=1 training.sample_every=1000000 only_bc=false offline=true online=true eval=false distill_phase=null use_wandb=true logging.mode="${WANDB_LOG_MODE}" logging.project=marvin-rl100-online-rl logging.name=marvin_usb_online_rl_k5_seed1000 checkpoint.save_ckpt=false env_num=1 feature_type=2D with_pointcloud=false use_agent_pos=true action_norm=true chunk_as_single_action=true ft_all_actions=false horizon=17 n_obs_steps=2 n_action_steps="${EXECUTED_STEPS}" num_inference_steps=10 policy._target_=rl_100.policy.rl100_2d.RL1002D policy.scheduler_type=ddim policy.model=dp3 policy.w_pc=false policy.use_visual=true policy.use_aug=false policy.img_shape='[3,128,128]' policy.down_dims='[128,256,512]' policy.chunk_as_single_action=true encoder_type=resnet encoders.resnet.resize_shape='[128,128]' encoders.resnet.crop_shape=null encoders.resnet.random_crop=false encoders.resnet.share_rgb_model=true encoders.resnet.rgb_model.weights=IMAGENET1K_V1 encoders.resnet.imagenet_norm=true use_vib=false use_recon=false kl_annealing=false use_action_embed=false use_conv_action_embed=false task.dataset._target_=lerobot_robot_marvin.rl100_offline.MarvinOfflineDataset task.dataset.zarr_path="${OFFLINE_DATASET}" +task.dataset.normalizer_zarr_path="${BC_DATASET}" +task.dataset.executed_steps="${EXECUTED_STEPS}" task.dataset.sequence_stride="${EXECUTED_STEPS}" task.dataset.val_ratio=0.1 ++task.env_runner._target_=lerobot_robot_marvin.rl100_offline.MarvinOfflineEvalRunner ++task.env_runner.zarr_path="${OFFLINE_DATASET}" ++task.env_runner.normalizer_zarr_path="${BC_DATASET}" ++task.env_runner.horizon=17 ++task.env_runner.n_obs_steps=2 ++task.env_runner.executed_steps="${EXECUTED_STEPS}" ++task.env_runner.seed=1000 ++task.env_runner.val_ratio=0.1 ++task.env_runner.batch_size=4 ++task.env_runner.max_batches=1 ++task.env_runner.env_num=1 ++task.env_runner.eval_episodes=1 dataloader.batch_size=8 dataloader.num_workers=0 dataloader.pin_memory=false dataloader.persistent_workers=false val_dataloader.batch_size=8 val_dataloader.num_workers=0 val_dataloader.pin_memory=false val_dataloader.persistent_workers=false critic.q_hidden_dim=512 critic.q_depth=3 critic.q_lr=1e-4 critic.v_hidden_dim=512 critic.v_depth=3 critic.v_lr=1e-4 critic.omega=0.7 critic.gamma=0.99 critic.tau=0.005 critic.target_update_freq=2 critic.is_double_q=true critic.is_iql=true critic.is_share_encoder=true critic.fix_encoder=true critic.load_pretrain=true dynamics_type=mlp dynamics.prediction_mode=full dynamics.fix_encoder=true predict_r=true +unio4.global_best_dir="${STAGE2_POLICY_DIR}" unio4.fix_encoder=true ppo.batch_size="${ONLINE_BATCH_SIZE}" ppo.mini_batch_size="${MINI_BATCH_SIZE}" ppo.K_epochs="${PPO_EPOCHS}" ppo.lr_a="${ACTOR_LR}" ppo.lr_c="${CRITIC_LR}" ppo.fix_encoder=true ppo.share_encoder=false ppo.iql_ft=false ppo.use_bc=false ppo.actor_grad=true ppo.use_lr_decay=false ppo.force_stochastic_online=true +marvin_online.stage2_dir="${STAGE2_DIR}" +marvin_online.stage2_policy_dir="${STAGE2_POLICY_DIR}" +marvin_online.base_checkpoint="${BASE_CHECKPOINT}" +marvin_online.host="${HOST}" +marvin_online.port="${PORT}" +marvin_online.timeout=1.0 +marvin_online.connect_timeout=30.0 +marvin_online.reset_timeout=120.0 +marvin_online.heartbeat_interval=0.5 +marvin_online.allow_real_robot="${ALLOW_REAL_ROBOT}" +marvin_online.actions_per_inference="${EXECUTED_STEPS}" +marvin_online.max_episode_steps="${MAX_EPISODE_STEPS}" +marvin_online.action_match_atol=1e-6 +marvin_online.batch_size="${ONLINE_BATCH_SIZE}" +marvin_online.max_updates="${MAX_UPDATES}" +marvin_online.collect_only="${COLLECT_ONLY}" +marvin_online.resume_online="${RESUME_ONLINE}"
